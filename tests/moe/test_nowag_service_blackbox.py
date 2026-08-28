"""Opt-in black-box service matrix for real NoWAG models and sidecars."""

import concurrent.futures
import contextlib
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get("FREETOKEN_RUN_NOWAG_E2E") != "1",
    reason="set FREETOKEN_RUN_NOWAG_E2E=1 to start real model services",
)

_PROJECT = Path(__file__).resolve().parents[3]
_FREETOKEN = _PROJECT / "FreeToken"
_FREETOKEN_PYTHON = _FREETOKEN / "python"
_NOWAG_PLUGIN = _PROJECT / ".kernel-worktrees/nowag_final2_tail64_profile/src"
_BATCH_SIZES = (1, 2, 3, 4, 5, 7, 8)

_MODELS = {
    "qwen": {
        "model_env": "FREETOKEN_E2E_QWEN_MODEL",
        "model": "/data1/lmcache_kv/models/Qwen3.6-35B-A3B",
        "sidecar_env": "FREETOKEN_E2E_QWEN_SIDECAR",
        "sidecar": (
            "/data1/lmcache_kv/nowag_qwen36_experiment/quantized/"
            "qwen36_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
        ),
        "slots_env": "FREETOKEN_E2E_QWEN_CACHE_SLOTS",
        "extra_env": "FREETOKEN_E2E_QWEN_EXTRA_ARGS",
        "fixed_args": (),
    },
    "dsv4": {
        "model_env": "FREETOKEN_E2E_DSV4_MODEL",
        "model": "/data1/lmcache_kv/models/DeepSeek-V4-Flash-0731",
        "sidecar_env": "FREETOKEN_E2E_DSV4_SIDECAR",
        "sidecar": (
            "/data1/lmcache_kv/nowag_4090_experiment/quantized/"
            "dsv4_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
        ),
        "slots_env": "FREETOKEN_E2E_DSV4_CACHE_SLOTS",
        "default_slots": "1649",
        "extra_env": "FREETOKEN_E2E_DSV4_EXTRA_ARGS",
        "fixed_args": (
            "--max-seq-len-override",
            "17190",
            "--memory-ratio",
            "0.95",
            "--num-tokens",
            "69120",
        ),
    },
}


def _free_local_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _tail(path: Path, limit=12000):
    if not path.exists():
        return "<no server log>"
    text = path.read_text(errors="replace")
    return text[-limit:]


def _wait_for_model(proc, base_url, log_path):
    timeout = float(os.environ.get("FREETOKEN_E2E_STARTUP_TIMEOUT", "1800"))
    deadline = time.monotonic() + timeout
    last_error = None
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            pytest.fail(f"server exited with {proc.returncode}\n{_tail(log_path)}")
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
                health = json.load(response)
        except urllib.error.HTTPError as error:
            pytest.fail(f"health probe returned HTTP {error.code}\n{_tail(log_path)}")
        except (OSError, urllib.error.URLError) as error:
            last_error = error
            time.sleep(2)
            continue

        status = health.get("status")
        if status == "loading":
            last_error = health
            time.sleep(2)
            continue
        if status == "error":
            pytest.fail(
                f"model worker reported error: {health.get('message')}\n{_tail(log_path)}"
            )
        if status != "ok":
            pytest.fail(f"unexpected health response: {health}\n{_tail(log_path)}")

        try:
            with urllib.request.urlopen(f"{base_url}/v1/models", timeout=5) as response:
                payload = json.load(response)
            return payload["data"][0]["id"]
        except (OSError, KeyError, IndexError, json.JSONDecodeError) as error:
            pytest.fail(f"ready server did not expose a model id: {error}\n{_tail(log_path)}")
    pytest.fail(f"server did not become ready: {last_error}\n{_tail(log_path)}")


def _assert_healthy(base_url):
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=5) as response:
            health = json.load(response)
    except urllib.error.HTTPError as error:
        pytest.fail(f"health probe returned HTTP {error.code}")
    except OSError as error:
        pytest.fail(f"health probe failed: {error}")
    assert health.get("status") == "ok", health


def _stop_server(proc):
    if proc.poll() is not None:
        return
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        proc.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)


def _service_command(config, backend, port):
    model = os.environ.get(config["model_env"], config["model"])
    sidecar = os.environ.get(config["sidecar_env"], config["sidecar"])
    if not Path(model).exists():
        pytest.fail(f"model does not exist: {model}")
    if not Path(sidecar).exists():
        pytest.fail(f"NoWAG sidecar does not exist: {sidecar}")

    command = [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        model,
        "--nowag-expert-path",
        sidecar,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--moe-backend",
        backend,
        "--max-running-requests",
        "8",
        "--cuda-graph-max-bs",
        "8",
        *config["fixed_args"],
    ]
    if backend == "hybrid":
        slots = os.environ.get(config["slots_env"], config.get("default_slots"))
        if slots is None:
            pytest.fail(f"hybrid requires {config['slots_env']}")
        command += [
            "--moe-cache-size",
            slots,
            "--moe-hybrid-max-fetch",
            "-1",
        ]

    for env_name in (
        "FREETOKEN_E2E_EXTRA_ARGS",
        config["extra_env"],
        f"FREETOKEN_E2E_{backend.upper()}_EXTRA_ARGS",
    ):
        command += shlex.split(os.environ.get(env_name, ""))
    return command


@contextlib.contextmanager
def _running_service(config, backend, tmp_path):
    port = _free_local_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = tmp_path / f"{backend}.log"
    command = _service_command(config, backend, port)
    env = os.environ.copy()
    python_paths = [str(_FREETOKEN_PYTHON), str(_NOWAG_PLUGIN)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)

    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=_FREETOKEN,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            model_id = _wait_for_model(proc, base_url, log_path)
            yield base_url, model_id
            if proc.poll() is not None:
                pytest.fail(f"server exited with {proc.returncode}\n{_tail(log_path)}")
            _assert_healthy(base_url)
        finally:
            _stop_server(proc)


def _stream_completion(base_url, model_id, prompt, max_tokens):
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1,
        "top_k": -1,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    timeout = float(os.environ.get("FREETOKEN_E2E_REQUEST_TIMEOUT", "600"))
    pieces = []
    usage = None
    done_seen = False
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            assert response.status == 200
            for raw_line in response:
                line = raw_line.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    done_seen = True
                    break
                event = json.loads(data)
                assert event.get("error") is None, event["error"]
                if event.get("usage") is not None:
                    usage = event["usage"]
                for choice in event.get("choices", []):
                    delta = choice.get("delta") or {}
                    pieces.append(delta.get("reasoning_content") or "")
                    pieces.append(delta.get("content") or "")
    except urllib.error.HTTPError as error:
        detail = error.read().decode(errors="replace")
        pytest.fail(f"completion returned HTTP {error.code}: {detail}")

    text = "".join(pieces)
    assert done_seen
    assert text
    assert isinstance(usage, dict)
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")
    assert isinstance(prompt_tokens, int) and prompt_tokens > 0
    assert isinstance(completion_tokens, int) and completion_tokens > 0
    assert isinstance(total_tokens, int)
    assert total_tokens >= prompt_tokens + completion_tokens
    return {"content": text, "usage": usage}


def _exercise_batches(base_url, model_id):
    max_tokens = int(os.environ.get("FREETOKEN_E2E_MAX_TOKENS", "16"))
    results = {}
    for batch_size in _BATCH_SIZES:
        prompts = [
            f"Return the single word OK. Batch {batch_size}, request {index}."
            for index in range(batch_size)
        ]
        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [
                pool.submit(
                    _stream_completion, base_url, model_id, prompt, max_tokens
                )
                for prompt in prompts
            ]
            completed = [future.result() for future in futures]
        assert len(completed) == batch_size
        results[f"batch-{batch_size}"] = completed
        _assert_healthy(base_url)

    if os.environ.get("FREETOKEN_RUN_NOWAG_E2E_LONG") == "1":
        aime_path = Path(
            os.environ.get(
                "FREETOKEN_E2E_AIME_PATH",
                "/data1/lmcache_kv/datasets/aime25/test.jsonl",
            )
        )
        with aime_path.open() as dataset:
            item = json.loads(next(line for line in dataset if line.strip()))
        prompt = item["problem"]
        long_tokens = int(os.environ.get("FREETOKEN_E2E_LONG_MAX_TOKENS", "1024"))
        results["aime-long"] = [
            _stream_completion(base_url, model_id, prompt, long_tokens)
        ]
        _assert_healthy(base_url)
    return results


@pytest.mark.parametrize("model_name", ("qwen", "dsv4"))
def test_real_service_cpu_and_hybrid_match_all_batches(model_name, tmp_path):
    config = _MODELS[model_name]
    backend_results = {}
    for backend in ("cpu", "hybrid"):
        with _running_service(config, backend, tmp_path) as (base_url, model_id):
            backend_results[backend] = _exercise_batches(base_url, model_id)

    assert backend_results["hybrid"].keys() == backend_results["cpu"].keys()
    hybrid_counts = {
        batch: len(responses)
        for batch, responses in backend_results["hybrid"].items()
    }
    cpu_counts = {
        batch: len(responses) for batch, responses in backend_results["cpu"].items()
    }
    assert hybrid_counts == cpu_counts
