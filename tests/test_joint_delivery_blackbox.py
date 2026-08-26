"""Black-box delivery tests for the public ``ft serve`` interface."""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

import pytest

HOST = "127.0.0.1"
STARTUP_TIMEOUT = 180
REQUEST_TIMEOUT = 120


@pytest.fixture(scope="session")
def runtime():
    model = os.environ.get("FT_JOINT_DELIVERY_MODEL")
    if not model:
        pytest.skip("FT_JOINT_DELIVERY_MODEL is not set")

    cli = shlex.split(os.environ.get("FT_CLI", "ft"))
    if not cli or shutil.which(cli[0]) is None:
        pytest.skip("FT_CLI is not available")

    hidden_gpus = os.environ.get("CUDA_VISIBLE_DEVICES") in {"", "-1"}
    nvidia_smi = shutil.which("nvidia-smi")
    if hidden_gpus or nvidia_smi is None:
        pytest.skip("no configured GPU is available")
    gpu_check = subprocess.run(
        [nvidia_smi, "-L"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if gpu_check.returncode or not gpu_check.stdout.strip():
        pytest.skip("no configured GPU is available")
    return cli, model


def _free_port():
    with socket.socket() as sock:
        sock.bind((HOST, 0))
        return sock.getsockname()[1]


class _Server:
    def __init__(self, cli, model, policy, extra=()):
        self.port = _free_port()
        self.base_url = f"http://{HOST}:{self.port}"
        common_extra = shlex.split(
            os.environ.get("FT_JOINT_DELIVERY_EXTRA_ARGS", "")
        )
        command = [*cli, "serve", "--model-path", model, "--host", HOST]
        command += ["--port", str(self.port), "--batching-policy", policy]
        command += [*extra, *common_extra]
        self.process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        self._lines = []
        self._reader = threading.Thread(target=self._drain_logs, daemon=True)
        self._reader.start()
        try:
            self._wait_until_listening()
            self.model_id = self._wait_until_ready()
        except BaseException:
            self.close()
            raise

    def _drain_logs(self):
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._lines.append(line)

    def _wait_until_listening(self):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(
                    f"server exited with {self.process.returncode}\n{self.logs}"
                )
            with socket.socket() as probe:
                probe.settimeout(0.2)
                if probe.connect_ex((HOST, self.port)) == 0:
                    return
            time.sleep(0.2)
        raise AssertionError(f"server did not listen in time\n{self.logs}")

    def _wait_until_ready(self):
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(
                    f"server exited with {self.process.returncode}\n{self.logs}"
                )
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/v1/models", timeout=5
                ) as response:
                    models = json.load(response)["data"]
                assert models, "/v1/models returned no served model"
                model_id = models[0]["id"]
                ready = _completion(self.base_url, model_id, "Ready", 1)
                _assert_completion_shape(ready)
                return model_id
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")
                if error.code == 503 and "still loading" in detail.lower():
                    time.sleep(0.5)
                    continue
                raise AssertionError(
                    f"readiness HTTP {error.code}: {detail}"
                ) from error
            except (ConnectionError, TimeoutError, urllib.error.URLError):
                time.sleep(0.5)
        raise AssertionError(f"model did not become ready in time\n{self.logs}")

    @property
    def logs(self):
        return "".join(self._lines)

    def close(self):
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        if self.process.poll() is None:
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
        self._reader.join(timeout=2)


@contextlib.contextmanager
def _serve(runtime, policy="joint", *extra):
    cli, model = runtime
    server = _Server(cli, model, policy, extra)
    try:
        yield server
    finally:
        server.close()


def _request(base_url, model, prompt, max_tokens, *, stream=False, seed=1234):
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens}
    body.update(temperature=0, seed=seed, stream=stream)
    return urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )


def _completion(base_url, model, prompt, max_tokens, *, seed=1234):
    request = _request(base_url, model, prompt, max_tokens, seed=seed)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        assert response.status == 200
        return json.load(response)


def _stream_completion(base_url, model, prompt, max_tokens, started=None):
    request = _request(base_url, model, prompt, max_tokens, stream=True)
    text_parts = []
    response_ids = set()
    finished = False
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        assert response.status == 200
        for raw_line in response:
            line = raw_line.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                finished = True
                break
            chunk = json.loads(payload)
            if started is not None:
                started.set()
            if "id" in chunk:
                response_ids.add(chunk["id"])
            choice = chunk["choices"][0]
            text_parts.append(choice.get("text", ""))
            finished |= choice.get("finish_reason") is not None
    assert finished
    assert len(response_ids) == 1
    return "".join(text_parts), next(iter(response_ids))


def _abort_after_first_event(base_url, model):
    request = _request(base_url, model, "Continue: one two three", 128, stream=True)
    with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT) as response:
        for raw_line in response:
            line = raw_line.decode().strip()
            if line.startswith("data:") and line[5:].strip() != "[DONE]":
                json.loads(line[5:].strip())
                return
    raise AssertionError("stream ended before an observable event could be aborted")


def _assert_completion_shape(result):
    assert result["object"] == "text_completion"
    assert len(result["choices"]) == 1
    assert isinstance(result["choices"][0]["text"], str)
    assert result["choices"][0]["finish_reason"] is not None


def _assert_logged_value(logs, option, value):
    name = re.escape(option).replace(r"\-", "[-_ ]")
    pattern = rf"{name}[\"']?\s*(?:=|:)\s*{value}\b"
    assert re.search(pattern, logs, re.IGNORECASE), (
        f"server logs did not report {option}={value}\n{logs}"
    )


def test_serve_help_lists_all_batching_policies():
    cli = shlex.split(os.environ.get("FT_CLI", "ft"))
    if not cli or shutil.which(cli[0]) is None:
        pytest.skip("FT_CLI is not available")
    result = subprocess.run(
        [*cli, "serve", "--help"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    help_text = result.stdout + result.stderr
    for policy in ("legacy", "mixed", "layered", "joint"):
        assert policy in help_text


@pytest.mark.parametrize(
    "option,value",
    [
        ("--prefill-layer-group-size", "0"),
        ("--prefill-layer-group-size", "-1"),
        ("--prefill-wave-max-chunks", "0"),
        ("--prefill-wave-max-chunks", "-1"),
    ],
)
def test_joint_rejects_nonpositive_integer_options(runtime, option, value):
    cli, model = runtime
    command = [
        *cli,
        "serve",
        "--model-path",
        model,
        "--batching-policy",
        "joint",
    ]
    command += [option, value]
    command += shlex.split(os.environ.get("FT_JOINT_DELIVERY_EXTRA_ARGS", ""))
    result = subprocess.run(command, capture_output=True, text=True, timeout=20)
    output = result.stdout + result.stderr
    assert result.returncode != 0
    assert option.lstrip("-") in output
    assert re.search(
        r"positive|greater than 0|at least 1|invalid|must be\s*>=\s*1", output, re.I
    )


def test_joint_default_delivery_scenarios(runtime):
    long_prompt = "The quick brown fox crosses the quiet field. " * 40
    options = ("--max-prefill-length", "8", "--prefill-layer-group-size", "2")
    with _serve(runtime, "joint", *options) as server:
        model_id = server.model_id
        long_prefill = _completion(server.base_url, model_id, long_prompt, 1)
        _assert_completion_shape(long_prefill)

        decode = _completion(server.base_url, model_id, "A", 16)
        _assert_completion_shape(decode)

        multi_chunk = _completion(server.base_url, model_id, long_prompt, 8)
        _assert_completion_shape(multi_chunk)

        starts = [threading.Event() for _ in range(3)]
        prompts = ["Count upward:", "Name colors:", "Write a short sentence:"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(
                    _stream_completion,
                    server.base_url,
                    model_id,
                    prompt,
                    64,
                    started,
                )
                for prompt, started in zip(prompts, starts)
            ]
            assert all(started.wait(60) for started in starts)
            late_prefill = _completion(server.base_url, model_id, long_prompt, 8)
            _assert_completion_shape(late_prefill)
            streams = [future.result(timeout=REQUEST_TIMEOUT) for future in futures]
        assert len({response_id for _, response_id in streams}) == len(streams)

        _abort_after_first_event(server.base_url, model_id)
        for prompt in ("After abort one", "After abort two", "After abort three"):
            follow_up = _completion(server.base_url, model_id, prompt, 8)
            _assert_completion_shape(follow_up)

    _assert_logged_value(server.logs, "prefill-wave-max-chunks", 1)
    _assert_logged_value(server.logs, "prefill-layer-group-size", 2)


def test_joint_explicit_wave_limit_is_used(runtime):
    with _serve(
        runtime,
        "joint",
        "--max-prefill-length",
        "8",
        "--prefill-wave-max-chunks",
        "2",
    ) as server:
        result = _completion(
            server.base_url, server.model_id, "Chunk this prompt. " * 30, 8
        )
        _assert_completion_shape(result)
    _assert_logged_value(server.logs, "prefill-wave-max-chunks", 2)


def test_legacy_and_joint_match_for_seeded_greedy_completion(runtime):
    prompt = "Complete this deterministic statement: The sky above the city"
    with _serve(runtime, "legacy") as legacy:
        legacy_result = _completion(
            legacy.base_url, legacy.model_id, prompt, 24, seed=8675309
        )
    with _serve(runtime, "joint") as joint:
        joint_result = _completion(
            joint.base_url, joint.model_id, prompt, 24, seed=8675309
        )

    _assert_completion_shape(legacy_result)
    _assert_completion_shape(joint_result)
    assert (
        joint_result["choices"][0]["text"]
        == legacy_result["choices"][0]["text"]
    )
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert joint_result["usage"][key] == legacy_result["usage"][key]
