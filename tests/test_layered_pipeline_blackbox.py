from __future__ import annotations

import importlib.util
import json
import math
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_ROOT = PROJECT_ROOT / "python"
SERVED_MODEL = "lab-agent-qwen3-moe"
HOST = "127.0.0.1"
PROMPT_LENGTHS = {
    "concurrent_decode": 64,
    "concurrent_prefill": 900,
    "pure_decode": 48,
    "pure_prefill": 700,
}

for import_root in (str(PYTHON_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

WAVE_PATTERN = re.compile(
    r"Layered pipeline wave complete: "
    r"chunks=(?P<chunks>\d+), "
    r"wave_reqs=(?P<wave_reqs>\d+), "
    r"frontier_batches=(?P<frontier_batches>\d+), "
    r"resident_groups=(?P<resident_groups>\d+), "
    r"chunk_group_steps=(?P<chunk_group_steps>\d+), "
    r"frontier_group_forwards=(?P<frontier_group_forwards>\d+), "
    r"iterations=(?P<iterations>\d+), "
    r"decode_iterations=(?P<decode_iterations>\d+), "
    r"prefill_layer_prepares=(?P<prefill_layer_prepares>\d+), "
    r"cross_group_prefetches=(?P<cross_group_prefetches>\d+), "
    r"deferred_cross_group_prefetches="
    r"(?P<deferred_cross_group_prefetches>\d+)(?:\r?\n|$)"
)
JOINT_WAVE_PATTERN = re.compile(
    r"Joint wave complete: "
    r"chunks=(?P<chunks>\d+), "
    r"wave_reqs=(?P<wave_reqs>\d+), "
    r"frontier_batches=(?P<frontier_batches>\d+), "
    r"groups=(?P<groups>\d+), "
    r"effective_group_size=(?P<effective_group_size>\d+), "
    r"prefill_layer_prepares=(?P<prefill_layer_prepares>\d+)"
    r"(?:\r?\n|$)"
)
LAYERED_PREFILL_WAVE_PATTERN = re.compile(
    r"Layered prefill wave complete: "
    r"reqs=(?P<reqs>\d+), "
    r"groups=(?P<groups>\d+), "
    r"group_forwards=(?P<group_forwards>\d+), "
    r"iterations=(?P<iterations>\d+), "
    r"decode_iterations=(?P<decode_iterations>\d+), "
    r"prefill_layer_prepares=(?P<prefill_layer_prepares>\d+)"
    r"(?:\r?\n|$)"
)
CACHE_PATTERN = re.compile(
    r"Layered pipeline cache: "
    r"requested_group_size=(?P<requested_group_size>\d+), "
    r"effective_group_size=(?P<effective_group_size>\d+), "
    r"shared_expert_slots=(?P<shared_expert_slots>\d+)"
)
WAVE_FIELDS = {
    "chunks",
    "wave_reqs",
    "frontier_batches",
    "resident_groups",
    "chunk_group_steps",
    "frontier_group_forwards",
    "iterations",
    "decode_iterations",
    "prefill_layer_prepares",
    "cross_group_prefetches",
    "deferred_cross_group_prefetches",
}
JOINT_WAVE_FIELDS = {
    "chunks",
    "wave_reqs",
    "frontier_batches",
    "groups",
    "effective_group_size",
    "prefill_layer_prepares",
}
LAYERED_PREFILL_WAVE_FIELDS = {
    "reqs",
    "groups",
    "group_forwards",
    "iterations",
    "decode_iterations",
    "prefill_layer_prepares",
}
SCALED_MOE_STATS_FIELDS = {
    "decode_active_rows",
    "decode_missing_rows",
    "decode_layer_calls",
    "decode_fetched_rows",
    "prefill_hit_rows",
    "prefill_rows",
    "prefill_layer_prepares",
    "prefill_h2d_bytes_total",
    "expert_row_bytes",
    "decode_h2d_bytes",
    "total_expert_h2d_bytes",
}
RETIRED_PIPELINE_MODES = (
    "layered-pipeline-wave3",
    "layeredPipelineG2-wave1",
    "layeredPipelineG2-wave3",
    "layered_pipeline_g2_wave1",
    "layered_pipeline_g2_wave3",
)
def _subprocess_env(*, python_optimize: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(PYTHON_ROOT), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    env["PYTHONUNBUFFERED"] = "1"
    if python_optimize:
        env["PYTHONOPTIMIZE"] = "1"
    return env


@contextmanager
def _blackbox_case(label: str) -> Iterator[None]:
    try:
        yield
    except AssertionError as exc:
        exc.add_note(f"public black-box case: {label}")
        raise


def _require_cuda_e2e() -> None:
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        pytest.skip("CUDA is unavailable")
    for dependency in ("transformers", "flashlib"):
        if importlib.util.find_spec(dependency) is None:
            pytest.skip(f"{dependency} is unavailable")
    try:
        torch.cuda.get_device_properties(torch.cuda.current_device())
    except RuntimeError as exc:
        pytest.skip(f"no usable CUDA device: {exc}")


@pytest.fixture(scope="module")
def tiny_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_cuda_e2e()
    from benchmarks.bench_lab_agent_policies import create_small_qwen3_moe

    destination = tmp_path_factory.mktemp("layered-pipeline-model") / "model"
    create_small_qwen3_moe(destination)
    return destination


@pytest.fixture(scope="module")
def scaled_model(tmp_path_factory: pytest.TempPathFactory) -> Path:
    _require_cuda_e2e()
    from benchmarks.bench_scaled_expert_contention import (
        create_scaled_qwen3_moe,
    )

    destination = tmp_path_factory.mktemp("scaled-pipeline-model") / "model"
    create_scaled_qwen3_moe(destination)
    return destination


def _public_prompt_materializer(
    model_path: Path,
) -> Callable[[int, int, str, int], str]:
    from benchmarks.bench_lab_agent_policies import (
        continuation_token_pieces,
        load_tokenizer,
        materialize_segment_text,
    )

    tokenizer = load_tokenizer(model_path)
    pieces = continuation_token_pieces(tokenizer, required=64)
    assert len(pieces) >= 32
    assert len(set(pieces[:32])) == 32

    def materialize(
        length: int,
        seed: int,
        label: str,
        first_token_index: int,
    ) -> str:
        first_token = pieces[first_token_index % len(pieces)]
        assert (
            isinstance(first_token, tuple)
            and len(first_token) == 2
            and type(first_token[0]) is int
            and isinstance(first_token[1], str)
        )
        text, token_ids = materialize_segment_text(
            tokenizer,
            length=length,
            seed=seed,
            label=label,
            continuation_pieces=pieces,
            first_token=first_token,
        )
        assert len(token_ids) == length
        assert token_ids[0] == first_token[0]
        assert tokenizer.encode(text) == token_ids
        return text

    return materialize


@pytest.fixture(scope="module")
def tiny_prompt_materializer(
    tiny_model: Path,
) -> Callable[[int, int, str, int], str]:
    return _public_prompt_materializer(tiny_model)


@pytest.fixture(scope="module")
def scaled_prompt_materializer(
    scaled_model: Path,
) -> Callable[[int, int, str, int], str]:
    return _public_prompt_materializer(scaled_model)


@pytest.fixture(scope="module")
def public_prompts(tiny_model: Path) -> dict[str, str]:
    from benchmarks.bench_lab_agent_policies import (
        continuation_token_pieces,
        load_tokenizer,
        materialize_segment_text,
    )

    tokenizer = load_tokenizer(tiny_model)
    pieces = continuation_token_pieces(tokenizer, required=64)
    seeds = {
        "concurrent_decode": 11,
        "concurrent_prefill": 23,
        "pure_decode": 37,
        "pure_prefill": 41,
    }
    prompts: dict[str, str] = {}
    for label, length in PROMPT_LENGTHS.items():
        text, token_ids = materialize_segment_text(
            tokenizer,
            length=length,
            seed=seeds[label],
            label=label,
            continuation_pieces=pieces,
            first_token=None,
        )
        assert len(token_ids) == length
        prompts[label] = text
    return prompts


@pytest.fixture(scope="module")
def graph_workload_prompts(tiny_model: Path) -> dict[str, Any]:
    from benchmarks.bench_lab_agent_policies import (
        continuation_token_pieces,
        load_tokenizer,
        materialize_segment_text,
    )

    tokenizer = load_tokenizer(tiny_model)
    pieces = continuation_token_pieces(tokenizer, required=64)

    def segment(length: int, seed: int, label: str) -> str:
        text, token_ids = materialize_segment_text(
            tokenizer,
            length=length,
            seed=seed,
            label=label,
            continuation_pieces=pieces,
            first_token=None,
        )
        assert len(token_ids) == length
        return text

    batches: dict[int, dict[str, Any]] = {}
    for batch_size, prompt_length in ((1, 640), (4, 2_048), (8, 4_096)):
        prefixes: list[str] = []
        followups: list[str] = []
        for request_index in range(batch_size):
            label = f"graph-bs{batch_size}-request{request_index}"
            prefix = segment(
                prompt_length,
                seed=10_000 + prompt_length + request_index,
                label=f"{label}-prefix",
            )
            continuation = segment(
                32,
                seed=20_000 + prompt_length + request_index,
                label=f"{label}-continuation",
            )
            prefixes.append(prefix)
            followups.append(prefix + continuation)
        batches[batch_size] = {
            "prompt_length": prompt_length,
            "prefixes": prefixes,
            "followups": followups,
        }

    return {
        "batches": batches,
        "concurrent_decode": segment(64, 30_001, "graph-concurrent-decode"),
        "concurrent_prefill": segment(
            4_096, 30_002, "graph-concurrent-prefill"
        ),
    }


def _free_port() -> int:
    for _ in range(100):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((HOST, 0))
            server_port = int(server_socket.getsockname()[1])
            if server_port == 65_535:
                continue
            with socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            ) as backend_socket:
                try:
                    backend_socket.bind((HOST, server_port + 1))
                except OSError:
                    continue
                return server_port
    raise RuntimeError("no consecutive server and backend TCP ports are available")


def _service_command(
    model_path: Path,
    *,
    policy: str,
    port: int,
    cache_size: int = 24,
    group_size: int = 2,
    chunks_per_iteration: int | None = None,
    pipeline_wave_max_chunks: int | None = None,
    max_prefill_length: int = 1024,
    max_seq_len_override: int = 8192,
    cuda_graph_max_bs: int = 0,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model-path",
        str(model_path),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        HOST,
        "--port",
        str(port),
        "--dtype",
        "float16",
        "--attention-backend",
        "triton",
        "--moe-backend",
        "offload",
        "--moe-cache-size",
        str(cache_size),
        "--cuda-graph-max-bs",
        str(cuda_graph_max_bs),
        "--cache-type",
        "radix",
        "--max-prefill-length",
        str(max_prefill_length),
        "--max-running-requests",
        "8",
        "--max-seq-len-override",
        str(max_seq_len_override),
        "--enable-cache-report",
        "--batching-policy",
        policy,
        "--prefill-layer-group-size",
        str(group_size),
    ]
    if chunks_per_iteration is not None:
        command.extend(
            [
                "--layered-pipeline-chunks-per-iteration",
                str(chunks_per_iteration),
            ]
        )
    if pipeline_wave_max_chunks is not None:
        command.extend(
            [
                "--prefill-wave-max-chunks",
                str(pipeline_wave_max_chunks),
            ]
        )
    command.extend(
        [
            "--tensor-parallel-size",
            "1",
            "--disable-pynccl",
        ]
    )
    return command


def _completion_payload(
    prompt: str, *, max_tokens: int, stream: bool
) -> dict[str, Any]:
    return {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.0,
        "top_p": 1.0,
    }


def _post_completion(
    base_url: str, payload: dict[str, Any], *, timeout: float = 300.0
) -> dict[str, Any]:
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            assert response.status == 200
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(
            f"completion returned HTTP {exc.code}: {body}"
        ) from exc


def _completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    assert isinstance(choices, list) and choices
    text = choices[0].get("text")
    assert isinstance(text, str)
    assert isinstance(response.get("usage"), dict)
    return text


def _log_tail(log_path: Path, limit: int = 12_000) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")[-limit:]


def _wait_until_ready(
    process: subprocess.Popen[str], base_url: str, log_path: Path
) -> None:
    deadline = time.monotonic() + 600.0
    last_error = "service has not accepted a request"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"service exited with {process.returncode}:\n{_log_tail(log_path)}"
            )
        try:
            models_request = urllib.request.Request(
                f"{base_url}/v1/models", method="GET"
            )
            with urllib.request.urlopen(models_request, timeout=2.0) as response:
                if response.status != 200:
                    time.sleep(0.1)
                    continue
            _post_completion(
                base_url,
                _completion_payload("ready", max_tokens=1, stream=False),
                timeout=30.0,
            )
            return
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
            if exc.code != 503:
                time.sleep(0.1)
        except (urllib.error.URLError, TimeoutError, AssertionError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise AssertionError(
        f"service did not become ready ({last_error}):\n{_log_tail(log_path)}"
    )


@contextmanager
def _running_service(
    model_path: Path,
    log_path: Path,
    *,
    policy: str,
    cache_size: int = 24,
    group_size: int = 2,
    chunks_per_iteration: int | None = None,
    pipeline_wave_max_chunks: int | None = None,
    max_prefill_length: int = 1024,
    max_seq_len_override: int = 8192,
    cuda_graph_max_bs: int = 0,
    python_optimize: bool = False,
) -> Iterator[tuple[str, Path]]:
    port = _free_port()
    command = _service_command(
        model_path,
        policy=policy,
        port=port,
        cache_size=cache_size,
        group_size=group_size,
        chunks_per_iteration=chunks_per_iteration,
        pipeline_wave_max_chunks=pipeline_wave_max_chunks,
        max_prefill_length=max_prefill_length,
        max_seq_len_override=max_seq_len_override,
        cuda_graph_max_bs=cuda_graph_max_bs,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=_subprocess_env(python_optimize=python_optimize),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        base_url = f"http://{HOST}:{port}"
        try:
            _wait_until_ready(process, base_url, log_path)
            yield base_url, log_path
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30.0)


def _records_from_text(
    content: str, pattern: re.Pattern[str]
) -> list[dict[str, int]]:
    return [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in pattern.finditer(content)
    ]


def _records(log_path: Path, pattern: re.Pattern[str]) -> list[dict[str, int]]:
    return _records_from_text(
        _log_tail(log_path, limit=1_000_000), pattern
    )


def _wait_for_record(
    log_path: Path,
    pattern: re.Pattern[str],
    predicate: Callable[[dict[str, int]], bool],
    *,
    timeout: float = 60.0,
) -> dict[str, int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        for record in _records(log_path, pattern):
            if predicate(record):
                return record
        time.sleep(0.1)
    raise AssertionError(
        f"expected public log record was not emitted:\n{_log_tail(log_path)}"
    )


def _wait_for_wave_accounting(
    log_path: Path,
    pattern: re.Pattern[str],
    *,
    offset: int,
    expected_requests: int,
    expected_chunks: int,
    timeout: float = 120.0,
) -> list[dict[str, int]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waves = _records(log_path, pattern)[offset:]
        observed_requests = sum(wave["wave_reqs"] for wave in waves)
        observed_chunks = sum(wave["chunks"] for wave in waves)
        if (
            observed_requests == expected_requests
            and observed_chunks == expected_chunks
        ):
            return waves
        assert observed_requests <= expected_requests, waves
        assert observed_chunks <= expected_chunks, waves
        time.sleep(0.1)
    raise AssertionError(
        "public wave accounting did not close: "
        f"requests={expected_requests}, chunks={expected_chunks}\n"
        f"{_log_tail(log_path)}"
    )


def _wait_for_layered_prefill_requests(
    log_path: Path,
    *,
    offset: int,
    expected_requests: int,
    timeout: float = 120.0,
) -> list[dict[str, int]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waves = _records(log_path, LAYERED_PREFILL_WAVE_PATTERN)[offset:]
        observed_requests = sum(wave["reqs"] for wave in waves)
        if observed_requests == expected_requests:
            return waves
        assert observed_requests <= expected_requests, waves
        time.sleep(0.1)
    raise AssertionError(
        "layered-prefill wave membership did not close: "
        f"requests={expected_requests}\n{_log_tail(log_path)}"
    )


def _assert_wave_contract(
    wave: dict[str, int],
    *,
    chunks: int,
    resident_groups: int,
    chunks_per_iteration: int,
    num_layers: int,
    wave_reqs: int | None = 1,
    frontier_batches: int | None = None,
    frontier_group_forwards: int | None = None,
    expected_iterations: int | None = None,
    expected_decode_iterations: int | None = None,
) -> None:
    assert set(wave) == WAVE_FIELDS
    assert all(
        type(wave[field]) is int and wave[field] >= 0
        for field in WAVE_FIELDS
    )
    assert wave["chunks"] == chunks
    if wave_reqs is None:
        assert 1 <= wave["wave_reqs"] <= chunks
    else:
        assert wave["wave_reqs"] == wave_reqs
    if frontier_batches is None and wave_reqs == 1:
        frontier_batches = chunks
    if frontier_batches is None:
        assert 1 <= wave["frontier_batches"] <= chunks
    else:
        assert wave["frontier_batches"] == frontier_batches
    assert wave["resident_groups"] == resident_groups
    assert wave["chunk_group_steps"] == chunks * resident_groups
    if frontier_group_forwards is None and wave_reqs == 1:
        frontier_group_forwards = chunks * resident_groups
    if frontier_group_forwards is None:
        assert wave["frontier_group_forwards"] >= (
            wave["frontier_batches"] * resident_groups
        )
    else:
        assert wave["frontier_group_forwards"] == frontier_group_forwards
    if expected_iterations is None:
        expected_iterations = resident_groups * math.ceil(
            chunks / chunks_per_iteration
        )
    assert wave["iterations"] == expected_iterations
    if expected_decode_iterations is None:
        assert 0 <= wave["decode_iterations"] <= wave["iterations"]
    else:
        assert wave["decode_iterations"] == expected_decode_iterations
    assert wave["prefill_layer_prepares"] == num_layers


def _assert_joint_wave_contract(
    wave: dict[str, int],
    *,
    chunks: int,
    wave_reqs: int,
    frontier_batches: int,
    num_layers: int,
    effective_group_size: int,
) -> None:
    assert set(wave) == JOINT_WAVE_FIELDS
    assert all(
        type(wave[field]) is int and wave[field] >= 0
        for field in JOINT_WAVE_FIELDS
    )
    assert wave["chunks"] == chunks
    assert wave["wave_reqs"] == wave_reqs
    assert wave["frontier_batches"] == frontier_batches
    assert wave["groups"] == math.ceil(
        num_layers / effective_group_size
    )
    assert wave["effective_group_size"] == effective_group_size
    assert wave["prefill_layer_prepares"] == num_layers


def _assert_layered_prefill_wave_contract(
    wave: dict[str, int],
    *,
    groups: int,
    num_layers: int,
    expected_reqs: int | None = None,
) -> None:
    assert set(wave) == LAYERED_PREFILL_WAVE_FIELDS
    assert all(
        type(wave[field]) is int and wave[field] >= 0
        for field in LAYERED_PREFILL_WAVE_FIELDS
    )
    assert wave["reqs"] >= 1
    if expected_reqs is not None:
        assert wave["reqs"] == expected_reqs
    assert wave["groups"] == groups
    assert wave["group_forwards"] == groups
    assert wave["iterations"] == groups
    assert 0 <= wave["decode_iterations"] <= wave["iterations"]
    assert wave["prefill_layer_prepares"] == num_layers


def _assert_layered_prefill_structure(
    structure: dict[str, int],
    waves: list[dict[str, int]],
) -> None:
    assert set(structure) == LAYERED_PREFILL_WAVE_FIELDS
    assert structure == {
        field: sum(wave[field] for wave in waves)
        for field in LAYERED_PREFILL_WAVE_FIELDS
    }


def _assert_dynamic_pipeline_wave_contract(
    wave: dict[str, int],
    *,
    resident_groups: int,
    chunks_per_iteration: int,
    num_layers: int,
) -> None:
    assert 1 <= wave["wave_reqs"] <= wave["chunks"]
    assert 1 <= wave["frontier_batches"] <= wave["chunks"]
    assert wave["iterations"] >= resident_groups * math.ceil(
        wave["chunks"] / chunks_per_iteration
    )
    group_zero_forwards = wave["frontier_group_forwards"] - (
        (resident_groups - 1) * wave["frontier_batches"]
    )
    assert wave["frontier_batches"] <= group_zero_forwards
    assert group_zero_forwards <= wave["chunks"]
    _assert_wave_contract(
        wave,
        chunks=wave["chunks"],
        resident_groups=resident_groups,
        chunks_per_iteration=chunks_per_iteration,
        num_layers=num_layers,
        wave_reqs=wave["wave_reqs"],
        frontier_batches=wave["frontier_batches"],
        frontier_group_forwards=wave["frontier_group_forwards"],
        expected_iterations=wave["iterations"],
    )


def _assert_wave_soft_cap(
    waves: list[dict[str, int]],
    *,
    wave_max_chunks: int,
) -> None:
    assert wave_max_chunks >= 1
    for wave in waves:
        assert (
            wave["chunks"] <= wave_max_chunks or wave["wave_reqs"] == 1
        ), wave


def _stream_completion(
    base_url: str,
    payload: dict[str, Any],
    first_text: threading.Event,
    result: dict[str, Any],
    *,
    timeout: float = 300.0,
) -> None:
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    events: list[dict[str, Any]] = []
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            assert response.status == 200
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                usage = event.get("usage")
                if usage is not None:
                    assert isinstance(usage, dict)
                    result["usage"] = usage
                choices = event.get("choices", [])
                assert isinstance(choices, list)
                if not choices:
                    continue
                text = choices[0].get("text", "")
                assert isinstance(text, str)
                if text:
                    events.append({"at_seconds": time.monotonic(), "text": text})
                    first_text.set()
        result["events"] = events
        result["output_text"] = "".join(event["text"] for event in events)
    except BaseException as exc:
        result["error"] = exc
        first_text.set()


def _run_decode_while_prefilling(
    base_url: str,
    *,
    decode_payload: dict[str, Any],
    prefill_payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    first_text = threading.Event()
    stream_result: dict[str, Any] = {}
    decode_thread = threading.Thread(
        target=_stream_completion,
        args=(base_url, decode_payload, first_text, stream_result),
        daemon=True,
    )
    decode_thread.start()
    assert first_text.wait(timeout=180.0), "decode emitted no SSE text"
    assert "error" not in stream_result, stream_result.get("error")

    prefill_submitted = time.monotonic()
    prefill_response = _post_completion(base_url, prefill_payload)
    prefill_completed = time.monotonic()

    decode_thread.join(timeout=300.0)
    assert not decode_thread.is_alive(), "streaming decode did not finish"
    assert "error" not in stream_result, stream_result.get("error")
    events = stream_result.get("events")
    assert isinstance(events, list) and events
    assert any(
        prefill_submitted <= event["at_seconds"] <= prefill_completed
        for event in events
    ), "decode emitted no token while prefill was active"
    return stream_result, prefill_response


def _run_decode_with_prefill_batch(
    base_url: str,
    *,
    decode_payload: dict[str, Any],
    prefill_payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_text = threading.Event()
    stream_result: dict[str, Any] = {}
    decode_thread = threading.Thread(
        target=_stream_completion,
        args=(base_url, decode_payload, first_text, stream_result),
        daemon=True,
    )
    decode_thread.start()
    assert first_text.wait(timeout=180.0), "decode emitted no SSE text"
    assert "error" not in stream_result, stream_result.get("error")

    prefill_submitted = time.monotonic()
    prefill_responses = _run_nonstream_batch(base_url, prefill_payloads)
    prefill_completed = time.monotonic()

    decode_thread.join(timeout=300.0)
    assert not decode_thread.is_alive(), "streaming decode did not finish"
    assert "error" not in stream_result, stream_result.get("error")
    events = stream_result.get("events")
    assert isinstance(events, list) and events
    assert any(
        prefill_submitted <= event["at_seconds"] <= prefill_completed
        for event in events
    ), "decode emitted no token while ragged prefill was active"
    return stream_result, prefill_responses


def _run_nonstream_batch(
    base_url: str,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    start_gate = threading.Barrier(len(payloads) + 1)
    results: list[dict[str, Any] | BaseException | None] = [None] * len(
        payloads
    )

    def worker(index: int) -> None:
        try:
            start_gate.wait()
            results[index] = _post_completion(
                base_url, payloads[index], timeout=600.0
            )
        except BaseException as exc:
            results[index] = exc

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(len(payloads))
    ]
    for thread in threads:
        thread.start()
    start_gate.wait()
    for thread in threads:
        thread.join(timeout=700.0)
        assert not thread.is_alive(), "concurrent completion did not finish"
    assert all(not isinstance(result, BaseException) for result in results), results
    assert all(isinstance(result, dict) for result in results)
    return [result for result in results if isinstance(result, dict)]


def _run_stream_batch(
    base_url: str,
    payloads: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], float]:
    start_gate = threading.Barrier(len(payloads) + 1)
    results: list[dict[str, Any]] = [{} for _ in payloads]

    def worker(index: int) -> None:
        start_gate.wait()
        results[index]["submitted_at_seconds"] = time.monotonic()
        _stream_completion(
            base_url,
            payloads[index],
            threading.Event(),
            results[index],
            timeout=600.0,
        )
        results[index]["completed_at_seconds"] = time.monotonic()

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(len(payloads))
    ]
    for thread in threads:
        thread.start()
    batch_started = time.monotonic()
    start_gate.wait()
    for thread in threads:
        thread.join(timeout=700.0)
        assert not thread.is_alive(), "concurrent streaming completion did not finish"
    assert all("error" not in result for result in results), [
        result.get("error") for result in results
    ]
    assert all(result.get("events") for result in results)
    assert all(
        result["events"][0]["at_seconds"]
        >= result["submitted_at_seconds"]
        for result in results
    )
    batch_completed = max(
        result["completed_at_seconds"] for result in results
    )
    return results, batch_completed - batch_started


def _observable_usage(response: dict[str, Any]) -> dict[str, int]:
    usage = response.get("usage")
    assert isinstance(usage, dict)
    prompt_details = usage.get("prompt_tokens_details")
    if prompt_details is None:
        cached_tokens = 0
    else:
        assert isinstance(prompt_details, dict)
        cached_tokens = prompt_details.get("cached_tokens")
    observed = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
    }
    assert all(type(value) is int and value >= 0 for value in observed.values())
    return observed


def _assert_fresh_completion_usage(
    response: dict[str, Any],
    *,
    prompt_tokens: int,
    completion_tokens: int,
) -> None:
    _completion_text(response)
    usage = _observable_usage(response)
    assert usage == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": 0,
    }


def _response_observation(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "output_text": _completion_text(response),
        "usage": _observable_usage(response),
    }


def _stream_observation(result: dict[str, Any]) -> dict[str, Any]:
    events = result.get("events")
    output_text = result.get("output_text")
    assert isinstance(events, list) and events
    assert isinstance(output_text, str)
    event_texts = [event["text"] for event in events]
    assert all(isinstance(text, str) and text for text in event_texts)
    assert "".join(event_texts) == output_text
    return {"output_text": output_text, "event_texts": event_texts}


def _exercise_graph_configuration(
    tiny_model: Path,
    prompts: dict[str, Any],
    log_path: Path,
    *,
    cuda_graph_max_bs: int,
) -> dict[str, Any]:
    observations: dict[str, Any] = {"batches": {}}
    with _running_service(
        tiny_model,
        log_path,
        policy="layered-pipeline",
        cache_size=24,
        group_size=2,
        chunks_per_iteration=3,
        max_prefill_length=128,
        cuda_graph_max_bs=cuda_graph_max_bs,
        python_optimize=True,
    ) as (base_url, service_log):
        wave_offset = len(_records(service_log, WAVE_PATTERN))
        for batch_size in (1, 4, 8):
            batch = prompts["batches"][batch_size]
            initial_payloads = [
                _completion_payload(prompt, max_tokens=16, stream=False)
                for prompt in batch["prefixes"]
            ]
            initial_responses = _run_nonstream_batch(base_url, initial_payloads)

            cached_payloads = [
                _completion_payload(prompt, max_tokens=16, stream=False)
                for prompt in batch["followups"]
            ]
            cached_responses = _run_nonstream_batch(base_url, cached_payloads)
            cached_observations = [
                _response_observation(response) for response in cached_responses
            ]
            assert all(
                observation["usage"]["cached_tokens"] > 0
                for observation in cached_observations
            )

            stream_payloads = [
                {**payload, "stream": True} for payload in cached_payloads
            ]
            stream_results, _ = _run_stream_batch(base_url, stream_payloads)
            stream_observations = [
                _stream_observation(result) for result in stream_results
            ]
            assert [
                observation["output_text"]
                for observation in stream_observations
            ] == [
                observation["output_text"]
                for observation in cached_observations
            ]
            observations["batches"][batch_size] = {
                "prompt_length": batch["prompt_length"],
                "initial": [
                    _response_observation(response)
                    for response in initial_responses
                ],
                "cached": cached_observations,
                "stream": stream_observations,
            }

        concurrent_wave_offset = len(_records(service_log, WAVE_PATTERN))
        decode_payload = _completion_payload(
            prompts["concurrent_decode"], max_tokens=64, stream=True
        )
        prefill_payload = _completion_payload(
            prompts["concurrent_prefill"], max_tokens=2, stream=False
        )
        decode_result, prefill_response = _run_decode_while_prefilling(
            base_url,
            decode_payload=decode_payload,
            prefill_payload=prefill_payload,
        )
        pure_decode_response = _post_completion(
            base_url, {**decode_payload, "stream": False}
        )
        pure_decode_observation = _response_observation(pure_decode_response)
        assert pure_decode_observation["usage"]["cached_tokens"] > 0
        assert decode_result["output_text"] == pure_decode_observation[
            "output_text"
        ]

        concurrent_waves = _records(service_log, WAVE_PATTERN)[
            concurrent_wave_offset:
        ]
        prefill_waves = [
            wave for wave in concurrent_waves if wave["chunks"] == 32
        ]
        assert len(prefill_waves) == 1
        concurrent_wave = prefill_waves[0]
        _assert_wave_contract(
            concurrent_wave,
            chunks=32,
            resident_groups=3,
            chunks_per_iteration=3,
            num_layers=5,
        )
        assert concurrent_wave["decode_iterations"] > 0
        assert concurrent_wave["cross_group_prefetches"] == 0
        assert concurrent_wave["deferred_cross_group_prefetches"] == 2

        all_waves = _records(service_log, WAVE_PATTERN)[wave_offset:]
        assert all_waves
        for wave in all_waves:
            _assert_wave_contract(
                wave,
                chunks=wave["chunks"],
                resident_groups=3,
                chunks_per_iteration=3,
                num_layers=5,
                wave_reqs=None,
            )

        observations.update(
            {
                "concurrent_decode": _stream_observation(decode_result),
                "concurrent_prefill": _response_observation(prefill_response),
                "pure_decode": pure_decode_observation,
                "concurrent_wave": concurrent_wave,
            }
        )
    return observations


def _exercise_layered_prefill_graph_configuration(
    scaled_model: Path,
    prompts: dict[str, Any],
    log_path: Path,
    *,
    cuda_graph_max_bs: int,
) -> dict[str, Any]:
    with _running_service(
        scaled_model,
        log_path,
        policy="layered-prefill",
        cache_size=24,
        group_size=2,
        pipeline_wave_max_chunks=4,
        max_prefill_length=32,
        max_seq_len_override=4096,
        cuda_graph_max_bs=cuda_graph_max_bs,
        python_optimize=True,
    ) as (base_url, service_log):
        wave_offset = len(
            _records(service_log, LAYERED_PREFILL_WAVE_PATTERN)
        )
        decode_payload = _completion_payload(
            prompts["driver"], max_tokens=8, stream=True
        )
        decode_payload["stream_options"] = {"include_usage": True}
        prefill_payloads = [
            _completion_payload(prompt, max_tokens=1, stream=False)
            for prompt in prompts["prefill"]
        ]
        decode_result, prefill_responses = (
            _run_decode_with_prefill_batch(
                base_url,
                decode_payload=decode_payload,
                prefill_payloads=prefill_payloads,
            )
        )
        decode_observation = _stream_observation(decode_result)
        decode_usage = _observable_usage(decode_result)
        assert decode_usage == {
            "prompt_tokens": 32,
            "completion_tokens": 8,
            "total_tokens": 40,
            "cached_tokens": 0,
        }
        for response in prefill_responses:
            _assert_fresh_completion_usage(
                response,
                prompt_tokens=56,
                completion_tokens=1,
            )

        waves = _wait_for_layered_prefill_requests(
            service_log,
            offset=wave_offset,
            expected_requests=3,
        )
        assert waves
        driver_wave = waves[0]
        _assert_layered_prefill_wave_contract(
            driver_wave,
            groups=4,
            num_layers=8,
            expected_reqs=1,
        )
        assert driver_wave["decode_iterations"] == 0
        burst_waves = waves[1:]
        assert burst_waves
        assert sum(wave["reqs"] for wave in burst_waves) == 2
        for wave in burst_waves:
            assert wave["reqs"] in (1, 2)
            _assert_layered_prefill_wave_contract(
                wave,
                groups=4,
                num_layers=8,
            )
        assert any(wave["decode_iterations"] > 0 for wave in burst_waves)
        return {
            "decode": {
                **decode_observation,
                "usage": decode_usage,
            },
            "prefill": [
                _response_observation(response)
                for response in prefill_responses
            ],
            "waves": waves,
        }


def _run_lab_benchmark(
    tiny_model: Path,
    output_path: Path,
    *,
    modes: list[str],
    profile: str,
    max_prefill_length: int | None = None,
    repetitions: int = 1,
) -> dict[str, Any]:
    ft_executable = _ft_executable()
    command = [
        sys.executable,
        "benchmarks/bench_lab_agent_policies.py",
        "--ft-executable",
        str(ft_executable),
        "--model",
        str(tiny_model),
        "--modes",
        *modes,
        "--profile",
        profile,
        "--repetitions",
        str(repetitions),
        "--gpu",
        str(torch.cuda.current_device()),
        "--output",
        str(output_path),
    ]
    if max_prefill_length is not None:
        command.extend(["--max-prefill-length", str(max_prefill_length)])
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=1_800.0,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout[-8_000:] + completed.stderr[-8_000:]
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_scaled_benchmark(
    scaled_model: Path,
    output_path: Path,
    *,
    prefill_requests: int,
    prefill_tokens: int,
    max_prefill_length: int,
    repetitions: int = 1,
    prefill_stagger_ms: int = 0,
    mode: str = "layered-pipeline-g1-cpi16-wave64",
) -> dict[str, Any]:
    command = [
        sys.executable,
        "benchmarks/bench_scaled_expert_contention.py",
        "--ft-executable",
        str(_ft_executable()),
        "--model",
        str(scaled_model),
        "--modes",
        mode,
        "--repetitions",
        str(repetitions),
        "--gpu",
        str(torch.cuda.current_device()),
        "--host",
        HOST,
        "--port",
        str(_free_port()),
        "--server-timeout",
        "900",
        "--driver-prompt-tokens",
        "128",
        "--driver-decode-tokens",
        "512",
        "--prefill-requests",
        str(prefill_requests),
        "--prefill-tokens",
        str(prefill_tokens),
        "--prefill-decode-tokens",
        "1",
        "--burst-trigger",
        "first-sse",
        "--prefill-submit-stagger-ms",
        str(prefill_stagger_ms),
        "--max-prefill-length",
        str(max_prefill_length),
        "--moe-cache-size",
        "24",
        "--cuda-graph-max-bs",
        "8",
        "--output",
        str(output_path),
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(python_optimize=True),
        capture_output=True,
        text=True,
        timeout=1_800.0,
        check=False,
    )
    assert completed.returncode == 0, (
        completed.stdout[-8_000:] + completed.stderr[-8_000:]
    )
    return json.loads(output_path.read_text(encoding="utf-8"))


def _ft_executable() -> Path:
    executable = Path(sys.executable).with_name("ft")
    assert executable.is_file()
    assert os.access(executable, os.X_OK)
    return executable


def _assert_successful_benchmark_request(
    request: dict[str, Any], *, require_nonempty_text: bool = True
) -> None:
    assert request["measurement_failed"] is False
    assert "output_mismatch" in request
    submitted_at = request["submitted_at_seconds"]
    output_text = request["output_text"]
    usage = request["usage"]
    events = request["nonempty_text_events"]
    assert isinstance(submitted_at, (int, float)) and submitted_at >= 0
    assert isinstance(output_text, str)
    assert isinstance(usage, dict)
    assert isinstance(events, list)
    if require_nonempty_text:
        assert output_text
        assert events
    for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert type(usage.get(field)) is int and usage[field] >= 0
    assert usage["total_tokens"] == (
        usage["prompt_tokens"] + usage["completion_tokens"]
    )
    for event in events:
        assert event["at_seconds"] >= submitted_at
        assert isinstance(event["text"], str) and event["text"]
    assert "".join(event["text"] for event in events) == output_text


def _scaled_pipeline_mode(report: dict[str, Any]) -> dict[str, Any]:
    required_top_level = {
        "schema",
        "created_at_unix_seconds",
        "model_path",
        "auto_generated_model",
        "model_contract",
        "workload_contract",
        "reference_mode",
        "modes",
    }
    optional_top_level = {"model_path_removed_after_run"}
    assert set(report) in (
        required_top_level,
        required_top_level | optional_top_level,
    )
    assert isinstance(report["model_contract"], dict)
    assert isinstance(report["workload_contract"], dict)
    assert isinstance(report["modes"], list)
    matches = [
        mode
        for mode in report["modes"]
        if mode.get("name") == "layered_pipeline_g1_cpi16_wave64"
    ]
    assert len(matches) == 1
    mode = matches[0]
    required_mode_fields = {
        "name",
        "server_command",
        "repetitions",
        "requests",
        "joint_waves",
        "layered_pipeline_waves",
        "server_log_tail",
        "error",
        "readiness_prompt_token_id",
        "summary",
        "layered_pipeline_structure",
    }
    assert set(mode) == required_mode_fields
    assert mode["error"] in (None, "")
    assert isinstance(mode["joint_waves"], list)
    assert isinstance(mode["layered_pipeline_waves"], list)
    assert isinstance(mode["server_log_tail"], str)
    assert isinstance(mode["summary"], dict)
    return mode


def _scaled_layered_prefill_mode(report: dict[str, Any]) -> dict[str, Any]:
    required_top_level = {
        "schema",
        "created_at_unix_seconds",
        "model_path",
        "auto_generated_model",
        "model_contract",
        "workload_contract",
        "reference_mode",
        "modes",
    }
    optional_top_level = {"model_path_removed_after_run"}
    assert set(report) in (
        required_top_level,
        required_top_level | optional_top_level,
    )
    assert isinstance(report["modes"], list)
    matches = [
        mode
        for mode in report["modes"]
        if str(mode.get("name", "")).replace("-", "_")
        == "layered_prefill_g1_wave64"
    ]
    assert len(matches) == 1
    mode = matches[0]
    required_mode_fields = {
        "name",
        "repetitions",
        "requests",
        "server_log_tail",
        "error",
        "summary",
        "layered_prefill_waves",
        "layered_prefill_structure",
    }
    assert required_mode_fields <= set(mode)
    assert mode["error"] in (None, "")
    assert isinstance(mode["server_log_tail"], str)
    assert isinstance(mode["summary"], dict)
    assert isinstance(mode["layered_prefill_waves"], list)
    assert isinstance(mode["layered_prefill_structure"], dict)
    return mode


def _assert_scaled_accounting(
    mode: dict[str, Any],
    *,
    request_count: int,
    prompt_tokens: int,
    completion_tokens: int,
    repetition_count: int = 1,
    prefill_rows: int | None = None,
    prefill_layer_prepares: int | None = None,
) -> None:
    requests = mode["requests"]
    repetitions = mode["repetitions"]
    assert isinstance(requests, list)
    assert len(requests) == request_count * repetition_count
    assert isinstance(repetitions, list)
    assert len(repetitions) == repetition_count
    for request in requests:
        _assert_successful_benchmark_request(
            request, require_nonempty_text=False
        )
    assert sum(request["usage"]["prompt_tokens"] for request in requests) == (
        prompt_tokens * repetition_count
    )
    assert sum(
        request["usage"]["completion_tokens"] for request in requests
    ) == completion_tokens * repetition_count

    required_repetition_fields = {
        "burst_released_at_seconds",
        "makespan_seconds",
        "request_count",
        "prompt_tokens",
        "decode_tokens",
        "prompt_throughput_tokens_per_second",
        "decode_throughput_tokens_per_second",
        "measurement_failed_requests",
        "moe_stats_before",
        "moe_stats_after",
        "moe_stats_delta",
        "output_mismatch_requests",
    }
    for repetition in repetitions:
        assert required_repetition_fields <= set(repetition)
        assert isinstance(
            repetition["burst_released_at_seconds"], (int, float)
        )
        assert repetition["burst_released_at_seconds"] >= 0
        assert isinstance(repetition["makespan_seconds"], (int, float))
        assert repetition["makespan_seconds"] >= 0
        assert repetition["request_count"] == request_count
        assert repetition["prompt_tokens"] == prompt_tokens
        assert repetition["decode_tokens"] == completion_tokens
        assert repetition["measurement_failed_requests"] == 0
        assert type(repetition["output_mismatch_requests"]) is int
        assert repetition["output_mismatch_requests"] >= 0
        for field in (
            "prompt_throughput_tokens_per_second",
            "decode_throughput_tokens_per_second",
        ):
            value = repetition[field]
            assert type(value) in (int, float)
            assert math.isfinite(value) and value >= 0
        assert isinstance(repetition["moe_stats_before"], dict)
        assert isinstance(repetition["moe_stats_after"], dict)
        delta = repetition["moe_stats_delta"]
        assert isinstance(delta, dict)
        assert set(delta) == SCALED_MOE_STATS_FIELDS
        assert all(
            type(value) is int and value >= 0 for value in delta.values()
        )
        if prefill_rows is not None:
            assert delta["prefill_rows"] == prefill_rows
        if prefill_layer_prepares is not None:
            assert delta["prefill_layer_prepares"] == prefill_layer_prepares


def test_cli_exposes_layered_pipeline_without_removing_existing_policies() -> None:
    help_result = subprocess.run(
        [sys.executable, "-m", "freetoken.cli", "serve", "--help"],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    help_text = help_result.stdout + help_result.stderr
    for policy in (
        "layered-prefill",
        "layered-pipeline",
        "layered",
        "joint",
    ):
        assert re.search(rf"(?<![\w-]){re.escape(policy)}(?![\w-])", help_text)
    assert "--layered-pipeline-chunks-per-iteration" in help_text
    assert "--prefill-wave-max-chunks" in help_text
    assert all(mode not in help_text for mode in RETIRED_PIPELINE_MODES)

    unknown_policy = "unsupported-layered-policy"
    rejected = subprocess.run(
        [
            sys.executable,
            "-m",
            "freetoken.cli",
            "serve",
            "--model-path",
            str(PROJECT_ROOT),
            "--batching-policy",
            unknown_policy,
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert rejected.returncode != 0

    invalid_message = (
        "--layered-pipeline-chunks-per-iteration must be at least 1"
    )
    for invalid_value in (0, -1):
        invalid = subprocess.run(
            _service_command(
                PROJECT_ROOT,
                policy="layered-pipeline",
                port=_free_port(),
                chunks_per_iteration=invalid_value,
            ),
            cwd=PROJECT_ROOT,
            env=_subprocess_env(),
            capture_output=True,
            text=True,
            timeout=30.0,
            check=False,
        )
        assert invalid.returncode != 0
        assert invalid_message in invalid.stdout + invalid.stderr

    invalid_wave = subprocess.run(
        _service_command(
            PROJECT_ROOT,
            policy="layered-pipeline",
            port=_free_port(),
            pipeline_wave_max_chunks=0,
        ),
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert invalid_wave.returncode != 0
    assert (
        "argument --prefill-wave-max-chunks: must be >= 1"
        in invalid_wave.stdout + invalid_wave.stderr
    )


@pytest.mark.parametrize(
    "policy", ["joint", "layered-pipeline", "layered-prefill"]
)
@pytest.mark.parametrize(
    "invalid_options",
    [
        {"max_prefill_length": 0},
        {"group_size": 0},
    ],
    ids=["max-prefill-length-zero", "layer-group-size-zero"],
)
def test_service_rejects_zero_chunk_and_group_boundaries(
    tiny_model: Path,
    policy: str,
    invalid_options: dict[str, int],
) -> None:
    completed = subprocess.run(
        _service_command(
            tiny_model,
            policy=policy,
            port=_free_port(),
            **invalid_options,
        ),
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600.0,
        check=False,
    )
    assert completed.returncode != 0, (
        f"{policy} accepted invalid public options {invalid_options}"
    )


@pytest.mark.parametrize("policy", ["joint", "layered-prefill"])
def test_joint_and_layered_prefill_reject_zero_wave_capacity(
    tiny_model: Path,
    policy: str,
) -> None:
    completed = subprocess.run(
        _service_command(
            tiny_model,
            policy=policy,
            port=_free_port(),
            pipeline_wave_max_chunks=0,
        ),
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600.0,
        check=False,
    )
    assert completed.returncode != 0
    assert (
        "argument --prefill-wave-max-chunks: must be >= 1"
        in completed.stdout + completed.stderr
    )


@pytest.mark.parametrize(
    (
        "requested_mode",
        "canonical_mode",
        "chunks_per_iteration",
        "expected_primary",
    ),
    [
        ("layered-pipeline", "layered_pipeline_g2_cpi1", "1", True),
        (
            "layered-pipeline-cpi1",
            "layered_pipeline_g2_cpi1",
            "1",
            True,
        ),
        (
            "layered-pipeline-cpi2",
            "layered_pipeline_g2_cpi2",
            "2",
            False,
        ),
        (
            "layered-pipeline-cpi3",
            "layered_pipeline_g2_cpi3",
            "3",
            None,
        ),
        (
            "layered_pipeline_g2_cpi1",
            "layered_pipeline_g2_cpi1",
            "1",
            True,
        ),
        (
            "layered_pipeline_g2_cpi2",
            "layered_pipeline_g2_cpi2",
            "2",
            False,
        ),
        (
            "layered_pipeline_g2_cpi3",
            "layered_pipeline_g2_cpi3",
            "3",
            None,
        ),
    ],
)
def test_benchmark_dry_run_resolves_chunk_iteration_modes(
    requested_mode: str,
    canonical_mode: str,
    chunks_per_iteration: str,
    expected_primary: bool | None,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_lab_agent_policies.py",
            "--ft-executable",
            str(_ft_executable()),
            "--modes",
            requested_mode,
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert report["dry_run"] is True
    assert isinstance(report["commands"], list)

    def command_for_mode(mode: str) -> dict[str, Any]:
        matches = [
            command
            for command in report["commands"]
            if command.get("mode") == mode
        ]
        assert len(matches) == 1
        assert isinstance(matches[0].get("primary"), bool)
        if expected_primary is not None:
            assert matches[0]["primary"] is expected_primary
        assert isinstance(matches[0].get("argv"), list)
        assert all(isinstance(argument, str) for argument in matches[0]["argv"])
        return matches[0]

    def assert_adjacent(argv: list[str], flag: str, value: str) -> None:
        assert any(
            argv[index : index + 2] == [flag, value]
            for index in range(len(argv) - 1)
        )

    argv = command_for_mode(canonical_mode)["argv"]
    assert_adjacent(argv, "--batching-policy", "layered-pipeline")
    assert_adjacent(argv, "--prefill-layer-group-size", "2")
    assert_adjacent(
        argv,
        "--layered-pipeline-chunks-per-iteration",
        chunks_per_iteration,
    )


def test_scaled_benchmark_dry_run_resolves_canonical_wave_mode() -> None:
    command = [
        sys.executable,
        "benchmarks/bench_scaled_expert_contention.py",
        "--ft-executable",
        str(_ft_executable()),
        "--modes",
        "layered-pipeline-g1-cpi16-wave64",
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert set(report) == {"model_contract", "workload", "commands"}
    assert isinstance(report["model_contract"], dict)
    assert isinstance(report["workload"], dict)
    commands = report["commands"]
    assert isinstance(commands, list)
    assert len(commands) == 1
    argv = commands[0]
    assert isinstance(argv, list)
    assert all(isinstance(argument, str) for argument in argv)

    def assert_adjacent(flag: str, value: str) -> None:
        assert any(
            argv[index : index + 2] == [flag, value]
            for index in range(len(argv) - 1)
        )

    for flag, value in (
        ("--batching-policy", "layered-pipeline"),
        ("--prefill-layer-group-size", "1"),
        ("--layered-pipeline-chunks-per-iteration", "16"),
        ("--prefill-wave-max-chunks", "64"),
        ("--max-prefill-length", "128"),
        ("--moe-cache-size", "24"),
    ):
        assert_adjacent(flag, value)

    rejected_command = command.copy()
    mode_index = rejected_command.index(
        "layered-pipeline-g1-cpi16-wave64"
    )
    rejected_command[mode_index] = "layered-pipeline-g1-cpi16-wave0"
    rejected = subprocess.run(
        rejected_command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert rejected.returncode != 0
    assert "pipeline wave value must be positive" in (
        rejected.stdout + rejected.stderr
    )


@pytest.mark.parametrize(
    "requested_mode",
    ["layered-prefill-g1-wave64", "layered_prefill_g1_wave64"],
)
def test_scaled_dry_run_resolves_layered_prefill_aliases(
    requested_mode: str,
) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_scaled_expert_contention.py",
            "--ft-executable",
            str(_ft_executable()),
            "--modes",
            requested_mode,
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout)
    assert set(report) == {"model_contract", "workload", "commands"}
    commands = report["commands"]
    assert isinstance(commands, list) and len(commands) == 1
    argv = commands[0]
    assert isinstance(argv, list)
    assert all(isinstance(argument, str) for argument in argv)
    for flag, value in (
        ("--batching-policy", "layered-prefill"),
        ("--prefill-layer-group-size", "1"),
        ("--prefill-wave-max-chunks", "64"),
        ("--max-prefill-length", "128"),
        ("--moe-cache-size", "24"),
        ("--cuda-graph-max-bs", "8"),
        ("--attention-backend", "triton"),
        ("--moe-backend", "offload"),
    ):
        assert any(
            argv[index : index + 2] == [flag, value]
            for index in range(len(argv) - 1)
        ), f"{requested_mode} omitted {flag}={value}"


@pytest.mark.parametrize(
    "retired_mode",
    RETIRED_PIPELINE_MODES,
)
def test_benchmark_rejects_retired_pipeline_wave_modes(retired_mode: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_lab_agent_policies.py",
            "--ft-executable",
            str(_ft_executable()),
            "--modes",
            retired_mode,
            "--dry-run",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert completed.returncode != 0


@pytest.mark.parametrize(
    (
        "max_prefill_length",
        "chunk_distribution",
        "total_chunks",
        "total_steps",
        "cpi1_iterations",
        "cpi2_iterations",
    ),
    [
        (64, {10: 4, 13: 16}, 248, 744, 744, 396),
        (128, {5: 4, 7: 16}, 132, 396, 396, 228),
    ],
)
def test_benchmark_chunk_iterations_preserve_accounting_and_wave_geometry(
    tiny_model: Path,
    tmp_path: Path,
    max_prefill_length: int,
    chunk_distribution: dict[int, int],
    total_chunks: int,
    total_steps: int,
    cpi1_iterations: int,
    cpi2_iterations: int,
) -> None:
    report = _run_lab_benchmark(
        tiny_model,
        tmp_path / f"cpi-t{max_prefill_length}.json",
        modes=["legacy", "layered-pipeline-cpi1", "layered-pipeline-cpi2"],
        profile="main",
        max_prefill_length=max_prefill_length,
    )
    assert "schema" in report
    assert isinstance(report.get("modes"), list)
    modes = {mode["name"]: mode for mode in report["modes"]}
    pipeline_modes = {
        "layered_pipeline_g2_cpi1": (1, cpi1_iterations),
        "layered_pipeline_g2_cpi2": (2, cpi2_iterations),
    }
    assert ({"legacy"} | set(pipeline_modes)) <= set(modes)

    for mode_name in ("legacy", *pipeline_modes):
        mode = modes[mode_name]
        assert mode["error"] in (None, "")
        assert isinstance(mode["summary"], dict)
        assert isinstance(mode["requests"], list)
        assert len(mode["requests"]) == 20
        for request in mode["requests"]:
            _assert_successful_benchmark_request(request)

    legacy_requests = modes["legacy"]["requests"]
    legacy_usages = [request["usage"] for request in legacy_requests]
    legacy_actual_tokens = [
        request["actual_new_prefill_tokens"] for request in legacy_requests
    ]
    for mode_name in pipeline_modes:
        requests = modes[mode_name]["requests"]
        assert [request["usage"] for request in requests] == legacy_usages
        actual_tokens = [
            request["actual_new_prefill_tokens"] for request in requests
        ]
        assert actual_tokens == legacy_actual_tokens
        assert actual_tokens.count(640) == 4
        assert actual_tokens.count(832) == 16
        derived_chunks = [
            math.ceil(token_count / max_prefill_length)
            for token_count in actual_tokens
        ]
        for chunks, count in chunk_distribution.items():
            assert derived_chunks.count(chunks) == count
        assert sum(derived_chunks) == total_chunks

    for mode_name, mode_contract in pipeline_modes.items():
        chunks_per_iteration, expected_iterations = mode_contract
        waves = modes[mode_name]["layered_pipeline_waves"]
        assert len(waves) == 20
        observed_chunks = [wave["chunks"] for wave in waves]
        for chunks, count in chunk_distribution.items():
            assert observed_chunks.count(chunks) == count
        assert sum(observed_chunks) == total_chunks
        assert sum(wave["chunk_group_steps"] for wave in waves) == total_steps
        assert sum(wave["iterations"] for wave in waves) == expected_iterations
        for wave in waves:
            _assert_wave_contract(
                wave,
                chunks=wave["chunks"],
                resident_groups=3,
                chunks_per_iteration=chunks_per_iteration,
                num_layers=5,
            )
        assert any(wave["decode_iterations"] > 0 for wave in waves)


def test_cpi3_group_major_wave_preserves_functional_contract(
    tiny_model: Path,
    tmp_path: Path,
) -> None:
    report = _run_lab_benchmark(
        tiny_model,
        tmp_path / "cpi3-t128-three-repetitions.json",
        modes=["legacy", "layered-pipeline-cpi3"],
        profile="main",
        max_prefill_length=128,
        repetitions=3,
    )
    assert "schema" in report
    assert isinstance(report.get("modes"), list)
    modes = {mode["name"]: mode for mode in report["modes"]}
    pipeline_name = "layered_pipeline_g2_cpi3"
    assert {"legacy", pipeline_name} <= set(modes)

    for mode_name in ("legacy", pipeline_name):
        mode = modes[mode_name]
        assert mode["error"] in (None, "")
        assert isinstance(mode["summary"], dict)
        assert isinstance(mode["requests"], list)
        assert len(mode["requests"]) == 60
        for request in mode["requests"]:
            _assert_successful_benchmark_request(request)

    legacy_requests = modes["legacy"]["requests"]
    pipeline_requests = modes[pipeline_name]["requests"]
    assert [request["usage"] for request in pipeline_requests] == [
        request["usage"] for request in legacy_requests
    ]

    actual_tokens = [
        request["actual_new_prefill_tokens"] for request in pipeline_requests
    ]
    assert actual_tokens == [
        request["actual_new_prefill_tokens"] for request in legacy_requests
    ]
    assert actual_tokens.count(640) == 12
    assert actual_tokens.count(832) == 48
    derived_chunks = [math.ceil(token_count / 128) for token_count in actual_tokens]
    assert derived_chunks.count(5) == 12
    assert derived_chunks.count(7) == 48
    assert sum(derived_chunks) == 396

    waves = modes[pipeline_name]["layered_pipeline_waves"]
    assert len(waves) == 60
    assert sum(wave["chunks"] for wave in waves) == 396
    assert sum(wave["chunk_group_steps"] for wave in waves) == 1_188
    assert sum(wave["iterations"] for wave in waves) == 504
    assert sum(wave["chunks"] == 5 for wave in waves) == 12
    assert sum(wave["chunks"] == 7 for wave in waves) == 48
    for wave in waves:
        _assert_wave_contract(
            wave,
            chunks=wave["chunks"],
            resident_groups=3,
            chunks_per_iteration=3,
            num_layers=5,
        )


def test_scaled_layered_prefill_work_is_independent_of_wave_membership(
    scaled_model: Path,
    tmp_path: Path,
) -> None:
    report = _run_scaled_benchmark(
        scaled_model,
        tmp_path / "scaled-layered-prefill-g1-wave64.json",
        prefill_requests=4,
        prefill_tokens=2_048,
        max_prefill_length=128,
        mode="layered-prefill-g1-wave64",
    )
    mode = _scaled_layered_prefill_mode(report)
    _assert_scaled_accounting(
        mode,
        request_count=5,
        prompt_tokens=8_320,
        completion_tokens=516,
    )
    waves = mode["layered_prefill_waves"]
    assert waves
    assert sum(wave["reqs"] for wave in waves) == 5
    for wave in waves:
        _assert_layered_prefill_wave_contract(
            wave,
            groups=8,
            num_layers=8,
        )
    _assert_layered_prefill_structure(
        mode["layered_prefill_structure"], waves
    )


def test_scaled_canonical_arrival_forms_driver_and_burst_waves(
    scaled_model: Path,
    tmp_path: Path,
) -> None:
    report = _run_scaled_benchmark(
        scaled_model,
        tmp_path / "scaled-canonical-wave64.json",
        prefill_requests=4,
        prefill_tokens=2_048,
        max_prefill_length=128,
    )
    mode = _scaled_pipeline_mode(report)
    _assert_scaled_accounting(
        mode,
        request_count=5,
        prompt_tokens=8_320,
        completion_tokens=516,
        prefill_rows=128,
        prefill_layer_prepares=16,
    )

    waves = mode["layered_pipeline_waves"]
    assert len(waves) == 2
    driver_waves = [
        wave
        for wave in waves
        if wave["wave_reqs"] == 1 and wave["chunks"] == 1
    ]
    burst_waves = [
        wave
        for wave in waves
        if wave["wave_reqs"] == 4 and wave["chunks"] == 64
    ]
    assert len(driver_waves) == 1
    assert len(burst_waves) == 1
    driver_wave = driver_waves[0]
    burst_wave = burst_waves[0]
    _assert_wave_contract(
        driver_wave,
        chunks=1,
        resident_groups=8,
        chunks_per_iteration=16,
        num_layers=8,
    )
    _assert_wave_contract(
        burst_wave,
        chunks=64,
        resident_groups=8,
        chunks_per_iteration=16,
        num_layers=8,
        wave_reqs=4,
        frontier_batches=16,
        frontier_group_forwards=128,
    )
    assert sum(wave["prefill_layer_prepares"] for wave in waves) == 16
    assert sum(wave["iterations"] for wave in waves) == 40

    log_waves = _records_from_text(mode["server_log_tail"], WAVE_PATTERN)
    assert driver_wave in log_waves
    assert burst_wave in log_waves


def test_scaled_t256_completes_without_backend_worker_exit(
    scaled_model: Path,
    tmp_path: Path,
) -> None:
    report = _run_scaled_benchmark(
        scaled_model,
        tmp_path / "scaled-t256-backend-lifecycle.json",
        prefill_requests=4,
        prefill_tokens=2_048,
        max_prefill_length=256,
        repetitions=2,
    )
    mode = _scaled_pipeline_mode(report)
    _assert_scaled_accounting(
        mode,
        request_count=5,
        prompt_tokens=8_320,
        completion_tokens=516,
        repetition_count=2,
        prefill_layer_prepares=16,
    )
    assert re.search(
        r"backend worker.*exit", mode["server_log_tail"], re.IGNORECASE
    ) is None, mode["server_log_tail"]

    waves = mode["layered_pipeline_waves"]
    assert len(waves) == 4
    for repetition_index in range(2):
        repetition_waves = waves[
            repetition_index * 2 : (repetition_index + 1) * 2
        ]
        driver_waves = [
            wave
            for wave in repetition_waves
            if wave["wave_reqs"] == 1 and wave["chunks"] == 1
        ]
        burst_waves = [
            wave
            for wave in repetition_waves
            if wave["wave_reqs"] == 4 and wave["chunks"] == 32
        ]
        assert len(driver_waves) == 1
        assert len(burst_waves) == 1
        _assert_wave_contract(
            driver_waves[0],
            chunks=1,
            resident_groups=8,
            chunks_per_iteration=16,
            num_layers=8,
        )
        burst_wave = burst_waves[0]
        if repetition_index == 0:
            expected_frontier_batches = 8
            expected_frontier_group_forwards = 64
        else:
            assert 8 <= burst_wave["frontier_batches"] <= 32
            group_zero_forwards = burst_wave[
                "frontier_group_forwards"
            ] - (7 * burst_wave["frontier_batches"])
            assert (
                burst_wave["frontier_batches"]
                <= group_zero_forwards
                <= burst_wave["chunks"]
            )
            expected_frontier_batches = burst_wave["frontier_batches"]
            expected_frontier_group_forwards = burst_wave[
                "frontier_group_forwards"
            ]
        assert burst_wave["iterations"] >= 16
        expected_iterations = (
            17 if repetition_index == 0 else burst_wave["iterations"]
        )
        _assert_wave_contract(
            burst_wave,
            chunks=32,
            resident_groups=8,
            chunks_per_iteration=16,
            num_layers=8,
            wave_reqs=4,
            frontier_batches=expected_frontier_batches,
            frontier_group_forwards=expected_frontier_group_forwards,
            expected_iterations=expected_iterations,
            expected_decode_iterations=expected_iterations,
        )
        assert burst_wave["cross_group_prefetches"] == 6
        assert burst_wave["deferred_cross_group_prefetches"] == 0
    assert sum(wave["chunks"] for wave in waves) == 66
    assert sum(wave["prefill_layer_prepares"] for wave in waves) == 32


def test_scaled_wave_soft_cap_keeps_oversized_requests_whole(
    scaled_model: Path,
    tmp_path: Path,
) -> None:
    report = _run_scaled_benchmark(
        scaled_model,
        tmp_path / "scaled-oversized-wave64.json",
        prefill_requests=2,
        prefill_tokens=2_080,
        max_prefill_length=32,
        prefill_stagger_ms=5,
    )
    mode = _scaled_pipeline_mode(report)
    _assert_scaled_accounting(
        mode,
        request_count=3,
        prompt_tokens=4_288,
        completion_tokens=514,
    )

    waves = mode["layered_pipeline_waves"]
    assert len(waves) == 3
    assert sorted(wave["chunks"] for wave in waves) == [4, 65, 65]
    driver_waves = [
        wave
        for wave in waves
        if wave["wave_reqs"] == 1 and wave["chunks"] == 4
    ]
    oversized_waves = [
        wave
        for wave in waves
        if wave["wave_reqs"] == 1 and wave["chunks"] == 65
    ]
    assert len(driver_waves) == 1
    assert len(oversized_waves) == 2
    _assert_wave_contract(
        driver_waves[0],
        chunks=4,
        resident_groups=8,
        chunks_per_iteration=16,
        num_layers=8,
    )
    for wave in oversized_waves:
        _assert_wave_contract(
            wave,
            chunks=65,
            resident_groups=8,
            chunks_per_iteration=16,
            num_layers=8,
        )
    assert sum(wave["iterations"] for wave in waves) == 88

    log_waves = _records_from_text(mode["server_log_tail"], WAVE_PATTERN)
    assert driver_waves[0] in log_waves
    assert sum(wave["chunks"] == 65 for wave in log_waves) >= 2


def test_release_graph8_matches_graph0_for_public_decode_workloads(
    tiny_model: Path,
    graph_workload_prompts: dict[str, Any],
    tmp_path: Path,
) -> None:
    graph_results = {
        graph_size: _exercise_graph_configuration(
            tiny_model,
            graph_workload_prompts,
            tmp_path / f"pipeline-graph{graph_size}.log",
            cuda_graph_max_bs=graph_size,
        )
        for graph_size in (0, 8)
    }
    graph0 = graph_results[0]
    graph8 = graph_results[8]

    def assert_streams_match(
        graph0_stream: dict[str, Any],
        graph8_stream: dict[str, Any],
        completion_tokens: int,
    ) -> None:
        assert graph8_stream["output_text"] == graph0_stream["output_text"]
        graph0_events = graph0_stream["event_texts"]
        graph8_events = graph8_stream["event_texts"]
        if (
            len(graph0_events) == completion_tokens
            and len(graph8_events) == completion_tokens
        ):
            assert graph8_events == graph0_events

    for batch_size, prompt_length in ((1, 640), (4, 2_048), (8, 4_096)):
        graph0_batch = graph0["batches"][batch_size]
        graph8_batch = graph8["batches"][batch_size]
        assert graph0_batch["prompt_length"] == prompt_length
        assert graph8_batch["prompt_length"] == prompt_length
        assert graph8_batch["initial"] == graph0_batch["initial"]
        assert graph8_batch["cached"] == graph0_batch["cached"]
        assert len(graph0_batch["stream"]) == batch_size
        assert len(graph8_batch["stream"]) == batch_size
        for index in range(batch_size):
            completion_tokens = graph0_batch["cached"][index]["usage"][
                "completion_tokens"
            ]
            assert_streams_match(
                graph0_batch["stream"][index],
                graph8_batch["stream"][index],
                completion_tokens,
            )

    assert graph8["concurrent_prefill"] == graph0["concurrent_prefill"]
    assert graph8["pure_decode"] == graph0["pure_decode"]
    assert_streams_match(
        graph0["concurrent_decode"],
        graph8["concurrent_decode"],
        graph0["pure_decode"]["usage"]["completion_tokens"],
    )
    for field in WAVE_FIELDS - {"decode_iterations"}:
        assert graph8["concurrent_wave"][field] == graph0[
            "concurrent_wave"
        ][field]


def test_layered_prefill_graph0_and_graph8_preserve_public_accounting(
    scaled_model: Path,
    scaled_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
) -> None:
    prompts = {
        "driver": scaled_prompt_materializer(
            32,
            600_000,
            "layered-prefill-driver32",
            0,
        ),
        "prefill": [
            scaled_prompt_materializer(
                56,
                600_001 + request_index,
                f"layered-prefill-burst56-request{request_index}",
                1 + request_index,
            )
            for request_index in range(2)
        ],
    }
    results = {
        graph_size: _exercise_layered_prefill_graph_configuration(
            scaled_model,
            prompts,
            tmp_path / f"layered-prefill-graph{graph_size}.log",
            cuda_graph_max_bs=graph_size,
        )
        for graph_size in (0, 8)
    }
    graph0 = results[0]
    graph8 = results[8]
    assert graph8["decode"]["usage"] == graph0["decode"]["usage"]
    for result in results.values():
        decode = result["decode"]
        assert len(decode["event_texts"]) == 8
        assert "".join(decode["event_texts"]) == decode["output_text"]
        assert decode["usage"] == {
            "prompt_tokens": 32,
            "completion_tokens": 8,
            "total_tokens": 40,
            "cached_tokens": 0,
        }
        assert len(result["prefill"]) == 2
        for prefill in result["prefill"]:
            assert isinstance(prefill["output_text"], str)
            assert prefill["usage"] == {
                "prompt_tokens": 56,
                "completion_tokens": 1,
                "total_tokens": 57,
                "cached_tokens": 0,
            }
        assert sum(wave["reqs"] for wave in result["waves"]) == 3
        assert all(
            wave["groups"]
            == wave["group_forwards"]
            == wave["iterations"]
            == 4
            for wave in result["waves"]
        )


def test_layered_prefill_minimum_cache_and_wave_soft_cap(
    tiny_model: Path,
    tiny_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
) -> None:
    max_prefill_length = 32
    with _running_service(
        tiny_model,
        tmp_path / "layered-prefill-minimum-cache-soft-cap.log",
        policy="layered-prefill",
        cache_size=16,
        group_size=1,
        pipeline_wave_max_chunks=3,
        max_prefill_length=max_prefill_length,
    ) as (base_url, service_log):
        wave_offset = len(
            _records(service_log, LAYERED_PREFILL_WAVE_PATTERN)
        )
        oversized_prompt = tiny_prompt_materializer(
            4 * max_prefill_length,
            610_000,
            "layered-prefill-oversized-K4-W3",
            0,
        )
        oversized_response = _post_completion(
            base_url,
            _completion_payload(
                oversized_prompt, max_tokens=1, stream=False
            ),
        )
        _assert_fresh_completion_usage(
            oversized_response,
            prompt_tokens=4 * max_prefill_length,
            completion_tokens=1,
        )
        oversized_waves = _wait_for_layered_prefill_requests(
            service_log,
            offset=wave_offset,
            expected_requests=1,
        )
        assert len(oversized_waves) == 1
        _assert_layered_prefill_wave_contract(
            oversized_waves[0],
            groups=5,
            num_layers=5,
            expected_reqs=1,
        )
        wave_offset += 1

        ragged_chunks = (1, 3)
        ragged_payloads = []
        for request_index, chunk_count in enumerate(ragged_chunks):
            prompt_tokens = chunk_count * max_prefill_length
            prompt = tiny_prompt_materializer(
                prompt_tokens,
                610_100 + request_index,
                (
                    f"layered-prefill-not-fit-W3-K{chunk_count}"
                    f"-request{request_index}"
                ),
                1 + request_index,
            )
            ragged_payloads.append(
                _completion_payload(prompt, max_tokens=1, stream=False)
            )
        ragged_responses = _run_nonstream_batch(base_url, ragged_payloads)
        for response, chunk_count in zip(
            ragged_responses, ragged_chunks, strict=True
        ):
            _assert_fresh_completion_usage(
                response,
                prompt_tokens=chunk_count * max_prefill_length,
                completion_tokens=1,
            )
        ragged_waves = _wait_for_layered_prefill_requests(
            service_log,
            offset=wave_offset,
            expected_requests=2,
        )
        assert len(ragged_waves) == 2
        for wave in ragged_waves:
            _assert_layered_prefill_wave_contract(
                wave,
                groups=5,
                num_layers=5,
                expected_reqs=1,
            )


@pytest.mark.parametrize(
    (
        "max_prefill_length",
        "requested_group_size",
        "cache_size",
        "chunks_per_iteration",
        "expected_group_size",
    ),
    [
        pytest.param(32, 1, 16, 1, 1, id="t32-g1-c16-p1"),
        pytest.param(128, 2, 24, 2, 2, id="t128-g2-c24-p2"),
        pytest.param(256, 4, 40, 4, 4, id="t256-g4-c40-p4"),
        pytest.param(512, 4, 48, 8, 4, id="t512-g4-c48-p8"),
    ],
)
def test_pipeline_token_boundaries_and_shared_pool_geometry(
    tiny_model: Path,
    tiny_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
    max_prefill_length: int,
    requested_group_size: int,
    cache_size: int,
    chunks_per_iteration: int,
    expected_group_size: int,
) -> None:
    num_layers = 5
    num_experts = 8
    assert expected_group_size == min(
        requested_group_size,
        num_layers,
        (cache_size // num_experts) - 1,
    )
    resident_groups = math.ceil(num_layers / expected_group_size)
    token_lengths = (
        max_prefill_length - 1,
        max_prefill_length,
        max_prefill_length + 1,
        (2 * max_prefill_length) - 1,
        2 * max_prefill_length,
        (2 * max_prefill_length) + 1,
    )
    log_name = (
        f"pipeline-t{max_prefill_length}-g{requested_group_size}"
        f"-c{cache_size}-p{chunks_per_iteration}.log"
    )
    with _running_service(
        tiny_model,
        tmp_path / log_name,
        policy="layered-pipeline",
        cache_size=cache_size,
        group_size=requested_group_size,
        chunks_per_iteration=chunks_per_iteration,
        pipeline_wave_max_chunks=64,
        max_prefill_length=max_prefill_length,
    ) as (base_url, service_log):
        geometry = _wait_for_record(
            service_log,
            CACHE_PATTERN,
            lambda record: record["requested_group_size"]
            == requested_group_size,
        )
        assert geometry == {
            "requested_group_size": requested_group_size,
            "effective_group_size": expected_group_size,
            "shared_expert_slots": cache_size,
        }
        wave_offset = len(_records(service_log, WAVE_PATTERN))
        for case_index, prompt_tokens in enumerate(token_lengths):
            case_label = (
                f"T={max_prefill_length}, n={prompt_tokens}, "
                f"G={requested_group_size}, C={cache_size}, "
                f"P={chunks_per_iteration}"
            )
            with _blackbox_case(case_label):
                prompt = tiny_prompt_materializer(
                    prompt_tokens,
                    100_000 + max_prefill_length + case_index,
                    f"boundary-{case_label}",
                    case_index,
                )
                response = _post_completion(
                    base_url,
                    _completion_payload(
                        prompt, max_tokens=1, stream=False
                    ),
                )
                _assert_fresh_completion_usage(
                    response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=1,
                )
                expected_chunks = math.ceil(
                    prompt_tokens / max_prefill_length
                )
                waves = _wait_for_wave_accounting(
                    service_log,
                    WAVE_PATTERN,
                    offset=wave_offset,
                    expected_requests=1,
                    expected_chunks=expected_chunks,
                )
                assert len(waves) == 1
                wave = waves[0]
                _assert_wave_contract(
                    wave,
                    chunks=expected_chunks,
                    resident_groups=resident_groups,
                    chunks_per_iteration=chunks_per_iteration,
                    num_layers=num_layers,
                )
                assert wave["decode_iterations"] == 0
                assert wave["cross_group_prefetches"] == 0
                assert wave["deferred_cross_group_prefetches"] == 0
                wave_offset += 1


@pytest.mark.parametrize(
    ("chunks_per_iteration", "wave_max_chunks", "ragged_chunks"),
    [
        pytest.param(1, 1, (2,), id="p1-w1-concurrency1"),
        pytest.param(2, 3, (1, 2), id="p2-w3-concurrency2"),
        pytest.param(4, 16, (3, 4, 5, 17), id="p4-w16-concurrency4"),
        pytest.param(
            8,
            64,
            (1, 2, 3, 4, 5, 13, 35, 65),
            id="p8-w64-concurrency8",
        ),
    ],
)
def test_pipeline_soft_cap_and_ragged_concurrency_matrix(
    tiny_model: Path,
    tiny_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
    chunks_per_iteration: int,
    wave_max_chunks: int,
    ragged_chunks: tuple[int, ...],
) -> None:
    max_prefill_length = 8
    resident_groups = 3
    log_name = (
        f"pipeline-p{chunks_per_iteration}-w{wave_max_chunks}"
        f"-c{len(ragged_chunks)}.log"
    )
    with _running_service(
        tiny_model,
        tmp_path / log_name,
        policy="layered-pipeline",
        cache_size=24,
        group_size=2,
        chunks_per_iteration=chunks_per_iteration,
        pipeline_wave_max_chunks=wave_max_chunks,
        max_prefill_length=max_prefill_length,
    ) as (base_url, service_log):
        wave_offset = len(_records(service_log, WAVE_PATTERN))
        boundary_chunks = sorted(
            chunk_count
            for chunk_count in {
                wave_max_chunks - 1,
                wave_max_chunks,
                wave_max_chunks + 1,
            }
            if chunk_count > 0
        )
        for case_index, chunk_count in enumerate(boundary_chunks):
            prompt_tokens = chunk_count * max_prefill_length
            case_label = (
                f"single UID P={chunks_per_iteration}, "
                f"W={wave_max_chunks}, K={chunk_count}"
            )
            with _blackbox_case(case_label):
                prompt = tiny_prompt_materializer(
                    prompt_tokens,
                    200_000 + (wave_max_chunks * 100) + case_index,
                    case_label,
                    case_index,
                )
                response = _post_completion(
                    base_url,
                    _completion_payload(
                        prompt, max_tokens=1, stream=False
                    ),
                )
                _assert_fresh_completion_usage(
                    response,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=1,
                )
                waves = _wait_for_wave_accounting(
                    service_log,
                    WAVE_PATTERN,
                    offset=wave_offset,
                    expected_requests=1,
                    expected_chunks=chunk_count,
                )
                assert len(waves) == 1
                _assert_wave_contract(
                    waves[0],
                    chunks=chunk_count,
                    resident_groups=resident_groups,
                    chunks_per_iteration=chunks_per_iteration,
                    num_layers=5,
                )
                assert waves[0]["decode_iterations"] == 0
                assert waves[0]["cross_group_prefetches"] == 0
                assert waves[0]["deferred_cross_group_prefetches"] == 0
                _assert_wave_soft_cap(
                    waves, wave_max_chunks=wave_max_chunks
                )
                wave_offset += 1

        batch_payloads: list[dict[str, Any]] = []
        batch_prompt_tokens: list[int] = []
        for request_index, chunk_count in enumerate(ragged_chunks):
            prompt_tokens = chunk_count * max_prefill_length
            prompt = tiny_prompt_materializer(
                prompt_tokens,
                300_000 + (wave_max_chunks * 100) + request_index,
                (
                    f"ragged-P{chunks_per_iteration}-W{wave_max_chunks}"
                    f"-K{chunk_count}-request{request_index}"
                ),
                16 + request_index,
            )
            batch_prompt_tokens.append(prompt_tokens)
            batch_payloads.append(
                _completion_payload(prompt, max_tokens=1, stream=False)
            )

        batch_responses = _run_nonstream_batch(base_url, batch_payloads)
        for response, prompt_tokens in zip(
            batch_responses, batch_prompt_tokens, strict=True
        ):
            _assert_fresh_completion_usage(
                response,
                prompt_tokens=prompt_tokens,
                completion_tokens=1,
            )
        batch_waves = _wait_for_wave_accounting(
            service_log,
            WAVE_PATTERN,
            offset=wave_offset,
            expected_requests=len(ragged_chunks),
            expected_chunks=sum(ragged_chunks),
        )
        _assert_wave_soft_cap(
            batch_waves, wave_max_chunks=wave_max_chunks
        )
        for wave in batch_waves:
            _assert_dynamic_pipeline_wave_contract(
                wave,
                resident_groups=resident_groups,
                chunks_per_iteration=chunks_per_iteration,
                num_layers=5,
            )
        oversized_chunks = [
            chunk_count
            for chunk_count in ragged_chunks
            if chunk_count > wave_max_chunks
        ]
        for chunk_count in oversized_chunks:
            assert any(
                wave["chunks"] == chunk_count and wave["wave_reqs"] == 1
                for wave in batch_waves
            )


@pytest.mark.parametrize(
    (
        "max_prefill_length",
        "requested_group_size",
        "cache_size",
        "wave_max_chunks",
        "ragged_chunks",
    ),
    [
        pytest.param(32, 1, 8, 3, (1, 2), id="g1-c8-w3-concurrency2"),
        pytest.param(
            128,
            4,
            32,
            16,
            (1, 2, 3, 4),
            id="g4-c32-w16-concurrency4",
        ),
    ],
)
def test_joint_wave_geometry_for_single_and_coalesced_requests(
    tiny_model: Path,
    tiny_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
    max_prefill_length: int,
    requested_group_size: int,
    cache_size: int,
    wave_max_chunks: int,
    ragged_chunks: tuple[int, ...],
) -> None:
    num_layers = 5
    num_experts = 8
    effective_group_size = min(
        requested_group_size,
        num_layers,
        cache_size // num_experts,
    )
    with _running_service(
        tiny_model,
        tmp_path
        / (
            f"joint-t{max_prefill_length}-g{requested_group_size}"
            f"-c{cache_size}-w{wave_max_chunks}.log"
        ),
        policy="joint",
        cache_size=cache_size,
        group_size=requested_group_size,
        pipeline_wave_max_chunks=wave_max_chunks,
        max_prefill_length=max_prefill_length,
    ) as (base_url, service_log):
        wave_offset = len(_records(service_log, JOINT_WAVE_PATTERN))
        single_chunks = 2
        single_prompt_tokens = single_chunks * max_prefill_length
        single_prompt = tiny_prompt_materializer(
            single_prompt_tokens,
            400_000 + max_prefill_length,
            (
                f"joint-single-T{max_prefill_length}"
                f"-G{requested_group_size}-C{cache_size}"
            ),
            0,
        )
        single_response = _post_completion(
            base_url,
            _completion_payload(single_prompt, max_tokens=1, stream=False),
        )
        _assert_fresh_completion_usage(
            single_response,
            prompt_tokens=single_prompt_tokens,
            completion_tokens=1,
        )
        single_waves = _wait_for_wave_accounting(
            service_log,
            JOINT_WAVE_PATTERN,
            offset=wave_offset,
            expected_requests=1,
            expected_chunks=single_chunks,
        )
        assert len(single_waves) == 1
        _assert_joint_wave_contract(
            single_waves[0],
            chunks=single_chunks,
            wave_reqs=1,
            frontier_batches=single_chunks,
            num_layers=num_layers,
            effective_group_size=effective_group_size,
        )
        wave_offset += 1

        payloads: list[dict[str, Any]] = []
        prompt_lengths: list[int] = []
        for request_index, chunk_count in enumerate(ragged_chunks):
            prompt_tokens = chunk_count * max_prefill_length
            prompt_lengths.append(prompt_tokens)
            prompt = tiny_prompt_materializer(
                prompt_tokens,
                410_000 + max_prefill_length + request_index,
                (
                    f"joint-ragged-T{max_prefill_length}"
                    f"-K{chunk_count}-request{request_index}"
                ),
                8 + request_index,
            )
            payloads.append(
                _completion_payload(prompt, max_tokens=1, stream=False)
            )
        responses = _run_nonstream_batch(base_url, payloads)
        for response, prompt_tokens in zip(
            responses, prompt_lengths, strict=True
        ):
            _assert_fresh_completion_usage(
                response,
                prompt_tokens=prompt_tokens,
                completion_tokens=1,
            )
        batch_waves = _wait_for_wave_accounting(
            service_log,
            JOINT_WAVE_PATTERN,
            offset=wave_offset,
            expected_requests=len(ragged_chunks),
            expected_chunks=sum(ragged_chunks),
        )
        _assert_wave_soft_cap(
            batch_waves, wave_max_chunks=wave_max_chunks
        )
        assert max(ragged_chunks) <= sum(
            wave["frontier_batches"] for wave in batch_waves
        ) <= sum(ragged_chunks)
        for wave in batch_waves:
            assert 1 <= wave["frontier_batches"] <= wave["chunks"]
            _assert_joint_wave_contract(
                wave,
                chunks=wave["chunks"],
                wave_reqs=wave["wave_reqs"],
                frontier_batches=wave["frontier_batches"],
                num_layers=num_layers,
                effective_group_size=effective_group_size,
            )


def test_pipeline_request_lifecycles_remain_independent(
    tiny_model: Path,
    tiny_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
) -> None:
    max_prefill_length = 32
    prompt_lengths = (31, 32, 33, 65)
    completion_limits = (1, 2, 4, 8)
    prompts = [
        tiny_prompt_materializer(
            prompt_tokens,
            500_000 + request_index,
            f"lifecycle-request{request_index}-n{prompt_tokens}",
            request_index,
        )
        for request_index, prompt_tokens in enumerate(prompt_lengths)
    ]
    with _running_service(
        tiny_model,
        tmp_path / "pipeline-independent-request-lifecycles.log",
        policy="layered-pipeline",
        cache_size=24,
        group_size=2,
        chunks_per_iteration=4,
        pipeline_wave_max_chunks=16,
        max_prefill_length=max_prefill_length,
        cuda_graph_max_bs=8,
        python_optimize=True,
    ) as (base_url, service_log):
        wave_offset = len(_records(service_log, WAVE_PATTERN))
        responses = _run_nonstream_batch(
            base_url,
            [
                _completion_payload(
                    prompt,
                    max_tokens=completion_limit,
                    stream=False,
                )
                for prompt, completion_limit in zip(
                    prompts, completion_limits, strict=True
                )
            ],
        )
        for request_index, response in enumerate(responses):
            with _blackbox_case(
                f"independent finish request={request_index}, "
                f"max_tokens={completion_limits[request_index]}"
            ):
                _completion_text(response)
                usage = _observable_usage(response)
                assert usage["prompt_tokens"] == prompt_lengths[request_index]
                assert usage["cached_tokens"] == 0
                assert 1 <= usage["completion_tokens"] <= (
                    completion_limits[request_index]
                )
                assert usage["total_tokens"] == (
                    usage["prompt_tokens"] + usage["completion_tokens"]
                )
        assert _observable_usage(responses[0])["completion_tokens"] == 1
        expected_chunks = sum(
            math.ceil(prompt_tokens / max_prefill_length)
            for prompt_tokens in prompt_lengths
        )
        lifecycle_waves = _wait_for_wave_accounting(
            service_log,
            WAVE_PATTERN,
            offset=wave_offset,
            expected_requests=len(prompts),
            expected_chunks=expected_chunks,
        )
        _assert_wave_soft_cap(lifecycle_waves, wave_max_chunks=16)
        for wave in lifecycle_waves:
            _assert_dynamic_pipeline_wave_contract(
                wave,
                resident_groups=3,
                chunks_per_iteration=4,
                num_layers=5,
            )

        cached_payload = _completion_payload(
            prompts[0], max_tokens=completion_limits[0], stream=False
        )
        cached_response = _post_completion(base_url, cached_payload)
        cached_usage = _observable_usage(cached_response)
        assert cached_usage["prompt_tokens"] == prompt_lengths[0]
        assert cached_usage["cached_tokens"] > 0
        assert cached_usage["total_tokens"] == (
            cached_usage["prompt_tokens"]
            + cached_usage["completion_tokens"]
        )
        assert re.search(
            r"backend worker.*exit", _log_tail(service_log), re.IGNORECASE
        ) is None, _log_tail(service_log)


@pytest.mark.parametrize("policy", ["layered", "joint"])
def test_existing_policies_still_serve_completions(
    tiny_model: Path,
    public_prompts: dict[str, str],
    tmp_path: Path,
    policy: str,
) -> None:
    with _running_service(
        tiny_model, tmp_path / f"{policy}.log", policy=policy
    ) as (base_url, _):
        response = _post_completion(
            base_url,
            _completion_payload(
                public_prompts["pure_decode"], max_tokens=2, stream=False
            ),
        )
        _completion_text(response)


@pytest.mark.parametrize(
    ("cache_size", "expected_cross", "expected_deferred"),
    [(24, 0, 2), (40, 2, 0)],
)
def test_layered_pipeline_prefetch_counts_with_continuous_decode(
    tiny_model: Path,
    public_prompts: dict[str, str],
    tmp_path: Path,
    cache_size: int,
    expected_cross: int,
    expected_deferred: int,
) -> None:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(tiny_model)
    num_layers = int(config.num_hidden_layers)
    assert num_layers == 5
    assert int(config.num_experts) == 8
    effective_group_size = 2
    resident_groups = math.ceil(num_layers / effective_group_size)
    assert resident_groups == 3
    max_prefill_length = 64
    prefill_chunks = math.ceil(
        PROMPT_LENGTHS["concurrent_prefill"] / max_prefill_length
    )
    expected_steps = prefill_chunks * resident_groups
    expected_iterations = resident_groups * prefill_chunks

    baseline_payloads = {
        "concurrent_decode": _completion_payload(
            public_prompts["concurrent_decode"], max_tokens=192, stream=False
        ),
        "concurrent_prefill": _completion_payload(
            public_prompts["concurrent_prefill"], max_tokens=1, stream=False
        ),
        "pure_decode": _completion_payload(
            public_prompts["pure_decode"], max_tokens=64, stream=False
        ),
        "pure_prefill": _completion_payload(
            public_prompts["pure_prefill"], max_tokens=1, stream=False
        ),
    }
    baseline_outputs: dict[str, str] = {}
    with _running_service(
        tiny_model,
        tmp_path / f"pipeline-c{cache_size}-baseline.log",
        policy="layered-pipeline",
        cache_size=cache_size,
        max_prefill_length=max_prefill_length,
    ) as (base_url, _):
        for name, payload in baseline_payloads.items():
            response = _post_completion(base_url, payload)
            baseline_outputs[name] = _completion_text(response)
            if name == "concurrent_decode":
                completion_tokens = response["usage"].get("completion_tokens")
                assert isinstance(completion_tokens, int)
                assert completion_tokens > expected_iterations

    pipeline_log = tmp_path / f"pipeline-c{cache_size}-concurrent.log"
    with _running_service(
        tiny_model,
        pipeline_log,
        policy="layered-pipeline",
        cache_size=cache_size,
        max_prefill_length=max_prefill_length,
    ) as (base_url, log_path):
        geometry = _wait_for_record(
            log_path,
            CACHE_PATTERN,
            lambda record: record["requested_group_size"]
            == effective_group_size,
        )
        assert geometry == {
            "requested_group_size": effective_group_size,
            "effective_group_size": effective_group_size,
            "shared_expert_slots": cache_size,
        }
        stream_result, prefill_response = _run_decode_while_prefilling(
            base_url,
            decode_payload=_completion_payload(
                public_prompts["concurrent_decode"],
                max_tokens=192,
                stream=True,
            ),
            prefill_payload=baseline_payloads["concurrent_prefill"],
        )
        assert stream_result["output_text"] == baseline_outputs[
            "concurrent_decode"
        ]
        assert _completion_text(prefill_response) == baseline_outputs[
            "concurrent_prefill"
        ]

        wave = _wait_for_record(
            log_path,
            WAVE_PATTERN,
            lambda record: record["chunks"] == prefill_chunks
            and record["resident_groups"] == resident_groups
            and record["chunk_group_steps"] == expected_steps,
        )
        _assert_wave_contract(
            wave,
            chunks=prefill_chunks,
            resident_groups=resident_groups,
            chunks_per_iteration=1,
            num_layers=num_layers,
        )
        assert 0 < wave["decode_iterations"] <= expected_iterations
        assert wave["cross_group_prefetches"] == expected_cross
        assert wave["deferred_cross_group_prefetches"] == expected_deferred

        pure_decode = _post_completion(base_url, baseline_payloads["pure_decode"])
        pure_prefill = _post_completion(
            base_url, baseline_payloads["pure_prefill"]
        )
        assert _completion_text(pure_decode) == baseline_outputs["pure_decode"]
        assert _completion_text(pure_prefill) == baseline_outputs["pure_prefill"]


def test_layered_pipeline_rejects_one_expert_layer_of_shared_cache(
    tiny_model: Path,
) -> None:
    rejected = subprocess.run(
        _service_command(
            tiny_model,
            policy="layered-pipeline",
            port=_free_port(),
            cache_size=8,
            group_size=99,
        ),
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600.0,
        check=False,
    )
    assert rejected.returncode != 0
    assert (
        "layered-pipeline requires at least two expert layers of shared cache"
        in rejected.stdout + rejected.stderr
    )


def test_layered_prefill_rejects_one_expert_layer_of_shared_cache(
    tiny_model: Path,
) -> None:
    rejected = subprocess.run(
        _service_command(
            tiny_model,
            policy="layered-prefill",
            port=_free_port(),
            cache_size=8,
            group_size=1,
            pipeline_wave_max_chunks=3,
            max_prefill_length=32,
        ),
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=600.0,
        check=False,
    )
    assert rejected.returncode != 0


def test_layered_pipeline_runs_with_two_expert_layers_of_shared_cache(
    tiny_model: Path, public_prompts: dict[str, str], tmp_path: Path
) -> None:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(tiny_model)
    num_layers = int(config.num_hidden_layers)
    assert num_layers == 5
    num_experts = int(config.num_experts)
    assert num_experts == 8
    cache_size = 2 * num_experts
    requested_group_size = 99
    max_prefill_length = 64
    chunks_per_iteration = 2
    prefill_chunks = math.ceil(
        PROMPT_LENGTHS["concurrent_prefill"] / max_prefill_length
    )
    resident_groups = num_layers
    expected_steps = prefill_chunks * resident_groups
    expected_iterations = resident_groups * math.ceil(
        prefill_chunks / chunks_per_iteration
    )
    decode_payload = _completion_payload(
        public_prompts["concurrent_decode"], max_tokens=192, stream=False
    )
    prefill_payload = _completion_payload(
        public_prompts["concurrent_prefill"], max_tokens=1, stream=False
    )

    with _running_service(
        tiny_model,
        tmp_path / "minimum-shared-pool-baseline.log",
        policy="layered-pipeline",
        cache_size=cache_size,
        group_size=requested_group_size,
        chunks_per_iteration=chunks_per_iteration,
        max_prefill_length=max_prefill_length,
    ) as (base_url, _):
        baseline_decode = _completion_text(
            _post_completion(base_url, decode_payload)
        )
        baseline_prefill = _completion_text(
            _post_completion(base_url, prefill_payload)
        )

    log_path = tmp_path / "minimum-shared-pool-concurrent.log"
    with _running_service(
        tiny_model,
        log_path,
        policy="layered-pipeline",
        cache_size=cache_size,
        group_size=requested_group_size,
        chunks_per_iteration=chunks_per_iteration,
        max_prefill_length=max_prefill_length,
    ) as (base_url, service_log):
        geometry = _wait_for_record(
            service_log,
            CACHE_PATTERN,
            lambda record: record["requested_group_size"]
            == requested_group_size,
        )
        assert geometry == {
            "requested_group_size": requested_group_size,
            "effective_group_size": 1,
            "shared_expert_slots": cache_size,
        }
        stream_result, prefill_response = _run_decode_while_prefilling(
            base_url,
            decode_payload={**decode_payload, "stream": True},
            prefill_payload=prefill_payload,
        )
        assert stream_result["output_text"] == baseline_decode
        assert _completion_text(prefill_response) == baseline_prefill

        wave = _wait_for_record(
            service_log,
            WAVE_PATTERN,
            lambda record: record["chunks"] == prefill_chunks
            and record["resident_groups"] == resident_groups
            and record["chunk_group_steps"] == expected_steps,
        )
        _assert_wave_contract(
            wave,
            chunks=prefill_chunks,
            resident_groups=resident_groups,
            chunks_per_iteration=chunks_per_iteration,
            num_layers=num_layers,
        )
        assert 0 < wave["decode_iterations"] <= expected_iterations
