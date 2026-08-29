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
HOST = "127.0.0.1"
SERVED_MODEL = "lab-agent-qwen3-moe"
NUM_LAYERS = 8
NUM_EXPERTS = 8
QWEN36_MODEL_PATH = Path("/data1/lmcache_kv/models/Qwen3.6-35B-A3B")
DSV4_MODEL_PATH = Path("/data1/lmcache_kv/models/DeepSeek-V4-Flash-0731")
DSV4_NOWAG_EXPERT_PATH = Path(
    "/data1/lmcache_kv/nowag_4090_experiment/quantized/"
    "dsv4_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
)
DSV4_NOWAG_PLUGIN_SRC = Path(
    "/home/nengneng/AIPrometheus/servebig/servebig-project/"
    ".kernel-worktrees/nowag_final2_tail64_profile/src"
)

for import_root in (str(PYTHON_ROOT), str(PROJECT_ROOT)):
    if import_root not in sys.path:
        sys.path.insert(0, import_root)

PIPELINE_WAVE_PATTERN = re.compile(
    r"Layered pipeline wave complete: "
    r"reqs=(?P<reqs>\d+), "
    r"groups=(?P<groups>\d+), "
    r"group_forwards=(?P<group_forwards>\d+), "
    r"iterations=(?P<iterations>\d+), "
    r"decode_iterations=(?P<decode_iterations>\d+), "
    r"prefill_layer_prepares=(?P<prefill_layer_prepares>\d+)"
    r"(?:\r?\n|$)"
)
PIPELINE_ITERATION_LIMIT_PATTERN = re.compile(
    r"Layered pipeline iteration limit: "
    r"requested_tokens=(?P<requested_tokens>\d+), "
    r"effective_tokens=(?P<effective_tokens>\d+), "
    r"event=(?P<event>startup|cache_rebuild|rollback)"
    r"(?:\r?\n|$)"
)
PIPELINE_WAVE_FIELDS = {
    "reqs",
    "groups",
    "group_forwards",
    "iterations",
    "decode_iterations",
    "prefill_layer_prepares",
}
PIPELINE_CROSS_GRAPH_STRUCTURE_FIELDS = {
    "reqs",
    "groups",
    "group_forwards",
    "iterations",
    "prefill_layer_prepares",
}
MOE_STATS_FIELDS = {
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
REMOVED_MODE_NAMES = (
    "layered-pipeline-g1-cpi16-wave64",
    "layered_pipeline_g1_cpi16_wave64",
    "layered-prefill-g1-wave64",
    "layered_prefill_g1_wave64",
)
SHARED_POOL_MISS_PROMPT_SPECS = {
    "readiness": (8, 91_000, "shared-pool-readiness", 4),
    "driver_0": (128, 91_101, "shared-pool-driver-0", 0),
    "driver_1": (128, 91_102, "shared-pool-driver-1", 1),
    "driver_2": (128, 91_103, "shared-pool-driver-2", 2),
    "prefill": (2_048, 91_201, "shared-pool-prefill", 3),
}
_CLAIMED_SERVICE_PORTS: set[int] = set()
_PORT_SELECTION_LOCK = threading.Lock()


def _subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    python_paths = [str(PYTHON_ROOT), str(PROJECT_ROOT)]
    if env.get("PYTHONPATH"):
        python_paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_paths)
    return env


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
def scaled_prompt_materializer(
    scaled_model: Path,
) -> Callable[[int, int, str, int], str]:
    return _public_prompt_materializer(scaled_model)


def _ft_executable() -> Path:
    executable = Path(sys.executable).with_name("ft")
    assert executable.is_file()
    assert os.access(executable, os.X_OK)
    return executable


def _free_port() -> int:
    with _PORT_SELECTION_LOCK:
        for _ in range(200):
            with socket.socket(
                socket.AF_INET, socket.SOCK_STREAM
            ) as api_socket:
                api_socket.bind(("0.0.0.0", 0))
                port = int(api_socket.getsockname()[1])
                required_ports = {port, port + 1}
                if port == 65_535 or (
                    required_ports & _CLAIMED_SERVICE_PORTS
                ):
                    continue
                with socket.socket(
                    socket.AF_INET, socket.SOCK_STREAM
                ) as backend_socket:
                    try:
                        backend_socket.bind(("0.0.0.0", port + 1))
                        api_socket.listen(1)
                        backend_socket.listen(1)
                    except OSError:
                        continue
                    _CLAIMED_SERVICE_PORTS.update(required_ports)
                    return port
    raise RuntimeError("no consecutive API/backend TCP port pair is available")


def _service_command(
    model_path: Path,
    *,
    port: int,
    group_size: int,
    wave_max_chunks: int,
    max_prefill_length: int,
    cache_size: int,
    cuda_graph_max_bs: int,
    policy: str = "layered-pipeline",
    collect_moe_stats: bool = False,
) -> list[str]:
    command = [
        str(_ft_executable()),
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
        "16",
        "--max-seq-len-override",
        "4096",
        "--enable-cache-report",
        "--batching-policy",
        policy,
        "--prefill-layer-group-size",
        str(group_size),
        "--prefill-wave-max-chunks",
        str(wave_max_chunks),
        "--tensor-parallel-size",
        "1",
        "--disable-pynccl",
    ]
    if collect_moe_stats:
        command.append("--moe-collect-stats")
    return command


def _real_model_service_command(
    model_path: Path,
    *,
    port: int,
    attention_backend: str,
    cuda_graph_max_bs: int,
    num_tokens: int | None,
    max_prefill_length: int = 32,
    max_seq_len_override: int = 4_096,
    wave_max_chunks: int = 8,
    max_running_requests: int = 8,
    memory_ratio: float | None = None,
) -> list[str]:
    command = [
        str(_ft_executable()),
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
        "bfloat16",
        "--max-running-requests",
        str(max_running_requests),
        "--max-seq-len-override",
        str(max_seq_len_override),
        "--max-prefill-length",
        str(max_prefill_length),
        "--attention-backend",
        attention_backend,
        "--moe-backend",
        "offload",
        "--moe-cache-size",
        "512",
        "--cuda-graph-max-bs",
        str(cuda_graph_max_bs),
        "--cache-type",
        "radix",
        "--enable-cache-report",
        "--batching-policy",
        "layered-pipeline",
        "--prefill-layer-group-size",
        "1",
        "--prefill-wave-max-chunks",
        str(wave_max_chunks),
        "--moe-collect-stats",
    ]
    insertion_index = command.index("--max-prefill-length")
    planner_arguments = []
    if num_tokens is not None:
        planner_arguments.extend(["--num-tokens", str(num_tokens)])
    if memory_ratio is not None:
        planner_arguments.extend(["--memory-ratio", str(memory_ratio)])
    command[insertion_index:insertion_index] = planner_arguments
    return command


def _completion_payload(
    prompt: str,
    *,
    max_tokens: int,
    stream: bool,
    ignore_eos: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": SERVED_MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    if ignore_eos:
        payload["ignore_eos"] = True
    if stream:
        payload["stream_options"] = {"include_usage": True}
    return payload


def _post_completion(
    base_url: str,
    payload: dict[str, Any],
    *,
    timeout: float = 900.0,
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


def _stream_completion(
    base_url: str,
    payload: dict[str, Any],
    first_text: threading.Event,
    result: dict[str, Any],
    *,
    timeout: float = 900.0,
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
                    events.append(
                        {"at_seconds": time.monotonic(), "text": text}
                    )
                    first_text.set()
        result["events"] = events
        result["output_text"] = "".join(
            event["text"] for event in events
        )
    except BaseException as exc:
        result["error"] = exc
        first_text.set()


def _usage(observation: dict[str, Any]) -> dict[str, int]:
    usage = observation.get("usage")
    assert isinstance(usage, dict)
    prompt_details = usage.get("prompt_tokens_details")
    if prompt_details is None:
        cached_tokens = 0
    else:
        assert isinstance(prompt_details, dict)
        cached_tokens = prompt_details.get("cached_tokens", 0)
    normalized = {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
        "cached_tokens": cached_tokens,
    }
    assert all(
        type(value) is int and value >= 0 for value in normalized.values()
    )
    assert normalized["total_tokens"] == (
        normalized["prompt_tokens"] + normalized["completion_tokens"]
    )
    return normalized


def _completion_text(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    assert isinstance(choices, list) and choices
    text = choices[0].get("text")
    assert isinstance(text, str)
    return text


def _assert_nonstream_completion(
    response: dict[str, Any],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> str:
    text = _completion_text(response)
    assert _usage(response) == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
    }
    return text


def _assert_stream_completion(
    result: dict[str, Any],
    *,
    prompt_tokens: int,
    completion_tokens: int,
    cached_tokens: int = 0,
) -> str:
    assert "error" not in result, result.get("error")
    events = result.get("events")
    output_text = result.get("output_text")
    assert isinstance(events, list) and events
    assert isinstance(output_text, str)
    event_times = []
    for event in events:
        assert set(event) == {"at_seconds", "text"}
        assert type(event["at_seconds"]) is float
        assert event["at_seconds"] >= 0
        assert isinstance(event["text"], str) and event["text"]
        event_times.append(event["at_seconds"])
    assert event_times == sorted(event_times)
    assert output_text == "".join(event["text"] for event in events)
    assert _usage(result) == {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "cached_tokens": cached_tokens,
    }
    return output_text


def _log_text(log_path: Path) -> str:
    if not log_path.exists():
        return ""
    return log_path.read_text(encoding="utf-8", errors="replace")


def _wait_until_ready(
    process: subprocess.Popen[str],
    base_url: str,
    log_path: Path,
    readiness_prompt_text: str,
) -> None:
    deadline = time.monotonic() + 600.0
    last_error = "service has not accepted a request"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"service exited with {process.returncode}:\n"
                f"{_log_text(log_path)[-12_000:]}"
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
                _completion_payload(
                    readiness_prompt_text, max_tokens=1, stream=False
                ),
                timeout=30.0,
            )
            return
        except urllib.error.HTTPError as exc:
            last_error = f"HTTP {exc.code}"
        except (urllib.error.URLError, TimeoutError, AssertionError) as exc:
            last_error = str(exc)
        time.sleep(0.1)
    raise AssertionError(
        f"service did not become ready ({last_error}):\n"
        f"{_log_text(log_path)[-12_000:]}"
    )


@contextmanager
def _running_service(
    model_path: Path,
    log_path: Path,
    *,
    group_size: int,
    wave_max_chunks: int,
    max_prefill_length: int,
    cache_size: int,
    cuda_graph_max_bs: int = 8,
    readiness_prompt_text: str = "ready",
) -> Iterator[tuple[str, Path, subprocess.Popen[str]]]:
    port = _free_port()
    command = _service_command(
        model_path,
        port=port,
        group_size=group_size,
        wave_max_chunks=wave_max_chunks,
        max_prefill_length=max_prefill_length,
        cache_size=cache_size,
        cuda_graph_max_bs=cuda_graph_max_bs,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=_subprocess_env(),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        base_url = f"http://{HOST}:{port}"
        try:
            _wait_until_ready(
                process,
                base_url,
                log_path,
                readiness_prompt_text,
            )
            _wait_for_pipeline_requests(
                log_path,
                offset=0,
                expected_requests=1,
            )
            yield base_url, log_path, process
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=30.0)


def _records_from_text(content: str) -> list[dict[str, int]]:
    return [
        {name: int(value) for name, value in match.groupdict().items()}
        for match in PIPELINE_WAVE_PATTERN.finditer(content)
    ]


def _records(log_path: Path) -> list[dict[str, int]]:
    return _records_from_text(_log_text(log_path))


def _iteration_limits_from_text(
    content: str,
) -> list[dict[str, int | str]]:
    limits: list[dict[str, int | str]] = []
    for match in PIPELINE_ITERATION_LIMIT_PATTERN.finditer(content):
        requested_tokens = int(match.group("requested_tokens"))
        effective_tokens = int(match.group("effective_tokens"))
        assert requested_tokens > 0
        assert 0 < effective_tokens <= requested_tokens
        limits.append(
            {
                "requested_tokens": requested_tokens,
                "effective_tokens": effective_tokens,
                "event": match.group("event"),
            }
        )
    return limits


def _latest_iteration_token_limit(
    content: str,
    *,
    requested_tokens: int,
) -> int:
    limits = _iteration_limits_from_text(content)
    assert limits, "service emitted no public pipeline iteration limit"
    assert any(limit["event"] == "startup" for limit in limits)
    latest = limits[-1]
    assert latest["requested_tokens"] == requested_tokens
    effective_tokens = latest["effective_tokens"]
    assert type(effective_tokens) is int
    return effective_tokens


def _wait_for_pipeline_requests(
    log_path: Path,
    *,
    offset: int,
    expected_requests: int,
    timeout: float = 180.0,
) -> list[dict[str, int]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waves = _records(log_path)[offset:]
        observed_requests = sum(wave["reqs"] for wave in waves)
        if observed_requests >= expected_requests:
            assert observed_requests == expected_requests
            return waves
        time.sleep(0.1)
    raise AssertionError(
        f"pipeline accounted for fewer than {expected_requests} requests:\n"
        f"{_log_text(log_path)[-12_000:]}"
    )


def _wait_for_public_server_pipeline_requests(
    server: Any,
    *,
    offset: int,
    expected_requests: int,
    timeout: float = 300.0,
) -> list[dict[str, int]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        waves = _records_from_text(server.log_tail())[offset:]
        observed_requests = sum(wave["reqs"] for wave in waves)
        if observed_requests >= expected_requests:
            assert observed_requests == expected_requests
            return waves
        time.sleep(0.1)
    raise AssertionError(
        f"pipeline accounted for fewer than {expected_requests} requests:\n"
        f"{server.log_tail()}"
    )


def _effective_group_size(
    group_size: int,
    cache_size: int,
    *,
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
) -> int:
    effective = min(
        group_size,
        num_layers,
        (cache_size // num_experts) - 1,
    )
    assert effective >= 1
    return effective


def _assert_pipeline_wave(
    wave: dict[str, int],
    *,
    group_size: int,
    cache_size: int,
    expected_reqs: int | None = None,
    expected_tiles: int | None = 1,
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
) -> int:
    assert set(wave) == PIPELINE_WAVE_FIELDS
    assert all(
        type(wave[field]) is int and wave[field] >= 0
        for field in PIPELINE_WAVE_FIELDS
    )
    assert wave["reqs"] >= 1
    if expected_reqs is not None:
        assert wave["reqs"] == expected_reqs
    effective_group_size = _effective_group_size(
        group_size,
        cache_size,
        num_layers=num_layers,
        num_experts=num_experts,
    )
    groups = math.ceil(num_layers / effective_group_size)
    assert wave["groups"] == groups
    assert wave["group_forwards"] > 0
    assert wave["group_forwards"] % groups == 0
    tiles = wave["group_forwards"] // groups
    if expected_tiles is not None:
        assert tiles == expected_tiles
    assert wave["iterations"] == wave["group_forwards"]
    assert 0 <= wave["decode_iterations"] <= wave["iterations"]
    assert wave["prefill_layer_prepares"] == num_layers
    return tiles


def _assert_fifo_packed_waves(
    waves: list[dict[str, int]],
    *,
    request_prefill_tokens: list[int],
    iteration_token_limit: int,
    group_size: int,
    cache_size: int,
    num_layers: int = NUM_LAYERS,
    num_experts: int = NUM_EXPERTS,
) -> None:
    request_cursor = 0
    for wave in waves:
        request_count = wave["reqs"]
        wave_prefill_tokens = request_prefill_tokens[
            request_cursor : request_cursor + request_count
        ]
        assert len(wave_prefill_tokens) == request_count
        expected_tiles = math.ceil(
            sum(wave_prefill_tokens) / iteration_token_limit
        )
        _assert_pipeline_wave(
            wave,
            group_size=group_size,
            cache_size=cache_size,
            expected_reqs=request_count,
            expected_tiles=expected_tiles,
            num_layers=num_layers,
            num_experts=num_experts,
        )
        request_cursor += request_count
    assert request_cursor == len(request_prefill_tokens)


def _assert_pipeline_structure(
    structure: dict[str, int],
    waves: list[dict[str, int]],
) -> None:
    assert set(structure) == PIPELINE_WAVE_FIELDS
    assert structure == {
        field: sum(wave[field] for wave in waves)
        for field in PIPELINE_WAVE_FIELDS
    }


def _run_nonstream_batch(
    base_url: str,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gate = threading.Barrier(len(payloads) + 1)
    results: list[dict[str, Any] | BaseException | None] = [None] * len(
        payloads
    )

    def worker(index: int) -> None:
        try:
            gate.wait()
            results[index] = _post_completion(base_url, payloads[index])
        except BaseException as exc:
            results[index] = exc

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(len(payloads))
    ]
    for thread in threads:
        thread.start()
    gate.wait()
    for thread in threads:
        thread.join(timeout=1_000.0)
        assert not thread.is_alive(), "concurrent completion did not finish"
    assert all(not isinstance(result, BaseException) for result in results), results
    assert all(isinstance(result, dict) for result in results)
    return [result for result in results if isinstance(result, dict)]


def _run_staggered_batch(
    base_url: str,
    payloads: list[dict[str, Any]],
    *,
    interval_seconds: float,
) -> list[dict[str, Any]]:
    start_events = [threading.Event() for _ in payloads]
    submitted_events = [threading.Event() for _ in payloads]
    results: list[dict[str, Any] | BaseException | None] = [None] * len(
        payloads
    )

    def worker(index: int) -> None:
        try:
            start_events[index].wait()
            submitted_events[index].set()
            results[index] = _post_completion(base_url, payloads[index])
        except BaseException as exc:
            results[index] = exc

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(len(payloads))
    ]
    for thread in threads:
        thread.start()
    for index in range(len(payloads)):
        start_events[index].set()
        assert submitted_events[index].wait(timeout=10.0)
        if index + 1 < len(payloads):
            time.sleep(interval_seconds)
    for thread in threads:
        thread.join(timeout=1_000.0)
        assert not thread.is_alive(), "staggered completion did not finish"
    assert all(not isinstance(result, BaseException) for result in results), results
    assert all(isinstance(result, dict) for result in results)
    return [result for result in results if isinstance(result, dict)]


def _run_stream_batch(
    base_url: str,
    payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    gate = threading.Barrier(len(payloads) + 1)
    results: list[dict[str, Any]] = [{} for _ in payloads]

    def worker(index: int) -> None:
        gate.wait()
        _stream_completion(
            base_url,
            payloads[index],
            threading.Event(),
            results[index],
        )

    threads = [
        threading.Thread(target=worker, args=(index,), daemon=True)
        for index in range(len(payloads))
    ]
    for thread in threads:
        thread.start()
    gate.wait()
    for thread in threads:
        thread.join(timeout=1_000.0)
        assert not thread.is_alive(), "streaming completion did not finish"
    assert all("error" not in result for result in results), [
        result.get("error") for result in results
    ]
    return results


def _run_decode_then_prefill_batch(
    base_url: str,
    *,
    decode_payload: dict[str, Any],
    prefill_payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_text = threading.Event()
    decode_result: dict[str, Any] = {}
    decode_thread = threading.Thread(
        target=_stream_completion,
        args=(base_url, decode_payload, first_text, decode_result),
        daemon=True,
    )
    decode_thread.start()
    assert first_text.wait(timeout=300.0), "decode emitted no nonempty SSE"
    assert "error" not in decode_result, decode_result.get("error")
    assert decode_thread.is_alive(), "decode ended before prefill admission"

    prefill_started = time.monotonic()
    prefill_responses = _run_nonstream_batch(base_url, prefill_payloads)
    prefill_completed = time.monotonic()

    decode_thread.join(timeout=1_000.0)
    assert not decode_thread.is_alive(), "decode did not finish"
    assert "error" not in decode_result, decode_result.get("error")
    events = decode_result.get("events")
    assert isinstance(events, list) and events
    assert any(
        prefill_started <= event["at_seconds"] <= prefill_completed
        for event in events
    ), "decode emitted no SSE text while prefill was active"
    return decode_result, prefill_responses


def _run_decode_then_stream_prefill_batch(
    base_url: str,
    *,
    decode_payload: dict[str, Any],
    prefill_payloads: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_text = threading.Event()
    decode_result: dict[str, Any] = {}
    decode_thread = threading.Thread(
        target=_stream_completion,
        args=(base_url, decode_payload, first_text, decode_result),
        daemon=True,
    )
    decode_thread.start()
    assert first_text.wait(timeout=900.0), "decode emitted no nonempty SSE"
    assert "error" not in decode_result, decode_result.get("error")
    assert decode_thread.is_alive(), "decode ended before prefill admission"

    prefill_started = time.monotonic()
    prefill_results = _run_stream_batch(base_url, prefill_payloads)
    prefill_completed = time.monotonic()

    decode_thread.join(timeout=1_800.0)
    assert not decode_thread.is_alive(), "decode did not finish"
    assert "error" not in decode_result, decode_result.get("error")
    events = decode_result.get("events")
    assert isinstance(events, list) and events
    assert any(
        prefill_started <= event["at_seconds"] <= prefill_completed
        for event in events
    ), "decode emitted no SSE text while prefill was active"
    return decode_result, prefill_results


def _has_pair(argv: list[str], flag: str, value: str) -> bool:
    return any(
        argv[index : index + 2] == [flag, value]
        for index in range(len(argv) - 1)
    )


def _assert_startup_rejected(
    command: list[str],
    *,
    timeout: float = 180.0,
) -> None:
    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            output, _ = process.communicate(timeout=30.0)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            output, _ = process.communicate(timeout=30.0)
        raise AssertionError(
            "invalid public configuration started instead of failing:\n"
            f"{output[-12_000:]}"
        )
    assert process.returncode != 0, output[-12_000:]


def _scaled_dry_run(mode: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_scaled_expert_contention.py",
            "--modes",
            mode,
            "--dry-run",
            "--ft-executable",
            str(_ft_executable()),
            "--moe-cache-size",
            "24",
            "--max-prefill-length",
            "128",
        ],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=60.0,
        check=False,
    )


def _run_scaled_benchmark(
    scaled_model: Path,
    output_path: Path,
) -> dict[str, Any]:
    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/bench_scaled_expert_contention.py",
            "--ft-executable",
            str(_ft_executable()),
            "--model",
            str(scaled_model),
            "--modes",
            "layered-pipeline-g1-wave64",
            "--repetitions",
            "1",
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
            "4",
            "--prefill-tokens",
            "2048",
            "--prefill-decode-tokens",
            "1",
            "--burst-trigger",
            "first-sse",
            "--prefill-submit-stagger-ms",
            "0",
            "--max-prefill-length",
            "128",
            "--moe-cache-size",
            "24",
            "--cuda-graph-max-bs",
            "8",
            "--output",
            str(output_path),
        ],
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


def _assert_benchmark_request(request: dict[str, Any]) -> None:
    assert request["measurement_failed"] is False
    assert request.get("error") in (None, "")
    output_text = request["output_text"]
    events = request["nonempty_text_events"]
    assert isinstance(output_text, str)
    assert isinstance(events, list)
    assert output_text == "".join(event["text"] for event in events)
    for event in events:
        assert set(event) == {"at_seconds", "text"}
        assert type(event["at_seconds"]) is float
        assert event["at_seconds"] >= 0
        assert isinstance(event["text"], str) and event["text"]
    _usage(request)
    if "output_mismatch" in request:
        assert request["output_mismatch"] is None


def test_cli_exposes_only_the_final_layered_pipeline_surface() -> None:
    completed = subprocess.run(
        [str(_ft_executable()), "serve", "--help"],
        cwd=PROJECT_ROOT,
        env=_subprocess_env(),
        capture_output=True,
        text=True,
        timeout=30.0,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    help_text = completed.stdout + completed.stderr
    assert "layered-pipeline" in help_text
    assert "--prefill-layer-group-size" in help_text
    assert "--prefill-wave-max-chunks" in help_text
    assert "--max-prefill-length" in help_text
    assert "layered-prefill" not in help_text
    assert "--layered-pipeline-chunks-per-iteration" not in help_text


@pytest.mark.parametrize(
    "removed_surface",
    [
        "policy",
        "option",
    ],
)
def test_removed_cli_surfaces_are_rejected_before_service_start(
    scaled_model: Path,
    removed_surface: str,
) -> None:
    command = _service_command(
        scaled_model,
        port=_free_port(),
        group_size=1,
        wave_max_chunks=64,
        max_prefill_length=128,
        cache_size=24,
        cuda_graph_max_bs=0,
        policy="layered-prefill" if removed_surface == "policy" else (
            "layered-pipeline"
        ),
    )
    if removed_surface == "option":
        command.extend(
            ["--layered-pipeline-chunks-per-iteration", "2"]
        )
    _assert_startup_rejected(command, timeout=120.0)


@pytest.mark.parametrize(
    ("flag", "invalid_value"),
    [
        ("--prefill-layer-group-size", "0"),
        ("--prefill-wave-max-chunks", "0"),
        ("--max-prefill-length", "0"),
        ("--moe-cache-size", "8"),
    ],
)
def test_invalid_public_geometry_is_rejected(
    scaled_model: Path,
    flag: str,
    invalid_value: str,
) -> None:
    command = _service_command(
        scaled_model,
        port=_free_port(),
        group_size=1,
        wave_max_chunks=3,
        max_prefill_length=128,
        cache_size=24,
        cuda_graph_max_bs=0,
    )
    value_index = command.index(flag) + 1
    command[value_index] = invalid_value
    _assert_startup_rejected(command)


def test_scaled_dry_run_uses_final_mode_argv_and_rejects_removed_modes() -> None:
    accepted_modes = (
        ("layered-pipeline-g1-wave1", "1", "1"),
        ("layered-pipeline-g2-wave3", "2", "3"),
        ("layered-pipeline-g4-wave16", "4", "16"),
        ("layered-pipeline-g1-wave64", "1", "64"),
    )
    for mode_name, group_size, wave_max_chunks in accepted_modes:
        completed = _scaled_dry_run(mode_name)
        assert completed.returncode == 0, completed.stderr
        report = json.loads(completed.stdout)
        commands = report["commands"]
        assert isinstance(commands, list) and len(commands) == 1
        argv = commands[0]
        assert isinstance(argv, list)
        assert all(isinstance(argument, str) for argument in argv)
        for flag, value in (
            ("--batching-policy", "layered-pipeline"),
            ("--prefill-layer-group-size", group_size),
            ("--prefill-wave-max-chunks", wave_max_chunks),
            ("--max-prefill-length", "128"),
            ("--moe-cache-size", "24"),
        ):
            assert _has_pair(argv, flag, value), f"missing {flag}={value}"
        assert "--layered-pipeline-chunks-per-iteration" not in argv
        assert "layered-prefill" not in argv

    for removed_mode in REMOVED_MODE_NAMES:
        rejected = _scaled_dry_run(removed_mode)
        assert rejected.returncode != 0, removed_mode


@pytest.mark.parametrize(
    (
        "max_prefill_length",
        "requested_group_size",
        "cache_size",
        "wave_max_chunks",
        "concurrency",
        "equal_k",
    ),
    [
        pytest.param(32, 4, 16, 64, 8, 8, id="t32-g4-c16-w64-bs8"),
        pytest.param(128, 2, 24, 16, 4, 4, id="t128-g2-c24-w16-bs4"),
        pytest.param(256, 4, 40, 3, 2, 2, id="t256-g4-c40-w3-bs2"),
        pytest.param(512, 1, 48, 1, 1, 1, id="t512-g1-c48-w1-bs1"),
        pytest.param(8192, 2, 24, 16, 2, 1, id="t8192-g2-c24-w16-bs2"),
    ],
)
def test_public_parameter_matrix_and_admission_boundaries(
    scaled_model: Path,
    scaled_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
    max_prefill_length: int,
    requested_group_size: int,
    cache_size: int,
    wave_max_chunks: int,
    concurrency: int,
    equal_k: int,
) -> None:
    case_label = (
        f"t{max_prefill_length}-g{requested_group_size}"
        f"-c{cache_size}-w{wave_max_chunks}-bs{concurrency}"
    )
    readiness_prompt = scaled_prompt_materializer(
        8,
        99_000 + max_prefill_length,
        f"matrix-{case_label}-readiness",
        31,
    )
    with _running_service(
        scaled_model,
        tmp_path / f"matrix-{case_label}.log",
        group_size=requested_group_size,
        wave_max_chunks=wave_max_chunks,
        max_prefill_length=max_prefill_length,
        cache_size=cache_size,
        readiness_prompt_text=readiness_prompt,
    ) as (base_url, service_log, process):
        iteration_token_limit = _latest_iteration_token_limit(
            _log_text(service_log),
            requested_tokens=max_prefill_length,
        )
        sequential_offset = len(_records(service_log))
        if max_prefill_length <= 512:
            token_lengths = [
                max_prefill_length - 1,
                max_prefill_length,
                max_prefill_length + 1,
                (2 * max_prefill_length) - 1,
                2 * max_prefill_length,
                (2 * max_prefill_length) + 1,
            ]
            if wave_max_chunks == 1:
                soft_cap_ks = [1, 2]
            else:
                soft_cap_ks = [
                    wave_max_chunks - 1,
                    wave_max_chunks,
                    wave_max_chunks + 1,
                ]
            token_lengths.extend(
                k_value * max_prefill_length for k_value in soft_cap_ks
            )
        else:
            token_lengths = [2_048]

        observed_ks = []
        actual_new_prefill_tokens = []
        for request_index, prompt_tokens in enumerate(token_lengths):
            prompt = scaled_prompt_materializer(
                prompt_tokens,
                100_000 + max_prefill_length + request_index,
                f"matrix-{case_label}-request-{request_index}",
                request_index,
            )
            response = _post_completion(
                base_url,
                _completion_payload(prompt, max_tokens=1, stream=False),
            )
            _assert_nonstream_completion(
                response,
                prompt_tokens=prompt_tokens,
                completion_tokens=1,
            )
            actual_new_prefill = (
                _usage(response)["prompt_tokens"]
                - _usage(response)["cached_tokens"]
            )
            actual_new_prefill_tokens.append(actual_new_prefill)
            observed_ks.append(
                math.ceil(actual_new_prefill / max_prefill_length)
            )

        sequential_waves = _wait_for_pipeline_requests(
            service_log,
            offset=sequential_offset,
            expected_requests=len(token_lengths),
        )
        assert len(sequential_waves) == len(token_lengths)
        for wave, actual_new_prefill in zip(
            sequential_waves,
            actual_new_prefill_tokens,
        ):
            _assert_pipeline_wave(
                wave,
                group_size=requested_group_size,
                cache_size=cache_size,
                expected_reqs=1,
                expected_tiles=math.ceil(
                    actual_new_prefill / iteration_token_limit
                ),
            )
        assert observed_ks == [
            math.ceil(prompt_tokens / max_prefill_length)
            for prompt_tokens in token_lengths
        ]
        if max_prefill_length <= 512:
            assert observed_ks[:6] == [1, 1, 2, 2, 2, 3]
            assert observed_ks[-len(soft_cap_ks) :] == soft_cap_ks
            assert observed_ks[-1] == wave_max_chunks + 1

        concurrent_offset = len(_records(service_log))
        if max_prefill_length == 8_192:
            concurrent_prompt_tokens = 2_048
            concurrent_k = 1
        else:
            concurrent_prompt_tokens = equal_k * max_prefill_length
            concurrent_k = equal_k
        payloads = []
        for request_index in range(concurrency):
            prompt = scaled_prompt_materializer(
                concurrent_prompt_tokens,
                200_000 + max_prefill_length + request_index,
                f"matrix-{case_label}-batch-{request_index}",
                20 + request_index,
            )
            payloads.append(
                _completion_payload(prompt, max_tokens=1, stream=False)
            )
        responses = _run_nonstream_batch(base_url, payloads)
        for response in responses:
            _assert_nonstream_completion(
                response,
                prompt_tokens=concurrent_prompt_tokens,
                completion_tokens=1,
            )
        concurrent_waves = _wait_for_pipeline_requests(
            service_log,
            offset=concurrent_offset,
            expected_requests=concurrency,
        )
        max_wave_reqs = (
            1
            if concurrent_k > wave_max_chunks
            else wave_max_chunks // concurrent_k
        )
        for wave in concurrent_waves:
            expected_tiles = math.ceil(
                (
                    wave["reqs"]
                    * concurrent_prompt_tokens
                )
                / iteration_token_limit
            )
            _assert_pipeline_wave(
                wave,
                group_size=requested_group_size,
                cache_size=cache_size,
                expected_tiles=expected_tiles,
            )
            assert wave["reqs"] <= max_wave_reqs
        assert sum(wave["reqs"] for wave in concurrent_waves) == concurrency

        if cache_size == 2 * NUM_EXPERTS:
            fallback_offset = len(_records(service_log))
            fallback_prompt = scaled_prompt_materializer(
                16,
                210_000,
                "minimum-cache-small-working-set",
                29,
            )
            fallback_result: dict[str, Any] = {}
            _stream_completion(
                base_url,
                _completion_payload(
                    fallback_prompt,
                    max_tokens=8,
                    stream=True,
                    ignore_eos=True,
                ),
                threading.Event(),
                fallback_result,
            )
            _assert_stream_completion(
                fallback_result,
                prompt_tokens=16,
                completion_tokens=8,
            )
            fallback_waves = _wait_for_pipeline_requests(
                service_log,
                offset=fallback_offset,
                expected_requests=1,
            )
            assert len(fallback_waves) == 1
            _assert_pipeline_wave(
                fallback_waves[0],
                group_size=requested_group_size,
                cache_size=cache_size,
                expected_reqs=1,
                expected_tiles=math.ceil(
                    16 / iteration_token_limit
                ),
            )
        assert process.poll() is None


def test_first_sse_ragged_fifo_cached_and_repeated_lifecycle(
    scaled_model: Path,
    scaled_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
) -> None:
    from benchmarks.bench_lab_agent_policies import load_tokenizer

    max_prefill_length = 128
    group_size = 2
    cache_size = 24
    wave_max_chunks = 16
    readiness_prompt = scaled_prompt_materializer(
        8, 299_999, "ragged-readiness", 31
    )
    with _running_service(
        scaled_model,
        tmp_path / "ragged-first-sse-cache-repeat.log",
        group_size=group_size,
        wave_max_chunks=wave_max_chunks,
        max_prefill_length=max_prefill_length,
        cache_size=cache_size,
        readiness_prompt_text=readiness_prompt,
    ) as (base_url, service_log, process):
        iteration_token_limit = _latest_iteration_token_limit(
            _log_text(service_log),
            requested_tokens=max_prefill_length,
        )
        wave_offset = len(_records(service_log))
        driver_prompt = scaled_prompt_materializer(
            128, 300_000, "ragged-driver", 0
        )
        ragged_ks = [1, 2, 5, 8]
        assert sum(ragged_ks) == wave_max_chunks
        ragged_prompts = [
            scaled_prompt_materializer(
                k_value * max_prefill_length,
                300_100 + request_index,
                f"ragged-prefill-k{k_value}",
                1 + request_index,
            )
            for request_index, k_value in enumerate(ragged_ks)
        ]
        decode_result, prefill_responses = _run_decode_then_prefill_batch(
            base_url,
            decode_payload=_completion_payload(
                driver_prompt,
                max_tokens=512,
                stream=True,
                ignore_eos=True,
            ),
            prefill_payloads=[
                _completion_payload(prompt, max_tokens=1, stream=False)
                for prompt in ragged_prompts
            ],
        )
        _assert_stream_completion(
            decode_result,
            prompt_tokens=128,
            completion_tokens=512,
        )
        for response, k_value in zip(prefill_responses, ragged_ks):
            _assert_nonstream_completion(
                response,
                prompt_tokens=k_value * max_prefill_length,
                completion_tokens=1,
            )

        waves = _wait_for_pipeline_requests(
            service_log,
            offset=wave_offset,
            expected_requests=5,
        )
        assert waves[0]["reqs"] == 1
        assert waves[0]["decode_iterations"] == 0
        _assert_pipeline_wave(
            waves[0],
            group_size=group_size,
            cache_size=cache_size,
            expected_reqs=1,
            expected_tiles=math.ceil(
                128 / iteration_token_limit
            ),
        )
        burst_waves = waves[1:]
        assert burst_waves
        assert sum(wave["reqs"] for wave in burst_waves) == len(ragged_ks)
        assert all(1 <= wave["reqs"] <= len(ragged_ks) for wave in burst_waves)
        assert any(
            wave["decode_iterations"] == wave["iterations"]
            for wave in burst_waves
        )
        for wave in burst_waves:
            _assert_pipeline_wave(
                wave,
                group_size=group_size,
                cache_size=cache_size,
                expected_tiles=None,
            )

        tokenizer = load_tokenizer(scaled_model)
        cached_prefix = ragged_prompts[2]
        prefix_ids = tokenizer.encode(cached_prefix)
        continuation = scaled_prompt_materializer(
            32, 300_200, "ragged-cached-continuation", 6
        )
        cached_followup = cached_prefix + continuation
        followup_ids = tokenizer.encode(cached_followup)
        assert followup_ids[: len(prefix_ids)] == prefix_ids

        first_followup = _post_completion(
            base_url,
            _completion_payload(
                cached_followup, max_tokens=1, stream=False
            ),
        )
        first_usage = _usage(first_followup)
        assert first_usage["prompt_tokens"] == len(followup_ids)
        assert first_usage["completion_tokens"] == 1
        assert first_usage["total_tokens"] == len(followup_ids) + 1
        assert first_usage["cached_tokens"] > 0

        repeated_followup = _post_completion(
            base_url,
            _completion_payload(
                cached_followup, max_tokens=1, stream=False
            ),
        )
        repeated_usage = _usage(repeated_followup)
        assert repeated_usage["prompt_tokens"] == len(followup_ids)
        assert repeated_usage["completion_tokens"] == 1
        assert repeated_usage["total_tokens"] == len(followup_ids) + 1
        assert repeated_usage["cached_tokens"] > 0
        _completion_text(first_followup)
        _completion_text(repeated_followup)

        pure_decode_result: dict[str, Any] = {}
        _stream_completion(
            base_url,
            _completion_payload(
                cached_followup,
                max_tokens=8,
                stream=True,
                ignore_eos=True,
            ),
            threading.Event(),
            pure_decode_result,
        )
        pure_decode_usage = _usage(pure_decode_result)
        _assert_stream_completion(
            pure_decode_result,
            prompt_tokens=len(followup_ids),
            completion_tokens=8,
            cached_tokens=pure_decode_usage["cached_tokens"],
        )
        assert pure_decode_usage["cached_tokens"] > 0

        packed_ragged_offset = len(_records(service_log))
        packed_ragged_lengths = [
            max_prefill_length - 1,
            max_prefill_length - 1,
            max_prefill_length + 1,
            max_prefill_length + 1,
        ]
        packed_ragged_payloads = [
            _completion_payload(
                scaled_prompt_materializer(
                    prompt_tokens,
                    300_250 + request_index,
                    f"static-ragged-{prompt_tokens}-{request_index}",
                    13 + request_index,
                ),
                max_tokens=1,
                stream=False,
            )
            for request_index, prompt_tokens in enumerate(
                packed_ragged_lengths
            )
        ]
        packed_ragged_responses = _run_staggered_batch(
            base_url,
            packed_ragged_payloads,
            interval_seconds=0.005,
        )
        for response, prompt_tokens in zip(
            packed_ragged_responses,
            packed_ragged_lengths,
        ):
            _assert_nonstream_completion(
                response,
                prompt_tokens=prompt_tokens,
                completion_tokens=1,
            )
        packed_ragged_waves = _wait_for_pipeline_requests(
            service_log,
            offset=packed_ragged_offset,
            expected_requests=len(packed_ragged_lengths),
        )
        _assert_fifo_packed_waves(
            packed_ragged_waves,
            request_prefill_tokens=packed_ragged_lengths,
            iteration_token_limit=iteration_token_limit,
            group_size=group_size,
            cache_size=cache_size,
        )

        not_fit_offset = len(_records(service_log))
        not_fit_ks = [9, 8]
        not_fit_payloads = [
            _completion_payload(
                scaled_prompt_materializer(
                    k_value * max_prefill_length,
                    300_300 + request_index,
                    f"ragged-not-fit-k{k_value}",
                    8 + request_index,
                ),
                max_tokens=1,
                stream=False,
            )
            for request_index, k_value in enumerate(not_fit_ks)
        ]
        not_fit_responses = _run_staggered_batch(
            base_url,
            not_fit_payloads,
            interval_seconds=0.005,
        )
        for response, k_value in zip(not_fit_responses, not_fit_ks):
            _assert_nonstream_completion(
                response,
                prompt_tokens=k_value * max_prefill_length,
                completion_tokens=1,
            )
        not_fit_waves = _wait_for_pipeline_requests(
            service_log,
            offset=not_fit_offset,
            expected_requests=2,
        )
        assert len(not_fit_waves) == 2
        _assert_fifo_packed_waves(
            not_fit_waves,
            request_prefill_tokens=[
                k_value * max_prefill_length
                for k_value in not_fit_ks
            ],
            iteration_token_limit=iteration_token_limit,
            group_size=group_size,
            cache_size=cache_size,
        )

        oversized_offset = len(_records(service_log))
        oversized_and_following_ks = [17, 8, 8]
        oversized_payloads = [
            _completion_payload(
                scaled_prompt_materializer(
                    k_value * max_prefill_length,
                    300_400 + request_index,
                    f"fifo-oversized-then-fit-k{k_value}-{request_index}",
                    10 + request_index,
                ),
                max_tokens=1,
                stream=False,
            )
            for request_index, k_value in enumerate(
                oversized_and_following_ks
            )
        ]
        oversized_responses = _run_staggered_batch(
            base_url,
            oversized_payloads,
            interval_seconds=0.005,
        )
        for response, k_value in zip(
            oversized_responses, oversized_and_following_ks
        ):
            _assert_nonstream_completion(
                response,
                prompt_tokens=k_value * max_prefill_length,
                completion_tokens=1,
            )
        oversized_waves = _wait_for_pipeline_requests(
            service_log,
            offset=oversized_offset,
            expected_requests=3,
        )
        assert oversized_waves[0]["reqs"] == 1
        assert sum(wave["reqs"] for wave in oversized_waves) == 3
        _assert_fifo_packed_waves(
            oversized_waves,
            request_prefill_tokens=[
                k_value * max_prefill_length
                for k_value in oversized_and_following_ks
            ],
            iteration_token_limit=iteration_token_limit,
            group_size=group_size,
            cache_size=cache_size,
        )
        for wave in _records(service_log)[wave_offset:]:
            _assert_pipeline_wave(
                wave,
                group_size=group_size,
                cache_size=cache_size,
                expected_tiles=None,
            )
        assert process.poll() is None


def _exercise_graph_configuration(
    scaled_model: Path,
    materialize: Callable[[int, int, str, int], str],
    log_path: Path,
    *,
    cuda_graph_max_bs: int,
) -> dict[str, Any]:
    observations: dict[str, Any] = {"batches": {}}
    readiness_prompt = materialize(
        8,
        399_999,
        f"graph{cuda_graph_max_bs}-readiness",
        31,
    )
    with _running_service(
        scaled_model,
        log_path,
        group_size=2,
        wave_max_chunks=64,
        max_prefill_length=128,
        cache_size=24,
        cuda_graph_max_bs=cuda_graph_max_bs,
        readiness_prompt_text=readiness_prompt,
    ) as (base_url, service_log, process):
        iteration_token_limit = _latest_iteration_token_limit(
            _log_text(service_log),
            requested_tokens=128,
        )
        all_waves = []
        first_token_bases = {1: 0, 4: 1, 8: 5}
        for batch_size, prompt_tokens in ((1, 128), (4, 256), (8, 512)):
            wave_offset = len(_records(service_log))
            payloads = []
            for request_index in range(batch_size):
                prompt = materialize(
                    prompt_tokens,
                    400_000 + (batch_size * 100) + request_index,
                    f"graph-bs{batch_size}-request-{request_index}",
                    first_token_bases[batch_size] + request_index,
                )
                payloads.append(
                    _completion_payload(
                        prompt,
                        max_tokens=4,
                        stream=True,
                        ignore_eos=True,
                    )
                )
            results = _run_stream_batch(base_url, payloads)
            batch_observations = []
            for result in results:
                output_text = _assert_stream_completion(
                    result,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=4,
                )
                batch_observations.append(
                    {"output_text": output_text, "usage": _usage(result)}
                )
            waves = _wait_for_pipeline_requests(
                service_log,
                offset=wave_offset,
                expected_requests=batch_size,
            )
            for wave in waves:
                _assert_pipeline_wave(
                    wave,
                    group_size=2,
                    cache_size=24,
                    expected_tiles=math.ceil(
                        wave["reqs"]
                        * prompt_tokens
                        / iteration_token_limit
                    ),
                )
            assert sum(wave["reqs"] for wave in waves) == batch_size
            observations["batches"][batch_size] = batch_observations
            all_waves.extend(waves)
        observations["structure"] = {
            field: sum(wave[field] for wave in all_waves)
            for field in PIPELINE_WAVE_FIELDS
        }
        observations["wave_count"] = len(all_waves)
        assert process.poll() is None
    return observations


def test_graph0_and_graph8_preserve_public_function_and_wave_structure(
    scaled_model: Path,
    scaled_prompt_materializer: Callable[[int, int, str, int], str],
    tmp_path: Path,
) -> None:
    observations = {
        graph_size: _exercise_graph_configuration(
            scaled_model,
            scaled_prompt_materializer,
            tmp_path / f"pipeline-graph{graph_size}.log",
            cuda_graph_max_bs=graph_size,
        )
        for graph_size in (0, 8)
    }
    for batch_size in (1, 4, 8):
        graph0_batch = observations[0]["batches"][batch_size]
        graph8_batch = observations[8]["batches"][batch_size]
        assert [item["usage"] for item in graph8_batch] == [
            item["usage"] for item in graph0_batch
        ]
    assert observations[8]["batches"][1][0]["output_text"] == (
        observations[0]["batches"][1][0]["output_text"]
    )
    for graph_size in (0, 8):
        assert observations[graph_size]["structure"]["reqs"] == 13
        assert observations[graph_size]["wave_count"] >= 3


def _exercise_real_model_backend(
    model_path: Path,
    prompts: dict[str, str],
    *,
    attention_backend: str,
    cuda_graph_max_bs: int,
    num_layers: int,
    num_experts: int,
    num_tokens: int,
) -> dict[str, Any]:
    from benchmarks.bench_lab_agent_policies import PublicServer
    from benchmarks.bench_scaled_expert_contention import (
        wait_for_snapshot_count,
    )

    port = _free_port()
    base_url = f"http://{HOST}:{port}"
    server = PublicServer(
        command=_real_model_service_command(
            model_path,
            port=port,
            attention_backend=attention_backend,
            cuda_graph_max_bs=cuda_graph_max_bs,
            num_tokens=num_tokens,
        ),
        gpu=str(torch.cuda.current_device()),
        base_url=base_url,
        timeout=3_600.0,
        readiness_prompt_text=prompts["readiness"],
    )
    try:
        server.start()
    except Exception as exc:
        raise AssertionError(
            "real-model service failed during public startup: "
            f"{type(exc).__name__}: {exc}\n{server.log_tail()}"
        ) from exc
    try:
        wait_for_snapshot_count(server, minimum=1, timeout=600.0)
        readiness_cursor = len(_records_from_text(server.log_tail()))
        assert readiness_cursor >= 1
        iteration_token_limit = _latest_iteration_token_limit(
            server.log_tail(),
            requested_tokens=32,
        )
        server.mark_measurement_start()
        wave_offset = len(_records_from_text(server.log_tail()))
        assert wave_offset >= readiness_cursor

        driver_result, prefill_results = (
            _run_decode_then_stream_prefill_batch(
                base_url,
                decode_payload=_completion_payload(
                    prompts["driver"],
                    max_tokens=16,
                    stream=True,
                    ignore_eos=True,
                ),
                prefill_payloads=[
                    _completion_payload(
                        prompts[f"prefill_{request_index}"],
                        max_tokens=1,
                        stream=True,
                        ignore_eos=True,
                    )
                    for request_index in range(2)
                ],
            )
        )
        driver_output = _assert_stream_completion(
            driver_result,
            prompt_tokens=32,
            completion_tokens=16,
        )
        for result in prefill_results:
            _assert_stream_completion(
                result,
                prompt_tokens=64,
                completion_tokens=1,
            )
        results = [driver_result, *prefill_results]
        assert sum(_usage(result)["prompt_tokens"] for result in results) == 160
        assert sum(
            _usage(result)["completion_tokens"] for result in results
        ) == 18

        waves = _wait_for_public_server_pipeline_requests(
            server,
            offset=wave_offset,
            expected_requests=3,
            timeout=600.0,
        )
        assert len(waves) == 2
        first_wave, second_wave = waves
        _assert_pipeline_wave(
            first_wave,
            group_size=1,
            cache_size=512,
            expected_reqs=1,
            expected_tiles=math.ceil(32 / iteration_token_limit),
            num_layers=num_layers,
            num_experts=num_experts,
        )
        assert first_wave["decode_iterations"] == 0
        _assert_pipeline_wave(
            second_wave,
            group_size=1,
            cache_size=512,
            expected_reqs=2,
            expected_tiles=math.ceil(128 / iteration_token_limit),
            num_layers=num_layers,
            num_experts=num_experts,
        )
        assert 0 < second_wave["decode_iterations"] <= second_wave[
            "iterations"
        ]
        return {
            "driver_output": driver_output,
            "usages": [_usage(result) for result in results],
            "waves": waves,
        }
    except AssertionError as exc:
        exc.add_note(server.log_tail())
        raise
    finally:
        server.stop()


def _exercise_activation_budgeted_dsv4(
    model_path: Path,
    prompts: dict[str, Any],
    *,
    cuda_graph_max_bs: int,
) -> dict[str, Any]:
    from benchmarks.bench_real_conversation_concurrency import (
        BenchmarkServer,
    )
    from benchmarks.bench_scaled_expert_contention import (
        wait_for_snapshot_count,
    )

    port = _free_port()
    base_url = f"http://{HOST}:{port}"
    command = _real_model_service_command(
        model_path,
        port=port,
        attention_backend="dsv4_sparse",
        cuda_graph_max_bs=cuda_graph_max_bs,
        num_tokens=None,
        max_prefill_length=8_192,
        max_seq_len_override=131_072,
        wave_max_chunks=1,
        max_running_requests=16,
        memory_ratio=0.7,
    )
    command.extend(
        ["--nowag-expert-path", str(DSV4_NOWAG_EXPERT_PATH)]
    )
    assert "--num-tokens" not in command
    assert _has_pair(command, "--memory-ratio", "0.7")
    assert _has_pair(
        command,
        "--nowag-expert-path",
        str(DSV4_NOWAG_EXPERT_PATH),
    )
    server = BenchmarkServer(
        command=command,
        gpu=str(torch.cuda.current_device()),
        base_url=base_url,
        timeout=3_600.0,
        readiness_prompt_text=prompts["readiness"],
        nowag_plugin_src=DSV4_NOWAG_PLUGIN_SRC,
    )
    try:
        try:
            server.start()
        except Exception as exc:
            raise AssertionError(
                "activation-budgeted DSV4 failed during public startup: "
                f"{type(exc).__name__}: {exc}\n{server.log_tail()}"
            ) from exc
        wait_for_snapshot_count(server, minimum=1, timeout=600.0)
        readiness_cursor = len(_records_from_text(server.log_tail()))
        assert readiness_cursor >= 1
        iteration_token_limit = _latest_iteration_token_limit(
            server.log_tail(),
            requested_tokens=8_192,
        )
        server.mark_measurement_start()

        short_offset = len(_records_from_text(server.log_tail()))
        short_result: dict[str, Any] = {}
        _stream_completion(
            base_url,
            _completion_payload(
                prompts["short"],
                max_tokens=4,
                stream=True,
                ignore_eos=True,
            ),
            threading.Event(),
            short_result,
        )
        _assert_stream_completion(
            short_result,
            prompt_tokens=32,
            completion_tokens=4,
        )
        short_waves = _wait_for_public_server_pipeline_requests(
            server,
            offset=short_offset,
            expected_requests=1,
            timeout=600.0,
        )
        assert len(short_waves) == 1
        _assert_pipeline_wave(
            short_waves[0],
            group_size=1,
            cache_size=512,
            expected_reqs=1,
            expected_tiles=math.ceil(32 / iteration_token_limit),
            num_layers=43,
            num_experts=256,
        )
        assert short_waves[0]["decode_iterations"] == 0

        concurrent_offset = len(_records_from_text(server.log_tail()))
        driver_result, ragged_results = (
            _run_decode_then_stream_prefill_batch(
                base_url,
                decode_payload=_completion_payload(
                    prompts["driver"],
                    max_tokens=16,
                    stream=True,
                    ignore_eos=True,
                ),
                prefill_payloads=[
                    _completion_payload(
                        prompts["long"],
                        max_tokens=1,
                        stream=True,
                        ignore_eos=True,
                    ),
                    _completion_payload(
                        prompts["ragged_short"],
                        max_tokens=1,
                        stream=True,
                        ignore_eos=True,
                    ),
                ],
            )
        )
        _assert_stream_completion(
            driver_result,
            prompt_tokens=32,
            completion_tokens=16,
        )
        _assert_stream_completion(
            ragged_results[0],
            prompt_tokens=40_000,
            completion_tokens=1,
        )
        long_actual_prefill_tokens = (
            _usage(ragged_results[0])["prompt_tokens"]
            - _usage(ragged_results[0])["cached_tokens"]
        )
        long_tiles = math.ceil(
            long_actual_prefill_tokens / iteration_token_limit
        )
        assert long_tiles > 1
        _assert_stream_completion(
            ragged_results[1],
            prompt_tokens=128,
            completion_tokens=1,
        )
        concurrent_results = [driver_result, *ragged_results]
        assert sum(
            _usage(result)["prompt_tokens"]
            for result in concurrent_results
        ) == 40_160
        assert sum(
            _usage(result)["completion_tokens"]
            for result in concurrent_results
        ) == 18

        concurrent_waves = _wait_for_public_server_pipeline_requests(
            server,
            offset=concurrent_offset,
            expected_requests=3,
            timeout=1_800.0,
        )
        assert len(concurrent_waves) == 3
        assert concurrent_waves[0]["decode_iterations"] == 0
        _assert_pipeline_wave(
            concurrent_waves[0],
            group_size=1,
            cache_size=512,
            expected_reqs=1,
            expected_tiles=math.ceil(32 / iteration_token_limit),
            num_layers=43,
            num_experts=256,
        )
        long_tiled_waves = [
            wave
            for wave in concurrent_waves[1:]
            if wave["group_forwards"] > 43
        ]
        short_single_tile_waves = [
            wave
            for wave in concurrent_waves[1:]
            if wave["group_forwards"] == 43
        ]
        assert len(long_tiled_waves) == 1
        assert len(short_single_tile_waves) == 1
        _assert_pipeline_wave(
            long_tiled_waves[0],
            group_size=1,
            cache_size=512,
            expected_reqs=1,
            expected_tiles=long_tiles,
            num_layers=43,
            num_experts=256,
        )
        _assert_pipeline_wave(
            short_single_tile_waves[0],
            group_size=1,
            cache_size=512,
            expected_reqs=1,
            expected_tiles=math.ceil(128 / iteration_token_limit),
            num_layers=43,
            num_experts=256,
        )
        assert any(
            0 < wave["decode_iterations"] <= wave["iterations"]
            for wave in concurrent_waves[1:]
        )

        cached_offset = len(_records_from_text(server.log_tail()))
        cached_result: dict[str, Any] = {}
        _stream_completion(
            base_url,
            _completion_payload(
                prompts["cached_followup"],
                max_tokens=1,
                stream=True,
                ignore_eos=True,
            ),
            threading.Event(),
            cached_result,
        )
        cached_usage = _usage(cached_result)
        assert cached_usage["cached_tokens"] >= 128
        cached_actual_prefill_tokens = (
            cached_usage["prompt_tokens"]
            - cached_usage["cached_tokens"]
        )
        _assert_stream_completion(
            cached_result,
            prompt_tokens=prompts["cached_followup_tokens"],
            completion_tokens=1,
            cached_tokens=cached_usage["cached_tokens"],
        )
        cached_waves = _wait_for_public_server_pipeline_requests(
            server,
            offset=cached_offset,
            expected_requests=1,
            timeout=600.0,
        )
        assert len(cached_waves) == 1
        _assert_pipeline_wave(
            cached_waves[0],
            group_size=1,
            cache_size=512,
            expected_reqs=1,
            expected_tiles=math.ceil(
                cached_actual_prefill_tokens / iteration_token_limit
            ),
            num_layers=43,
            num_experts=256,
        )
        return {
            "short_usage": _usage(short_result),
            "concurrent_usages": [
                _usage(result) for result in concurrent_results
            ],
            "cached_usage": cached_usage,
            "wave_structure": {
                "single_tile_short": short_waves[0],
                "decode_driver": concurrent_waves[0],
                "long_tiled": long_tiled_waves[0],
                "ragged_short": short_single_tile_waves[0],
                "cached_followup": cached_waves[0],
            },
        }
    except AssertionError as exc:
        exc.add_note(server.log_tail())
        raise
    finally:
        server.close()


@pytest.mark.parametrize(
    (
        "model_path",
        "attention_backend",
        "num_layers",
        "num_experts",
        "num_tokens",
        "requires_flashinfer",
    ),
    [
        pytest.param(
            QWEN36_MODEL_PATH,
            "triton",
            40,
            256,
            4_096,
            False,
            id="qwen36-triton",
        ),
        pytest.param(
            QWEN36_MODEL_PATH,
            "fi",
            40,
            256,
            4_096,
            True,
            id="qwen36-flashinfer",
        ),
        pytest.param(
            DSV4_MODEL_PATH,
            "dsv4_sparse",
            43,
            256,
            8_192,
            False,
            id="dsv4-sparse",
        ),
    ],
)
def test_real_moe_backends_support_layered_pipeline_graph0_and_graph8(
    model_path: Path,
    attention_backend: str,
    num_layers: int,
    num_experts: int,
    num_tokens: int,
    requires_flashinfer: bool,
) -> None:
    _require_cuda_e2e()
    if not model_path.exists():
        pytest.skip(f"public checkpoint is unavailable: {model_path}")
    if (
        attention_backend == "dsv4_sparse"
        and not DSV4_NOWAG_EXPERT_PATH.exists()
    ):
        pytest.skip(
            "public DSV4 NoWAG expert checkpoint is unavailable: "
            f"{DSV4_NOWAG_EXPERT_PATH}"
        )
    if (
        attention_backend == "dsv4_sparse"
        and not DSV4_NOWAG_PLUGIN_SRC.exists()
    ):
        pytest.skip(
            "public DSV4 NoWAG plugin source is unavailable: "
            f"{DSV4_NOWAG_PLUGIN_SRC}"
        )
    if requires_flashinfer and importlib.util.find_spec("flashinfer") is None:
        pytest.skip("flashinfer is unavailable")

    materialize = _public_prompt_materializer(model_path)
    if attention_backend == "dsv4_sparse":
        from benchmarks.bench_lab_agent_policies import load_tokenizer

        ragged_short = materialize(
            128, 510_003, "activation-ragged-page-prefix", 3
        )
        continuation = materialize(
            32, 510_004, "activation-cached-continuation", 4
        )
        cached_followup = ragged_short + continuation
        tokenizer = load_tokenizer(model_path)
        ragged_short_ids = tokenizer.encode(ragged_short)
        cached_followup_ids = tokenizer.encode(cached_followup)
        assert len(ragged_short_ids) == 128
        assert len(cached_followup_ids) == 160
        assert cached_followup_ids[: len(ragged_short_ids)] == (
            ragged_short_ids
        )
        activation_prompts: dict[str, Any] = {
            "readiness": materialize(
                8, 510_000, "activation-readiness", 5
            ),
            "short": materialize(
                32, 510_001, "activation-single-tile-short", 0
            ),
            "driver": materialize(
                32, 510_002, "activation-decode-driver", 1
            ),
            "long": materialize(
                40_000, 510_005, "activation-long-40k", 2
            ),
            "ragged_short": ragged_short,
            "cached_followup": cached_followup,
            "cached_followup_tokens": len(cached_followup_ids),
        }
        observations = {
            graph_size: _exercise_activation_budgeted_dsv4(
                model_path,
                activation_prompts,
                cuda_graph_max_bs=graph_size,
            )
            for graph_size in (0, 8)
        }
        assert observations[8]["short_usage"] == observations[0][
            "short_usage"
        ]
        assert observations[8]["concurrent_usages"] == observations[0][
            "concurrent_usages"
        ]
        assert observations[8]["cached_usage"] == observations[0][
            "cached_usage"
        ]
        assert set(observations[8]["wave_structure"]) == set(
            observations[0]["wave_structure"]
        )
        for wave_name, graph0_wave in observations[0][
            "wave_structure"
        ].items():
            graph8_wave = observations[8]["wave_structure"][wave_name]
            assert {
                field: graph8_wave[field]
                for field in PIPELINE_CROSS_GRAPH_STRUCTURE_FIELDS
            } == {
                field: graph0_wave[field]
                for field in PIPELINE_CROSS_GRAPH_STRUCTURE_FIELDS
            }
        return

    prompts = {
        "readiness": materialize(
            8, 500_000, "real-model-readiness", 3
        ),
        "driver": materialize(32, 500_001, "real-model-driver", 0),
        "prefill_0": materialize(
            64, 500_002, "real-model-prefill-0", 1
        ),
        "prefill_1": materialize(
            64, 500_003, "real-model-prefill-1", 2
        ),
    }
    observations = {
        graph_size: _exercise_real_model_backend(
            model_path,
            prompts,
            attention_backend=attention_backend,
            cuda_graph_max_bs=graph_size,
            num_layers=num_layers,
            num_experts=num_experts,
            num_tokens=num_tokens,
        )
        for graph_size in (0, 8)
    }
    assert observations[8]["usages"] == observations[0]["usages"]
    assert observations[8]["driver_output"] == observations[0][
        "driver_output"
    ]


def test_layered_pipeline_shared_pool_bounds_three_driver_decode_misses(
    scaled_model: Path,
    scaled_prompt_materializer: Callable[[int, int, str, int], str],
) -> None:
    from benchmarks.bench_lab_agent_policies import PublicServer
    from benchmarks.bench_scaled_expert_contention import (
        snapshot_delta,
        wait_for_snapshot_count,
    )

    assert len(
        {spec[3] for spec in SHARED_POOL_MISS_PROMPT_SPECS.values()}
    ) == len(SHARED_POOL_MISS_PROMPT_SPECS)
    prompts = {
        name: scaled_prompt_materializer(*spec)
        for name, spec in SHARED_POOL_MISS_PROMPT_SPECS.items()
    }
    port = _free_port()
    base_url = f"http://{HOST}:{port}"
    command = _service_command(
        scaled_model,
        port=port,
        group_size=1,
        wave_max_chunks=64,
        max_prefill_length=128,
        cache_size=24,
        cuda_graph_max_bs=8,
        collect_moe_stats=True,
    )
    server = PublicServer(
        command=command,
        gpu=str(torch.cuda.current_device()),
        base_url=base_url,
        timeout=900.0,
        readiness_prompt_text=prompts["readiness"],
    )

    server.start()
    try:
        baseline = wait_for_snapshot_count(server, minimum=1, timeout=180.0)
        iteration_token_limit = _latest_iteration_token_limit(
            server.log_tail(),
            requested_tokens=128,
        )
        server.mark_measurement_start()
        wave_offset = len(_records_from_text(server.log_tail()))

        payloads = []
        for driver_index in range(3):
            payloads.append(
                _completion_payload(
                    prompts[f"driver_{driver_index}"],
                    max_tokens=512,
                    stream=True,
                    ignore_eos=True,
                )
            )
        driver_results: list[dict[str, Any]] = [{}, {}, {}]
        first_text_events = [threading.Event() for _ in payloads]
        gate = threading.Barrier(4)

        def run_driver(driver_index: int) -> None:
            gate.wait()
            _stream_completion(
                base_url,
                payloads[driver_index],
                first_text_events[driver_index],
                driver_results[driver_index],
            )

        driver_threads = [
            threading.Thread(
                target=run_driver,
                args=(driver_index,),
                daemon=True,
            )
            for driver_index in range(3)
        ]
        for thread in driver_threads:
            thread.start()
        gate.wait()
        for driver_index, first_text in enumerate(first_text_events):
            assert first_text.wait(timeout=300.0), (
                f"driver {driver_index} emitted no nonempty SSE"
            )
            assert "error" not in driver_results[driver_index], (
                driver_results[driver_index].get("error")
            )
        assert all(thread.is_alive() for thread in driver_threads)

        prefill_result: dict[str, Any] = {}
        _stream_completion(
            base_url,
            _completion_payload(
                prompts["prefill"],
                max_tokens=1,
                stream=True,
                ignore_eos=True,
            ),
            threading.Event(),
            prefill_result,
        )
        for thread in driver_threads:
            thread.join(timeout=1_000.0)
            assert not thread.is_alive(), "decode driver did not finish"

        results = [*driver_results, prefill_result]
        for result in driver_results:
            _assert_stream_completion(
                result,
                prompt_tokens=128,
                completion_tokens=512,
            )
        _assert_stream_completion(
            prefill_result,
            prompt_tokens=2_048,
            completion_tokens=1,
        )
        assert sum(_usage(result)["prompt_tokens"] for result in results) == 2_432
        assert sum(
            _usage(result)["completion_tokens"] for result in results
        ) == 1_537

        final = wait_for_snapshot_count(
            server,
            minimum=len(baseline) + 1,
            timeout=180.0,
        )
        delta = snapshot_delta(baseline[-1], final[-1])
        assert set(delta) == MOE_STATS_FIELDS
        assert all(
            type(value) is int and value >= 0 for value in delta.values()
        )
        active_rows = delta["decode_active_rows"]
        missing_rows = delta["decode_missing_rows"]
        assert active_rows > 20_000
        assert 0 < missing_rows
        assert missing_rows * 10 <= active_rows * 7
        assert delta["decode_layer_calls"] >= 4_000
        assert delta["prefill_rows"] == 192
        assert delta["prefill_layer_prepares"] == 24

        waves = _records_from_text(server.log_tail())[wave_offset:]
        assert waves
        assert sum(wave["reqs"] for wave in waves) == 4
        assert any(
            wave["decode_iterations"] == wave["iterations"]
            for wave in waves
        )
        _assert_fifo_packed_waves(
            waves,
            request_prefill_tokens=[128, 128, 128, 2_048],
            iteration_token_limit=iteration_token_limit,
            group_size=1,
            cache_size=24,
        )
    except AssertionError as exc:
        exc.add_note(server.log_tail())
        raise
    finally:
        server.stop()


def test_scaled_runtime_json_uses_only_final_pipeline_wave_schema(
    scaled_model: Path,
    tmp_path: Path,
) -> None:
    report = _run_scaled_benchmark(
        scaled_model,
        tmp_path / "scaled-final-layered-pipeline.json",
    )
    modes = report["modes"]
    assert isinstance(modes, list) and len(modes) == 1
    mode = modes[0]
    assert mode["name"] == "layered_pipeline_g1_wave64"
    assert mode.get("error") in (None, "")
    assert all("layered_prefill" not in key for key in mode)
    assert "layered_pipeline_waves" in mode
    assert "layered_pipeline_structure" in mode
    assert "--layered-pipeline-chunks-per-iteration" not in mode[
        "server_command"
    ]
    assert "layered-prefill" not in mode["server_command"]

    requests = mode["requests"]
    assert isinstance(requests, list) and len(requests) == 5
    for request in requests:
        _assert_benchmark_request(request)
    assert sum(_usage(request)["prompt_tokens"] for request in requests) == 8_320
    assert sum(
        _usage(request)["completion_tokens"] for request in requests
    ) == 516

    waves = mode["layered_pipeline_waves"]
    assert isinstance(waves, list) and waves
    assert sum(wave["reqs"] for wave in waves) == 5
    iteration_token_limit = _latest_iteration_token_limit(
        mode["server_log_tail"],
        requested_tokens=128,
    )
    _assert_fifo_packed_waves(
        waves,
        request_prefill_tokens=[128, 2_048, 2_048, 2_048, 2_048],
        iteration_token_limit=iteration_token_limit,
        group_size=1,
        cache_size=24,
    )
    _assert_pipeline_structure(mode["layered_pipeline_structure"], waves)
