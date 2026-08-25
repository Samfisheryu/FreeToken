import concurrent.futures
from dataclasses import dataclass
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import pytest


pytestmark = pytest.mark.needs_weights

PHYSICAL_GPU_INDEX = "2"
DEFAULT_DENSE_MODEL = Path(
    "/data1/lmcache_kv/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
SERVED_MODEL_NAME = "mixed-test"
SCENARIO_NAMES = (
    "late_short_prefill",
    "late_chunked_prefill",
    "decode4_prefill2",
)


@dataclass(frozen=True)
class StreamResult:
    http_status: int
    saw_done: bool
    output: str
    completion_tokens: int


@dataclass(frozen=True)
class ScenarioPlan:
    name: str
    repeat_id: int
    request_a: tuple
    request_b: tuple


@dataclass(frozen=True)
class ScenarioResult:
    repeat_id: int
    requests: dict
    elapsed: float
    completion_tokens: int
    completion_tokens_per_second: float
    mixed_marker_found: bool


def _required_existing_path(variable_name, default_path=None):
    raw_path = os.environ.get(variable_name)
    if raw_path:
        path = Path(raw_path).expanduser()
    elif default_path is not None:
        path = default_path
    else:
        pytest.skip(f"{variable_name} is required for the mixed-batch A/B E2E suite")
    if not path.exists():
        pytest.skip(f"{variable_name} does not exist: {path}")
    return path.resolve()


def _require_free_gpu_memory(min_free_gib):
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                PHYSICAL_GPU_INDEX,
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        pytest.skip(f"cannot query free GPU memory: {exc}")

    free_mib = [float(line.strip()) for line in result.stdout.splitlines() if line.strip()]
    if not free_mib:
        pytest.skip("nvidia-smi reported no GPUs")
    if free_mib[0] < min_free_gib * 1024:
        pytest.skip(
            f"mixed-batch A/B E2E requires {min_free_gib:g} GiB free on physical "
            f"GPU {PHYSICAL_GPU_INDEX}; available is {free_mib[0] / 1024:.1f} GiB"
        )


def _find_free_port_pair():
    for _ in range(100):
        first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port == 65535:
                continue
            second.bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            continue
        finally:
            first.close()
            second.close()
    raise RuntimeError("could not find two consecutive free localhost ports")


def _log_tail(log_path, line_count=120):
    if not log_path.exists():
        return "<log file was not created>"
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-line_count:])


def _log_offset(log_path):
    return log_path.stat().st_size


def _read_log_since(log_path, offset):
    with log_path.open("rb") as log_file:
        log_file.seek(offset)
        return log_file.read().decode("utf-8", errors="replace")


def _wait_until_serving(process, base_url, log_path, timeout):
    deadline = time.monotonic() + timeout
    status_url = f"{base_url}/v1/cache/status"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            pytest.fail(
                f"server exited with code {return_code} before becoming ready\n"
                f"--- server log tail ---\n{_log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(status_url, timeout=2) as response:
                if response.getcode() == 200:
                    try:
                        status = json.load(response)
                    except (json.JSONDecodeError, UnicodeDecodeError):
                        status = None
                    if isinstance(status, dict) and status.get("state") == "serving":
                        return
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.5)

    pytest.fail(
        f"server did not reach cache state 'serving' within {timeout:g} seconds\n"
        f"--- server log tail ---\n{_log_tail(log_path)}"
    )


def _stream_chat(base_url, payload, timeout, on_first_text=None):
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    reasoning_parts = []
    content_parts = []
    saw_done = False
    notified_first_text = False

    def consume_event(event_data):
        nonlocal saw_done, notified_first_text
        event_data = event_data.strip()
        if not event_data:
            return False
        if event_data == "[DONE]":
            saw_done = True
            return True
        try:
            event = json.loads(event_data)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"invalid SSE JSON event: {event_data!r}") from exc
        choices = event.get("choices") or []
        if not choices:
            return False
        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning_content") or ""
        content = delta.get("content") or ""
        if reasoning:
            reasoning_parts.append(reasoning)
        if content:
            content_parts.append(content)
        if not notified_first_text and (reasoning + content).strip():
            notified_first_text = True
            if on_first_text is not None:
                on_first_text()
        return False

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status = response.getcode()
            if status != 200:
                body = response.read().decode("utf-8", errors="replace")
                raise AssertionError(f"chat completion returned HTTP {status}: {body}")
            data_lines = []
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data_lines:
                        should_stop = consume_event("\n".join(data_lines))
                        data_lines = []
                        if should_stop:
                            break
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    value = line[5:]
                    if value.startswith(" "):
                        value = value[1:]
                    data_lines.append(value)
            if data_lines and not saw_done:
                consume_event("\n".join(data_lines))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(
            f"chat completion returned HTTP {exc.code}: {body}"
        ) from exc

    return StreamResult(
        http_status=status,
        saw_done=saw_done,
        output="".join(reasoning_parts) + "".join(content_parts),
        completion_tokens=int(payload["max_tokens"]),
    )


def _assert_complete_stream(result, label):
    assert result.http_status == 200, f"{label} returned HTTP {result.http_status}"
    assert result.saw_done, f"{label} ended without a [DONE] event"
    assert result.output.strip(), f"{label} returned no reasoning_content or content"


def _chat_request(prompt, max_tokens):
    return {
        "model": SERVED_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": True,
        "max_tokens": max_tokens,
        "stream": True,
    }


def _decode_prompt(scenario_name, repeat_id, request_id):
    repeat_label = "" if repeat_id is None else f", repeat {repeat_id}"
    return (
        f"Scenario {scenario_name}{repeat_label}, request {request_id}. "
        "Output a numbered list starting at 1, with one concise item per line. "
        "Continue until stopped and add no introduction."
    )


def _short_prefill_prompt(scenario_name, repeat_id, request_id):
    return (
        f"Scenario {scenario_name}, repeat {repeat_id}, request {request_id}. "
        "In one short sentence, explain why a library labels its shelves."
    )


def _long_prefill_prompt(scenario_name, repeat_id, request_id):
    repeated_sentence = (
        "During the quiet afternoon, the village library recorded each returned "
        "book, arranged the shelves, and prepared a simple reading list for the next day."
    )
    long_context = " ".join([repeated_sentence] * 85)
    repeat_label = "" if repeat_id is None else f", repeat {repeat_id}"
    return (
        f"Scenario {scenario_name}{repeat_label}, request {request_id}. "
        "Read the following ordinary repeated passage, then describe its main activity "
        f"in one short sentence.\n\n{long_context}"
    )


def _build_scenario_plans(repeats):
    plans = [
        ScenarioPlan(
            name="late_short_prefill",
            repeat_id=0,
            request_a=(
                (
                    "A0",
                    _chat_request(
                        _decode_prompt("late_short_prefill", 0, "A0"),
                        16,
                    ),
                ),
            ),
            request_b=(
                (
                    "B0",
                    _chat_request(
                        _short_prefill_prompt("late_short_prefill", 0, "B0"),
                        16,
                    ),
                ),
            ),
        ),
        ScenarioPlan(
            name="late_chunked_prefill",
            repeat_id=0,
            request_a=(
                (
                    "A0",
                    _chat_request(
                        _decode_prompt("late_chunked_prefill", 0, "A0"),
                        64,
                    ),
                ),
            ),
            request_b=(
                (
                    "B0",
                    _chat_request(
                        _long_prefill_prompt("late_chunked_prefill", 0, "B0"),
                        16,
                    ),
                ),
            ),
        ),
    ]
    for repeat_id in range(repeats):
        plans.append(
            ScenarioPlan(
                name="decode4_prefill2",
                repeat_id=repeat_id,
                request_a=tuple(
                    (
                        f"A{request_id}",
                        _chat_request(
                            _decode_prompt(
                                "decode4_prefill2",
                                None,
                                f"A{request_id}",
                            ),
                            64,
                        ),
                    )
                    for request_id in range(4)
                ),
                request_b=tuple(
                    (
                        f"B{request_id}",
                        _chat_request(
                            _long_prefill_prompt(
                                "decode4_prefill2",
                                None,
                                f"B{request_id}",
                            ),
                            16,
                        ),
                    )
                    for request_id in range(2)
                ),
            )
        )
    return tuple(plans)


def _run_late_prefill(base_url, plan, timeout):
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        request_b_futures = {}

        def submit_request_b():
            request_id, payload = plan.request_b[0]
            request_b_futures[request_id] = executor.submit(
                _stream_chat,
                base_url,
                payload,
                timeout,
            )

        request_a_id, request_a_payload = plan.request_a[0]
        request_a_future = executor.submit(
            _stream_chat,
            base_url,
            request_a_payload,
            timeout,
            submit_request_b,
        )
        requests = {request_a_id: request_a_future.result(timeout=timeout)}
        assert len(request_b_futures) == 1, "request A produced no non-empty text delta"
        for request_id, future in request_b_futures.items():
            requests[request_id] = future.result(timeout=timeout)
    return requests, time.monotonic() - started


def _run_decode4_prefill2(base_url, plan, timeout):
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        first_delta_lock = threading.Lock()
        first_delta_count = 0
        request_b_futures = {}

        def note_first_delta():
            nonlocal first_delta_count
            with first_delta_lock:
                first_delta_count += 1
                if first_delta_count == len(plan.request_a):
                    for request_id, payload in plan.request_b:
                        request_b_futures[request_id] = executor.submit(
                            _stream_chat,
                            base_url,
                            payload,
                            timeout,
                        )

        request_a_futures = {
            request_id: executor.submit(
                _stream_chat,
                base_url,
                payload,
                timeout,
                note_first_delta,
            )
            for request_id, payload in plan.request_a
        }
        requests = {
            request_id: future.result(timeout=timeout)
            for request_id, future in request_a_futures.items()
        }
        assert first_delta_count == 4, "not all four decode requests produced a text delta"
        assert len(request_b_futures) == 2, "both prefill requests were not submitted"
        for request_id, future in request_b_futures.items():
            requests[request_id] = future.result(timeout=timeout)
    return requests, time.monotonic() - started


def _marker_since(process, log_path, offset, marker, timeout):
    deadline = time.monotonic() + timeout
    while True:
        if marker in _read_log_since(log_path, offset):
            return True
        return_code = process.poll()
        if return_code is not None:
            pytest.fail(
                f"server exited with code {return_code} while waiting for {marker!r}\n"
                f"--- server log tail ---\n{_log_tail(log_path)}"
            )
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.25)


def _stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def _run_variant(
    policy,
    command,
    project_root,
    server_environment,
    base_url,
    log_path,
    plans,
    warmup_request,
    timeout,
):
    results = {scenario_name: [] for scenario_name in SCENARIO_NAMES}
    variant_command = [*command, "--batching-policy", policy]
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            variant_command,
            cwd=project_root,
            env=server_environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            _wait_until_serving(process, base_url, log_path, timeout)
            warmup_result = _stream_chat(base_url, warmup_request, timeout)
            _assert_complete_stream(warmup_result, f"{policy} warmup")

            for plan in plans:
                offset = _log_offset(log_path)
                if plan.name == "decode4_prefill2":
                    requests, elapsed = _run_decode4_prefill2(base_url, plan, timeout)
                else:
                    requests, elapsed = _run_late_prefill(base_url, plan, timeout)

                marker_wait = 60 if policy == "mixed" else 0.5
                marker_found = _marker_since(
                    process,
                    log_path,
                    offset,
                    "Mixed batch,",
                    marker_wait,
                )
                completion_tokens = sum(
                    request.completion_tokens for request in requests.values()
                )
                results[plan.name].append(
                    ScenarioResult(
                        repeat_id=plan.repeat_id,
                        requests=requests,
                        elapsed=elapsed,
                        completion_tokens=completion_tokens,
                        completion_tokens_per_second=completion_tokens / elapsed,
                        mixed_marker_found=marker_found,
                    )
                )
        except Exception as exc:
            return_code = process.poll()
            if return_code is not None:
                raise AssertionError(
                    f"{policy} server exited with code {return_code}\n"
                    f"--- server log tail ---\n{_log_tail(log_path)}"
                ) from exc
            raise
        finally:
            _stop_process(process)
    return results


@pytest.fixture(scope="module")
def mixed_ab_results(tmp_path_factory):
    project_root = Path(__file__).resolve().parents[2]
    model_path = _required_existing_path(
        "FREETOKEN_MIXED_TEST_MODEL",
        DEFAULT_DENSE_MODEL,
    )
    boot_timeout = float(os.environ.get("FREETOKEN_MIXED_BOOT_TIMEOUT", "600"))
    min_free_gib = float(os.environ.get("FREETOKEN_MIXED_MIN_FREE_GIB", "20"))
    repeats = int(os.environ.get("FREETOKEN_MIXED_AB_REPEATS", "3"))
    if repeats < 1:
        pytest.fail("FREETOKEN_MIXED_AB_REPEATS must be at least 1")
    _require_free_gpu_memory(min_free_gib)

    port = _find_free_port_pair()
    base_url = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        str(model_path),
        "--served-model-name",
        SERVED_MODEL_NAME,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-running-requests",
        "8",
        "--max-prefill-length",
        "256",
        "--decode-log-interval",
        "1",
        "--cache-type",
        "naive",
    ]
    server_environment = os.environ.copy()
    server_environment["PYTHONPATH"] = "python"
    server_environment["PYTHONUNBUFFERED"] = "1"
    server_environment["CUDA_VISIBLE_DEVICES"] = PHYSICAL_GPU_INDEX
    warmup_request = _chat_request(
        "Reply with the single word warmup.",
        4,
    )
    plans = _build_scenario_plans(repeats)
    output_directory = tmp_path_factory.mktemp("mixed-batch-ab")

    legacy = _run_variant(
        policy="legacy",
        command=command,
        project_root=project_root,
        server_environment=server_environment,
        base_url=base_url,
        log_path=output_directory / "legacy.log",
        plans=plans,
        warmup_request=warmup_request,
        timeout=boot_timeout,
    )
    mixed = _run_variant(
        policy="mixed",
        command=command,
        project_root=project_root,
        server_environment=server_environment,
        base_url=base_url,
        log_path=output_directory / "mixed.log",
        plans=plans,
        warmup_request=warmup_request,
        timeout=boot_timeout,
    )
    return {
        "legacy": legacy,
        "mixed": mixed,
        "repeats": repeats,
    }


def _assert_scenario_structure(mixed_ab_results, scenario_name):
    legacy_runs = mixed_ab_results["legacy"][scenario_name]
    mixed_runs = mixed_ab_results["mixed"][scenario_name]
    expected_run_count = (
        mixed_ab_results["repeats"] if scenario_name == "decode4_prefill2" else 1
    )
    expected_completion_tokens = {
        "late_short_prefill": 32,
        "late_chunked_prefill": 80,
        "decode4_prefill2": 288,
    }[scenario_name]
    expected_request_ids = (
        {"A0", "A1", "A2", "A3", "B0", "B1"}
        if scenario_name == "decode4_prefill2"
        else {"A0", "B0"}
    )
    expected_repeat_ids = (
        list(range(expected_run_count))
        if scenario_name == "decode4_prefill2"
        else [0]
    )
    expected_request_completion_tokens = {
        request_id: (16 if request_id.startswith("B") else 64)
        for request_id in expected_request_ids
    }
    if scenario_name == "late_short_prefill":
        expected_request_completion_tokens["A0"] = 16

    for policy, runs in (("legacy", legacy_runs), ("mixed", mixed_runs)):
        assert len(runs) == expected_run_count
        assert [run.repeat_id for run in runs] == expected_repeat_ids
        for run in runs:
            if policy == "legacy":
                assert not run.mixed_marker_found, (
                    f"legacy logged a mixed batch in {scenario_name} "
                    f"repeat {run.repeat_id}"
                )
            else:
                assert run.mixed_marker_found, (
                    f"mixed logged no mixed batch in {scenario_name} "
                    f"repeat {run.repeat_id}"
                )
            assert run.completion_tokens == expected_completion_tokens
            assert set(run.requests) == expected_request_ids
            for request_id, request in run.requests.items():
                assert request.completion_tokens == expected_request_completion_tokens[
                    request_id
                ]
                _assert_complete_stream(
                    request,
                    f"{policy} {scenario_name} repeat {run.repeat_id} {request_id}",
                )

    return legacy_runs, mixed_runs


def _difference_description(left_label, left, right_label, right):
    shared_length = min(len(left), len(right))
    difference_index = next(
        (index for index in range(shared_length) if left[index] != right[index]),
        shared_length,
    )
    snippet_start = max(0, difference_index - 24)
    snippet_end = difference_index + 24
    return (
        f"first differing character={difference_index}; "
        f"{left_label}[{snippet_start}:{snippet_end}]="
        f"{left[snippet_start:snippet_end]!r}; "
        f"{right_label}[{snippet_start}:{snippet_end}]="
        f"{right[snippet_start:snippet_end]!r}"
    )


def _assert_matching_scenario_runs(mixed_ab_results, scenario_name):
    legacy_runs, mixed_runs = _assert_scenario_structure(
        mixed_ab_results,
        scenario_name,
    )
    mismatches = []
    for legacy_run, mixed_run in zip(legacy_runs, mixed_runs):
        assert legacy_run.repeat_id == mixed_run.repeat_id
        for request_id in legacy_run.requests:
            legacy_request = legacy_run.requests[request_id]
            mixed_request = mixed_run.requests[request_id]
            assert mixed_request.completion_tokens == legacy_request.completion_tokens
            if mixed_request.output != legacy_request.output:
                mismatches.append(
                    f"policy=legacy-vs-mixed repeat={legacy_run.repeat_id} "
                    f"request={request_id}: "
                    + _difference_description(
                        "legacy",
                        legacy_request.output,
                        "mixed",
                        mixed_request.output,
                    )
                )

    assert not mismatches, (
        f"{scenario_name} output mismatches:\n" + "\n".join(mismatches)
    )


@pytest.mark.parametrize(
    "scenario_name",
    ("late_short_prefill", "late_chunked_prefill"),
    ids=("late_short_prefill", "late_chunked_prefill"),
)
def test_mixed_scenario_matches_legacy(mixed_ab_results, scenario_name):
    _assert_matching_scenario_runs(mixed_ab_results, scenario_name)


def test_decode4_prefill2_late_prefills_match_legacy(mixed_ab_results):
    legacy_runs, mixed_runs = _assert_scenario_structure(
        mixed_ab_results,
        "decode4_prefill2",
    )
    mismatches = []
    for legacy_run, mixed_run in zip(legacy_runs, mixed_runs):
        for request_id in ("B0", "B1"):
            legacy_output = legacy_run.requests[request_id].output
            mixed_output = mixed_run.requests[request_id].output
            if mixed_output != legacy_output:
                mismatches.append(
                    f"repeat={legacy_run.repeat_id} request={request_id}: "
                    + _difference_description(
                        "legacy",
                        legacy_output,
                        "mixed",
                        mixed_output,
                    )
                )

    assert not mismatches, (
        "decode4_prefill2 late-prefill output mismatches:\n" + "\n".join(mismatches)
    )


def test_decode4_prefill2_request_accounting_is_consistent(mixed_ab_results):
    legacy_runs, mixed_runs = _assert_scenario_structure(
        mixed_ab_results,
        "decode4_prefill2",
    )
    expected = {
        "A0": 64,
        "A1": 64,
        "A2": 64,
        "A3": 64,
        "B0": 16,
        "B1": 16,
    }

    for runs in (legacy_runs, mixed_runs):
        accounting = [
            {
                request_id: request.completion_tokens
                for request_id, request in run.requests.items()
            }
            for run in runs
        ]
        assert accounting == [expected] * mixed_ab_results["repeats"]


def test_mixed_improves_service_throughput(mixed_ab_results):
    legacy_runs, mixed_runs = _assert_scenario_structure(
        mixed_ab_results,
        "decode4_prefill2",
    )
    legacy_elapsed = [run.elapsed for run in legacy_runs]
    mixed_elapsed = [run.elapsed for run in mixed_runs]
    legacy_throughput = [run.completion_tokens_per_second for run in legacy_runs]
    mixed_throughput = [run.completion_tokens_per_second for run in mixed_runs]
    legacy_median_elapsed = statistics.median(legacy_elapsed)
    mixed_median_elapsed = statistics.median(mixed_elapsed)
    legacy_median_throughput = statistics.median(legacy_throughput)
    mixed_median_throughput = statistics.median(mixed_throughput)
    speedup = mixed_median_throughput / legacy_median_throughput
    summary = (
        "decode4_prefill2 A/B performance\n"
        f"legacy elapsed={legacy_elapsed}, median={legacy_median_elapsed:.3f}s, "
        f"median throughput={legacy_median_throughput:.2f} completion tokens/s\n"
        f"mixed elapsed={mixed_elapsed}, median={mixed_median_elapsed:.3f}s, "
        f"median throughput={mixed_median_throughput:.2f} completion tokens/s\n"
        f"throughput speedup={speedup:.3f}x"
    )
    print(summary)
    assert mixed_median_throughput > legacy_median_throughput, summary
