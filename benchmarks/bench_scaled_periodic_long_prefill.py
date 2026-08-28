#!/usr/bin/env python3
"""Three live decoders under a serial stream of long scaled-MoE prefills."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
from pathlib import Path
import shutil
import tempfile
import threading
import time
from typing import Any, Iterable

if __package__:
    from . import bench_lab_agent_policies as lab
    from . import bench_scaled_expert_contention as scaled
else:
    import bench_lab_agent_policies as lab
    import bench_scaled_expert_contention as scaled


HERE = Path(__file__).resolve().parent
DRIVER_COUNT = 3
DRIVER_PROMPT_TOKENS = 128
DRIVER_DECODE_TOKENS = 2048
LONG_PREFILL_REQUESTS = 6
LONG_PREFILL_PROMPT_TOKENS = 12288
LONG_PREFILL_DECODE_TOKENS = 1
LONG_PREFILL_THINK_SECONDS = 0.100
MAX_PREFILL_LENGTH = 8192
LONG_PREFILL_CHUNKS = math.ceil(LONG_PREFILL_PROMPT_TOKENS / MAX_PREFILL_LENGTH)
MAX_RUNNING_REQUESTS = 4
MAX_SEQ_LEN = 16384
CUDA_GRAPH_MAX_BS = 8
WARMUP_DRIVER_DECODE_TOKENS = 256
REQUESTS_PER_REPETITION = DRIVER_COUNT + LONG_PREFILL_REQUESTS
PROMPT_TOKENS_PER_REPETITION = (
    DRIVER_COUNT * DRIVER_PROMPT_TOKENS
    + LONG_PREFILL_REQUESTS * LONG_PREFILL_PROMPT_TOKENS
)
DECODE_TOKENS_PER_REPETITION = (
    DRIVER_COUNT * DRIVER_DECODE_TOKENS
    + LONG_PREFILL_REQUESTS * LONG_PREFILL_DECODE_TOKENS
)
DEFAULT_MODES = (
    {
        "name": "legacy",
        "batching_policy": "legacy",
    },
    {
        "name": "layered_pipeline_g1_wave2",
        "batching_policy": "layered-pipeline",
        "prefill_layer_group_size": 1,
        "prefill_wave_max_chunks": 2,
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18360)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--ft-executable")
    parser.add_argument("--moe-cache-size", type=int, default=24)
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if args.moe_cache_size < 1:
        parser.error("--moe-cache-size must be at least 1")
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    if args.server_timeout <= 0:
        parser.error("--server-timeout must be positive")
    return args


def mode_contracts() -> list[dict[str, Any]]:
    return [dict(mode) for mode in DEFAULT_MODES]


def server_workload(cache_size: int) -> dict[str, Any]:
    workload = lab.load_workload()
    workload["public_server_config"].update(
        {
            "max_running_requests": MAX_RUNNING_REQUESTS,
            "max_seq_len_override": MAX_SEQ_LEN,
            "max_prefill_length": MAX_PREFILL_LENGTH,
            "dtype": "float16",
            "attention_backend": "triton",
            "moe_cache_size": cache_size,
            "cuda_graph_max_bs": CUDA_GRAPH_MAX_BS,
            "cache_type": "radix",
        }
    )
    return workload


def server_command(
    ft_executable: str,
    model_path: Path | str,
    mode: dict[str, Any],
    host: str,
    port: int,
    cache_size: int,
) -> list[str]:
    return lab.server_command(
        ft_executable,
        model_path,
        mode,
        host,
        port,
        server_workload(cache_size),
    ) + ["--moe-collect-stats"]


def create_long_context_scaled_model(model_path: Path) -> None:
    scaled.create_scaled_qwen3_moe(model_path)
    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["max_position_embeddings"] = MAX_SEQ_LEN
    lab.write_json(config_path, config)


def checkpoint_max_position_embeddings(model_path: Path) -> int:
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    value = int(config.get("max_position_embeddings", 0))
    if value < 1:
        raise ValueError("scaled checkpoint has no positive max_position_embeddings")
    return value


def materialize_inputs(tokenizer: Any, repetitions: int) -> dict[str, Any]:
    measured_requests = repetitions * REQUESTS_PER_REPETITION
    first_tokens = lab.first_token_candidates(tokenizer, measured_requests + 3)
    continuation = lab.continuation_token_pieces(tokenizer)
    materials: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        offset = repetition * REQUESTS_PER_REPETITION
        drivers: list[dict[str, Any]] = []
        for driver_index in range(DRIVER_COUNT):
            text, token_ids = lab.materialize_segment_text(
                tokenizer,
                DRIVER_PROMPT_TOKENS,
                20260828 + repetition * 10000 + driver_index,
                f"periodic-driver-r{repetition}-d{driver_index}",
                continuation,
                first_token=first_tokens[offset + driver_index],
            )
            drivers.append({"text": text, "token_count": len(token_ids)})

        prefills: list[dict[str, Any]] = []
        for prefill_index in range(LONG_PREFILL_REQUESTS):
            material_index = offset + DRIVER_COUNT + prefill_index
            text, token_ids = lab.materialize_segment_text(
                tokenizer,
                LONG_PREFILL_PROMPT_TOKENS,
                30260828 + repetition * 10000 + prefill_index,
                f"periodic-long-prefill-r{repetition}-p{prefill_index}",
                continuation,
                first_token=first_tokens[material_index],
            )
            prefills.append({"text": text, "token_count": len(token_ids)})
        materials.append({"drivers": drivers, "long_prefills": prefills})

    warmup_driver_text, warmup_driver_ids = lab.materialize_segment_text(
        tokenizer,
        DRIVER_PROMPT_TOKENS,
        40260828,
        "periodic-long-prefill-warmup-driver",
        continuation,
        first_token=first_tokens[-3],
    )
    warmup_prefill_text, warmup_prefill_ids = lab.materialize_segment_text(
        tokenizer,
        LONG_PREFILL_PROMPT_TOKENS,
        40260829,
        "periodic-long-prefill-warmup-prefill",
        continuation,
        first_token=first_tokens[-2],
    )
    readiness_id, readiness_text = first_tokens[-1]
    return {
        "repetitions": materials,
        "warmup": {
            "driver": {
                "text": warmup_driver_text,
                "token_count": len(warmup_driver_ids),
            },
            "long_prefill": {
                "text": warmup_prefill_text,
                "token_count": len(warmup_prefill_ids),
            },
        },
        "readiness_prompt_token_id": readiness_id,
        "readiness_prompt_text": readiness_text,
    }


def wait_for_idle_snapshot(
    server: lab.PublicServer,
    minimum_count: int,
    timeout: float,
) -> list[dict[str, int]]:
    deadline = time.monotonic() + timeout
    stable_since: float | None = None
    last_count = -1
    while time.monotonic() < deadline:
        snapshots = server.moe_cache_stats_snapshots()
        count = len(snapshots)
        if server.process is not None and server.process.poll() is not None:
            raise RuntimeError(
                f"server exited before idle stats snapshot: {server.log_tail()}"
            )
        if count >= minimum_count:
            if count != last_count:
                last_count = count
                stable_since = time.monotonic()
            elif stable_since is not None and time.monotonic() - stable_since >= 0.2:
                return snapshots
        time.sleep(0.05)
    raise TimeoutError(f"idle stats did not stabilize within {timeout}s")


def value_distribution(
    values: Iterable[float | int], percents: Iterable[int] = (50, 95)
) -> dict[str, float | None]:
    return lab.distribution((float(value) for value in values), percents)


def gap_distribution(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = list(values)
    return {
        "sample_count": len(samples),
        **value_distribution(samples, (50, 95, 99)),
        "max": max(samples) if samples else None,
    }


def finalize_request(
    record: dict[str, Any],
    *,
    repetition: int,
    request_id: str,
    role: str,
    expected_prompt_tokens: int,
    expected_completion_tokens: int,
) -> None:
    failures: list[str] = []
    usage = record.get("usage")
    events = record["nonempty_text_events"]
    if record.get("error"):
        failures.append("HTTP or stream error")
    if not events:
        failures.append("no non-empty SSE text event")
        record["ttft_seconds"] = None
    else:
        record["ttft_seconds"] = (
            events[0]["at_seconds"] - record["submitted_at_seconds"]
        )

    record["tpot_seconds"] = None
    record["cached_tokens"] = None
    record["actual_new_prefill_tokens"] = None
    if not isinstance(usage, dict):
        failures.append("missing usage")
    else:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if "prompt_tokens_details" not in usage:
            cached_tokens = 0
        else:
            details = usage["prompt_tokens_details"]
            cached_tokens = (
                details.get("cached_tokens") if isinstance(details, dict) else None
            )
        if prompt_tokens != expected_prompt_tokens:
            failures.append(
                f"prompt_tokens {prompt_tokens!r} != {expected_prompt_tokens}"
            )
        if completion_tokens != expected_completion_tokens:
            failures.append(
                f"completion_tokens {completion_tokens!r} != "
                f"{expected_completion_tokens}"
            )
        if not isinstance(cached_tokens, int):
            failures.append("usage.prompt_tokens_details.cached_tokens unavailable")
        elif not isinstance(prompt_tokens, int):
            failures.append("usage.prompt_tokens is not an integer")
        else:
            record["cached_tokens"] = cached_tokens
            record["actual_new_prefill_tokens"] = prompt_tokens - cached_tokens
            if not 0 <= cached_tokens <= prompt_tokens:
                failures.append("cached_tokens is outside the prompt")
            if role == "long_prefill" and cached_tokens != 0:
                failures.append(
                    f"independent long prefill unexpectedly cached {cached_tokens} tokens"
                )
        if events and isinstance(completion_tokens, int) and completion_tokens > 1:
            record["tpot_seconds"] = (
                record["response_complete_at_seconds"]
                - events[0]["at_seconds"]
            ) / (completion_tokens - 1)

    record.update(
        {
            "repetition": repetition,
            "request_id": request_id,
            "role": role,
            "nonempty_text_event_count": len(events),
            "choice_event_count": len(record["choice_events"]),
            "full_request_latency_seconds": (
                record["response_complete_at_seconds"]
                - record["submitted_at_seconds"]
            ),
            "measurement_failed": bool(failures),
            "measurement_failures": failures,
        }
    )


def drivers_still_active(
    driver_futures: list[concurrent.futures.Future[dict[str, Any]]],
) -> tuple[bool, list[int]]:
    completed = [index for index, future in enumerate(driver_futures) if future.done()]
    return not completed, completed


def wait_for_all_first_text_events(
    events: list[threading.Event],
    driver_futures: list[concurrent.futures.Future[dict[str, Any]]],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while not all(event.is_set() for event in events):
        for index, (event, future) in enumerate(zip(events, driver_futures)):
            if not event.is_set() and future.done():
                result = future.result()
                raise RuntimeError(
                    f"driver {index} completed before its first non-empty SSE event: "
                    f"{result.get('error') or result.get('usage')}"
                )
        if time.monotonic() >= deadline:
            missing = [index for index, event in enumerate(events) if not event.is_set()]
            raise TimeoutError(f"drivers did not emit first SSE events: {missing}")
        time.sleep(0.01)


def run_path_warmup(
    *,
    base_url: str,
    material: dict[str, Any],
    server_timeout: float,
) -> dict[str, Any]:
    origin = time.perf_counter()
    first_text_event = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        driver_future = executor.submit(
            lab.request_completion,
            base_url,
            material["driver"]["text"],
            material["driver"]["token_count"],
            WARMUP_DRIVER_DECODE_TOKENS,
            70260828,
            origin,
            first_text_event=first_text_event,
        )
        wait_for_all_first_text_events(
            [first_text_event], [driver_future], server_timeout
        )
        if driver_future.done():
            raise RuntimeError("path-warmup driver completed before long prefill")
        long_prefill = lab.request_completion(
            base_url,
            material["long_prefill"]["text"],
            material["long_prefill"]["token_count"],
            LONG_PREFILL_DECODE_TOKENS,
            70260829,
            origin,
        )
        driver_active_at_prefill_completion = not driver_future.done()
        driver = driver_future.result()

    finalize_request(
        driver,
        repetition=-1,
        request_id="warmup_driver",
        role="warmup_driver",
        expected_prompt_tokens=DRIVER_PROMPT_TOKENS,
        expected_completion_tokens=WARMUP_DRIVER_DECODE_TOKENS,
    )
    finalize_request(
        long_prefill,
        repetition=-1,
        request_id="warmup_long_prefill",
        role="long_prefill",
        expected_prompt_tokens=LONG_PREFILL_PROMPT_TOKENS,
        expected_completion_tokens=LONG_PREFILL_DECODE_TOKENS,
    )
    failures = [
        *driver["measurement_failures"],
        *long_prefill["measurement_failures"],
    ]
    if not driver_active_at_prefill_completion:
        failures.append("path-warmup driver completed before long prefill")
    if failures:
        raise RuntimeError("public path warmup failed: " + "; ".join(failures))
    return {
        "excluded_from_measurement": True,
        "driver_active_at_long_prefill_completion": True,
        "driver": {
            "usage": driver["usage"],
            "ttft_seconds": driver["ttft_seconds"],
            "tpot_seconds": driver["tpot_seconds"],
            "latency_seconds": driver["full_request_latency_seconds"],
        },
        "long_prefill": {
            "usage": long_prefill["usage"],
            "cached_tokens": long_prefill["cached_tokens"],
            "ttft_seconds": long_prefill["ttft_seconds"],
            "latency_seconds": long_prefill["full_request_latency_seconds"],
        },
    }


def request_driver_after_barrier(
    *,
    barrier: threading.Barrier,
    clock: dict[str, float],
    base_url: str,
    prompt: dict[str, Any],
    request_seed: int,
    first_text_event: threading.Event,
) -> dict[str, Any]:
    barrier.wait()
    return lab.request_completion(
        base_url,
        prompt["text"],
        prompt["token_count"],
        DRIVER_DECODE_TOKENS,
        request_seed,
        clock["origin"],
        first_text_event=first_text_event,
    )


def driver_gap_samples(
    drivers: list[dict[str, Any]],
    prefill_windows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for driver in drivers:
        events = driver["nonempty_text_events"]
        for ordinal, (earlier, later) in enumerate(zip(events, events[1:])):
            start = earlier["at_seconds"]
            end = later["at_seconds"]
            intersecting_windows = [
                window["request_id"]
                for window in prefill_windows
                if start <= window["response_complete_at_seconds"]
                and end >= window["submitted_at_seconds"]
            ]
            samples.append(
                {
                    "driver_request_id": driver["request_id"],
                    "gap_ordinal": ordinal,
                    "start_at_seconds": start,
                    "end_at_seconds": end,
                    "gap_seconds": end - start,
                    "intersects_any_prefill_active_window": bool(
                        intersecting_windows
                    ),
                    "intersecting_prefill_request_ids": intersecting_windows,
                }
            )
    return samples


def run_one_repetition(
    *,
    base_url: str,
    material: dict[str, Any],
    repetition: int,
    server_timeout: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    clock: dict[str, float] = {}

    def set_origin() -> None:
        clock["origin"] = time.perf_counter()

    first_text_events = [threading.Event() for _ in range(DRIVER_COUNT)]
    barrier = threading.Barrier(DRIVER_COUNT + 1, action=set_origin)
    prefill_records: list[dict[str, Any]] = []
    prefill_windows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=MAX_RUNNING_REQUESTS
    ) as executor:
        driver_futures = [
            executor.submit(
                request_driver_after_barrier,
                barrier=barrier,
                clock=clock,
                base_url=base_url,
                prompt=prompt,
                request_seed=50260828 + repetition * 10000 + driver_index,
                first_text_event=first_text_events[driver_index],
            )
            for driver_index, prompt in enumerate(material["drivers"])
        ]
        barrier.wait()
        wait_for_all_first_text_events(
            first_text_events, driver_futures, server_timeout
        )
        burst_released_at = lab.relative_now(clock["origin"])

        for prefill_index, prompt in enumerate(material["long_prefills"]):
            active, completed = drivers_still_active(driver_futures)
            if not active:
                raise RuntimeError(
                    "drivers completed before long prefill submission "
                    f"{prefill_index}: {completed}"
                )
            future = executor.submit(
                lab.request_completion,
                base_url,
                prompt["text"],
                prompt["token_count"],
                LONG_PREFILL_DECODE_TOKENS,
                60260828 + repetition * 10000 + prefill_index,
                clock["origin"],
            )
            completed_during_window: set[int] = set()
            while True:
                try:
                    record = future.result(timeout=0.01)
                    break
                except concurrent.futures.TimeoutError:
                    completed_during_window.update(
                        index
                        for index, driver_future in enumerate(driver_futures)
                        if driver_future.done()
                    )
            completed_during_window.update(
                index
                for index, driver_future in enumerate(driver_futures)
                if driver_future.done()
            )
            request_id = f"long_prefill_{prefill_index}"
            finalize_request(
                record,
                repetition=repetition,
                request_id=request_id,
                role="long_prefill",
                expected_prompt_tokens=LONG_PREFILL_PROMPT_TOKENS,
                expected_completion_tokens=LONG_PREFILL_DECODE_TOKENS,
            )
            record["drivers_active_throughout_request"] = not completed_during_window
            record["drivers_completed_during_request"] = sorted(
                completed_during_window
            )
            prefill_records.append(record)
            prefill_windows.append(
                {
                    "request_id": request_id,
                    "submitted_at_seconds": record["submitted_at_seconds"],
                    "response_complete_at_seconds": record[
                        "response_complete_at_seconds"
                    ],
                    "duration_seconds": record["full_request_latency_seconds"],
                    "drivers_active_at_submission": True,
                    "drivers_active_at_completion": not completed_during_window,
                    "drivers_completed_during_window": sorted(
                        completed_during_window
                    ),
                }
            )
            if completed_during_window:
                raise RuntimeError(
                    f"drivers completed during long prefill {prefill_index}: "
                    f"{sorted(completed_during_window)}"
                )
            if prefill_index + 1 < LONG_PREFILL_REQUESTS:
                time.sleep(LONG_PREFILL_THINK_SECONDS)

        active, completed = drivers_still_active(driver_futures)
        if not active:
            raise RuntimeError(
                f"drivers completed when final long prefill finished: {completed}"
            )
        driver_records = [future.result() for future in driver_futures]

    for driver_index, record in enumerate(driver_records):
        finalize_request(
            record,
            repetition=repetition,
            request_id=f"driver_{driver_index}",
            role="driver",
            expected_prompt_tokens=DRIVER_PROMPT_TOKENS,
            expected_completion_tokens=DRIVER_DECODE_TOKENS,
        )

    records = driver_records + prefill_records
    gaps = driver_gap_samples(driver_records, prefill_windows)
    intersecting_gaps = [
        sample
        for sample in gaps
        if sample["intersects_any_prefill_active_window"]
    ]
    makespan = max(record["response_complete_at_seconds"] for record in records)
    summary = {
        "benchmark_start_perf_counter": clock["origin"],
        "all_drivers_first_nonempty_sse_at_seconds": burst_released_at,
        "request_count": len(records),
        "submitted_prompt_tokens": sum(
            record["submitted_prompt_tokens"] for record in records
        ),
        "usage_prompt_tokens": sum(
            record["usage"]["prompt_tokens"]
            for record in records
            if isinstance(record.get("usage"), dict)
        ),
        "cached_tokens": sum(
            record["cached_tokens"]
            for record in records
            if isinstance(record.get("cached_tokens"), int)
        ),
        "actual_new_prefill_tokens": sum(
            record["actual_new_prefill_tokens"]
            for record in records
            if isinstance(record.get("actual_new_prefill_tokens"), int)
        ),
        "decode_tokens": sum(
            record["usage"]["completion_tokens"]
            for record in records
            if isinstance(record.get("usage"), dict)
        ),
        "makespan_seconds": makespan,
        "requests_per_second": len(records) / makespan,
        "submitted_prompt_throughput_tokens_per_second": (
            PROMPT_TOKENS_PER_REPETITION / makespan
        ),
        "decode_throughput_tokens_per_second": (
            DECODE_TOKENS_PER_REPETITION / makespan
        ),
        "measurement_failed_requests": sum(
            record["measurement_failed"] for record in records
        ),
        "prefill_active_windows": prefill_windows,
        "driver_adjacent_nonempty_sse_text_event_gap_samples": gaps,
        "driver_adjacent_nonempty_sse_text_event_gap_seconds": {
            "overall": gap_distribution(
                sample["gap_seconds"] for sample in gaps
            ),
            "intersecting_any_prefill_active_window": gap_distribution(
                sample["gap_seconds"] for sample in intersecting_gaps
            ),
        },
    }
    return records, summary


def sum_wave_fields(waves: list[dict[str, int]]) -> dict[str, int]:
    if not waves:
        return {}
    return {
        name: sum(wave[name] for wave in waves)
        for name in waves[0]
    }


def validate_pipeline_waves(
    waves: list[dict[str, int]],
) -> tuple[list[str], list[dict[str, int]], dict[str, Any]]:
    failures: list[str] = []
    structure = sum_wave_fields(waves)
    if structure.get("reqs") != REQUESTS_PER_REPETITION:
        failures.append(
            f"pipeline wave reqs {structure.get('reqs')} != "
            f"{REQUESTS_PER_REPETITION}"
        )
    long_waves = waves[-LONG_PREFILL_REQUESTS:]
    if len(long_waves) != LONG_PREFILL_REQUESTS:
        failures.append(
            f"found {len(long_waves)} ordered long-prefill waves, expected "
            f"{LONG_PREFILL_REQUESTS}"
        )
    for wave in long_waves:
        expected = {
            "reqs": 1,
            "groups": scaled.MODEL_CONTRACT["layers"],
            "group_forwards": scaled.MODEL_CONTRACT["layers"],
            "iterations": scaled.MODEL_CONTRACT["layers"],
            "decode_iterations": scaled.MODEL_CONTRACT["layers"],
            "prefill_layer_prepares": scaled.MODEL_CONTRACT["layers"],
        }
        for name, value in expected.items():
            if wave[name] != value:
                failures.append(
                    f"long pipeline wave {name} {wave[name]} != {value}"
                )
    identification = {
        "method": "last six wave logs in completion order",
        "limitation": (
            "pipeline wave logs contain neither request ids nor token counts"
        ),
    }
    return failures, long_waves, identification


def validate_stats(delta: dict[str, int]) -> list[str]:
    failures: list[str] = []
    if delta["decode_missing_rows"] > delta["decode_active_rows"]:
        failures.append("decode missing rows exceed active rows")
    if delta["prefill_hit_rows"] > delta["prefill_rows"]:
        failures.append("prefill hit rows exceed prefill rows")
    row_bytes = delta["expert_row_bytes"]
    if row_bytes <= 0 or delta["prefill_h2d_bytes_total"] % row_bytes:
        failures.append("prefill H2D bytes are not integral expert rows")
    return failures


def validate_repetition(
    records: list[dict[str, Any]],
    summary: dict[str, Any],
    stats_delta: dict[str, int],
    mode: dict[str, Any],
    waves: list[dict[str, int]],
) -> tuple[list[str], list[dict[str, int]], dict[str, Any] | None]:
    failures: list[str] = []
    if len(records) != REQUESTS_PER_REPETITION:
        failures.append(
            f"request count {len(records)} != {REQUESTS_PER_REPETITION}"
        )
    if summary["submitted_prompt_tokens"] != PROMPT_TOKENS_PER_REPETITION:
        failures.append(
            f"submitted prompt tokens {summary['submitted_prompt_tokens']} != "
            f"{PROMPT_TOKENS_PER_REPETITION}"
        )
    if summary["usage_prompt_tokens"] != PROMPT_TOKENS_PER_REPETITION:
        failures.append(
            f"usage prompt tokens {summary['usage_prompt_tokens']} != "
            f"{PROMPT_TOKENS_PER_REPETITION}"
        )
    if summary["decode_tokens"] != DECODE_TOKENS_PER_REPETITION:
        failures.append(
            f"decode tokens {summary['decode_tokens']} != "
            f"{DECODE_TOKENS_PER_REPETITION}"
        )
    if summary["measurement_failed_requests"]:
        failures.append(
            f"{summary['measurement_failed_requests']} requests failed public accounting"
        )
    long_prefills = [record for record in records if record["role"] == "long_prefill"]
    if len(long_prefills) != LONG_PREFILL_REQUESTS:
        failures.append(
            f"long prefill request count {len(long_prefills)} != "
            f"{LONG_PREFILL_REQUESTS}"
        )
    if any(record.get("cached_tokens") != 0 for record in long_prefills):
        failures.append("one or more independent long prefills reported cached tokens")
    if any(
        not record.get("drivers_active_throughout_request", False)
        for record in long_prefills
    ):
        failures.append("one or more long prefills outlived a driver")
    failures.extend(validate_stats(stats_delta))

    long_waves: list[dict[str, int]] = []
    identification: dict[str, Any] | None = None
    if mode["batching_policy"] == "layered-pipeline":
        wave_failures, long_waves, identification = validate_pipeline_waves(waves)
        failures.extend(wave_failures)
    elif waves:
        failures.append("legacy mode unexpectedly emitted resident-wave logs")
    return failures, long_waves, identification


def metric_distribution(
    records: list[dict[str, Any]], role: str, field: str
) -> dict[str, float | None]:
    return value_distribution(
        record[field]
        for record in records
        if record["role"] == role and record.get(field) is not None
    )


def summarize_mode(
    records: list[dict[str, Any]], repetitions: list[dict[str, Any]]
) -> dict[str, Any]:
    all_gap_samples = [
        sample
        for repetition in repetitions
        for sample in repetition[
            "driver_adjacent_nonempty_sse_text_event_gap_samples"
        ]
    ]
    intersecting = [
        sample
        for sample in all_gap_samples
        if sample["intersects_any_prefill_active_window"]
    ]
    stats_fields = (
        "prefill_h2d_bytes_total",
        "decode_h2d_bytes",
        "total_expert_h2d_bytes",
        "decode_active_rows",
        "decode_missing_rows",
        "decode_layer_calls",
        "decode_fetched_rows",
        "prefill_rows",
        "prefill_hit_rows",
        "prefill_layer_prepares",
    )
    return {
        "driver_ttft_seconds": metric_distribution(records, "driver", "ttft_seconds"),
        "driver_tpot_seconds": metric_distribution(records, "driver", "tpot_seconds"),
        "long_prefill_ttft_seconds": metric_distribution(
            records, "long_prefill", "ttft_seconds"
        ),
        "long_prefill_active_window_seconds": value_distribution(
            window["duration_seconds"]
            for repetition in repetitions
            for window in repetition["prefill_active_windows"]
        ),
        "driver_adjacent_nonempty_sse_text_event_gap_seconds": {
            "overall": gap_distribution(
                sample["gap_seconds"] for sample in all_gap_samples
            ),
            "intersecting_any_prefill_active_window": gap_distribution(
                sample["gap_seconds"] for sample in intersecting
            ),
        },
        "makespan_seconds": value_distribution(
            repetition["makespan_seconds"] for repetition in repetitions
        ),
        "requests_per_second": value_distribution(
            repetition["requests_per_second"] for repetition in repetitions
        ),
        "submitted_prompt_throughput_tokens_per_second": value_distribution(
            repetition["submitted_prompt_throughput_tokens_per_second"]
            for repetition in repetitions
        ),
        "decode_throughput_tokens_per_second": value_distribution(
            repetition["decode_throughput_tokens_per_second"]
            for repetition in repetitions
        ),
        "moe_stats": {
            field: {
                **value_distribution(
                    repetition["moe_stats_delta"][field]
                    for repetition in repetitions
                ),
                "total": sum(
                    repetition["moe_stats_delta"][field]
                    for repetition in repetitions
                ),
            }
            for field in stats_fields
        },
        "request_count": len(records),
        "measurement_failed_requests": sum(
            record["measurement_failed"] for record in records
        ),
        "output_mismatch_requests": sum(
            record.get("output_mismatch") is not None for record in records
        ),
        "token_totals": {
            "prompt_tokens": sum(
                repetition["usage_prompt_tokens"] for repetition in repetitions
            ),
            "cached_tokens": sum(
                repetition["cached_tokens"] for repetition in repetitions
            ),
            "actual_new_prefill_tokens": sum(
                repetition["actual_new_prefill_tokens"]
                for repetition in repetitions
            ),
            "decode_tokens": sum(
                repetition["decode_tokens"] for repetition in repetitions
            ),
        },
    }


def dry_run_plan(
    args: argparse.Namespace,
    modes: list[dict[str, Any]],
) -> dict[str, Any]:
    executable = lab.find_ft_executable(args.ft_executable)
    model_path: Path | str = args.model or "<auto-generated-scaled-qwen3-moe>"
    return {
        "dry_run": True,
        "schema": "freetoken.scaled_periodic_long_prefill.v1",
        "model_contract": {
            **scaled.MODEL_CONTRACT,
            "max_position_embeddings": MAX_SEQ_LEN,
        },
        "workload_contract": workload_contract(args.repetitions, args.moe_cache_size),
        "modes": modes,
        "commands": [
            server_command(
                executable,
                model_path,
                mode,
                args.host,
                args.port,
                args.moe_cache_size,
            )
            for mode in modes
        ],
    }


def workload_contract(repetitions: int, cache_size: int) -> dict[str, Any]:
    return {
        "repetitions": repetitions,
        "driver_count": DRIVER_COUNT,
        "driver_prompt_tokens_each": DRIVER_PROMPT_TOKENS,
        "driver_completion_tokens_each": DRIVER_DECODE_TOKENS,
        "driver_release_barrier": "all first non-empty SSE text events",
        "long_prefill_requests": LONG_PREFILL_REQUESTS,
        "long_prefill_prompt_tokens_each": LONG_PREFILL_PROMPT_TOKENS,
        "long_prefill_completion_tokens_each": LONG_PREFILL_DECODE_TOKENS,
        "long_prefill_submission": "serial fourth lane while all drivers remain active",
        "long_prefill_think_time_ms": int(LONG_PREFILL_THINK_SECONDS * 1000),
        "requests_per_repetition": REQUESTS_PER_REPETITION,
        "prompt_tokens_per_repetition": PROMPT_TOKENS_PER_REPETITION,
        "decode_tokens_per_repetition": DECODE_TOKENS_PER_REPETITION,
        "max_prefill_length": MAX_PREFILL_LENGTH,
        "long_prefill_chunks_each": LONG_PREFILL_CHUNKS,
        "max_running_requests": MAX_RUNNING_REQUESTS,
        "max_seq_len_override": MAX_SEQ_LEN,
        "attention_backend": "triton",
        "cache_type": "radix",
        "cuda_graph_max_bs": CUDA_GRAPH_MAX_BS,
        "moe_cache_size": cache_size,
        "ignore_eos": True,
        "gap_metric": "adjacent non-empty SSE text-event gap",
        "measurement_excluded_path_warmup": {
            "driver_prompt_tokens": DRIVER_PROMPT_TOKENS,
            "driver_completion_tokens": WARMUP_DRIVER_DECODE_TOKENS,
            "release_long_prefill_after": "first non-empty driver SSE text event",
            "long_prefill_prompt_tokens": LONG_PREFILL_PROMPT_TOKENS,
            "long_prefill_completion_tokens": LONG_PREFILL_DECODE_TOKENS,
            "require_driver_active_at_long_prefill_completion": True,
        },
    }


def main() -> int:
    args = parse_args()
    modes = mode_contracts()
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, modes), indent=2, default=str))
        return 0

    executable = lab.find_ft_executable(args.ft_executable)
    owned_model_root: Path | None = None
    if args.model is None:
        owned_model_root = Path(tempfile.mkdtemp(prefix="freetoken-scaled-periodic-"))
        model_path = owned_model_root / "model"
        create_long_context_scaled_model(model_path)
    else:
        model_path = args.model.resolve()
        max_positions = checkpoint_max_position_embeddings(model_path)
        if max_positions < MAX_SEQ_LEN:
            raise ValueError(
                f"scaled checkpoint supports {max_positions} positions; "
                f"periodic long-prefill workload requires at least {MAX_SEQ_LEN}"
            )

    tokenizer = lab.load_tokenizer(model_path)
    materials = materialize_inputs(tokenizer, args.repetitions)
    del tokenizer
    output = args.output or (
        HERE
        / "results"
        / f"scaled_periodic_long_prefill_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result: dict[str, Any] = {
        "schema": "freetoken.scaled_periodic_long_prefill.v1",
        "created_at_unix_seconds": time.time(),
        "model_path": str(model_path),
        "auto_generated_model": args.model is None,
        "model_contract": {
            **scaled.MODEL_CONTRACT,
            "max_position_embeddings": MAX_SEQ_LEN,
        },
        "workload_contract": workload_contract(
            args.repetitions, args.moe_cache_size
        ),
        "metric_definitions": {
            "driver_tpot_seconds": (
                "(response complete - first non-empty SSE text event) / "
                "(usage completion tokens - 1)"
            ),
            "driver_adjacent_nonempty_sse_text_event_gap_seconds": (
                "client-observed interval between adjacent non-empty SSE text events; "
                "one event may contain more than one token"
            ),
            "intersecting_any_prefill_active_window": (
                "the closed SSE event interval overlaps at least one long-prefill "
                "submission-to-response-complete interval"
            ),
        },
        "reference_mode": modes[0]["name"],
        "modes": [],
    }
    references: dict[tuple[int, str], str] = {}
    base_url = f"http://{args.host}:{args.port}"
    try:
        for mode_index, mode in enumerate(modes):
            command = server_command(
                executable,
                model_path,
                mode,
                args.host,
                args.port,
                args.moe_cache_size,
            )
            mode_result: dict[str, Any] = {
                "name": mode["name"],
                "contract": mode,
                "server_command": command,
                "warmup": None,
                "repetitions": [],
                "requests": [],
                "layered_pipeline_waves": [],
                "layered_pipeline_structure": None,
                "summary": None,
                "server_log_tail": None,
                "error": None,
                "readiness_prompt_token_id": materials[
                    "readiness_prompt_token_id"
                ],
            }
            result["modes"].append(mode_result)
            server = lab.PublicServer(
                command,
                args.gpu,
                base_url,
                args.server_timeout,
                materials["readiness_prompt_text"],
            )
            try:
                server.start()
                readiness_snapshots = wait_for_idle_snapshot(
                    server, 1, args.server_timeout
                )
                warmup = run_path_warmup(
                    base_url=base_url,
                    material=materials["warmup"],
                    server_timeout=args.server_timeout,
                )
                snapshots = wait_for_idle_snapshot(
                    server, len(readiness_snapshots) + 1, args.server_timeout
                )
                baseline_snapshot = snapshots[-1]
                snapshot_count = len(snapshots)
                mode_result["warmup"] = warmup
                server.mark_measurement_start()
                pipeline_wave_cursor = 0

                for repetition in range(args.repetitions):
                    records, repetition_summary = run_one_repetition(
                        base_url=base_url,
                        material=materials["repetitions"][repetition],
                        repetition=repetition,
                        server_timeout=args.server_timeout,
                    )
                    final_snapshots = wait_for_idle_snapshot(
                        server, snapshot_count + 1, args.server_timeout
                    )
                    after_snapshot = final_snapshots[-1]
                    stats_delta = scaled.snapshot_delta(
                        baseline_snapshot, after_snapshot
                    )
                    stats_delta["prefill_h2d_rows"] = (
                        stats_delta["prefill_h2d_bytes_total"]
                        // stats_delta["expert_row_bytes"]
                    )

                    if mode["batching_policy"] == "layered-pipeline":
                        all_waves = server.layered_pipeline_waves()
                        waves = all_waves[pipeline_wave_cursor:]
                        pipeline_wave_cursor = len(all_waves)
                    else:
                        waves = []

                    for record in records:
                        key = (repetition, record["request_id"])
                        reference = references.get(key)
                        record["output_mismatch"] = (
                            lab.first_difference(reference, record["output_text"])
                            if reference is not None
                            else None
                        )
                    failures, long_waves, identification = validate_repetition(
                        records,
                        repetition_summary,
                        stats_delta,
                        mode,
                        waves,
                    )
                    repetition_summary.update(
                        {
                            "moe_stats_before": baseline_snapshot,
                            "moe_stats_after": after_snapshot,
                            "moe_stats_delta": stats_delta,
                            "idle_snapshot_count_delta": (
                                len(final_snapshots) - snapshot_count
                            ),
                            "layered_pipeline_waves": waves,
                            "layered_pipeline_structure": (
                                sum_wave_fields(waves)
                                if mode["batching_policy"] == "layered-pipeline"
                                else None
                            ),
                            "identified_long_prefill_waves": long_waves,
                            "long_prefill_wave_identification": identification,
                            "output_mismatch_requests": sum(
                                record["output_mismatch"] is not None
                                for record in records
                            ),
                            "validation_failures": failures,
                        }
                    )
                    mode_result["requests"].extend(records)
                    mode_result["repetitions"].append(repetition_summary)
                    if mode["batching_policy"] == "layered-pipeline":
                        mode_result["layered_pipeline_waves"].extend(waves)
                    lab.write_json(output, result)
                    if failures:
                        raise RuntimeError(
                            f"mode {mode['name']} repetition {repetition} failed: "
                            + "; ".join(failures)
                        )
                    if mode_index == 0:
                        for record in records:
                            references[(repetition, record["request_id"])] = record[
                                "output_text"
                            ]
                    baseline_snapshot = after_snapshot
                    snapshot_count = len(final_snapshots)

                mode_result["summary"] = summarize_mode(
                    mode_result["requests"], mode_result["repetitions"]
                )
                if mode["batching_policy"] == "layered-pipeline":
                    mode_result["layered_pipeline_structure"] = sum_wave_fields(
                        mode_result["layered_pipeline_waves"]
                    )
            except Exception as exc:
                mode_result["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                server.stop()
                mode_result["server_log_tail"] = server.log_tail()
                server.close()
                lab.write_json(output, result)
    finally:
        if owned_model_root is not None:
            shutil.rmtree(owned_model_root)
            result["model_path_removed_after_run"] = True
        lab.write_json(output, result)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
