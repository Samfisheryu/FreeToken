#!/usr/bin/env python3
"""Replay a fixed real repo-agent trace at its natural arrival times.

The same ten SWE-bench repository prompts, BurstGPT arrival offsets, and
BurstGPT response lengths are replayed against fresh legacy and
layered-pipeline servers.  Arrival times are absolute and open-loop: a slow
earlier request never delays submission of a later request.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import time
from typing import Any, Iterable

if __package__:
    from . import bench_lab_agent_policies as lab
    from . import bench_real_conversation_concurrency as real
    from . import bench_scaled_expert_contention as scaled
else:
    import bench_lab_agent_policies as lab
    import bench_real_conversation_concurrency as real
    import bench_scaled_expert_contention as scaled


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = Path(
    "/data1/lmcache_kv/experiments/freetoken_dsv4_repo/"
    "swebench_burstgpt_dsv4_v1.json"
)
MANIFEST_SCHEMA = "freetoken.dsv4_repo_concurrency_manifest.v1"
RESULT_SCHEMA = "freetoken.natural_repo_agent.v1"
CASE_NAME = "repo4k_x10"
REQUEST_COUNT = 10
SOURCE_PROMPT_TOKENS = 4_000
ARRIVAL_POLICY = "open_loop_fixed"
TIME_COMPRESSION = 1
RESPONSE_LENGTH_POLICY = "burstgpt_source_response_tokens"
MODES = ("legacy", "layered-pipeline")
MODEL_PROFILES: dict[str, dict[str, Any]] = {
    "qwen36": {
        "model": Path("/data1/lmcache_kv/models/Qwen3.6-35B-A3B"),
        "expert": Path(
            "/data1/lmcache_kv/nowag_qwen36_experiment/quantized/"
            "qwen36_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
        ),
        "attention_backend": "triton",
        "max_seq_len": 16_384,
        "num_tokens": 150_000,
        "memory_ratio": None,
    },
    "dsv4": {
        "model": Path("/data1/lmcache_kv/models/DeepSeek-V4-Flash-0731"),
        "expert": Path(
            "/data1/lmcache_kv/nowag_4090_experiment/quantized/"
            "dsv4_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
        ),
        "attention_backend": "dsv4_sparse",
        "max_seq_len": 131_072,
        "num_tokens": None,
        "memory_ratio": 0.7,
    },
}
MOE_CACHE_SIZE = 512
MAX_PREFILL_LENGTH = 8_192
MAX_RUNNING_REQUESTS = 16
CUDA_GRAPH_MAX_BS = 8
PREFILL_LAYER_GROUP_SIZE = 1
PREFILL_WAVE_MAX_CHUNKS = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-profile",
        required=True,
        choices=tuple(MODEL_PROFILES),
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--expert", type=Path)
    parser.add_argument(
        "--nowag-plugin-src",
        type=Path,
        help="optional NoWAG plugin source to prepend to the server PYTHONPATH",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18420)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=1800.0)
    parser.add_argument("--ft-executable")
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=MODES,
        default=MODES,
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if len(set(args.modes)) != len(args.modes):
        parser.error("--modes must not contain duplicates")
    if args.server_timeout <= 0 or args.request_timeout <= 0:
        parser.error("timeouts must be positive")
    profile = MODEL_PROFILES[args.model_profile]
    args.model = args.model or profile["model"]
    args.expert = args.expert or profile["expert"]
    return args


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unexpected manifest schema in {path}")
    case = payload.get("case_specs", {}).get(CASE_NAME)
    expected_case = {
        "repo_requests": REQUEST_COUNT,
        "prompt_tokens": SOURCE_PROMPT_TOKENS,
    }
    if case != expected_case:
        raise ValueError(f"manifest {CASE_NAME} geometry is not {expected_case}")
    tasks = payload.get("tasks")
    schedules = payload.get("arrival_schedules")
    if not isinstance(tasks, list) or len(tasks) < REQUEST_COUNT:
        raise ValueError("manifest does not contain ten repository tasks")
    if not isinstance(schedules, list) or len(schedules) < REQUEST_COUNT:
        raise ValueError("manifest does not contain ten arrival schedules")

    previous_arrival = -1.0
    for index in range(REQUEST_COUNT):
        task = tasks[index]
        variant = task.get("prompt_variants", {}).get(str(SOURCE_PROMPT_TOKENS))
        if not isinstance(variant, dict):
            raise ValueError(f"task {index} lacks its 4K prompt variant")
        if variant.get("token_count") != SOURCE_PROMPT_TOKENS:
            raise ValueError(f"task {index} has unexpected source prompt length")
        if not isinstance(variant.get("text"), str) or not variant["text"]:
            raise ValueError(f"task {index} has no prompt text")
        schedule = schedules[index]
        arrival = schedule.get("start_offset_seconds")
        if not isinstance(arrival, (int, float)) or arrival < previous_arrival:
            raise ValueError("arrival offsets must be non-negative and nondecreasing")
        previous_arrival = float(arrival)
        source_requests = schedule.get("source_requests")
        if not isinstance(source_requests, list) or not source_requests:
            raise ValueError(f"schedule {index} has no source request")
        response_tokens = source_requests[0].get("response_tokens")
        if not isinstance(response_tokens, int) or response_tokens < 1:
            raise ValueError(f"schedule {index} has no positive response length")
    return payload


def trace_requests(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    requests: list[dict[str, Any]] = []
    for index in range(REQUEST_COUNT):
        task = manifest["tasks"][index]
        prompt = task["prompt_variants"][str(SOURCE_PROMPT_TOKENS)]
        schedule = manifest["arrival_schedules"][index]
        source_request = schedule["source_requests"][0]
        requests.append(
            {
                "task_index": index,
                "instance_id": task["instance_id"],
                "repo": task["repo"],
                "base_commit": task["base_commit"],
                "prompt_text": prompt["text"],
                "source_prompt_tokens": prompt["token_count"],
                "arrival_offset_seconds": float(schedule["start_offset_seconds"]),
                "target_completion_tokens": source_request["response_tokens"],
                "burstgpt_session_id": schedule["burstgpt_session_id"],
                "burstgpt_source_request": source_request,
            }
        )
    return requests


def public_request_spec(request: dict[str, Any]) -> dict[str, Any]:
    return {
        key: request[key]
        for key in (
            "task_index",
            "instance_id",
            "repo",
            "base_commit",
            "source_prompt_tokens",
            "arrival_offset_seconds",
            "target_completion_tokens",
            "burstgpt_session_id",
            "burstgpt_source_request",
        )
    }


def profile_config(profile_name: str) -> dict[str, Any]:
    profile = MODEL_PROFILES[profile_name]
    return {
        "dtype": "bfloat16",
        "attention_backend": profile["attention_backend"],
        "moe_backend": "offload",
        "moe_cache_size": MOE_CACHE_SIZE,
        "max_prefill_length": MAX_PREFILL_LENGTH,
        "prefill_layer_group_size": PREFILL_LAYER_GROUP_SIZE,
        "prefill_wave_max_chunks": PREFILL_WAVE_MAX_CHUNKS,
        "max_running_requests": MAX_RUNNING_REQUESTS,
        "cuda_graph_max_bs": CUDA_GRAPH_MAX_BS,
        "max_seq_len": profile["max_seq_len"],
        "num_tokens": profile["num_tokens"],
        "memory_ratio": profile["memory_ratio"],
        "cache_type": "radix",
    }


def server_command(
    args: argparse.Namespace,
    executable: str,
    mode: str,
) -> list[str]:
    config = profile_config(args.model_profile)
    command = [
        executable,
        "serve",
        "--model-path",
        str(args.model.resolve()),
        "--served-model-name",
        lab.SERVED_MODEL,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        config["dtype"],
        "--max-running-requests",
        str(config["max_running_requests"]),
        "--max-seq-len-override",
        str(config["max_seq_len"]),
        "--max-prefill-length",
        str(config["max_prefill_length"]),
        "--attention-backend",
        config["attention_backend"],
        "--moe-backend",
        config["moe_backend"],
        "--moe-cache-size",
        str(config["moe_cache_size"]),
        "--cuda-graph-max-bs",
        str(config["cuda_graph_max_bs"]),
        "--cache-type",
        config["cache_type"],
        "--enable-cache-report",
        "--moe-collect-stats",
        "--nowag-expert-path",
        str(args.expert.resolve()),
        "--batching-policy",
        mode,
    ]
    if config["num_tokens"] is not None:
        command.extend(("--num-tokens", str(config["num_tokens"])))
    if config["memory_ratio"] is not None:
        command.extend(("--memory-ratio", str(config["memory_ratio"])))
    if mode == "layered-pipeline":
        command.extend(
            (
                "--prefill-layer-group-size",
                str(config["prefill_layer_group_size"]),
                "--prefill-wave-max-chunks",
                str(config["prefill_wave_max_chunks"]),
            )
        )
    return command


def add_model_prompt_lengths(
    requests: list[dict[str, Any]], model: Path
) -> None:
    tokenizer = lab.load_tokenizer(model)
    try:
        for request in requests:
            request["model_prompt_tokens"] = len(
                tokenizer.encode(request["prompt_text"], add_special_tokens=False)
            )
    finally:
        del tokenizer


def finish_record(record: dict[str, Any], request: dict[str, Any]) -> None:
    failures: list[str] = []
    events = record["nonempty_text_events"]
    usage = record.get("usage")
    completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
    prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None

    if record.get("error"):
        failures.append("HTTP or stream error")
    if not events:
        failures.append("no non-empty SSE text event")
        ttft = None
        tpot = None
        max_gap = None
    else:
        first_event = events[0]["at_seconds"]
        last_event = events[-1]["at_seconds"]
        ttft = first_event - record["submitted_at_seconds"]
        tpot = (
            (last_event - first_event) / (completion_tokens - 1)
            if isinstance(completion_tokens, int) and completion_tokens > 1
            else None
        )
        gaps = [
            later["at_seconds"] - earlier["at_seconds"]
            for earlier, later in zip(events, events[1:])
        ]
        max_gap = max(gaps) if gaps else None
    if not isinstance(usage, dict):
        failures.append("missing usage")
    elif completion_tokens != request["target_completion_tokens"]:
        failures.append("completion usage mismatch")

    record.update(
        {
            "task_index": request["task_index"],
            "instance_id": request["instance_id"],
            "repo": request["repo"],
            "base_commit": request["base_commit"],
            "arrival_offset_seconds": request["arrival_offset_seconds"],
            "submission_delay_seconds": (
                record["submitted_at_seconds"] - request["arrival_offset_seconds"]
            ),
            "source_prompt_tokens": request["source_prompt_tokens"],
            "model_prompt_tokens": request["model_prompt_tokens"],
            "target_completion_tokens": request["target_completion_tokens"],
            "actual_prompt_tokens": prompt_tokens,
            "actual_completion_tokens": completion_tokens,
            "ttft_seconds": ttft,
            "tpot_seconds": tpot,
            "latency_seconds": (
                record["response_complete_at_seconds"]
                - record["submitted_at_seconds"]
            ),
            "max_nonempty_sse_event_gap_seconds": max_gap,
            "output_mismatch": None,
            "measurement_failed": bool(failures),
            "measurement_failures": failures,
        }
    )


def run_request(
    base_url: str,
    request: dict[str, Any],
    origin: float,
    request_timeout: float,
) -> dict[str, Any]:
    real.wait_until(origin + request["arrival_offset_seconds"])
    record = lab.request_completion(
        base_url,
        request["prompt_text"],
        request["model_prompt_tokens"],
        request["target_completion_tokens"],
        8_200_000 + request["task_index"],
        origin,
        request_timeout=request_timeout,
    )
    finish_record(record, request)
    return record


def run_trace(
    base_url: str,
    requests: list[dict[str, Any]],
    request_timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    origin = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(requests)
    ) as executor:
        futures = [
            executor.submit(run_request, base_url, request, origin, request_timeout)
            for request in requests
        ]
        records = [future.result() for future in futures]
    records.sort(key=lambda record: record["task_index"])
    return records, time.perf_counter() - origin


def distribution(values: Iterable[float | None]) -> dict[str, float | None]:
    valid = [value for value in values if value is not None]
    return {
        "p50": real.percentile(valid, 50),
        "p95": real.percentile(valid, 95),
    }


def summarize_records(
    records: list[dict[str, Any]], elapsed_seconds: float
) -> dict[str, Any]:
    valid_usage = [record for record in records if isinstance(record.get("usage"), dict)]
    result: dict[str, Any] = {
        "request_count": len(records),
        "measurement_failed_requests": sum(
            bool(record["measurement_failed"]) for record in records
        ),
        "http_or_stream_error_requests": sum(
            bool(record.get("error")) for record in records
        ),
        "actual_prompt_tokens": sum(
            record["actual_prompt_tokens"] or 0 for record in valid_usage
        ),
        "actual_completion_tokens": sum(
            record["actual_completion_tokens"] or 0 for record in valid_usage
        ),
        "trace_elapsed_seconds": elapsed_seconds,
        "active_span_seconds": (
            max(record["response_complete_at_seconds"] for record in records)
            - min(record["submitted_at_seconds"] for record in records)
        ),
        **real.concurrency_summary(records),
    }
    fields = (
        "ttft_seconds",
        "tpot_seconds",
        "latency_seconds",
        "max_nonempty_sse_event_gap_seconds",
        "submission_delay_seconds",
    )
    for field in fields:
        result[field] = distribution(record[field] for record in records)
    return result


def paired_requests(
    requests: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    modes: list[str],
) -> list[dict[str, Any]]:
    records_by_mode = {
        run["mode"]: {record["task_index"]: record for record in run["requests"]}
        for run in runs
        if run["requests"]
    }
    fields = (
        "submitted_at_seconds",
        "submission_delay_seconds",
        "actual_prompt_tokens",
        "actual_completion_tokens",
        "ttft_seconds",
        "tpot_seconds",
        "latency_seconds",
        "max_nonempty_sse_event_gap_seconds",
        "error",
        "measurement_failed",
        "output_mismatch",
    )
    paired: list[dict[str, Any]] = []
    for request in requests:
        index = request["task_index"]
        mode_records = {
            mode: (
                {
                    field: records_by_mode[mode][index].get(field)
                    for field in fields
                }
                if mode in records_by_mode and index in records_by_mode[mode]
                else None
            )
            for mode in modes
        }
        paired.append(
            {
                **public_request_spec(request),
                "policies": mode_records,
            }
        )
    return paired


def fairness_checks(
    requests: list[dict[str, Any]],
    runs: list[dict[str, Any]],
    modes: list[str],
) -> dict[str, Any]:
    complete_runs = len(runs) == len(modes) and all(
        len(run["requests"]) == len(requests) for run in runs
    )
    records = [record for run in runs for record in run["requests"]]
    no_request_failures = complete_runs and not any(
        record["measurement_failed"] for record in records
    )
    completion_usage_matches_target = complete_runs and all(
        record["actual_completion_tokens"] == record["target_completion_tokens"]
        for record in records
    )
    per_mode = {
        run["mode"]: {record["task_index"]: record for record in run["requests"]}
        for run in runs
    }
    same_actual_prompt_usage = complete_runs and all(
        len(
            {
                per_mode[mode][request["task_index"]]["actual_prompt_tokens"]
                for mode in modes
            }
        )
        == 1
        for request in requests
    )
    return {
        "same_task_and_prompt_by_task_index": True,
        "same_target_completion_tokens_by_task_index": True,
        "all_requested_modes_complete": complete_runs,
        "same_actual_prompt_usage_by_task_index": same_actual_prompt_usage,
        "completion_usage_matches_target": completion_usage_matches_target,
        "zero_http_or_request_failures": no_request_failures,
        "passed": (
            complete_runs
            and same_actual_prompt_usage
            and completion_usage_matches_target
            and no_request_failures
        ),
        "output_text_equality_is_reported_but_not_a_pass_condition": True,
    }


def dry_run_plan(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    executable: str,
) -> dict[str, Any]:
    requests = trace_requests(manifest)
    return {
        "dry_run": True,
        "schema": RESULT_SCHEMA,
        "manifest_path": str(args.manifest.resolve()),
        "model_profile": args.model_profile,
        "model": str(args.model.resolve()),
        "expert": str(args.expert.resolve()),
        "gpu": args.gpu,
        "workload": {
            "case": CASE_NAME,
            "arrival_policy": ARRIVAL_POLICY,
            "time_compression": TIME_COMPRESSION,
            "response_length_policy": RESPONSE_LENGTH_POLICY,
            "fresh_server_per_mode": True,
            "requests": [public_request_spec(request) for request in requests],
        },
        "server_config": profile_config(args.model_profile),
        "commands": [
            {"mode": mode, "command": server_command(args, executable, mode)}
            for mode in args.modes
        ],
    }


def main() -> int:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    executable = lab.find_ft_executable(args.ft_executable)
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, manifest, executable), indent=2))
        return 0

    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.expert.exists():
        raise FileNotFoundError(args.expert)
    if args.nowag_plugin_src is not None and not args.nowag_plugin_src.is_dir():
        raise FileNotFoundError(args.nowag_plugin_src)

    requests = trace_requests(manifest)
    add_model_prompt_lengths(requests, args.model)
    output = args.output or (
        HERE
        / "results"
        / f"natural_repo_agent_{args.model_profile}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result: dict[str, Any] = {
        "schema": RESULT_SCHEMA,
        "created_at_unix_seconds": time.time(),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_note": "companion reproduction input; not copied or hashed",
        "model_profile": args.model_profile,
        "model": str(args.model.resolve()),
        "expert": str(args.expert.resolve()),
        "gpu": args.gpu,
        "workload": {
            "case": CASE_NAME,
            "arrival_policy": ARRIVAL_POLICY,
            "time_compression": TIME_COMPRESSION,
            "response_length_policy": RESPONSE_LENGTH_POLICY,
            "fresh_server_per_mode": True,
            "requests": [public_request_spec(request) for request in requests],
        },
        "server_config": profile_config(args.model_profile),
        "nowag_plugin_src": (
            str(args.nowag_plugin_src.resolve())
            if args.nowag_plugin_src is not None
            else None
        ),
        "runs": [],
        "paired_requests": [],
        "fairness_checks": None,
    }
    references: dict[int, str] = {}
    base_url = f"http://{args.host}:{args.port}"
    for mode in args.modes:
        command = server_command(args, executable, mode)
        run: dict[str, Any] = {
            "mode": mode,
            "server_command": command,
            "requests": [],
            "summary": None,
            "moe_stats_before": None,
            "moe_stats_after": None,
            "moe_stats_delta": None,
            "layered_pipeline_waves": [],
            "error": None,
        }
        result["runs"].append(run)
        server = real.BenchmarkServer(
            command,
            args.gpu,
            base_url,
            args.server_timeout,
            "Natural repo-agent benchmark readiness probe.",
            nowag_plugin_src=args.nowag_plugin_src,
        )
        try:
            server.start()
            snapshots = scaled.wait_for_snapshot_count(
                server, 1, args.server_timeout
            )
            baseline = snapshots[-1]
            log_offset = real.log_offset(server)
            records, elapsed_seconds = run_trace(
                base_url, requests, args.request_timeout
            )
            final_snapshots = real.wait_for_settled_snapshot(
                server, len(snapshots), args.server_timeout
            )
            after = final_snapshots[-1]
            for record in records:
                reference = references.get(record["task_index"])
                record["output_mismatch"] = (
                    None
                    if reference is None
                    else lab.first_difference(reference, record["output_text"])
                )
                if reference is None:
                    references[record["task_index"]] = record["output_text"]
            run.update(
                {
                    "requests": records,
                    "summary": summarize_records(records, elapsed_seconds),
                    "moe_stats_before": baseline,
                    "moe_stats_after": after,
                    "moe_stats_delta": scaled.snapshot_delta(baseline, after),
                    "layered_pipeline_waves": (
                        real.pipeline_waves_since(server, log_offset)
                        if mode == "layered-pipeline"
                        else []
                    ),
                }
            )
            result["paired_requests"] = paired_requests(
                requests, result["runs"], list(args.modes)
            )
            result["fairness_checks"] = fairness_checks(
                requests, result["runs"], list(args.modes)
            )
            lab.write_json(output, result)
            if run["summary"]["measurement_failed_requests"]:
                raise RuntimeError(f"{mode} contains failed requests")
        except Exception as exc:
            run["error"] = f"{type(exc).__name__}: {exc}"
            result["paired_requests"] = paired_requests(
                requests, result["runs"], list(args.modes)
            )
            result["fairness_checks"] = fairness_checks(
                requests, result["runs"], list(args.modes)
            )
            lab.write_json(output, result)
            raise
        finally:
            server.stop()
            run["server_log_tail"] = server.log_tail()
            server.close()
            lab.write_json(output, result)

    result["paired_requests"] = paired_requests(
        requests, result["runs"], list(args.modes)
    )
    result["fairness_checks"] = fairness_checks(
        requests, result["runs"], list(args.modes)
    )
    lab.write_json(output, result)
    if not result["fairness_checks"]["passed"]:
        raise RuntimeError("benchmark fairness contract failed")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
