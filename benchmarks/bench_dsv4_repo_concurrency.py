#!/usr/bin/env python3
"""Compare FreeToken policies on real SWE-bench repository contexts.

The workload combines official SWE-bench BM25 retrieval prompts with BurstGPT
session-start timing.  Three short-prompt drivers begin decoding first; real
repository prompts then arrive while those drivers are active.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import subprocess
import threading
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
DEFAULT_MODEL = Path("/data1/lmcache_kv/models/DeepSeek-V4-Flash-0731")
DEFAULT_EXPERT = Path(
    "/data1/lmcache_kv/nowag_4090_experiment/quantized/"
    "dsv4_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
)
DEFAULT_PARQUET = Path(
    "/data1/lmcache_kv/experiments/freetoken_dsv4_repo/"
    "swebench_bm25_40k_test.parquet"
)
DATASET = "princeton-nlp/SWE-bench_bm25_40K"
DATASET_SPLIT = "test"
MEASUREMENT_OFFSETS = (121, 1207, 1328, 1448, 1569, 1810, 1086, 0, 1931, 241)
WARMUP_REPO_OFFSET = 5
DRIVER_OFFSETS = (965, 2052, 2293)
CASE_SPECS = {
    "repo4k_x10": {"repo_requests": 10, "prompt_tokens": 4_000},
    "repo8k_x5": {"repo_requests": 5, "prompt_tokens": 8_000},
    "repo12k_x4": {"repo_requests": 4, "prompt_tokens": 12_000},
    "repo24k_x2": {"repo_requests": 2, "prompt_tokens": 24_000},
    "repo40k_x1": {"repo_requests": 1, "prompt_tokens": 40_000},
}
DEFAULT_CASES = tuple(CASE_SPECS)
SERVED_MODEL = lab.SERVED_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prepare-manifest",
        type=Path,
        help="materialize fixed SWE-bench prompts and BurstGPT schedules",
    )
    source.add_argument("--manifest", type=Path)
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    parser.add_argument(
        "--nowag-plugin-src",
        type=Path,
        help="NoWAG plugin source to prepend to PYTHONPATH; omitted by default",
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
        choices=("legacy", "layered-pipeline"),
        default=("legacy", "layered-pipeline"),
    )
    parser.add_argument(
        "--cases", nargs="+", choices=tuple(CASE_SPECS), default=DEFAULT_CASES
    )
    parser.add_argument("--driver-count", type=int, default=3)
    parser.add_argument("--driver-decode-tokens", type=int, default=512)
    parser.add_argument("--repo-decode-tokens", type=int, default=1)
    parser.add_argument("--time-compression", type=float, default=120.0)
    parser.add_argument("--moe-cache-size", type=int, default=512)
    parser.add_argument("--memory-ratio", type=float, default=0.8)
    parser.add_argument(
        "--max-prefill-length",
        type=int,
        default=65_536,
        help="maximum prefill tokens per resident-group iteration",
    )
    parser.add_argument("--max-seq-len", type=int, default=131_072)
    parser.add_argument("--max-running-requests", type=int, default=16)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--prefill-layer-group-size", type=int, default=1)
    parser.add_argument("--prefill-wave-max-chunks", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.driver_count < 1 or args.driver_count > len(DRIVER_OFFSETS):
        parser.error(f"--driver-count must be between 1 and {len(DRIVER_OFFSETS)}")
    if args.driver_decode_tokens < 2 or args.repo_decode_tokens < 1:
        parser.error("decode token counts must be positive; driver count must exceed one")
    if args.time_compression <= 0:
        parser.error("--time-compression must be positive")
    if not 0 < args.memory_ratio < 1:
        parser.error("--memory-ratio must be between zero and one")
    if args.max_prefill_length < 1:
        parser.error("--max-prefill-length must be at least 1")
    if args.prefill_layer_group_size < 1:
        parser.error("--prefill-layer-group-size must be at least 1")
    if args.prefill_wave_max_chunks < 1:
        parser.error("--prefill-wave-max-chunks must be at least 1")
    needed = args.driver_count + max(
        CASE_SPECS[name]["repo_requests"] for name in args.cases
    )
    if args.max_running_requests < needed:
        parser.error("--max-running-requests must cover drivers plus repo requests")
    return args


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def prompt_variant(tokenizer: Any, text: str, target: int) -> dict[str, Any]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < target:
        raise ValueError(f"SWE-bench prompt has {len(ids)} tokens, needs {target}")
    selected = ids[:target]
    selected_text = tokenizer.decode(selected, skip_special_tokens=False)
    if tokenizer.encode(selected_text, add_special_tokens=False) != selected:
        raise RuntimeError("DSV4 prompt truncation did not round-trip")
    return {"text": selected_text, "token_count": target}


def prepare_manifest(path: Path, parquet: Path, model: Path) -> None:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("manifest preparation requires pyarrow") from exc
    if not parquet.is_file():
        raise FileNotFoundError(parquet)
    table = pq.read_table(
        parquet,
        columns=["instance_id", "repo", "base_commit", "problem_statement", "text"],
    )
    tokenizer = lab.load_tokenizer(model)

    def row(offset: int) -> dict[str, Any]:
        return {
            key: table[key][offset].as_py()
            for key in ("instance_id", "repo", "base_commit", "problem_statement", "text")
        }

    tasks: list[dict[str, Any]] = []
    for index, offset in enumerate(MEASUREMENT_OFFSETS):
        source = row(offset)
        variants: dict[str, Any] = {}
        for case_name, spec in CASE_SPECS.items():
            if index < spec["repo_requests"]:
                target = spec["prompt_tokens"]
                variants[str(target)] = prompt_variant(tokenizer, source["text"], target)
        tasks.append(
            {
                "row_offset": offset,
                "instance_id": source["instance_id"],
                "repo": source["repo"],
                "base_commit": source["base_commit"],
                "prompt_variants": variants,
            }
        )

    warmup_source = row(WARMUP_REPO_OFFSET)
    warmup_variants = {
        str(spec["prompt_tokens"]): prompt_variant(
            tokenizer, warmup_source["text"], spec["prompt_tokens"]
        )
        for spec in CASE_SPECS.values()
    }
    driver_prompts = []
    for offset in DRIVER_OFFSETS:
        source = row(offset)
        variant = prompt_variant(tokenizer, source["text"], 128)
        driver_prompts.append(
            {
                "row_offset": offset,
                "instance_id": source["instance_id"],
                "repo": source["repo"],
                **variant,
            }
        )
    payload = {
        "schema": "freetoken.dsv4_repo_concurrency_manifest.v1",
        "created_at_unix_seconds": time.time(),
        "sources": {
            "swebench": {
                "dataset": DATASET,
                "split": DATASET_SPLIT,
                "parquet": str(parquet.resolve()),
                "measurement_offsets": list(MEASUREMENT_OFFSETS),
                "warmup_repo_offset": WARMUP_REPO_OFFSET,
                "driver_offsets": list(DRIVER_OFFSETS),
            },
            "burstgpt": {
                "url": real.BURSTGPT_URL,
                "max_think_seconds": real.MAX_BURST_THINK_SECONDS,
            },
        },
        "model": str(model.resolve()),
        "case_specs": CASE_SPECS,
        "tasks": tasks,
        "warmup_repo": {
            "row_offset": WARMUP_REPO_OFFSET,
            "instance_id": warmup_source["instance_id"],
            "repo": warmup_source["repo"],
            "prompt_variants": warmup_variants,
        },
        "driver_prompts": driver_prompts,
        "arrival_schedules": real.burstgpt_schedules(10),
    }
    del tokenizer
    write_json(path, payload)


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "freetoken.dsv4_repo_concurrency_manifest.v1":
        raise ValueError("unexpected DSV4 repo manifest schema")
    if payload.get("case_specs") != CASE_SPECS:
        raise ValueError("manifest case geometry differs from benchmark")
    if len(payload.get("tasks", ())) != len(MEASUREMENT_OFFSETS):
        raise ValueError("manifest measurement task count changed")
    if len(payload.get("arrival_schedules", ())) < 10:
        raise ValueError("manifest lacks BurstGPT schedules")
    return payload


def server_command(args: argparse.Namespace, executable: str, mode: str) -> list[str]:
    command = [
        executable,
        "serve",
        "--model-path",
        str(args.model.resolve()),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        "bfloat16",
        "--max-running-requests",
        str(args.max_running_requests),
        "--max-seq-len-override",
        str(args.max_seq_len),
        "--max-prefill-length",
        str(args.max_prefill_length),
        "--attention-backend",
        "dsv4_sparse",
        "--moe-backend",
        "offload",
        "--moe-cache-size",
        str(args.moe_cache_size),
        "--memory-ratio",
        str(args.memory_ratio),
        "--cuda-graph-max-bs",
        str(args.cuda_graph_max_bs),
        "--cache-type",
        "radix",
        "--enable-cache-report",
        "--moe-collect-stats",
        "--sampling-defaults",
        "none",
        "--nowag-expert-path",
        str(args.expert.resolve()),
        "--batching-policy",
        mode,
    ]
    if mode == "layered-pipeline":
        command.extend(
            (
                "--prefill-layer-group-size",
                str(args.prefill_layer_group_size),
                "--prefill-wave-max-chunks",
                str(args.prefill_wave_max_chunks),
            )
        )
    return command


def relative_now(origin: float) -> float:
    return time.perf_counter() - origin


def finish_record(
    record: dict[str, Any], expected_prompt: int, expected_decode: int
) -> None:
    failures: list[str] = []
    usage = record.get("usage")
    events = record["nonempty_text_events"]
    if record.get("error"):
        failures.append("HTTP or stream error")
    if not events:
        failures.append("no non-empty SSE text event")
        record["ttft_seconds"] = None
        record["tpot_seconds"] = None
        record["max_sse_gap_seconds"] = None
    else:
        record["ttft_seconds"] = events[0]["at_seconds"] - record["submitted_at_seconds"]
        record["tpot_seconds"] = (
            (record["response_complete_at_seconds"] - events[0]["at_seconds"])
            / (expected_decode - 1)
            if expected_decode > 1
            else None
        )
        gaps = [
            later["at_seconds"] - earlier["at_seconds"]
            for earlier, later in zip(events, events[1:])
        ]
        record["max_sse_gap_seconds"] = max(gaps) if gaps else None
    if not isinstance(usage, dict):
        failures.append("missing usage")
    else:
        if usage.get("prompt_tokens") != expected_prompt:
            failures.append("prompt usage mismatch")
        if usage.get("completion_tokens") != expected_decode:
            failures.append("completion usage mismatch")
    record["latency_seconds"] = (
        record["response_complete_at_seconds"] - record["submitted_at_seconds"]
    )
    record["measurement_failed"] = bool(failures)
    record["measurement_failures"] = failures


def run_prompt(
    base_url: str,
    prompt: dict[str, Any],
    decode_tokens: int,
    seed: int,
    origin: float,
    role: str,
    request_id: str,
    first_text_event: threading.Event | None = None,
    wait_until_seconds: float | None = None,
    request_timeout: float = 1800.0,
) -> dict[str, Any]:
    if wait_until_seconds is not None:
        real.wait_until(wait_until_seconds)
    record = lab.request_completion(
        base_url,
        prompt["text"],
        prompt["token_count"],
        decode_tokens,
        seed,
        origin,
        first_text_event,
        request_timeout=request_timeout,
    )
    finish_record(record, prompt["token_count"], decode_tokens)
    record.update({"role": role, "request_id": request_id})
    return record


def run_path_warmup(
    base_url: str,
    manifest: dict[str, Any],
    case: dict[str, int],
    request_timeout: float,
) -> list[dict[str, Any]]:
    origin = time.perf_counter()
    event = threading.Event()
    driver = manifest["driver_prompts"][-1]
    repo = manifest["warmup_repo"]["prompt_variants"][str(case["prompt_tokens"])]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        driver_future = executor.submit(
            run_prompt,
            base_url,
            driver,
            256,
            7_000_001,
            origin,
            "warmup_driver",
            "warmup_driver",
            event,
            None,
            request_timeout,
        )
        if not event.wait(timeout=request_timeout):
            raise TimeoutError("warmup driver did not stream its first token")
        repo_future = executor.submit(
            run_prompt,
            base_url,
            repo,
            1,
            7_000_002,
            origin,
            "warmup_repo",
            "warmup_repo",
            None,
            None,
            request_timeout,
        )
        records = [driver_future.result(), repo_future.result()]
    if any(record["measurement_failed"] for record in records):
        raise RuntimeError("path warmup failed")
    return records


def run_measurement(
    base_url: str,
    manifest: dict[str, Any],
    case: dict[str, int],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    origin = time.perf_counter()
    events = [threading.Event() for _ in range(args.driver_count)]
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.driver_count + case["repo_requests"]
    ) as executor:
        driver_futures = [
            executor.submit(
                run_prompt,
                base_url,
                manifest["driver_prompts"][index],
                args.driver_decode_tokens,
                8_000_000 + index,
                origin,
                "driver",
                f"driver_{index}",
                events[index],
                None,
                args.request_timeout,
            )
            for index in range(args.driver_count)
        ]
        for event in events:
            if not event.wait(timeout=args.request_timeout):
                raise TimeoutError("measurement driver did not stream its first token")
        release_at = time.perf_counter()
        repo_futures = []
        for index in range(case["repo_requests"]):
            task = manifest["tasks"][index]
            prompt = task["prompt_variants"][str(case["prompt_tokens"])]
            schedule = manifest["arrival_schedules"][index]
            target = release_at + schedule["start_offset_seconds"] / args.time_compression
            repo_futures.append(
                executor.submit(
                    run_prompt,
                    base_url,
                    prompt,
                    args.repo_decode_tokens,
                    8_100_000 + index,
                    origin,
                    "repo",
                    task["instance_id"],
                    None,
                    target,
                    args.request_timeout,
                )
            )
        records.extend(future.result() for future in driver_futures)
        records.extend(future.result() for future in repo_futures)
    records.sort(key=lambda item: item["submitted_at_seconds"])
    repo_records = [record for record in records if record["role"] == "repo"]
    driver_records = [record for record in records if record["role"] == "driver"]
    first_repo = min(record["submitted_at_seconds"] for record in repo_records)
    last_repo = max(record["response_complete_at_seconds"] for record in repo_records)
    active_at_first_repo = sum(
        record["response_complete_at_seconds"] > first_repo for record in driver_records
    )
    active_at_each_repo = [
        sum(
            driver["response_complete_at_seconds"] > repo["submitted_at_seconds"]
            for driver in driver_records
        )
        for repo in repo_records
    ]
    window_gaps = [
        later["at_seconds"] - earlier["at_seconds"]
        for driver in driver_records
        for earlier, later in zip(
            driver["nonempty_text_events"], driver["nonempty_text_events"][1:]
        )
        if earlier["at_seconds"] < last_repo and later["at_seconds"] > first_repo
    ]
    return records, {
        "repo_release_at_seconds": release_at - origin,
        "repo_window_start_seconds": first_repo,
        "repo_window_end_seconds": last_repo,
        "repo_window_seconds": last_repo - first_repo,
        "drivers_active_at_first_repo": active_at_first_repo,
        "drivers_active_at_each_repo_submission": active_at_each_repo,
        "driver_gap_seconds_during_repo_window": window_gaps,
    }


def distribution(values: Iterable[float | None]) -> dict[str, float | None]:
    valid = [value for value in values if value is not None]
    return {"p50": real.percentile(valid, 50), "p95": real.percentile(valid, 95)}


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    good = [record for record in records if not record["measurement_failed"]]
    result: dict[str, Any] = {
        "request_count": len(records),
        "measurement_failed_requests": len(records) - len(good),
        "prompt_tokens": sum(record["usage"]["prompt_tokens"] for record in good),
        "decode_tokens": sum(record["usage"]["completion_tokens"] for record in good),
        "makespan_seconds": (
            max(record["response_complete_at_seconds"] for record in records)
            - min(record["submitted_at_seconds"] for record in records)
        ),
        **real.concurrency_summary(records),
    }
    for role in ("driver", "repo"):
        selected = [record for record in good if record["role"] == role]
        result[role] = {
            "request_count": len(selected),
            "ttft_seconds": distribution(record["ttft_seconds"] for record in selected),
            "tpot_seconds": distribution(record["tpot_seconds"] for record in selected),
            "latency_seconds": distribution(record["latency_seconds"] for record in selected),
            "max_sse_gap_seconds": distribution(
                record["max_sse_gap_seconds"] for record in selected
            ),
        }
    return result


def dry_run(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    executable = lab.find_ft_executable(args.ft_executable)
    return {
        "dry_run": True,
        "schema": "freetoken.dsv4_repo_concurrency.v1",
        "gpu": args.gpu,
        "manifest_path": str(args.manifest),
        "manifest_note": "companion reproduction input; not copied or hashed",
        "cases": {name: CASE_SPECS[name] for name in args.cases},
        "commands": [server_command(args, executable, mode) for mode in args.modes],
    }


def main() -> int:
    args = parse_args()
    if args.prepare_manifest is not None:
        prepare_manifest(args.prepare_manifest, args.parquet, args.model)
        print(args.prepare_manifest)
        return 0

    assert args.manifest is not None
    manifest = load_manifest(args.manifest)
    if args.dry_run:
        print(json.dumps(dry_run(args, manifest), indent=2))
        return 0
    for path in (args.model, args.expert):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.nowag_plugin_src is not None and not args.nowag_plugin_src.exists():
        raise FileNotFoundError(args.nowag_plugin_src)
    executable = lab.find_ft_executable(args.ft_executable)
    output = args.output or (
        HERE / "results" / f"dsv4_repo_concurrency_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result: dict[str, Any] = {
        "schema": "freetoken.dsv4_repo_concurrency.v1",
        "created_at_unix_seconds": time.time(),
        "gpu": args.gpu,
        "manifest_path": str(args.manifest.resolve()),
        "manifest_note": "companion reproduction input; not copied or hashed",
        "model": str(args.model.resolve()),
        "expert": str(args.expert.resolve()),
        "config": {
            "cases": list(args.cases),
            "driver_count": args.driver_count,
            "driver_decode_tokens": args.driver_decode_tokens,
            "repo_decode_tokens": args.repo_decode_tokens,
            "time_compression": args.time_compression,
            "moe_cache_size": args.moe_cache_size,
            "memory_ratio": args.memory_ratio,
            "max_prefill_length": args.max_prefill_length,
            "max_seq_len": args.max_seq_len,
            "max_running_requests": args.max_running_requests,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "nowag_plugin_src": (
                str(args.nowag_plugin_src.resolve())
                if args.nowag_plugin_src is not None
                else None
            ),
        },
        "runs": [],
    }
    references: dict[tuple[str, str, str], str] = {}
    for case_name in args.cases:
        case_spec = CASE_SPECS[case_name]
        for mode in args.modes:
            command = server_command(args, executable, mode)
            run: dict[str, Any] = {
                "case": case_name,
                "case_spec": case_spec,
                "mode": mode.replace("-", "_"),
                "server_command": command,
                "warmup": None,
                "requests": [],
                "summary": None,
                "repo_window": None,
                "moe_stats_before": None,
                "moe_stats_after": None,
                "moe_stats_delta": None,
                "layered_pipeline_waves": [],
                "error": None,
            }
            result["runs"].append(run)
            base_url = f"http://{args.host}:{args.port}"
            server = real.BenchmarkServer(
                command,
                args.gpu,
                base_url,
                args.server_timeout,
                "DSV4 repository benchmark readiness probe.",
                nowag_plugin_src=args.nowag_plugin_src,
            )
            try:
                server.start()
                snapshots = scaled.wait_for_snapshot_count(server, 1, args.server_timeout)
                run["warmup"] = run_path_warmup(
                    base_url, manifest, case_spec, args.request_timeout
                )
                snapshots = real.wait_for_settled_snapshot(
                    server, len(snapshots), args.server_timeout
                )
                baseline = snapshots[-1]
                offset = real.log_offset(server)
                records, repo_window = run_measurement(
                    base_url, manifest, case_spec, args
                )
                final_snapshots = real.wait_for_settled_snapshot(
                    server, len(snapshots), args.server_timeout
                )
                after = final_snapshots[-1]
                for record in records:
                    key = (case_name, record["role"], record["request_id"])
                    reference = references.get(key)
                    record["output_mismatch"] = (
                        None
                        if reference is None
                        else lab.first_difference(reference, record["output_text"])
                    )
                    if reference is None:
                        references[key] = record["output_text"]
                run.update(
                    {
                        "requests": records,
                        "summary": summarize_records(records),
                        "repo_window": repo_window,
                        "moe_stats_before": baseline,
                        "moe_stats_after": after,
                        "moe_stats_delta": scaled.snapshot_delta(baseline, after),
                        "layered_pipeline_waves": (
                            real.pipeline_waves_since(server, offset)
                            if mode == "layered-pipeline"
                            else []
                        ),
                    }
                )
                if run["summary"]["measurement_failed_requests"]:
                    raise RuntimeError("measurement contains failed requests")
                if repo_window["drivers_active_at_first_repo"] != args.driver_count:
                    raise RuntimeError("not all drivers remained active at first repo arrival")
                if min(repo_window["drivers_active_at_each_repo_submission"]) < 1:
                    raise RuntimeError("a repo request arrived after every driver completed")
                write_json(output, result)
            except Exception as exc:
                run["error"] = f"{type(exc).__name__}: {exc}"
                write_json(output, result)
                raise
            finally:
                server.stop()
                run["server_log_tail"] = server.log_tail()
                server.close()
                write_json(output, result)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
