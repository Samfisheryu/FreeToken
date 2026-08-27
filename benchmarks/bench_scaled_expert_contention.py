#!/usr/bin/env python3
"""Public expert-transfer contention benchmark for a scaled Qwen3-MoE model.

One long decode request starts first.  As soon as its first SSE text event reaches
the client, four independent long-prefill requests are submitted concurrently.  The
shape keeps a decode stream runnable while forcing the offload policies to schedule a
large, identical prompt workload through an expert cache smaller than the model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

if __package__:
    from . import bench_lab_agent_policies as lab
else:
    import bench_lab_agent_policies as lab


HERE = Path(__file__).resolve().parent
MODEL_CONTRACT = {
    "architecture": "Qwen3MoeForCausalLM",
    "layers": 8,
    "hidden_size": 512,
    "dense_intermediate_size": 1024,
    "moe_intermediate_size": 4096,
    "experts_per_layer": 8,
    "experts_per_token": 2,
    "dtype": "float16",
    "expert_row_bytes": 12 * 2**20,
    "total_expert_bytes": 8 * 8 * 12 * 2**20,
}
COUNTER_FIELDS = (
    "decode_active_rows",
    "decode_missing_rows",
    "decode_layer_calls",
    "decode_fetched_rows",
    "prefill_hit_rows",
    "prefill_rows",
    "prefill_layer_prepares",
    "prefill_h2d_bytes_total",
)
PIPELINE_MODE_RE = re.compile(
    r"layered(?:-|_)pipeline(?:-|_)g(\d+)(?:-|_)cpi(\d+)"
    r"(?:(?:-|_)wave(\d+))?"
)
LAYERED_PREFILL_MODE_RE = re.compile(
    r"layered(?:-|_)prefill(?:-|_)g(\d+)(?:(?:-|_)wave(\d+))?"
)
JOINT_MODE_RE = re.compile(r"joint(?:-|_)g(\d+)(?:-|_)wave(\d+)")
LAYERED_MODE_RE = re.compile(r"layered(?:-|_)g(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "legacy",
            "mixed",
            "layeredG2",
            "jointG2-wave2",
            "layered-pipeline-cpi3",
        ],
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18180)
    parser.add_argument("--server-timeout", type=float, default=600.0)
    parser.add_argument("--ft-executable")
    parser.add_argument("--driver-prompt-tokens", type=int, default=128)
    parser.add_argument("--driver-decode-tokens", type=int, default=512)
    parser.add_argument("--prefill-requests", type=int, default=4)
    parser.add_argument("--prefill-tokens", type=int, default=2048)
    parser.add_argument("--prefill-decode-tokens", type=int, default=1)
    parser.add_argument(
        "--burst-trigger",
        choices=("first-sse", "immediate"),
        default="first-sse",
        help="Release prefill requests after the driver's first text event or immediately.",
    )
    parser.add_argument("--prefill-submit-stagger-ms", type=float, default=0.0)
    parser.add_argument("--max-prefill-length", type=int, default=128)
    parser.add_argument("--moe-cache-size", type=int, default=24)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    args = parser.parse_args()
    for name in (
        "repetitions",
        "driver_prompt_tokens",
        "driver_decode_tokens",
        "prefill_requests",
        "prefill_tokens",
        "prefill_decode_tokens",
        "max_prefill_length",
        "moe_cache_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    if args.cuda_graph_max_bs < 0:
        parser.error("--cuda-graph-max-bs must be non-negative")
    if args.prefill_submit_stagger_ms < 0:
        parser.error("--prefill-submit-stagger-ms must be non-negative")
    max_sequence = max(
        args.driver_prompt_tokens + args.driver_decode_tokens,
        args.prefill_tokens + args.prefill_decode_tokens,
    )
    if max_sequence > 4096:
        parser.error("the fixed scaled model supports at most 4096 total tokens per request")
    return args


def create_scaled_qwen3_moe(destination: Path) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    except ImportError as exc:
        raise RuntimeError("scaled model creation requires torch and transformers") from exc

    torch.manual_seed(20260827)
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            lab.AUTO_TOKENIZER, local_files_only=True
        )
    except OSError:
        print(
            f"Tokenizer {lab.AUTO_TOKENIZER} is not cached; downloading it explicitly.",
            file=sys.stderr,
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(lab.AUTO_TOKENIZER)
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=MODEL_CONTRACT["hidden_size"],
        intermediate_size=MODEL_CONTRACT["dense_intermediate_size"],
        num_hidden_layers=MODEL_CONTRACT["layers"],
        num_attention_heads=8,
        num_key_value_heads=2,
        head_dim=64,
        max_position_embeddings=4096,
        decoder_sparse_step=1,
        moe_intermediate_size=MODEL_CONTRACT["moe_intermediate_size"],
        num_experts_per_tok=MODEL_CONTRACT["experts_per_token"],
        num_experts=MODEL_CONTRACT["experts_per_layer"],
        norm_topk_prob=True,
        attention_dropout=0.0,
        attention_bias=False,
        tie_word_embeddings=True,
        torch_dtype="float16",
    )
    model = Qwen3MoeForCausalLM(config).half()
    destination.mkdir(parents=True, exist_ok=False)
    model.save_pretrained(destination, safe_serialization=True)
    tokenizer.save_pretrained(destination)
    del model


def build_server_workload(args: argparse.Namespace) -> dict[str, Any]:
    workload = lab.load_workload()
    workload["public_server_config"].update(
        {
            "max_running_requests": max(8, args.prefill_requests + 1),
            "max_seq_len_override": 4096,
            "max_prefill_length": args.max_prefill_length,
            "dtype": "float16",
            "attention_backend": "triton",
            "moe_cache_size": args.moe_cache_size,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "cache_type": "radix",
        }
    )
    return workload


def resolve_modes(raw_modes: Iterable[str], workload: dict[str, Any]) -> list[dict[str, Any]]:
    """Resolve standard aliases plus explicit scaled-workload policy geometry."""
    modes: list[dict[str, Any]] = []
    tokens = [piece for item in raw_modes for piece in item.split(",") if piece]
    for token in tokens:
        if match := LAYERED_PREFILL_MODE_RE.fullmatch(token):
            group_size = int(match.group(1))
            wave_chunks = int(match.group(2)) if match.group(2) is not None else 1
            if group_size < 1 or wave_chunks < 1:
                raise ValueError(
                    f"layered-prefill G/wave values must be positive: {token!r}"
                )
            modes.append(
                {
                    "name": f"layered_prefill_g{group_size}_wave{wave_chunks}",
                    "batching_policy": "layered-prefill",
                    "prefill_layer_group_size": group_size,
                    "prefill_wave_max_chunks": wave_chunks,
                    "primary": False,
                }
            )
        elif match := PIPELINE_MODE_RE.fullmatch(token):
            group_size = int(match.group(1))
            chunks_per_iteration = int(match.group(2))
            wave_chunks = int(match.group(3)) if match.group(3) is not None else None
            if group_size < 1 or chunks_per_iteration < 1:
                raise ValueError(f"pipeline G/CPI values must be positive: {token!r}")
            mode = {
                "name": f"layered_pipeline_g{group_size}_cpi{chunks_per_iteration}",
                "batching_policy": "layered-pipeline",
                "prefill_layer_group_size": group_size,
                "layered_pipeline_chunks_per_iteration": chunks_per_iteration,
                "primary": False,
            }
            if wave_chunks is not None:
                if wave_chunks < 1:
                    raise ValueError(f"pipeline wave value must be positive: {token!r}")
                mode["name"] += f"_wave{wave_chunks}"
                mode["prefill_wave_max_chunks"] = wave_chunks
            modes.append(mode)
        elif match := JOINT_MODE_RE.fullmatch(token):
            group_size, wave_chunks = (int(value) for value in match.groups())
            if group_size < 1 or wave_chunks < 1:
                raise ValueError(f"joint G/wave values must be positive: {token!r}")
            modes.append(
                {
                    "name": f"joint_g{group_size}_wave{wave_chunks}",
                    "batching_policy": "joint",
                    "prefill_layer_group_size": group_size,
                    "prefill_wave_max_chunks": wave_chunks,
                    "primary": False,
                }
            )
        elif match := LAYERED_MODE_RE.fullmatch(token):
            group_size = int(match.group(1))
            if group_size < 1:
                raise ValueError(f"layered group size must be positive: {token!r}")
            modes.append(
                {
                    "name": f"layered_g{group_size}_serial",
                    "batching_policy": "layered",
                    "prefill_layer_group_size": group_size,
                    "prefill_execution": "serial",
                    "primary": False,
                }
            )
        else:
            modes.extend(lab.resolve_modes([token], workload))
    unique: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for mode in modes:
        identity = (
            mode["batching_policy"],
            mode.get("prefill_layer_group_size"),
            mode.get("prefill_wave_max_chunks"),
            mode.get("layered_pipeline_chunks_per_iteration"),
        )
        if identity not in seen:
            seen.add(identity)
            unique.append(mode)
    return unique


def materialize_requests(
    tokenizer,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], int, str]:
    request_count = 1 + args.prefill_requests
    candidates = lab.first_token_candidates(
        tokenizer, args.repetitions * request_count + 1
    )
    continuation = lab.continuation_token_pieces(tokenizer)
    repetitions: list[dict[str, Any]] = []
    for repetition in range(args.repetitions):
        offset = repetition * request_count
        driver_text, driver_ids = lab.materialize_segment_text(
            tokenizer,
            args.driver_prompt_tokens,
            20260827 + repetition * 10000,
            f"scaled-driver-r{repetition}",
            continuation,
            first_token=candidates[offset],
        )
        prefill = []
        for index in range(args.prefill_requests):
            text, token_ids = lab.materialize_segment_text(
                tokenizer,
                args.prefill_tokens,
                20260827 + repetition * 10000 + index + 1,
                f"scaled-prefill-{index}-r{repetition}",
                continuation,
                first_token=candidates[offset + index + 1],
            )
            prefill.append({"text": text, "token_count": len(token_ids)})
        repetitions.append(
            {
                "driver": {"text": driver_text, "token_count": len(driver_ids)},
                "prefill": prefill,
            }
        )
    readiness_id, readiness_text = candidates[-1]
    return repetitions, readiness_id, readiness_text


def wait_for_snapshot_count(
    server: lab.PublicServer,
    minimum: int,
    timeout: float,
) -> list[dict[str, int]]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snapshots = server.moe_cache_stats_snapshots()
        if len(snapshots) >= minimum:
            return snapshots
        if server.process is not None and server.process.poll() is not None:
            raise RuntimeError(
                f"server exited before idle stats snapshot: {server.log_tail()}"
            )
        time.sleep(0.05)
    raise TimeoutError(f"no MoE idle stats snapshot within {timeout}s")


def snapshot_delta(
    before: dict[str, int], after: dict[str, int]
) -> dict[str, int]:
    if before["expert_row_bytes"] != after["expert_row_bytes"]:
        raise RuntimeError("expert row size changed within one server lifetime")
    delta = {name: after[name] - before[name] for name in COUNTER_FIELDS}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError(f"MoE cumulative counters moved backwards: {delta}")
    row_bytes = after["expert_row_bytes"]
    delta["expert_row_bytes"] = row_bytes
    delta["decode_h2d_bytes"] = delta["decode_missing_rows"] * row_bytes
    delta["total_expert_h2d_bytes"] = (
        delta["decode_h2d_bytes"] + delta["prefill_h2d_bytes_total"]
    )
    return delta


def run_one_repetition(
    *,
    base_url: str,
    prompts: dict[str, Any],
    args: argparse.Namespace,
    repetition: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    origin = time.perf_counter()
    first_text = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.prefill_requests + 1
    ) as executor:
        driver_prompt = prompts["driver"]
        driver = executor.submit(
            lab.request_completion,
            base_url,
            driver_prompt["text"],
            driver_prompt["token_count"],
            args.driver_decode_tokens,
            30260827 + repetition * 100,
            origin,
            first_text_event=first_text,
        )
        if args.burst_trigger == "first-sse":
            deadline = time.monotonic() + args.server_timeout
            while not first_text.wait(timeout=0.05):
                if driver.done():
                    result = driver.result()
                    raise RuntimeError(
                        "driver completed before a non-empty SSE event: "
                        f"{result.get('error') or result.get('usage')}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError("driver did not emit its first SSE event")
        burst_released_at = lab.relative_now(origin)
        prefill_futures = []
        for index, prompt in enumerate(prompts["prefill"]):
            prefill_futures.append(
                executor.submit(
                    lab.request_completion,
                    base_url,
                    prompt["text"],
                    prompt["token_count"],
                    args.prefill_decode_tokens,
                    30260828 + repetition * 100 + index,
                    origin,
                )
            )
            if args.prefill_submit_stagger_ms:
                time.sleep(args.prefill_submit_stagger_ms / 1000.0)
        records = [driver.result()] + [future.result() for future in prefill_futures]

    for index, record in enumerate(records):
        role = "driver" if index == 0 else "prefill"
        expected_prompt = (
            args.driver_prompt_tokens if role == "driver" else args.prefill_tokens
        )
        expected_decode = (
            args.driver_decode_tokens if role == "driver" else args.prefill_decode_tokens
        )
        failures = []
        usage = record.get("usage")
        events = record["nonempty_text_events"]
        if record.get("error"):
            failures.append("HTTP or stream error")
        if not events:
            failures.append("no non-empty SSE text event")
            record["ttft_seconds"] = None
            record["tpot_seconds"] = None
        else:
            record["ttft_seconds"] = (
                events[0]["at_seconds"] - record["submitted_at_seconds"]
            )
            record["tpot_seconds"] = (
                (events[-1]["at_seconds"] - events[0]["at_seconds"])
                / (expected_decode - 1)
                if expected_decode > 1
                else None
            )
        if not isinstance(usage, dict):
            failures.append("missing usage")
        else:
            if usage.get("prompt_tokens") != expected_prompt:
                failures.append(
                    f"prompt_tokens {usage.get('prompt_tokens')!r} != {expected_prompt}"
                )
            if usage.get("completion_tokens") != expected_decode:
                failures.append(
                    f"completion_tokens {usage.get('completion_tokens')!r} != {expected_decode}"
                )
            details = usage.get("prompt_tokens_details")
            cached = details.get("cached_tokens") if isinstance(details, dict) else 0
            if cached not in (None, 0):
                failures.append(f"unexpected cached_tokens {cached}")
        record.update(
            {
                "repetition": repetition,
                "request_id": "driver" if index == 0 else f"prefill_{index - 1}",
                "role": role,
                "measurement_failed": bool(failures),
                "measurement_failures": failures,
            }
        )

    makespan = max(item["response_complete_at_seconds"] for item in records)
    total_prompt = args.driver_prompt_tokens + args.prefill_requests * args.prefill_tokens
    total_decode = args.driver_decode_tokens + args.prefill_requests * args.prefill_decode_tokens
    repetition_summary = {
        "burst_released_at_seconds": burst_released_at,
        "makespan_seconds": makespan,
        "request_count": len(records),
        "prompt_tokens": total_prompt,
        "decode_tokens": total_decode,
        "prompt_throughput_tokens_per_second": total_prompt / makespan,
        "decode_throughput_tokens_per_second": total_decode / makespan,
        "measurement_failed_requests": sum(item["measurement_failed"] for item in records),
    }
    return records, repetition_summary


def distribution(values: Iterable[float]) -> dict[str, float | None]:
    return lab.distribution(values, [50, 95])


def summarize_mode(
    records: list[dict[str, Any]], repetitions: list[dict[str, Any]]
) -> dict[str, Any]:
    driver = [record for record in records if record["role"] == "driver"]
    prefill = [record for record in records if record["role"] == "prefill"]
    return {
        "driver_ttft_seconds": distribution(
            record["ttft_seconds"] for record in driver if record["ttft_seconds"] is not None
        ),
        "driver_tpot_seconds": distribution(
            record["tpot_seconds"] for record in driver if record["tpot_seconds"] is not None
        ),
        "prefill_ttft_seconds": distribution(
            record["ttft_seconds"] for record in prefill if record["ttft_seconds"] is not None
        ),
        "makespan_seconds": distribution(item["makespan_seconds"] for item in repetitions),
        "prompt_throughput_tokens_per_second": distribution(
            item["prompt_throughput_tokens_per_second"] for item in repetitions
        ),
        "decode_throughput_tokens_per_second": distribution(
            item["decode_throughput_tokens_per_second"] for item in repetitions
        ),
        "expert_h2d_bytes": distribution(
            item["moe_stats_delta"]["total_expert_h2d_bytes"] for item in repetitions
        ),
        "request_count": len(records),
        "measurement_failed_requests": sum(
            record["measurement_failed"] for record in records
        ),
        "output_mismatch_requests": sum(
            record.get("output_mismatch") is not None for record in records
        ),
    }


def main() -> int:
    args = parse_args()
    workload = build_server_workload(args)
    modes = resolve_modes(args.modes, workload)
    ft_executable = lab.find_ft_executable(args.ft_executable)
    model_label: str | Path = args.model or "<auto-generated-scaled-qwen3-moe>"
    if args.dry_run:
        print(
            json.dumps(
                {
                    "model_contract": MODEL_CONTRACT,
                    "workload": vars(args),
                    "commands": [
                        lab.server_command(
                            ft_executable,
                            model_label,
                            mode,
                            args.host,
                            args.port,
                            workload,
                        )
                        + ["--moe-collect-stats"]
                        for mode in modes
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    owned_model_root: Path | None = None
    if args.model is None:
        owned_model_root = Path(tempfile.mkdtemp(prefix="freetoken-scaled-moe-"))
        model_path = owned_model_root / "model"
        create_scaled_qwen3_moe(model_path)
    else:
        model_path = args.model.resolve()
    tokenizer = lab.load_tokenizer(model_path)
    prompts, readiness_id, readiness_text = materialize_requests(tokenizer, args)
    del tokenizer

    output = args.output or (
        HERE / "results" / f"scaled_expert_contention_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result: dict[str, Any] = {
        "schema": "freetoken.scaled_expert_contention.v1",
        "created_at_unix_seconds": time.time(),
        "model_path": str(model_path),
        "auto_generated_model": args.model is None,
        "model_contract": MODEL_CONTRACT,
        "workload_contract": {
            "driver_prompt_tokens": args.driver_prompt_tokens,
            "driver_decode_tokens": args.driver_decode_tokens,
            "prefill_requests": args.prefill_requests,
            "prefill_tokens_each": args.prefill_tokens,
            "prefill_decode_tokens_each": args.prefill_decode_tokens,
            "burst_trigger": args.burst_trigger,
            "prefill_submit_stagger_ms": args.prefill_submit_stagger_ms,
            "max_prefill_length": args.max_prefill_length,
            "moe_cache_size": args.moe_cache_size,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
        },
        "reference_mode": modes[0]["name"],
        "modes": [],
    }
    references: dict[tuple[int, str], str] = {}
    base_url = f"http://{args.host}:{args.port}"
    try:
        for mode_index, mode in enumerate(modes):
            command = lab.server_command(
                ft_executable, model_path, mode, args.host, args.port, workload
            ) + ["--moe-collect-stats"]
            mode_result: dict[str, Any] = {
                "name": mode["name"],
                "server_command": command,
                "repetitions": [],
                "requests": [],
                "joint_waves": [],
                "layered_pipeline_waves": [],
                "layered_prefill_waves": [],
                "server_log_tail": None,
                "error": None,
                "readiness_prompt_token_id": readiness_id,
            }
            result["modes"].append(mode_result)
            server = lab.PublicServer(
                command, args.gpu, base_url, args.server_timeout, readiness_text
            )
            try:
                server.start()
                baseline_snapshots = wait_for_snapshot_count(
                    server, 1, args.server_timeout
                )
                server.mark_measurement_start()
                for repetition in range(args.repetitions):
                    before_count = len(baseline_snapshots)
                    before = baseline_snapshots[-1]
                    records, repetition_summary = run_one_repetition(
                        base_url=base_url,
                        prompts=prompts[repetition],
                        args=args,
                        repetition=repetition,
                    )
                    final_snapshots = wait_for_snapshot_count(
                        server, before_count + 1, args.server_timeout
                    )
                    after = final_snapshots[-1]
                    repetition_summary["moe_stats_before"] = before
                    repetition_summary["moe_stats_after"] = after
                    repetition_summary["moe_stats_delta"] = snapshot_delta(before, after)
                    baseline_snapshots = final_snapshots

                    for record in records:
                        key = (repetition, record["request_id"])
                        reference = references.get(key)
                        record["output_mismatch"] = (
                            lab.first_difference(reference, record["output_text"])
                            if reference is not None
                            else None
                        )
                        if mode_index == 0:
                            references[key] = record["output_text"]
                    repetition_summary["output_mismatch_requests"] = sum(
                        record["output_mismatch"] is not None for record in records
                    )
                    mode_result["requests"].extend(records)
                    mode_result["repetitions"].append(repetition_summary)

                mode_result["summary"] = summarize_mode(
                    mode_result["requests"], mode_result["repetitions"]
                )
                if mode_result["summary"]["measurement_failed_requests"]:
                    lab.write_json(output, result)
                    raise RuntimeError(
                        f"mode {mode['name']} has measurement-failed requests"
                    )
            except Exception as exc:
                mode_result["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                server.stop()
                if mode["batching_policy"] == "layered-pipeline":
                    mode_result["layered_pipeline_waves"] = (
                        server.layered_pipeline_waves()
                    )
                    waves = mode_result["layered_pipeline_waves"]
                    mode_result["layered_pipeline_structure"] = {
                        name: sum(wave[name] for wave in waves)
                        for name in (
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
                        )
                    }
                elif mode["batching_policy"] == "layered-prefill":
                    mode_result["layered_prefill_waves"] = (
                        server.layered_prefill_waves()
                    )
                    waves = mode_result["layered_prefill_waves"]
                    mode_result["layered_prefill_structure"] = {
                        name: sum(wave[name] for wave in waves)
                        for name in (
                            "reqs",
                            "groups",
                            "group_forwards",
                            "iterations",
                            "decode_iterations",
                            "prefill_layer_prepares",
                        )
                    }
                elif mode["batching_policy"] == "joint":
                    mode_result["joint_waves"] = server.joint_waves()
                    waves = mode_result["joint_waves"]
                    mode_result["joint_structure"] = {
                        name: sum(wave[name] for wave in waves)
                        for name in (
                            "chunks",
                            "wave_reqs",
                            "frontier_batches",
                            "groups",
                            "prefill_layer_prepares",
                        )
                    }
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
