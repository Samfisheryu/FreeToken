#!/usr/bin/env python3
"""Eight-session closed-loop benchmark for the scaled public Qwen3-MoE model."""

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
SESSION_COUNT = 8
TURNS_PER_SESSION = 5
INITIAL_PROMPT_TOKENS = 2048
USER_SEGMENT_TOKENS = 512
COMPLETION_TOKENS = 512
THINK_TIME_SECONDS = 0.100
MAX_SEQ_LEN = 16384
DEFAULT_MODES = (
    "legacy_t8192_c24",
    "legacy_t8192_c40",
    "layered_pipeline_g1_t512_cpi16_wave64_c24",
    "layered_pipeline_g1_t512_cpi16_wave64_c40",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18340)
    parser.add_argument("--server-timeout", type=float, default=1200.0)
    parser.add_argument("--ft-executable")
    args = parser.parse_args()
    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    unknown = [name for name in args.modes if name not in DEFAULT_MODES]
    if unknown:
        parser.error(f"unknown closed-loop modes: {', '.join(unknown)}")
    return args


def mode_contract(name: str) -> dict[str, Any]:
    cache_size = 40 if name.endswith("_c40") else 24
    if name.startswith("legacy_"):
        return {
            "name": name,
            "batching_policy": "legacy",
            "max_prefill_length": 8192,
            "moe_cache_size": cache_size,
        }
    return {
        "name": name,
        "batching_policy": "layered-pipeline",
        "prefill_layer_group_size": 1,
        "prefill_wave_max_chunks": 64,
        "layered_pipeline_chunks_per_iteration": 16,
        "max_prefill_length": 512,
        "moe_cache_size": cache_size,
    }


def server_workload(mode: dict[str, Any]) -> dict[str, Any]:
    workload = lab.load_workload()
    workload["public_server_config"].update(
        {
            "max_running_requests": SESSION_COUNT,
            "max_seq_len_override": MAX_SEQ_LEN,
            "max_prefill_length": mode["max_prefill_length"],
            "dtype": "float16",
            "attention_backend": "triton",
            "moe_cache_size": mode["moe_cache_size"],
            "cuda_graph_max_bs": SESSION_COUNT,
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
) -> list[str]:
    command = lab.server_command(
        ft_executable,
        model_path,
        mode,
        host,
        port,
        server_workload(mode),
    )
    command.append("--moe-collect-stats")
    return command


def materialize_inputs(tokenizer: Any, repetitions: int) -> dict[str, Any]:
    required_initials = repetitions * SESSION_COUNT
    first_tokens = lab.first_token_candidates(tokenizer, required_initials + 2)
    continuation = lab.continuation_token_pieces(tokenizer)
    materials: list[list[dict[str, Any]]] = []
    for repetition in range(repetitions):
        sessions: list[dict[str, Any]] = []
        for session_index in range(SESSION_COUNT):
            material_index = repetition * SESSION_COUNT + session_index
            initial_text, initial_ids = lab.materialize_segment_text(
                tokenizer,
                INITIAL_PROMPT_TOKENS,
                20260827 + repetition * 10000 + session_index * 100,
                f"closed-loop-r{repetition}-s{session_index}-initial",
                continuation,
                first_token=first_tokens[material_index],
            )
            segments: list[dict[str, Any]] = []
            for turn_index in range(1, TURNS_PER_SESSION):
                text, token_ids = lab.materialize_segment_text(
                    tokenizer,
                    USER_SEGMENT_TOKENS,
                    30260827
                    + repetition * 10000
                    + session_index * 100
                    + turn_index,
                    f"closed-loop-r{repetition}-s{session_index}-turn{turn_index}",
                    continuation,
                )
                segments.append(
                    {
                        "text": text,
                        "token_count": len(token_ids),
                        "seed": 30260827
                        + repetition * 10000
                        + session_index * 100
                        + turn_index,
                    }
                )
            sessions.append(
                {
                    "session_index": session_index,
                    "initial_text": initial_text,
                    "initial_token_count": len(initial_ids),
                    "segments": segments,
                }
            )
        materials.append(sessions)

    warmup_text, warmup_ids = lab.materialize_segment_text(
        tokenizer,
        INITIAL_PROMPT_TOKENS,
        40260827,
        "closed-loop-warmup",
        continuation,
        first_token=first_tokens[-2],
    )
    readiness_id, readiness_text = first_tokens[-1]
    return {
        "repetitions": materials,
        "warmup": {"text": warmup_text, "token_count": len(warmup_ids)},
        "readiness_prompt_token_id": readiness_id,
        "readiness_prompt_text": readiness_text,
    }


def checkpoint_max_position_embeddings(model_path: Path) -> int:
    with (model_path / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    value = int(config.get("max_position_embeddings", 0))
    if value < 1:
        raise ValueError("scaled checkpoint has no positive max_position_embeddings")
    return value


def create_long_context_scaled_model(model_path: Path) -> None:
    scaled.create_scaled_qwen3_moe(model_path)
    config_path = model_path / "config.json"
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    config["max_position_embeddings"] = MAX_SEQ_LEN
    lab.write_json(config_path, config)


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


def distribution(values: Iterable[float | int]) -> dict[str, float | None]:
    return lab.distribution((float(value) for value in values), [50, 95])


def finalize_request(
    record: dict[str, Any],
    *,
    turn_index: int,
    previous_prompt_tokens: int | None,
    reference_output: str | None,
) -> None:
    failures: list[str] = []
    usage = record.get("usage")
    events = record["nonempty_text_events"]
    if record.get("error"):
        failures.append("HTTP or stream error")
    if not events:
        failures.append("no non-empty SSE text event")
        record["ttft_seconds"] = None
        gaps: list[float] = []
    else:
        record["ttft_seconds"] = (
            events[0]["at_seconds"] - record["submitted_at_seconds"]
        )
        gaps = [
            later["at_seconds"] - earlier["at_seconds"]
            for earlier, later in zip(events, events[1:])
        ]
    record["tpot_seconds"] = None
    record["full_turn_latency_seconds"] = (
        record["response_complete_at_seconds"] - record["submitted_at_seconds"]
    )
    record["max_adjacent_nonempty_text_event_gap_seconds"] = (
        max(gaps) if gaps else None
    )
    record["nonempty_text_event_count"] = len(events)
    record["cached_tokens"] = None
    record["actual_new_prefill_tokens"] = None
    if not isinstance(usage, dict):
        failures.append("missing usage")
    else:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else 0
        if prompt_tokens != record["submitted_prompt_tokens"]:
            failures.append(
                f"usage.prompt_tokens {prompt_tokens!r} != submitted "
                f"{record['submitted_prompt_tokens']}"
            )
        if completion_tokens != COMPLETION_TOKENS:
            failures.append(
                f"completion_tokens {completion_tokens!r} != {COMPLETION_TOKENS}"
            )
        if events and isinstance(completion_tokens, int) and completion_tokens > 1:
            record["tpot_seconds"] = (
                record["response_complete_at_seconds"] - events[0]["at_seconds"]
            ) / (completion_tokens - 1)
        if not isinstance(cached, int) or not isinstance(prompt_tokens, int):
            failures.append("prompt cache accounting is unavailable")
        else:
            record["cached_tokens"] = cached
            record["actual_new_prefill_tokens"] = prompt_tokens - cached
            if not 0 <= cached <= prompt_tokens:
                failures.append("cached_tokens is outside the prompt")
            if turn_index == 0 and cached != 0:
                failures.append(f"isolated first turn unexpectedly cached {cached} tokens")
            if turn_index > 0:
                assert previous_prompt_tokens is not None
                if cached <= 0:
                    failures.append("continuation prompt did not hit the radix cache")
                if cached < previous_prompt_tokens - 1:
                    failures.append(
                        f"cached prefix {cached} is shorter than prior prompt "
                        f"{previous_prompt_tokens}"
                    )
    record["output_mismatch"] = (
        lab.first_difference(reference_output, record["output_text"])
        if reference_output is not None
        else None
    )
    record["measurement_failed"] = bool(failures)
    record["measurement_failures"] = failures


def run_session(
    *,
    tokenizer: Any,
    material: dict[str, Any],
    base_url: str,
    repetition: int,
    start_barrier: threading.Barrier,
    clock: dict[str, float],
    references: dict[tuple[int, int, int], str],
) -> list[dict[str, Any]]:
    session_index = material["session_index"]
    prompt_text = material["initial_text"]
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_assembly: dict[str, Any] = {
        "target_prompt_tokens": INITIAL_PROMPT_TOKENS,
        "previous_prompt_tokens": None,
        "previous_output_retokenized_tokens": None,
        "user_segment_tokens": None,
        "prior_prompt_token_prefix_preserved": None,
    }
    start_barrier.wait()
    origin = clock["origin"]
    records: list[dict[str, Any]] = []
    for turn_index in range(TURNS_PER_SESSION):
        previous_prompt_tokens = prompt_assembly["previous_prompt_tokens"]
        request_seed = (
            50260827 + repetition * 10000 + session_index * 100 + turn_index
        )
        record = lab.request_completion(
            base_url,
            prompt_text,
            len(prompt_ids),
            COMPLETION_TOKENS,
            request_seed,
            origin,
        )
        record.update(
            {
                "repetition": repetition,
                "session_index": session_index,
                "session_id": f"session_{session_index}",
                "turn_index": turn_index,
                "request_seed": request_seed,
                **prompt_assembly,
                "prompt_token_delta_from_target": (
                    len(prompt_ids) - prompt_assembly["target_prompt_tokens"]
                ),
            }
        )
        reference = references.get((repetition, session_index, turn_index))
        finalize_request(
            record,
            turn_index=turn_index,
            previous_prompt_tokens=previous_prompt_tokens,
            reference_output=reference,
        )
        records.append(record)

        if turn_index + 1 == TURNS_PER_SESSION:
            continue
        think_deadline = (
            origin + record["response_complete_at_seconds"] + THINK_TIME_SECONDS
        )
        previous_text = prompt_text
        previous_ids = prompt_ids
        output_text = record["output_text"]
        output_ids = tokenizer.encode(output_text, add_special_tokens=False)
        segment = material["segments"][turn_index]
        prompt_text = previous_text + output_text + segment["text"]
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_assembly = {
            "target_prompt_tokens": (
                INITIAL_PROMPT_TOKENS
                + (turn_index + 1) * (COMPLETION_TOKENS + USER_SEGMENT_TOKENS)
            ),
            "previous_prompt_tokens": len(previous_ids),
            "previous_output_retokenized_tokens": len(output_ids),
            "user_segment_tokens": segment["token_count"],
            "user_segment_seed": segment["seed"],
            "prior_prompt_token_prefix_preserved": (
                prompt_ids[: len(previous_ids)] == previous_ids
            ),
            "appended_prompt_tokens": len(prompt_ids) - len(previous_ids),
        }
        remaining = think_deadline - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
    return records


def validate_stats(delta: dict[str, int]) -> list[str]:
    failures: list[str] = []
    if delta["decode_missing_rows"] > delta["decode_active_rows"]:
        failures.append("decode missing rows exceed active rows")
    if delta["prefill_hit_rows"] > delta["prefill_rows"]:
        failures.append("prefill hit rows exceed admitted rows")
    row_bytes = delta["expert_row_bytes"]
    if row_bytes <= 0 or delta["prefill_h2d_bytes_total"] % row_bytes:
        failures.append("prefill H2D bytes are not integral expert rows")
    return failures


def validate_pipeline_waves(
    waves: list[dict[str, int]],
    records: list[dict[str, Any]],
    chunk_tokens: int,
) -> list[str]:
    failures: list[str] = []
    expected_chunks = sum(
        math.ceil(record["actual_new_prefill_tokens"] / chunk_tokens)
        for record in records
        if isinstance(record.get("actual_new_prefill_tokens"), int)
        and record["actual_new_prefill_tokens"] > 0
    )
    if sum(wave["chunks"] for wave in waves) != expected_chunks:
        failures.append(
            f"pipeline chunks {sum(wave['chunks'] for wave in waves)} != "
            f"expected {expected_chunks}"
        )
    for wave in waves:
        if wave["resident_groups"] != scaled.MODEL_CONTRACT["layers"]:
            failures.append("pipeline wave did not visit every resident group")
        if wave["chunk_group_steps"] != wave["chunks"] * wave["resident_groups"]:
            failures.append("pipeline chunk-group steps do not close")
        if (
            wave["frontier_group_forwards"]
            != wave["frontier_batches"] * wave["resident_groups"]
        ):
            failures.append("pipeline frontier forwards do not close")
        if wave["prefill_layer_prepares"] != scaled.MODEL_CONTRACT["layers"]:
            failures.append("pipeline layer prepares do not close")
    return failures


def summarize_repetition(
    records: list[dict[str, Any]],
    origin: float,
    stats_delta: dict[str, int],
    waves: list[dict[str, int]],
    idle_snapshot_count_delta: int,
) -> dict[str, Any]:
    makespan = max(record["response_complete_at_seconds"] for record in records)
    actual_new_prefill_tokens = sum(
        record["actual_new_prefill_tokens"] for record in records
    )
    decode_tokens = sum(record["usage"]["completion_tokens"] for record in records)
    return {
        "benchmark_start_perf_counter": origin,
        "request_count": len(records),
        "makespan_seconds": makespan,
        "turns_per_second": len(records) / makespan,
        "submitted_prompt_tokens": sum(
            record["submitted_prompt_tokens"] for record in records
        ),
        "usage_prompt_tokens": sum(record["usage"]["prompt_tokens"] for record in records),
        "cached_tokens": sum(record["cached_tokens"] for record in records),
        "actual_new_prefill_tokens": actual_new_prefill_tokens,
        "decode_tokens": decode_tokens,
        "actual_new_prefill_throughput_tokens_per_second": (
            actual_new_prefill_tokens / makespan
        ),
        "decode_throughput_tokens_per_second": decode_tokens / makespan,
        "measurement_failed_requests": sum(
            record["measurement_failed"] for record in records
        ),
        "output_mismatch_requests": sum(
            record["output_mismatch"] is not None for record in records
        ),
        "first_submission_spread_seconds": (
            max(
                record["submitted_at_seconds"]
                for record in records
                if record["turn_index"] == 0
            )
            - min(
                record["submitted_at_seconds"]
                for record in records
                if record["turn_index"] == 0
            )
        ),
        "idle_snapshot_count_delta": idle_snapshot_count_delta,
        "moe_stats_delta": stats_delta,
        "layered_pipeline_waves": waves,
    }


def metric_summary(records: list[dict[str, Any]], field: str) -> dict[str, Any]:
    def values(selected: Iterable[dict[str, Any]]) -> list[float]:
        return [
            float(record[field])
            for record in selected
            if record.get(field) is not None
        ]

    return {
        "aggregate": distribution(values(records)),
        "per_turn": {
            str(turn): distribution(
                values(record for record in records if record["turn_index"] == turn)
            )
            for turn in range(TURNS_PER_SESSION)
        },
    }


def summarize_mode(
    records: list[dict[str, Any]], repetitions: list[dict[str, Any]]
) -> dict[str, Any]:
    token_accounting = {}
    for turn in range(TURNS_PER_SESSION):
        selected = [record for record in records if record["turn_index"] == turn]
        token_accounting[str(turn)] = {
            "target_prompt_tokens_per_request": (
                INITIAL_PROMPT_TOKENS
                + turn * (COMPLETION_TOKENS + USER_SEGMENT_TOKENS)
            ),
            "actual_prompt_tokens": distribution(
                record["usage"]["prompt_tokens"] for record in selected
            ),
            "cached_tokens": distribution(record["cached_tokens"] for record in selected),
            "actual_new_prefill_tokens": distribution(
                record["actual_new_prefill_tokens"] for record in selected
            ),
        }
    stats_fields = (
        "prefill_h2d_bytes_total",
        "decode_h2d_bytes",
        "total_expert_h2d_bytes",
        "decode_active_rows",
        "decode_missing_rows",
        "prefill_rows",
        "prefill_hit_rows",
        "prefill_layer_prepares",
    )
    return {
        "ttft_seconds": metric_summary(records, "ttft_seconds"),
        "tpot_seconds": metric_summary(records, "tpot_seconds"),
        "max_adjacent_nonempty_text_event_gap_seconds": metric_summary(
            records, "max_adjacent_nonempty_text_event_gap_seconds"
        ),
        "full_turn_latency_seconds": metric_summary(
            records, "full_turn_latency_seconds"
        ),
        "makespan_seconds": distribution(
            repetition["makespan_seconds"] for repetition in repetitions
        ),
        "turns_per_second": distribution(
            repetition["turns_per_second"] for repetition in repetitions
        ),
        "actual_new_prefill_throughput_tokens_per_second": distribution(
            repetition["actual_new_prefill_throughput_tokens_per_second"]
            for repetition in repetitions
        ),
        "decode_throughput_tokens_per_second": distribution(
            repetition["decode_throughput_tokens_per_second"]
            for repetition in repetitions
        ),
        "token_totals": {
            "actual_prompt_tokens": sum(
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
        "token_accounting_by_turn": token_accounting,
        "moe_stats": {
            field: {
                **distribution(
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
            record["output_mismatch"] is not None for record in records
        ),
    }


def run_repetition(
    *,
    tokenizer: Any,
    materials: list[dict[str, Any]],
    base_url: str,
    repetition: int,
    references: dict[tuple[int, int, int], str],
) -> tuple[list[dict[str, Any]], float]:
    clock: dict[str, float] = {}

    def set_origin() -> None:
        clock["origin"] = time.perf_counter()

    start_barrier = threading.Barrier(SESSION_COUNT + 1, action=set_origin)
    with concurrent.futures.ThreadPoolExecutor(max_workers=SESSION_COUNT) as executor:
        futures = [
            executor.submit(
                run_session,
                tokenizer=tokenizer,
                material=material,
                base_url=base_url,
                repetition=repetition,
                start_barrier=start_barrier,
                clock=clock,
                references=references,
            )
            for material in materials
        ]
        start_barrier.wait()
        records = [record for future in futures for record in future.result()]
    records.sort(
        key=lambda record: (
            record["submitted_at_seconds"],
            record["session_index"],
            record["turn_index"],
        )
    )
    return records, clock["origin"]


def validate_repetition(
    records: list[dict[str, Any]],
    stats_delta: dict[str, int],
    waves: list[dict[str, int]],
    mode: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    if len(records) != SESSION_COUNT * TURNS_PER_SESSION:
        failures.append(f"request count {len(records)} != 40")
    for session_index in range(SESSION_COUNT):
        turns = sorted(
            record["turn_index"]
            for record in records
            if record["session_index"] == session_index
        )
        if turns != list(range(TURNS_PER_SESSION)):
            failures.append(f"session {session_index} turn sequence is incomplete")
    failed = sum(record["measurement_failed"] for record in records)
    if failed:
        failures.append(f"{failed} requests failed public accounting")
    failures.extend(validate_stats(stats_delta))
    if mode["batching_policy"] == "layered-pipeline":
        failures.extend(
            validate_pipeline_waves(waves, records, mode["max_prefill_length"])
        )
    elif waves:
        failures.append("legacy mode unexpectedly emitted pipeline waves")
    return failures


def dry_run_plan(
    args: argparse.Namespace,
    modes: list[dict[str, Any]],
) -> dict[str, Any]:
    executable = lab.find_ft_executable(args.ft_executable)
    model_path: Path | str = args.model or "<auto-generated-scaled-qwen3-moe>"
    return {
        "dry_run": True,
        "schema": "freetoken.scaled_agent_closed_loop.v1",
        "repetitions": args.repetitions,
        "sessions": SESSION_COUNT,
        "turns_per_session": TURNS_PER_SESSION,
        "requests_per_repetition": SESSION_COUNT * TURNS_PER_SESSION,
        "target_prompt_tokens_by_turn": [
            INITIAL_PROMPT_TOKENS
            + turn * (COMPLETION_TOKENS + USER_SEGMENT_TOKENS)
            for turn in range(TURNS_PER_SESSION)
        ],
        "completion_tokens": COMPLETION_TOKENS,
        "think_time_ms": int(THINK_TIME_SECONDS * 1000),
        "commands": [
            {
                "mode": mode["name"],
                "argv": server_command(
                    executable, model_path, mode, args.host, args.port
                ),
            }
            for mode in modes
        ],
    }


def main() -> int:
    args = parse_args()
    modes = [mode_contract(name) for name in args.modes]
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, modes), indent=2))
        return 0

    executable = lab.find_ft_executable(args.ft_executable)
    owned_model_root: Path | None = None
    if args.model is None:
        owned_model_root = Path(tempfile.mkdtemp(prefix="freetoken-scaled-agent-"))
        model_path = owned_model_root / "model"
        create_long_context_scaled_model(model_path)
    else:
        model_path = args.model.resolve()
        max_positions = checkpoint_max_position_embeddings(model_path)
        if max_positions < MAX_SEQ_LEN:
            raise ValueError(
                f"scaled checkpoint supports {max_positions} positions; "
                f"closed-loop workload requires at least {MAX_SEQ_LEN}"
            )
    tokenizer = lab.load_tokenizer(model_path)
    materials = materialize_inputs(tokenizer, args.repetitions)
    output = args.output or (
        HERE / "results" / f"scaled_agent_closed_loop_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result: dict[str, Any] = {
        "schema": "freetoken.scaled_agent_closed_loop.v1",
        "created_at_unix_seconds": time.time(),
        "model_path": str(model_path),
        "auto_generated_model": args.model is None,
        "model_contract": scaled.MODEL_CONTRACT,
        "workload_contract": {
            "sessions": SESSION_COUNT,
            "turns_per_session": TURNS_PER_SESSION,
            "requests_per_repetition": SESSION_COUNT * TURNS_PER_SESSION,
            "initial_prompt_tokens": INITIAL_PROMPT_TOKENS,
            "real_previous_output_in_next_prompt": True,
            "deterministic_user_segment_tokens": USER_SEGMENT_TOKENS,
            "completion_tokens": COMPLETION_TOKENS,
            "ignore_eos": True,
            "think_time_ms": int(THINK_TIME_SECONDS * 1000),
            "target_prompt_tokens_by_turn": [
                INITIAL_PROMPT_TOKENS
                + turn * (COMPLETION_TOKENS + USER_SEGMENT_TOKENS)
                for turn in range(TURNS_PER_SESSION)
            ],
            "max_seq_len_override": MAX_SEQ_LEN,
            "attention_backend": "triton",
            "cuda_graph_max_bs": SESSION_COUNT,
        },
        "reference_mode": modes[0]["name"],
        "modes": [],
    }
    references: dict[tuple[int, int, int], str] = {}
    base_url = f"http://{args.host}:{args.port}"
    try:
        for mode_index, mode in enumerate(modes):
            command = server_command(
                executable, model_path, mode, args.host, args.port
            )
            mode_result: dict[str, Any] = {
                "name": mode["name"],
                "contract": mode,
                "server_command": command,
                "warmup": None,
                "repetitions": [],
                "requests": [],
                "layered_pipeline_waves": [],
                "summary": None,
                "server_log_tail": None,
                "error": None,
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
                warmup_origin = time.perf_counter()
                warmup = lab.request_completion(
                    base_url,
                    materials["warmup"]["text"],
                    materials["warmup"]["token_count"],
                    COMPLETION_TOKENS,
                    60260827,
                    warmup_origin,
                )
                warmup_usage = warmup.get("usage")
                if (
                    warmup.get("error")
                    or not isinstance(warmup_usage, dict)
                    or warmup_usage.get("prompt_tokens") != INITIAL_PROMPT_TOKENS
                    or warmup_usage.get("completion_tokens") != COMPLETION_TOKENS
                ):
                    raise RuntimeError(f"public warmup failed: {warmup}")
                snapshots = wait_for_idle_snapshot(server, 1, args.server_timeout)
                baseline_snapshot = snapshots[-1]
                snapshot_count = len(snapshots)
                mode_result["warmup"] = {
                    "usage": warmup_usage,
                    "full_turn_latency_seconds": warmup[
                        "response_complete_at_seconds"
                    ],
                }
                server.mark_measurement_start()
                wave_cursor = 0

                for repetition in range(args.repetitions):
                    records, origin = run_repetition(
                        tokenizer=tokenizer,
                        materials=materials["repetitions"][repetition],
                        base_url=base_url,
                        repetition=repetition,
                        references=references,
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
                    all_waves = (
                        server.layered_pipeline_waves()
                        if mode["batching_policy"] == "layered-pipeline"
                        else []
                    )
                    waves = all_waves[wave_cursor:]
                    wave_cursor = len(all_waves)
                    repetition_failures = validate_repetition(
                        records, stats_delta, waves, mode
                    )
                    repetition_summary = summarize_repetition(
                        records,
                        origin,
                        stats_delta,
                        waves,
                        len(final_snapshots) - snapshot_count,
                    )
                    repetition_summary["validation_failures"] = repetition_failures
                    mode_result["requests"].extend(records)
                    mode_result["repetitions"].append(repetition_summary)
                    mode_result["layered_pipeline_waves"].extend(waves)
                    lab.write_json(output, result)
                    if repetition_failures:
                        raise RuntimeError(
                            f"mode {mode['name']} repetition {repetition} failed: "
                            + "; ".join(repetition_failures)
                        )
                    if mode_index == 0:
                        for record in records:
                            references[
                                (
                                    repetition,
                                    record["session_index"],
                                    record["turn_index"],
                                )
                            ] = record["output_text"]
                    baseline_snapshot = after_snapshot
                    snapshot_count = len(final_snapshots)

                mode_result["summary"] = summarize_mode(
                    mode_result["requests"], mode_result["repetitions"]
                )
            except Exception as exc:
                mode_result["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                server.stop()
                mode_result["server_log_tail"] = server.log_tail()
                server.close()
                lab.write_json(output, result)

        reference_accounting = {
            (
                record["repetition"],
                record["session_index"],
                record["turn_index"],
            ): (
                record["usage"]["prompt_tokens"],
                record["usage"]["completion_tokens"],
            )
            for record in result["modes"][0]["requests"]
        }
        result["cross_mode_prompt_and_completion_accounting_equal"] = all(
            (
                record["usage"]["prompt_tokens"],
                record["usage"]["completion_tokens"],
            )
            == reference_accounting[
                (
                    record["repetition"],
                    record["session_index"],
                    record["turn_index"],
                )
            ]
            for mode_result in result["modes"][1:]
            for record in mode_result["requests"]
        )
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
