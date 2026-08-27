#!/usr/bin/env python3
"""Black-box lab-agent policy benchmark for the public FreeToken API."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable
import urllib.error
import urllib.request


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_WORKLOAD = HERE / "workloads" / "lab_agent_burst_v1.json"
AUTO_TOKENIZER = "Qwen/Qwen3-0.6B"
SERVED_MODEL = "lab-agent-qwen3-moe"
MODE_ALIASES = {
    "legacy": "legacy",
    "mixed": "mixed",
    "layeredG2": "layered_g2_serial",
    "layered_g2_serial": "layered_g2_serial",
    "jointG2-wave1": "joint_g2_wave1",
    "joint_g2_wave1": "joint_g2_wave1",
    "jointG2-wave2": "joint_g2_wave2",
    "joint_g2_wave2": "joint_g2_wave2",
    "jointG2-wave4": "joint_g2_wave4_exploratory",
    "wave4": "joint_g2_wave4_exploratory",
    "joint_g2_wave4_exploratory": "joint_g2_wave4_exploratory",
    "layered-pipeline": "layered_pipeline_g2_cpi1",
    "layered-pipeline-cpi1": "layered_pipeline_g2_cpi1",
    "layered_pipeline_g2_cpi1": "layered_pipeline_g2_cpi1",
    "layered-pipeline-cpi2": "layered_pipeline_g2_cpi2",
    "layered_pipeline_g2_cpi2": "layered_pipeline_g2_cpi2",
    "layered-pipeline-cpi3": "layered_pipeline_g2_cpi3",
    "layered_pipeline_g2_cpi3": "layered_pipeline_g2_cpi3",
    "layered-pipeline-cpi4": "layered_pipeline_g2_cpi4",
    "layered_pipeline_g2_cpi4": "layered_pipeline_g2_cpi4",
    "layered-prefill": "layered_prefill_g2_wave1",
    "layered_prefill_g2_wave1": "layered_prefill_g2_wave1",
}

LAYERED_PIPELINE_WAVE_RE = re.compile(
    r"Layered pipeline wave complete: "
    r"chunks=(\d+), wave_reqs=(\d+), frontier_batches=(\d+), "
    r"resident_groups=(\d+), chunk_group_steps=(\d+), "
    r"frontier_group_forwards=(\d+), "
    r"iterations=(\d+), decode_iterations=(\d+), "
    r"prefill_layer_prepares=(\d+), cross_group_prefetches=(\d+), "
    r"deferred_cross_group_prefetches=(\d+)"
)
JOINT_WAVE_RE = re.compile(
    r"Joint wave complete: "
    r"chunks=(\d+), wave_reqs=(\d+), frontier_batches=(\d+), groups=(\d+), "
    r"effective_group_size=(\d+), prefill_layer_prepares=(\d+)"
)
LAYERED_PREFILL_WAVE_RE = re.compile(
    r"Layered prefill wave complete: "
    r"reqs=(\d+), groups=(\d+), group_forwards=(\d+), "
    r"iterations=(\d+), decode_iterations=(\d+), "
    r"prefill_layer_prepares=(\d+)"
)
MOE_CACHE_STATS_RE = re.compile(r"MoE cache stats snapshot: ([^\n]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run four closed-loop lab-agent sessions against one public `ft serve` "
            "process per scheduling mode and save raw SSE timings as JSON."
        )
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=[
            "legacy",
            "mixed",
            "layeredG2",
            "jointG2-wave1",
            "jointG2-wave2",
            "layered-pipeline",
        ],
        help=(
            "Modes, separated by spaces or commas. Primary defaults: legacy mixed "
            "layeredG2 jointG2-wave1 jointG2-wave2 layered-pipeline (G2/CPI1). "
            "Use layered-pipeline-cpi2/cpi3/cpi4 or 'all' for CPI experiments."
        ),
    )
    parser.add_argument(
        "--joint-groups",
        nargs="+",
        help=(
            "Joint layer-group sizes to sweep, separated by spaces or commas. "
            "Requires --joint-waves."
        ),
    )
    parser.add_argument(
        "--joint-waves",
        nargs="+",
        help=(
            "Joint prefill wave chunk counts to sweep, separated by spaces or commas. "
            "Requires --joint-groups."
        ),
    )
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument(
        "--max-prefill-length",
        type=int,
        help="Override the workload's server-side prefill chunk length.",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value for each server")
    parser.add_argument("--profile", default="main", help="Workload profile name")
    parser.add_argument(
        "--model",
        type=Path,
        help="Local model checkpoint. If omitted, create a small public Qwen3-MoE checkpoint.",
    )
    parser.add_argument("--output", type=Path, help="Raw JSON output path")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the plan only")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--server-timeout", type=float, default=300.0)
    parser.add_argument(
        "--ft-executable",
        help="Public ft executable (default: PATH)",
    )
    args = parser.parse_args()
    if args.max_prefill_length is not None and args.max_prefill_length < 1:
        parser.error("--max-prefill-length must be at least 1")
    if (args.joint_groups is None) != (args.joint_waves is None):
        parser.error("--joint-groups and --joint-waves must be provided together")
    if args.joint_groups is not None:
        args.joint_groups = parse_positive_int_values(
            args.joint_groups, "--joint-groups", parser
        )
        args.joint_waves = parse_positive_int_values(
            args.joint_waves, "--joint-waves", parser
        )
    return args


def parse_positive_int_values(
    raw_values: Iterable[str], option: str, parser: argparse.ArgumentParser
) -> list[int]:
    pieces = [
        piece.strip()
        for raw_value in raw_values
        for piece in raw_value.split(",")
        if piece.strip()
    ]
    if not pieces:
        parser.error(f"{option} requires at least one positive integer")
    values: list[int] = []
    for piece in pieces:
        try:
            value = int(piece)
        except ValueError:
            parser.error(f"{option} values must be positive integers; got {piece!r}")
        if value <= 0:
            parser.error(f"{option} values must be positive integers; got {piece!r}")
        if value not in values:
            values.append(value)
    return values


def load_workload() -> dict[str, Any]:
    with DEFAULT_WORKLOAD.open("r", encoding="utf-8") as handle:
        workload = json.load(handle)
    if workload.get("schema") != "freetoken.public_workload.v1":
        raise ValueError("unsupported workload schema")
    return workload


def validate_workload(workload: dict[str, Any], profile_name: str) -> dict[str, Any]:
    if profile_name not in workload["profiles"]:
        raise ValueError(f"unknown profile {profile_name!r}")
    profile = workload["profiles"][profile_name]
    turns = workload["closed_loop_submission"]["turns_per_session"]
    sessions = workload["sessions"]
    if turns != len(profile["prompt_tokens_by_turn"]):
        raise ValueError("turn count and prompt_tokens_by_turn disagree")
    if len(sessions) != 4 or turns != 5:
        raise ValueError("this benchmark contract requires four sessions of five turns")
    if len(sessions) * turns != workload["closed_loop_submission"]["total_requests"]:
        raise ValueError("total_requests disagrees with sessions times turns")
    expected_lengths = [profile["initial_prompt_tokens"]]
    for _ in range(1, turns):
        expected_lengths.append(expected_lengths[-1] + profile["incremental_prompt_tokens_after_first_turn"])
    if expected_lengths != profile["prompt_tokens_by_turn"]:
        raise ValueError("profile prompt lengths do not match the declared increment")
    if profile_name == "main":
        if expected_lengths != [640, 1472, 2304, 3136, 3968]:
            raise ValueError("main profile prompt lengths changed")
        if profile["requested_decode_tokens"] != 512:
            raise ValueError("main profile must request exactly 512 output tokens")
        if profile["expected_actual_new_prefill_tokens_by_turn"] != [640, 832, 832, 832, 832]:
            raise ValueError("main profile new-prefill expectations changed")
    if max(profile["prompt_plus_requested_output_tokens_by_turn"]) > profile["max_seq_len"]:
        raise ValueError("profile exceeds max_seq_len")
    return profile


def resolve_modes(
    raw_modes: Iterable[str],
    workload: dict[str, Any],
    joint_groups: Iterable[int] | None = None,
    joint_waves: Iterable[int] | None = None,
) -> list[dict[str, Any]]:
    available = {mode["name"]: mode for mode in workload["comparison_modes"]}
    tokens = [piece for item in raw_modes for piece in item.split(",") if piece]
    if tokens == ["all"]:
        names = list(available)
    else:
        names = []
        for token in tokens:
            if token not in MODE_ALIASES:
                raise ValueError(f"unknown mode {token!r}")
            name = MODE_ALIASES[token]
            if name not in names:
                names.append(name)
    modes = [available[name] for name in names]
    if joint_groups is not None and joint_waves is not None:
        modes.extend(
            {
                "name": f"joint_g{group_size}_wave{wave_chunks}",
                "batching_policy": "joint",
                "prefill_layer_group_size": group_size,
                "prefill_wave_max_chunks": wave_chunks,
                "primary": False,
            }
            for group_size in joint_groups
            for wave_chunks in joint_waves
        )

    unique_modes: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for mode in modes:
        identity = (
            mode["batching_policy"],
            mode.get("prefill_layer_group_size"),
            mode.get("prefill_execution"),
            mode.get("prefill_wave_max_chunks"),
            mode.get("layered_pipeline_chunks_per_iteration"),
        )
        if identity not in seen:
            seen.add(identity)
            unique_modes.append(mode)
    return unique_modes


def find_ft_executable(explicit: str | None) -> str:
    if explicit:
        return explicit
    found = shutil.which("ft")
    if found:
        return found
    raise FileNotFoundError("ft executable not found on PATH; pass --ft-executable")


def server_command(
    ft_executable: str,
    model_path: Path | str,
    mode: dict[str, Any],
    host: str,
    port: int,
    workload: dict[str, Any],
) -> list[str]:
    cfg = workload["public_server_config"]
    command = [
        ft_executable,
        "serve",
        "--model-path",
        str(model_path),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        cfg["dtype"],
        "--max-running-requests",
        str(cfg["max_running_requests"]),
        "--max-seq-len-override",
        str(cfg["max_seq_len_override"]),
        "--max-prefill-length",
        str(cfg["max_prefill_length"]),
        "--attention-backend",
        cfg["attention_backend"],
        "--moe-backend",
        "offload",
        "--moe-cache-size",
        str(cfg["moe_cache_size"]),
        "--cuda-graph-max-bs",
        str(cfg["cuda_graph_max_bs"]),
        "--cache-type",
        cfg["cache_type"],
        "--enable-cache-report",
        "--batching-policy",
        mode["batching_policy"],
    ]
    if "prefill_layer_group_size" in mode:
        command += ["--prefill-layer-group-size", str(mode["prefill_layer_group_size"])]
    if mode["batching_policy"] == "layered":
        command += ["--prefill-execution", mode["prefill_execution"]]
    if "prefill_wave_max_chunks" in mode:
        command += ["--prefill-wave-max-chunks", str(mode["prefill_wave_max_chunks"])]
    if "layered_pipeline_chunks_per_iteration" in mode:
        command += [
            "--layered-pipeline-chunks-per-iteration",
            str(mode["layered_pipeline_chunks_per_iteration"]),
        ]
    return command


def create_small_qwen3_moe(destination: Path) -> None:
    try:
        import torch
        from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM
    except ImportError as exc:
        raise RuntimeError("automatic model creation requires torch and transformers") from exc

    torch.manual_seed(20260825)
    try:
        tokenizer = AutoTokenizer.from_pretrained(AUTO_TOKENIZER, local_files_only=True)
    except OSError:
        print(
            f"Tokenizer {AUTO_TOKENIZER} is not cached locally; downloading it explicitly.",
            file=sys.stderr,
            flush=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(AUTO_TOKENIZER)
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=5,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        max_position_embeddings=8192,
        decoder_sparse_step=1,
        moe_intermediate_size=1024,
        num_experts_per_tok=2,
        num_experts=8,
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


def load_tokenizer(model_path: Path):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("benchmark prompt materialization requires transformers") from exc
    return AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)


def first_token_candidates(tokenizer, required: int) -> list[tuple[int, str]]:
    special = set(tokenizer.all_special_ids)
    candidates: list[tuple[int, str]] = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if (
            not text
            or not text.isascii()
            or not text.isalpha()
            or text[:1].isspace()
            or "\ufffd" in text
        ):
            continue
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if encoded == [token_id]:
            candidates.append((token_id, text))
            if len(candidates) >= required:
                return candidates
    raise RuntimeError(f"tokenizer has fewer than {required} usable distinct first tokens")


def continuation_token_pieces(tokenizer, required: int = 64) -> list[tuple[int, str]]:
    special = set(tokenizer.all_special_ids)
    candidates: list[tuple[int, str]] = []
    for token_id in range(len(tokenizer)):
        if token_id in special:
            continue
        text = tokenizer.decode([token_id], skip_special_tokens=False)
        if (
            len(text) < 2
            or not text.isascii()
            or text[0] != " "
            or not text[1:].isalpha()
            or "\ufffd" in text
        ):
            continue
        if tokenizer.encode(text, add_special_tokens=False) == [token_id]:
            candidates.append((token_id, text))
            if len(candidates) >= required:
                return candidates
    raise RuntimeError(f"tokenizer has fewer than {required} composable text tokens")


def materialize_segment_text(
    tokenizer,
    length: int,
    seed: int,
    label: str,
    continuation_pieces: list[tuple[int, str]],
    first_token: tuple[int, str] | None = None,
) -> tuple[str, list[int]]:
    rng = random.Random(seed)
    text_pieces: list[str] = []
    expected_ids: list[int] = []
    if first_token is not None:
        first_id, first_text = first_token
        text_pieces.append(first_text)
        expected_ids.append(first_id)
    while len(expected_ids) < length:
        token_id, text = continuation_pieces[rng.randrange(len(continuation_pieces))]
        text_pieces.append(text)
        expected_ids.append(token_id)
    text = "".join(text_pieces)
    actual_ids = tokenizer.encode(text, add_special_tokens=False)
    special = set(tokenizer.all_special_ids)
    if (
        actual_ids != expected_ids
        or tokenizer.decode(actual_ids, skip_special_tokens=False) != text
        or any(token_id in special for token_id in actual_ids)
    ):
        raise RuntimeError(
            f"failed to materialize {label} as exactly {length} round-trip-safe text tokens"
        )
    return text, actual_ids


def materialize_prompts(
    tokenizer,
    workload: dict[str, Any],
    profile: dict[str, Any],
    repetition: int,
    first_tokens: list[tuple[int, str]],
    continuation_pieces: list[tuple[int, str]],
) -> dict[str, list[dict[str, Any]]]:
    base_seed = workload["content_materialization"]["seed"] + 10000 * repetition
    turns = workload["closed_loop_submission"]["turns_per_session"]
    prompts: dict[str, list[dict[str, Any]]] = {}
    observed_first: list[int] = []
    for session in workload["sessions"]:
        user_index = session["user_index"]
        first_token = first_tokens[repetition * len(workload["sessions"]) + user_index]
        initial_text, initial_ids = materialize_segment_text(
            tokenizer,
            profile["initial_prompt_tokens"],
            base_seed + 100 * user_index,
            f"project-context-user-{user_index}",
            continuation_pieces,
            first_token=first_token,
        )
        if initial_ids[0] != first_token[0]:
            raise RuntimeError("initial prompt did not preserve its selected isolation token")
        observed_first.append(initial_ids[0])
        user_prompts = [{"text": initial_text, "token_count": len(initial_ids)}]
        current_text = initial_text
        current_ids = initial_ids
        for transition in range(turns - 1):
            assistant_text, assistant_ids = materialize_segment_text(
                tokenizer,
                profile["fixed_assistant_transcript_tokens"],
                base_seed + 100 * user_index + 10 * transition + 1,
                f"fixed-assistant-user-{user_index}-turn-{transition}",
                continuation_pieces,
            )
            tool_text, tool_ids = materialize_segment_text(
                tokenizer,
                profile["fixed_tool_result_tokens"],
                base_seed + 100 * user_index + 10 * transition + 2,
                f"fixed-tool-user-{user_index}-turn-{transition + 1}",
                continuation_pieces,
            )
            next_text = current_text + assistant_text + tool_text
            next_ids = tokenizer.encode(next_text, add_special_tokens=False)
            expected_ids = current_ids + assistant_ids + tool_ids
            if (
                next_ids != expected_ids
                or next_ids[: len(current_ids)] != current_ids
                or tokenizer.decode(next_ids, skip_special_tokens=False) != next_text
            ):
                raise RuntimeError(
                    f"text prompt boundary changed tokenization for user {user_index}, "
                    f"turn {transition + 1}"
                )
            current_text = next_text
            current_ids = next_ids
            user_prompts.append({"text": current_text, "token_count": len(current_ids)})
        expected = profile["prompt_tokens_by_turn"]
        actual = [prompt["token_count"] for prompt in user_prompts]
        if actual != expected:
            raise RuntimeError(f"materialized prompt lengths {actual} != {expected}")
        prompts[session["user_id"]] = user_prompts
    if len(observed_first) != len(set(observed_first)):
        raise RuntimeError("session-isolation first tokens are not distinct")
    return prompts


def relative_now(origin: float) -> float:
    return time.perf_counter() - origin


def request_completion(
    base_url: str,
    prompt_text: str,
    prompt_token_count: int,
    decode_tokens: int,
    request_seed: int,
    origin: float,
    first_text_event: threading.Event | None = None,
) -> dict[str, Any]:
    payload = {
        "model": SERVED_MODEL,
        "prompt": prompt_text,
        "max_tokens": decode_tokens,
        "temperature": 0.0,
        "ignore_eos": True,
        "seed": request_seed,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    submitted_at = relative_now(origin)
    result: dict[str, Any] = {
        "submitted_at_seconds": submitted_at,
        "submitted_prompt_tokens": prompt_token_count,
        "choice_events": [],
        "nonempty_text_events": [],
        "usage": None,
        "output_text": "",
        "error": None,
    }
    request = urllib.request.Request(
        base_url + "/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=3600) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                timestamp = relative_now(origin)
                if event.get("usage") is not None:
                    result["usage"] = event["usage"]
                choices = event.get("choices") or []
                for choice in choices:
                    text = choice.get("text") or ""
                    result["choice_events"].append(
                        {"at_seconds": timestamp, "text": text}
                    )
                    if text:
                        result["nonempty_text_events"].append(
                            {"at_seconds": timestamp, "text": text}
                        )
                        result["output_text"] += text
                        if first_text_event is not None:
                            first_text_event.set()
    except Exception as exc:  # The raw result must retain observable HTTP/stream failures.
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["response_complete_at_seconds"] = relative_now(origin)
    return result


def percentile(values: Iterable[float], percent: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def distribution(values: Iterable[float], percents: Iterable[int]) -> dict[str, float | None]:
    materialized = list(values)
    return {f"p{percent}": percentile(materialized, percent) for percent in percents}


def first_difference(reference: str, actual: str) -> dict[str, Any] | None:
    if reference == actual:
        return None
    limit = min(len(reference), len(actual))
    offset = next((index for index in range(limit) if reference[index] != actual[index]), limit)
    return {
        "character_offset": offset,
        "reference_length": len(reference),
        "actual_length": len(actual),
        "reference_fragment": reference[offset : offset + 80],
        "actual_fragment": actual[offset : offset + 80],
    }


def finalize_request(
    result: dict[str, Any],
    profile: dict[str, Any],
    turn_index: int,
    reference_output: str | None,
) -> None:
    failures: list[str] = []
    events = result["nonempty_text_events"]
    usage = result.get("usage")
    if result.get("error"):
        failures.append("HTTP or stream error")
    if not events:
        failures.append("no non-empty SSE choice text event")
        result["ttft_seconds"] = None
        result["tpot_seconds"] = None
        gaps: list[float] = []
    else:
        result["ttft_seconds"] = events[0]["at_seconds"] - result["submitted_at_seconds"]
        gaps = [
            later["at_seconds"] - earlier["at_seconds"]
            for earlier, later in zip(events, events[1:])
        ]
        completion_tokens = usage.get("completion_tokens") if usage else None
        result["tpot_seconds"] = (
            (events[-1]["at_seconds"] - events[0]["at_seconds"]) / (completion_tokens - 1)
            if isinstance(completion_tokens, int) and completion_tokens > 1
            else None
        )
    result["choice_event_count"] = len(result["choice_events"])
    result["nonempty_text_event_count"] = len(events)
    result["sse_text_event_gap_seconds"] = {
        **distribution(gaps, [50, 95, 99]),
        "max": max(gaps) if gaps else None,
    }
    result["gap_interpretation"] = "SSE text-event gap"
    result["cached_tokens"] = None
    result["actual_new_prefill_tokens"] = None
    if not isinstance(usage, dict):
        failures.append("missing usage")
    else:
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        details = usage.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else None
        expected_cached = profile["expected_cached_tokens_by_turn"][turn_index]
        if completion_tokens != profile["requested_decode_tokens"]:
            failures.append(
                f"completion_tokens {completion_tokens!r} != {profile['requested_decode_tokens']}"
            )
        if prompt_tokens != result["submitted_prompt_tokens"]:
            failures.append(
                f"usage.prompt_tokens {prompt_tokens!r} != submitted {result['submitted_prompt_tokens']}"
            )
        if cached is None and expected_cached == 0:
            cached = 0
        if not isinstance(cached, int):
            failures.append("usage.prompt_tokens_details.cached_tokens unavailable")
        elif isinstance(prompt_tokens, int):
            actual_new = prompt_tokens - cached
            result["cached_tokens"] = cached
            result["actual_new_prefill_tokens"] = actual_new
            expected_new = profile["expected_actual_new_prefill_tokens_by_turn"][turn_index]
            if cached != expected_cached:
                failures.append(f"cached_tokens {cached} != expected {expected_cached}")
            if actual_new != expected_new:
                failures.append(f"actual_new_prefill_tokens {actual_new} != expected {expected_new}")
        if result["nonempty_text_event_count"] == completion_tokens:
            result["gap_interpretation"] = "inter-token gap"
    result["output_mismatch"] = (
        first_difference(reference_output, result["output_text"])
        if reference_output is not None
        else None
    )
    result["measurement_failed"] = bool(failures)
    result["measurement_failures"] = failures


def run_session(
    session: dict[str, Any],
    prompts: list[dict[str, Any]],
    profile: dict[str, Any],
    workload: dict[str, Any],
    base_url: str,
    origin: float,
    repetition: int,
    references: dict[tuple[int, str, int], str],
) -> list[dict[str, Any]]:
    delay = session["start_delay_ms"] / 1000.0
    remaining = origin + delay - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)
    records = []
    for turn_index, prompt in enumerate(prompts):
        if turn_index:
            time.sleep(session["think_time_after_completion_ms"] / 1000.0)
        request_seed = (
            workload["content_materialization"]["seed"]
            + 100 * session["user_index"]
            + turn_index
        )
        record = request_completion(
            base_url,
            prompt["text"],
            prompt["token_count"],
            profile["requested_decode_tokens"],
            request_seed,
            origin,
        )
        record.update(
            {
                "repetition": repetition,
                "user_id": session["user_id"],
                "user_index": session["user_index"],
                "turn_index": turn_index,
                "request_seed": request_seed,
            }
        )
        reference = references.get((repetition, session["user_id"], turn_index))
        finalize_request(record, profile, turn_index, reference)
        records.append(record)
    return records


def summarize_repetition(records: list[dict[str, Any]], origin: float) -> dict[str, Any]:
    makespan = max(record["response_complete_at_seconds"] for record in records)
    usages = [record["usage"] for record in records if isinstance(record.get("usage"), dict)]
    submitted_prompt_tokens = sum(record["submitted_prompt_tokens"] for record in records)
    prompt_tokens = sum(item.get("prompt_tokens", 0) for item in usages)
    completion_tokens = sum(item.get("completion_tokens", 0) for item in usages)
    actual_new = sum(
        record["actual_new_prefill_tokens"]
        for record in records
        if isinstance(record.get("actual_new_prefill_tokens"), int)
    )
    return {
        "benchmark_start_perf_counter": origin,
        "request_count": len(records),
        "makespan_seconds": makespan,
        "submitted_prompt_tokens": submitted_prompt_tokens,
        "usage_prompt_tokens": prompt_tokens,
        "actual_new_prefill_tokens": actual_new,
        "decode_tokens": completion_tokens,
        "submitted_prompt_throughput_tokens_per_second": submitted_prompt_tokens / makespan,
        "prompt_throughput_tokens_per_second": prompt_tokens / makespan,
        "actual_new_prefill_throughput_tokens_per_second": actual_new / makespan,
        "decode_throughput_tokens_per_second": completion_tokens / makespan,
        "measurement_failed_requests": sum(record["measurement_failed"] for record in records),
        "output_mismatch_requests": sum(record["output_mismatch"] is not None for record in records),
    }


def summarize_mode(records: list[dict[str, Any]], repetitions: list[dict[str, Any]]) -> dict[str, Any]:
    overall = [
        record["ttft_seconds"]
        for record in records
        if record["ttft_seconds"] is not None
    ]
    first = [record["ttft_seconds"] for record in records if record["turn_index"] == 0 and record["ttft_seconds"] is not None]
    later = [record["ttft_seconds"] for record in records if record["turn_index"] > 0 and record["ttft_seconds"] is not None]
    tpot = [record["tpot_seconds"] for record in records if record["tpot_seconds"] is not None]
    all_gaps = []
    true_token_gaps = []
    for record in records:
        timestamps = [event["at_seconds"] for event in record["nonempty_text_events"]]
        gaps = [later_at - earlier for earlier, later_at in zip(timestamps, timestamps[1:])]
        all_gaps.extend(gaps)
        if record["gap_interpretation"] == "inter-token gap":
            true_token_gaps.extend(gaps)
    return {
        "ttft_seconds": {
            "overall": distribution(overall, [50, 95]),
            "first_turn": distribution(first, [50, 95]),
            "later_turns": distribution(later, [50, 95]),
        },
        "tpot_seconds": distribution(tpot, [50, 95]),
        "sse_text_event_gap_seconds": {
            **distribution(all_gaps, [50, 95, 99]),
            "max": max(all_gaps) if all_gaps else None,
        },
        "inter_token_gap_seconds": (
            {**distribution(true_token_gaps, [50, 95, 99]), "max": max(true_token_gaps)}
            if true_token_gaps
            else None
        ),
        "makespan_seconds": distribution((item["makespan_seconds"] for item in repetitions), [50, 95]),
        "submitted_prompt_throughput_tokens_per_second": distribution(
            (item["submitted_prompt_throughput_tokens_per_second"] for item in repetitions), [50, 95]
        ),
        "prompt_throughput_tokens_per_second": distribution(
            (item["prompt_throughput_tokens_per_second"] for item in repetitions), [50, 95]
        ),
        "actual_new_prefill_throughput_tokens_per_second": distribution(
            (item["actual_new_prefill_throughput_tokens_per_second"] for item in repetitions), [50, 95]
        ),
        "decode_throughput_tokens_per_second": distribution(
            (item["decode_throughput_tokens_per_second"] for item in repetitions), [50, 95]
        ),
        "request_count": len(records),
        "measurement_failed_requests": sum(record["measurement_failed"] for record in records),
        "output_mismatch_requests": sum(record["output_mismatch"] is not None for record in records),
    }


class PublicServer:
    def __init__(
        self,
        command: list[str],
        gpu: str,
        base_url: str,
        timeout: float,
        readiness_prompt_text: str,
    ):
        self.command = command
        self.gpu = gpu
        self.base_url = base_url
        self.timeout = timeout
        self.readiness_prompt_text = readiness_prompt_text
        self.process: subprocess.Popen[bytes] | None = None
        self.log = tempfile.TemporaryFile(mode="w+b")
        self.measurement_log_offset: int | None = None

    def start(self) -> None:
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = self.gpu
        environment["PYTHONPATH"] = f"{REPO / 'python'}:{REPO}"
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=REPO,
            start_new_session=True,
        )
        self.wait_until_ready()

    def wait_until_ready(self) -> None:
        """Wait for a process already assigned to this server and warm its public API."""
        if self.process is None:
            raise RuntimeError("server process has not been launched")
        deadline = time.monotonic() + self.timeout
        url = self.base_url + "/v1/models"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with {self.process.returncode}: {self.log_tail()}")
            try:
                with urllib.request.urlopen(url, timeout=2) as response:
                    if response.status == 200:
                        break
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"/v1/models returned HTTP {exc.code}: {body}") from exc
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(0.25)
        else:
            raise TimeoutError(f"server did not listen within {self.timeout}s: {self.log_tail()}")

        self._wait_for_completion_ready(deadline)
        self._minimal_completion(retry_still_loading=True, deadline=deadline)

    def mark_measurement_start(self) -> None:
        self.log.flush()
        self.log.seek(0, os.SEEK_END)
        self.measurement_log_offset = self.log.tell()

    def layered_pipeline_waves(self) -> list[dict[str, int]]:
        if self.measurement_log_offset is None:
            return []
        self.log.flush()
        self.log.seek(self.measurement_log_offset)
        text = self.log.read().decode("utf-8", errors="replace")
        fields = (
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
        return [
            dict(zip(fields, (int(value) for value in match.groups())))
            for match in LAYERED_PIPELINE_WAVE_RE.finditer(text)
        ]

    def joint_waves(self) -> list[dict[str, int]]:
        if self.measurement_log_offset is None:
            return []
        self.log.flush()
        self.log.seek(self.measurement_log_offset)
        text = self.log.read().decode("utf-8", errors="replace")
        fields = (
            "chunks",
            "wave_reqs",
            "frontier_batches",
            "groups",
            "effective_group_size",
            "prefill_layer_prepares",
        )
        return [
            dict(zip(fields, (int(value) for value in match.groups())))
            for match in JOINT_WAVE_RE.finditer(text)
        ]

    def layered_prefill_waves(self) -> list[dict[str, int]]:
        if self.measurement_log_offset is None:
            return []
        self.log.flush()
        self.log.seek(self.measurement_log_offset)
        text = self.log.read().decode("utf-8", errors="replace")
        fields = (
            "reqs",
            "groups",
            "group_forwards",
            "iterations",
            "decode_iterations",
            "prefill_layer_prepares",
        )
        return [
            dict(zip(fields, (int(value) for value in match.groups())))
            for match in LAYERED_PREFILL_WAVE_RE.finditer(text)
        ]

    def moe_cache_stats_snapshots(self) -> list[dict[str, int]]:
        """Parse every cumulative idle snapshot, including readiness before measurement."""
        self.log.flush()
        self.log.seek(0)
        text = self.log.read().decode("utf-8", errors="replace")
        snapshots: list[dict[str, int]] = []
        for match in MOE_CACHE_STATS_RE.finditer(text):
            fields: dict[str, int] = {}
            for item in match.group(1).split(", "):
                name, value = item.split("=", 1)
                fields[name] = int(value)
            snapshots.append(fields)
        return snapshots

    def _completion_request(self) -> urllib.request.Request:
        payload = {
            "model": SERVED_MODEL,
            "prompt": self.readiness_prompt_text,
            "max_tokens": 1,
            "temperature": 0.0,
            "ignore_eos": True,
            "seed": 20260825,
            "stream": False,
        }
        return urllib.request.Request(
            self.base_url + "/v1/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _minimal_completion(self, retry_still_loading: bool, deadline: float) -> None:
        while True:
            if self.process is None or self.process.poll() is not None:
                return_code = None if self.process is None else self.process.returncode
                raise RuntimeError(f"server exited with {return_code}: {self.log_tail()}")
            try:
                with urllib.request.urlopen(self._completion_request(), timeout=60) as response:
                    body = response.read()
                    if response.status != 200:
                        raise RuntimeError(f"warmup completion returned HTTP {response.status}")
                    json.loads(body)
                    return
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                still_loading = exc.code == 503 and "still loading" in body.lower()
                if retry_still_loading and still_loading and time.monotonic() < deadline:
                    time.sleep(0.25)
                    continue
                raise RuntimeError(f"/v1/completions returned HTTP {exc.code}: {body}") from exc

    def _wait_for_completion_ready(self, deadline: float) -> None:
        self._minimal_completion(retry_still_loading=True, deadline=deadline)

    def stop(self) -> None:
        if self.process is None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        if self.process.poll() is not None:
            return
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=10)

    def log_tail(self) -> str:
        self.log.flush()
        self.log.seek(0, os.SEEK_END)
        size = self.log.tell()
        self.log.seek(max(0, size - 16384))
        return self.log.read().decode("utf-8", errors="replace")

    def close(self) -> None:
        self.stop()
        self.log.close()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def dry_run_plan(args: argparse.Namespace, workload: dict[str, Any], profile: dict[str, Any], modes: list[dict[str, Any]]) -> dict[str, Any]:
    model_path: str | Path = args.model if args.model else "<auto-generated-5-layer-qwen3-moe>"
    ft_executable = find_ft_executable(args.ft_executable)
    return {
        "dry_run": True,
        "workload": str(DEFAULT_WORKLOAD),
        "profile": args.profile,
        "repetitions": args.repetitions,
        "sessions": len(workload["sessions"]),
        "requests_per_repetition": workload["closed_loop_submission"]["total_requests"],
        "prompt_tokens_by_turn": profile["prompt_tokens_by_turn"],
        "requested_decode_tokens": profile["requested_decode_tokens"],
        "commands": [
            {
                "mode": mode["name"],
                "primary": mode["primary"],
                "argv": server_command(ft_executable, model_path, mode, args.host, args.port, workload),
            }
            for mode in modes
        ],
    }


def main() -> int:
    args = parse_args()
    if args.repetitions < 1:
        raise ValueError("--repetitions must be at least 1")
    workload = load_workload()
    if args.max_prefill_length is not None:
        workload["public_server_config"]["max_prefill_length"] = args.max_prefill_length
    profile = validate_workload(workload, args.profile)
    modes = resolve_modes(
        args.modes,
        workload,
        joint_groups=args.joint_groups,
        joint_waves=args.joint_waves,
    )
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, workload, profile, modes), indent=2))
        return 0

    ft_executable = find_ft_executable(args.ft_executable)
    owned_model_root: Path | None = None
    if args.model is None:
        owned_model_root = Path(tempfile.mkdtemp(prefix="freetoken-lab-agent-"))
        model_path = owned_model_root / "model"
        create_small_qwen3_moe(model_path)
    else:
        model_path = args.model.resolve()
    tokenizer = load_tokenizer(model_path)
    config = getattr(tokenizer, "model_max_length", None)
    required_first_tokens = args.repetitions * len(workload["sessions"])
    token_candidates = first_token_candidates(tokenizer, required_first_tokens + 1)
    isolation_tokens = token_candidates[:required_first_tokens]
    readiness_prompt_token_id, readiness_prompt_text = token_candidates[-1]
    continuation_pieces = continuation_token_pieces(tokenizer)
    prompts_by_repetition = [
        materialize_prompts(
            tokenizer,
            workload,
            profile,
            repetition,
            isolation_tokens,
            continuation_pieces,
        )
        for repetition in range(args.repetitions)
    ]
    del tokenizer

    output = args.output or (
        HERE / "results" / f"lab_agent_policies_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result: dict[str, Any] = {
        "schema": "freetoken.lab_agent_policy_benchmark.v1",
        "created_at_unix_seconds": time.time(),
        "workload": str(DEFAULT_WORKLOAD),
        "profile": args.profile,
        "model_path": str(model_path),
        "auto_generated_model": args.model is None,
        "auto_model_contract": {
            "architecture": "Qwen3MoeForCausalLM",
            "layers": 5,
            "experts_per_layer": 8,
            "experts_per_token": 2,
            "moe_intermediate_size": 1024,
            "max_position_embeddings": 8192,
        } if args.model is None else None,
        "tokenizer_model_max_length": config,
        "reference_mode": modes[0]["name"],
        "modes": [],
    }
    references: dict[tuple[int, str, int], str] = {}
    base_url = f"http://{args.host}:{args.port}"
    try:
        for mode_index, mode in enumerate(modes):
            command = server_command(ft_executable, model_path, mode, args.host, args.port, workload)
            mode_result: dict[str, Any] = {
                "name": mode["name"],
                "primary": mode["primary"],
                "server_command": command,
                "repetitions": [],
                "requests": [],
                "server_log_tail": None,
                "joint_waves": [],
                "layered_pipeline_waves": [],
                "layered_prefill_waves": [],
                "error": None,
                "readiness_prompt_token_id": readiness_prompt_token_id,
            }
            result["modes"].append(mode_result)
            server = PublicServer(
                command,
                args.gpu,
                base_url,
                args.server_timeout,
                readiness_prompt_text,
            )
            try:
                server.start()
                server.mark_measurement_start()
                for repetition in range(args.repetitions):
                    origin = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                        futures = [
                            executor.submit(
                                run_session,
                                session,
                                prompts_by_repetition[repetition][session["user_id"]],
                                profile,
                                workload,
                                base_url,
                                origin,
                                repetition,
                                references,
                            )
                            for session in workload["sessions"]
                        ]
                        repetition_records = [record for future in futures for record in future.result()]
                    repetition_records.sort(key=lambda item: (item["submitted_at_seconds"], item["user_index"], item["turn_index"]))
                    mode_result["requests"].extend(repetition_records)
                    mode_result["repetitions"].append(summarize_repetition(repetition_records, origin))
                    if mode_index == 0:
                        for record in repetition_records:
                            references[(repetition, record["user_id"], record["turn_index"])] = record["output_text"]
                mode_result["summary"] = summarize_mode(mode_result["requests"], mode_result["repetitions"])
                failed_measurements = mode_result["summary"]["measurement_failed_requests"]
                if failed_measurements:
                    write_json(output, result)
                    raise RuntimeError(
                        f"mode {mode['name']} has {failed_measurements} measurement-failed requests"
                    )
            except Exception as exc:
                mode_result["error"] = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                server.stop()
                if mode["batching_policy"] == "layered-pipeline":
                    mode_result["layered_pipeline_waves"] = server.layered_pipeline_waves()
                elif mode["batching_policy"] == "layered-prefill":
                    mode_result["layered_prefill_waves"] = server.layered_prefill_waves()
                elif mode["batching_policy"] == "joint":
                    mode_result["joint_waves"] = server.joint_waves()
                mode_result["server_log_tail"] = server.log_tail()
                server.close()
                write_json(output, result)
    finally:
        if owned_model_root is not None:
            shutil.rmtree(owned_model_root)
            result["model_path_removed_after_run"] = True
        write_json(output, result)
    print(output)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
