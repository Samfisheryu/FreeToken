#!/usr/bin/env python3
"""Replay real chat text under real request timing against FreeToken policies.

The manifest combines two public sources with complementary fields:

* WildChat supplies real, multi-turn user/assistant text.
* BurstGPT v2 supplies real conversation arrival and think-time traces.

The mapping is materialized once and then reused unchanged by every policy.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import io
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import tempfile
import threading
import time
from typing import Any, Iterable
import urllib.parse
import urllib.request

if __package__:
    from . import bench_lab_agent_policies as lab
    from . import bench_scaled_expert_contention as scaled
else:
    import bench_lab_agent_policies as lab
    import bench_scaled_expert_contention as scaled


HERE = Path(__file__).resolve().parent
DEFAULT_MODEL = Path("/data1/lmcache_kv/models/Qwen3.6-35B-A3B")
DEFAULT_EXPERT = Path(
    "/data1/lmcache_kv/nowag_qwen36_experiment/quantized/"
    "qwen36_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
)
WILDCHAT_DATASET = "allenai/WildChat"
WILDCHAT_SPLIT = "train"
WILDCHAT_TOTAL_ROWS = 529_000
WILDCHAT_BATCH_ROWS = 100
WILDCHAT_MAX_BATCHES = 220
BURSTGPT_URL = (
    "https://github.com/HPMLL/BurstGPT/releases/download/v2.0/"
    "BurstGPT_without_fails_3.csv"
)
BURSTGPT_RANGE_BYTES = 16 * 1024 * 1024
MAX_BURST_THINK_SECONDS = 600.0
DEFAULT_USER_COUNTS = (5, 10, 20)
DEFAULT_PROFILES = ("short", "natural", "long")
TURNS_PER_USER = 2
WARMUP_USERS = 5
MAX_PROMPT_TOKENS = 8192
MAX_SEQ_LEN = 16384
NUM_EXPERTS = 256
SERVED_MODEL = lab.SERVED_MODEL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--prepare-manifest",
        type=Path,
        help="write a fixed WildChat/BurstGPT manifest and exit",
    )
    source.add_argument("--manifest", type=Path, help="prepared manifest to replay")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    parser.add_argument(
        "--nowag-plugin-src",
        type=Path,
        help="optional pushed nowag_vllm src directory for the server subprocess",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18420)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--ft-executable")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=("legacy", "layered-pipeline-g1-wave64"),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        choices=DEFAULT_PROFILES,
        default=DEFAULT_PROFILES,
    )
    parser.add_argument(
        "--user-counts",
        nargs="+",
        type=int,
        default=DEFAULT_USER_COUNTS,
    )
    parser.add_argument("--time-compression", type=float, default=120.0)
    parser.add_argument("--response-token-cap", type=int, default=128)
    parser.add_argument("--moe-cache-size", type=int, default=2 * NUM_EXPERTS)
    parser.add_argument("--max-prefill-length", type=int, default=MAX_PROMPT_TOKENS)
    parser.add_argument("--max-running-requests", type=int, default=24)
    parser.add_argument("--num-tokens", type=int, default=269000)
    parser.add_argument("--cuda-graph-max-bs", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if any(count < 1 or count > 20 for count in args.user_counts):
        parser.error("--user-counts values must be between 1 and 20")
    if len(set(args.user_counts)) != len(args.user_counts):
        parser.error("--user-counts must not contain duplicates")
    if args.time_compression <= 0:
        parser.error("--time-compression must be positive")
    if args.request_timeout <= 0:
        parser.error("--request-timeout must be positive")
    if args.response_token_cap < 1:
        parser.error("--response-token-cap must be at least 1")
    if args.max_running_requests < max(args.user_counts):
        parser.error("--max-running-requests must cover the largest user count")
    if args.max_prefill_length < 1:
        parser.error("--max-prefill-length must be at least 1")
    return args


def fetch_json(url: str, attempts: int = 3) -> dict[str, Any]:
    error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                return json.load(response)
        except Exception as exc:
            error = exc
            if attempt + 1 < attempts:
                time.sleep(0.5)
    assert error is not None
    raise error


def wildchat_offsets() -> list[int]:
    last = WILDCHAT_TOTAL_ROWS - WILDCHAT_BATCH_ROWS
    return [
        round(index * last / (WILDCHAT_MAX_BATCHES - 1))
        for index in range(WILDCHAT_MAX_BATCHES)
    ]


def wildchat_rows(offset: int) -> list[dict[str, Any]]:
    query = urllib.parse.urlencode(
        {
            "dataset": WILDCHAT_DATASET,
            "config": "default",
            "split": WILDCHAT_SPLIT,
            "offset": offset,
            "length": WILDCHAT_BATCH_ROWS,
        }
    )
    payload = fetch_json("https://datasets-server.huggingface.co/rows?" + query)
    return [
        {"row_index": item["row_idx"], **item["row"]}
        for item in payload["rows"]
    ]


def encode_length(tokenizer: Any, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def conversation_turns(tokenizer: Any, row: dict[str, Any]) -> list[dict[str, Any]]:
    if row.get("language") != "English" or row.get("toxic"):
        return []
    history: list[dict[str, str]] = []
    turns: list[dict[str, Any]] = []
    for raw_message in row.get("conversation") or []:
        role = raw_message.get("role")
        content = raw_message.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            return []
        if not content.strip():
            return []
        message = {"role": role, "content": content}
        if role == "assistant" and history and history[-1]["role"] == "user":
            prompt = tokenizer.apply_chat_template(
                history,
                tokenize=False,
                add_generation_prompt=True,
            )
            prompt_tokens = encode_length(tokenizer, prompt)
            response_tokens = encode_length(tokenizer, content)
            if prompt_tokens > 0 and response_tokens > 0:
                turns.append(
                    {
                        "prompt_text": prompt,
                        "prompt_tokens": prompt_tokens,
                        "source_response_tokens": response_tokens,
                    }
                )
        history.append(message)
    return turns


def two_turn_windows(turns: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    return [
        turns[end - TURNS_PER_USER : end]
        for end in range(TURNS_PER_USER, len(turns) + 1)
        if all(turn["prompt_tokens"] <= MAX_PROMPT_TOKENS for turn in turns[end - 2 : end])
    ]


def choose_profile_window(
    profile: str,
    windows: list[list[dict[str, Any]]],
) -> list[dict[str, Any]] | None:
    if not windows:
        return None
    if profile == "natural":
        return windows[-1]
    bounds = {
        "short": (1, 512),
        "long": (4096, MAX_PROMPT_TOKENS + 1),
    }
    low, high = bounds[profile]
    return next(
        (
            window
            for window in windows
            if low <= window[-1]["prompt_tokens"] < high
        ),
        None,
    )


def required_sessions_per_profile(user_counts: Iterable[int]) -> int:
    return WARMUP_USERS + sum(user_counts)


def collect_wildchat_profiles(
    tokenizer: Any,
    user_counts: tuple[int, ...],
) -> tuple[dict[str, list[dict[str, Any]]], list[int]]:
    required = required_sessions_per_profile(user_counts)
    selected: dict[str, list[dict[str, Any]]] = {
        profile: [] for profile in DEFAULT_PROFILES
    }
    seen: dict[str, set[str]] = {profile: set() for profile in DEFAULT_PROFILES}
    used_offsets: list[int] = []
    for offset in wildchat_offsets():
        used_offsets.append(offset)
        for row in wildchat_rows(offset):
            conversation_id = row["conversation_id"]
            windows = two_turn_windows(conversation_turns(tokenizer, row))
            for profile in DEFAULT_PROFILES:
                if len(selected[profile]) >= required:
                    continue
                if conversation_id in seen[profile]:
                    continue
                window = choose_profile_window(profile, windows)
                if window is None:
                    continue
                seen[profile].add(conversation_id)
                selected[profile].append(
                    {
                        "conversation_id": conversation_id,
                        "wildchat_row_index": row["row_index"],
                        "wildchat_timestamp": row["timestamp"],
                        "turns": window,
                    }
                )
        if all(len(items) >= required for items in selected.values()):
            return selected, used_offsets
    counts = {profile: len(items) for profile, items in selected.items()}
    raise RuntimeError(
        f"WildChat fixed-offset sample did not contain {required} sessions per profile: "
        f"{counts}"
    )


def fetch_burstgpt_rows() -> list[dict[str, Any]]:
    request = urllib.request.Request(
        BURSTGPT_URL,
        headers={"Range": f"bytes=0-{BURSTGPT_RANGE_BYTES - 1}"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        raw = response.read()
    raw = raw[: raw.rfind(b"\n") + 1]
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))


def burstgpt_schedules(required: int = 20) -> list[dict[str, Any]]:
    sessions: dict[str, list[dict[str, Any]]] = {}
    for row in fetch_burstgpt_rows():
        if row.get("Log Type") != "Conversation log" or not row.get("Session ID"):
            continue
        sessions.setdefault(row["Session ID"], []).append(row)
    eligible: list[tuple[float, str, list[dict[str, Any]]]] = []
    for session_id, rows in sessions.items():
        ordered = sorted(rows, key=lambda item: float(item["Timestamp"]))
        if len(ordered) < TURNS_PER_USER:
            continue
        used = ordered[:TURNS_PER_USER]
        think_seconds = [
            max(
                0.0,
                float(following["Timestamp"])
                - float(current["Timestamp"])
                - float(current["Elapsed time"]),
            )
            for current, following in zip(used, used[1:])
        ]
        if max(think_seconds, default=0.0) <= MAX_BURST_THINK_SECONDS:
            eligible.append((float(ordered[0]["Timestamp"]), session_id, ordered))
    eligible.sort()
    if len(eligible) < required:
        raise RuntimeError(f"BurstGPT range contains only {len(eligible)} eligible sessions")
    start = min(
        range(len(eligible) - required + 1),
        key=lambda index: eligible[index + required - 1][0] - eligible[index][0],
    )
    chosen = eligible[start : start + required]
    origin = chosen[0][0]
    schedules: list[dict[str, Any]] = []
    for first_timestamp, session_id, rows in chosen:
        used = rows[:TURNS_PER_USER]
        think_seconds: list[float] = []
        for current, following in zip(used, used[1:]):
            gap = float(following["Timestamp"]) - float(current["Timestamp"])
            elapsed = float(current["Elapsed time"])
            think_seconds.append(max(0.0, gap - elapsed))
        schedules.append(
            {
                "burstgpt_session_id": session_id,
                "start_offset_seconds": first_timestamp - origin,
                "think_seconds": think_seconds,
                "source_requests": [
                    {
                        "timestamp": float(row["Timestamp"]),
                        "elapsed_seconds": float(row["Elapsed time"]),
                        "request_tokens": int(row["Request tokens"]),
                        "response_tokens": int(row["Response tokens"]),
                        "model": row["Model"],
                    }
                    for row in used
                ],
            }
        )
    return schedules


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def prepare_manifest(path: Path, model: Path, user_counts: tuple[int, ...]) -> None:
    tokenizer = lab.load_tokenizer(model)
    profiles, offsets = collect_wildchat_profiles(tokenizer, user_counts)
    del tokenizer
    payload = {
        "schema": "freetoken.real_conversation_manifest.v1",
        "created_at_unix_seconds": time.time(),
        "sources": {
            "wildchat": {
                "dataset": WILDCHAT_DATASET,
                "split": WILDCHAT_SPLIT,
                "dataset_server_offsets": offsets,
                "batch_rows": WILDCHAT_BATCH_ROWS,
            },
            "burstgpt": {
                "url": BURSTGPT_URL,
                "byte_range": [0, BURSTGPT_RANGE_BYTES - 1],
                "max_active_session_think_seconds": MAX_BURST_THINK_SECONDS,
            },
        },
        "model_tokenizer": str(model.resolve()),
        "turns_per_user": TURNS_PER_USER,
        "user_counts": list(user_counts),
        "warmup_users": WARMUP_USERS,
        "profiles": profiles,
        "arrival_schedules": burstgpt_schedules(max(user_counts)),
    }
    write_json(path, payload)


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "freetoken.real_conversation_manifest.v1":
        raise ValueError(f"unsupported manifest schema in {path}")
    return manifest


def parse_mode(value: str) -> dict[str, Any]:
    normalized = value.lower().replace("_", "-")
    if normalized == "legacy":
        return {"name": "legacy", "batching_policy": "legacy"}
    if normalized == "mixed":
        return {"name": "mixed", "batching_policy": "mixed"}
    parts = normalized.split("-")
    if len(parts) == 4 and parts[:2] == ["layered", "pipeline"]:
        group_text, wave_text = parts[2], parts[3]
        if not group_text.startswith("g") or not wave_text.startswith("wave"):
            raise ValueError(f"invalid mode {value!r}")
        group = int(group_text[1:])
        wave = int(wave_text[4:])
        if group < 1 or wave < 1:
            raise ValueError(f"invalid mode {value!r}")
        return {
            "name": f"layered_pipeline_g{group}_wave{wave}",
            "batching_policy": "layered-pipeline",
            "prefill_layer_group_size": group,
            "prefill_wave_max_chunks": wave,
        }
    raise ValueError(f"unsupported mode {value!r}")


def server_workload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "public_server_config": {
            "max_running_requests": args.max_running_requests,
            "max_seq_len_override": MAX_SEQ_LEN,
            "max_prefill_length": args.max_prefill_length,
            "dtype": "bfloat16",
            "attention_backend": "triton",
            "moe_cache_size": args.moe_cache_size,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "cache_type": "radix",
        }
    }


def server_command(
    args: argparse.Namespace,
    executable: str,
    mode: dict[str, Any],
) -> list[str]:
    return lab.server_command(
        executable,
        args.model.resolve(),
        mode,
        args.host,
        args.port,
        server_workload(args),
    ) + [
        "--nowag-expert-path",
        str(args.expert.resolve()),
        "--num-tokens",
        str(args.num_tokens),
        "--moe-collect-stats",
    ]


def wait_until(target: float) -> None:
    while True:
        remaining = target - time.perf_counter()
        if remaining <= 0:
            return
        time.sleep(min(remaining, 0.05))


def finish_record(
    record: dict[str, Any],
    session: dict[str, Any],
    user_index: int,
    turn_index: int,
    expected_decode: int,
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
        completion = usage.get("completion_tokens") if isinstance(usage, dict) else None
        record["tpot_seconds"] = (
            (record["response_complete_at_seconds"] - events[0]["at_seconds"])
            / (completion - 1)
            if isinstance(completion, int) and completion > 1
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
        if usage.get("prompt_tokens") != record["submitted_prompt_tokens"]:
            failures.append("prompt usage mismatch")
        if usage.get("completion_tokens") != expected_decode:
            failures.append("completion usage mismatch")
    record.update(
        {
            "conversation_id": session["conversation_id"],
            "wildchat_row_index": session["wildchat_row_index"],
            "user_index": user_index,
            "turn_index": turn_index,
            "source_response_tokens": session["turns"][turn_index][
                "source_response_tokens"
            ],
            "requested_completion_tokens": expected_decode,
            "latency_seconds": (
                record["response_complete_at_seconds"] - record["submitted_at_seconds"]
            ),
            "measurement_failed": bool(failures),
            "measurement_failures": failures,
        }
    )


def run_user(
    base_url: str,
    session: dict[str, Any],
    schedule: dict[str, Any],
    user_index: int,
    origin: float,
    time_compression: float,
    response_cap: int,
    seed_base: int,
    request_timeout: float,
) -> list[dict[str, Any]]:
    wait_until(origin + schedule["start_offset_seconds"] / time_compression)
    records: list[dict[str, Any]] = []
    for turn_index, turn in enumerate(session["turns"]):
        expected_decode = min(turn["source_response_tokens"], response_cap)
        record = lab.request_completion(
            base_url,
            turn["prompt_text"],
            turn["prompt_tokens"],
            expected_decode,
            seed_base + user_index * 10 + turn_index,
            origin,
            request_timeout=request_timeout,
        )
        finish_record(record, session, user_index, turn_index, expected_decode)
        records.append(record)
        if turn_index < len(session["turns"]) - 1:
            time.sleep(schedule["think_seconds"][turn_index] / time_compression)
    return records


def run_case(
    base_url: str,
    sessions: list[dict[str, Any]],
    schedules: list[dict[str, Any]],
    time_compression: float,
    response_cap: int,
    seed_base: int,
    request_timeout: float,
) -> tuple[list[dict[str, Any]], float]:
    origin = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(sessions)) as executor:
        futures = [
            executor.submit(
                run_user,
                base_url,
                session,
                schedules[index],
                index,
                origin,
                time_compression,
                response_cap,
                seed_base,
                request_timeout,
            )
            for index, session in enumerate(sessions)
        ]
        records = [record for future in futures for record in future.result()]
    records.sort(key=lambda item: (item["submitted_at_seconds"], item["user_index"]))
    return records, time.perf_counter() - origin


def percentile(values: Iterable[float], percent: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percent / 100
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - position) + ordered[high] * (position - low)


def concurrency_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    events: list[tuple[float, int]] = []
    for record in records:
        events.append((record["submitted_at_seconds"], 1))
        events.append((record["response_complete_at_seconds"], -1))
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    area = 0.0
    previous = events[0][0]
    for timestamp, delta in events:
        area += active * (timestamp - previous)
        active += delta
        peak = max(peak, active)
        previous = timestamp
    span = events[-1][0] - events[0][0]
    return {
        "peak_inflight_requests": peak,
        "average_inflight_requests": area / span if span > 0 else 0.0,
    }


def summarize_case(records: list[dict[str, Any]], makespan: float) -> dict[str, Any]:
    good = [record for record in records if not record["measurement_failed"]]
    prompt_tokens = sum(record["usage"]["prompt_tokens"] for record in good)
    decode_tokens = sum(record["usage"]["completion_tokens"] for record in good)
    result: dict[str, Any] = {
        "request_count": len(records),
        "measurement_failed_requests": len(records) - len(good),
        "prompt_tokens": prompt_tokens,
        "decode_tokens": decode_tokens,
        "makespan_seconds": makespan,
        "prompt_throughput_tokens_per_second": prompt_tokens / makespan,
        "decode_throughput_tokens_per_second": decode_tokens / makespan,
        **concurrency_summary(records),
    }
    for field in ("ttft_seconds", "tpot_seconds", "latency_seconds", "max_sse_gap_seconds"):
        values = [record[field] for record in good if record.get(field) is not None]
        result[field] = {
            "p50": percentile(values, 50),
            "p95": percentile(values, 95),
        }
    return result


def wait_for_settled_snapshot(
    server: lab.PublicServer,
    prior_count: int,
    timeout: float,
) -> list[dict[str, int]]:
    deadline = time.monotonic() + timeout
    last_count = -1
    stable_polls = 0
    while time.monotonic() < deadline:
        snapshots = server.moe_cache_stats_snapshots()
        count = len(snapshots)
        if count > prior_count:
            stable_polls = stable_polls + 1 if count == last_count else 0
            if stable_polls >= 3:
                return snapshots
        last_count = count
        if server.process is not None and server.process.poll() is not None:
            raise RuntimeError(f"server exited before idle snapshot: {server.log_tail()}")
        time.sleep(0.1)
    raise TimeoutError("MoE idle snapshot did not settle")


class BenchmarkServer(lab.PublicServer):
    def __init__(self, *args: Any, nowag_plugin_src: Path | None, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.nowag_plugin_src = nowag_plugin_src

    def start(self) -> None:
        if self.nowag_plugin_src is None:
            super().start()
            return
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = self.gpu
        environment["PYTHONPATH"] = ":".join(
            (
                str(self.nowag_plugin_src.resolve()),
                str(lab.REPO / "python"),
                str(lab.REPO),
            )
        )
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log,
            stderr=subprocess.STDOUT,
            env=environment,
            cwd=lab.REPO,
            start_new_session=True,
        )
        self.wait_until_ready()


def log_offset(server: lab.PublicServer) -> int:
    server.log.flush()
    server.log.seek(0, io.SEEK_END)
    return server.log.tell()


def pipeline_waves_since(server: lab.PublicServer, offset: int) -> list[dict[str, int]]:
    server.log.flush()
    server.log.seek(offset)
    text = server.log.read().decode("utf-8", errors="replace")
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
        for match in lab.LAYERED_PIPELINE_WAVE_RE.finditer(text)
    ]


def profile_session_slices(
    profile: list[dict[str, Any]],
    user_counts: tuple[int, ...],
) -> tuple[list[dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    warmup = profile[:WARMUP_USERS]
    cursor = WARMUP_USERS
    cases: dict[int, list[dict[str, Any]]] = {}
    for count in user_counts:
        cases[count] = profile[cursor : cursor + count]
        cursor += count
    if len(warmup) != WARMUP_USERS or any(len(cases[n]) != n for n in user_counts):
        raise ValueError("manifest profile does not contain enough disjoint sessions")
    return warmup, cases


def dry_run_plan(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    modes: list[dict[str, Any]],
) -> dict[str, Any]:
    executable = lab.find_ft_executable(args.ft_executable)
    return {
        "dry_run": True,
        "schema": "freetoken.real_conversation_concurrency.v1",
        "manifest": str(args.manifest),
        "profiles": list(args.profiles),
        "user_counts": list(args.user_counts),
        "turns_per_user": manifest["turns_per_user"],
        "commands": [server_command(args, executable, mode) for mode in modes],
    }


def main() -> int:
    args = parse_args()
    user_counts = tuple(args.user_counts)
    if args.prepare_manifest is not None:
        prepare_manifest(args.prepare_manifest, args.model.resolve(), user_counts)
        print(args.prepare_manifest)
        return 0

    assert args.manifest is not None
    manifest = load_manifest(args.manifest)
    manifest_user_counts = tuple(manifest["user_counts"])
    missing_user_counts = [
        count for count in user_counts if count not in manifest_user_counts
    ]
    if missing_user_counts:
        raise ValueError(
            f"requested user counts {missing_user_counts} are not present in the "
            f"manifest; available counts: {list(manifest_user_counts)}"
        )
    modes = [parse_mode(value) for value in args.modes]
    if args.dry_run:
        print(json.dumps(dry_run_plan(args, manifest, modes), indent=2))
        return 0
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.expert.exists():
        raise FileNotFoundError(args.expert)
    executable = lab.find_ft_executable(args.ft_executable)
    output = args.output or (
        HERE / "results" / f"real_conversation_concurrency_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    result: dict[str, Any] = {
        "schema": "freetoken.real_conversation_concurrency.v1",
        "created_at_unix_seconds": time.time(),
        "manifest": str(args.manifest.resolve()),
        "model": str(args.model.resolve()),
        "expert": str(args.expert.resolve()),
        "config": {
            "profiles": list(args.profiles),
            "user_counts": list(user_counts),
            "turns_per_user": TURNS_PER_USER,
            "time_compression": args.time_compression,
            "response_token_cap": args.response_token_cap,
            "moe_cache_size": args.moe_cache_size,
            "max_prefill_length": args.max_prefill_length,
            "max_running_requests": args.max_running_requests,
            "num_tokens": args.num_tokens,
            "request_timeout": args.request_timeout,
            "cuda_graph_max_bs": args.cuda_graph_max_bs,
            "attention_backend": "triton",
            "dtype": "bfloat16",
            "nowag_plugin_src": (
                str(args.nowag_plugin_src.resolve())
                if args.nowag_plugin_src is not None
                else None
            ),
        },
        "runs": [],
    }
    references: dict[tuple[str, int, str, int], str] = {}
    schedules = manifest["arrival_schedules"]
    for profile_index, profile_name in enumerate(args.profiles):
        warmup_sessions, manifest_cases = profile_session_slices(
            manifest["profiles"][profile_name], manifest_user_counts
        )
        cases = {count: manifest_cases[count] for count in user_counts}
        for mode in modes:
            run: dict[str, Any] = {
                "profile": profile_name,
                "mode": mode["name"],
                "server_command": server_command(args, executable, mode),
                "cases": [],
                "error": None,
            }
            result["runs"].append(run)
            base_url = f"http://{args.host}:{args.port}"
            readiness = "Real workload benchmark readiness probe; answer briefly."
            server = BenchmarkServer(
                run["server_command"],
                args.gpu,
                base_url,
                args.server_timeout,
                readiness,
                nowag_plugin_src=args.nowag_plugin_src,
            )
            try:
                server.start()
                snapshots = scaled.wait_for_snapshot_count(
                    server, 1, args.server_timeout
                )
                warmup_records, _ = run_case(
                    base_url,
                    warmup_sessions,
                    [
                        {
                            "start_offset_seconds": 0.0,
                            "think_seconds": [0.0],
                        }
                        for _ in warmup_sessions
                    ],
                    args.time_compression,
                    min(args.response_token_cap, 32),
                    7_000_000 + profile_index * 1000,
                    args.request_timeout,
                )
                if any(record["measurement_failed"] for record in warmup_records):
                    raise RuntimeError("profile warmup request failed")
                snapshots = wait_for_settled_snapshot(
                    server, len(snapshots), args.server_timeout
                )
                baseline = snapshots[-1]
                snapshot_count = len(snapshots)
                offset = log_offset(server)
                for case_index, count in enumerate(user_counts):
                    records, makespan = run_case(
                        base_url,
                        cases[count],
                        schedules[:count],
                        args.time_compression,
                        args.response_token_cap,
                        8_000_000
                        + profile_index * 100_000
                        + case_index * 1000,
                        args.request_timeout,
                    )
                    final_snapshots = wait_for_settled_snapshot(
                        server, snapshot_count, args.server_timeout
                    )
                    after = final_snapshots[-1]
                    waves = (
                        pipeline_waves_since(server, offset)
                        if mode["batching_policy"] == "layered-pipeline"
                        else []
                    )
                    for record in records:
                        key = (
                            profile_name,
                            count,
                            record["conversation_id"],
                            record["turn_index"],
                        )
                        reference = references.get(key)
                        record["output_mismatch"] = (
                            None
                            if reference is None
                            else lab.first_difference(reference, record["output_text"])
                        )
                        if reference is None:
                            references[key] = record["output_text"]
                    case = {
                        "configured_users": count,
                        "requests": records,
                        "summary": summarize_case(records, makespan),
                        "moe_stats_before": baseline,
                        "moe_stats_after": after,
                        "moe_stats_delta": scaled.snapshot_delta(baseline, after),
                        "layered_pipeline_waves": waves,
                    }
                    run["cases"].append(case)
                    if case["summary"]["measurement_failed_requests"]:
                        raise RuntimeError(
                            f"{profile_name}/{mode['name']}/{count} users has failed requests"
                        )
                    baseline = after
                    snapshot_count = len(final_snapshots)
                    offset = log_offset(server)
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
