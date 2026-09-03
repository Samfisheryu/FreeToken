#!/usr/bin/env python3
"""Measure frozen long-context mixed online task goodput on one FreeToken arm.

The runner warms one server with twenty fixed requests, preserves the resulting
cache, and then drives twenty closed-loop users.  The measured denominator is
from user 0's first accepted submission through the maximum of fixed window
close, the last submitted request terminal, and the first public server-idle
acknowledgement.  SWE-bench Docker scoring begins only after that denominator
is closed.

``--server-args`` consumes the rest of the command line and must be last.
"""

from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Iterable

if __package__:
    from . import bench_aime_task_goodput as serving
    from . import bench_mixed_task_goodput as runtime
else:
    import bench_aime_task_goodput as serving
    import bench_mixed_task_goodput as runtime


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = Path(
    "/data1/lmcache_kv/goodput_campaign/long_mixed_workload_v2/manifest.json"
)
MANIFEST_SCHEMA = "freetoken.long_mixed_task_goodput_manifest.v2"
TASK_SCHEMA = "freetoken.long_mixed_task_goodput_task.v2"
RESULT_SCHEMA = "freetoken.long_mixed_task_goodput_result.v2"
FAMILIES = ("numeric", "code", "knowledge")
TARGETS = (8192, 16384, 32768)
SYSTEM_PROMPT = "Follow the requested output format exactly."
HARD_REQUEST_TIMEOUT = 210.0
SLO_SECONDS = {"numeric": 90.0, "code": 180.0, "knowledge": 60.0}
TOKEN_CAPS = {"numeric": 128, "code": 2048, "knowledge": 32}
NUMERIC_RE = re.compile(
    r"\s*FINAL:\s*(\$?\s*[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?\s*%?)\s*\Z"
)
KNOWLEDGE_RE = re.compile(r"\s*FINAL:\s*([ABCD])\s*\Z")
DIFF_FIRST_RE = re.compile(r"(?:diff --git a/\S+ b/\S+|--- (?:a/)?\S+)")
SWE_RESULT_SCHEMA = "freetoken.swebench_verified_subset_result.v2"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--arm-name", required=True)
    parser.add_argument("--model", required=True, help="local checkpoint, FTW, or HF model id")
    parser.add_argument("--stream", choices=("dev", "final"), default="dev")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--freetoken-root", type=Path, default=REPO)
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--pythonpath", action="append", default=[], metavar="PATH")
    parser.add_argument("--env", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout", type=float, default=90.0)
    parser.add_argument("--numeric-max-tokens", type=int, default=128)
    parser.add_argument("--code-max-tokens", type=int, default=2048)
    parser.add_argument("--knowledge-max-tokens", type=int, default=32)
    parser.add_argument("--seed-base", type=int, default=20260902)
    parser.add_argument(
        "--swebench-python",
        default=sys.executable,
        help="Python executable containing dependencies for SWE-bench v5.0.1",
    )
    parser.add_argument(
        "--swebench-root",
        type=Path,
        required=True,
        help=(
            "official SWE-bench checkout at commit "
            "87ab1f6ced28f75ba73ca899dc759b019310944a"
        ),
    )
    parser.add_argument("--swebench-max-workers", type=int, default=2)
    parser.add_argument("--swebench-instance-timeout", type=int, default=1800)
    parser.add_argument("--swebench-scorer-timeout", type=float, default=21600.0)
    parser.add_argument("--stats-poll-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "remaining arguments for `ft serve`; must include exactly "
            "--max-seq-len-override 40960 and this option must be last"
        ),
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in (
        "server_timeout",
        "shutdown_timeout",
        "swebench_scorer_timeout",
        "stats_poll_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.swebench_max_workers < 1 or args.swebench_instance_timeout < 1:
        parser.error("SWE-bench worker count and instance timeout must be positive")
    for family, cap in TOKEN_CAPS.items():
        value = getattr(args, f"{family}_max_tokens")
        if not 1 <= value <= cap:
            parser.error(f"--{family}-max-tokens must be in 1..{cap}")
    args.request_timeout = HARD_REQUEST_TIMEOUT
    args.server_env = serving.parse_environment(args.env, parser)
    serving.validate_server_args(args.server_args, parser)
    validate_max_sequence_args(args.server_args, parser)
    return args


def validate_max_sequence_args(
    server_args: list[str], parser: argparse.ArgumentParser
) -> None:
    values: list[str] = []
    index = 0
    while index < len(server_args):
        token = server_args[index]
        if token == "--max-seq-len-override":
            if index + 1 >= len(server_args):
                parser.error("--max-seq-len-override needs value 40960")
            values.append(server_args[index + 1])
            index += 2
            continue
        if token.startswith("--max-seq-len-override="):
            values.append(token.split("=", 1)[1])
        index += 1
    if values != ["40960"]:
        parser.error("--server-args must contain exactly one --max-seq-len-override 40960")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported long mixed workload manifest: {path}")
    traffic = manifest.get("traffic")
    policy = manifest.get("request_policy")
    if not isinstance(traffic, dict) or not isinstance(policy, dict):
        raise ValueError("manifest needs traffic and request_policy objects")
    expected_traffic = {
        "user_count": 20,
        "first_submission_stagger_seconds": 0.5,
        "think_time_seconds": 2.0,
        "hard_request_timeout_seconds": 210,
    }
    for field, expected in expected_traffic.items():
        if traffic.get(field) != expected:
            raise ValueError(f"traffic.{field} must remain {expected}")
    for split, expected in (
        ("dev", (120, 12, 80, {"8192": 32, "16384": 32, "32768": 16})),
        ("final", (180, 51, 340, {"8192": 136, "16384": 136, "32768": 68})),
    ):
        row = traffic.get(split)
        actual = (
            row.get("submission_window_seconds") if isinstance(row, dict) else None,
            row.get("turns_per_user") if isinstance(row, dict) else None,
            row.get("tasks_per_family") if isinstance(row, dict) else None,
            row.get("length_bucket_counts_per_family") if isinstance(row, dict) else None,
        )
        if actual != expected:
            raise ValueError(f"traffic.{split} differs from the frozen contract")
    if policy.get("system_prompt") != SYSTEM_PROMPT:
        raise ValueError("the frozen common system prompt changed")
    if policy.get("greedy") is not True or policy.get("enable_thinking") is not False:
        raise ValueError("v2 requires greedy decoding with thinking disabled")
    if policy.get("max_sequence_length") != 40960:
        raise ValueError("v2 max sequence length must remain 40960")
    families = policy.get("families")
    for family in FAMILIES:
        row = families.get(family) if isinstance(families, dict) else None
        if not isinstance(row, dict) or (
            row.get("max_tokens_cap"), row.get("slo_seconds")
        ) != (TOKEN_CAPS[family], int(SLO_SECONDS[family])):
            raise ValueError(f"{family} cap/SLO changed")
    tokenizer = manifest.get("tokenizer")
    if not isinstance(tokenizer, dict) or not isinstance(tokenizer.get("identifier"), str):
        raise ValueError("manifest tokenizer contract is missing")
    if tokenizer.get("chat_template_kwargs") != {"enable_thinking": False}:
        raise ValueError("manifest tokenizer thinking policy changed")
    evaluator = manifest.get("evaluators", {}).get("swebench")
    if not isinstance(evaluator, dict) or (
        evaluator.get("version"), evaluator.get("revision")
    ) != ("5.0.1", "87ab1f6ced28f75ba73ca899dc759b019310944a"):
        raise ValueError("manifest SWE-bench evaluator pin changed")
    return manifest


def resolve_task_path(
    manifest_path: Path, manifest: dict[str, Any], split: str
) -> Path:
    relative = manifest.get("task_files", {}).get(split)
    if not isinstance(relative, str):
        raise ValueError(f"manifest omits task_files.{split}")
    path = (manifest_path.parent / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_tasks(path: Path, split: str) -> dict[str, dict[str, Any]]:
    tasks: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != TASK_SCHEMA:
                raise ValueError(f"unsupported task row at {path}:{line_number}")
            task_id = row.get("task_id")
            family = row.get("family")
            bucket = row.get("length_bucket_tokens")
            prompt_tokens = row.get("prompt_tokens")
            if not isinstance(task_id, str) or not task_id or task_id in tasks:
                raise ValueError(f"invalid/duplicate task_id at {path}:{line_number}")
            if family not in FAMILIES or row.get("split") != split:
                raise ValueError(f"invalid task family/split at {path}:{line_number}")
            if bucket not in TARGETS or not isinstance(prompt_tokens, int):
                raise ValueError(f"invalid task length bucket at {path}:{line_number}")
            if not math.ceil(bucket * 0.9) <= prompt_tokens <= bucket:
                raise ValueError(f"task {task_id} falls outside its 90-100% token bucket")
            if not isinstance(row.get("task_text"), str) or not row["task_text"].strip():
                raise ValueError(f"task {task_id} has no frozen prompt")
            reference = row.get("reference")
            expected_kind = {
                "numeric": "finqa_numeric",
                "code": "swebench_verified",
                "knowledge": "choice",
            }[family]
            if not isinstance(reference, dict) or reference.get("kind") != expected_kind:
                raise ValueError(f"task {task_id} reference kind changed")
            tasks[task_id] = row
    if not tasks:
        raise ValueError(f"empty task stream: {path}")
    return tasks


def validate_assignments(
    manifest: dict[str, Any], split: str, tasks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    users = manifest.get("assignments", {}).get(split)
    if not isinstance(users, list) or len(users) != 20:
        raise ValueError(f"{split} assignments must contain 20 users")
    turns = 1 if split == "warmup" else int(manifest["traffic"][split]["turns_per_user"])
    assigned: list[str] = []
    for user_index, user in enumerate(users):
        if not isinstance(user, dict) or user.get("user_index") != user_index:
            raise ValueError(f"{split} user indices must be ordered 0..19")
        task_ids = user.get("task_ids")
        if not isinstance(task_ids, list) or len(task_ids) != turns:
            raise ValueError(f"{split} user {user_index} must have {turns} tasks")
        for turn_index, task_id in enumerate(task_ids):
            if task_id not in tasks:
                raise ValueError(f"{split} assignment references unknown task {task_id!r}")
            expected = FAMILIES[(user_index + turn_index) % 3]
            if tasks[task_id]["family"] != expected:
                raise ValueError(
                    f"{split} user {user_index} turn {turn_index} expected {expected}"
                )
        assigned.extend(task_ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(tasks):
        raise ValueError(f"{split} assignments must consume every task exactly once")
    if split != "warmup":
        expected_counts = manifest["traffic"][split]["length_bucket_counts_per_family"]
        for family in FAMILIES:
            actual = Counter(
                str(tasks[task_id]["length_bucket_tokens"])
                for task_id in assigned
                if tasks[task_id]["family"] == family
            )
            if dict(actual) != expected_counts:
                raise ValueError(f"{split} {family} bucket counts changed: {dict(actual)}")
    return users


def prepare_prompts(
    tasks: Iterable[dict[str, Any]], token_counter: Any
) -> None:
    for task in tasks:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": task["task_text"]},
        ]
        observation = token_counter.prompt_observation(
            messages, {"chat_template_kwargs": {"enable_thinking": False}}
        )
        if observation.get("estimated") or observation.get("tokens") != task["prompt_tokens"]:
            raise ValueError(
                f"task {task['task_id']} client token count differs from frozen exact count: "
                f"{observation.get('tokens')} vs {task['prompt_tokens']}"
            )
        task["messages"] = messages
        task["prompt_token_observation"] = observation


def max_tokens_by_family(args: argparse.Namespace) -> dict[str, int]:
    return {family: int(getattr(args, f"{family}_max_tokens")) for family in FAMILIES}


def wait_until(target: float) -> None:
    remaining = target - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def run_warmup(
    *,
    args: argparse.Namespace,
    users: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    model_id: str,
    token_caps: dict[str, int],
) -> list[dict[str, Any]]:
    origin = time.perf_counter()

    def one(user: dict[str, Any]) -> dict[str, Any]:
        user_index = int(user["user_index"])
        scheduled = origin + user_index * 0.5
        wait_until(scheduled)
        task = tasks[user["task_ids"][0]]
        record = runtime.run_task(
            args=args,
            phase="warmup",
            user_index=user_index,
            turn_index=0,
            task=task,
            model_id=model_id,
            scheduled_perf=scheduled,
            seed=args.seed_base - 1000 + user_index,
            max_tokens=token_caps[task["family"]],
        )
        record["length_bucket_tokens"] = task["length_bucket_tokens"]
        return record

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        records = list(executor.map(one, users))
    return sorted(records, key=lambda row: row["user_index"])


class MeasurementWindow:
    def __init__(self, seconds: float) -> None:
        self.seconds = seconds
        self.condition = threading.Condition()
        self.start: float | None = None
        self.end: float | None = None
        self.exhaustion: list[dict[str, Any]] = []

    def claim_submission(self, user_index: int, turn_index: int, submitted: float) -> bool:
        with self.condition:
            if user_index == 0 and turn_index == 0:
                if self.start is not None:
                    raise RuntimeError("user 0 first submission was recorded twice")
                self.start = submitted
                self.end = submitted + self.seconds
                self.condition.notify_all()
                return True
            while self.start is None:
                self.condition.wait(timeout=0.1)
            assert self.end is not None
            return submitted < self.end

    def first_schedule(self, user_index: int, provisional: float) -> float:
        if user_index == 0:
            return provisional
        with self.condition:
            while self.start is None:
                self.condition.wait(timeout=0.1)
            assert self.start is not None
            return self.start + user_index * 0.5

    def wait_for_submission_time(self, scheduled: float) -> bool:
        with self.condition:
            while self.start is None:
                self.condition.wait(timeout=0.1)
            assert self.end is not None
            while True:
                now = time.perf_counter()
                if now >= self.end:
                    return False
                if now >= scheduled:
                    return True
                self.condition.wait(timeout=min(scheduled, self.end) - now)

    def record_exhaustion(self, user_index: int, next_scheduled: float) -> None:
        with self.condition:
            if self.end is not None and next_scheduled < self.end:
                self.exhaustion.append(
                    {"user_index": user_index, "next_scheduled_perf": next_scheduled}
                )


def run_measured(
    *,
    args: argparse.Namespace,
    users: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    model_id: str,
    token_caps: dict[str, int],
    window_seconds: float,
) -> tuple[list[dict[str, Any]], MeasurementWindow]:
    provisional_origin = time.perf_counter()
    window = MeasurementWindow(window_seconds)
    sink: list[dict[str, Any]] = []
    sink_lock = threading.Lock()

    def user_loop(user: dict[str, Any]) -> None:
        user_index = int(user["user_index"])
        scheduled = window.first_schedule(user_index, provisional_origin)
        last_terminal: float | None = None
        for turn_index, task_id in enumerate(user["task_ids"]):
            if turn_index:
                assert last_terminal is not None
                scheduled = last_terminal + 2.0
            if user_index == 0 and turn_index == 0:
                wait_until(scheduled)
            elif not window.wait_for_submission_time(scheduled):
                return
            task = tasks[task_id]
            callback = lambda submitted, u=user_index, k=turn_index: window.claim_submission(
                u, k, submitted
            )
            record = runtime.run_task(
                args=args,
                phase=args.stream,
                user_index=user_index,
                turn_index=turn_index,
                task=task,
                model_id=model_id,
                scheduled_perf=scheduled,
                seed=args.seed_base + int(task["source_ordinal"]),
                max_tokens=token_caps[task["family"]],
                on_submit=callback,
            )
            if record.get("_submission_rejected"):
                return
            record["length_bucket_tokens"] = task["length_bucket_tokens"]
            last_terminal = record["_terminal_perf"]
            with sink_lock:
                sink.append(record)
        if last_terminal is not None:
            window.record_exhaustion(user_index, last_terminal + 2.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(user_loop, user) for user in users]
        for future in futures:
            future.result()
    with sink_lock:
        records = list(sink)
    return records, window


def finish_valid(record: dict[str, Any]) -> bool:
    return runtime.finish_valid(record) and record.get("finish_reason") != "length"


def warmup_terminal_valid(record: dict[str, Any]) -> bool:
    return bool(
        record.get("error") is None
        and record.get("finish_reason") in {"stop", "length"}
        and record.get("terminal_reason")
        == f"server_{record.get('finish_reason')}"
    )


def warmup_aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = [record for record in records if not warmup_terminal_valid(record)]
    return {
        "submitted": len(records),
        "completed": len(records) - len(invalid),
        "terminal_reason_counts": dict(
            Counter(str(record.get("terminal_reason")) for record in records)
        ),
        "invalid_terminal_count": len(invalid),
        "invalid_terminal_reason_counts": dict(
            Counter(str(record.get("terminal_reason")) for record in invalid)
        ),
    }


def normalize_finqa_number(value: str) -> float:
    text = value.replace("$", "").strip().split("(", 1)[0].strip().replace(",", "")
    percent = "%" in text
    if percent:
        text = text.replace("%", "")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError("non-finite numeric answer")
    if percent:
        number /= 100.0
    return round(number, 5)


def parse_numeric_or_knowledge(record: dict[str, Any], task: dict[str, Any]) -> None:
    if not finish_valid(record):
        record["parse_status"] = "invalid_finish"
        return
    if task["family"] == "numeric":
        match = NUMERIC_RE.fullmatch(record["raw_output"])
        if match is None:
            record["parse_status"] = "format_error"
            return
        try:
            observed = normalize_finqa_number(match.group(1))
        except (ValueError, OverflowError):
            record["parse_status"] = "number_error"
            return
        record["parse_status"] = "parsed"
        record["judge_correct"] = observed == task["reference"]["normalized_value"]
        return
    match = KNOWLEDGE_RE.fullmatch(record["raw_output"])
    if match is None:
        record["parse_status"] = "format_error"
        return
    record["parse_status"] = "parsed"
    record["judge_correct"] = match.group(1) == task["reference"]["value"]


def parse_unified_diff(record: dict[str, Any]) -> str | None:
    if not finish_valid(record):
        record["parse_status"] = "invalid_finish"
        return None
    raw = record["raw_output"]
    patch = raw.strip()
    if not patch or "```" in patch:
        record["parse_status"] = "format_error"
        return None
    lines = patch.splitlines()
    if DIFF_FIRST_RE.fullmatch(lines[0]) is None:
        record["parse_status"] = "format_error"
        return None
    has_old = any(line.startswith("--- ") for line in lines)
    has_new = any(line.startswith("+++ ") for line in lines)
    has_hunk = any(re.match(r"@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@", line) for line in lines)
    if not (has_old and has_new and has_hunk):
        record["parse_status"] = "format_error"
        return None
    record["parse_status"] = "parsed"
    return patch + ("\n" if raw.endswith("\n") else "")


def verified_parquet(manifest: dict[str, Any]) -> Path:
    source_cache = manifest.get("source_cache")
    if not isinstance(source_cache, dict):
        raise ValueError("manifest source_cache is missing")
    directory = source_cache.get("directory")
    filename = source_cache.get("files", {}).get("swe_verified_test.parquet")
    if not isinstance(directory, str) or not isinstance(filename, str):
        raise ValueError("manifest omits fixed SWE-bench Verified Parquet")
    path = (Path(directory) / filename).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def run_swebench(
    *,
    args: argparse.Namespace,
    swebench_python: str,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items: list[dict[str, str]] = []
    for record in records:
        if record["family"] != "code":
            continue
        patch = parse_unified_diff(record)
        if patch is None:
            continue
        items.append(
            {
                "record_id": record["record_id"],
                "instance_id": tasks[record["task_id"]]["reference"]["instance_id"],
                "model_patch": patch,
            }
        )
    if not items:
        return {
            "status": "complete",
            "python": swebench_python,
            "root": str(args.swebench_root.expanduser().resolve()),
            "submitted_to_verifier": 0,
            "error": None,
        }
    with tempfile.TemporaryDirectory(prefix="freetoken-swebench-subset-v2-") as temporary:
        root = Path(temporary)
        input_path = root / "input.json"
        output_path = root / "output.json"
        serving.write_json(
            input_path,
            {
                "schema": "freetoken.swebench_verified_subset_input.v2",
                "items": items,
            },
        )
        completed = subprocess.run(
            [
                swebench_python,
                str(HERE / "score_swebench_verified_subset.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--verified-parquet",
                str(verified_parquet(manifest)),
                "--swebench-root",
                str(args.swebench_root.expanduser().resolve()),
                "--max-workers",
                str(args.swebench_max_workers),
                "--instance-timeout",
                str(args.swebench_instance_timeout),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.swebench_scorer_timeout,
        )
        if completed.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                f"SWE-bench scorer exited {completed.returncode}; output tail:\n"
                + completed.stdout[-12000:]
            )
        result = load_json(output_path)
    if result.get("schema") != SWE_RESULT_SCHEMA:
        raise ValueError("SWE-bench scorer returned unsupported schema")
    by_record = {item["record_id"]: item for item in result.get("items", [])}
    if set(by_record) != {item["record_id"] for item in items}:
        raise ValueError("SWE-bench scorer did not return every submitted record once")
    for record in records:
        verdict = by_record.get(record["record_id"])
        if verdict is None:
            continue
        record["verifier"] = {
            "status": verdict.get("status"),
            "report_available": verdict.get("report_available"),
        }
        record["judge_correct"] = verdict.get("resolved") is True
    return {
        "status": "complete",
        "python": swebench_python,
        "root": str(args.swebench_root.expanduser().resolve()),
        "submitted_to_verifier": len(items),
        "evaluator": result.get("evaluator"),
        "dataset": result.get("dataset"),
        "error": None,
    }


def attach_metrics(
    records: list[dict[str, Any]], tasks: dict[str, dict[str, Any]], token_counter: Any
) -> None:
    for record in records:
        task = tasks[record["task_id"]]
        record["latency_seconds"] = record["_terminal_perf"] - record["_submitted_perf"]
        record["deadline_seconds"] = SLO_SECONDS[record["family"]]
        record["finish_valid"] = finish_valid(record)
        record["slo_success"] = bool(
            record["judge_correct"]
            and record["finish_valid"]
            and record["latency_seconds"] <= record["deadline_seconds"]
        )
        record["ttft_seconds"] = (
            record["_first_text_perf"] - record["_submitted_perf"]
            if record["_first_text_perf"] is not None
            else None
        )
        usage = record.get("server_usage")
        completion_tokens = usage.get("completion_tokens") if isinstance(usage, dict) else None
        observation = token_counter.output_observation(record["raw_output"])
        record["output_token_observation"] = observation
        if not isinstance(completion_tokens, int):
            completion_tokens = int(observation["tokens"])
        record["completion_tokens"] = completion_tokens
        record["prompt_token_observation"] = task["prompt_token_observation"]
        record["length_bucket_tokens"] = task["length_bucket_tokens"]
        record["tpot_seconds"] = (
            (record["_last_text_perf"] - record["_first_text_perf"])
            / (completion_tokens - 1)
            if completion_tokens > 1
            and record["_first_text_perf"] is not None
            and record["_last_text_perf"] is not None
            else None
        )


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    output = args.output or HERE / "results" / f"long_mixed_goodput_{args.arm_name}_{args.stream}_{timestamp}.json"
    log = args.server_log or output.with_suffix(".server.log")
    return output.expanduser().absolute(), log.expanduser().absolute()


def token_counter_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | None]:
    before_requests = before.get("requests") if isinstance(before.get("requests"), dict) else {}
    after_requests = after.get("requests") if isinstance(after.get("requests"), dict) else {}
    result: dict[str, int | None] = {}
    for field in ("prompt_tokens_total", "completion_tokens_total"):
        left, right = before_requests.get(field), after_requests.get(field)
        result[field] = right - left if isinstance(left, int) and isinstance(right, int) else None
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    warmup_tasks = load_tasks(resolve_task_path(manifest_path, manifest, "warmup"), "warmup")
    measured_tasks = load_tasks(resolve_task_path(manifest_path, manifest, args.stream), args.stream)
    warmup_users = validate_assignments(manifest, "warmup", warmup_tasks)
    measured_users = validate_assignments(manifest, args.stream, measured_tasks)
    output, log = output_paths(args)
    root = args.freetoken_root.expanduser().resolve()
    python = serving.resolve_python(args.python_executable)
    swebench_python = serving.resolve_python(args.swebench_python)
    model = serving.model_argument(args.model)
    environment, environment_overrides = serving.server_environment(args, root)
    command = serving.server_command(args, python, model)
    token_caps = max_tokens_by_family(args)
    window_seconds = float(manifest["traffic"][args.stream]["submission_window_seconds"])

    if args.dry_run:
        payload = {
            "mode": "dry-run",
            "arm_name": args.arm_name,
            "stream": args.stream,
            "manifest": str(manifest_path),
            "command": command,
            "environment_overrides": environment_overrides,
            "server_python": python,
            "swebench_python": swebench_python,
            "swebench_root": str(args.swebench_root.expanduser().absolute()),
            "request_timeout_seconds": HARD_REQUEST_TIMEOUT,
            "token_caps": token_caps,
            "traffic": manifest["traffic"],
            "length_buckets": manifest["length_buckets"],
            "theoretical": manifest["theoretical"],
            "selected_stream_task_count": len(measured_tasks),
            "output": str(output),
            "server_log": str(log),
            "server_started": False,
            "docker_started": False,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 0

    token_counter = serving.ClientTokenCounter(manifest["tokenizer"]["identifier"])
    prepare_prompts(warmup_tasks.values(), token_counter)
    prepare_prompts(measured_tasks.values(), token_counter)
    base_url = f"http://{args.host}:{args.port}"
    server = serving.ManagedServer(
        command=command,
        environment=environment,
        cwd=root,
        base_url=base_url,
        log_path=log,
        startup_timeout=args.server_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    poller: runtime.StatsPoller | None = None
    warmup: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    shutdown_error: str | None = None
    try:
        server.start()
        assert server.model_id is not None
        warmup = run_warmup(
            args=args,
            users=warmup_users,
            tasks=warmup_tasks,
            model_id=server.model_id,
            token_caps=token_caps,
        )
        warmup_idle = server.wait_idle(HARD_REQUEST_TIMEOUT)
        warmup_status = warmup_aggregate(warmup)
        if warmup_status["invalid_terminal_count"]:
            raise RuntimeError(
                "fixed warmup contained "
                f"{warmup_status['invalid_terminal_count']} invalid terminal(s): "
                f"{warmup_status['invalid_terminal_reason_counts']}"
            )
        stats_before = server.stats()
        poller = runtime.StatsPoller(server, args.stats_poll_seconds)
        poller.start()
        clock_perf = time.perf_counter()
        clock_wall = time.time()
        records, window = run_measured(
            args=args,
            users=measured_users,
            tasks=measured_tasks,
            model_id=server.model_id,
            token_caps=token_caps,
            window_seconds=window_seconds,
        )
        drain_stats = server.wait_idle(HARD_REQUEST_TIMEOUT)
        first_server_idle_ack = time.perf_counter()
        if poller is not None:
            poller.stop()
        if window.start is None or window.end is None:
            raise RuntimeError("user 0 never established the measured submission window")
        if not records:
            raise RuntimeError("measured stream submitted no tasks")
        measurement_wall = clock_wall + (window.start - clock_perf)
        last_terminal = max(record["_terminal_perf"] for record in records)
        denominator_end = max(window.end, last_terminal, first_server_idle_ack)
        denominator = denominator_end - window.start

        for record in records:
            if record["family"] != "code":
                parse_numeric_or_knowledge(record, measured_tasks[record["task_id"]])
        verifier = run_swebench(
            args=args,
            swebench_python=swebench_python,
            manifest=manifest,
            records=records,
            tasks=measured_tasks,
        )
        attach_metrics(records, measured_tasks, token_counter)
        by_family = {
            family: runtime.summarize_group(
                [record for record in records if record["family"] == family], denominator
            )
            for family in FAMILIES
        }
        by_bucket = {
            str(target): runtime.summarize_group(
                [record for record in records if record["length_bucket_tokens"] == target],
                denominator,
            )
            for target in TARGETS
        }
        summary = {
            "denominator_seconds": denominator,
            "submission_window_seconds": window_seconds,
            "drain_tail_seconds": max(0.0, denominator_end - window.end),
            "total": runtime.summarize_group(records, denominator),
            "families": by_family,
            "length_buckets": by_bucket,
            "concurrency": runtime.concurrency_summary(records, denominator),
        }
        valid = not window.exhaustion
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "stream": args.stream,
            "valid_task_goodput": valid,
            "invalid_reason": (
                "frozen per-user queue exhausted before window end" if not valid else None
            ),
            "manifest": str(manifest_path),
            "source_provenance": manifest["sources"],
            "training_forbidden_texts": manifest["training_forbidden_texts"],
            "traffic": manifest["traffic"],
            "frozen_policy": {
                "system_prompt": SYSTEM_PROMPT,
                "greedy": True,
                "enable_thinking": False,
                "max_sequence_length": 40960,
                "hard_request_timeout_seconds": HARD_REQUEST_TIMEOUT,
                "max_tokens": token_caps,
                "slo_seconds": SLO_SECONDS,
            },
            "server": {
                "freetoken_root": str(root),
                "git": serving.git_info(root),
                "python": python,
                "model_argument": model,
                "served_model_id": server.model_id,
                "gpu": args.gpu,
                "base_url": base_url,
                "command": command,
                "environment_overrides": environment_overrides,
                "log": str(log),
                "shutdown_error": None,
            },
            "client_tokenizer": token_counter.info(),
            "warmup": {
                **warmup_status,
                "family_counts": {
                    family: sum(record["family"] == family for record in warmup)
                    for family in FAMILIES
                },
                "fully_drained": warmup_idle.get("requests", {}).get("active") == 0,
                "stats_after": warmup_idle,
            },
            "online_timing": {
                "measurement_start_unix_seconds": measurement_wall,
                "submission_window_end_seconds": window.end - window.start,
                "last_client_terminal_seconds": last_terminal - window.start,
                "server_idle_ack_seconds": first_server_idle_ack - window.start,
                "denominator_end_seconds": denominator_end - window.start,
                "denominator_seconds": denominator,
                "drain_tail_seconds": max(0.0, denominator_end - window.end),
            },
            "verifier": verifier,
            "summary": summary,
            "server_observability": {
                "stats_before": stats_before,
                "stats_after_drain": drain_stats,
                "request_counter_delta": token_counter_delta(stats_before, drain_stats),
                "peak_observed_vram_bytes": poller.peak_vram_bytes if poller else None,
                "peak_observed_vram_source": "/v1/stats vram_bytes polling",
                "poll_samples": poller.samples if poller else 0,
                "poll_error": poller.error if poller else None,
                "expert_cache_misses": None,
                "h2d_bytes": None,
                "missing_metric_reason": (
                    "the production /v1/stats schema does not expose expert-cache misses or H2D bytes"
                ),
            },
            "queue_exhaustion": window.exhaustion,
            "records": [
                runtime.public_record(record, window.start, measurement_wall)
                for record in sorted(
                    records, key=lambda row: (row["user_index"], row["turn_index"])
                )
            ],
        }
    except BaseException as exc:
        failure = exc
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "stream": args.stream,
            "valid_task_goodput": False,
            "invalid_reason": f"{type(exc).__name__}: {exc}",
            "manifest": str(manifest_path),
            "warmup": warmup_aggregate(warmup),
            "server": {
                "command": command,
                "environment_overrides": environment_overrides,
                "log": str(log),
                "shutdown_error": None,
            },
            "partial_terminal_record_count": len(records),
        }
    finally:
        if poller is not None:
            poller.stop()
        try:
            server.stop()
        except Exception as exc:
            shutdown_error = f"{type(exc).__name__}: {exc}"
        finally:
            server.close()
        assert result is not None
        result["server"]["shutdown_error"] = shutdown_error
        if shutdown_error is not None:
            result["valid_task_goodput"] = False
            result["invalid_reason"] = result.get("invalid_reason") or shutdown_error
        serving.write_json(output, result)

    stdout = {
        "output": str(output),
        "stream": args.stream,
        "valid_task_goodput": result["valid_task_goodput"],
        "invalid_reason": (
            "see the result file for the final-stream failure diagnostic"
            if args.stream == "final" and result.get("invalid_reason")
            else result.get("invalid_reason")
        ),
        "summary": result.get("summary"),
        "per_task_stdout": False,
    }
    print(json.dumps(stdout, indent=2, ensure_ascii=False))
    if failure is not None:
        return 2
    return 0 if result["valid_task_goodput"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
