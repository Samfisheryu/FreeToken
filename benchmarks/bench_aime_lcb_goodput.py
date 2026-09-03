#!/usr/bin/env python3
"""Measure AIME + LiveCodeBench accuracy and maximum serving throughput.

One invocation owns one FreeToken server and one independently tuned arm.  It
runs two fixed warmup requests to natural server completion and full drain,
then drives twenty closed-loop users with at most one request in flight per
user and zero think time.  Development admits new requests for 300 seconds
without wrapping its finite queue.  Final submits every one of its 50 tasks
exactly once.  Both modes fully drain before the online denominator closes.

No response is stopped at the first boxed answer or code block.  Only natural
server ``stop`` and ``length`` finishes are protocol-valid.  LiveCodeBench's
official hidden-test scorer runs after online timing has closed.

Pass ``--server-args`` last; every following token belongs to ``ft serve``.
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
import socket
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
    "/data1/lmcache_kv/goodput_campaign/aime25_lcb50_familythink_v1/manifest.json"
)
RESULT_SCHEMA = "freetoken.aime_lcb50_familythink_goodput_result.v1"
MANIFEST_SCHEMA = "freetoken.aime_lcb50_familythink_goodput_manifest.v1"
TASK_SCHEMA = "freetoken.aime_lcb50_familythink_goodput_task.v1"
SCORER_INPUT_SCHEMA = "freetoken.livecodebench_subset_input.v1"
SCORER_OUTPUT_SCHEMA = "freetoken.livecodebench_subset_result.v1"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RUNNER_REVISION = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
FAMILIES = ("aime", "code")
THINKING_BY_FAMILY = {"aime": True, "code": False}
VALID_FINISH_REASONS = {"stop", "length"}
BOXED_INTEGER_RE = re.compile(r"\\boxed\s*\{\s*([+-]?\d+)\s*\}")
ALLOWED_MAX_SEQUENCE_LENGTHS = (32768, 65536)
MAX_RUNNING_REQUESTS = 20
DEV_WINDOW_SECONDS = 300.0
DEFAULT_REQUEST_TIMEOUT = 3600.0
CAP_LIMITS = {"aime": (1, 49152), "code": (1, 32768)}
DEFAULT_CAPS = {"aime": 49152, "code": 32768}
AIME_SOURCES = {
    "warmup": {
        "name": "amc23",
        "repository": "math-ai/amc23",
        "revision": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
        "source_file": "amc23-test.parquet",
    },
    "dev": {
        "name": "aime24",
        "repository": "math-ai/aime24",
        "revision": "83a7f387baaa524a8bda0022eac0541582297103",
        "source_file": "aime24-test.parquet",
    },
    "final": {
        "name": "aime25",
        "repository": "math-ai/aime25",
        "revision": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
        "source_file": "aime25-test.jsonl",
    },
}
LCB_INCREMENT_BY_SPLIT = {"warmup": "v4", "dev": "v5", "final": "v6"}
LCB_SOURCE_FILE_BY_SPLIT = {
    "warmup": "lcb-test4.jsonl",
    "dev": "lcb-test5.jsonl",
    "final": "lcb-test6.jsonl",
}


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
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="per-request transport safety ceiling; any hit invalidates the run",
    )
    parser.add_argument(
        "--aime-max-tokens",
        type=int,
        default=DEFAULT_CAPS["aime"],
        help="AIME new-token cap; formal default and maximum are 49152",
    )
    parser.add_argument(
        "--code-max-tokens",
        type=int,
        default=DEFAULT_CAPS["code"],
        help="code new-token cap; formal default and maximum are 32768",
    )
    parser.add_argument("--seed-base", type=int, default=20260903)
    parser.add_argument(
        "--lcb-root",
        type=Path,
        required=True,
        help=f"official LiveCodeBench checkout at {RUNNER_REVISION}",
    )
    parser.add_argument(
        "--lcb-python",
        default=sys.executable,
        help="Python environment containing the pinned LiveCodeBench dependencies",
    )
    parser.add_argument(
        "--lcb-source-cache",
        type=Path,
        help="override manifest source_cache for lcb-test5.jsonl/lcb-test6.jsonl",
    )
    parser.add_argument("--lcb-workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--lcb-test-timeout", type=int, default=6)
    parser.add_argument("--lcb-scorer-timeout", type=float, default=21600.0)
    parser.add_argument("--stats-poll-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print frozen commands/policy without starting a server, judge, or GPU and without writing",
    )
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "remaining arguments for `ft serve`; must contain exactly one "
            "--max-running-requests 20 and one --max-seq-len-override "
            "equal to 32768 or 65536; this option must be last"
        ),
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in (
        "server_timeout",
        "shutdown_timeout",
        "request_timeout",
        "lcb_scorer_timeout",
        "stats_poll_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.request_timeout < DEFAULT_REQUEST_TIMEOUT:
        parser.error("--request-timeout must be at least 3600 seconds")
    if args.lcb_workers < 1 or args.lcb_test_timeout < 1:
        parser.error("LiveCodeBench worker count and test timeout must be positive")
    for family, limits in CAP_LIMITS.items():
        value = int(getattr(args, f"{family}_max_tokens"))
        if not limits[0] <= value <= limits[1]:
            parser.error(
                f"--{family}-max-tokens must be in {limits[0]}..{limits[1]}"
            )
    args.server_env = serving.parse_environment(args.env, parser)
    serving.validate_server_args(args.server_args, parser)
    args.max_running_requests = required_server_int(
        args.server_args, "--max-running-requests", parser
    )
    if args.max_running_requests != MAX_RUNNING_REQUESTS:
        parser.error(
            f"--server-args must set --max-running-requests {MAX_RUNNING_REQUESTS}"
        )
    args.max_sequence_length = required_server_int(
        args.server_args, "--max-seq-len-override", parser
    )
    if args.max_sequence_length not in ALLOWED_MAX_SEQUENCE_LENGTHS:
        allowed = " or ".join(str(value) for value in ALLOWED_MAX_SEQUENCE_LENGTHS)
        parser.error(f"--max-seq-len-override must be {allowed}")
    for family in FAMILIES:
        cap = int(getattr(args, f"{family}_max_tokens"))
        if cap > args.max_sequence_length:
            parser.error(
                f"--{family}-max-tokens ({cap}) exceeds server max sequence length "
                f"({args.max_sequence_length})"
            )
    return args


def required_server_int(
    server_args: list[str], option: str, parser: argparse.ArgumentParser
) -> int:
    values: list[str] = []
    index = 0
    while index < len(server_args):
        token = server_args[index]
        if token == option:
            if index + 1 >= len(server_args):
                parser.error(f"{option} needs an integer value")
            values.append(server_args[index + 1])
            index += 2
            continue
        if token.startswith(f"{option}="):
            values.append(token.split("=", 1)[1])
        index += 1
    if len(values) != 1:
        parser.error(f"--server-args must contain exactly one {option}")
    try:
        return int(values[0])
    except ValueError:
        parser.error(f"{option} needs an integer value")


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def git_commit(root: Path) -> str:
    environment = os.environ.copy()
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    return completed.stdout.strip()


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported AIME+LCB manifest: {path}")
    traffic = manifest.get("traffic")
    policy = manifest.get("request_policy")
    sources = manifest.get("sources")
    if not all(isinstance(value, dict) for value in (traffic, policy, sources)):
        raise ValueError("manifest needs traffic, request_policy, and sources objects")
    observed_traffic = (
        traffic.get("user_count"),
        traffic.get("think_time_seconds"),
        traffic.get("one_request_in_flight_per_user"),
        traffic.get("fixed_queue_once"),
        traffic.get("queue_wrap"),
        traffic.get("dev_submission_window_seconds"),
        traffic.get("final_submission_window_seconds"),
        traffic.get("warmup_task_count"),
    )
    if observed_traffic != (20, 0, True, True, False, 300, None, 2):
        raise ValueError("manifest traffic differs from the frozen contract")
    if "thinking" in policy or (
        policy.get("thinking_by_family"),
        policy.get("n"),
        policy.get("temperature"),
        policy.get("top_p"),
        policy.get("top_k"),
        policy.get("default_request_timeout_seconds"),
    ) != (THINKING_BY_FAMILY, 1, 0.0, 1.0, -1, 3600):
        raise ValueError("manifest request policy differs from the frozen contract")
    lcb = sources.get("livecodebench")
    evaluator = lcb.get("evaluator") if isinstance(lcb, dict) else None
    if not isinstance(lcb, dict) or (
        lcb.get("repository"), lcb.get("revision")
    ) != ("livecodebench/code_generation_lite", DATASET_REVISION):
        raise ValueError("manifest LiveCodeBench data revision changed")
    if not isinstance(evaluator, dict) or evaluator.get("revision") != RUNNER_REVISION:
        raise ValueError("manifest LiveCodeBench evaluator revision changed")
    for split, expected in AIME_SOURCES.items():
        source = sources.get(expected["name"])
        if not isinstance(source, dict) or (
            source.get("repository"),
            source.get("revision"),
            source.get("cache_file"),
        ) != (
            expected["repository"],
            expected["revision"],
            expected["source_file"],
        ):
            raise ValueError(f"manifest {split} AIME source provenance changed")
    expected_counts = {
        "warmup": (2, 1, 1),
        "dev": (156, 30, 126),
        "final": (50, 30, 20),
    }
    counts = manifest.get("task_counts")
    if not isinstance(counts, dict):
        raise ValueError("manifest task_counts is missing")
    for split, expected in expected_counts.items():
        row = counts.get(split)
        actual = (
            row.get("total") if isinstance(row, dict) else None,
            row.get("families", {}).get("aime") if isinstance(row, dict) else None,
            row.get("families", {}).get("code") if isinstance(row, dict) else None,
        )
        if actual != expected:
            raise ValueError(f"manifest {split} task counts changed: {actual} != {expected}")
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
    common_fields = {
        "schema",
        "task_id",
        "split",
        "family",
        "source",
        "source_revision",
        "source_file",
        "source_ordinal",
        "difficulty",
        "scored",
        "problem_text",
        "task_text",
        "reference",
    }
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("schema") != TASK_SCHEMA:
                raise ValueError(f"unsupported task row at {path}:{line_number}")
            family = row.get("family")
            expected_fields = common_fields | ({"source_increment"} if family == "code" else set())
            if set(row) != expected_fields:
                raise ValueError(f"task fields changed at {path}:{line_number}")
            task_id = row.get("task_id")
            if not isinstance(task_id, str) or not task_id or task_id in tasks:
                raise ValueError(f"invalid/duplicate task_id at {path}:{line_number}")
            if row.get("split") != split or family not in FAMILIES:
                raise ValueError(f"invalid task split/family at {path}:{line_number}")
            if not all(
                isinstance(row.get(field), str) and row[field].strip()
                for field in ("source", "source_revision", "source_file", "problem_text", "task_text")
            ):
                raise ValueError(f"task {task_id} has invalid text/provenance fields")
            if not isinstance(row.get("source_ordinal"), int):
                raise ValueError(f"task {task_id} has invalid source_ordinal")
            scored = row.get("scored")
            if not isinstance(scored, bool) or scored != (split != "warmup"):
                raise ValueError(f"task {task_id} has invalid scored state")
            reference = row.get("reference")
            if family == "aime":
                expected_kind = "aime_integer" if scored else "unscored"
            else:
                expected_kind = "livecodebench" if scored else "unscored"
            if not isinstance(reference, dict) or reference.get("kind") != expected_kind:
                raise ValueError(f"task {task_id} reference kind changed")
            if family == "aime":
                if scored:
                    answer = reference.get("answer")
                    if not isinstance(answer, int) or not 0 <= answer <= 999:
                        raise ValueError(f"task {task_id} AIME answer is outside 000..999")
                elif set(reference) != {"kind"}:
                    raise ValueError(f"task {task_id} unscored reference contains answer data")
                expected = AIME_SOURCES[split]
                if (
                    not task_id.startswith(expected["name"] + "/")
                    or row.get("source") != expected["repository"]
                    or row.get("source_revision") != expected["revision"]
                    or row.get("source_file") != expected["source_file"]
                ):
                    raise ValueError(f"task {task_id} AIME provenance changed")
            else:
                if not scored:
                    if set(reference) != {"kind"}:
                        raise ValueError(
                            f"task {task_id} unscored reference contains judge data"
                        )
                elif (
                    reference.get("question_id") != task_id.removeprefix("lcb/")
                    or reference.get("source_file") != row.get("source_file")
                ):
                    raise ValueError(f"task {task_id} LiveCodeBench reference changed")
                if (
                    row.get("source") != "livecodebench/code_generation_lite"
                    or row.get("source_revision") != DATASET_REVISION
                    or row.get("source_increment") != LCB_INCREMENT_BY_SPLIT[split]
                    or row.get("source_file") != LCB_SOURCE_FILE_BY_SPLIT[split]
                    or row.get("difficulty") not in {"medium", "hard"}
                ):
                    raise ValueError(f"task {task_id} LiveCodeBench provenance changed")
            row["_stream_index"] = len(tasks)
            tasks[task_id] = row
    if not tasks:
        raise ValueError(f"empty task stream: {path}")
    return tasks


def validate_assignments(
    manifest: dict[str, Any], split: str, tasks: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    users = manifest.get("assignments", {}).get(split)
    expected_user_count = 2 if split == "warmup" else 20
    if not isinstance(users, list) or len(users) != expected_user_count:
        raise ValueError(
            f"{split} assignments must contain exactly {expected_user_count} users"
        )
    assigned: list[str] = []
    for user_index, user in enumerate(users):
        if (
            not isinstance(user, dict)
            or user.get("user_index") != user_index
            or user.get("user_id") != f"user-{user_index:02d}"
        ):
            raise ValueError(f"{split} users must be ordered user-00 through user-19")
        task_ids = user.get("task_ids")
        if not isinstance(task_ids, list) or (split == "warmup" and len(task_ids) != 1):
            raise ValueError(f"{split} user {user_index} has an invalid fixed queue")
        if any(task_id not in tasks for task_id in task_ids):
            raise ValueError(f"{split} user {user_index} references an unknown task")
        assigned.extend(task_ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(tasks):
        raise ValueError(f"{split} assignments must consume every task exactly once")
    if split == "final" and Counter(len(user["task_ids"]) for user in users) != Counter(
        {2: 10, 3: 10}
    ):
        raise ValueError("final users must each receive two or three tasks")
    return users


def token_caps(args: argparse.Namespace) -> dict[str, int]:
    return {
        "aime": int(args.aime_max_tokens),
        "code": int(args.code_max_tokens),
    }


def prepare_prompts(
    tasks: Iterable[dict[str, Any]], manifest: dict[str, Any], token_counter: Any
) -> None:
    policy = manifest["request_policy"]
    for task in tasks:
        system = (
            policy["aime_system_prompt"]
            if task["family"] == "aime"
            else policy["code_system_prompt"]
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": task["task_text"]})
        task["messages"] = messages
        enable_thinking = policy["thinking_by_family"][task["family"]]
        task["prompt_token_observation"] = token_counter.prompt_observation(
            messages, {"chat_template_kwargs": {"enable_thinking": enable_thinking}}
        )


def validate_sequence_capacity(
    tasks: Iterable[dict[str, Any]], caps: dict[str, int], max_sequence_length: int
) -> dict[str, Any]:
    maximum_by_family = {family: 0 for family in FAMILIES}
    for task in tasks:
        observation = task["prompt_token_observation"]
        tokens = observation.get("tokens")
        if observation.get("estimated") or not isinstance(tokens, int):
            raise ValueError(
                f"task {task['task_id']} lacks an exact rendered prompt token count; "
                "max-sequence compatibility cannot be proven"
            )
        family = task["family"]
        maximum_by_family[family] = max(maximum_by_family[family], tokens)
        if tokens + caps[family] > max_sequence_length:
            raise ValueError(
                f"task {task['task_id']} prompt ({tokens}) + {family} max tokens "
                f"({caps[family]}) exceeds {max_sequence_length}"
            )
    return {
        "max_prompt_tokens_by_family": maximum_by_family,
        "max_new_tokens_by_family": caps,
        "max_sequence_length": max_sequence_length,
        "all_tasks_compatible": True,
    }


def base_record(
    *, phase: str, user_index: int, turn_index: int, task: dict[str, Any], seed: int
) -> dict[str, Any]:
    return {
        "record_id": f"{phase}:{user_index}:{turn_index}:{task['task_id']}",
        "phase": phase,
        "user_id": f"user-{user_index:02d}",
        "user_index": user_index,
        "turn_index": turn_index,
        "stream_index": task["_stream_index"],
        "task_id": task["task_id"],
        "family": task["family"],
        "difficulty": task["difficulty"],
        "source": task["source"],
        "source_revision": task["source_revision"],
        "source_file": task["source_file"],
        "request_seed": seed,
        "messages": task["messages"],
        "max_tokens": None,
        "finish_reason": None,
        "terminal_reason": None,
        "error": None,
        "saw_stream_done": False,
        "raw_output": "",
        "reasoning_output": "",
        "content_output": "",
        "server_usage": None,
        "parse_status": None,
        "observed_answer": None,
        "judge_correct": False,
        "verifier": None,
        "cap_hit": False,
        "_submitted_perf": None,
        "_first_text_perf": None,
        "_last_text_perf": None,
        "_terminal_perf": None,
    }


def finish_valid(record: dict[str, Any]) -> bool:
    return bool(
        record.get("error") is None
        and record.get("saw_stream_done") is True
        and record.get("finish_reason") in VALID_FINISH_REASONS
        and record.get("terminal_reason")
        == f"server_{record.get('finish_reason')}"
    )


class SubmissionGate:
    def __init__(self, window_seconds: float | None) -> None:
        self.window_seconds = window_seconds
        self.start: float | None = None
        self.end: float | None = None
        self.lock = threading.Lock()
        self.closed_rejections = 0

    def claim(self, submitted: float) -> bool:
        with self.lock:
            if self.start is None:
                self.start = submitted
                self.end = (
                    submitted + self.window_seconds
                    if self.window_seconds is not None
                    else None
                )
            if self.end is not None and submitted >= self.end:
                self.closed_rejections += 1
                return False
            return True

    def open_now(self) -> bool:
        with self.lock:
            return self.end is None or time.perf_counter() < self.end


def run_task(
    *,
    args: argparse.Namespace,
    phase: str,
    user_index: int,
    turn_index: int,
    task: dict[str, Any],
    model_id: str,
    seed: int,
    max_tokens: int,
    enable_thinking: bool,
    gate: SubmissionGate | None,
) -> dict[str, Any]:
    record = base_record(
        phase=phase,
        user_index=user_index,
        turn_index=turn_index,
        task=task,
        seed=seed,
    )
    record["max_tokens"] = max_tokens
    body = {
        "model": model_id,
        "messages": task["messages"],
        "max_tokens": max_tokens,
        "n": 1,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "chat_template_kwargs": {"enable_thinking": enable_thinking},
    }
    connection = None
    response = None
    try:
        connection = serving.http.client.HTTPConnection(
            args.host, args.port, timeout=args.request_timeout
        )
        record["_submitted_perf"] = time.perf_counter()
        if gate is not None and not gate.claim(record["_submitted_perf"]):
            record["terminal_reason"] = "submission_window_closed"
            record["_submission_rejected"] = True
            return record
        deadline = record["_submitted_perf"] + args.request_timeout
        payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
        connection.request(
            "POST",
            "/v1/chat/completions",
            body=payload,
            headers={
                "Content-Type": "application/json",
                "Content-Length": str(len(payload)),
                "Connection": "close",
            },
        )
        serving.remaining_timeout(connection, deadline)
        response = connection.getresponse()
        if response.status != 200:
            serving.remaining_timeout(connection, deadline)
            detail = response.read(4096).decode("utf-8", errors="replace")
            record["terminal_reason"] = "http_error"
            record["error"] = f"HTTP {response.status}: {detail}"
        else:
            while True:
                event = serving.read_sse_payload(response, connection, deadline)
                if event is None:
                    record["terminal_reason"] = "stream_eof"
                    record["error"] = "SSE stream ended before [DONE]"
                    break
                if event is serving._SSE_DONE:
                    record["saw_stream_done"] = True
                    if record["finish_reason"] in VALID_FINISH_REASONS:
                        record["terminal_reason"] = f"server_{record['finish_reason']}"
                    else:
                        record["terminal_reason"] = "server_done_without_valid_finish"
                        record["error"] = (
                            "server [DONE] arrived without finish_reason stop or length"
                        )
                    break
                if not isinstance(event, dict):
                    raise ValueError("stream event was not an object")
                if event.get("error") is not None:
                    record["terminal_reason"] = "stream_error"
                    record["error"] = json.dumps(event["error"], ensure_ascii=False)
                    break
                if isinstance(event.get("usage"), dict):
                    record["server_usage"] = event["usage"]
                for choice in event.get("choices") or []:
                    finish_reason = choice.get("finish_reason")
                    if finish_reason:
                        finish_reason = str(finish_reason)
                        previous = record["finish_reason"]
                        if previous is not None and previous != finish_reason:
                            raise ValueError(
                                f"stream reported conflicting finish reasons {previous!r} and "
                                f"{finish_reason!r}"
                            )
                        record["finish_reason"] = finish_reason
                for channel, piece in runtime.event_pieces(event):
                    now = time.perf_counter()
                    if record["_first_text_perf"] is None:
                        record["_first_text_perf"] = now
                    record["_last_text_perf"] = now
                    record["raw_output"] += piece
                    record[f"{channel}_output"] += piece
    except (socket.timeout, TimeoutError) as exc:
        record["terminal_reason"] = "request_timeout"
        record["error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        record["terminal_reason"] = "stream_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
        if record["_submitted_perf"] is None:
            record["_submitted_perf"] = time.perf_counter()
        record["_terminal_perf"] = time.perf_counter()
        record["cap_hit"] = record["finish_reason"] == "length"
    return record


def run_warmup(
    *,
    args: argparse.Namespace,
    users: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    model_id: str,
    caps: dict[str, int],
    thinking_by_family: dict[str, bool],
) -> list[dict[str, Any]]:
    start = threading.Event()

    def one(user: dict[str, Any]) -> dict[str, Any]:
        start.wait()
        user_index = int(user["user_index"])
        task = tasks[user["task_ids"][0]]
        return run_task(
            args=args,
            phase="warmup",
            user_index=user_index,
            turn_index=0,
            task=task,
            model_id=model_id,
            seed=args.seed_base - 10000 + task["_stream_index"],
            max_tokens=caps[task["family"]],
            enable_thinking=thinking_by_family[task["family"]],
            gate=None,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(users)) as executor:
        futures = [executor.submit(one, user) for user in users]
        start.set()
        records = [future.result() for future in futures]
    records.sort(key=lambda row: row["user_index"])
    return records


def run_measured(
    *,
    args: argparse.Namespace,
    users: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
    model_id: str,
    caps: dict[str, int],
    thinking_by_family: dict[str, bool],
) -> tuple[list[dict[str, Any]], SubmissionGate, bool]:
    gate = SubmissionGate(DEV_WINDOW_SECONDS if args.stream == "dev" else None)
    start = threading.Event()
    invalid = threading.Event()
    records: list[dict[str, Any]] = []
    records_lock = threading.Lock()
    exhausted_users: set[int] = set()

    def user_loop(user: dict[str, Any]) -> None:
        start.wait()
        user_index = int(user["user_index"])
        for turn_index, task_id in enumerate(user["task_ids"]):
            if invalid.is_set() or not gate.open_now():
                return
            task = tasks[task_id]
            record = run_task(
                args=args,
                phase=args.stream,
                user_index=user_index,
                turn_index=turn_index,
                task=task,
                model_id=model_id,
                seed=args.seed_base + task["_stream_index"],
                max_tokens=caps[task["family"]],
                enable_thinking=thinking_by_family[task["family"]],
                gate=gate,
            )
            if record.get("_submission_rejected"):
                return
            with records_lock:
                records.append(record)
            if not finish_valid(record):
                invalid.set()
                return
        with records_lock:
            exhausted_users.add(user_index)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(user_loop, user) for user in users]
        start.set()
        for future in futures:
            future.result()
    if args.stream == "dev" and not invalid.is_set() and gate.end is not None:
        remaining = gate.end - time.perf_counter()
        if remaining > 0:
            time.sleep(remaining)
    with records_lock:
        copied = list(records)
        all_users_exhausted = len(exhausted_users) == 20
    return copied, gate, all_users_exhausted


def warmup_summary(records: list[dict[str, Any]] | None) -> dict[str, Any]:
    records = records or []
    return {
        "submitted": len(records),
        "completed": sum(finish_valid(record) for record in records),
        "terminal_reason_counts": dict(
            Counter(str(record.get("terminal_reason")) for record in records)
        ),
        "finish_reason_counts": dict(
            Counter(str(record.get("finish_reason")) for record in records)
        ),
        "cap_hits": sum(bool(record.get("cap_hit")) for record in records),
        "fully_natural": bool(records) and all(finish_valid(record) for record in records),
        "prompts_or_outputs_in_summary": False,
    }


def parse_aime(record: dict[str, Any], task: dict[str, Any]) -> None:
    if not finish_valid(record):
        record["parse_status"] = "invalid_finish"
        return
    answer_text = record["content_output"] or record["raw_output"]
    matches = BOXED_INTEGER_RE.findall(answer_text)
    if not matches:
        record["parse_status"] = "no_boxed_integer"
        return
    observed = int(matches[-1])
    if not 0 <= observed <= 999:
        record["parse_status"] = "boxed_integer_outside_000_999"
        return
    record["parse_status"] = "parsed_final_boxed_integer"
    record["observed_answer"] = observed
    record["judge_correct"] = observed == task["reference"]["answer"]


def run_lcb_scorer(
    *,
    args: argparse.Namespace,
    lcb_python: str,
    lcb_root: Path,
    source_cache: Path,
    records: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    code_records = [record for record in records if record["family"] == "code"]
    items = []
    for record in code_records:
        task = tasks[record["task_id"]]
        model_output = record["content_output"] or record["raw_output"]
        record["judge_output_channel"] = (
            "content_output" if record["content_output"] else "combined_reasoning_content"
        )
        items.append(
            {
                "record_id": record["record_id"],
                "question_id": task["reference"]["question_id"],
                "source_file": task["reference"]["source_file"],
                "model_output": model_output,
            }
        )
    with tempfile.TemporaryDirectory(prefix="freetoken-aime-lcb-score-") as temporary:
        root = Path(temporary)
        input_path = root / "input.json"
        output_path = root / "output.json"
        serving.write_json(
            input_path,
            {
                "schema": SCORER_INPUT_SCHEMA,
                "dataset_revision": DATASET_REVISION,
                "runner_revision": RUNNER_REVISION,
                "items": items,
            },
        )
        completed = subprocess.run(
            [
                lcb_python,
                str(HERE / "score_livecodebench_subset.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
                "--lcb-root",
                str(lcb_root),
                "--source-cache",
                str(source_cache),
                "--workers",
                str(args.lcb_workers),
                "--test-timeout",
                str(args.lcb_test_timeout),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.lcb_scorer_timeout,
        )
        if completed.returncode != 0 or not output_path.is_file():
            raise RuntimeError(
                f"LiveCodeBench scorer exited {completed.returncode}; "
                f"diagnostic tail:\n{completed.stdout[-8000:]}"
            )
        result = load_json(output_path)
    if result.get("schema") != SCORER_OUTPUT_SCHEMA:
        raise ValueError("LiveCodeBench scorer returned an unsupported schema")
    if (
        result.get("dataset", {}).get("revision") != DATASET_REVISION
        or result.get("runner", {}).get("revision") != RUNNER_REVISION
    ):
        raise ValueError("LiveCodeBench scorer provenance changed")
    returned = result.get("items")
    if not isinstance(returned, list):
        raise ValueError("LiveCodeBench scorer omitted its item list")
    by_record = {
        item.get("record_id"): item for item in returned if isinstance(item, dict)
    }
    if len(by_record) != len(returned) or set(by_record) != {
        record["record_id"] for record in code_records
    }:
        raise ValueError("LiveCodeBench scorer did not return each code record exactly once")
    for record in code_records:
        verdict = by_record[record["record_id"]]
        record["parse_status"] = verdict.get("extraction_status")
        record["judge_correct"] = bool(verdict.get("passed"))
        record["verifier"] = {
            "status": verdict.get("status"),
            "official_error_code": verdict.get("official_error_code"),
        }
    return {
        "status": "complete",
        "timing": "after online denominator closed",
        "included_in_denominator": False,
        "submitted": len(items),
        "passed": sum(bool(item.get("passed")) for item in returned),
        "dataset_revision": DATASET_REVISION,
        "runner_revision": RUNNER_REVISION,
        "python": lcb_python,
        "root": str(lcb_root),
        "source_cache": str(source_cache),
        "workers": args.lcb_workers,
        "test_timeout_seconds": args.lcb_test_timeout,
    }


def attach_metrics(
    records: list[dict[str, Any]], tasks: dict[str, dict[str, Any]], token_counter: Any
) -> None:
    for record in records:
        task = tasks[record["task_id"]]
        record["finish_valid"] = finish_valid(record)
        record["latency_seconds"] = record["_terminal_perf"] - record["_submitted_perf"]
        record["ttft_seconds"] = (
            record["_first_text_perf"] - record["_submitted_perf"]
            if record["_first_text_perf"] is not None
            else None
        )
        output_observation = token_counter.output_observation(record["raw_output"])
        usage = record.get("server_usage")
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        prompt_tokens = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        if not isinstance(completion_tokens, int):
            completion_tokens = int(output_observation["tokens"])
        if not isinstance(prompt_tokens, int):
            prompt_tokens = int(task["prompt_token_observation"]["tokens"])
        record["completion_tokens"] = completion_tokens
        record["prompt_tokens"] = prompt_tokens
        record["output_token_observation"] = output_observation
        record["prompt_token_observation"] = task["prompt_token_observation"]
        record["tpot_seconds"] = (
            (record["_last_text_perf"] - record["_first_text_perf"])
            / (completion_tokens - 1)
            if completion_tokens > 1
            and record["_first_text_perf"] is not None
            and record["_last_text_perf"] is not None
            else None
        )


def summarize_group(records: list[dict[str, Any]], denominator: float) -> dict[str, Any]:
    submitted = len(records)
    correct = sum(bool(record["judge_correct"]) for record in records)
    completion_total = sum(int(record["completion_tokens"]) for record in records)
    prompt_total = sum(int(record["prompt_tokens"]) for record in records)
    return {
        "submitted": submitted,
        "correct": correct,
        "accuracy": correct / submitted if submitted else None,
        "pass_at_1": correct / submitted if submitted else None,
        "latency_seconds": runtime.distribution(record["latency_seconds"] for record in records),
        "ttft_seconds": runtime.distribution(record["ttft_seconds"] for record in records),
        "tpot_seconds": runtime.distribution(record["tpot_seconds"] for record in records),
        "completion_tokens": {
            "total": completion_total,
            "average_per_request": completion_total / submitted if submitted else None,
            "per_online_second": completion_total / denominator if denominator > 0 else None,
        },
        "prompt_tokens": {
            "total": prompt_total,
            "average_per_request": prompt_total / submitted if submitted else None,
            "per_online_second": prompt_total / denominator if denominator > 0 else None,
        },
        "terminal_reason_counts": dict(
            Counter(str(record["terminal_reason"]) for record in records)
        ),
        "finish_reason_counts": dict(
            Counter(str(record["finish_reason"]) for record in records)
        ),
        "cap_hits": sum(bool(record["cap_hit"]) for record in records),
    }


def build_summary(
    records: list[dict[str, Any]], denominator: float, stream: str
) -> dict[str, Any]:
    by_family = {
        family: summarize_group(
            [record for record in records if record["family"] == family], denominator
        )
        for family in FAMILIES
    }
    total = summarize_group(records, denominator)
    aime_accuracy = by_family["aime"]["accuracy"]
    code_accuracy = by_family["code"]["accuracy"]
    if aime_accuracy is None or code_accuracy is None:
        raise ValueError("both task families must have at least one submitted measured task")
    macro_quality = (aime_accuracy + code_accuracy) / 2.0
    raw_throughput = len(records) / denominator if denominator > 0 else 0.0
    correct = total["correct"]
    return {
        "stream": stream,
        "denominator_seconds": denominator,
        "aime": {
            **by_family["aime"],
            "required_final_denominator": 30,
        },
        "code": by_family["code"],
        "total": total,
        "micro_accuracy": correct / len(records) if records else None,
        "macro_quality": macro_quality,
        "raw_throughput_requests_per_second": raw_throughput,
        "literal_correct_throughput_per_second": (
            correct / denominator if denominator > 0 else 0.0
        ),
        "balanced_goodput_per_second": raw_throughput * macro_quality,
        "concurrency": runtime.concurrency_summary(records, denominator),
        "cap_hits": {
            "total": sum(bool(record["cap_hit"]) for record in records),
            "aime": by_family["aime"]["cap_hits"],
            "code": by_family["code"]["cap_hits"],
        },
    }


def public_record(
    record: dict[str, Any], first_submit: float, first_submit_wall: float
) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if not key.startswith("_")}
    for public, private in (
        ("submitted_at", "_submitted_perf"),
        ("first_text_at", "_first_text_perf"),
        ("last_text_at", "_last_text_perf"),
        ("terminal_at", "_terminal_perf"),
    ):
        value = record.get(private)
        result[public + "_seconds"] = value - first_submit if value is not None else None
        result[public + "_unix_seconds"] = (
            first_submit_wall + value - first_submit if value is not None else None
        )
    return result


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_arm = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.arm_name).strip("_") or "arm"
    output = args.output or HERE / "results" / f"aime_lcb_{args.stream}_{safe_arm}_{stamp}.json"
    output = output.expanduser().resolve()
    log = args.server_log or output.with_suffix(".server.log")
    return output, log.expanduser().resolve()


def source_cache_path(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> Path:
    value = args.lcb_source_cache or Path(str(manifest.get("source_cache", "")))
    if not str(value):
        raise ValueError("manifest omits source_cache and --lcb-source-cache was not supplied")
    return value.expanduser().resolve()


def frozen_policy(args: argparse.Namespace, manifest: dict[str, Any]) -> dict[str, Any]:
    mode = (
        "fixed_300_second_submission_window_then_full_drain"
        if args.stream == "dev"
        else "submit_all_50_tasks_once_then_full_drain"
    )
    return {
        "traffic_mode": mode,
        "user_count": 20,
        "max_running_requests": args.max_running_requests,
        "one_inflight_per_user": True,
        "think_time_seconds": 0,
        "queue_wrap": False,
        "sampling": {
            "thinking_by_family": manifest["request_policy"]["thinking_by_family"],
            "n": 1,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed_base": args.seed_base,
        },
        "max_new_tokens": token_caps(args),
        "cap_bounds": {
            family: {"minimum": limits[0], "maximum": limits[1]}
            for family, limits in CAP_LIMITS.items()
        },
        "max_sequence_length": args.max_sequence_length,
        "request_timeout_seconds": args.request_timeout,
        "request_timeout_role": "transport safety ceiling; any hit invalidates the run",
        "valid_finish_reasons": sorted(VALID_FINISH_REASONS),
        "client_early_stop": False,
        "aime_answer": "final boxed integer in 000..999",
        "code_verifier": manifest["sources"]["livecodebench"]["evaluator"],
    }


def dry_run_payload(
    *,
    args: argparse.Namespace,
    manifest_path: Path,
    manifest: dict[str, Any],
    root: Path,
    python: str,
    lcb_python: str,
    lcb_root: Path,
    source_cache: Path,
    command: list[str],
    environment_overrides: dict[str, str],
    output: Path,
    log: Path,
) -> dict[str, Any]:
    candidate_count = manifest["task_counts"][args.stream]["total"]
    return {
        "dry_run": True,
        "schema": RESULT_SCHEMA,
        "arm_name": args.arm_name,
        "stream": args.stream,
        "manifest": str(manifest_path),
        "candidate_task_count": candidate_count,
        "queue": {
            "fixed_once": True,
            "wrap": False,
            "dev": {
                "submission_window_seconds": 300,
                "may_submit_fewer_than_candidate_count": True,
                "denominator": (
                    "first measured submit to max(first submit + 300s, last submitted "
                    "client terminal, first post-close server-idle acknowledgement)"
                ),
            },
            "final": {
                "submission_window_seconds": None,
                "must_submit_every_candidate_once": True,
                "required_submitted": 50,
                "denominator": (
                    "first measured submit to max(last client terminal, first server-idle "
                    "acknowledgement after all 50 terminals)"
                ),
            },
        },
        "frozen_policy": frozen_policy(args, manifest),
        "runtime_sequence_capacity_check": (
            "Before server start, exact model-rendered prompt tokens + selected family cap "
            f"must be <= {args.max_sequence_length} for every warmup and measured task."
        ),
        "server": {
            "command": command,
            "environment_overrides": environment_overrides,
            "freetoken_root": str(root),
            "python": python,
            "git_commit": git_commit(root),
            "git_dirty": "not inspected in zero-write dry-run",
        },
        "offline_scorer": {
            "command_prefix": [
                lcb_python,
                str(HERE / "score_livecodebench_subset.py"),
                "--lcb-root",
                str(lcb_root),
                "--source-cache",
                str(source_cache),
            ],
            "dataset_revision": DATASET_REVISION,
            "runner_revision": RUNNER_REVISION,
            "included_in_online_denominator": False,
        },
        "output": str(output),
        "server_log": str(log),
        "starts_server_or_gpu": False,
        "starts_judge": False,
        "writes_files": False,
        "per_task_stdout": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    thinking_by_family = manifest["request_policy"]["thinking_by_family"]
    warmup_tasks = load_tasks(
        resolve_task_path(manifest_path, manifest, "warmup"), "warmup"
    )
    measured_tasks = load_tasks(
        resolve_task_path(manifest_path, manifest, args.stream), args.stream
    )
    warmup_users = validate_assignments(manifest, "warmup", warmup_tasks)
    measured_users = validate_assignments(manifest, args.stream, measured_tasks)
    root = args.freetoken_root.expanduser().resolve()
    if not (root / "python" / "freetoken" / "cli.py").is_file():
        raise FileNotFoundError(f"not a FreeToken source checkout: {root}")
    lcb_root = args.lcb_root.expanduser().resolve()
    if not (lcb_root / "lcb_runner" / "evaluation" / "compute_code_generation_metrics.py").is_file():
        raise FileNotFoundError(f"not a LiveCodeBench checkout: {lcb_root}")
    observed_lcb_commit = git_commit(lcb_root)
    if observed_lcb_commit != RUNNER_REVISION:
        raise ValueError(
            f"LiveCodeBench checkout must be {RUNNER_REVISION}; found {observed_lcb_commit}"
        )
    python = serving.resolve_python(args.python_executable)
    lcb_python = serving.resolve_python(args.lcb_python)
    model = serving.model_argument(args.model)
    command = serving.server_command(args, python, model)
    environment, environment_overrides = serving.server_environment(args, root)
    source_cache = source_cache_path(args, manifest)
    output, log = output_paths(args)
    policy = frozen_policy(args, manifest)
    if args.dry_run:
        print(
            json.dumps(
                dry_run_payload(
                    args=args,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    root=root,
                    python=python,
                    lcb_python=lcb_python,
                    lcb_root=lcb_root,
                    source_cache=source_cache,
                    command=command,
                    environment_overrides=environment_overrides,
                    output=output,
                    log=log,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    token_counter = serving.ClientTokenCounter(model)
    prepare_prompts(warmup_tasks.values(), manifest, token_counter)
    prepare_prompts(measured_tasks.values(), manifest, token_counter)
    capacity = validate_sequence_capacity(
        [*warmup_tasks.values(), *measured_tasks.values()],
        token_caps(args),
        args.max_sequence_length,
    )
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
    result: dict[str, Any] | None = None
    records: list[dict[str, Any]] = []
    warmup_records: list[dict[str, Any]] | None = None
    failure: BaseException | None = None
    shutdown_error: str | None = None
    poller: runtime.StatsPoller | None = None
    first_submit: float | None = None
    first_submit_wall: float | None = None
    verifier: dict[str, Any] | None = None
    try:
        server.start()
        assert server.model_id is not None
        warmup_records = run_warmup(
            args=args,
            users=warmup_users,
            tasks=warmup_tasks,
            model_id=server.model_id,
            caps=token_caps(args),
            thinking_by_family=thinking_by_family,
        )
        warmup_idle = server.wait_idle(args.request_timeout)
        if len(warmup_records) != 2 or any(
            not finish_valid(record) for record in warmup_records
        ):
            counts = Counter(record["terminal_reason"] for record in warmup_records)
            raise RuntimeError(f"warmup did not complete naturally: {dict(counts)}")
        if warmup_idle.get("requests", {}).get("active") != 0:
            raise RuntimeError("server did not drain after warmup")
        stats_before = server.stats()
        clock_perf = time.perf_counter()
        clock_wall = time.time()
        poller = runtime.StatsPoller(server, args.stats_poll_seconds)
        poller.start()
        records, gate, queue_exhausted = run_measured(
            args=args,
            users=measured_users,
            tasks=measured_tasks,
            model_id=server.model_id,
            caps=token_caps(args),
            thinking_by_family=thinking_by_family,
        )
        if not records or gate.start is None:
            raise RuntimeError("measured workload submitted no tasks")
        first_submit = gate.start
        first_submit_wall = clock_wall + (first_submit - clock_perf)
        drain_stats = server.wait_idle(args.request_timeout)
        idle_ack = time.perf_counter()
        poller.stop()
        last_terminal = max(record["_terminal_perf"] for record in records)
        if args.stream == "dev":
            if gate.end is None:
                raise RuntimeError("development submission window was never established")
            denominator_end = max(gate.end, last_terminal, idle_ack)
        else:
            denominator_end = max(last_terminal, idle_ack)
        denominator = denominator_end - first_submit
        invalid_terminals = [record for record in records if not finish_valid(record)]
        if invalid_terminals:
            reasons = Counter(record["terminal_reason"] for record in invalid_terminals)
            raise RuntimeError(f"measured request failure invalidated run: {dict(reasons)}")
        if args.stream == "final" and len(records) != 50:
            raise RuntimeError(f"final submitted {len(records)} tasks instead of all 50")
        server.stop()
        for record in records:
            if record["family"] == "aime":
                parse_aime(record, measured_tasks[record["task_id"]])
        verifier = run_lcb_scorer(
            args=args,
            lcb_python=lcb_python,
            lcb_root=lcb_root,
            source_cache=source_cache,
            records=records,
            tasks=measured_tasks,
        )
        attach_metrics(records, measured_tasks, token_counter)
        summary = build_summary(records, denominator, args.stream)
        token_delta = serving.stats_token_delta(stats_before, drain_stats)
        summary["server_token_throughput_per_second"] = {
            "prompt_tokens": (
                token_delta["prompt_tokens_total"] / denominator
                if token_delta.get("prompt_tokens_total") is not None and denominator > 0
                else None
            ),
            "completion_tokens": (
                token_delta["completion_tokens_total"] / denominator
                if token_delta.get("completion_tokens_total") is not None and denominator > 0
                else None
            ),
        }
        candidate_count = len(measured_tasks)
        unsubmitted = candidate_count - len(records)
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "stream": args.stream,
            "valid_benchmark_run": True,
            "invalid_reason": None,
            "manifest": str(manifest_path),
            "source_provenance": manifest["sources"],
            "training_forbidden_texts": manifest["training_forbidden_texts"],
            "frozen_policy": policy,
            "sequence_capacity_preflight": capacity,
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
                **warmup_summary(warmup_records),
                "fully_drained": warmup_idle.get("requests", {}).get("active") == 0,
                "stats_after": warmup_idle,
            },
            "queue": {
                "mode": (
                    "fixed_300_second_submission_window"
                    if args.stream == "dev"
                    else "complete_fixed_queue"
                ),
                "candidate_tasks": candidate_count,
                "submitted_tasks": len(records),
                "unsubmitted_tasks": unsubmitted,
                "fixed_once": True,
                "wrapped": False,
                "all_user_queues_exhausted": queue_exhausted,
                "window_closed_rejections": gate.closed_rejections,
                "final_all_tasks_submitted": args.stream == "final" and unsubmitted == 0,
            },
            "online_timing": {
                "measurement_start_unix_seconds": first_submit_wall,
                "first_submit_seconds": 0.0,
                "submission_window_seconds": (
                    DEV_WINDOW_SECONDS if args.stream == "dev" else None
                ),
                "submission_window_end_seconds": (
                    gate.end - first_submit if gate.end is not None else None
                ),
                "last_client_terminal_seconds": last_terminal - first_submit,
                "first_server_idle_ack_seconds": idle_ack - first_submit,
                "denominator_end_seconds": denominator_end - first_submit,
                "denominator_seconds": denominator,
                "denominator_definition": manifest["metrics"]["denominator"],
                "offline_scorer_included": False,
            },
            "verifier": verifier,
            "summary": summary,
            "server_observability": {
                "stats_before": stats_before,
                "stats_after_drain": drain_stats,
                "request_counter_delta": token_delta,
                "peak_observed_vram_bytes": poller.peak_vram_bytes,
                "peak_observed_vram_source": "/v1/stats vram_bytes polling",
                "poll_samples": poller.samples,
                "poll_error": poller.error,
            },
            "records": [
                public_record(record, first_submit, first_submit_wall)
                for record in sorted(records, key=lambda row: row["stream_index"])
            ],
        }
    except BaseException as exc:
        failure = exc
        if first_submit is None and records:
            first_submit = min(record["_submitted_perf"] for record in records)
            first_submit_wall = time.time()
        partial_records = []
        if first_submit is not None and first_submit_wall is not None:
            partial_records = [
                public_record(record, first_submit, first_submit_wall)
                for record in sorted(records, key=lambda row: row["stream_index"])
            ]
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "stream": args.stream,
            "valid_benchmark_run": False,
            "invalid_reason": f"{type(exc).__name__}: {exc}",
            "manifest": str(manifest_path),
            "frozen_policy": policy,
            "warmup": warmup_summary(warmup_records),
            "verifier": verifier,
            "server": {
                "command": command,
                "environment_overrides": environment_overrides,
                "log": str(log),
                "shutdown_error": None,
            },
            "partial_terminal_record_count": len(records),
            "records": partial_records,
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
            result["valid_benchmark_run"] = False
            result["invalid_reason"] = result.get("invalid_reason") or shutdown_error
        serving.write_json(output, result)

    print(
        json.dumps(
            {
                "output": str(output),
                "stream": args.stream,
                "valid_benchmark_run": result["valid_benchmark_run"],
                "invalid_reason": result.get("invalid_reason"),
                "summary": result.get("summary"),
                "per_task_stdout": False,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if failure is not None:
        return 2
    return 0 if result["valid_benchmark_run"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
