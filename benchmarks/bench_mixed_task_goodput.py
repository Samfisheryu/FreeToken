#!/usr/bin/env python3
"""Measure frozen 20-user mixed online task goodput for one FreeToken arm.

The runner starts one server, runs the fixed 20-request staggered warmup to
drain, and then preserves that cache while driving either the five-minute
development stream or ten-minute final stream.  Users keep at most one request
in flight and wait 30 seconds after every terminal.  No request is submitted at
or after the fixed window end; already submitted requests fully drain.

EvalPlus scoring runs only after the online denominator has closed.  Pass
``--server-args`` last; every following token belongs to ``ft serve``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from decimal import Decimal, InvalidOperation
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
from typing import Any, Callable, Iterable

if __package__:
    from . import bench_aime_task_goodput as serving
else:
    import bench_aime_task_goodput as serving


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = Path(
    "/data1/lmcache_kv/goodput_campaign/mixed_workload_sources_v1/manifest.json"
)
RESULT_SCHEMA = "freetoken.mixed_task_goodput_result.v1"
MANIFEST_SCHEMA = "freetoken.mixed_task_goodput_manifest.v1"
TASK_SCHEMA = "freetoken.mixed_task_goodput_task.v1"
FAMILIES = ("numeric", "code", "knowledge")
NUMERIC_RE = re.compile(
    r"\s*FINAL:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*\Z"
)
KNOWLEDGE_RE = re.compile(r"\s*FINAL:\s*([ABCD])\s*\Z")
CODE_RE = re.compile(r"\s*```python[ \t]*\r?\n([\s\S]*?)\r?\n```\s*\Z")


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
        default=300.0,
        help="hard transport timeout; family SLOs remain fixed at 30/60/10 seconds",
    )
    parser.add_argument("--numeric-max-tokens", type=int, default=512)
    parser.add_argument("--code-max-tokens", type=int, default=768)
    parser.add_argument("--knowledge-max-tokens", type=int, default=64)
    parser.add_argument("--seed-base", type=int, default=20260901)
    parser.add_argument(
        "--evalplus-python",
        default=sys.executable,
        help="Python executable containing EvalPlus 0.3.1",
    )
    parser.add_argument(
        "--evalplus-pythonpath",
        action="append",
        default=[],
        metavar="PATH",
        help="prepend an EvalPlus 0.3.1 source checkout; repeat as needed",
    )
    parser.add_argument("--evalplus-timeout", type=float, default=3600.0)
    parser.add_argument("--stats-poll-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="remaining arguments for `ft serve`; this option must be last",
    )
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in (
        "server_timeout",
        "shutdown_timeout",
        "request_timeout",
        "evalplus_timeout",
        "stats_poll_seconds",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    caps = {"numeric": 512, "code": 768, "knowledge": 64}
    for family, cap in caps.items():
        value = getattr(args, f"{family}_max_tokens")
        if not 1 <= value <= cap:
            parser.error(f"--{family}-max-tokens must be in 1..{cap}")
    if args.request_timeout <= 60:
        parser.error("--request-timeout must be greater than the largest fixed family SLO (60s)")
    args.server_env = serving.parse_environment(args.env, parser)
    serving.validate_server_args(args.server_args, parser)
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"unsupported mixed workload manifest: {path}")
    traffic = manifest.get("traffic")
    policy = manifest.get("request_policy")
    if not isinstance(traffic, dict) or not isinstance(policy, dict):
        raise ValueError("mixed workload manifest needs traffic and request_policy objects")
    frozen = {
        "user_count": 20,
        "think_time_seconds": 30,
        "warmup_task_count": 20,
    }
    for field, expected in frozen.items():
        if traffic.get(field) != expected:
            raise ValueError(f"traffic.{field} must remain {expected}")
    for stream, seconds, turns, family_count in (
        ("dev", 300, 12, 80),
        ("final", 600, 21, 140),
    ):
        contract = traffic.get(stream)
        if not isinstance(contract, dict):
            raise ValueError(f"traffic.{stream} is missing")
        expected = (seconds, turns, family_count)
        actual = (
            contract.get("submission_window_seconds"),
            contract.get("turns_per_user"),
            contract.get("tasks_per_family"),
        )
        if actual != expected:
            raise ValueError(f"traffic.{stream} changed: expected {expected}, got {actual}")
    if policy.get("system_prompt") != "Follow the requested output format exactly.":
        raise ValueError("the common system prompt changed")
    if policy.get("greedy") is not True or policy.get("enable_thinking") is not False:
        raise ValueError("mixed workload requires greedy decoding with thinking disabled")
    families = policy.get("families")
    for family, cap, slo in (("numeric", 512, 30), ("code", 768, 60), ("knowledge", 64, 10)):
        if not isinstance(families, dict) or not isinstance(families.get(family), dict):
            raise ValueError(f"request policy omits {family}")
        if families[family].get("max_tokens_cap") != cap or families[family].get("slo_seconds") != slo:
            raise ValueError(f"request policy changed the {family} cap/SLO")
    return manifest


def load_tasks(path: Path) -> dict[str, dict[str, Any]]:
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
            if not isinstance(task_id, str) or not task_id or task_id in tasks:
                raise ValueError(f"invalid or duplicate task_id at {path}:{line_number}")
            if family not in FAMILIES or not isinstance(row.get("task_text"), str):
                raise ValueError(f"invalid task family/text at {path}:{line_number}")
            if not isinstance(row.get("reference"), dict):
                raise ValueError(f"task reference missing at {path}:{line_number}")
            tasks[task_id] = row
    if not tasks:
        raise ValueError(f"task stream is empty: {path}")
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
            expected_family = FAMILIES[(user_index + turn_index) % 3]
            if tasks[task_id]["family"] != expected_family:
                raise ValueError(
                    f"{split} user {user_index} turn {turn_index} expected {expected_family}"
                )
        assigned.extend(task_ids)
    if len(assigned) != len(set(assigned)) or set(assigned) != set(tasks):
        raise ValueError(f"{split} assignments must consume every task exactly once")
    return users


def resolve_task_path(manifest_path: Path, manifest: dict[str, Any], split: str) -> Path:
    relative = manifest.get("task_files", {}).get(split)
    if not isinstance(relative, str):
        raise ValueError(f"manifest omits task_files.{split}")
    path = (manifest_path.parent / relative).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def prepare_prompts(
    tasks: Iterable[dict[str, Any]], manifest: dict[str, Any], token_counter: Any
) -> None:
    policy = manifest["request_policy"]
    for task in tasks:
        instruction = policy["families"][task["family"]]["instruction"]
        user_text = task["task_text"].rstrip() + "\n\n" + instruction
        messages = [
            {"role": "system", "content": policy["system_prompt"]},
            {"role": "user", "content": user_text},
        ]
        task["messages"] = messages
        task["prompt_token_observation"] = token_counter.prompt_observation(
            messages, {"chat_template_kwargs": {"enable_thinking": False}}
        )


def max_tokens_by_family(args: argparse.Namespace) -> dict[str, int]:
    return {family: int(getattr(args, f"{family}_max_tokens")) for family in FAMILIES}


def base_record(
    *, phase: str, user_index: int, turn_index: int, task: dict[str, Any], seed: int
) -> dict[str, Any]:
    return {
        "record_id": f"{phase}:{user_index}:{turn_index}:{task['task_id']}",
        "phase": phase,
        "user_id": f"user-{user_index:02d}",
        "user_index": user_index,
        "turn_index": turn_index,
        "task_id": task["task_id"],
        "family": task["family"],
        "source": task["source"],
        "source_split": task["source_split"],
        "request_seed": seed,
        "finish_reason": None,
        "terminal_reason": None,
        "error": None,
        "raw_output": "",
        "reasoning_output": "",
        "content_output": "",
        "server_usage": None,
        "judge_correct": False,
        "slo_success": False,
        "parse_status": None,
        "verifier": None,
        "_scheduled_perf": None,
        "_submitted_perf": None,
        "_first_text_perf": None,
        "_last_text_perf": None,
        "_terminal_perf": None,
    }


def event_pieces(event: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        for field, channel in (("reasoning_content", "reasoning"), ("content", "content")):
            value = delta.get(field)
            if isinstance(value, str) and value:
                yield channel, value


def run_task(
    *,
    args: argparse.Namespace,
    phase: str,
    user_index: int,
    turn_index: int,
    task: dict[str, Any],
    model_id: str,
    scheduled_perf: float,
    seed: int,
    max_tokens: int,
    on_submit: Callable[[float], bool] | None = None,
) -> dict[str, Any]:
    record = base_record(
        phase=phase, user_index=user_index, turn_index=turn_index, task=task, seed=seed
    )
    record["_scheduled_perf"] = scheduled_perf
    body = {
        "model": model_id,
        "messages": task["messages"],
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    connection = None
    response = None
    try:
        connection = serving.http.client.HTTPConnection(
            args.host, args.port, timeout=args.request_timeout
        )
        record["_submitted_perf"] = time.perf_counter()
        if on_submit is not None and not on_submit(record["_submitted_perf"]):
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
                    record["terminal_reason"] = (
                        f"server_{record['finish_reason']}"
                        if record["finish_reason"]
                        else "stream_eof"
                    )
                    break
                if event is serving._SSE_DONE:
                    record["terminal_reason"] = (
                        f"server_{record['finish_reason']}"
                        if record["finish_reason"]
                        else "server_done"
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
                    if choice.get("finish_reason"):
                        record["finish_reason"] = str(choice["finish_reason"])
                for channel, piece in event_pieces(event):
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
    return record


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
        scheduled = origin + user_index
        wait_until(scheduled)
        task = tasks[user["task_ids"][0]]
        return run_task(
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

    def first_turn_schedule(self, user_index: int, provisional_origin: float) -> float:
        if user_index == 0:
            return provisional_origin
        with self.condition:
            while self.start is None:
                self.condition.wait(timeout=0.1)
            return self.start + user_index

    def wait_for_turn(self, *, user_index: int, turn_index: int, scheduled: float) -> bool:
        if user_index == 0 and turn_index == 0:
            wait_until(scheduled)
            return True
        with self.condition:
            while self.start is None:
                self.condition.wait(timeout=0.1)
            assert self.end is not None
            window_end = self.end
        if scheduled >= window_end:
            return False
        wait_until(scheduled)
        return time.perf_counter() < window_end

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
    schedule_origin = time.perf_counter()
    window = MeasurementWindow(window_seconds)
    sink: list[dict[str, Any]] = []
    sink_lock = threading.Lock()

    def user_loop(user: dict[str, Any]) -> None:
        user_index = int(user["user_index"])
        scheduled = window.first_turn_schedule(user_index, schedule_origin)
        last_terminal: float | None = None
        for turn_index, task_id in enumerate(user["task_ids"]):
            if turn_index > 0:
                assert last_terminal is not None
                scheduled = last_terminal + 30.0
            if not window.wait_for_turn(
                user_index=user_index, turn_index=turn_index, scheduled=scheduled
            ):
                return
            task = tasks[task_id]
            callback = lambda submitted, u=user_index, k=turn_index: window.claim_submission(
                u, k, submitted
            )
            record = run_task(
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
            last_terminal = record["_terminal_perf"]
            with sink_lock:
                sink.append(record)
        if last_terminal is not None:
            window.record_exhaustion(user_index, last_terminal + 30.0)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(user_loop, user) for user in users]
        for future in futures:
            future.result()
    with sink_lock:
        records = list(sink)
    return records, window


class StatsPoller:
    def __init__(self, server: Any, interval: float) -> None:
        self.server = server
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.samples = 0
        self.peak_vram_bytes: int | None = None
        self.error: str | None = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._run, name="mixed-goodput-stats", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                stats = self.server.stats()
                value = stats.get("vram_bytes")
                if isinstance(value, int):
                    self.peak_vram_bytes = (
                        value if self.peak_vram_bytes is None else max(self.peak_vram_bytes, value)
                    )
                self.samples += 1
            except Exception as exc:
                if self.error is None:
                    self.error = f"{type(exc).__name__}: {exc}"
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=max(1.0, self.interval * 3))


def finish_valid(record: dict[str, Any]) -> bool:
    if record["error"] is not None:
        return False
    if record["terminal_reason"] in {"request_timeout", "http_error", "stream_error", "stream_eof"}:
        return False
    return record["finish_reason"] in {None, "stop"}


def parse_non_code(record: dict[str, Any], task: dict[str, Any]) -> None:
    if not finish_valid(record):
        record["parse_status"] = "invalid_finish"
        return
    text = record["raw_output"]
    if task["family"] == "numeric":
        match = NUMERIC_RE.fullmatch(text)
        if match is None:
            record["parse_status"] = "format_error"
            return
        try:
            observed = Decimal(match.group(1))
            expected = Decimal(str(task["reference"]["value"]))
        except InvalidOperation:
            record["parse_status"] = "number_error"
            return
        if not observed.is_finite():
            record["parse_status"] = "number_error"
            return
        record["parse_status"] = "parsed"
        record["judge_correct"] = observed == expected
    elif task["family"] == "knowledge":
        match = KNOWLEDGE_RE.fullmatch(text)
        if match is None:
            record["parse_status"] = "format_error"
            return
        record["parse_status"] = "parsed"
        record["judge_correct"] = match.group(1) == task["reference"]["value"]


def parse_code(record: dict[str, Any]) -> str | None:
    if not finish_valid(record):
        record["parse_status"] = "invalid_finish"
        return None
    match = CODE_RE.fullmatch(record["raw_output"])
    if match is None or not match.group(1).strip() or "```" in match.group(1):
        record["parse_status"] = "format_error"
        return None
    record["parse_status"] = "parsed"
    return match.group(1)


def run_evalplus(
    *,
    args: argparse.Namespace,
    evalplus_python: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    records: list[dict[str, Any]],
    tasks: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    items = []
    for record in records:
        if record["family"] != "code":
            continue
        solution = parse_code(record)
        if solution is None:
            continue
        reference = tasks[record["task_id"]]["reference"]
        items.append(
            {
                "record_id": record["record_id"],
                "dataset": reference["dataset"],
                "task_id": record["task_id"],
                "solution": solution,
            }
        )
    if not items:
        return {
            "status": "complete",
            "python": evalplus_python,
            "submitted_to_verifier": 0,
            "error": None,
        }
    source_files = manifest["source_files"]
    environment = os.environ.copy()
    environment["HUMANEVAL_OVERRIDE_PATH"] = str(
        (manifest_path.parent / source_files["humaneval_plus"]).resolve()
    )
    environment["MBPP_OVERRIDE_PATH"] = str(
        (manifest_path.parent / source_files["mbpp_plus"]).resolve()
    )
    if args.evalplus_pythonpath:
        additions = [str(Path(item).expanduser().resolve()) for item in args.evalplus_pythonpath]
        existing = environment.get("PYTHONPATH")
        if existing:
            additions.append(existing)
        environment["PYTHONPATH"] = os.pathsep.join(additions)
    with tempfile.TemporaryDirectory(prefix="freetoken-evalplus-") as temporary:
        root = Path(temporary)
        input_path = root / "input.json"
        output_path = root / "output.json"
        serving.write_json(
            input_path,
            {"schema": "freetoken.evalplus_subset_input.v1", "items": items},
        )
        completed = subprocess.run(
            [
                evalplus_python,
                str(HERE / "score_evalplus_subset.py"),
                "--input",
                str(input_path),
                "--output",
                str(output_path),
            ],
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=args.evalplus_timeout,
        )
        if completed.returncode != 0 or not output_path.is_file():
            detail = completed.stdout[-8000:]
            raise RuntimeError(
                f"EvalPlus scorer exited {completed.returncode}; output tail:\n{detail}"
            )
        result = load_json(output_path)
    if result.get("schema") != "freetoken.evalplus_subset_result.v1":
        raise ValueError("EvalPlus scorer returned an unsupported schema")
    by_record = {item["record_id"]: item for item in result.get("items", [])}
    if set(by_record) != {item["record_id"] for item in items}:
        raise ValueError("EvalPlus scorer did not return every submitted record exactly once")
    for record in records:
        if record["record_id"] not in by_record:
            continue
        verdict = by_record[record["record_id"]]
        record["verifier"] = {
            "base_status": verdict["base_status"],
            "plus_status": verdict["plus_status"],
        }
        record["judge_correct"] = bool(verdict["passed"])
    return {
        "status": "complete",
        "python": evalplus_python,
        "submitted_to_verifier": len(items),
        "evalplus_version": result.get("evalplus_version"),
        "dataset_versions": result.get("dataset_versions"),
        "error": None,
    }


def attach_metrics(
    records: list[dict[str, Any]], tasks: dict[str, dict[str, Any]], token_counter: Any
) -> None:
    for record in records:
        task = tasks[record["task_id"]]
        record["latency_seconds"] = record["_terminal_perf"] - record["_submitted_perf"]
        record["deadline_seconds"] = float(
            {"numeric": 30, "code": 60, "knowledge": 10}[record["family"]]
        )
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
        completion_tokens = (
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        record["output_token_observation"] = token_counter.output_observation(
            record["raw_output"]
        )
        if not isinstance(completion_tokens, int):
            completion_tokens = int(record["output_token_observation"]["tokens"])
        record["completion_tokens"] = completion_tokens
        record["prompt_token_observation"] = task["prompt_token_observation"]
        record["tpot_seconds"] = (
            (record["_last_text_perf"] - record["_first_text_perf"])
            / (completion_tokens - 1)
            if completion_tokens > 1
            and record["_first_text_perf"] is not None
            and record["_last_text_perf"] is not None
            else None
        )


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    present = [float(value) for value in values if value is not None]
    return {
        "count": len(present),
        "mean": sum(present) / len(present) if present else None,
        "p50": percentile(present, 0.50),
        "p95": percentile(present, 0.95),
        "p99": percentile(present, 0.99),
    }


def summarize_group(records: list[dict[str, Any]], denominator: float) -> dict[str, Any]:
    submitted = len(records)
    correct = sum(bool(record["judge_correct"]) for record in records)
    successes = sum(bool(record["slo_success"]) for record in records)
    reasons: dict[str, int] = {}
    for record in records:
        key = str(record["terminal_reason"])
        reasons[key] = reasons.get(key, 0) + 1
    return {
        "submitted": submitted,
        "judge_correct": correct,
        "slo_success": successes,
        "raw_accuracy": correct / submitted if submitted else 0.0,
        "slo_success_rate": successes / submitted if submitted else 0.0,
        "goodput_per_second": successes / denominator if denominator > 0 else 0.0,
        "goodput_per_hour": successes / denominator * 3600.0 if denominator > 0 else 0.0,
        "latency_seconds": distribution(record["latency_seconds"] for record in records),
        "ttft_seconds": distribution(record["ttft_seconds"] for record in records),
        "tpot_seconds": distribution(record["tpot_seconds"] for record in records),
        "output_tokens": {
            "total": sum(int(record["completion_tokens"]) for record in records),
            "per_second": (
                sum(int(record["completion_tokens"]) for record in records) / denominator
                if denominator > 0
                else 0.0
            ),
            **distribution(float(record["completion_tokens"]) for record in records),
        },
        "terminal_reason_counts": reasons,
    }


def concurrency_summary(records: list[dict[str, Any]], denominator: float) -> dict[str, Any]:
    events = [
        event
        for record in records
        for event in (
            (record["_submitted_perf"], 1),
            (record["_terminal_perf"], -1),
        )
    ]
    events.sort(key=lambda item: (item[0], item[1]))
    active = 0
    peak = 0
    area = 0.0
    previous = events[0][0] if events else 0.0
    for timestamp, delta in events:
        area += active * (timestamp - previous)
        active += delta
        peak = max(peak, active)
        previous = timestamp
    return {
        "peak_inflight": peak,
        "average_inflight_over_denominator": area / denominator if denominator > 0 else 0.0,
    }


def public_record(record: dict[str, Any], start: float, wall_start: float) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if not key.startswith("_")}
    for public, private in (
        ("scheduled_at", "_scheduled_perf"),
        ("submitted_at", "_submitted_perf"),
        ("first_text_at", "_first_text_perf"),
        ("last_text_at", "_last_text_perf"),
        ("terminal_at", "_terminal_perf"),
    ):
        value = record.get(private)
        result[public + "_seconds"] = value - start if value is not None else None
        result[public + "_unix_seconds"] = wall_start + value - start if value is not None else None
    result["submit_lag_seconds"] = record["_submitted_perf"] - record["_scheduled_perf"]
    return result


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_arm = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.arm_name).strip("_") or "arm"
    output = args.output or HERE / "results" / f"mixed_task_goodput_{args.stream}_{safe_arm}_{stamp}.json"
    output = output.expanduser().resolve()
    log = args.server_log or output.with_suffix(".server.log")
    return output, log.expanduser().resolve()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest_path = args.manifest.expanduser().resolve()
    manifest = load_manifest(manifest_path)
    warmup_tasks = load_tasks(resolve_task_path(manifest_path, manifest, "warmup"))
    measured_tasks = load_tasks(resolve_task_path(manifest_path, manifest, args.stream))
    warmup_users = validate_assignments(manifest, "warmup", warmup_tasks)
    measured_users = validate_assignments(manifest, args.stream, measured_tasks)
    root = args.freetoken_root.expanduser().resolve()
    if not (root / "python" / "freetoken" / "cli.py").is_file():
        raise FileNotFoundError(f"not a FreeToken source checkout: {root}")
    python = serving.resolve_python(args.python_executable)
    evalplus_python = serving.resolve_python(args.evalplus_python)
    model = serving.model_argument(args.model)
    command = serving.server_command(args, python, model)
    environment, environment_overrides = serving.server_environment(args, root)
    output, log = output_paths(args)
    token_caps = max_tokens_by_family(args)
    window_seconds = float(manifest["traffic"][args.stream]["submission_window_seconds"])
    frozen_policy = {
        "system_prompt": manifest["request_policy"]["system_prompt"],
        "family_instructions": {
            family: manifest["request_policy"]["families"][family]["instruction"]
            for family in FAMILIES
        },
        "sampling": {"temperature": 0.0, "top_p": 1.0, "top_k": -1},
        "enable_thinking": False,
        "max_tokens": token_caps,
        "slo_seconds": {"numeric": 30, "code": 60, "knowledge": 10},
        "hard_request_timeout_seconds": args.request_timeout,
        "seed_base": args.seed_base,
    }
    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "schema": RESULT_SCHEMA,
                    "arm_name": args.arm_name,
                    "stream": args.stream,
                    "manifest": str(manifest_path),
                    "task_counts": manifest["task_counts"],
                    "traffic": manifest["traffic"],
                    "frozen_policy": frozen_policy,
                    "server": {
                        "command": command,
                        "environment_overrides": environment_overrides,
                        "freetoken_git": serving.git_info(root),
                        "benchmark_git": serving.git_info(REPO),
                    },
                    "evalplus": {
                        "python": evalplus_python,
                        "pythonpath": args.evalplus_pythonpath,
                        "version": "0.3.1",
                        "commit": manifest["sources"]["code"]["evaluator_commit"],
                        "timing": "after online drain; excluded from denominator",
                    },
                    "output": str(output),
                    "server_log": str(log),
                    "per_task_stdout": False,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    token_counter = serving.ClientTokenCounter(model)
    prepare_prompts(warmup_tasks.values(), manifest, token_counter)
    prepare_prompts(measured_tasks.values(), manifest, token_counter)
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
    failure: BaseException | None = None
    shutdown_error: str | None = None
    poller: StatsPoller | None = None
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
        warmup_idle = server.wait_idle(args.request_timeout)
        stats_before = server.stats()
        clock_perf = time.perf_counter()
        clock_wall = time.time()
        poller = StatsPoller(server, args.stats_poll_seconds)
        poller.start()
        records, window = run_measured(
            args=args,
            users=measured_users,
            tasks=measured_tasks,
            model_id=server.model_id,
            token_caps=token_caps,
            window_seconds=window_seconds,
        )
        drain_stats = server.wait_idle(args.request_timeout)
        first_server_idle_ack = time.perf_counter()
        poller.stop()
        if window.start is None or window.end is None:
            raise RuntimeError("user 0 never established the measured submission window")
        measurement_wall = clock_wall + (window.start - clock_perf)
        if not records:
            raise RuntimeError("the measured stream submitted no tasks")
        last_terminal = max(record["_terminal_perf"] for record in records)
        denominator_end = max(window.end, last_terminal, first_server_idle_ack)
        denominator = denominator_end - window.start
        for record in records:
            task = measured_tasks[record["task_id"]]
            if record["family"] != "code":
                parse_non_code(record, task)
        verifier = run_evalplus(
            args=args,
            evalplus_python=evalplus_python,
            manifest_path=manifest_path,
            manifest=manifest,
            records=records,
            tasks=measured_tasks,
        )
        attach_metrics(records, measured_tasks, token_counter)
        by_family = {
            family: summarize_group(
                [record for record in records if record["family"] == family], denominator
            )
            for family in FAMILIES
        }
        summary = {
            "denominator_seconds": denominator,
            "submission_window_seconds": window_seconds,
            "drain_tail_seconds": max(0.0, denominator_end - window.end),
            "total": summarize_group(records, denominator),
            "families": by_family,
            "concurrency": concurrency_summary(records, denominator),
        }
        token_delta = serving.stats_token_delta(stats_before, drain_stats)
        summary["server_token_throughput_per_second"] = {
            field: (value / denominator if value is not None and denominator > 0 else None)
            for field, value in token_delta.items()
            if field in {"prompt_tokens_total", "completion_tokens_total"}
        }
        valid = not window.exhaustion
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "stream": args.stream,
            "valid_task_goodput": valid,
            "invalid_reason": "frozen per-user queue exhausted before window end" if not valid else None,
            "manifest": str(manifest_path),
            "source_provenance": manifest["sources"],
            "training_forbidden_texts": manifest["training_forbidden_texts"],
            "traffic": manifest["traffic"],
            "frozen_policy": frozen_policy,
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
                "task_count": len(warmup),
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
                "request_counter_delta": token_delta,
                "peak_observed_vram_bytes": poller.peak_vram_bytes,
                "peak_observed_vram_source": "/v1/stats vram_bytes polling",
                "poll_samples": poller.samples,
                "poll_error": poller.error,
                "expert_cache_misses": None,
                "h2d_bytes": None,
                "missing_metric_reason": (
                    "the production /v1/stats schema does not expose expert-cache misses or H2D bytes"
                ),
            },
            "queue_exhaustion": window.exhaustion,
            "records": [
                public_record(record, window.start, measurement_wall)
                for record in sorted(records, key=lambda row: (row["user_index"], row["turn_index"]))
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
        "invalid_reason": result.get("invalid_reason"),
        "summary": result.get("summary"),
        "per_task_stdout": False,
    }
    print(json.dumps(stdout, indent=2, ensure_ascii=False))
    if failure is not None:
        return 2
    return 0 if result["valid_task_goodput"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
