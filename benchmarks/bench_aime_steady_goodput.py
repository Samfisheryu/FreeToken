#!/usr/bin/env python3
"""Measure steady-state AIME task goodput for one independently frozen arm.

The runner owns one FreeToken server and drives staggered closed-loop users from
an immutable local JSONL task stream.  Warmup continues until every user has a
terminal request.  A fixed submission window starts at that instant; requests
submitted before it remain live but never count.  The first streamed
``\\boxed{integer}`` is the task terminal and closes the real HTTP stream.

``--server-args`` consumes the remainder of the command line and must be last.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
from pathlib import Path
import re
import socket
import sys
import threading
import time
from typing import Any, Iterable

if __package__:
    from . import bench_aime_task_goodput as task_bench
else:
    import bench_aime_task_goodput as task_bench


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_WORKLOAD = HERE / "workloads" / "aime_steady_goodput_v1.json"
RESULT_SCHEMA = "freetoken.aime_steady_goodput_result.v1"
WORKLOAD_SCHEMA = "freetoken.aime_steady_goodput_workload.v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--arm-name", required=True, help="stable name for this frozen arm")
    parser.add_argument("--model", required=True, help="local checkpoint, FTW, or HF model id")
    parser.add_argument(
        "--freetoken-root",
        type=Path,
        default=REPO,
        help="FreeToken source checkout used by the spawned server",
    )
    parser.add_argument(
        "--python",
        dest="python_executable",
        default=sys.executable,
        help="Python executable from the arm's environment",
    )
    parser.add_argument(
        "--pythonpath",
        action="append",
        default=[],
        metavar="PATH",
        help="prepend an additional server import path; repeat as needed",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="extra server environment entry; repeat as needed",
    )
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES for this arm")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=1919)
    parser.add_argument("--server-timeout", type=float, default=1800.0)
    parser.add_argument("--shutdown-timeout", type=float, default=90.0)
    parser.add_argument("--abort-timeout", type=float, default=30.0)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--workload", type=Path, default=DEFAULT_WORKLOAD)
    parser.add_argument(
        "--task-jsonl",
        type=Path,
        default=(
            Path(os.environ["FREETOKEN_AIME_STEADY_JSONL"])
            if os.environ.get("FREETOKEN_AIME_STEADY_JSONL")
            else None
        ),
        help=(
            "pre-locked local JSONL task stream; may also be set through "
            "FREETOKEN_AIME_STEADY_JSONL"
        ),
    )
    parser.add_argument(
        "--user-count",
        type=int,
        help="closed-loop user count; workload default is 20",
    )
    parser.add_argument(
        "--start-cadence-seconds",
        "--cadence-seconds",
        dest="start_cadence_seconds",
        type=float,
        help="gap between consecutive users' first submissions; workload default is 10",
    )
    parser.add_argument(
        "--think-time-seconds",
        type=float,
        help="delay from one task terminal to that user's next submit; default is 30",
    )
    parser.add_argument(
        "--measurement-seconds",
        "--window-seconds",
        dest="measurement_seconds",
        type=float,
        help="measured submission-window duration; workload default is 1800",
    )
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--seed-base", type=int, default=20260831)
    parser.add_argument(
        "--system-prompt",
        default="",
        help="one global system message for every task; empty means no system message",
    )
    parser.add_argument(
        "--answer-instruction",
        "--user-suffix",
        dest="answer_instruction",
        default=None,
        help=(
            "one global user-message suffix; default is the workload instruction, "
            "and an empty string disables it"
        ),
    )
    parser.add_argument(
        "--request-extra-json",
        default='{"chat_template_kwargs":{"enable_thinking":true}}',
        help="extra chat request fields as a JSON object (measurement fields are reserved)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--server-log", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--server-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="all remaining tokens are appended to `ft serve`; this option must be last",
    )
    args = parser.parse_args(argv)

    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    for name in ("server_timeout", "shutdown_timeout", "abort_timeout", "request_timeout"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_tokens < 32:
        parser.error("--max-tokens must be at least 32 so disconnect-to-abort can be verified")
    if args.greedy and args.temperature is not None:
        parser.error("--greedy and --temperature cannot be used together")
    if args.top_p is not None and not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")

    try:
        request_extra = json.loads(args.request_extra_json)
    except json.JSONDecodeError as exc:
        parser.error(f"--request-extra-json is not valid JSON: {exc}")
    if not isinstance(request_extra, dict):
        parser.error("--request-extra-json must decode to an object")
    template_kwargs = request_extra.get("chat_template_kwargs")
    if template_kwargs is not None and not isinstance(template_kwargs, dict):
        parser.error("request_extra_json.chat_template_kwargs must be an object")
    overlap = sorted(task_bench.RESERVED_REQUEST_FIELDS.intersection(request_extra))
    if overlap:
        parser.error(
            "--request-extra-json cannot override frozen fields: " + ", ".join(overlap)
        )
    args.request_extra = request_extra
    args.server_env = task_bench.parse_environment(args.env, parser)
    task_bench.validate_server_args(args.server_args, parser)
    return args


def load_workload(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        workload = json.load(handle)
    if not isinstance(workload, dict) or workload.get("schema") != WORKLOAD_SCHEMA:
        raise ValueError(f"unsupported steady-goodput workload schema in {path}")
    defaults = workload.get("defaults")
    prompt_contract = workload.get("prompt_contract")
    if not isinstance(defaults, dict) or not isinstance(prompt_contract, dict):
        raise ValueError("steady workload must contain defaults and prompt_contract")
    for field in (
        "user_count",
        "start_cadence_seconds",
        "think_time_seconds",
        "measurement_seconds",
    ):
        if field not in defaults:
            raise ValueError(f"steady workload defaults omit {field}")
    return workload


def resolve_traffic(args: argparse.Namespace, workload: dict[str, Any]) -> dict[str, Any]:
    defaults = workload["defaults"]
    traffic = {
        "user_count": (
            int(defaults["user_count"]) if args.user_count is None else args.user_count
        ),
        "start_cadence_seconds": (
            float(defaults["start_cadence_seconds"])
            if args.start_cadence_seconds is None
            else args.start_cadence_seconds
        ),
        "think_time_seconds": (
            float(defaults["think_time_seconds"])
            if args.think_time_seconds is None
            else args.think_time_seconds
        ),
        "measurement_seconds": (
            float(defaults["measurement_seconds"])
            if args.measurement_seconds is None
            else args.measurement_seconds
        ),
    }
    if traffic["user_count"] < 1:
        raise ValueError("--user-count must be positive")
    if traffic["start_cadence_seconds"] < 0:
        raise ValueError("--start-cadence-seconds must be non-negative")
    if traffic["think_time_seconds"] < 0:
        raise ValueError("--think-time-seconds must be non-negative")
    if traffic["measurement_seconds"] <= 0:
        raise ValueError("--measurement-seconds must be positive")
    return traffic


def resolve_prompt_policy(
    args: argparse.Namespace, workload: dict[str, Any]
) -> dict[str, Any]:
    contract = workload["prompt_contract"]
    answer_instruction = (
        contract["answer_instruction"]
        if args.answer_instruction is None
        else args.answer_instruction
    )
    return {
        "system_prompt": args.system_prompt,
        "answer_instruction": answer_instruction,
        "answer_instruction_source": (
            "workload default" if args.answer_instruction is None else "--answer-instruction"
        ),
        "existing_instruction_marker": contract["append_instruction_when_missing"],
        "user_suffix_separator": "\n",
        "message_construction": [
            "Start with an empty messages list.",
            "If system_prompt is non-empty, append one system message.",
            "Strip trailing whitespace from the task problem.",
            "If answer_instruction is non-empty and its marker is absent, append one newline and the instruction.",
            "Append the resulting text as one user message; no task-specific policy is allowed.",
        ],
        "scope": "Identical global policy for every task in the stream.",
    }


def load_task_stream(path: Path) -> list[dict[str, Any]]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    tasks: list[dict[str, Any]] = []
    task_ids: set[str] = set()
    with resolved.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"task JSONL line {line_number} is invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"task JSONL line {line_number} is not an object")
            task_id = row.get("task_id")
            problem = row.get("problem")
            source = row.get("source")
            if not isinstance(task_id, str) or not task_id.strip():
                raise ValueError(f"task JSONL line {line_number} has no non-empty task_id")
            if task_id in task_ids:
                raise ValueError(f"duplicate task_id {task_id!r} on line {line_number}")
            if not isinstance(problem, str) or not problem.strip():
                raise ValueError(f"task JSONL line {line_number} has no non-empty problem")
            if not isinstance(source, str) or not source.strip():
                raise ValueError(f"task JSONL line {line_number} has no non-empty source")
            if "answer" not in row:
                raise ValueError(f"task JSONL line {line_number} has no answer")
            task_ids.add(task_id)
            tasks.append(
                {
                    "problem_index": len(tasks),
                    "task_id": task_id,
                    "problem": problem,
                    "reference_answer": task_bench.parse_reference_answer(row["answer"]),
                    "source": source,
                }
            )
    if not tasks:
        raise ValueError("task JSONL contains no tasks")
    return tasks


def prepare_task_prompts(
    tasks: Iterable[dict[str, Any]],
    prompt_policy: dict[str, Any],
    request_extra: dict[str, Any],
    token_counter: task_bench.ClientTokenCounter,
) -> None:
    for task in tasks:
        messages, appended = task_bench.build_messages(task["problem"], prompt_policy)
        task["messages"] = messages
        task["answer_instruction_appended"] = appended
        task["prompt_token_observation"] = token_counter.prompt_observation(
            messages, request_extra
        )


class SteadyState:
    def __init__(self, *, user_count: int, measurement_seconds: float) -> None:
        self.user_count = user_count
        self.measurement_seconds = measurement_seconds
        self.condition = threading.Condition()
        self.cancelled = False
        self.entered_users: set[int] = set()
        self.first_terminal_by_user: dict[int, float] = {}
        self.measurement_start: float | None = None
        self.measurement_end: float | None = None
        self.records: list[dict[str, Any]] = []
        self.exhaustion: dict[str, Any] | None = None
        self.thread_errors: list[dict[str, Any]] = []
        self.threads: list[threading.Thread] = []

    def wait_for_submission(self, scheduled_perf: float) -> bool:
        with self.condition:
            while True:
                if self.cancelled:
                    return False
                if self.measurement_end is not None and scheduled_perf >= self.measurement_end:
                    return False
                now = time.perf_counter()
                if now >= scheduled_perf:
                    return True
                timeout = scheduled_perf - now
                if self.measurement_end is not None:
                    timeout = min(timeout, max(0.001, self.measurement_end - now))
                self.condition.wait(timeout=timeout)

    def claim_submission(
        self,
        *,
        user_index: int,
        turn_index: int,
        task_index: int,
        task_count: int,
    ) -> tuple[str, float] | None:
        with self.condition:
            if self.cancelled:
                return None
            now = time.perf_counter()
            if self.measurement_end is not None and now >= self.measurement_end:
                return None
            phase = "warmup" if self.measurement_start is None else "measured"
            if task_index >= task_count:
                self.exhaustion = {
                    "phase": phase,
                    "user_index": user_index,
                    "turn_index": turn_index,
                    "requested_row_index": task_index,
                    "available_task_count": task_count,
                    "at_perf_counter": now,
                }
                self.cancelled = True
                self.condition.notify_all()
                return None
            self.entered_users.add(user_index)
            return phase, now

    def record_terminal(self, record: dict[str, Any]) -> None:
        with self.condition:
            self.records.append(record)
            user_index = int(record["user_index"])
            if user_index not in self.first_terminal_by_user:
                self.first_terminal_by_user[user_index] = record["_terminal_perf"]
            if (
                self.measurement_start is None
                and len(self.entered_users) == self.user_count
                and len(self.first_terminal_by_user) == self.user_count
            ):
                self.measurement_start = max(self.first_terminal_by_user.values())
                self.measurement_end = self.measurement_start + self.measurement_seconds
            self.condition.notify_all()

    def record_thread_error(self, user_index: int, exc: BaseException) -> None:
        with self.condition:
            self.thread_errors.append(
                {
                    "user_index": user_index,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            self.cancelled = True
            self.condition.notify_all()

    def cancel(self) -> None:
        with self.condition:
            self.cancelled = True
            self.condition.notify_all()

    def snapshot(self) -> list[dict[str, Any]]:
        with self.condition:
            return list(self.records)

    def join_threads(self, timeout: float) -> None:
        deadline = time.monotonic() + timeout
        for thread in self.threads:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            thread.join(remaining)


def _base_record(
    *,
    user_index: int,
    turn_index: int,
    task: dict[str, Any],
    seed: int,
    scheduled_perf: float,
    submitted_perf: float,
    phase: str,
) -> dict[str, Any]:
    return {
        "user_id": f"aime_user_{user_index:02d}",
        "user_index": user_index,
        "turn_index": turn_index,
        "problem_index": task["problem_index"],
        "task_id": task["task_id"],
        "source": task["source"],
        "reference_answer": task["reference_answer"],
        "request_seed": seed,
        "boxed_answer": None,
        "correct": False,
        "finish_reason": None,
        "terminal_reason": None,
        "output_text": "",
        "error": None,
        "client_disconnect_requested": False,
        "boxed_character_start": None,
        "boxed_character_end": None,
        "_phase_at_submit": phase,
        "_scheduled_perf": scheduled_perf,
        "_submitted_perf": submitted_perf,
        "_first_text_perf": None,
        "_first_box_perf": None,
        "_terminal_perf": None,
        "_disconnect_perf": None,
        "_server_usage": None,
    }


def run_task(
    *,
    host: str,
    port: int,
    model_id: str,
    user_index: int,
    turn_index: int,
    task: dict[str, Any],
    seed: int,
    scheduled_perf: float,
    submitted_perf: float,
    phase: str,
    max_tokens: int,
    request_timeout: float,
    sampling: dict[str, Any],
    request_extra: dict[str, Any],
) -> dict[str, Any]:
    record = _base_record(
        user_index=user_index,
        turn_index=turn_index,
        task=task,
        seed=seed,
        scheduled_perf=scheduled_perf,
        submitted_perf=submitted_perf,
        phase=phase,
    )
    body = {
        "model": model_id,
        "messages": task["messages"],
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": True,
        "stream_options": {"include_usage": True},
        **sampling,
        **request_extra,
    }
    payload = json.dumps(body, ensure_ascii=False).encode("utf-8")
    connection: http.client.HTTPConnection | None = None
    response: http.client.HTTPResponse | None = None
    deadline = submitted_perf + request_timeout
    try:
        connection = http.client.HTTPConnection(host, port, timeout=request_timeout)
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
        task_bench.remaining_timeout(connection, deadline)
        response = connection.getresponse()
        if response.status != 200:
            task_bench.remaining_timeout(connection, deadline)
            detail = response.read(4096).decode("utf-8", errors="replace")
            record["terminal_reason"] = "http_error"
            record["error"] = f"HTTP {response.status}: {detail}"
            record["_terminal_perf"] = time.perf_counter()
        else:
            while record["_terminal_perf"] is None:
                event = task_bench.read_sse_payload(response, connection, deadline)
                if event is None:
                    record["terminal_reason"] = (
                        f"server_{record['finish_reason']}"
                        if record["finish_reason"]
                        else "stream_eof"
                    )
                    record["_terminal_perf"] = time.perf_counter()
                    break
                if event is task_bench._SSE_DONE:
                    record["terminal_reason"] = (
                        f"server_{record['finish_reason']}"
                        if record["finish_reason"]
                        else "server_done"
                    )
                    record["_terminal_perf"] = time.perf_counter()
                    break
                assert isinstance(event, dict)
                if event.get("error") is not None:
                    record["terminal_reason"] = "stream_error"
                    record["error"] = json.dumps(event["error"], ensure_ascii=False)
                    record["_terminal_perf"] = time.perf_counter()
                    break
                if isinstance(event.get("usage"), dict):
                    record["_server_usage"] = event["usage"]
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason"):
                        record["finish_reason"] = str(choice["finish_reason"])
                for piece in task_bench.event_text(event):
                    now = time.perf_counter()
                    if record["_first_text_perf"] is None:
                        record["_first_text_perf"] = now
                    record["output_text"] += piece
                    match = task_bench.BOXED_INTEGER_RE.search(record["output_text"])
                    if match is None:
                        continue
                    record["boxed_answer"] = int(match.group(1))
                    record["correct"] = record["boxed_answer"] == record["reference_answer"]
                    record["boxed_character_start"] = match.start()
                    record["boxed_character_end"] = match.end()
                    record["_first_box_perf"] = now
                    record["_terminal_perf"] = now
                    record["terminal_reason"] = "boxed"
                    record["client_disconnect_requested"] = True
                    break
    except (socket.timeout, TimeoutError) as exc:
        record["terminal_reason"] = "request_timeout"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["_terminal_perf"] = time.perf_counter()
        record["client_disconnect_requested"] = True
    except Exception as exc:
        record["terminal_reason"] = "stream_error"
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["_terminal_perf"] = time.perf_counter()
    finally:
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
        if record["client_disconnect_requested"]:
            record["_disconnect_perf"] = time.perf_counter()
    if record["_terminal_perf"] is None:
        record["terminal_reason"] = "client_error"
        record["error"] = record["error"] or "request returned without a terminal state"
        record["_terminal_perf"] = time.perf_counter()
    return record


def run_user(
    *,
    user_index: int,
    tasks: list[dict[str, Any]],
    traffic: dict[str, Any],
    schedule_origin: float,
    state: SteadyState,
    host: str,
    port: int,
    model_id: str,
    max_tokens: int,
    request_timeout: float,
    sampling: dict[str, Any],
    request_extra: dict[str, Any],
    seed_base: int,
) -> None:
    try:
        user_count = int(traffic["user_count"])
        scheduled = schedule_origin + user_index * float(traffic["start_cadence_seconds"])
        turn_index = 0
        while state.wait_for_submission(scheduled):
            task_index = turn_index * user_count + user_index
            claim = state.claim_submission(
                user_index=user_index,
                turn_index=turn_index,
                task_index=task_index,
                task_count=len(tasks),
            )
            if claim is None:
                return
            phase, submitted_perf = claim
            task = tasks[task_index]
            record = run_task(
                host=host,
                port=port,
                model_id=model_id,
                user_index=user_index,
                turn_index=turn_index,
                task=task,
                seed=seed_base + task_index,
                scheduled_perf=scheduled,
                submitted_perf=submitted_perf,
                phase=phase,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                sampling=sampling,
                request_extra=request_extra,
            )
            state.record_terminal(record)
            turn_index += 1
            scheduled = record["_terminal_perf"] + float(traffic["think_time_seconds"])
    except BaseException as exc:
        state.record_thread_error(user_index, exc)


def run_workload(
    *,
    tasks: list[dict[str, Any]],
    traffic: dict[str, Any],
    schedule_origin: float,
    state: SteadyState,
    host: str,
    port: int,
    model_id: str,
    max_tokens: int,
    request_timeout: float,
    sampling: dict[str, Any],
    request_extra: dict[str, Any],
    seed_base: int,
) -> list[dict[str, Any]]:
    for user_index in range(int(traffic["user_count"])):
        thread = threading.Thread(
            target=run_user,
            kwargs={
                "user_index": user_index,
                "tasks": tasks,
                "traffic": traffic,
                "schedule_origin": schedule_origin,
                "state": state,
                "host": host,
                "port": port,
                "model_id": model_id,
                "max_tokens": max_tokens,
                "request_timeout": request_timeout,
                "sampling": sampling,
                "request_extra": request_extra,
                "seed_base": seed_base,
            },
            name=f"aime-steady-user-{user_index}",
            daemon=True,
        )
        state.threads.append(thread)
        thread.start()
    try:
        while any(thread.is_alive() for thread in state.threads):
            for thread in state.threads:
                thread.join(0.1)
    except BaseException:
        state.cancel()
        raise
    if state.thread_errors:
        first = state.thread_errors[0]
        raise RuntimeError(
            f"user {first['user_index']} failed with {first['type']}: {first['message']}"
        )
    return state.snapshot()


def attach_client_usage(
    records: Iterable[dict[str, Any]],
    tasks: list[dict[str, Any]],
    token_counter: task_bench.ClientTokenCounter,
) -> None:
    task_bench.attach_client_usage(
        records,
        {task["problem_index"]: task for task in tasks},
        token_counter,
    )


def _record_phase(
    record: dict[str, Any], measurement_start: float | None, measurement_end: float | None
) -> str:
    submitted = record["_submitted_perf"]
    terminal = record["_terminal_perf"]
    if measurement_start is None or submitted < measurement_start:
        if measurement_start is not None and terminal >= measurement_start:
            return "warmup_crossing"
        return "warmup"
    if measurement_end is not None and submitted < measurement_end:
        return "measured"
    return "outside_window"


def normalize_record(
    record: dict[str, Any],
    *,
    schedule_origin: float,
    wall_origin: float,
    measurement_start: float | None,
    measurement_end: float | None,
) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if not key.startswith("_")}
    result["phase"] = _record_phase(record, measurement_start, measurement_end)
    mapping = {
        "scheduled": "_scheduled_perf",
        "submitted": "_submitted_perf",
        "first_text": "_first_text_perf",
        "first_box": "_first_box_perf",
        "terminal": "_terminal_perf",
        "client_disconnect": "_disconnect_perf",
    }
    for public, private in mapping.items():
        value = record.get(private)
        result[public + "_from_schedule_origin_seconds"] = (
            value - schedule_origin if value is not None else None
        )
        result[public + "_from_measurement_start_seconds"] = (
            value - measurement_start
            if value is not None and measurement_start is not None
            else None
        )
        result[public + "_at_unix_seconds"] = (
            wall_origin + value - schedule_origin if value is not None else None
        )
    result["latency_seconds"] = record["_terminal_perf"] - record["_submitted_perf"]
    result["submit_lag_seconds"] = record["_submitted_perf"] - record["_scheduled_perf"]
    return result


def concurrency_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    if not records:
        return {"peak_inflight_tasks": 0, "average_inflight_tasks": 0.0}
    return task_bench.concurrency_summary(records)


def count_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    reasons: dict[str, int] = {}
    for record in records:
        reason = str(record["terminal_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    return {
        "submitted_task_count": len(records),
        "terminal_task_count": sum(record["_terminal_perf"] is not None for record in records),
        "correct_task_count": sum(bool(record["correct"]) for record in records),
        "boxed_task_count": sum(record["boxed_answer"] is not None for record in records),
        "incorrect_boxed_task_count": sum(
            record["boxed_answer"] is not None and not record["correct"] for record in records
        ),
        "tasks_without_box": sum(record["boxed_answer"] is None for record in records),
        "terminal_reason_counts": reasons,
        **concurrency_summary(records),
    }


def summarize(
    records: list[dict[str, Any]],
    *,
    measurement_start: float | None,
    measurement_end: float | None,
) -> dict[str, Any]:
    warmup = [
        record
        for record in records
        if measurement_start is None or record["_submitted_perf"] < measurement_start
    ]
    measured = [
        record
        for record in records
        if measurement_start is not None
        and measurement_end is not None
        and measurement_start <= record["_submitted_perf"] < measurement_end
    ]
    warmup_summary = count_summary(warmup)
    warmup_summary["crossing_task_count"] = sum(
        measurement_start is not None and record["_terminal_perf"] >= measurement_start
        for record in warmup
    )
    measured_summary = count_summary(measured)
    if measurement_start is None or measurement_end is None:
        denominator = None
        last_terminal = None
        goodput = None
    else:
        last_terminal = max(
            (record["_terminal_perf"] for record in measured),
            default=measurement_start,
        )
        denominator = max(measurement_end, last_terminal) - measurement_start
        correct = int(measured_summary["correct_task_count"])
        goodput = correct / denominator * 3600.0 if denominator > 0 else 0.0
    measured_summary.update(
        {
            "submission_window_seconds": (
                measurement_end - measurement_start
                if measurement_start is not None and measurement_end is not None
                else None
            ),
            "last_measured_terminal_from_start_seconds": (
                last_terminal - measurement_start
                if last_terminal is not None and measurement_start is not None
                else None
            ),
            "denominator_seconds": denominator,
            "denominator_formula": (
                "max(measurement_window_end, last_measured_terminal) - measurement_start"
            ),
            "task_goodput_per_hour": goodput,
        }
    )
    return {
        "warmup": warmup_summary,
        "measured": measured_summary,
        "all": count_summary(records),
    }


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_arm = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.arm_name).strip("_") or "arm"
    output = args.output or HERE / "results" / f"aime_steady_goodput_{safe_arm}_{stamp}.json"
    output = output.expanduser().resolve()
    server_log = args.server_log or output.with_suffix(".server.log")
    return output, server_log.expanduser().resolve()


def invocation_command(argv: list[str] | None) -> list[str]:
    arguments = list(sys.argv[1:] if argv is None else argv)
    return [str(Path(sys.executable).resolve()), str(Path(__file__).resolve()), *arguments]


def dry_run_payload(
    *,
    args: argparse.Namespace,
    workload: dict[str, Any],
    traffic: dict[str, Any],
    tasks: list[dict[str, Any]],
    task_path: Path,
    root: Path,
    python: str,
    model: str,
    command: list[str],
    environment_overrides: dict[str, str],
    sampling: dict[str, Any],
    sampling_source: str,
    prompt_policy: dict[str, Any],
    output: Path,
    server_log: Path,
    argv: list[str] | None,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "schema": RESULT_SCHEMA,
        "arm_name": args.arm_name,
        "harness_command": invocation_command(argv),
        "model": model,
        "freetoken_root": str(root),
        "python": python,
        "gpu": args.gpu,
        "server_command": command,
        "server_environment_overrides": environment_overrides,
        "freetoken_git": task_bench.git_info(root),
        "benchmark_git": task_bench.git_info(REPO),
        "workload_path": str(args.workload.expanduser().resolve()),
        "workload": workload,
        "traffic": traffic,
        "task_stream": {
            "path": str(task_path),
            "task_count": len(tasks),
            "assignment": "row_index = turn_index * user_count + user_index",
            "repeats_allowed": False,
        },
        "frozen_request_policy": {
            "max_tokens": args.max_tokens,
            "request_timeout_seconds": args.request_timeout,
            "sampling": sampling,
            "sampling_source": sampling_source,
            "seed_base": args.seed_base,
            "extra": args.request_extra,
            "prompt_policy": prompt_policy,
            "first_box_regex": task_bench.BOXED_INTEGER_RE.pattern,
        },
        "output": str(output),
        "server_log": str(server_log),
    }


def failure_payload(
    *,
    args: argparse.Namespace,
    argv: list[str] | None,
    output: Path,
    server_log: Path,
    failure: BaseException,
    shutdown_error: str | None,
    state: SteadyState | None,
    tasks: list[dict[str, Any]],
    token_counter: task_bench.ClientTokenCounter | None,
    schedule_origin: float | None,
    wall_origin: float | None,
    context: dict[str, Any],
) -> dict[str, Any]:
    records = state.snapshot() if state is not None else []
    if token_counter is not None and tasks:
        attach_client_usage(records, tasks, token_counter)
    measurement_start = state.measurement_start if state is not None else None
    measurement_end = state.measurement_end if state is not None else None
    normalized = []
    if schedule_origin is not None and wall_origin is not None:
        normalized = [
            normalize_record(
                record,
                schedule_origin=schedule_origin,
                wall_origin=wall_origin,
                measurement_start=measurement_start,
                measurement_end=measurement_end,
            )
            for record in sorted(records, key=lambda item: item["_submitted_perf"])
        ]
    summary = summarize(
        records,
        measurement_start=measurement_start,
        measurement_end=measurement_end,
    )
    return {
        "schema": RESULT_SCHEMA,
        "created_at_unix_seconds": time.time(),
        "arm_name": args.arm_name,
        "run_status": "failed",
        "valid_steady_goodput": False,
        "invalid_reason": "server or workload raised before a valid steady-state result",
        "failure": {
            "type": type(failure).__name__,
            "message": str(failure),
            "terminal_rows_preserved": len(normalized),
        },
        "harness": {
            "script": str(Path(__file__).resolve()),
            "command": invocation_command(argv),
            "git": task_bench.git_info(REPO),
        },
        "server": {
            "log": str(server_log),
            "shutdown_error": shutdown_error,
        },
        "partial_context": context,
        "measurement": {
            "start_from_schedule_origin_seconds": (
                measurement_start - schedule_origin
                if measurement_start is not None and schedule_origin is not None
                else None
            ),
            "end_from_schedule_origin_seconds": (
                measurement_end - schedule_origin
                if measurement_end is not None and schedule_origin is not None
                else None
            ),
        },
        "summary": summary,
        "tasks": normalized,
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output, server_log = output_paths(args)
    root = args.freetoken_root.expanduser().resolve()
    python = ""
    model = ""
    workload: dict[str, Any] = {}
    traffic: dict[str, Any] = {}
    tasks: list[dict[str, Any]] = []
    task_path: Path | None = None
    token_counter: task_bench.ClientTokenCounter | None = None
    server: task_bench.ManagedServer | None = None
    state: SteadyState | None = None
    schedule_origin: float | None = None
    wall_origin: float | None = None
    shutdown_error: str | None = None
    context: dict[str, Any] = {}
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    try:
        if not (root / "python" / "freetoken" / "cli.py").is_file():
            raise FileNotFoundError(f"not a FreeToken source checkout: {root}")
        python = task_bench.resolve_python(args.python_executable)
        model = task_bench.model_argument(args.model)
        workload = load_workload(args.workload.expanduser().resolve())
        traffic = resolve_traffic(args, workload)
        if args.task_jsonl is None:
            raise ValueError(
                "--task-jsonl is required (or set FREETOKEN_AIME_STEADY_JSONL)"
            )
        task_path = args.task_jsonl.expanduser().resolve()
        tasks = load_task_stream(task_path)
        prompt_policy = resolve_prompt_policy(args, workload)
        sampling, sampling_source = task_bench.resolve_sampling(args, model)
        command = task_bench.server_command(args, python, model)
        environment, environment_overrides = task_bench.server_environment(args, root)
        context = {
            "model": model,
            "freetoken_root": str(root),
            "python": python,
            "gpu": args.gpu,
            "server_command": command,
            "server_environment_overrides": environment_overrides,
            "workload_path": str(args.workload.expanduser().resolve()),
            "traffic": traffic,
            "task_stream_path": str(task_path),
            "task_stream_count": len(tasks),
        }
        if args.dry_run:
            print(
                json.dumps(
                    dry_run_payload(
                        args=args,
                        workload=workload,
                        traffic=traffic,
                        tasks=tasks,
                        task_path=task_path,
                        root=root,
                        python=python,
                        model=model,
                        command=command,
                        environment_overrides=environment_overrides,
                        sampling=sampling,
                        sampling_source=sampling_source,
                        prompt_policy=prompt_policy,
                        output=output,
                        server_log=server_log,
                        argv=argv,
                    ),
                    indent=2,
                    ensure_ascii=False,
                )
            )
            return 0

        token_counter = task_bench.ClientTokenCounter(model)
        prepare_task_prompts(tasks, prompt_policy, args.request_extra, token_counter)
        base_url = f"http://{args.host}:{args.port}"
        server = task_bench.ManagedServer(
            command=command,
            environment=environment,
            cwd=root,
            base_url=base_url,
            log_path=server_log,
            startup_timeout=args.server_timeout,
            shutdown_timeout=args.shutdown_timeout,
        )
        server.start()
        assert server.model_id is not None
        abort_probe = task_bench.confirm_disconnect_aborts(
            server,
            host=args.host,
            port=args.port,
            model_id=server.model_id,
            max_tokens=args.max_tokens,
            timeout=args.request_timeout,
            abort_timeout=args.abort_timeout,
        )
        stats_before = server.wait_idle(args.abort_timeout)
        schedule_origin = time.perf_counter()
        wall_origin = time.time()
        state = SteadyState(
            user_count=int(traffic["user_count"]),
            measurement_seconds=float(traffic["measurement_seconds"]),
        )
        records = run_workload(
            tasks=tasks,
            traffic=traffic,
            schedule_origin=schedule_origin,
            state=state,
            host=args.host,
            port=args.port,
            model_id=server.model_id,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            sampling=sampling,
            request_extra=args.request_extra,
            seed_base=args.seed_base,
        )
        attach_client_usage(records, tasks, token_counter)
        post_idle_error: str | None = None
        try:
            stats_after = server.wait_idle(args.abort_timeout)
        except Exception as exc:
            post_idle_error = f"{type(exc).__name__}: {exc}"
            try:
                stats_after = server.stats()
            except Exception:
                stats_after = {}
        measurement_start = state.measurement_start
        measurement_end = state.measurement_end
        normalized = [
            normalize_record(
                record,
                schedule_origin=schedule_origin,
                wall_origin=wall_origin,
                measurement_start=measurement_start,
                measurement_end=measurement_end,
            )
            for record in sorted(records, key=lambda item: item["_submitted_perf"])
        ]
        post_active = (
            stats_after.get("requests", {}).get("active")
            if isinstance(stats_after.get("requests"), dict)
            else None
        )
        invalid_reasons: list[str] = []
        if measurement_start is None:
            invalid_reasons.append("measurement never started")
        if state.exhaustion is not None:
            invalid_reasons.append(
                f"task stream exhausted during {state.exhaustion['phase']}"
            )
        if post_idle_error is not None or post_active != 0:
            invalid_reasons.append("server did not report active=0 after measured drain")
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "valid_steady_goodput": not invalid_reasons,
            "invalid_reason": "; ".join(invalid_reasons) if invalid_reasons else None,
            "harness": {
                "script": str(Path(__file__).resolve()),
                "command": invocation_command(argv),
                "git": task_bench.git_info(REPO),
            },
            "server": {
                "freetoken_root": str(root),
                "git": task_bench.git_info(root),
                "python": python,
                "model_argument": model,
                "served_model_id": server.model_id,
                "gpu": args.gpu,
                "base_url": base_url,
                "command": command,
                "environment_overrides": environment_overrides,
                "log": str(server_log),
                "shutdown_error": None,
            },
            "client_tokenizer": token_counter.info(),
            "workload_path": str(args.workload.expanduser().resolve()),
            "workload": workload,
            "task_stream": {
                "path": str(task_path),
                "task_count": len(tasks),
                "assignment": "row_index = turn_index * user_count + user_index",
                "repeats_allowed": False,
                "exhaustion": state.exhaustion,
            },
            "frozen_policy": {
                "traffic": traffic,
                "max_tokens": args.max_tokens,
                "request_timeout_seconds": args.request_timeout,
                "server_timeout_seconds": args.server_timeout,
                "abort_timeout_seconds": args.abort_timeout,
                "sampling": sampling,
                "sampling_source": sampling_source,
                "seed_base": args.seed_base,
                "request_extra": args.request_extra,
                "prompt_policy": prompt_policy,
                "server_args": args.server_args,
                "first_box_regex": task_bench.BOXED_INTEGER_RE.pattern,
            },
            "measurement": {
                "start_from_schedule_origin_seconds": (
                    measurement_start - schedule_origin
                    if measurement_start is not None
                    else None
                ),
                "end_from_schedule_origin_seconds": (
                    measurement_end - schedule_origin
                    if measurement_end is not None
                    else None
                ),
                "start_at_unix_seconds": (
                    wall_origin + measurement_start - schedule_origin
                    if measurement_start is not None
                    else None
                ),
                "end_at_unix_seconds": (
                    wall_origin + measurement_end - schedule_origin
                    if measurement_end is not None
                    else None
                ),
                "submitted_interval": "[start, end)",
            },
            "abort_confirmation": {
                "preflight": abort_probe,
                "post_workload_confirmed": post_idle_error is None and post_active == 0,
                "post_workload_error": post_idle_error,
                "stats_before_workload": stats_before,
                "stats_after_workload": stats_after,
            },
            "server_measurement_delta": task_bench.stats_token_delta(
                stats_before, stats_after
            ),
            "summary": summarize(
                records,
                measurement_start=measurement_start,
                measurement_end=measurement_end,
            ),
            "tasks": normalized,
        }
    except BaseException as exc:
        failure = exc
        if state is not None:
            state.cancel()
    finally:
        if server is not None:
            try:
                server.stop()
            except Exception as exc:
                shutdown_error = f"{type(exc).__name__}: {exc}"
            server.close()
        if state is not None:
            state.join_threads(5.0)

    if failure is not None:
        diagnostic = failure_payload(
            args=args,
            argv=argv,
            output=output,
            server_log=server_log,
            failure=failure,
            shutdown_error=shutdown_error,
            state=state,
            tasks=tasks,
            token_counter=token_counter,
            schedule_origin=schedule_origin,
            wall_origin=wall_origin,
            context=context,
        )
        task_bench.write_json(output, diagnostic)
        print(
            f"[{args.arm_name}] failed after {len(diagnostic['tasks'])} terminal tasks; "
            f"diagnostic result: {output}\n{type(failure).__name__}: {failure}",
            file=sys.stderr,
            flush=True,
        )
        return 130 if isinstance(failure, KeyboardInterrupt) else 1

    assert result is not None
    result["server"]["shutdown_error"] = shutdown_error
    result["run_status"] = "completed"
    if shutdown_error is not None:
        result["valid_steady_goodput"] = False
        result["invalid_reason"] = "server process group did not shut down cleanly"
    task_bench.write_json(output, result)
    measured = result["summary"]["measured"]
    goodput = measured["task_goodput_per_hour"]
    goodput_text = "n/a" if goodput is None else f"{goodput:.3f}/h"
    print(
        f"[{args.arm_name}] correct={measured['correct_task_count']}/"
        f"{measured['submitted_task_count']} denominator="
        f"{measured['denominator_seconds']}s task_goodput={goodput_text} "
        f"valid={result['valid_steady_goodput']}\n{output}",
        flush=True,
    )
    return 0 if result["valid_steady_goodput"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
