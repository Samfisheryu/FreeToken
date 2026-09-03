#!/usr/bin/env python3
"""Run one frozen FreeToken arm and measure AIME task goodput.

The benchmark owns one server process, replays the fixed twenty-user closed-loop
manifest, and stops each task at the first streamed ``\\boxed{integer}``.  Closing
that stream is part of the measured client behavior.  A public-API probe verifies
before measurement that disconnecting a stream removes the live request from
``/v1/stats`` without incrementing the server's completed-request count.

``--server-args`` consumes the remainder of the command line and must therefore be
last.  It lets each arm freeze any FreeToken serving policy independently.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from typing import Any, Iterable
import urllib.error
import urllib.request

if __package__:
    from . import bench_decode_moe as decode_bench
else:
    import bench_decode_moe as decode_bench


HERE = Path(__file__).resolve().parent
REPO = HERE.parent
DEFAULT_MANIFEST = HERE / "workloads" / "aime25_task_goodput_20user_v1.json"
RESULT_SCHEMA = "freetoken.aime_task_goodput_result.v1"
BOXED_INTEGER_RE = re.compile(r"\\boxed\s*\{\s*([+-]?\d+)\s*\}")
ENV_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
ABORT_PROBE_TOKEN_CAP = 2048
RESERVED_SERVER_OPTIONS = ("--model", "--model-path", "--host", "--port", "--gpu")
RESERVED_REQUEST_FIELDS = {
    "model",
    "messages",
    "max_tokens",
    "seed",
    "stream",
    "stream_options",
    "temperature",
    "top_p",
    "top_k",
}
_SSE_DONE = object()


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
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--aime-jsonl",
        type=Path,
        default=(Path(os.environ["FREETOKEN_AIME25_JSONL"]) if os.environ.get("FREETOKEN_AIME25_JSONL") else None),
        help="local AIME25 JSONL; otherwise download the manifest's Hub dataset",
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
            "one global user-message suffix; default is the manifest instruction, "
            "and an empty string disables it"
        ),
    )
    parser.add_argument(
        "--request-extra-json",
        default='{"chat_template_kwargs":{"enable_thinking":true}}',
        help="extra chat request fields as a JSON object (core measurement fields are reserved)",
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
    overlap = sorted(RESERVED_REQUEST_FIELDS.intersection(request_extra))
    if overlap:
        parser.error(
            "--request-extra-json cannot override frozen fields: " + ", ".join(overlap)
        )
    args.request_extra = request_extra
    args.server_env = parse_environment(args.env, parser)
    validate_server_args(args.server_args, parser)
    return args


def parse_environment(
    assignments: Iterable[str], parser: argparse.ArgumentParser
) -> dict[str, str]:
    result: dict[str, str] = {}
    for assignment in assignments:
        name, separator, value = assignment.partition("=")
        if not separator or not ENV_NAME_RE.fullmatch(name):
            parser.error(f"--env expects KEY=VALUE; got {assignment!r}")
        if name in {"CUDA_VISIBLE_DEVICES", "PYTHONPATH"}:
            parser.error(f"use --gpu/--pythonpath instead of overriding {name} with --env")
        result[name] = value
    return result


def validate_server_args(args: Iterable[str], parser: argparse.ArgumentParser) -> None:
    for token in args:
        for option in RESERVED_SERVER_OPTIONS:
            if token == option or token.startswith(option + "="):
                parser.error(
                    f"{option} is controlled by a top-level benchmark option and cannot "
                    "appear after --server-args"
                )


def load_manifest(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != "freetoken.aime_task_goodput_workload.v1":
        raise ValueError(f"unsupported workload schema in {path}")
    users = manifest.get("users")
    closed_loop = manifest.get("closed_loop")
    if not isinstance(users, list) or not isinstance(closed_loop, dict):
        raise ValueError("manifest must contain users and closed_loop")
    if len(users) != 20 or closed_loop.get("user_count") != 20:
        raise ValueError("the AIME goodput contract requires exactly 20 users")
    if [user.get("user_index") for user in users] != list(range(20)):
        raise ValueError("manifest user_index values must be ordered 0 through 19")
    if len({user.get("user_id") for user in users}) != 20:
        raise ValueError("manifest user_id values must be unique")
    expected_counts = [2] * 10 + [1] * 10
    observed_counts = [len(user.get("problem_indices") or []) for user in users]
    if observed_counts != expected_counts:
        raise ValueError("users 0-9 need two tasks and users 10-19 need one task")
    indices = [index for user in users for index in user["problem_indices"]]
    if sorted(indices) != list(range(30)) or closed_loop.get("task_count") != 30:
        raise ValueError("manifest must assign each AIME25 problem index exactly once")
    for user in users:
        if float(user.get("start_offset_seconds", -1)) < 0:
            raise ValueError("start offsets must be non-negative")
        if float(user.get("think_time_seconds", -1)) < 0:
            raise ValueError("think times must be non-negative")
    return manifest


def resolve_prompt_policy(
    args: argparse.Namespace, manifest: dict[str, Any]
) -> dict[str, Any]:
    prompt_contract = manifest["prompt_contract"]
    answer_instruction = (
        prompt_contract["answer_instruction"]
        if args.answer_instruction is None
        else args.answer_instruction
    )
    return {
        "system_prompt": args.system_prompt,
        "answer_instruction": answer_instruction,
        "answer_instruction_source": (
            "manifest default" if args.answer_instruction is None else "--answer-instruction"
        ),
        "existing_instruction_marker": prompt_contract["append_instruction_when_missing"],
        "user_suffix_separator": "\n",
        "message_construction": [
            "Start with an empty messages list.",
            "If system_prompt is non-empty, append {'role': 'system', 'content': system_prompt}.",
            "Strip trailing whitespace from the dataset problem to form user_content.",
            "If answer_instruction is non-empty and existing_instruction_marker is absent from user_content, append user_suffix_separator followed by answer_instruction.",
            "Append {'role': 'user', 'content': user_content}. No task-specific prompt text or policy is allowed.",
        ],
        "scope": "Identical global policy for all 30 tasks; only dataset problem text varies.",
    }


def build_messages(problem: str, prompt_policy: dict[str, Any]) -> tuple[list[dict[str, str]], bool]:
    messages: list[dict[str, str]] = []
    system_prompt = prompt_policy["system_prompt"]
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    user_content = problem.rstrip()
    instruction = prompt_policy["answer_instruction"]
    marker = prompt_policy["existing_instruction_marker"]
    appended = bool(instruction) and marker not in user_content
    if appended:
        user_content += prompt_policy["user_suffix_separator"] + instruction
    messages.append({"role": "user", "content": user_content})
    return messages, appended


def _thinking_mode(template_kwargs: dict[str, Any]) -> str:
    mode = str(template_kwargs.get("thinking_mode") or "chat")
    if template_kwargs.get("enable_thinking") or template_kwargs.get("thinking"):
        mode = "thinking"
    return mode if mode in {"chat", "thinking"} else "chat"


class ClientTokenCounter:
    """Count text with the served model's tokenizer and label what is estimated."""

    def __init__(self, model: str) -> None:
        load_source = "freetoken.utils.load_tokenizer"
        try:
            from freetoken.utils import load_tokenizer

            self.tokenizer = load_tokenizer(model)
        except Exception:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(model)
            load_source = "transformers.AutoTokenizer.from_pretrained"
        self.model = model
        self.load_source = load_source
        self.dsv4_encoder: Any = None
        model_path = Path(model)
        encoder_path = model_path / "encoding" / "encoding_dsv4.py"
        if not getattr(self.tokenizer, "chat_template", None) and encoder_path.is_file():
            spec = importlib.util.spec_from_file_location(
                "aime_goodput_encoding_dsv4", encoder_path
            )
            if spec is not None and spec.loader is not None:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                if hasattr(module, "encode_messages"):
                    self.dsv4_encoder = module

    def info(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "loader": self.load_source,
            "tokenizer_class": type(self.tokenizer).__name__,
            "tokenizer_name_or_path": str(
                getattr(self.tokenizer, "name_or_path", self.model)
            ),
            "has_chat_template": bool(getattr(self.tokenizer, "chat_template", None)),
            "has_dsv4_encoder": self.dsv4_encoder is not None,
        }

    def prompt_observation(
        self,
        messages: list[dict[str, str]],
        request_extra: dict[str, Any],
    ) -> dict[str, Any]:
        template_kwargs = dict(request_extra.get("chat_template_kwargs") or {})
        if getattr(self.tokenizer, "chat_template", None):
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **template_kwargs,
            )
            ids = self.tokenizer.encode(rendered, add_special_tokens=False)
            return {
                "tokens": len(ids),
                "source": (
                    "model tokenizer: apply_chat_template(tokenize=False, "
                    "add_generation_prompt=True), then encode(add_special_tokens=False)"
                ),
                "estimated": False,
                "chat_template_kwargs": template_kwargs,
            }
        if self.dsv4_encoder is not None:
            rendered = self.dsv4_encoder.encode_messages(
                [dict(message) for message in messages],
                thinking_mode=_thinking_mode(template_kwargs),
                reasoning_effort=template_kwargs.get("reasoning_effort"),
            )
            ids = self.tokenizer.encode(rendered, add_special_tokens=True)
            return {
                "tokens": len(ids),
                "source": (
                    "model tokenizer: encoding/encoding_dsv4.py encode_messages, "
                    "then encode(add_special_tokens=True)"
                ),
                "estimated": False,
                "chat_template_kwargs": template_kwargs,
            }
        fallback = "\n".join(
            f"{message['role']}: {message['content']}" for message in messages
        )
        ids = self.tokenizer.encode(fallback, add_special_tokens=True)
        return {
            "tokens": len(ids),
            "source": (
                "model tokenizer fallback: role/content text joined by newline because "
                "the client could not load the server's chat renderer"
            ),
            "estimated": True,
            "chat_template_kwargs": template_kwargs,
        }

    def output_observation(self, text: str) -> dict[str, Any]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return {
            "tokens": len(ids),
            "source": (
                "model tokenizer re-encoding of client-observed streamed reasoning/content "
                "text through the task terminal"
            ),
            "estimated": True,
            "estimate_reason": (
                "Retokenized text is exact for the observed text but may differ from the "
                "server's sampled-token count because decode/token boundaries are not exposed."
            ),
        }


def resolve_python(value: str) -> str:
    if os.sep in value:
        path = Path(os.path.abspath(str(Path(value).expanduser())))
        if not path.is_file():
            raise FileNotFoundError(f"Python executable not found: {path}")
        if not os.access(path, os.X_OK):
            raise PermissionError(f"Python executable is not executable: {path}")
        return str(path)
    found = shutil.which(value)
    if found is None:
        raise FileNotFoundError(f"Python executable not found on PATH: {value}")
    path = Path(os.path.abspath(found))
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Python executable is not executable: {path}")
    return str(path)


def model_argument(value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve()) if path.exists() else value


def server_command(args: argparse.Namespace, python: str, model: str) -> list[str]:
    return [
        python,
        "-m",
        "freetoken.cli",
        "serve",
        "--model-path",
        model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        *args.server_args,
    ]


def server_environment(
    args: argparse.Namespace, freetoken_root: Path
) -> tuple[dict[str, str], dict[str, str]]:
    import_paths = [
        *(str(Path(item).expanduser().resolve()) for item in args.pythonpath),
        str(freetoken_root / "python"),
        str(freetoken_root),
    ]
    existing = os.environ.get("PYTHONPATH")
    if existing:
        import_paths.append(existing)
    overrides = {
        "CUDA_VISIBLE_DEVICES": args.gpu,
        "PYTHONPATH": os.pathsep.join(import_paths),
        **args.server_env,
    }
    environment = os.environ.copy()
    environment.update(overrides)
    return environment, overrides


def git_info(root: Path) -> dict[str, Any]:
    def command(*values: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(root), *values],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        return completed.stdout.strip()

    commit = command("rev-parse", "HEAD")
    branch = command("rev-parse", "--abbrev-ref", "HEAD")
    dirty = bool(command("status", "--porcelain"))
    return {"commit": commit, "branch": branch, "dirty": dirty}


def read_json_url(url: str, timeout: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        payload = json.load(response)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object from {url}")
    return payload


class ManagedServer:
    def __init__(
        self,
        *,
        command: list[str],
        environment: dict[str, str],
        cwd: Path,
        base_url: str,
        log_path: Path,
        startup_timeout: float,
        shutdown_timeout: float,
    ) -> None:
        self.command = command
        self.environment = environment
        self.cwd = cwd
        self.base_url = base_url
        self.log_path = log_path
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.process: subprocess.Popen[bytes] | None = None
        self.log_handle: Any = None
        self.model_id: str | None = None

    def start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log_handle = self.log_path.open("wb")
        self.process = subprocess.Popen(
            self.command,
            stdout=self.log_handle,
            stderr=subprocess.STDOUT,
            cwd=self.cwd,
            env=self.environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited with {self.process.returncode} during startup: {self.log_tail()}"
                )
            try:
                health = read_json_url(self.base_url + "/health", timeout=3.0)
            except (OSError, ValueError, urllib.error.URLError):
                time.sleep(0.5)
                continue
            if health.get("status") == "error":
                raise RuntimeError(f"server reported startup error: {health}")
            if health.get("status") == "ok" and health.get("maintenance", "serving") == "serving":
                models = read_json_url(self.base_url + "/v1/models", timeout=10.0)
                entries = models.get("data")
                if not isinstance(entries, list) or not entries or not entries[0].get("id"):
                    raise RuntimeError(f"server returned no model id: {models}")
                self.model_id = str(entries[0]["id"])
                return
            time.sleep(0.5)
        raise TimeoutError(
            f"server was not ready after {self.startup_timeout:.1f}s: {self.log_tail()}"
        )

    def stats(self) -> dict[str, Any]:
        return read_json_url(self.base_url + "/v1/stats", timeout=10.0)

    def wait_idle(self, timeout: float) -> dict[str, Any]:
        deadline = time.monotonic() + timeout
        last: dict[str, Any] | None = None
        while time.monotonic() < deadline:
            if self.process is not None and self.process.poll() is not None:
                raise RuntimeError(
                    f"server exited before requests drained: {self.log_tail()}"
                )
            last = self.stats()
            requests = last.get("requests")
            if isinstance(requests, dict) and requests.get("active") == 0:
                return last
            time.sleep(0.05)
        active = None
        if isinstance(last, dict) and isinstance(last.get("requests"), dict):
            active = last["requests"].get("active")
        raise TimeoutError(f"server still reported {active!r} active requests after {timeout:.1f}s")

    def log_tail(self, bytes_to_read: int = 16384) -> str:
        if self.log_handle is None:
            return ""
        self.log_handle.flush()
        with self.log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - bytes_to_read))
            return handle.read().decode("utf-8", errors="replace")

    def stop(self) -> None:
        if self.process is None:
            return
        process = self.process
        group = process.pid
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=self.shutdown_timeout)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(group, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=min(15.0, self.shutdown_timeout))
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(group, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=15.0)
        live = self._wait_for_group_exit(group, self.shutdown_timeout)
        if live:
            try:
                os.killpg(group, signal.SIGKILL)
            except ProcessLookupError:
                pass
            live = self._wait_for_group_exit(group, 15.0)
        if live:
            raise RuntimeError(f"server process group {group} still has live members: {live}")

    @staticmethod
    def _live_group_members(group: int) -> list[int]:
        members: list[int] = []
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                state = fields[0]
                process_group = int(fields[2])
            except (FileNotFoundError, ProcessLookupError, PermissionError, IndexError):
                continue
            if process_group == group and state != "Z":
                members.append(int(entry.name))
        return members

    @classmethod
    def _wait_for_group_exit(cls, group: int, timeout: float) -> list[int]:
        deadline = time.monotonic() + timeout
        while True:
            members = cls._live_group_members(group)
            if not members:
                return []
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return members
            time.sleep(min(0.05, remaining))

    def close(self) -> None:
        if self.log_handle is not None:
            self.log_handle.close()
            self.log_handle = None


def remaining_timeout(connection: http.client.HTTPConnection, deadline: float) -> float:
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise TimeoutError("request wall-clock deadline expired")
    if connection.sock is not None:
        connection.sock.settimeout(remaining)
    return remaining


def open_chat_stream(
    *,
    host: str,
    port: int,
    body: dict[str, Any],
    deadline: float,
) -> tuple[http.client.HTTPConnection, http.client.HTTPResponse]:
    connection = http.client.HTTPConnection(host, port, timeout=max(0.001, deadline - time.perf_counter()))
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
    remaining_timeout(connection, deadline)
    response = connection.getresponse()
    if response.status != 200:
        remaining_timeout(connection, deadline)
        detail = response.read(4096).decode("utf-8", errors="replace")
        response.close()
        connection.close()
        raise RuntimeError(f"HTTP {response.status}: {detail}")
    return connection, response


def read_sse_payload(
    response: http.client.HTTPResponse,
    connection: http.client.HTTPConnection,
    deadline: float,
) -> dict[str, Any] | object | None:
    while True:
        remaining_timeout(connection, deadline)
        raw = response.readline()
        if not raw:
            return None
        line = raw.strip()
        if not line or not line.startswith(b"data:"):
            continue
        data = line[len(b"data:") :].strip()
        if data == b"[DONE]":
            return _SSE_DONE
        if not data:
            continue
        payload = json.loads(data)
        if not isinstance(payload, dict):
            raise ValueError("SSE data was not a JSON object")
        return payload


def event_text(event: dict[str, Any]) -> list[str]:
    pieces: list[str] = []
    for choice in event.get("choices") or []:
        delta = choice.get("delta") or {}
        for field in ("reasoning_content", "content"):
            value = delta.get(field)
            if isinstance(value, str) and value:
                pieces.append(value)
    return pieces


def request_counts(stats: dict[str, Any]) -> dict[str, int]:
    requests = stats.get("requests")
    if not isinstance(requests, dict):
        raise ValueError("/v1/stats has no requests object")
    required = ("active", "completed", "completion_tokens_total")
    if any(not isinstance(requests.get(name), int) for name in required):
        raise ValueError(f"/v1/stats requests object lacks integer fields {required}")
    return {name: int(requests[name]) for name in required}


def confirm_disconnect_aborts(
    server: ManagedServer,
    *,
    host: str,
    port: int,
    model_id: str,
    max_tokens: int,
    timeout: float,
    abort_timeout: float,
) -> dict[str, Any]:
    before_stats = server.wait_idle(abort_timeout)
    before = request_counts(before_stats)
    probe_tokens = min(max_tokens, ABORT_PROBE_TOKEN_CAP)
    body = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": "Emit a long sequence of decimal integers separated by spaces.",
            }
        ],
        "max_tokens": probe_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    started = time.perf_counter()
    deadline = started + timeout
    connection: http.client.HTTPConnection | None = None
    response: http.client.HTTPResponse | None = None
    first_text_at: float | None = None
    open_counts: dict[str, int] | None = None
    open_stats: dict[str, Any] | None = None
    try:
        connection, response = open_chat_stream(
            host=host,
            port=port,
            body=body,
            deadline=deadline,
        )
        while first_text_at is None:
            payload = read_sse_payload(response, connection, deadline)
            if payload is None or payload is _SSE_DONE:
                raise RuntimeError("abort probe finished before a streamed text delta arrived")
            assert isinstance(payload, dict)
            if event_text(payload):
                first_text_at = time.perf_counter()
        open_stats = server.stats()
        open_counts = request_counts(open_stats)
        if open_counts["active"] < 1:
            raise RuntimeError("abort probe was not observable as an active public request")
    finally:
        if response is not None:
            response.close()
        if connection is not None:
            connection.close()
    disconnected_at = time.perf_counter()
    after_stats = server.wait_idle(abort_timeout)
    after = request_counts(after_stats)
    if after["completed"] != before["completed"]:
        raise RuntimeError(
            "disconnect probe reached normal completion; client-close abort was not confirmed"
        )
    if after["completion_tokens_total"] - before["completion_tokens_total"] >= probe_tokens:
        raise RuntimeError("disconnect probe generated its full token budget before becoming idle")
    assert first_text_at is not None and open_counts is not None and open_stats is not None
    return {
        "confirmed": True,
        "method": (
            "Read one non-empty /v1/chat/completions SSE delta, observe requests.active >= 1, "
            "close the HTTP response/socket, then require requests.active == 0 without an "
            "increase in requests.completed."
        ),
        "probe_max_tokens": probe_tokens,
        "first_text_seconds": first_text_at - started,
        "disconnect_seconds": disconnected_at - started,
        "idle_ack_seconds_after_disconnect": time.perf_counter() - disconnected_at,
        "stats_before": before_stats,
        "stats_while_open": open_stats,
        "stats_after": after_stats,
    }


def resolve_dataset_path(
    manifest: dict[str, Any], explicit: Path | None
) -> tuple[Path, str]:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path, "--aime-jsonl"
    dataset = manifest["dataset"]
    from huggingface_hub import hf_hub_download

    downloaded = hf_hub_download(
        dataset["repository"],
        dataset["file"],
        repo_type=dataset.get("repository_type", "dataset"),
    )
    return Path(downloaded).resolve(), "huggingface_hub"


def parse_reference_answer(value: Any) -> int:
    text = str(value).strip()
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    match = BOXED_INTEGER_RE.search(text)
    if match is not None:
        return int(match.group(1))
    raise ValueError(f"AIME reference answer is not an integer: {value!r}")


def load_tasks(
    manifest: dict[str, Any], dataset_path: Path
) -> dict[int, dict[str, Any]]:
    with dataset_path.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    dataset = manifest["dataset"]
    count = int(dataset["problem_count"])
    if len(rows) < count:
        raise ValueError(f"AIME file has {len(rows)} rows; manifest needs {count}")
    tasks: dict[int, dict[str, Any]] = {}
    for index in range(count):
        row = rows[index]
        problem = row.get(dataset["problem_field"]) or row.get(
            dataset["fallback_problem_field"]
        )
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"AIME row {index} has no problem text")
        tasks[index] = {
            "problem_index": index,
            "problem": problem,
            "reference_answer": parse_reference_answer(row[dataset["answer_field"]]),
        }
    return tasks


def prepare_task_prompts(
    tasks: dict[int, dict[str, Any]],
    prompt_policy: dict[str, Any],
    request_extra: dict[str, Any],
    token_counter: ClientTokenCounter,
) -> None:
    for task in tasks.values():
        messages, appended = build_messages(task["problem"], prompt_policy)
        task["messages"] = messages
        task["answer_instruction_appended"] = appended
        task["prompt_token_observation"] = token_counter.prompt_observation(
            messages, request_extra
        )


def attach_client_usage(
    records: Iterable[dict[str, Any]],
    tasks: dict[int, dict[str, Any]],
    token_counter: ClientTokenCounter,
) -> None:
    for record in records:
        if "usage" in record:
            continue
        task = tasks[record["problem_index"]]
        server_usage = record.get("_server_usage")
        try:
            output_observation = token_counter.output_observation(record["output_text"])
        except Exception as exc:
            output_observation = {
                "tokens": None,
                "source": "model tokenizer re-encoding of client-observed output failed",
                "estimated": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        record["messages"] = task["messages"]
        record["answer_instruction_appended"] = task["answer_instruction_appended"]
        if isinstance(server_usage, dict):
            unavailable_reason = None
        elif record["terminal_reason"] in {"boxed", "request_timeout"}:
            unavailable_reason = (
                "The client closed the stream at its task terminal before FreeToken's final "
                "usage SSE chunk; waiting for that chunk would violate early termination."
            )
        else:
            unavailable_reason = (
                "The HTTP/SSE stream ended without a server usage object; the observable "
                "terminal reason is recorded separately."
            )
        record["usage"] = {
            "server_reported": server_usage if isinstance(server_usage, dict) else None,
            "server_reported_available": isinstance(server_usage, dict),
            "server_reported_unavailable_reason": unavailable_reason,
            "client_observed": {
                "prompt": task["prompt_token_observation"],
                "output": output_observation,
            },
        }


def resolve_sampling(args: argparse.Namespace, model: str) -> tuple[dict[str, Any], str]:
    sampling, source = decode_bench.resolve_sampling(model, args.greedy)
    overrides: list[str] = []
    for argument, field in (
        (args.temperature, "temperature"),
        (args.top_p, "top_p"),
        (args.top_k, "top_k"),
    ):
        if argument is not None:
            sampling[field] = argument
            overrides.append(field)
    if overrides:
        source += " + CLI overrides " + ",".join(overrides)
    return sampling, source


def wait_until(target: float) -> None:
    remaining = target - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def base_task_record(
    *,
    user: dict[str, Any],
    turn_index: int,
    task: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    return {
        "user_id": user["user_id"],
        "user_index": user["user_index"],
        "turn_index": turn_index,
        "problem_index": task["problem_index"],
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
        "_scheduled_perf": None,
        "_submitted_perf": None,
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
    user: dict[str, Any],
    turn_index: int,
    task: dict[str, Any],
    seed: int,
    scheduled_perf: float,
    max_tokens: int,
    request_timeout: float,
    sampling: dict[str, Any],
    request_extra: dict[str, Any],
) -> dict[str, Any]:
    record = base_task_record(user=user, turn_index=turn_index, task=task, seed=seed)
    record["_scheduled_perf"] = scheduled_perf
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
    connection: http.client.HTTPConnection | None = None
    response: http.client.HTTPResponse | None = None
    try:
        connection = http.client.HTTPConnection(host, port, timeout=request_timeout)
        record["_submitted_perf"] = time.perf_counter()
        deadline = record["_submitted_perf"] + request_timeout
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
        remaining_timeout(connection, deadline)
        response = connection.getresponse()
        if response.status != 200:
            remaining_timeout(connection, deadline)
            detail = response.read(4096).decode("utf-8", errors="replace")
            record["terminal_reason"] = "http_error"
            record["error"] = f"HTTP {response.status}: {detail}"
            record["_terminal_perf"] = time.perf_counter()
        else:
            while record["_terminal_perf"] is None:
                event = read_sse_payload(response, connection, deadline)
                if event is None:
                    record["terminal_reason"] = (
                        f"server_{record['finish_reason']}"
                        if record["finish_reason"]
                        else "stream_eof"
                    )
                    record["_terminal_perf"] = time.perf_counter()
                    break
                if event is _SSE_DONE:
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
                for piece in event_text(event):
                    now = time.perf_counter()
                    if record["_first_text_perf"] is None:
                        record["_first_text_perf"] = now
                    record["output_text"] += piece
                    match = BOXED_INTEGER_RE.search(record["output_text"])
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
    if record["_submitted_perf"] is None:
        record["_submitted_perf"] = time.perf_counter()
    if record["_terminal_perf"] is None:
        record["terminal_reason"] = "client_error"
        record["error"] = record["error"] or "request returned without a terminal state"
        record["_terminal_perf"] = time.perf_counter()
    return record


def run_user(
    *,
    user: dict[str, Any],
    tasks: dict[int, dict[str, Any]],
    schedule_origin: float,
    host: str,
    port: int,
    model_id: str,
    max_tokens: int,
    request_timeout: float,
    sampling: dict[str, Any],
    request_extra: dict[str, Any],
    seed_base: int,
    terminal_sink: Any,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    scheduled = schedule_origin + float(user["start_offset_seconds"])
    for turn_index, problem_index in enumerate(user["problem_indices"]):
        wait_until(scheduled)
        task = tasks[problem_index]
        record = run_task(
            host=host,
            port=port,
            model_id=model_id,
            user=user,
            turn_index=turn_index,
            task=task,
            seed=seed_base + problem_index,
            scheduled_perf=scheduled,
            max_tokens=max_tokens,
            request_timeout=request_timeout,
            sampling=sampling,
            request_extra=request_extra,
        )
        records.append(record)
        terminal_sink(record)
        scheduled = record["_terminal_perf"] + float(user["think_time_seconds"])
    return records


def run_workload(
    *,
    manifest: dict[str, Any],
    tasks: dict[int, dict[str, Any]],
    host: str,
    port: int,
    model_id: str,
    max_tokens: int,
    request_timeout: float,
    sampling: dict[str, Any],
    request_extra: dict[str, Any],
    seed_base: int,
    schedule_origin: float,
    terminal_records: list[dict[str, Any]],
    terminal_lock: threading.Lock,
) -> list[dict[str, Any]]:
    def terminal_sink(record: dict[str, Any]) -> None:
        with terminal_lock:
            terminal_records.append(record)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=20)
    try:
        futures = [
            executor.submit(
                run_user,
                user=user,
                tasks=tasks,
                schedule_origin=schedule_origin,
                host=host,
                port=port,
                model_id=model_id,
                max_tokens=max_tokens,
                request_timeout=request_timeout,
                sampling=sampling,
                request_extra=request_extra,
                seed_base=seed_base,
                terminal_sink=terminal_sink,
            )
            for user in manifest["users"]
        ]
        for future in futures:
            future.result()
    except BaseException:
        executor.shutdown(wait=False, cancel_futures=True)
        raise
    else:
        executor.shutdown(wait=True)
    with terminal_lock:
        records = list(terminal_records)
    if len(records) != 30:
        raise RuntimeError(f"workload returned {len(records)} terminal records instead of 30")
    return records


def concurrency_summary(records: list[dict[str, Any]]) -> dict[str, float | int]:
    events: list[tuple[float, int]] = []
    for record in records:
        events.append((record["_submitted_perf"], 1))
        events.append((record["_terminal_perf"], -1))
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
        "peak_inflight_tasks": peak,
        "average_inflight_tasks": area / span if span > 0 else 0.0,
    }


def normalize_record(
    record: dict[str, Any], *, first_submit: float, schedule_origin: float, wall_origin: float
) -> dict[str, Any]:
    result = {key: value for key, value in record.items() if not key.startswith("_")}
    mapping = {
        "scheduled_at": "_scheduled_perf",
        "submitted_at": "_submitted_perf",
        "first_text_at": "_first_text_perf",
        "first_box_at": "_first_box_perf",
        "terminal_at": "_terminal_perf",
        "client_disconnect_at": "_disconnect_perf",
    }
    for public, private in mapping.items():
        value = record.get(private)
        result[public + "_seconds"] = value - first_submit if value is not None else None
        result[public + "_unix_seconds"] = (
            wall_origin + value - schedule_origin if value is not None else None
        )
    result["latency_seconds"] = record["_terminal_perf"] - record["_submitted_perf"]
    result["submit_lag_seconds"] = record["_submitted_perf"] - record["_scheduled_perf"]
    return result


def summarize(
    records: list[dict[str, Any]], schedule_origin: float, wall_origin: float
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    first_submit = min(record["_submitted_perf"] for record in records)
    last_terminal = max(record["_terminal_perf"] for record in records)
    makespan = last_terminal - first_submit
    correct = sum(bool(record["correct"]) for record in records)
    reasons: dict[str, int] = {}
    for record in records:
        reason = str(record["terminal_reason"])
        reasons[reason] = reasons.get(reason, 0) + 1
    normalized = [
        normalize_record(
            record,
            first_submit=first_submit,
            schedule_origin=schedule_origin,
            wall_origin=wall_origin,
        )
        for record in sorted(records, key=lambda item: item["problem_index"])
    ]
    summary = {
        "task_count": len(records),
        "terminal_task_count": sum(record["_terminal_perf"] is not None for record in records),
        "correct_task_count": correct,
        "boxed_task_count": sum(record["boxed_answer"] is not None for record in records),
        "incorrect_boxed_task_count": sum(
            record["boxed_answer"] is not None and not record["correct"] for record in records
        ),
        "tasks_without_box": sum(record["boxed_answer"] is None for record in records),
        "makespan_seconds": makespan,
        "task_goodput_per_hour": correct / makespan * 3600.0 if makespan > 0 else 0.0,
        "terminal_reason_counts": reasons,
        **concurrency_summary(records),
    }
    return summary, normalized


def stats_token_delta(before: dict[str, Any], after: dict[str, Any]) -> dict[str, int | None]:
    left = before.get("requests") if isinstance(before.get("requests"), dict) else {}
    right = after.get("requests") if isinstance(after.get("requests"), dict) else {}
    fields = ("prompt_tokens_total", "completion_tokens_total", "completed")
    return {
        field: (
            int(right[field]) - int(left[field])
            if isinstance(left.get(field), int) and isinstance(right.get(field), int)
            else None
        )
        for field in fields
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_arm = re.sub(r"[^A-Za-z0-9_.-]+", "_", args.arm_name).strip("_") or "arm"
    output = args.output or HERE / "results" / f"aime_task_goodput_{safe_arm}_{stamp}.json"
    output = output.expanduser().resolve()
    log = args.server_log or output.with_suffix(".server.log")
    return output, log.expanduser().resolve()


def dry_run_payload(
    *,
    args: argparse.Namespace,
    manifest: dict[str, Any],
    model: str,
    python: str,
    root: Path,
    command: list[str],
    environment_overrides: dict[str, str],
    sampling: dict[str, Any],
    sampling_source: str,
    prompt_policy: dict[str, Any],
    output: Path,
    log: Path,
) -> dict[str, Any]:
    return {
        "dry_run": True,
        "schema": RESULT_SCHEMA,
        "arm_name": args.arm_name,
        "model": model,
        "freetoken_root": str(root),
        "python": python,
        "gpu": args.gpu,
        "server_command": command,
        "server_environment_overrides": environment_overrides,
        "freetoken_git": git_info(root),
        "benchmark_git": git_info(REPO),
        "manifest_path": str(args.manifest.expanduser().resolve()),
        "workload": manifest,
        "frozen_request_policy": {
            "max_tokens": args.max_tokens,
            "request_timeout_seconds": args.request_timeout,
            "sampling": sampling,
            "sampling_source": sampling_source,
            "seed_base": args.seed_base,
            "extra": args.request_extra,
            "prompt_policy": prompt_policy,
            "first_box_regex": BOXED_INTEGER_RE.pattern,
        },
        "output": str(output),
        "server_log": str(log),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.freetoken_root.expanduser().resolve()
    if not (root / "python" / "freetoken" / "cli.py").is_file():
        raise FileNotFoundError(f"not a FreeToken source checkout: {root}")
    python = resolve_python(args.python_executable)
    model = model_argument(args.model)
    manifest = load_manifest(args.manifest.expanduser().resolve())
    prompt_policy = resolve_prompt_policy(args, manifest)
    sampling, sampling_source = resolve_sampling(args, model)
    command = server_command(args, python, model)
    environment, environment_overrides = server_environment(args, root)
    output, server_log = output_paths(args)

    if args.dry_run:
        print(
            json.dumps(
                dry_run_payload(
                    args=args,
                    manifest=manifest,
                    model=model,
                    python=python,
                    root=root,
                    command=command,
                    environment_overrides=environment_overrides,
                    sampling=sampling,
                    sampling_source=sampling_source,
                    prompt_policy=prompt_policy,
                    output=output,
                    log=server_log,
                ),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0

    dataset_path, dataset_source = resolve_dataset_path(manifest, args.aime_jsonl)
    tasks = load_tasks(manifest, dataset_path)
    token_counter = ClientTokenCounter(model)
    prepare_task_prompts(
        tasks,
        prompt_policy,
        args.request_extra,
        token_counter,
    )
    base_url = f"http://{args.host}:{args.port}"
    server = ManagedServer(
        command=command,
        environment=environment,
        cwd=root,
        base_url=base_url,
        log_path=server_log,
        startup_timeout=args.server_timeout,
        shutdown_timeout=args.shutdown_timeout,
    )
    shutdown_error: str | None = None
    result: dict[str, Any] | None = None
    failure: BaseException | None = None
    terminal_records: list[dict[str, Any]] = []
    terminal_lock = threading.Lock()
    schedule_origin: float | None = None
    wall_origin: float | None = None
    abort_probe: dict[str, Any] | None = None
    stats_before: dict[str, Any] = {}
    stats_after: dict[str, Any] = {}
    try:
        server.start()
        assert server.model_id is not None
        abort_probe = confirm_disconnect_aborts(
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
        records = run_workload(
            manifest=manifest,
            tasks=tasks,
            host=args.host,
            port=args.port,
            model_id=server.model_id,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            sampling=sampling,
            request_extra=args.request_extra,
            seed_base=args.seed_base,
            schedule_origin=schedule_origin,
            terminal_records=terminal_records,
            terminal_lock=terminal_lock,
        )
        attach_client_usage(records, tasks, token_counter)
        last_terminal = max(record["_terminal_perf"] for record in records)
        post_abort_error: str | None = None
        try:
            stats_after = server.wait_idle(args.abort_timeout)
        except Exception as exc:
            post_abort_error = f"{type(exc).__name__}: {exc}"
            try:
                stats_after = (
                    server.stats()
                    if server.process is not None and server.process.poll() is None
                    else {}
                )
            except Exception:
                stats_after = {}
        summary, normalized = summarize(records, schedule_origin, wall_origin)
        post_active = (
            stats_after.get("requests", {}).get("active")
            if isinstance(stats_after.get("requests"), dict)
            else None
        )
        post_abort_confirmed = post_abort_error is None and post_active == 0
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "valid_task_goodput": post_abort_confirmed,
            "invalid_reason": None if post_abort_confirmed else "server work remained after client task terminals",
            "harness": {
                "script": str(Path(__file__).resolve()),
                "git": git_info(REPO),
            },
            "server": {
                "freetoken_root": str(root),
                "git": git_info(root),
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
            "dataset": {
                **manifest["dataset"],
                "resolved_path": str(dataset_path),
                "source": dataset_source,
            },
            "manifest_path": str(args.manifest.expanduser().resolve()),
            "workload": manifest,
            "frozen_policy": {
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
                "first_box_regex": BOXED_INTEGER_RE.pattern,
            },
            "abort_confirmation": {
                "preflight": abort_probe,
                "post_workload_confirmed": post_abort_confirmed,
                "post_workload_error": post_abort_error,
                "seconds_after_last_task_terminal": time.perf_counter() - last_terminal,
                "stats_before_workload": stats_before,
                "stats_after_workload": stats_after,
            },
            "server_measurement_delta": stats_token_delta(stats_before, stats_after),
            "summary": summary,
            "tasks": normalized,
        }
    except BaseException as exc:
        failure = exc
    finally:
        try:
            server.stop()
        except Exception as exc:
            shutdown_error = f"{type(exc).__name__}: {exc}"
        server.close()

    if result is None:
        with terminal_lock:
            partial_records = list(terminal_records)
        attach_client_usage(partial_records, tasks, token_counter)
        if partial_records and schedule_origin is not None and wall_origin is not None:
            partial_summary, partial_tasks = summarize(
                partial_records, schedule_origin, wall_origin
            )
        else:
            partial_summary = {
                "task_count": 0,
                "terminal_task_count": 0,
                "correct_task_count": 0,
                "boxed_task_count": 0,
                "incorrect_boxed_task_count": 0,
                "tasks_without_box": 0,
                "makespan_seconds": None,
                "task_goodput_per_hour": None,
                "terminal_reason_counts": {},
                "peak_inflight_tasks": 0,
                "average_inflight_tasks": 0.0,
            }
            partial_tasks = []
        assert failure is not None
        result = {
            "schema": RESULT_SCHEMA,
            "created_at_unix_seconds": time.time(),
            "arm_name": args.arm_name,
            "run_status": "failed",
            "valid_task_goodput": False,
            "invalid_reason": "server or workload raised before a valid 30-task result",
            "failure": {
                "type": type(failure).__name__,
                "message": str(failure),
                "terminal_rows_preserved": len(partial_tasks),
            },
            "harness": {
                "script": str(Path(__file__).resolve()),
                "git": git_info(REPO),
            },
            "server": {
                "freetoken_root": str(root),
                "git": git_info(root),
                "python": python,
                "model_argument": model,
                "served_model_id": server.model_id,
                "gpu": args.gpu,
                "base_url": base_url,
                "command": command,
                "environment_overrides": environment_overrides,
                "log": str(server_log),
                "shutdown_error": shutdown_error,
            },
            "client_tokenizer": token_counter.info(),
            "dataset": {
                **manifest["dataset"],
                "resolved_path": str(dataset_path),
                "source": dataset_source,
            },
            "manifest_path": str(args.manifest.expanduser().resolve()),
            "workload": manifest,
            "frozen_policy": {
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
                "first_box_regex": BOXED_INTEGER_RE.pattern,
            },
            "abort_confirmation": {
                "preflight": abort_probe,
                "stats_before_workload": stats_before,
                "stats_after_workload": stats_after,
            },
            "summary": partial_summary,
            "tasks": partial_tasks,
        }
        write_json(output, result)
        print(
            f"[{args.arm_name}] failed after {len(partial_tasks)}/30 terminal tasks; "
            f"diagnostic result: {output}\n{type(failure).__name__}: {failure}",
            file=sys.stderr,
            flush=True,
        )
        return 130 if isinstance(failure, KeyboardInterrupt) else 1

    result["server"]["shutdown_error"] = shutdown_error
    result["run_status"] = "completed"
    if shutdown_error is not None:
        result["valid_task_goodput"] = False
        result["invalid_reason"] = "server process group did not shut down cleanly"
    write_json(output, result)
    summary = result["summary"]
    print(
        f"[{args.arm_name}] correct={summary['correct_task_count']}/30 "
        f"makespan={summary['makespan_seconds']:.3f}s "
        f"task_goodput={summary['task_goodput_per_hour']:.3f}/h "
        f"valid={result['valid_task_goodput']}\n{output}",
        flush=True,
    )
    return 0 if result["valid_task_goodput"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
