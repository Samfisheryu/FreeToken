"""Independent black-box acceptance for layered-pipeline resident-group reuse.

This module deliberately imports no FreeToken or benchmark implementation code.
It talks only to the public ``serve`` CLI, ``/v1/completions`` SSE endpoint, and
documented server log lines.

GPU execution is opt-in.  Set ``FREETOKEN_RESIDENT_BLACKBOX_RUN=1`` and point
``FREETOKEN_RESIDENT_BLACKBOX_CONFIG`` at a JSON file with this public shape::

    {
      "launch_prefix": ["/opt/venv/bin/python", "{tree}/path/to/ft"],
      "gpu": "0",
      "candidate_tree": "/tmp/freetoken-layered-pipeline-modular",
      "baseline_tree": "/tmp/freetoken-layered-pipeline-lru-baseline",
      "synthetic": {
        "path": "/models/scaled-qwen3-moe",
        "served_name": "lab-agent-qwen3-moe",
        "layers": 8,
        "experts": 8,
        "dtype": "float16",
        "attention_backend": "triton",
        "max_seq_len": 4096,
        "max_running_requests": 12,
        "extra_args": []
      },
      "real_models": {
        "qwen36": {
          "path": "/models/Qwen3.6-MoE",
          "served_name": "lab-agent-qwen3-moe",
          "layers": 64,
          "experts": 128,
          "dtype": "bfloat16",
          "attention_backend": "flashinfer",
          "max_seq_len": 4096,
          "max_running_requests": 12,
          "extra_args": ["--expert-path", "/models/qwen36-experts"]
        },
        "dsv4": {
          "path": "/models/DeepSeek-V4",
          "served_name": "lab-agent-qwen3-moe",
          "layers": 61,
          "experts": 256,
          "dtype": "bfloat16",
          "attention_backend": "flashinfer",
          "max_seq_len": 4096,
          "max_running_requests": 16,
          "extra_args": ["--expert-path", "/models/dsv4-experts"]
        }
      }
    }

``launch_prefix`` is tokenized JSON, never a shell string.  ``{tree}`` is
expanded separately for candidate and baseline.  Model-specific public CLI
arguments, including a NoWAG expert/plugin path when required, belong in
``extra_args``.  Missing real-model paths are reported as pytest skips.

The suite starts at most eight successful GPU service lifecycles: two synthetic
configurations in candidate and baseline (four), plus Graph 0/8 for each of the
two optional real models (four).
"""

from __future__ import annotations

import concurrent.futures
import dataclasses
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import threading
import time
from typing import Any, Callable, Iterable, Mapping, Sequence
import urllib.error
import urllib.request

import pytest


RUN_ENV = "FREETOKEN_RESIDENT_BLACKBOX_RUN"
CONFIG_ENV = "FREETOKEN_RESIDENT_BLACKBOX_CONFIG"
PIPELINE_FILTER_ENV = "FREETOKEN_RESIDENT_BLACKBOX_PIPELINE"
FIXED_PORT_ENV = "FREETOKEN_RESIDENT_BLACKBOX_FIXED_PORT"

WAVE_RE = re.compile(
    r"Layered pipeline wave complete: "
    r"reqs=(?P<reqs>\d+), "
    r"groups=(?P<groups>\d+), "
    r"group_forwards=(?P<group_forwards>\d+), "
    r"iterations=(?P<iterations>\d+), "
    r"decode_iterations=(?P<decode_iterations>\d+), "
    r"prefill_layer_prepares=(?P<prefill_layer_prepares>\d+)"
)
ITERATION_RE = re.compile(
    r"Layered pipeline iteration limit: "
    r"requested_tokens=(?P<requested_tokens>\d+), "
    r"effective_tokens=(?P<effective_tokens>\d+), "
    r"event=(?P<event>[A-Za-z0-9_-]+)"
)
STATS_RE = re.compile(
    r"MoE cache stats snapshot: "
    r"decode_active_rows=(?P<decode_active_rows>\d+), "
    r"decode_missing_rows=(?P<decode_missing_rows>\d+), "
    r"decode_layer_calls=(?P<decode_layer_calls>\d+), "
    r"decode_fetched_rows=(?P<decode_fetched_rows>\d+), "
    r"prefill_hit_rows=(?P<prefill_hit_rows>\d+), "
    r"prefill_rows=(?P<prefill_rows>\d+), "
    r"prefill_layer_prepares=(?P<prefill_layer_prepares>\d+), "
    r"prefill_h2d_bytes_total=(?P<prefill_h2d_bytes_total>\d+), "
    r"expert_row_bytes=(?P<expert_row_bytes>\d+)"
)

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


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    path: Path
    served_name: str
    layers: int
    experts: int
    dtype: str
    attention_backend: str
    max_seq_len: int
    max_running_requests: int
    extra_args: tuple[str, ...]

    @classmethod
    def from_json(cls, key: str, raw: Mapping[str, Any]) -> "ModelSpec":
        required = (
            "path",
            "served_name",
            "layers",
            "experts",
            "dtype",
            "attention_backend",
        )
        missing = [name for name in required if name not in raw]
        if missing:
            raise ValueError(f"model {key!r} is missing fields: {missing}")
        extra_args = raw.get("extra_args", [])
        if not isinstance(extra_args, list) or not all(
            isinstance(item, str) for item in extra_args
        ):
            raise ValueError(f"model {key!r} extra_args must be a list of strings")
        model = cls(
            key=key,
            path=Path(str(raw["path"])),
            served_name=str(raw["served_name"]),
            layers=int(raw["layers"]),
            experts=int(raw["experts"]),
            dtype=str(raw["dtype"]),
            attention_backend=str(raw["attention_backend"]),
            max_seq_len=int(raw.get("max_seq_len", 4096)),
            max_running_requests=int(raw.get("max_running_requests", 12)),
            extra_args=tuple(extra_args),
        )
        if (
            model.layers <= 0
            or model.experts <= 0
            or model.max_seq_len <= 0
            or model.max_running_requests <= 0
        ):
            raise ValueError(f"model {key!r} has non-positive public dimensions")
        return model


@dataclasses.dataclass(frozen=True)
class PipelineSpec:
    name: str
    cuda_graph_max_bs: int
    group_size: int
    wave_max_chunks: int
    max_prefill_tokens: int
    cache_size: int

    def effective_group(self, model: ModelSpec) -> int:
        return min(
            self.group_size,
            model.layers,
            (self.cache_size // model.experts) - 1,
        )

    def groups(self, model: ModelSpec) -> int:
        effective = self.effective_group(model)
        if effective < 1:
            raise ValueError(
                f"{self.name}: cache C={self.cache_size} cannot hold one resident "
                f"group plus the decode reserve for E={model.experts}"
            )
        return math.ceil(model.layers / effective)

    def public_args(self) -> tuple[str, ...]:
        return (
            "--max-prefill-length",
            str(self.max_prefill_tokens),
            "--moe-cache-size",
            str(self.cache_size),
            "--cuda-graph-max-bs",
            str(self.cuda_graph_max_bs),
            "--batching-policy",
            "layered-pipeline",
            "--prefill-layer-group-size",
            str(self.group_size),
            "--prefill-wave-max-chunks",
            str(self.wave_max_chunks),
        )


@dataclasses.dataclass(frozen=True)
class BlackboxConfig:
    launch_prefix: tuple[str, ...]
    gpu: str
    candidate_tree: Path
    baseline_tree: Path
    synthetic: ModelSpec
    real_models: Mapping[str, ModelSpec]
    startup_timeout_s: float
    request_timeout_s: float
    idle_timeout_s: float
    shutdown_timeout_s: float
    submit_stagger_s: float
    driver_max_tokens: int

    @classmethod
    def load(cls) -> "BlackboxConfig":
        config_path = os.environ.get(CONFIG_ENV)
        if not config_path:
            pytest.skip(f"set {CONFIG_ENV} to the public black-box JSON config")
        raw = json.loads(Path(config_path).read_text(encoding="utf-8"))
        launch_prefix = raw.get("launch_prefix")
        if (
            not isinstance(launch_prefix, list)
            or not launch_prefix
            or not all(isinstance(item, str) for item in launch_prefix)
        ):
            raise ValueError("launch_prefix must be a non-empty JSON list of strings")
        real_raw = raw.get("real_models", {})
        if not isinstance(real_raw, dict):
            raise ValueError("real_models must be a JSON object")
        unexpected = set(real_raw) - {"qwen36", "dsv4"}
        if unexpected:
            raise ValueError(
                "only the contracted real-model keys qwen36 and dsv4 are accepted; "
                f"got {sorted(unexpected)}"
            )
        return cls(
            launch_prefix=tuple(launch_prefix),
            gpu=str(raw.get("gpu", "0")),
            candidate_tree=Path(
                raw.get(
                    "candidate_tree",
                    "/tmp/freetoken-layered-pipeline-modular",
                )
            ),
            baseline_tree=Path(
                raw.get(
                    "baseline_tree",
                    "/tmp/freetoken-layered-pipeline-lru-baseline",
                )
            ),
            synthetic=ModelSpec.from_json("synthetic", raw["synthetic"]),
            real_models={
                key: ModelSpec.from_json(key, value) for key, value in real_raw.items()
            },
            startup_timeout_s=float(raw.get("startup_timeout_s", 600.0)),
            request_timeout_s=float(raw.get("request_timeout_s", 600.0)),
            idle_timeout_s=float(raw.get("idle_timeout_s", 180.0)),
            shutdown_timeout_s=float(raw.get("shutdown_timeout_s", 30.0)),
            submit_stagger_s=float(raw.get("submit_stagger_s", 0.02)),
            driver_max_tokens=int(raw.get("driver_max_tokens", 512)),
        )

    def synthetic_pipelines(self) -> tuple[PipelineSpec, PipelineSpec]:
        experts = self.synthetic.experts
        return (
            PipelineSpec(
                name="graph0_wave1",
                cuda_graph_max_bs=0,
                group_size=2,
                wave_max_chunks=1,
                max_prefill_tokens=128,
                cache_size=3 * experts,
            ),
            PipelineSpec(
                name="graph8_partial_group",
                cuda_graph_max_bs=8,
                group_size=3,
                wave_max_chunks=4,
                max_prefill_tokens=96,
                cache_size=4 * experts,
            ),
        )


@dataclasses.dataclass(frozen=True)
class LogLine:
    index: int
    monotonic_s: float
    text: str


@dataclasses.dataclass(frozen=True)
class WaveRecord:
    line: LogLine
    reqs: int
    groups: int
    group_forwards: int
    iterations: int
    decode_iterations: int
    prefill_layer_prepares: int


@dataclasses.dataclass(frozen=True)
class IterationLimit:
    line: LogLine
    requested_tokens: int
    effective_tokens: int
    event: str


@dataclasses.dataclass(frozen=True)
class StatsSnapshot:
    line: LogLine
    decode_active_rows: int
    decode_missing_rows: int
    decode_layer_calls: int
    decode_fetched_rows: int
    prefill_hit_rows: int
    prefill_rows: int
    prefill_layer_prepares: int
    prefill_h2d_bytes_total: int
    expert_row_bytes: int


class LogCapture:
    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._lines: list[LogLine] = []

    def append(self, raw: str) -> None:
        with self._condition:
            line = LogLine(len(self._lines), time.monotonic(), raw.rstrip("\r\n"))
            self._lines.append(line)
            self._condition.notify_all()

    def cursor(self) -> int:
        with self._condition:
            return len(self._lines)

    def lines(self, start: int = 0, stop: int | None = None) -> list[LogLine]:
        with self._condition:
            return list(self._lines[start:stop])

    def wait_for(
        self,
        parser: Callable[[LogLine], Any | None],
        *,
        after_index: int,
        after_time_s: float = 0.0,
        timeout_s: float,
        process: subprocess.Popen[str],
    ) -> Any:
        deadline = time.monotonic() + timeout_s
        scan = after_index
        with self._condition:
            while True:
                for line in self._lines[scan:]:
                    scan = line.index + 1
                    if line.monotonic_s < after_time_s:
                        continue
                    parsed = parser(line)
                    if parsed is not None:
                        return parsed
                if process.poll() is not None:
                    tail = "\n".join(item.text for item in self._lines[-40:])
                    raise AssertionError(
                        f"server exited with {process.returncode} while waiting for a "
                        f"public log record; log tail:\n{tail}"
                    )
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    tail = "\n".join(item.text for item in self._lines[-40:])
                    raise AssertionError(
                        f"timed out waiting for a public log record; log tail:\n{tail}"
                    )
                self._condition.wait(timeout=min(remaining, 0.5))


def _parse_wave(line: LogLine) -> WaveRecord | None:
    match = WAVE_RE.search(line.text)
    if not match:
        return None
    values = {name: int(value) for name, value in match.groupdict().items()}
    return WaveRecord(line=line, **values)


def _parse_iteration(line: LogLine) -> IterationLimit | None:
    match = ITERATION_RE.search(line.text)
    if not match:
        return None
    return IterationLimit(
        line=line,
        requested_tokens=int(match.group("requested_tokens")),
        effective_tokens=int(match.group("effective_tokens")),
        event=match.group("event"),
    )


def _parse_stats(line: LogLine) -> StatsSnapshot | None:
    match = STATS_RE.search(line.text)
    if not match:
        return None
    values = {name: int(value) for name, value in match.groupdict().items()}
    return StatsSnapshot(line=line, **values)


@dataclasses.dataclass
class CompletionOutcome:
    label: str
    requested_max_tokens: int
    started_s: float
    ended_s: float
    first_text_s: float | None
    status: int | None
    content_type: str
    done: bool
    text: str
    text_event_count: int
    usage: Mapping[str, Any] | None
    malformed_lines: list[str]
    error: str | None

    @property
    def prompt_tokens(self) -> int:
        assert self.usage is not None
        return int(self.usage["prompt_tokens"])

    @property
    def completion_tokens(self) -> int:
        assert self.usage is not None
        return int(self.usage["completion_tokens"])

    @property
    def cached_tokens(self) -> int:
        assert self.usage is not None
        details = self.usage.get("prompt_tokens_details")
        if not isinstance(details, Mapping):
            return 0
        value = details.get("cached_tokens", 0)
        return int(value)

    @property
    def cache_details_present(self) -> bool:
        if self.usage is None:
            return False
        details = self.usage.get("prompt_tokens_details")
        return isinstance(details, Mapping) and "cached_tokens" in details

    @property
    def uncached_prompt_rows(self) -> int:
        return self.prompt_tokens - self.cached_tokens

    def contract_errors(self) -> list[str]:
        errors: list[str] = []
        if self.error:
            errors.append(self.error)
        if self.status != 200:
            errors.append(f"HTTP status is {self.status}, expected 200")
        if "text/event-stream" not in self.content_type.lower():
            errors.append(f"content-type is not SSE: {self.content_type!r}")
        if not self.done:
            errors.append("SSE stream did not end with data: [DONE]")
        if self.requested_max_tokens > 0 and self.text_event_count == 0:
            errors.append("SSE stream contained no choices[0].text event")
        errors.extend(f"malformed SSE line: {line!r}" for line in self.malformed_lines)
        if not isinstance(self.usage, Mapping):
            errors.append("final SSE usage object is missing")
            return errors
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = self.usage.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                errors.append(f"usage.{key} is not a non-negative integer: {value!r}")
        if errors:
            return errors
        prompt = int(self.usage["prompt_tokens"])
        completion = int(self.usage["completion_tokens"])
        total = int(self.usage["total_tokens"])
        if total != prompt + completion:
            errors.append(f"usage total mismatch: {total} != {prompt} + {completion}")
        if completion != self.requested_max_tokens:
            errors.append(
                "ignore_eos completion length mismatch: "
                f"{completion} != {self.requested_max_tokens}"
            )
        cached = self.cached_tokens
        if cached < 0 or cached > prompt:
            errors.append(f"cached_tokens={cached} is outside [0, {prompt}]")
        return errors


def _post_completion(
    *,
    base_url: str,
    served_name: str,
    prompt: str,
    max_tokens: int,
    timeout_s: float,
    label: str,
    first_text_callback: Callable[[float], None] | None = None,
) -> CompletionOutcome:
    payload = {
        "model": served_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "ignore_eos": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    started = time.monotonic()
    status: int | None = None
    content_type = ""
    done = False
    text_parts: list[str] = []
    text_event_count = 0
    usage: Mapping[str, Any] | None = None
    malformed: list[str] = []
    error: str | None = None
    first_text_s: float | None = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            status = int(response.status)
            content_type = response.headers.get("Content-Type", "")
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    malformed.append(line)
                    continue
                data = line[len("data:") :].strip()
                if data == "[DONE]":
                    done = True
                    continue
                if done:
                    malformed.append("data received after [DONE]")
                    continue
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    malformed.append(line)
                    continue
                if not isinstance(event, Mapping):
                    malformed.append(line)
                    continue
                event_usage = event.get("usage")
                if isinstance(event_usage, Mapping):
                    usage = event_usage
                choices = event.get("choices")
                if isinstance(choices, list) and choices:
                    choice = choices[0]
                    if isinstance(choice, Mapping):
                        piece = choice.get("text")
                        if isinstance(piece, str):
                            text_event_count += 1
                            text_parts.append(piece)
                            if piece and first_text_s is None:
                                first_text_s = time.monotonic()
                                if first_text_callback is not None:
                                    first_text_callback(first_text_s)
    except urllib.error.HTTPError as exc:
        status = int(exc.code)
        content_type = exc.headers.get("Content-Type", "") if exc.headers else ""
        body = exc.read().decode("utf-8", errors="replace")
        error = f"HTTP {exc.code}: {body[-1000:]}"
    except (OSError, TimeoutError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    return CompletionOutcome(
        label=label,
        requested_max_tokens=max_tokens,
        started_s=started,
        ended_s=time.monotonic(),
        first_text_s=first_text_s,
        status=status,
        content_type=content_type,
        done=done,
        text="".join(text_parts),
        text_event_count=text_event_count,
        usage=usage,
        malformed_lines=malformed,
        error=error,
    )


def _unused_local_port() -> int:
    fixed = os.environ.get(FIXED_PORT_ENV)
    if fixed:
        return int(fixed)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class PublicServer:
    def __init__(
        self,
        *,
        config: BlackboxConfig,
        tree: Path,
        model: ModelSpec,
        pipeline: PipelineSpec,
        port: int,
    ) -> None:
        self.config = config
        self.tree = tree
        self.model = model
        self.pipeline = pipeline
        self.port = port
        self.base_url = f"http://127.0.0.1:{port}"
        self.logs = LogCapture()
        self.process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self.measure_cursor = 0
        self.last_snapshot: StatsSnapshot | None = None

    @property
    def public_serve_args(self) -> tuple[str, ...]:
        return (
            "serve",
            "--model-path",
            str(self.model.path),
            "--served-model-name",
            self.model.served_name,
            "--host",
            "127.0.0.1",
            "--port",
            str(self.port),
            "--dtype",
            self.model.dtype,
            "--max-running-requests",
            str(self.model.max_running_requests),
            "--max-seq-len-override",
            str(self.model.max_seq_len),
            *self.pipeline.public_args(),
            "--attention-backend",
            self.model.attention_backend,
            "--moe-backend",
            "offload",
            "--cache-type",
            "radix",
            "--enable-cache-report",
            "--moe-collect-stats",
            *self.model.extra_args,
        )

    def _command(self) -> list[str]:
        prefix = [
            part.format(tree=str(self.tree)) for part in self.config.launch_prefix
        ]
        return [*prefix, *self.public_serve_args]

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["CUDA_VISIBLE_DEVICES"] = self.config.gpu
        env["PYTHONOPTIMIZE"] = "1"
        additions = [str(self.tree / "python"), str(self.tree / "benchmarks")]
        current = env.get("PYTHONPATH")
        if current:
            additions.append(current)
        env["PYTHONPATH"] = os.pathsep.join(additions)
        return env

    def __enter__(self) -> "PublicServer":
        self.pipeline.groups(self.model)
        command = self._command()
        print(
            "SERVICE START "
            f"tree={self.tree} model={self.model.key} "
            f"pipeline={self.pipeline.name} graph={self.pipeline.cuda_graph_max_bs} "
            f"port={self.port}",
            flush=True,
        )
        self.process = subprocess.Popen(
            command,
            cwd=self.tree,
            env=self._environment(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
        )
        assert self.process.stdout is not None

        def read_logs() -> None:
            assert self.process is not None
            assert self.process.stdout is not None
            for raw in self.process.stdout:
                self.logs.append(raw)

        self._reader = threading.Thread(target=read_logs, daemon=True)
        self._reader.start()
        try:
            self._await_ready()
        except BaseException as exc:
            if isinstance(exc, Exception):
                print(
                    "PUBLIC FAILURE "
                    f"tree={self.tree} model={self.model.key} "
                    f"pipeline={self.pipeline.name} startup={exc}",
                    flush=True,
                )
            self.__exit__(type(exc), exc, exc.__traceback__)
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if self.process is None:
            return
        shutdown_deadline = time.monotonic() + self.config.shutdown_timeout_s
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(
                    timeout=max(0.1, shutdown_deadline - time.monotonic())
                )
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait(timeout=10.0)
        while time.monotonic() < shutdown_deadline:
            try:
                os.killpg(self.process.pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.05)
        else:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if self._reader is not None:
            self._reader.join(timeout=2.0)
        print(
            "SERVICE END "
            f"tree={self.tree} model={self.model.key} "
            f"pipeline={self.pipeline.name} exit={self.process.returncode}",
            flush=True,
        )

    def _await_ready(self) -> None:
        assert self.process is not None
        deadline = time.monotonic() + self.config.startup_timeout_s
        last_error = "server did not accept a request"
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                complete_log = "\n".join(line.text for line in self.logs.lines())
                raise AssertionError(
                    f"server exited during startup with {self.process.returncode}; "
                    f"complete public server log:\n{complete_log}"
                )
            attempt_cursor = self.logs.cursor()
            outcome = _post_completion(
                base_url=self.base_url,
                served_name=self.model.served_name,
                prompt=f"resident-blackbox-ready-{self.port}",
                max_tokens=1,
                timeout_s=min(10.0, self.config.request_timeout_s),
                label="readiness",
            )
            errors = outcome.contract_errors()
            if not errors:
                snapshot = self.logs.wait_for(
                    _parse_stats,
                    after_index=attempt_cursor,
                    after_time_s=outcome.started_s,
                    timeout_s=self.config.idle_timeout_s,
                    process=self.process,
                )
                self.last_snapshot = snapshot
                self.measure_cursor = snapshot.line.index + 1
                return
            last_error = "; ".join(errors)
            time.sleep(0.25)
        tail = "\n".join(line.text for line in self.logs.lines()[-40:])
        raise AssertionError(
            f"server readiness timed out: {last_error}; log tail:\n{tail}"
        )

    def wait_for_idle_since(self, started_s: float) -> StatsSnapshot:
        assert self.process is not None
        snapshot = self.logs.wait_for(
            _parse_stats,
            after_index=self.measure_cursor,
            after_time_s=started_s,
            timeout_s=self.config.idle_timeout_s,
            process=self.process,
        )
        return snapshot

    def latest_iteration_limit(self, stop_index: int) -> IterationLimit:
        records = [
            record
            for line in self.logs.lines(0, stop_index)
            if (record := _parse_iteration(line)) is not None
        ]
        assert records, "missing public iteration-limit log line"
        latest = records[-1]
        assert latest.requested_tokens == self.pipeline.max_prefill_tokens
        assert 0 < latest.effective_tokens <= latest.requested_tokens
        return latest

    def finish_measurement(self, snapshot: StatsSnapshot) -> None:
        self.last_snapshot = snapshot
        self.measure_cursor = snapshot.line.index + 1


@dataclasses.dataclass(frozen=True)
class RequestSpec:
    label: str
    prompt: str
    max_tokens: int = 2


@dataclasses.dataclass
class PhaseResult:
    name: str
    requests: tuple[RequestSpec, ...]
    outcomes: tuple[CompletionOutcome, ...]
    waves: tuple[WaveRecord, ...]
    iteration: IterationLimit
    stats_before: StatsSnapshot
    stats_after: StatsSnapshot
    stats_delta: Mapping[str, int]
    total_expert_h2d: int

    @property
    def failure_count(self) -> int:
        return sum(bool(outcome.contract_errors()) for outcome in self.outcomes)


@dataclasses.dataclass
class ServiceResult:
    tree: Path
    pipeline: PipelineSpec
    public_serve_args: tuple[str, ...]
    phases: tuple[PhaseResult, ...]
    lifecycle_count: int = 1

    @property
    def outcomes(self) -> tuple[CompletionOutcome, ...]:
        return tuple(outcome for phase in self.phases for outcome in phase.outcomes)

    @property
    def failure_count(self) -> int:
        return sum(phase.failure_count for phase in self.phases)


def _unique_prompt(label: str, approximate_tokens: int) -> str:
    count = max(1, approximate_tokens)
    return f"{label}:" + " blackbox" * count


def _matrix_requests(pipeline: PipelineSpec) -> tuple[tuple[RequestSpec, ...], ...]:
    tokens = pipeline.max_prefill_tokens
    wave = pipeline.wave_max_chunks
    oversized = tokens * (wave + 8) + 17
    shapes = {
        1: [tokens + 13],
        2: [oversized, (tokens // 2) + 7],
        4: [oversized + 11, (tokens // 3) + 5, tokens + 19, 2 * tokens - 23],
        8: [
            oversized + 29,
            (tokens // 3) + 3,
            (tokens // 2) + 9,
            tokens + 7,
            2 * tokens - 17,
            3 * tokens + 23,
            (tokens // 4) + 11,
            tokens + 31,
        ],
    }
    phases: list[tuple[RequestSpec, ...]] = []
    for concurrency in (1, 2, 4, 8):
        phases.append(
            tuple(
                RequestSpec(
                    label=f"c{concurrency}-r{index}",
                    prompt=_unique_prompt(
                        f"{pipeline.name}-c{concurrency}-r{index}",
                        approximate,
                    ),
                )
                for index, approximate in enumerate(shapes[concurrency])
            )
        )
    repeated = _unique_prompt(f"{pipeline.name}-cached-followup", tokens * 2 + 37)
    phases.append((RequestSpec("cache-seed", repeated),))
    phases.append((RequestSpec("cache-followup", repeated),))
    return tuple(phases)


def _run_requests(
    server: PublicServer,
    requests: Sequence[RequestSpec],
) -> tuple[CompletionOutcome, ...]:
    if len(requests) == 1:
        request = requests[0]
        return (
            _post_completion(
                base_url=server.base_url,
                served_name=server.model.served_name,
                prompt=request.prompt,
                max_tokens=request.max_tokens,
                timeout_s=server.config.request_timeout_s,
                label=request.label,
            ),
        )
    barrier = threading.Barrier(len(requests))

    def issue(index: int, request: RequestSpec) -> CompletionOutcome:
        barrier.wait(timeout=10.0)
        time.sleep(index * server.config.submit_stagger_s)
        return _post_completion(
            base_url=server.base_url,
            served_name=server.model.served_name,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            timeout_s=server.config.request_timeout_s,
            label=request.label,
        )

    outcomes: list[CompletionOutcome | None] = [None] * len(requests)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
        futures = {
            pool.submit(issue, index, request): index
            for index, request in enumerate(requests)
        }
        for future, index in ((item, futures[item]) for item in futures):
            outcomes[index] = future.result(
                timeout=server.config.request_timeout_s + 20
            )
    assert all(outcome is not None for outcome in outcomes)
    return tuple(outcome for outcome in outcomes if outcome is not None)


def _logical_waves(
    outcomes: Sequence[CompletionOutcome],
    *,
    requested_tile_tokens: int,
    wave_max_chunks: int,
) -> list[list[CompletionOutcome]]:
    pending = [outcome for outcome in outcomes if outcome.uncached_prompt_rows > 0]
    planned: list[list[CompletionOutcome]] = []
    cursor = 0
    while cursor < len(pending):
        first = pending[cursor]
        members = [first]
        chunks = math.ceil(first.uncached_prompt_rows / requested_tile_tokens)
        cursor += 1
        while cursor < len(pending):
            candidate = pending[cursor]
            candidate_chunks = math.ceil(
                candidate.uncached_prompt_rows / requested_tile_tokens
            )
            if chunks + candidate_chunks > wave_max_chunks:
                break
            members.append(candidate)
            chunks += candidate_chunks
            cursor += 1
        planned.append(members)
    return planned


def _snapshot_delta(
    before: StatsSnapshot,
    after: StatsSnapshot,
) -> dict[str, int]:
    assert after.expert_row_bytes == before.expert_row_bytes
    assert after.expert_row_bytes > 0
    delta = {
        field: int(getattr(after, field)) - int(getattr(before, field))
        for field in COUNTER_FIELDS
    }
    assert all(value >= 0 for value in delta.values()), (
        "MoE stats counters moved backwards",
        delta,
    )
    return delta


def _assert_stats_accounting(
    *,
    delta: Mapping[str, int],
    expert_row_bytes: int,
    experts: int,
) -> int:
    assert delta["prefill_rows"] == delta["prefill_layer_prepares"] * experts
    assert 0 <= delta["prefill_hit_rows"] <= delta["prefill_rows"]
    total_expert_h2d = (
        delta["decode_missing_rows"] * expert_row_bytes
        + delta["prefill_h2d_bytes_total"]
    )
    assert total_expert_h2d >= 0
    return total_expert_h2d


def _assert_wave_formula(
    *,
    server: PublicServer,
    outcomes: Sequence[CompletionOutcome],
    waves: Sequence[WaveRecord],
    effective_tokens: int,
    require_decode_throughout: bool = False,
) -> None:
    planned = _logical_waves(
        outcomes,
        requested_tile_tokens=server.pipeline.max_prefill_tokens,
        wave_max_chunks=server.pipeline.wave_max_chunks,
    )
    assert len(waves) == len(planned), (
        "logical wave count mismatch",
        [len(group) for group in planned],
        [wave.reqs for wave in waves],
    )
    expected_groups = server.pipeline.groups(server.model)
    for observed, members in zip(waves, planned, strict=True):
        physical_tiles = math.ceil(
            sum(member.uncached_prompt_rows for member in members) / effective_tokens
        )
        iterations = expected_groups * physical_tiles
        assert observed.reqs == len(members)
        assert observed.groups == expected_groups
        assert observed.group_forwards == iterations
        assert observed.iterations == iterations
        assert observed.prefill_layer_prepares == server.model.layers
        assert 0 <= observed.decode_iterations <= observed.iterations
        if require_decode_throughout:
            assert observed.decode_iterations == observed.iterations


def _run_phase(
    server: PublicServer,
    *,
    name: str,
    requests: Sequence[RequestSpec],
) -> PhaseResult:
    assert server.last_snapshot is not None
    before = server.last_snapshot
    start_cursor = server.measure_cursor
    outcomes = _run_requests(server, requests)
    started_s = min(outcome.started_s for outcome in outcomes)
    after = server.wait_for_idle_since(started_s)
    phase_lines = server.logs.lines(start_cursor, after.line.index)
    waves = tuple(
        record for line in phase_lines if (record := _parse_wave(line)) is not None
    )
    iteration = server.latest_iteration_limit(after.line.index + 1)
    errors = {
        outcome.label: outcome.contract_errors()
        for outcome in outcomes
        if outcome.contract_errors()
    }
    if errors:
        print(f"PUBLIC FAILURE phase={name} errors={errors}", flush=True)
    assert not errors, f"{name} public completion failures: {errors}"
    _assert_wave_formula(
        server=server,
        outcomes=outcomes,
        waves=waves,
        effective_tokens=iteration.effective_tokens,
    )
    delta = _snapshot_delta(before, after)
    total_h2d = _assert_stats_accounting(
        delta=delta,
        expert_row_bytes=after.expert_row_bytes,
        experts=server.model.experts,
    )
    assert delta["prefill_layer_prepares"] == sum(
        wave.prefill_layer_prepares for wave in waves
    )
    server.finish_measurement(after)
    return PhaseResult(
        name=name,
        requests=tuple(requests),
        outcomes=outcomes,
        waves=waves,
        iteration=iteration,
        stats_before=before,
        stats_after=after,
        stats_delta=delta,
        total_expert_h2d=total_h2d,
    )


def _run_decode_overlap_phase(server: PublicServer) -> PhaseResult:
    assert server.last_snapshot is not None
    before = server.last_snapshot
    driver_first_text = threading.Event()
    driver_first_time: list[float] = []

    def first_text(at_s: float) -> None:
        driver_first_time.append(at_s)
        driver_first_text.set()

    driver_prompt = _unique_prompt(f"{server.pipeline.name}-decode-driver", 19)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        driver_future = pool.submit(
            _post_completion,
            base_url=server.base_url,
            served_name=server.model.served_name,
            prompt=driver_prompt,
            max_tokens=server.config.driver_max_tokens,
            timeout_s=server.config.request_timeout_s,
            label="decode-driver",
            first_text_callback=first_text,
        )
        assert driver_first_text.wait(
            timeout=server.config.request_timeout_s
        ), "decode driver never produced a public SSE text event"
        start_cursor = server.logs.cursor()
        requests = (
            RequestSpec(
                "decode-overlap-prefill-0",
                _unique_prompt(
                    f"{server.pipeline.name}-decode-overlap-0",
                    server.pipeline.max_prefill_tokens
                    * (server.pipeline.wave_max_chunks + 8)
                    + 41,
                ),
            ),
            RequestSpec(
                "decode-overlap-prefill-1",
                _unique_prompt(
                    f"{server.pipeline.name}-decode-overlap-1",
                    server.pipeline.max_prefill_tokens + 27,
                ),
            ),
        )
        outcomes = _run_requests(server, requests)
        driver = driver_future.result(timeout=server.config.request_timeout_s + 20)

    all_outcomes = (driver, *outcomes)
    errors = {
        outcome.label: outcome.contract_errors()
        for outcome in all_outcomes
        if outcome.contract_errors()
    }
    if errors:
        print(f"PUBLIC FAILURE phase=decode-overlap errors={errors}", flush=True)
    assert not errors, f"decode-overlap public completion failures: {errors}"
    after = server.wait_for_idle_since(min(item.started_s for item in all_outcomes))
    waves = tuple(
        record
        for line in server.logs.lines(start_cursor, after.line.index)
        if (record := _parse_wave(line)) is not None
    )
    assert waves, "decode-overlap phase produced no public wave completion log"
    assert driver_first_time
    assert all(wave.line.monotonic_s < driver.ended_s for wave in waves), (
        "decode driver ended before the measured prefill wave completed; increase "
        "driver_max_tokens in the black-box config"
    )
    iteration = server.latest_iteration_limit(after.line.index + 1)
    _assert_wave_formula(
        server=server,
        outcomes=outcomes,
        waves=waves,
        effective_tokens=iteration.effective_tokens,
        require_decode_throughout=True,
    )
    delta = _snapshot_delta(before, after)
    total_h2d = _assert_stats_accounting(
        delta=delta,
        expert_row_bytes=after.expert_row_bytes,
        experts=server.model.experts,
    )
    server.finish_measurement(after)
    return PhaseResult(
        name="decode-overlap",
        requests=requests,
        outcomes=all_outcomes,
        waves=waves,
        iteration=iteration,
        stats_before=before,
        stats_after=after,
        stats_delta=delta,
        total_expert_h2d=total_h2d,
    )


def _run_synthetic_service(
    *,
    config: BlackboxConfig,
    tree: Path,
    pipeline: PipelineSpec,
    port: int,
) -> ServiceResult:
    phases: list[PhaseResult] = []
    with PublicServer(
        config=config,
        tree=tree,
        model=config.synthetic,
        pipeline=pipeline,
        port=port,
    ) as server:
        public_args = server.public_serve_args
        for requests in _matrix_requests(pipeline):
            phases.append(
                _run_phase(
                    server,
                    name=requests[0].label.split("-r", 1)[0],
                    requests=requests,
                )
            )
        phases.append(_run_decode_overlap_phase(server))
    return ServiceResult(
        tree=tree,
        pipeline=pipeline,
        public_serve_args=public_args,
        phases=tuple(phases),
    )


def _assert_synthetic_coverage(
    model: ModelSpec,
    results: Iterable[ServiceResult],
) -> None:
    result_list = list(results)
    assert {result.pipeline.cuda_graph_max_bs for result in result_list} == {0, 8}
    assert any(
        model.layers % result.pipeline.effective_group(model) != 0
        for result in result_list
    ), "matrix does not exercise a partial final resident group"
    concurrencies: set[int] = set()
    saw_ragged_tile = False
    saw_oversized_logical_wave = False
    saw_reported_cache_hit = False
    for result in result_list:
        for phase in result.phases:
            if phase.name.startswith("c") and phase.name[1:2].isdigit():
                concurrencies.add(len(phase.requests))
            for outcome in phase.outcomes:
                if outcome.usage is None:
                    continue
                if outcome.uncached_prompt_rows % phase.iteration.effective_tokens:
                    saw_ragged_tile = True
                chunks = math.ceil(
                    outcome.uncached_prompt_rows / result.pipeline.max_prefill_tokens
                )
                if chunks > result.pipeline.wave_max_chunks:
                    saw_oversized_logical_wave = True
            if phase.name == "cache-followup":
                followup = phase.outcomes[0]
                if followup.cache_details_present:
                    assert followup.cached_tokens > 0
                    saw_reported_cache_hit = True
    assert concurrencies == {1, 2, 4, 8}
    assert saw_ragged_tile
    assert saw_oversized_logical_wave
    # Backends may omit prompt_tokens_details entirely.  When they expose it,
    # the repeated same-service request must observably use cached prompt rows.
    if any(
        phase.outcomes[0].cache_details_present
        for result in result_list
        for phase in result.phases
        if phase.name == "cache-followup"
    ):
        assert saw_reported_cache_hit


def _assert_ab_no_regression(
    candidate: ServiceResult,
    baseline: ServiceResult,
) -> None:
    assert candidate.public_serve_args == baseline.public_serve_args
    assert candidate.pipeline == baseline.pipeline
    assert [phase.name for phase in candidate.phases] == [
        phase.name for phase in baseline.phases
    ]
    assert [
        request.prompt for phase in candidate.phases for request in phase.requests
    ] == [request.prompt for phase in baseline.phases for request in phase.requests]
    assert candidate.failure_count <= baseline.failure_count
    assert candidate.failure_count == 0
    candidate_usage = [
        (
            outcome.prompt_tokens,
            outcome.completion_tokens,
        )
        for outcome in candidate.outcomes
    ]
    baseline_usage = [
        (
            outcome.prompt_tokens,
            outcome.completion_tokens,
        )
        for outcome in baseline.outcomes
    ]
    assert candidate_usage == baseline_usage


def _require_gpu_config() -> BlackboxConfig:
    if os.environ.get(RUN_ENV) != "1":
        pytest.skip(f"GPU black-box execution is opt-in; set {RUN_ENV}=1")
    return BlackboxConfig.load()


def test_public_cli_retains_layered_pipeline_flags() -> None:
    config = _require_gpu_config()
    required = (
        "--batching-policy",
        "--prefill-layer-group-size",
        "--prefill-wave-max-chunks",
        "--max-prefill-length",
        "--moe-cache-size",
    )
    for tree in (config.candidate_tree, config.baseline_tree):
        prefix = [part.format(tree=str(tree)) for part in config.launch_prefix]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(tree / "python"), str(tree / "benchmarks"), env.get("PYTHONPATH", "")]
        ).rstrip(os.pathsep)
        completed = subprocess.run(
            [*prefix, "serve", "--help"],
            cwd=tree,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30.0,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout
        for flag in required:
            assert flag in completed.stdout


def test_synthetic_matrix_candidate_vs_read_only_baseline() -> None:
    config = _require_gpu_config()
    assert (
        config.synthetic.path.exists()
    ), f"synthetic model path does not exist: {config.synthetic.path}"
    assert config.candidate_tree.exists()
    assert config.baseline_tree.exists()
    pipelines = config.synthetic_pipelines()
    pipeline_filter = os.environ.get(PIPELINE_FILTER_ENV)
    if pipeline_filter:
        pipelines = tuple(
            pipeline for pipeline in pipelines if pipeline.name == pipeline_filter
        )
        assert pipelines, f"unknown {PIPELINE_FILTER_ENV}={pipeline_filter!r}"
    for pipeline in pipelines:
        assert pipeline.effective_group(config.synthetic) >= 1
    if not pipeline_filter:
        assert any(
            config.synthetic.layers % pipeline.effective_group(config.synthetic) != 0
            for pipeline in pipelines
        ), "the configured synthetic L cannot cover a partial final group"

    candidate_results: list[ServiceResult] = []
    baseline_results: list[ServiceResult] = []
    for pipeline in pipelines:
        port = _unused_local_port()
        candidate = _run_synthetic_service(
            config=config,
            tree=config.candidate_tree,
            pipeline=pipeline,
            port=port,
        )
        baseline = _run_synthetic_service(
            config=config,
            tree=config.baseline_tree,
            pipeline=pipeline,
            port=port,
        )
        _assert_ab_no_regression(candidate, baseline)
        candidate_results.append(candidate)
        baseline_results.append(baseline)

    if not pipeline_filter:
        _assert_synthetic_coverage(config.synthetic, candidate_results)
        _assert_synthetic_coverage(config.synthetic, baseline_results)
        assert sum(result.lifecycle_count for result in candidate_results) == 2
        assert sum(result.lifecycle_count for result in baseline_results) == 2


@pytest.mark.parametrize("model_key", ("qwen36", "dsv4"))
@pytest.mark.parametrize("cuda_graph_max_bs", (0, 8))
def test_real_model_minimal_graph_modes(
    model_key: str,
    cuda_graph_max_bs: int,
) -> None:
    config = _require_gpu_config()
    model = config.real_models.get(model_key)
    if model is None:
        pytest.skip(f"{model_key} is not configured")
    if not model.path.exists():
        pytest.skip(f"{model_key} path does not exist: {model.path}")
    group_size = min(2, model.layers)
    pipeline = PipelineSpec(
        name=f"{model_key}-graph{cuda_graph_max_bs}",
        cuda_graph_max_bs=cuda_graph_max_bs,
        group_size=group_size,
        wave_max_chunks=1,
        max_prefill_tokens=128,
        cache_size=(group_size + 1) * model.experts,
    )
    port = _unused_local_port()
    with PublicServer(
        config=config,
        tree=config.candidate_tree,
        model=model,
        pipeline=pipeline,
        port=port,
    ) as server:
        result = _run_phase(
            server,
            name=f"{model_key}-graph{cuda_graph_max_bs}-minimal",
            requests=(
                RequestSpec(
                    label="minimal-ragged",
                    prompt=_unique_prompt(
                        f"{model_key}-graph{cuda_graph_max_bs}",
                        pipeline.max_prefill_tokens + 17,
                    ),
                    max_tokens=1,
                ),
            ),
        )
    assert result.failure_count == 0
    assert result.outcomes[0].uncached_prompt_rows % result.iteration.effective_tokens
