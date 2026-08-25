from __future__ import annotations

import json
import os
import queue
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path
from typing import Any, Callable

import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

from freetoken.core import Batch
from freetoken.moe.offload_cache import OffloadMoeCache, plan_expert_cache_partition
from freetoken.scheduler import SchedulerConfig
from freetoken.scheduler.layered_batch import (
    LayeredBatchComposer,
    LayeredBatchPlan,
    LayeredExecutionStats,
)
from freetoken.server.args import parse_args


def _config(**overrides: Any) -> SchedulerConfig:
    values: dict[str, Any] = {
        "model_path": "blackbox-unused",
        "tp_info": object(),
        "dtype": object(),
    }
    values.update(overrides)
    return SchedulerConfig(**values)


def test_scheduler_config_layered_defaults_are_public_contract() -> None:
    config = _config()

    assert config.batching_policy == "legacy"
    assert config.prefill_layer_group_size == 2
    assert config.prefill_execution == "serial"


@pytest.mark.parametrize("execution", ["serial", "concurrent"])
@pytest.mark.parametrize("group_size", [1, 2, 31])
def test_scheduler_config_accepts_layered_boundary_values(
    execution: str, group_size: int
) -> None:
    config = _config(
        batching_policy="layered",
        prefill_layer_group_size=group_size,
        prefill_execution=execution,
    )

    assert config.batching_policy == "layered"
    assert config.prefill_layer_group_size == group_size
    assert config.prefill_execution == execution


@pytest.mark.parametrize("group_size", [0, -1, -10_000])
def test_scheduler_config_rejects_non_positive_layer_groups(group_size: int) -> None:
    with pytest.raises(ValueError):
        _config(batching_policy="layered", prefill_layer_group_size=group_size)


@pytest.mark.parametrize("policy", ["legacy", "mixed"])
@pytest.mark.parametrize("execution", ["serial", "concurrent"])
def test_non_layered_config_keeps_selected_policy(
    policy: str, execution: str
) -> None:
    config = _config(
        batching_policy=policy,
        prefill_layer_group_size=7,
        prefill_execution=execution,
    )

    assert config.batching_policy == policy
    assert config.prefill_layer_group_size == 7
    assert config.prefill_execution == execution


class _DecodeManagerProbe:
    def __init__(self, result: Batch | None) -> None:
        self.result = result
        self.calls = 0

    def schedule_next_batch(self) -> Batch | None:
        self.calls += 1
        return self.result


class _PrefillManagerProbe:
    def __init__(self, result: Batch | None) -> None:
        self.result = result
        self.budgets: list[int] = []

    def schedule_next_batch(self, token_budget: int) -> Batch | None:
        self.budgets.append(token_budget)
        return self.result


def _composer(
    decode_result: Batch | None,
    prefill_result: Batch | None,
) -> tuple[LayeredBatchComposer, _DecodeManagerProbe, _PrefillManagerProbe]:
    decode = _DecodeManagerProbe(decode_result)
    prefill = _PrefillManagerProbe(prefill_result)
    composer = LayeredBatchComposer(
        prefill_manager=prefill,
        decode_manager=decode,
    )
    return composer, decode, prefill


def _decode_batch(count: int = 1) -> Batch:
    reqs = [object() for _ in range(count)]
    return Batch(reqs=reqs, decode_size=len(reqs))


def _prefill_batch(count: int = 1) -> Batch:
    return Batch(reqs=[object() for _ in range(count)], decode_size=0)


def test_layered_composer_returns_none_only_when_both_managers_are_idle() -> None:
    composer, decode, prefill = _composer(None, None)

    assert composer.schedule_next_plan(token_budget=64) is None
    assert decode.calls == 1
    assert prefill.budgets == [64]


@pytest.mark.parametrize(
    ("decode_present", "prefill_present"),
    [(True, False), (False, True), (True, True)],
)
def test_layered_composer_never_returns_a_double_empty_plan(
    decode_present: bool, prefill_present: bool
) -> None:
    decode_batch = _decode_batch(2) if decode_present else None
    prefill_batch = _prefill_batch() if prefill_present else None
    composer, decode, prefill = _composer(decode_batch, prefill_batch)

    plan = composer.schedule_next_plan(token_budget=17)

    assert plan is not None
    assert plan.decode_batch is decode_batch
    assert plan.prefill_batch is prefill_batch
    assert plan.decode_batch is not None or plan.prefill_batch is not None
    assert decode.calls == 1
    assert prefill.budgets == [17]


@pytest.mark.parametrize("token_budget", [0, 1, 64, 1_000_003])
def test_layered_composer_does_not_charge_decode_against_prefill_budget(
    token_budget: int,
) -> None:
    decode_batch = _decode_batch()
    prefill_batch = _prefill_batch()
    composer, decode, prefill = _composer(decode_batch, prefill_batch)

    plan = composer.schedule_next_plan(token_budget=token_budget)

    assert plan is not None
    assert plan.decode_batch is decode_batch
    assert plan.prefill_batch is prefill_batch
    assert decode.calls == 1
    assert prefill.budgets == [token_budget]


def test_layered_composer_keeps_batches_separate_and_requests_disjoint() -> None:
    decode_batch = _decode_batch(2)
    prefill_batch = _prefill_batch(2)
    composer, _, _ = _composer(decode_batch, prefill_batch)

    plan = composer.schedule_next_plan(token_budget=128)

    assert plan is not None
    assert plan.decode_batch is decode_batch
    assert plan.prefill_batch is prefill_batch
    assert plan.decode_batch is not plan.prefill_batch
    decode_req_ids = {id(req) for req in plan.decode_batch.reqs}
    prefill_req_ids = {id(req) for req in plan.prefill_batch.reqs}
    assert decode_req_ids.isdisjoint(prefill_req_ids)


def test_layered_composer_rejects_a_request_in_both_batches() -> None:
    shared_req = object()
    decode_batch = Batch(reqs=[shared_req], decode_size=1)
    prefill_batch = Batch(reqs=[shared_req], decode_size=0)
    composer, _, _ = _composer(decode_batch, prefill_batch)

    with pytest.raises(ValueError):
        composer.schedule_next_plan(token_budget=64)


def test_layered_composer_treats_empty_batches_as_no_work() -> None:
    composer, decode, prefill = _composer(
        Batch(reqs=[], decode_size=0),
        Batch(reqs=[], decode_size=0),
    )

    assert composer.schedule_next_plan(token_budget=64) is None
    assert decode.calls == 1
    assert prefill.budgets == [64]


def test_layered_batch_plan_is_frozen() -> None:
    decode_batch = _decode_batch()
    plan = LayeredBatchPlan(decode_batch=decode_batch, prefill_batch=None)

    with pytest.raises(FrozenInstanceError):
        plan.decode_batch = None  # type: ignore[misc]


@pytest.mark.parametrize(
    ("total_slots", "num_experts", "expected"),
    [
        (
            12,
            4,
            {
                "total_slots": 12,
                "decode_slots": 4,
                "prefill_buffer_slots": 8,
            },
        ),
        (
            13,
            4,
            {
                "total_slots": 13,
                "decode_slots": 5,
                "prefill_buffer_slots": 8,
            },
        ),
        (
            3,
            1,
            {
                "total_slots": 3,
                "decode_slots": 1,
                "prefill_buffer_slots": 2,
            },
        ),
    ],
)
def test_expert_cache_partition_reserves_two_full_prefill_buffers(
    total_slots: int, num_experts: int, expected: dict[str, int]
) -> None:
    partition = plan_expert_cache_partition(
        total_slots=total_slots,
        num_experts=num_experts,
        prefill_buffers=2,
    )

    assert partition == expected
    assert partition["decode_slots"] + partition["prefill_buffer_slots"] == total_slots


@pytest.mark.parametrize("total_slots", [4, 7, 12])
def test_expert_cache_partition_without_prefill_buffer_preserves_all_slots(
    total_slots: int,
) -> None:
    partition = plan_expert_cache_partition(
        total_slots=total_slots,
        num_experts=4,
        prefill_buffers=0,
    )

    assert partition == {
        "total_slots": total_slots,
        "decode_slots": total_slots,
        "prefill_buffer_slots": 0,
    }


@pytest.mark.parametrize("total_slots", [0, 1, 11])
def test_layered_cache_partition_rejects_less_than_three_expert_banks(
    total_slots: int,
) -> None:
    with pytest.raises(ValueError, match=r"requires at least 3 \* num_experts expert slots"):
        plan_expert_cache_partition(
            total_slots=total_slots,
            num_experts=4,
            prefill_buffers=2,
        )


def test_cache_object_reports_fixed_partition_and_reset_prefill_stats() -> None:
    cache = OffloadMoeCache(
        num_layers=2,
        num_experts=4,
        cache_size=12,
        device=torch.device("cpu"),
        prefill_overlap=True,
        separate_prefill_buffer=True,
    )

    assert cache.cache_partition() == {
        "total_slots": 12,
        "decode_slots": 4,
        "prefill_buffer_slots": 8,
    }
    before = cache.decode_miss_stats()
    assert "prefill_hit_rows" in before
    assert "prefill_rows" in before

    cache.reset_stats()

    after = cache.decode_miss_stats()
    assert after["prefill_hit_rows"] == 0
    assert after["prefill_rows"] == 0


def test_layered_execution_stats_snapshot_has_fixed_zero_schema() -> None:
    stats = LayeredExecutionStats()

    assert stats.snapshot() == {
        "joint_rounds": 0,
        "decode_forwards": 0,
        "prefill_group_steps": 0,
        "decode_gpu_ms": 0,
        "prefill_gpu_ms": 0,
        "joint_wall_ms": 0,
    }


def test_layered_execution_stats_snapshot_is_a_copy() -> None:
    stats = LayeredExecutionStats()
    first = stats.snapshot()
    first["joint_rounds"] = 999
    first["decode_gpu_ms"] = 123.0

    second = stats.snapshot()

    assert second["joint_rounds"] == 0
    assert second["decode_gpu_ms"] == 0


@pytest.fixture
def tiny_qwen3_moe_config_path(tmp_path: Path) -> Path:
    config = Qwen3MoeConfig(
        vocab_size=16,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    config.save_pretrained(tmp_path)
    return tmp_path


def test_cli_parser_exposes_layered_options_and_serial_default(
    tiny_qwen3_moe_config_path: Path,
) -> None:
    model_path = str(tiny_qwen3_moe_config_path)
    defaults, _ = parse_args(["--model-path", model_path])
    explicit, _ = parse_args(
        [
            "--model-path",
            model_path,
            "--batching-policy",
            "layered",
            "--prefill-layer-group-size",
            "7",
            "--prefill-execution",
            "concurrent",
            "--moe-backend",
            "offload",
            "--moe-cache-size",
            "12",
        ]
    )

    assert defaults.batching_policy == "legacy"
    assert defaults.prefill_layer_group_size == 2
    assert defaults.prefill_execution == "serial"
    assert explicit.batching_policy == "layered"
    assert explicit.prefill_layer_group_size == 7
    assert explicit.prefill_execution == "concurrent"


@pytest.mark.parametrize(
    "args",
    [
        ["--batching-policy", "not-a-policy"],
        ["--prefill-execution", "not-an-execution"],
    ],
)
def test_cli_parser_rejects_invalid_layered_choices(args: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--model-path", "blackbox-unused", *args])


def test_cli_help_lists_all_layered_flags_and_choices(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as raised:
        parse_args(["--help"])

    assert raised.value.code == 0
    help_text = capsys.readouterr().out
    assert "--batching-policy" in help_text
    assert "layered" in help_text
    assert "--prefill-layer-group-size" in help_text
    assert "--prefill-execution" in help_text
    assert "serial" in help_text
    assert "concurrent" in help_text


_E2E_MODEL_ENV = "FREETOKEN_LAYERED_MODEL_PATH"
_E2E_EXTRA_ARGS_ENV = "FREETOKEN_LAYERED_SERVER_EXTRA_ARGS"


@pytest.fixture(scope="session")
def tiny_qwen3_moe_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    model_path = tmp_path_factory.mktemp("layered-tiny-qwen3-moe")
    vocab = {
        "<pad>": 0,
        "<bos>": 1,
        "<eos>": 2,
        "<unk>": 3,
        "alpha": 4,
        "beta": 5,
        "gamma": 6,
        "delta": 7,
        "one": 8,
        "two": 9,
        "three": 10,
        "released": 11,
        ".": 12,
        ",": 13,
    }
    tokenizer_backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        bos_token="<bos>",
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<unk>",
    )
    tokenizer.save_pretrained(model_path)

    config = Qwen3MoeConfig(
        vocab_size=len(vocab),
        hidden_size=256,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=4096,
        bos_token_id=vocab["<bos>"],
        eos_token_id=vocab["<eos>"],
        pad_token_id=vocab["<pad>"],
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260825)
        model = Qwen3MoeForCausalLM(config).to(dtype=torch.bfloat16)
    model.eval()
    model.save_pretrained(model_path, safe_serialization=True)
    return model_path


def _e2e_model(tiny_qwen3_moe_model_path: Path) -> Path:
    value = os.environ.get(_E2E_MODEL_ENV)
    if not value:
        return tiny_qwen3_moe_model_path
    path = Path(value)
    if not path.exists():
        pytest.fail(f"{_E2E_MODEL_ENV} does not exist: {path}")
    return path


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _server_command(
    *,
    model_path: Path,
    port: int,
    policy: str,
    execution: str,
    max_running_requests: int,
    group_size: int = 2,
    attention_backend: str | None = None,
    tensor_parallel_size: int = 1,
    moe_cache_size: int | None = None,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model-path",
        str(model_path),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        "layered-blackbox",
        "--dtype",
        "bfloat16",
        "--max-running-requests",
        str(max_running_requests),
        "--max-seq-len-override",
        "4096",
        "--num-tokens",
        "4096",
        "--disable-pynccl",
        "--tensor-parallel-size",
        str(tensor_parallel_size),
        "--batching-policy",
        policy,
        "--prefill-layer-group-size",
        str(group_size),
        "--prefill-execution",
        execution,
        "--max-prefill-length",
        os.environ.get("FREETOKEN_LAYERED_MAX_PREFILL_TOKENS", "64"),
        "--moe-backend",
        "offload",
    ]
    if attention_backend is not None:
        command += ["--attention-backend", attention_backend]
    cache_size = moe_cache_size
    if cache_size is None and os.environ.get("FREETOKEN_LAYERED_MOE_CACHE_SIZE"):
        cache_size = int(os.environ["FREETOKEN_LAYERED_MOE_CACHE_SIZE"])
    if cache_size is None:
        cache_size = 12
    if cache_size is not None:
        command += ["--moe-cache-size", str(cache_size)]
    nowag_path = os.environ.get("FREETOKEN_LAYERED_NOWAG_EXPERT_PATH")
    if nowag_path:
        command += ["--nowag-expert-path", nowag_path]
    extra = os.environ.get(_E2E_EXTRA_ARGS_ENV)
    if extra:
        parsed = json.loads(extra)
        if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
            pytest.fail(f"{_E2E_EXTRA_ARGS_ENV} must be a JSON list of strings")
        command.extend(parsed)
    return command


@dataclass
class _ServerProcess:
    process: subprocess.Popen[str]
    log: tempfile._TemporaryFileWrapper[str]
    base_url: str
    started_at: float
    first_non_503_s: float | None = None

    def close(self) -> None:
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=15)
        self.log.close()


def _read_log_tail(log: tempfile._TemporaryFileWrapper[str], limit: int = 12_000) -> str:
    log.flush()
    log.seek(0)
    return log.read()[-limit:]


def _start_server(command: list[str], port: int) -> _ServerProcess:
    log = tempfile.NamedTemporaryFile(mode="w+", encoding="utf-8")
    started_at = time.monotonic()
    process = subprocess.Popen(
        command,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    server = _ServerProcess(
        process=process,
        log=log,
        base_url=f"http://127.0.0.1:{port}",
        started_at=started_at,
    )
    deadline = time.monotonic() + float(os.environ.get("FREETOKEN_LAYERED_START_TIMEOUT", "180"))
    health_path = os.environ.get("FREETOKEN_LAYERED_HEALTH_PATH", "/health")
    while time.monotonic() < deadline:
        if process.poll() is not None:
            pytest.fail(
                f"server exited with {process.returncode}\ncommand={command!r}\n{_read_log_tail(log)}"
            )
        try:
            with urllib.request.urlopen(server.base_url + health_path, timeout=2) as response:
                if 200 <= response.status < 300:
                    try:
                        _post_json(
                            server.base_url,
                            _completion_payload("alpha", max_tokens=1, stream=False),
                            timeout=float(
                                os.environ.get("FREETOKEN_LAYERED_503_RETRY_WINDOW", "180")
                            ),
                            retry_503=True,
                        )
                    except BaseException:
                        log_tail = _read_log_tail(log)
                        server.close()
                        pytest.fail(
                            "server was live but never accepted an inference request\n"
                            f"command={command!r}\n{log_tail}"
                        )
                    server.first_non_503_s = time.monotonic() - server.started_at
                    return server
        except (urllib.error.URLError, TimeoutError):
            time.sleep(0.2)
    log_tail = _read_log_tail(log)
    server.close()
    pytest.fail(f"server did not become healthy\ncommand={command!r}\n{log_tail}")


def _completion_payload(prompt: str, *, max_tokens: int, stream: bool) -> dict[str, Any]:
    endpoint = os.environ.get("FREETOKEN_LAYERED_COMPLETION_PATH", "/v1/completions")
    common: dict[str, Any] = {
        "model": "layered-blackbox",
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
        "ignore_eos": True,
    }
    if endpoint.endswith("/chat/completions"):
        common["messages"] = [{"role": "user", "content": prompt}]
    else:
        common["prompt"] = prompt
    return common


def _extract_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        choice = choices[0]
        if isinstance(choice, dict):
            if isinstance(choice.get("text"), str):
                return choice["text"]
            message = choice.get("message")
            if isinstance(message, dict) and isinstance(message.get("content"), str):
                return message["content"]
            delta = choice.get("delta")
            if isinstance(delta, dict) and isinstance(delta.get("content"), str):
                return delta["content"]
    if isinstance(payload.get("output_text"), str):
        return payload["output_text"]
    return ""


def _urlopen_with_bounded_503_retry(
    request: urllib.request.Request,
    *,
    timeout: float,
    retry_503: bool,
):
    retry_window = float(os.environ.get("FREETOKEN_LAYERED_503_RETRY_WINDOW", "180"))
    deadline = time.monotonic() + min(timeout, retry_window)
    while True:
        try:
            return urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if not retry_503 or exc.code != 503 or time.monotonic() >= deadline:
                raise
            exc.close()
            time.sleep(0.2)


def _post_json(
    base_url: str,
    payload: dict[str, Any],
    timeout: float = 120,
    *,
    retry_503: bool = False,
) -> dict[str, Any]:
    path = os.environ.get("FREETOKEN_LAYERED_COMPLETION_PATH", "/v1/completions")
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with _urlopen_with_bounded_503_retry(
        request,
        timeout=timeout,
        retry_503=retry_503,
    ) as response:
        return json.loads(response.read().decode("utf-8"))


def _stream_text(
    base_url: str,
    payload: dict[str, Any],
    *,
    first_token: threading.Event | None = None,
    close_after_first: bool = False,
    timeout: float = 120,
    on_event: Callable[[int], None] | None = None,
) -> tuple[str, int]:
    path = os.environ.get("FREETOKEN_LAYERED_COMPLETION_PATH", "/v1/completions")
    request = urllib.request.Request(
        base_url + path,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    pieces: list[str] = []
    data_events = 0
    with _urlopen_with_bounded_503_retry(
        request,
        timeout=timeout,
        retry_503=False,
    ) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            event = json.loads(body)
            data_events += 1
            if on_event is not None:
                on_event(data_events)
            if first_token is not None:
                first_token.set()
            piece = _extract_text(event)
            if piece:
                pieces.append(piece)
            if close_after_first:
                break
    return "".join(pieces), data_events


@dataclass(frozen=True)
class _JointRun:
    decode_text: str
    prefill_text: str
    decode_events: int
    prefill_events: int
    decode_completion_s: float
    prefill_completion_s: float
    wall_s: float
    overlap_window_s: float
    decode_events_during_prefill: int

    @property
    def event_throughput_per_s(self) -> float:
        return (self.decode_events + self.prefill_events) / self.wall_s

    def metrics(self) -> dict[str, float | int]:
        return {
            "decode_completion_s": self.decode_completion_s,
            "prefill_completion_s": self.prefill_completion_s,
            "wall_s": self.wall_s,
            "overlap_window_s": self.overlap_window_s,
            "decode_events": self.decode_events,
            "prefill_events": self.prefill_events,
            "decode_events_during_prefill": self.decode_events_during_prefill,
            "event_throughput_per_s": self.event_throughput_per_s,
        }


def _workload_nonce(workload_id: int) -> str:
    words = ("alpha", "beta", "gamma", "delta")
    value = workload_id
    digits: list[str] = []
    for _ in range(8):
        digits.append(words[value % len(words)])
        value //= len(words)
    return " ".join(digits)


def _run_joint_pair(base_url: str, *, workload_id: int = 0) -> _JointRun:
    first_token = threading.Event()
    result: queue.Queue[tuple[str, str | BaseException, int, float]] = queue.Queue()
    progress_lock = threading.Lock()
    decode_progress = 0
    decode_prompt = os.environ.get(
        "FREETOKEN_LAYERED_DECODE_PROMPT",
        "Count upward from one, writing one integer per token.",
    )
    prefill_body = os.environ.get(
        "FREETOKEN_LAYERED_PREFILL_PROMPT",
        "Summarize this repeated context in one sentence: " + "alpha beta gamma delta " * 512,
    )
    prefill_prompt = f"{_workload_nonce(workload_id)} {prefill_body}"
    decode_max_tokens = int(os.environ.get("FREETOKEN_LAYERED_DECODE_MAX_TOKENS", "128"))
    prefill_max_tokens = int(os.environ.get("FREETOKEN_LAYERED_PREFILL_MAX_TOKENS", "8"))
    timeout = float(os.environ.get("FREETOKEN_LAYERED_JOINT_TIMEOUT", "300"))

    def record_decode_progress(events: int) -> None:
        nonlocal decode_progress
        with progress_lock:
            decode_progress = events

    def consume_decode() -> None:
        try:
            text, events = _stream_text(
                base_url,
                _completion_payload(
                    decode_prompt,
                    max_tokens=decode_max_tokens,
                    stream=True,
                ),
                first_token=first_token,
                timeout=timeout,
                on_event=record_decode_progress,
            )
            result.put(("ok", text, events, time.monotonic()))
        except BaseException as exc:  # surfaced in the test thread
            result.put(("error", exc, 0, time.monotonic()))

    started_at = time.monotonic()
    thread = threading.Thread(target=consume_decode, daemon=True)
    thread.start()
    if not first_token.wait(timeout=min(120.0, timeout)):
        if not result.empty():
            status, decode_result, _, _ = result.get_nowait()
            if status == "error":
                raise decode_result  # type: ignore[misc]
        pytest.fail("decode stream produced no first token")

    assert thread.is_alive(), "decode completed before long prefill submission"
    with progress_lock:
        events_before_prefill = decode_progress
    prefill_submitted_at = time.monotonic()
    assert thread.is_alive(), "decode was not alive when long prefill was submitted"
    prefill_text, prefill_events = _stream_text(
        base_url,
        _completion_payload(
            prefill_prompt,
            max_tokens=prefill_max_tokens,
            stream=True,
        ),
        timeout=timeout,
    )
    prefill_done_at = time.monotonic()
    with progress_lock:
        events_at_prefill_done = decode_progress
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "decode stream did not finish"
    status, decode_result, decode_events, decode_done_at = result.get_nowait()
    if status == "error":
        raise decode_result  # type: ignore[misc]
    assert decode_done_at > prefill_submitted_at, (
        "decode and prefill request lifetimes did not overlap"
    )
    wall_done_at = max(decode_done_at, prefill_done_at)
    return _JointRun(
        decode_text=str(decode_result),
        prefill_text=prefill_text,
        decode_events=decode_events,
        prefill_events=prefill_events,
        decode_completion_s=decode_done_at - started_at,
        prefill_completion_s=prefill_done_at - started_at,
        wall_s=wall_done_at - started_at,
        overlap_window_s=min(decode_done_at, prefill_done_at) - prefill_submitted_at,
        decode_events_during_prefill=events_at_prefill_done - events_before_prefill,
    )


def test_real_server_group_sizes_and_execution_modes_match_outputs(
    tiny_qwen3_moe_model_path: Path,
) -> None:
    model_path = _e2e_model(tiny_qwen3_moe_model_path)
    reference_outputs: dict[int, tuple[str, str]] = {}
    group_observations: dict[str, list[dict[str, float | int]]] = {}
    modes = [
        ("legacy-serial", "legacy", "serial", 2),
        ("layered-serial-g1", "layered", "serial", 1),
        ("layered-serial-g2", "layered", "serial", 2),
        ("layered-serial-g4", "layered", "serial", 4),
        ("layered-concurrent-g2", "layered", "concurrent", 2),
    ]
    for name, policy, execution, group_size in modes:
        port = _free_port()
        command = _server_command(
            model_path=model_path,
            port=port,
            policy=policy,
            execution=execution,
            max_running_requests=2,
            group_size=group_size,
            attention_backend="triton",
        )
        server = _start_server(command, port)
        try:
            assert server.first_non_503_s is not None
            repetitions = 3 if policy == "layered" and execution == "serial" else 1
            for workload_id in range(repetitions):
                run = _run_joint_pair(server.base_url, workload_id=workload_id)
                assert run.decode_events > 0
                assert run.prefill_events > 0
                assert run.overlap_window_s > 0
                current_output = (run.decode_text, run.prefill_text)
                if workload_id in reference_outputs:
                    assert current_output == reference_outputs[workload_id], name
                else:
                    reference_outputs[workload_id] = current_output
                if policy == "layered" and execution == "serial":
                    group_observations.setdefault(name, []).append(run.metrics())
        finally:
            server.close()

    group_report = {
        "schema": "freetoken-layer-group-observation-v1",
        "num_hidden_layers": 16,
        "outputs_identical": True,
        "actual_overlap_verified": True,
        "groups": {
            name: {
                "runs": runs,
                "median": {
                    key: float(statistics.median(float(run[key]) for run in runs))
                    for key in (
                        "decode_completion_s",
                        "prefill_completion_s",
                        "wall_s",
                        "overlap_window_s",
                        "decode_events_during_prefill",
                        "event_throughput_per_s",
                    )
                },
            }
            for name, runs in group_observations.items()
        },
    }
    print(
        "FREETOKEN_LAYER_GROUP_JSON=" + json.dumps(group_report, sort_keys=True),
        flush=True,
    )


def test_real_server_abort_once_releases_single_request_capacity(
    tiny_qwen3_moe_model_path: Path,
) -> None:
    model_path = _e2e_model(tiny_qwen3_moe_model_path)
    port = _free_port()
    command = _server_command(
        model_path=model_path,
        port=port,
        policy="layered",
        execution="concurrent",
        max_running_requests=1,
        attention_backend="triton",
    )
    server = _start_server(command, port)
    try:
        prompt = "Continue producing short words until stopped."
        _, events = _stream_text(
            server.base_url,
            _completion_payload(prompt, max_tokens=256, stream=True),
            close_after_first=True,
        )
        assert events >= 1

        response = _post_json(
            server.base_url,
            _completion_payload("Reply with the word released.", max_tokens=4, stream=False),
            timeout=60,
            retry_503=False,
        )
        assert isinstance(_extract_text(response), str)
    finally:
        server.close()


def _expect_startup_error(command: list[str], expected: str) -> None:
    completed = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=float(os.environ.get("FREETOKEN_LAYERED_ERROR_TIMEOUT", "180")),
        env=os.environ.copy(),
    )
    assert completed.returncode != 0, completed.stdout
    assert expected in completed.stdout


def test_real_server_rejects_dense_model_for_layered_mode() -> None:
    value = os.environ.get("FREETOKEN_LAYERED_DENSE_MODEL_PATH")
    if not value:
        pytest.skip("set FREETOKEN_LAYERED_DENSE_MODEL_PATH to run startup validation")
    model_path = Path(value)
    command = _server_command(
        model_path=model_path,
        port=_free_port(),
        policy="layered",
        execution="serial",
        max_running_requests=1,
    )
    _expect_startup_error(command, "layered batching requires an offloaded MoE model")


def test_real_server_rejects_model_without_layer_group_support() -> None:
    value = os.environ.get("FREETOKEN_LAYERED_UNSUPPORTED_MODEL_PATH")
    if not value:
        pytest.skip("set FREETOKEN_LAYERED_UNSUPPORTED_MODEL_PATH to run startup validation")
    model_path = Path(value)
    command = _server_command(
        model_path=model_path,
        port=_free_port(),
        policy="layered",
        execution="serial",
        max_running_requests=1,
    )
    _expect_startup_error(command, "does not support layer-group prefill")


def test_real_server_rejects_too_small_layered_expert_cache(
    tiny_qwen3_moe_model_path: Path,
) -> None:
    model_path = _e2e_model(tiny_qwen3_moe_model_path)
    num_experts = int(os.environ.get("FREETOKEN_LAYERED_NUM_EXPERTS", "4"))
    command = _server_command(
        model_path=model_path,
        port=_free_port(),
        policy="layered",
        execution="serial",
        max_running_requests=1,
        moe_cache_size=3 * num_experts - 1,
    )
    _expect_startup_error(command, "requires at least 3 * num_experts expert slots")


def test_real_server_rejects_concurrent_layered_tensor_parallelism(
    tiny_qwen3_moe_model_path: Path,
) -> None:
    model_path = _e2e_model(tiny_qwen3_moe_model_path)
    command = _server_command(
        model_path=model_path,
        port=_free_port(),
        policy="layered",
        execution="concurrent",
        max_running_requests=1,
        attention_backend="triton",
        tensor_parallel_size=2,
    )
    _expect_startup_error(
        command,
        "concurrent layered prefill requires tensor_parallel_size=1",
    )


def test_real_server_rejects_concurrent_layered_without_triton(
    tiny_qwen3_moe_model_path: Path,
) -> None:
    model_path = _e2e_model(tiny_qwen3_moe_model_path)
    command = _server_command(
        model_path=model_path,
        port=_free_port(),
        policy="layered",
        execution="concurrent",
        max_running_requests=1,
        attention_backend=None,
    )
    _expect_startup_error(command, "use --attention-backend triton")


def _median_metrics(runs: list[_JointRun]) -> dict[str, float]:
    keys = (
        "decode_completion_s",
        "prefill_completion_s",
        "wall_s",
        "overlap_window_s",
        "decode_events_during_prefill",
        "event_throughput_per_s",
    )
    raw = [run.metrics() for run in runs]
    return {
        key: float(statistics.median(float(item[key]) for item in raw))
        for key in keys
    }


def test_real_server_environment_controlled_performance_ab(
    tiny_qwen3_moe_model_path: Path,
) -> None:
    if os.environ.get("FREETOKEN_LAYERED_RUN_PERF") != "1":
        pytest.skip("set FREETOKEN_LAYERED_RUN_PERF=1 to run the measured A/B")
    repeats = int(os.environ.get("FREETOKEN_LAYERED_PERF_REPEATS", "3"))
    assert repeats >= 3, "performance A/B requires at least three measured repetitions"
    model_path = _e2e_model(tiny_qwen3_moe_model_path)
    modes = [
        ("legacy-serial", "legacy", "serial", 2),
        ("layered-serial-g2", "layered", "serial", 2),
        ("layered-concurrent-g2", "layered", "concurrent", 2),
    ]
    report: dict[str, Any] = {
        "schema": "freetoken-layered-blackbox-perf-v1",
        "repeats": repeats,
        "outputs_identical": False,
        "modes": {},
    }
    reference_outputs: dict[int, tuple[str, str]] = {}
    for name, policy, execution, group_size in modes:
        port = _free_port()
        command = _server_command(
            model_path=model_path,
            port=port,
            policy=policy,
            execution=execution,
            max_running_requests=2,
            group_size=group_size,
            attention_backend="triton",
        )
        server = _start_server(command, port)
        try:
            warmup = _run_joint_pair(server.base_url, workload_id=10_000)
            assert warmup.overlap_window_s > 0
            measured = [
                _run_joint_pair(server.base_url, workload_id=workload_id)
                for workload_id in range(repeats)
            ]
            for workload_id, run in enumerate(measured):
                assert run.overlap_window_s > 0
                current_output = (run.decode_text, run.prefill_text)
                reference_for_id = reference_outputs.get(workload_id)
                if reference_for_id is None:
                    reference_outputs[workload_id] = current_output
                else:
                    assert current_output == reference_for_id
            report["modes"][name] = {
                "first_non_503_s": server.first_non_503_s,
                "warmup": warmup.metrics(),
                "runs": [run.metrics() for run in measured],
                "median": _median_metrics(measured),
            }
        finally:
            server.close()
    report["outputs_identical"] = True
    print("FREETOKEN_LAYERED_PERF_JSON=" + json.dumps(report, sort_keys=True), flush=True)
