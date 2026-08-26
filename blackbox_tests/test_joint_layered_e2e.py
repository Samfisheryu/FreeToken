"""Black-box contract tests for the public ``joint`` serve policy.

The tests deliberately know nothing about FreeToken's Python implementation.
They create a public Transformers checkpoint, launch ``ft serve``, and observe
only CLI exit status, process logs, HTTP responses, and SSE events.

The GPU/model tests are opt-in because they start the real server repeatedly::

    FT_RUN_JOINT_E2E=1 \
      pytest -q blackbox_tests/test_joint_layered_e2e.py -s

``FT_CLI`` may contain a shell-style executable prefix (for example
``uv run ft``). ``FT_SERVE_EXTRA_ARGS`` may contain additional public serve
arguments required by the test machine. Neither variable may replace the
policy, layer-group, model, port, cache-size, or chunk-size arguments under
test.

Every policy is run with ``--attention-backend triton`` and
``--cuda-graph-max-bs 0``. Joint supports only triton in this release, so the
same backend is required for strict output and timing comparisons. The
synthetic checkpoint is incompatible with graph capture in the public offload
path, and CUDA graphs are outside this scheduling experiment.
"""

from __future__ import annotations

import concurrent.futures
import contextlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import socket
import statistics
import subprocess
import threading
import time
from typing import Callable, Iterator
import urllib.error
import urllib.request

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUN_E2E = os.environ.get("FT_RUN_JOINT_E2E") == "1"

VOCAB_SIZE = 512
NUM_LAYERS = 5
NUM_EXPERTS = 8
PREFILL_CHUNK_SIZE = 32
PREFILL_WAVE_MAX_CHUNKS = 2
SERVER_START_TIMEOUT = float(os.environ.get("FT_SERVER_START_TIMEOUT", "180"))
REQUEST_TIMEOUT = float(os.environ.get("FT_REQUEST_TIMEOUT", "180"))
_NEXT_SERVER_PORT = 18000


def _tokens(start: int, count: int) -> str:
    return " ".join(f"tok{4 + ((start + index) % (VOCAB_SIZE - 4))}" for index in range(count))


CASES: dict[str, tuple[str, int]] = {
    "single_chunk": (_tokens(0, 12), 8),
    "multi_chunk": (_tokens(20, 160), 8),
    "decode_a": (_tokens(40, 9), 32),
    "decode_b": (_tokens(70, 11), 32),
    "decode_c": (_tokens(100, 13), 32),
    "decode_active": (_tokens(140, 16), 192),
    "prefill_a": (_tokens(180, 160), 1),
    "prefill_b": (_tokens(230, 176), 1),
    "abort_capacity": (_tokens(300, 64), 160),
}


def _public_environment() -> dict[str, str]:
    env = os.environ.copy()
    public_paths = f"{PROJECT_ROOT / 'python'}:{PROJECT_ROOT}"
    old_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = f"{public_paths}:{old_pythonpath}" if old_pythonpath else public_paths
    return env


def _ft_prefix() -> list[str]:
    configured = os.environ.get("FT_CLI")
    if configured:
        return shlex.split(configured)
    installed = shutil.which("ft")
    if installed:
        return [installed]
    pytest.skip("ft executable not found; set FT_CLI or add ft to PATH")


def _serve_help() -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*_ft_prefix(), "serve", "--help"],
        cwd=PROJECT_ROOT,
        env=_public_environment(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=30,
        check=False,
    )


def _free_port() -> int:
    global _NEXT_SERVER_PORT
    while _NEXT_SERVER_PORT < 28000:
        port = _NEXT_SERVER_PORT
        _NEXT_SERVER_PORT += 2
        sockets = [
            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
            socket.socket(socket.AF_INET, socket.SOCK_STREAM),
        ]
        try:
            sockets[0].bind(("127.0.0.1", port))
            sockets[1].bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            pass
        finally:
            for sock in sockets:
                sock.close()
    raise RuntimeError("no adjacent free ports in the non-ephemeral test range 18000-27999")


def _serve_command(
    checkpoint: Path,
    port: int,
    policy: str,
    *,
    group_size: int | None = None,
    moe_cache_size: int = 16,
    num_tokens: int | None = None,
    disable_prefill_overlap: bool = False,
    attention_backend: str | None = None,
    prefill_execution: str | None = None,
    prefill_wave_max_chunks: int = PREFILL_WAVE_MAX_CHUNKS,
    dtype: str = "float16",
) -> list[str]:
    command = [
        *_ft_prefix(),
        "serve",
        "--model-path",
        str(checkpoint),
        "--dtype",
        dtype,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--max-running-requests",
        "8",
        "--max-seq-len-override",
        "512",
        "--max-output-tokens",
        "256",
        "--cuda-graph-max-bs",
        "0",
        "--max-prefill-length",
        str(PREFILL_CHUNK_SIZE),
        "--prefill-wave-max-chunks",
        str(prefill_wave_max_chunks),
        "--batching-policy",
        policy,
        "--moe-backend",
        "offload",
        "--moe-cache-size",
        str(moe_cache_size),
    ]
    if group_size is not None:
        command.extend(["--prefill-layer-group-size", str(group_size)])
    if num_tokens is not None:
        command.extend(["--num-tokens", str(num_tokens)])
    if disable_prefill_overlap:
        command.append("--disable-moe-prefill-overlap")
    if attention_backend is None:
        attention_backend = "triton"
    if attention_backend is not None:
        command.extend(["--attention-backend", attention_backend])
    if prefill_execution is not None:
        command.extend(["--prefill-execution", prefill_execution])
    command.extend(shlex.split(os.environ.get("FT_SERVE_EXTRA_ARGS", "")))
    return command


def _read_url_error(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8", errors="replace")
    except Exception:
        return str(error)


class PublicCompletionClient:
    """Small client for the documented OpenAI-compatible surface."""

    def __init__(self, port: int, model: str):
        self.base_url = f"http://127.0.0.1:{port}"
        self.model = model

    def _request(self, body: dict[str, object]) -> urllib.response.addinfourl:
        request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            return urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT)
        except urllib.error.HTTPError as error:
            detail = _read_url_error(error)
            raise AssertionError(f"completion failed with HTTP {error.code}: {detail}") from error

    def complete(self, prompt: str, max_tokens: int, *, stream: bool = False) -> str:
        body: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        }
        if not stream:
            with self._request(body) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return str(payload["choices"][0]["text"])

        text_parts: list[str] = []
        with self._request(body) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                piece = str(event["choices"][0].get("text", ""))
                text_parts.append(piece)
        return "".join(text_parts)

    def stream_with_observer(
        self,
        prompt: str,
        max_tokens: int,
        on_text: Callable[[str, float], None],
        *,
        abort_after_first_text: bool = False,
    ) -> str:
        body: dict[str, object] = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }
        text_parts: list[str] = []
        response = self._request(body)
        try:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                piece = str(event["choices"][0].get("text", ""))
                if not piece:
                    continue
                text_parts.append(piece)
                on_text(piece, time.perf_counter())
                if abort_after_first_text:
                    break
        finally:
            response.close()
        return "".join(text_parts)


class PublicServer:
    def __init__(self, command: list[str], port: int, log_path: Path):
        self.command = command
        self.port = port
        self.log_path = log_path
        self.process: subprocess.Popen[str] | None = None
        self._log_file = None
        self.client: PublicCompletionClient | None = None

    def start(self) -> "PublicServer":
        self._log_file = self.log_path.open("w+", encoding="utf-8")
        self.process = subprocess.Popen(
            self.command,
            cwd=PROJECT_ROOT,
            env=_public_environment(),
            text=True,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.monotonic() + SERVER_START_TIMEOUT
        last_error = "server did not accept connections"
        model: str | None = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise AssertionError(
                    f"server exited with {self.process.returncode} before readiness\n{self.logs()}"
                )
            if model is None:
                request = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/models")
                try:
                    with urllib.request.urlopen(request, timeout=2) as response:
                        payload = json.loads(response.read().decode("utf-8"))
                    model = str(payload["data"][0]["id"])
                except urllib.error.HTTPError as error:
                    detail = _read_url_error(error)
                    raise AssertionError(
                        f"model discovery failed with HTTP {error.code}: {detail}"
                    ) from error
                except (KeyError, IndexError, json.JSONDecodeError) as error:
                    raise AssertionError(f"invalid /v1/models response: {error}") from error
                except OSError as error:
                    last_error = str(error)
                    time.sleep(0.25)
                    continue

            ready_body = {
                "model": model,
                "prompt": _tokens(0, 1),
                "max_tokens": 1,
                "temperature": 0.0,
                "stream": False,
            }
            ready_request = urllib.request.Request(
                f"http://127.0.0.1:{self.port}/v1/completions",
                data=json.dumps(ready_body).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(ready_request, timeout=REQUEST_TIMEOUT) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                if not payload.get("choices"):
                    raise AssertionError(f"invalid readiness completion response: {payload}")
                self.client = PublicCompletionClient(self.port, model)
                return self
            except urllib.error.HTTPError as error:
                detail = _read_url_error(error)
                if error.code == 503 and "model is still loading" in detail.lower():
                    last_error = f"HTTP 503: {detail}"
                    time.sleep(0.25)
                    continue
                raise AssertionError(
                    f"readiness completion failed with HTTP {error.code}: {detail}"
                ) from error
            except json.JSONDecodeError as error:
                raise AssertionError(f"invalid readiness completion JSON: {error}") from error
            except OSError as error:
                raise AssertionError(f"readiness completion connection failed: {error}") from error
        raise AssertionError(f"server readiness timed out: {last_error}\n{self.logs()}")

    def logs(self) -> str:
        if self._log_file is None:
            return ""
        self._log_file.flush()
        return self.log_path.read_text(encoding="utf-8", errors="replace")

    def stop(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        if self._log_file is not None:
            self._log_file.close()


@contextlib.contextmanager
def _running_server(
    checkpoint: Path,
    log_dir: Path,
    policy: str,
    *,
    group_size: int | None = None,
    moe_cache_size: int = 16,
    num_tokens: int | None = None,
    attention_backend: str | None = None,
    prefill_execution: str | None = None,
    prefill_wave_max_chunks: int = PREFILL_WAVE_MAX_CHUNKS,
    dtype: str = "float16",
) -> Iterator[PublicServer]:
    port = _free_port()
    unique = f"{policy}-g{group_size}-{port}"
    server = PublicServer(
        _serve_command(
            checkpoint,
            port,
            policy,
            group_size=group_size,
            moe_cache_size=moe_cache_size,
            num_tokens=num_tokens,
            attention_backend=attention_backend,
            prefill_execution=prefill_execution,
            prefill_wave_max_chunks=prefill_wave_max_chunks,
            dtype=dtype,
        ),
        port,
        log_dir / f"{unique}.log",
    )
    try:
        yield server.start()
    finally:
        server.stop()


@pytest.fixture(scope="session")
def checkpoint_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast, Qwen3MoeConfig, Qwen3MoeForCausalLM

    path = tmp_path_factory.mktemp("joint-qwen3-moe")
    vocab = {"[UNK]": 0, "[PAD]": 1, "<s>": 2, "</s>": 3}
    vocab.update({f"tok{index}": index for index in range(4, VOCAB_SIZE)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        bos_token="<s>",
        eos_token="</s>",
    )
    tokenizer.model_max_length = 512
    tokenizer.save_pretrained(path)

    torch.manual_seed(7)
    config = Qwen3MoeConfig(
        vocab_size=VOCAB_SIZE,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=512,
        decoder_sparse_step=1,
        moe_intermediate_size=1024,
        num_experts_per_tok=2,
        num_experts=NUM_EXPERTS,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=None,
        tie_word_embeddings=False,
    )
    config.architectures = ["Qwen3MoeForCausalLM"]
    model = Qwen3MoeForCausalLM(config)

    # The routed expert tensors alone exceed 60 MiB in float32. This keeps the
    # checkpoint small enough for E2E while making repeated expert transfers a
    # material part of the timed workload.
    routed_expert_bytes = NUM_LAYERS * NUM_EXPERTS * 3 * 128 * 1024 * 4
    assert routed_expert_bytes >= 60 * 1024 * 1024
    model.save_pretrained(path, safe_serialization=True, max_shard_size="2GB")
    return path


@pytest.fixture(scope="session")
def log_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("joint-server-logs")


@pytest.fixture(scope="session")
def reference_outputs(checkpoint_dir: Path, log_dir: Path) -> dict[str, str]:
    """Establish strict legacy/mixed output equality before joint comparisons.

    Detects: an existing-policy mismatch or nondeterministic checkpoint, which
    would make a joint-policy comparison ambiguous. Action: fail immediately
    and fix the public serving determinism/workload before judging joint.
    """

    if not RUN_E2E:
        pytest.skip("set FT_RUN_JOINT_E2E=1 to run real-server tests")

    by_policy: dict[str, dict[str, str]] = {}
    for policy in ("legacy", "mixed"):
        with _running_server(checkpoint_dir, log_dir, policy) as server:
            assert server.client is not None
            by_policy[policy] = {
                name: server.client.complete(prompt, max_tokens)
                for name, (prompt, max_tokens) in CASES.items()
            }
    assert by_policy["legacy"] == by_policy["mixed"]
    return by_policy["legacy"]


@pytest.fixture(scope="module")
def joint_group_2_server(checkpoint_dir: Path, log_dir: Path) -> Iterator[PublicServer]:
    if not RUN_E2E:
        pytest.skip("set FT_RUN_JOINT_E2E=1 to run real-server tests")
    with _running_server(
        checkpoint_dir,
        log_dir,
        "joint",
        group_size=2,
        moe_cache_size=3 * NUM_EXPERTS,
    ) as server:
        yield server


def test_help_advertises_joint_public_contract() -> None:
    """Detects a missing/renamed public policy; action is to reject the build."""

    result = _serve_help()
    assert result.returncode == 0, result.stdout
    assert "--prefill-layer-group-size" in result.stdout
    assert "--prefill-wave-max-chunks" in result.stdout
    choices_match = re.search(r"--batching-policy\s+\{([^}]*)\}", result.stdout)
    assert choices_match is not None, result.stdout
    assert "joint" in {choice.strip() for choice in choices_match.group(1).split(",")}


def test_joint_checkpoint_starts_and_serves_public_completion(
    checkpoint_dir: Path,
    log_dir: Path,
) -> None:
    """Detects an unusable public checkpoint/joint startup path.

    Action: stop the functional E2E run and report the first public process log
    or HTTP error before testing scheduling semantics.
    """

    if not RUN_E2E:
        pytest.skip("set FT_RUN_JOINT_E2E=1 to run real-server tests")
    with _running_server(
        checkpoint_dir,
        log_dir,
        "joint",
        group_size=1,
        moe_cache_size=2 * NUM_EXPERTS,
    ) as server:
        assert server.client is not None
        server.client.complete(*CASES["single_chunk"])
        assert server.process is not None and server.process.poll() is None


@pytest.mark.parametrize("group_size", [1, 3])
def test_single_multi_chunk_and_tail_groups_match_references(
    checkpoint_dir: Path,
    log_dir: Path,
    reference_outputs: dict[str, str],
    group_size: int,
) -> None:
    """Detects wrong chunk/group advancement; action is to reject joint output.

    Group 1 exercises per-layer progress. Group 3 over five decoder layers
    forces a non-divisible two-layer tail. The 160-token prompt forces at
    least five public 32-token prefill chunks.
    """

    assert NUM_LAYERS % group_size != 0 if group_size == 3 else True
    cache_slots = (group_size + 1) * NUM_EXPERTS
    with _running_server(
        checkpoint_dir,
        log_dir,
        "joint",
        group_size=group_size,
        moe_cache_size=cache_slots,
    ) as server:
        assert server.client is not None
        for name in ("single_chunk", "multi_chunk"):
            prompt, max_tokens = CASES[name]
            assert server.client.complete(prompt, max_tokens) == reference_outputs[name]


def test_group_2_single_and_multi_chunk_match_references(
    joint_group_2_server: PublicServer,
    reference_outputs: dict[str, str],
) -> None:
    """Detects a G=2 or one-layer tail error; action is to reject joint output."""

    assert NUM_LAYERS % 2 == 1
    assert joint_group_2_server.client is not None
    for name in ("single_chunk", "multi_chunk"):
        prompt, max_tokens = CASES[name]
        assert joint_group_2_server.client.complete(prompt, max_tokens) == reference_outputs[name]


def test_requested_group_is_reduced_to_publicly_logged_effective_group(
    checkpoint_dir: Path,
    log_dir: Path,
    reference_outputs: dict[str, str],
) -> None:
    """Detects rejection or silent misreporting when requested G cannot fit.

    With 16 slots, this eight-expert checkpoint has one complete layer for the
    decode cache and one for grouped prefill. Requested G=3 must therefore run
    as effective G=1, and the public startup log must make both values visible.
    Action: reject startup behavior that fails, chooses another effective G,
    hides the adjustment, or changes generated output.
    """

    with _running_server(
        checkpoint_dir,
        log_dir,
        "joint",
        group_size=3,
        moe_cache_size=2 * NUM_EXPERTS,
    ) as server:
        assert server.client is not None
        prompt, max_tokens = CASES["multi_chunk"]
        assert server.client.complete(prompt, max_tokens) == reference_outputs["multi_chunk"]
        relevant_lines = [
            line.lower()
            for line in server.logs().splitlines()
            if "group" in line.lower()
            and ("effective" in line.lower() or "reduc" in line.lower() or "clamp" in line.lower())
        ]
        assert relevant_lines, "startup log did not disclose the reduced effective group"
        assert any("3" in line and "1" in line for line in relevant_lines), relevant_lines


def test_multiple_decode_streams_match_reference(
    joint_group_2_server: PublicServer,
    reference_outputs: dict[str, str],
) -> None:
    """Detects cross-request decode corruption; action is to reject scheduling."""

    assert joint_group_2_server.client is not None
    names = ("decode_a", "decode_b", "decode_c")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(names)) as pool:
        futures = {
            name: pool.submit(
                joint_group_2_server.client.complete,
                CASES[name][0],
                CASES[name][1],
                stream=True,
            )
            for name in names
        }
        observed = {name: future.result(timeout=REQUEST_TIMEOUT) for name, future in futures.items()}
    assert observed == {name: reference_outputs[name] for name in names}


def test_decode_already_running_when_prefill_arrives(
    joint_group_2_server: PublicServer,
    reference_outputs: dict[str, str],
) -> None:
    """Detects a stalled/corrupted active decode when a long prefill arrives.

    Action: reject the joint scheduler if either output changes or no decode
    text is observed after the prefill is submitted. This is the black-box
    consequence of combining active decode with the arriving prefill work;
    it intentionally does not inspect or count internal forward calls.
    """

    assert joint_group_2_server.client is not None
    client = joint_group_2_server.client
    first_decode_text = threading.Event()
    decode_timestamps: list[float] = []

    def observe_decode(_piece: str, timestamp: float) -> None:
        decode_timestamps.append(timestamp)
        first_decode_text.set()

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        decode_future = pool.submit(
            client.stream_with_observer,
            CASES["decode_active"][0],
            CASES["decode_active"][1],
            observe_decode,
        )
        assert first_decode_text.wait(timeout=REQUEST_TIMEOUT), "decode never began streaming"
        assert not decode_future.done(), "decode workload completed before prefill could arrive"
        prefill_submitted_at = time.perf_counter()
        prefill_output = client.complete(*CASES["prefill_a"], stream=True)
        decode_output = decode_future.result(timeout=REQUEST_TIMEOUT)

    assert any(timestamp >= prefill_submitted_at for timestamp in decode_timestamps)
    assert decode_output == reference_outputs["decode_active"]
    assert prefill_output == reference_outputs["prefill_a"]


def test_prefill_only_and_decode_only_phases(
    joint_group_2_server: PublicServer,
    reference_outputs: dict[str, str],
) -> None:
    """Detects dependence on both work types being present; action is rejection.

    One-token long-prompt requests have no recurrent decode phase after their
    prefill wave. A lone short-prompt, long-output request spends nearly all of
    its lifetime in decode with no arriving prefill.
    """

    assert joint_group_2_server.client is not None
    client = joint_group_2_server.client
    prefill_names = ("prefill_a", "prefill_b")
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        futures = {
            name: pool.submit(client.complete, *CASES[name], stream=True)
            for name in prefill_names
        }
        prefill_observed = {
            name: future.result(timeout=REQUEST_TIMEOUT) for name, future in futures.items()
        }
    assert prefill_observed == {name: reference_outputs[name] for name in prefill_names}

    decode_observed = client.complete(*CASES["decode_active"], stream=True)
    assert decode_observed == reference_outputs["decode_active"]


def test_abort_releases_capacity_for_next_requests(
    checkpoint_dir: Path,
    log_dir: Path,
    reference_outputs: dict[str, str],
) -> None:
    """Detects leaked request/KV capacity after SSE disconnect.

    Action: reject the scheduler if either of two full-size follow-up requests
    cannot finish or changes output after an aborted request reserved most of a
    deliberately small 256-token public KV capacity.
    """

    with _running_server(
        checkpoint_dir,
        log_dir,
        "joint",
        group_size=2,
        moe_cache_size=3 * NUM_EXPERTS,
        num_tokens=256,
    ) as server:
        assert server.client is not None
        server.client.stream_with_observer(
            *CASES["abort_capacity"],
            lambda _piece, _timestamp: None,
            abort_after_first_text=True,
        )
        time.sleep(1.0)
        for _ in range(2):
            observed = server.client.complete(*CASES["abort_capacity"], stream=True)
            assert observed == reference_outputs["abort_capacity"]


@pytest.mark.parametrize(
    ("group_size", "cache_slots", "expected_word"),
    [(0, 16, "group"), (3, 2 * NUM_EXPERTS - 1, "cache")],
)
def test_invalid_group_or_insufficient_cache_is_rejected_at_startup(
    checkpoint_dir: Path,
    tmp_path: Path,
    group_size: int,
    cache_slots: int,
    expected_word: str,
) -> None:
    """Detects unsafe startup acceptance; action is to reject the configuration.

    G=0 is not a layer group. Any positive requested G may be reduced, but the
    checkpoint still needs one complete eight-expert layer for decode and one
    for grouped prefill, so 15 total slots is unambiguously too small.
    """

    port = _free_port()
    command = _serve_command(
        checkpoint_dir,
        port,
        "joint",
        group_size=group_size,
        moe_cache_size=cache_slots,
        disable_prefill_overlap=True,
    )
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_public_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"invalid server configuration did not reject within 60s: {error}")

    output = result.stdout.lower()
    (tmp_path / f"startup-rejection-g{group_size}.log").write_text(
        result.stdout, encoding="utf-8"
    )
    assert result.returncode != 0, "invalid server configuration reached serving state"
    assert "unrecognized arguments" not in output and "invalid choice" not in output
    assert expected_word in output or "expert" in output or "resident" in output


def test_joint_rejects_non_triton_attention_backend(
    checkpoint_dir: Path,
    tmp_path: Path,
) -> None:
    """Detects unsupported attention execution reaching joint startup.

    Action: reject the build unless a public error explicitly says that joint
    requires the triton attention backend. This prevents a missing optional
    flashinfer installation from being mistaken for the required contract
    rejection.
    """

    port = _free_port()
    command = _serve_command(
        checkpoint_dir,
        port,
        "joint",
        group_size=2,
        moe_cache_size=3 * NUM_EXPERTS,
        attention_backend="flashinfer",
    )
    try:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=_public_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        pytest.fail(f"unsupported joint attention backend did not reject within 60s: {error}")

    output = result.stdout.lower()
    (tmp_path / "startup-rejection-attention-flashinfer.log").write_text(
        result.stdout, encoding="utf-8"
    )
    assert result.returncode != 0, "joint accepted a non-triton attention backend"
    assert "unrecognized arguments" not in output and "invalid choice" not in output
    assert "joint" in output and "attention" in output and "triton" in output


def _timed_policy_workload(
    client: PublicCompletionClient,
    decode_prompt: str,
    prefill_prompt: str,
) -> tuple[dict[str, float | int | bool], str, str]:
    first_decode_text = threading.Event()
    decode_timestamps: list[float] = []

    def note_first(_piece: str, timestamp: float) -> None:
        decode_timestamps.append(timestamp)
        first_decode_text.set()

    start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        def run_decode() -> tuple[str, float]:
            output = client.stream_with_observer(decode_prompt, 192, note_first)
            return output, time.perf_counter()

        decode_future = pool.submit(run_decode)
        assert first_decode_text.wait(timeout=REQUEST_TIMEOUT)
        decode_active_at_prefill_submission = not decode_future.done()
        decode_text_events_before_prefill_submission = len(decode_timestamps)
        prefill_submitted_at = time.perf_counter()
        prefill_output = client.complete(prefill_prompt, 1, stream=True)
        prefill_completed_at = time.perf_counter()
        decode_output, decode_completed_at = decode_future.result(timeout=REQUEST_TIMEOUT)

    assert decode_timestamps
    completed_at = max(prefill_completed_at, decode_completed_at)
    decode_token_count = len(decode_output.split())
    prefill_output_token_count = len(prefill_output.split())
    decode_intervals = [
        later - earlier
        for earlier, later in zip(decode_timestamps, decode_timestamps[1:])
    ]
    wall_seconds = completed_at - start
    decode_latency_seconds = decode_completed_at - start
    decode_active_seconds = decode_completed_at - decode_timestamps[0]
    metrics: dict[str, float | int | bool] = {
        "wall_seconds": wall_seconds,
        "prefill_latency_seconds": prefill_completed_at - prefill_submitted_at,
        "decode_time_to_first_text_seconds": decode_timestamps[0] - start,
        "decode_latency_seconds": decode_latency_seconds,
        "decode_active_seconds": decode_active_seconds,
        "decode_max_inter_text_gap_seconds": max(decode_intervals, default=0.0),
        "decode_output_tokens": decode_token_count,
        "prefill_output_tokens": prefill_output_token_count,
        "decode_output_tokens_per_second": decode_token_count / decode_latency_seconds,
        "prefill_input_tokens_per_second": 160 / (prefill_completed_at - prefill_submitted_at),
        "combined_output_tokens_per_wall_second": (
            decode_token_count + prefill_output_token_count
        )
        / wall_seconds,
        "decode_active_at_prefill_submission": decode_active_at_prefill_submission,
        "decode_text_events_before_prefill_submission": decode_text_events_before_prefill_submission,
        "decode_progressed_after_prefill_submission": any(
            timestamp >= prefill_submitted_at for timestamp in decode_timestamps
        ),
    }
    return metrics, decode_output, prefill_output


def _public_log_metrics(log_text: str) -> dict[str, object]:
    wave_pattern = re.compile(
        r"chunks=(\d+),\s*groups=(\d+),\s*effective_group_size=(\d+),\s*"
        r"prefill_layer_prepares=(\d+),\s*prefill_h2d_bytes=(\d+)",
        re.IGNORECASE,
    )
    wave_records = [
        {
            "chunks": int(match.group(1)),
            "groups": int(match.group(2)),
            "effective_group_size": int(match.group(3)),
            "prefill_layer_prepares": int(match.group(4)),
            "prefill_h2d_bytes": int(match.group(5)),
        }
        for match in wave_pattern.finditer(log_text)
    ]
    lines = log_text.splitlines()
    cached_tokens = [
        int(value) for value in re.findall(r"#cached-token:\s*(\d+)", log_text)
    ]
    return {
        "wave_records": wave_records,
        "wave_count": len(wave_records),
        "chunks_sum": sum(record["chunks"] for record in wave_records),
        "prefill_layer_prepares_sum": sum(
            record["prefill_layer_prepares"] for record in wave_records
        ),
        "prefill_h2d_bytes_sum": sum(record["prefill_h2d_bytes"] for record in wave_records),
        "prepare_log_line_count": sum("prepare" in line.lower() for line in lines),
        "h2d_log_line_count": sum("h2d" in line.lower() for line in lines),
        "cached_token_values": cached_tokens,
        "cached_token_log_observed": bool(cached_tokens),
        "all_reported_cached_tokens_zero": (
            all(value == 0 for value in cached_tokens) if cached_tokens else None
        ),
    }


def _public_startup_metrics(log_text: str) -> dict[str, object]:
    residency = re.search(
        r"requested_group_size=(\d+),\s*effective_group_size=(\d+),\s*"
        r"prefill_slots=(\d+),\s*decode_slots=(\d+)",
        log_text,
    )
    configured_wave = re.search(r"prefill_wave_max_chunks=(\d+)", log_text)
    configured_chunk = re.search(r"max_extend_tokens=(\d+)", log_text)
    return {
        "requested_group_size": int(residency.group(1)) if residency else None,
        "effective_group_size": int(residency.group(2)) if residency else None,
        "prefill_slots": int(residency.group(3)) if residency else None,
        "decode_slots": int(residency.group(4)) if residency else None,
        "prefill_wave_max_chunks": int(configured_wave.group(1)) if configured_wave else None,
        "prefill_chunk_tokens": int(configured_chunk.group(1)) if configured_chunk else None,
    }


@pytest.mark.skipif(
    not RUN_E2E,
    reason="set FT_RUN_JOINT_E2E=1 to run real-server tests",
)
def test_end_to_end_policy_medians(
    checkpoint_dir: Path,
    log_dir: Path,
    tmp_path: Path,
) -> None:
    """Measure a same-configuration five-mode public policy A/B.

    Detects: an unmeasurable workload, accidental prefix-cache hits, or a mode
    that cannot keep decode active when prefill arrives. Action: reject invalid
    measurements. Output mismatch is recorded per sample and does not stop the
    timing run, because the focused correctness test already owns that verdict.
    No speed pass/fail ratio is invented without a public numerical boundary.
    """

    repetitions = max(3, int(os.environ.get("FT_TIMING_REPETITIONS", "3")))
    timing_cases = [
        {
            "sample": index,
            "decode_prompt": _tokens(50 + index * 47, 16),
            "prefill_prompt": _tokens(250 + index * 61, 160),
        }
        for index in range(repetitions)
    ]
    references: dict[int, tuple[str, str]] = {}
    with _running_server(
        checkpoint_dir,
        log_dir,
        "legacy",
        moe_cache_size=24,
        attention_backend="triton",
        dtype="float16",
    ) as server:
        assert server.client is not None
        for case in timing_cases:
            references[case["sample"]] = (
                server.client.complete(case["decode_prompt"], 192, stream=True),
                server.client.complete(case["prefill_prompt"], 1, stream=True),
            )

    modes = (
        ("legacy", "legacy", None, 2),
        ("mixed", "mixed", None, 2),
        ("layered_g2", "layered", 2, 2),
        ("joint_g2_wave1", "joint", 2, 1),
        ("joint_g2_wave2", "joint", 2, 2),
    )
    samples: dict[str, list[dict[str, object]]] = {}
    mode_summaries: dict[str, dict[str, object]] = {}
    for mode_name, policy, group_size, wave_chunks in modes:
        with _running_server(
            checkpoint_dir,
            log_dir,
            policy,
            group_size=group_size,
            moe_cache_size=24,
            attention_backend="triton",
            prefill_wave_max_chunks=wave_chunks,
            dtype="float16",
        ) as server:
            assert server.client is not None
            startup = _public_startup_metrics(server.logs())
            server.client.complete(_tokens(480, 8), 2)
            mode_samples: list[dict[str, object]] = []
            for case in timing_cases:
                before_log = server.logs()
                metrics, decode_output, prefill_output = _timed_policy_workload(
                    server.client,
                    case["decode_prompt"],
                    case["prefill_prompt"],
                )
                time.sleep(0.1)
                sample_log = server.logs()[len(before_log) :]
                public_log = _public_log_metrics(sample_log)
                assert public_log["cached_token_log_observed"], public_log
                assert public_log["all_reported_cached_tokens_zero"] is True, public_log
                assert metrics["decode_active_at_prefill_submission"]
                assert metrics["decode_progressed_after_prefill_submission"]
                decode_reference, prefill_reference = references[case["sample"]]
                sample = {
                    "sample": case["sample"],
                    **metrics,
                    "decode_output_match": decode_output == decode_reference,
                    "prefill_output_match": prefill_output == prefill_reference,
                    "public_log": public_log,
                }
                for value in metrics.values():
                    if isinstance(value, float):
                        assert math.isfinite(value) and value >= 0
                mode_samples.append(sample)
            samples[mode_name] = mode_samples

            median_fields = (
                "wall_seconds",
                "prefill_latency_seconds",
                "decode_time_to_first_text_seconds",
                "decode_text_events_before_prefill_submission",
                "decode_latency_seconds",
                "decode_active_seconds",
                "decode_max_inter_text_gap_seconds",
                "decode_output_tokens_per_second",
                "prefill_input_tokens_per_second",
                "combined_output_tokens_per_wall_second",
            )
            mode_summaries[mode_name] = {
                "configured_group_size": group_size,
                "configured_wave_max_chunks": wave_chunks,
                "moe_cache_slots": 24,
                "attention_backend": "triton",
                "dtype": "float16",
                "cuda_graph_max_bs": 0,
                "public_startup": startup,
                "sample_count": len(mode_samples),
                "decode_output_match_count": sum(
                    bool(sample["decode_output_match"]) for sample in mode_samples
                ),
                "prefill_output_match_count": sum(
                    bool(sample["prefill_output_match"]) for sample in mode_samples
                ),
                "medians": {
                    field: statistics.median(float(sample[field]) for sample in mode_samples)
                    for field in median_fields
                },
                "public_log_totals": {
                    "wave_count": sum(
                        int(sample["public_log"]["wave_count"]) for sample in mode_samples
                    ),
                    "chunks_sum": sum(
                        int(sample["public_log"]["chunks_sum"]) for sample in mode_samples
                    ),
                    "prefill_layer_prepares_sum": sum(
                        int(sample["public_log"]["prefill_layer_prepares_sum"])
                        for sample in mode_samples
                    ),
                    "prefill_h2d_bytes_sum": sum(
                        int(sample["public_log"]["prefill_h2d_bytes_sum"])
                        for sample in mode_samples
                    ),
                    "prepare_log_line_count": sum(
                        int(sample["public_log"]["prepare_log_line_count"])
                        for sample in mode_samples
                    ),
                    "h2d_log_line_count": sum(
                        int(sample["public_log"]["h2d_log_line_count"])
                        for sample in mode_samples
                    ),
                },
            }

    paired_ratios: dict[str, dict[str, object]] = {}
    ratio_pairs = (
        ("joint_wave2_over_layered", "joint_g2_wave2", "layered_g2"),
        ("joint_wave2_over_mixed", "joint_g2_wave2", "mixed"),
        ("joint_wave2_over_joint_wave1", "joint_g2_wave2", "joint_g2_wave1"),
    )
    ratio_fields = (
        "wall_seconds",
        "prefill_latency_seconds",
        "decode_max_inter_text_gap_seconds",
    )
    for pair_name, numerator_mode, denominator_mode in ratio_pairs:
        paired_ratios[pair_name] = {}
        for field in ratio_fields:
            values: list[dict[str, float | int]] = []
            for numerator, denominator in zip(
                samples[numerator_mode], samples[denominator_mode], strict=True
            ):
                assert numerator["sample"] == denominator["sample"]
                denominator_value = float(denominator[field])
                assert denominator_value > 0, (pair_name, field, denominator)
                values.append(
                    {
                        "sample": int(numerator["sample"]),
                        "ratio": float(numerator[field]) / denominator_value,
                    }
                )
            ratios = [float(value["ratio"]) for value in values]
            paired_ratios[pair_name][field] = {
                "samples": values,
                "median": statistics.median(ratios),
                "min": min(ratios),
                "max": max(ratios),
            }

    report = {
        "configuration": {
            "repetitions": repetitions,
            "decode_prompt_tokens": 16,
            "decode_output_tokens_requested": 192,
            "prefill_prompt_tokens": 160,
            "prefill_output_tokens_requested": 1,
            "prefill_chunk_tokens": PREFILL_CHUNK_SIZE,
            "unique_prompt_per_sample": True,
            "reference_policy": "legacy",
        },
        "mode_summaries": mode_summaries,
        "paired_ratios": paired_ratios,
        "samples": samples,
    }
    report_path = tmp_path / "joint-policy-timings.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    assert all(len(values) >= 3 for values in samples.values())
