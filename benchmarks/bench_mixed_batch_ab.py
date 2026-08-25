#!/usr/bin/env python3
"""Black-box legacy-versus-mixed serving benchmark for FreeToken.

The policies use separate server processes but identical prompts and the same
first-token arrival barrier.  Each server receives one unmeasured warmup before
three measured repetitions.  Dense mode is the default and launches no MoE or
NoWAG expert options; ``resident-moe`` selects a fully resident fused MoE, and
``nowag`` selects the expert-offload setup.
"""

from __future__ import annotations

import argparse
import concurrent.futures
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import re
import socket
import statistics
import subprocess
import threading
import time
import urllib.error
import urllib.request


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PYTHON = Path("/home/nengneng/miniconda3/envs/freetoken-dev/bin/python")
DEFAULT_DENSE_MODEL = Path(
    "/data1/lmcache_kv/hf-cache/models--Qwen--Qwen3-0.6B/snapshots/"
    "c1899de289a04d12100db370d81485cdf75e47ca"
)
DEFAULT_RESIDENT_MOE_MODEL = Path(
    "/data1/lmcache_kv/hf-cache/models--openai--gpt-oss-20b/snapshots/"
    "6cee5e81ee83917806bbde320786a8fb61efebee"
)
DEFAULT_NOWAG_MODEL = Path("/data1/lmcache_kv/models/Qwen3.6-35B-A3B")
DEFAULT_NOWAG_EXPERT = Path(
    "/data1/lmcache_kv/nowag_qwen36_experiment/quantized/"
    "qwen36_expert_only_global_d6b12_wikitext2_train_seed0_128x2048_kpp5"
)
DEFAULT_OUTPUT = Path("/data1/lmcache_kv/experiments/freetoken_mixed_ab")
SERVED_MODEL = "mixed-batch-benchmark"
POLICIES = ("legacy", "mixed")
ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
BATCH_LINE = re.compile(r"\b(Prefill|Mixed|Decode) batch,")
INTEGER_FIELD = re.compile(r"#([a-z-]+):\s*(\d+)")


@dataclass(frozen=True)
class Configuration:
    name: str
    max_prefill_length: int
    decode_requests: int
    prefill_requests: int = 2
    decode_tokens: int = 128
    prefill_tokens: int = 16
    max_running_requests: int = 8


CONFIGURATIONS = (
    Configuration("baseline_p256_d4_p2", 256, 4),
    Configuration("candidate_p64_d6_p2", 64, 6),
)
FALLBACK = Configuration("fallback_p32_d6_p2", 32, 6)
HIGH_CONCURRENCY = Configuration(
    "high_concurrency_p256_d12_p2",
    256,
    12,
    max_running_requests=16,
)
CONFIGURATION_BY_NAME = {
    configuration.name: configuration
    for configuration in (*CONFIGURATIONS, FALLBACK, HIGH_CONCURRENCY)
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--model-mode",
        choices=("dense", "resident-moe", "nowag"),
        default="dense",
        help=(
            "dense passes no expert options; resident-moe uses the fused backend "
            "with all experts resident; nowag enables the expert-offload setup"
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        help=(
            "model path; defaults to Qwen3-0.6B for dense, gpt-oss-20b for "
            "resident-moe, or Qwen3.6-35B-A3B for nowag"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    nowag = parser.add_argument_group("NoWAG mode")
    nowag.add_argument(
        "--expert",
        type=Path,
        help="expert artifact path; only used by --model-mode nowag",
    )
    nowag.add_argument("--moe-cache-size", type=int, default=10240)
    nowag.add_argument("--num-tokens", type=int, default=269000)
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGURATION_BY_NAME),
        help="run exactly these configurations; the default runs the baseline sweep",
    )
    args = parser.parse_args()
    if args.model is None:
        default_models = {
            "dense": DEFAULT_DENSE_MODEL,
            "resident-moe": DEFAULT_RESIDENT_MOE_MODEL,
            "nowag": DEFAULT_NOWAG_MODEL,
        }
        args.model = default_models[args.model_mode]
    if args.model_mode == "nowag" and args.expert is None:
        args.expert = DEFAULT_NOWAG_EXPERT
    return args


def find_free_port_pair():
    for _ in range(100):
        first = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        second = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            first.bind(("127.0.0.1", 0))
            port = first.getsockname()[1]
            if port == 65535:
                continue
            second.bind(("127.0.0.1", port + 1))
            return port
        except OSError:
            continue
        finally:
            first.close()
            second.close()
    raise RuntimeError("could not find two consecutive free localhost ports")


def read_log_since(path, offset):
    with path.open("rb") as log_file:
        log_file.seek(offset)
        return log_file.read().decode("utf-8", errors="replace")


def log_tail(path, line_count=100):
    if not path.exists():
        return "<log file was not created>"
    return "\n".join(
        path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]
    )


def wait_until_serving(process, base_url, log_path, timeout):
    deadline = time.monotonic() + timeout
    status_url = f"{base_url}/v1/cache/status"
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                f"server exited with {process.returncode} before ready\n{log_tail(log_path)}"
            )
        try:
            with urllib.request.urlopen(status_url, timeout=2) as response:
                status = json.load(response)
                if response.getcode() == 200 and status.get("state") == "serving":
                    return
        except (OSError, urllib.error.URLError, json.JSONDecodeError):
            pass
        time.sleep(0.5)
    raise TimeoutError(f"server did not become ready\n{log_tail(log_path)}")


def wait_for_batch_logs_to_settle(process, log_path, offset, timeout=10.0):
    """Wait after timing so final status lines belong to the current repetition."""
    deadline = time.monotonic() + timeout
    previous_size = -1
    stable_polls = 0
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}\n{log_tail(log_path)}")
        current_size = log_path.stat().st_size
        text = read_log_since(log_path, offset)
        has_idle_line = any(
            " batch," in line
            and "#running-req: 0" in line
            and "#queue-req: 0" in line
            for line in text.splitlines()
        )
        stable_polls = stable_polls + 1 if current_size == previous_size else 0
        if has_idle_line and stable_polls >= 2:
            return
        previous_size = current_size
        time.sleep(0.1)


def stop_process(process):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def chat_request(prompt, max_tokens):
    return {
        "model": SERVED_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "top_p": 1.0,
        "top_k": -1,
        "ignore_eos": True,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


def decode_prompt(request_id):
    return (
        f"Mixed batching benchmark, decode request {request_id}. "
        "Output a numbered list starting at 1, with one concise item per line. "
        "Continue until stopped and add no introduction."
    )


def long_prefill_prompt(request_id):
    sentence = (
        "During the quiet afternoon, the village library recorded each returned "
        "book, arranged the shelves, and prepared a simple reading list for the next day."
    )
    context = " ".join([sentence] * 85)
    return (
        f"Mixed batching benchmark, prefill request {request_id}. "
        "Read the following ordinary repeated passage, then describe its main activity "
        f"in one short sentence.\n\n{context}"
    )


def stream_chat(base_url, payload, timeout, on_first_text=None):
    request = urllib.request.Request(
        f"{base_url}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Accept": "text/event-stream", "Content-Type": "application/json"},
        method="POST",
    )
    reasoning_parts = []
    content_parts = []
    usage = None
    saw_done = False
    notified = False

    def consume(data):
        nonlocal usage, saw_done, notified
        data = data.strip()
        if not data:
            return False
        if data == "[DONE]":
            saw_done = True
            return True
        event = json.loads(data)
        if event.get("usage") is not None:
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            return False
        delta = choices[0].get("delta") or {}
        reasoning = delta.get("reasoning_content") or ""
        content = delta.get("content") or ""
        if reasoning:
            reasoning_parts.append(reasoning)
        if content:
            content_parts.append(content)
        if not notified and (reasoning + content).strip():
            notified = True
            if on_first_text is not None:
                on_first_text()
        return False

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            http_status = response.getcode()
            data_lines = []
            for raw_line in response:
                line = raw_line.decode("utf-8").rstrip("\r\n")
                if not line:
                    if data_lines:
                        if consume("\n".join(data_lines)):
                            break
                        data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                if line.startswith("data:"):
                    value = line[5:]
                    data_lines.append(value[1:] if value.startswith(" ") else value)
            if data_lines and not saw_done:
                consume("\n".join(data_lines))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc

    output = "".join(reasoning_parts) + "".join(content_parts)
    if http_status != 200 or not saw_done or not output.strip():
        raise RuntimeError(
            f"incomplete stream: status={http_status}, done={saw_done}, output={output!r}"
        )
    return {
        "http_status": http_status,
        "saw_done": saw_done,
        "output": output,
        "usage": usage,
        "requested_completion_tokens": int(payload["max_tokens"]),
    }


def run_workload(base_url, configuration, timeout):
    decode_payloads = {
        f"A{index}": chat_request(decode_prompt(f"A{index}"), configuration.decode_tokens)
        for index in range(configuration.decode_requests)
    }
    prefill_payloads = {
        f"B{index}": chat_request(
            long_prefill_prompt(f"B{index}"), configuration.prefill_tokens
        )
        for index in range(configuration.prefill_requests)
    }
    started = time.monotonic()
    barrier_time = None
    first_text_count = 0
    lock = threading.Lock()
    prefill_futures = {}
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=configuration.decode_requests + configuration.prefill_requests
    ) as executor:

        def on_first_decode_text():
            nonlocal barrier_time, first_text_count
            with lock:
                first_text_count += 1
                if first_text_count == configuration.decode_requests:
                    barrier_time = time.monotonic()
                    for request_id, payload in prefill_payloads.items():
                        prefill_futures[request_id] = executor.submit(
                            stream_chat, base_url, payload, timeout
                        )

        decode_futures = {
            request_id: executor.submit(
                stream_chat, base_url, payload, timeout, on_first_decode_text
            )
            for request_id, payload in decode_payloads.items()
        }
        requests = {
            request_id: future.result(timeout=timeout)
            for request_id, future in decode_futures.items()
        }
        if first_text_count != configuration.decode_requests:
            raise RuntimeError("not all decode requests reached the first-token barrier")
        if len(prefill_futures) != configuration.prefill_requests:
            raise RuntimeError("prefill requests were not submitted at the barrier")
        requests.update(
            {
                request_id: future.result(timeout=timeout)
                for request_id, future in prefill_futures.items()
            }
        )
    elapsed = time.monotonic() - started
    requested_tokens = sum(
        request["requested_completion_tokens"] for request in requests.values()
    )
    usage_available = all(
        request["usage"] is not None
        and request["usage"].get("completion_tokens") is not None
        for request in requests.values()
    )
    actual_tokens = (
        sum(int(request["usage"]["completion_tokens"]) for request in requests.values())
        if usage_available
        else None
    )
    return {
        "elapsed_s": elapsed,
        "barrier_s": barrier_time - started,
        "requested_completion_tokens": requested_tokens,
        "actual_completion_tokens": actual_tokens,
        "fixed_completion_tokens_per_s": requested_tokens / elapsed,
        "actual_completion_tokens_per_s": (
            actual_tokens / elapsed if actual_tokens is not None else None
        ),
        "requests": requests,
    }


def parse_batch_logs(text):
    lines = []
    counts = {"Prefill": 0, "Mixed": 0, "Decode": 0}
    integer_totals = {
        "Prefill": {"new-token": 0},
        "Mixed": {"new-token": 0},
        "Decode": {},
    }
    for raw_line in text.splitlines():
        line = ANSI_ESCAPE.sub("", raw_line)
        match = BATCH_LINE.search(line)
        if match is None:
            continue
        kind = match.group(1)
        counts[kind] += 1
        fields = {name: int(value) for name, value in INTEGER_FIELD.findall(line)}
        if "new-token" in fields and kind in ("Prefill", "Mixed"):
            integer_totals[kind]["new-token"] += fields["new-token"]
        lines.append(line)
    return {
        "line_counts": counts,
        "unique_forwards": counts["Prefill"] + counts["Decode"],
        "pure_decode_forwards": counts["Decode"] - counts["Mixed"],
        "logged_new_token_totals": integer_totals,
        "raw_relevant_line_count": len(lines),
        "raw_relevant_head": lines[:5],
        "raw_relevant_tail": lines[-5:],
        "forward_count_note": (
            "With decode_log_interval=1, a Mixed forward emits both a Mixed and a "
            "Decode status line. Therefore unique forwards are Prefill + Decode, "
            "and pure decode forwards are Decode - Mixed."
        ),
        "row_count_note": (
            "The public lines expose prefill new-token counts and running-request "
            "state, but not the exact decode-row count inside a Mixed batch."
        ),
    }


def server_command(args, configuration, policy, port):
    command = [
        str(args.python),
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        str(args.model),
        "--served-model-name",
        SERVED_MODEL,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    if args.model_mode == "nowag":
        command.extend(
            [
                "--moe-backend",
                "offload",
                "--nowag-expert-path",
                str(args.expert),
                "--moe-cache-size",
                str(args.moe_cache_size),
                "--num-tokens",
                str(args.num_tokens),
            ]
        )
    elif args.model_mode == "resident-moe":
        command.extend(["--moe-backend", "fused"])
    command.extend(
        [
        "--max-running-requests",
        str(configuration.max_running_requests),
        "--max-prefill-length",
        str(configuration.max_prefill_length),
        "--decode-log-interval",
        "1",
        "--cache-type",
        "naive",
        "--batching-policy",
        policy,
        ]
    )
    return command


def run_policy(args, configuration, policy, run_directory):
    port = find_free_port_pair()
    base_url = f"http://127.0.0.1:{port}"
    log_path = run_directory / f"{configuration.name}_{policy}.log"
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "python"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["CUDA_VISIBLE_DEVICES"] = args.gpu
    command = server_command(args, configuration, policy, port)
    print(
        f"START config={configuration.name} policy={policy} "
        f"gpu={args.gpu} log={log_path}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_until_serving(process, base_url, log_path, args.timeout)
            warmup = stream_chat(
                base_url,
                chat_request("Reply with the single word warmup.", 4),
                args.timeout,
            )
            warmup_offset = log_path.stat().st_size
            wait_for_batch_logs_to_settle(process, log_path, warmup_offset)
            repetitions = []
            for repeat_id in range(args.repeats):
                offset = log_path.stat().st_size
                result = run_workload(base_url, configuration, args.timeout)
                wait_for_batch_logs_to_settle(process, log_path, offset)
                result["repeat_id"] = repeat_id
                result["log"] = parse_batch_logs(read_log_since(log_path, offset))
                repetitions.append(result)
                print(
                    f"DONE config={configuration.name} policy={policy} "
                    f"repeat={repeat_id} elapsed={result['elapsed_s']:.6f}s "
                    f"fixed_tps={result['fixed_completion_tokens_per_s']:.3f} "
                    f"actual_tps={result['actual_completion_tokens_per_s']}",
                    flush=True,
                )
        except Exception as exc:
            if process.poll() is not None:
                raise RuntimeError(
                    f"{policy} server exited with {process.returncode}\n{log_tail(log_path)}"
                ) from exc
            raise
        finally:
            stop_process(process)
    return {
        "command": command,
        "log_path": str(log_path),
        "warmup": warmup,
        "repetitions": repetitions,
    }


def first_difference(left, right):
    shared = min(len(left), len(right))
    index = next((i for i in range(shared) if left[i] != right[i]), shared)
    start = max(0, index - 24)
    end = index + 24
    return {
        "character_index": index,
        "legacy_snippet": left[start:end],
        "mixed_snippet": right[start:end],
    }


def summarize_configuration(configuration, policies):
    legacy = policies["legacy"]["repetitions"]
    mixed = policies["mixed"]["repetitions"]
    legacy_elapsed = [run["elapsed_s"] for run in legacy]
    mixed_elapsed = [run["elapsed_s"] for run in mixed]
    legacy_fixed_tps = [run["fixed_completion_tokens_per_s"] for run in legacy]
    mixed_fixed_tps = [run["fixed_completion_tokens_per_s"] for run in mixed]
    output_differences = []
    usage_equal = []
    paired = []
    for repeat_id, (legacy_run, mixed_run) in enumerate(zip(legacy, mixed)):
        paired.append(
            {
                "repeat_id": repeat_id,
                "legacy_elapsed_s": legacy_run["elapsed_s"],
                "mixed_elapsed_s": mixed_run["elapsed_s"],
                "mixed_speedup": legacy_run["elapsed_s"] / mixed_run["elapsed_s"],
                "mixed_faster": mixed_run["elapsed_s"] < legacy_run["elapsed_s"],
            }
        )
        for request_id in sorted(legacy_run["requests"]):
            legacy_request = legacy_run["requests"][request_id]
            mixed_request = mixed_run["requests"][request_id]
            usage_equal.append(legacy_request["usage"] == mixed_request["usage"])
            if legacy_request["output"] != mixed_request["output"]:
                output_differences.append(
                    {
                        "repeat_id": repeat_id,
                        "request_id": request_id,
                        **first_difference(
                            legacy_request["output"], mixed_request["output"]
                        ),
                    }
                )
    legacy_median = statistics.median(legacy_fixed_tps)
    mixed_median = statistics.median(mixed_fixed_tps)
    return {
        "configuration": configuration.__dict__,
        "legacy_elapsed_s": legacy_elapsed,
        "mixed_elapsed_s": mixed_elapsed,
        "legacy_fixed_completion_tokens_per_s": legacy_fixed_tps,
        "mixed_fixed_completion_tokens_per_s": mixed_fixed_tps,
        "legacy_median_elapsed_s": statistics.median(legacy_elapsed),
        "mixed_median_elapsed_s": statistics.median(mixed_elapsed),
        "legacy_median_fixed_completion_tokens_per_s": legacy_median,
        "mixed_median_fixed_completion_tokens_per_s": mixed_median,
        "median_speedup": mixed_median / legacy_median,
        "paired": paired,
        "paired_wins": sum(pair["mixed_faster"] for pair in paired),
        "stable_win": (
            mixed_median > legacy_median and all(pair["mixed_faster"] for pair in paired)
        ),
        "all_reported_usage_equal_across_policies": all(usage_equal),
        "output_difference_count": len(output_differences),
        "output_differences": output_differences,
    }


def write_results(path, results):
    path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")


def validate_inputs(args):
    if args.repeats != 3:
        raise ValueError("this experiment requires exactly three steady-state repeats")
    required_paths = [("python", args.python), ("model", args.model)]
    if args.model_mode == "nowag":
        required_paths.append(("expert", args.expert))
    for label, path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
    if args.gpu != "2":
        raise ValueError("this benchmark is assigned exclusively to physical GPU 2")


def main():
    args = parse_args()
    validate_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_directory = args.output_dir / timestamp
    run_directory.mkdir()
    result_path = run_directory / "results.json"
    results = {
        "started_at": datetime.now().astimezone().isoformat(),
        "gpu": args.gpu,
        "model_mode": args.model_mode,
        "model": str(args.model.resolve()),
        "expert": str(args.expert.resolve()) if args.expert is not None else None,
        "repeats": args.repeats,
        "run_directory": str(run_directory),
        "configs": {},
    }
    configurations = (
        tuple(CONFIGURATION_BY_NAME[name] for name in args.configs)
        if args.configs
        else CONFIGURATIONS
    )
    for configuration in configurations:
        policies = {}
        for policy in POLICIES:
            policies[policy] = run_policy(
                args, configuration, policy, run_directory
            )
            results["configs"].setdefault(configuration.name, {})[policy] = policies[
                policy
            ]
            write_results(result_path, results)
        summary = summarize_configuration(configuration, policies)
        results["configs"][configuration.name]["summary"] = summary
        write_results(result_path, results)
        print(
            f"SUMMARY config={configuration.name} "
            f"median_speedup={summary['median_speedup']:.6f} "
            f"paired_wins={summary['paired_wins']}/{args.repeats}",
            flush=True,
        )

    candidate = results["configs"].get("candidate_p64_d6_p2", {}).get("summary")
    if (
        args.configs is None
        and candidate is not None
        and candidate["median_speedup"] <= 1.0
    ):
        policies = {}
        for policy in POLICIES:
            policies[policy] = run_policy(args, FALLBACK, policy, run_directory)
            results["configs"].setdefault(FALLBACK.name, {})[policy] = policies[policy]
            write_results(result_path, results)
        results["configs"][FALLBACK.name]["summary"] = summarize_configuration(
            FALLBACK, policies
        )

    results["finished_at"] = datetime.now().astimezone().isoformat()
    write_results(result_path, results)
    print(f"RESULTS {result_path}", flush=True)


if __name__ == "__main__":
    main()
