#!/usr/bin/env python3
"""Profile one p64 NoWAG legacy or mixed workload with Nsight Systems.

The server is launched inside an interactive Nsight Systems session, but
collection starts only after the server is ready and one complete p64 workload
has warmed it.  The measured interval contains exactly one further p64 workload
plus the short wait for its final scheduler log lines.

Run one policy at a time on the assigned physical GPU::

    python benchmarks/profile_nowag_mixed_nsys.py mixed \
        --gpu 2 --model-mode nowag

The request workload, server command, readiness checks, and batch-log parsing
are imported from ``bench_mixed_batch_ab.py`` rather than duplicated here.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from typing import Any

import bench_mixed_batch_ab as mixed_ab


CONFIGURATION = mixed_ab.CONFIGURATION_BY_NAME["candidate_p64_d6_p2"]
DEFAULT_OUTPUT = Path("/data1/lmcache_kv/experiments/freetoken_mixed_nsys")
STATS_REPORTS = (
    "cuda_gpu_mem_size_sum",
    "cuda_gpu_mem_time_sum",
    "cuda_gpu_kern_sum",
    "nvtx_gpu_proj_sum",
)
CONTROL_TIMEOUT_SECONDS = 120


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("policy", choices=mixed_ab.POLICIES)
    parser.add_argument("--gpu", required=True, choices=("2",))
    parser.add_argument("--model-mode", required=True, choices=("nowag",))
    parser.add_argument("--python", type=Path, default=mixed_ab.DEFAULT_PYTHON)
    parser.add_argument("--model", type=Path, default=mixed_ab.DEFAULT_NOWAG_MODEL)
    parser.add_argument("--expert", type=Path, default=mixed_ab.DEFAULT_NOWAG_EXPERT)
    parser.add_argument("--moe-cache-size", type=int, default=10240)
    parser.add_argument("--num-tokens", type=int, default=269000)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    if args.moe_cache_size <= 0:
        parser.error("--moe-cache-size must be positive")
    if args.num_tokens <= 0:
        parser.error("--num-tokens must be positive")
    return args


def validate_inputs(args: argparse.Namespace) -> None:
    if shutil.which("nsys") is None:
        raise FileNotFoundError("nsys is not available on PATH")
    for label, path in (
        ("python", args.python),
        ("model", args.model),
        ("expert", args.expert),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")


def launch_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "2"
    environment["PYTHONPATH"] = "python"
    environment["PYTHONUNBUFFERED"] = "1"
    return environment


def run_control_command(
    command: list[str], environment: dict[str, str]
) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        cwd=mixed_ab.PROJECT_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        timeout=CONTROL_TIMEOUT_SECONDS,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"command failed with exit code {completed.returncode}: {command!r}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return {
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }


def try_control_command(
    command: list[str], environment: dict[str, str]
) -> bool:
    try:
        run_control_command(command, environment)
        return True
    except Exception as exc:
        print(f"cleanup command failed: {exc}", file=sys.stderr, flush=True)
        return False


def terminate_launch_tree(process: subprocess.Popen[str] | None) -> None:
    """Terminate only the process group created for this launch."""
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    if process.poll() is not None:
        return
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=20)


def summarize_workload(workload: dict[str, Any]) -> dict[str, Any]:
    return {
        "elapsed_s": workload["elapsed_s"],
        "barrier_s": workload["barrier_s"],
        "requested_completion_tokens": workload["requested_completion_tokens"],
        "actual_completion_tokens": workload["actual_completion_tokens"],
        "fixed_completion_tokens_per_s": workload[
            "fixed_completion_tokens_per_s"
        ],
        "actual_completion_tokens_per_s": workload[
            "actual_completion_tokens_per_s"
        ],
        "usage": {
            request_id: request["usage"]
            for request_id, request in workload["requests"].items()
        },
    }


def write_json(path: Path, result: dict[str, Any]) -> None:
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def generate_stats(
    report_path: Path,
    profile_base: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    sqlite_path = profile_base.with_suffix(".sqlite")
    outputs = {}
    for report_name in STATS_REPORTS:
        output_path = Path(f"{profile_base}_{report_name}.csv")
        command = [
            "nsys",
            "stats",
            "--report",
            report_name,
            "--format",
            "csv",
            "--output",
            str(profile_base),
            "--force-overwrite=true",
            "--sqlite",
            str(sqlite_path),
            str(report_path),
        ]
        completed = subprocess.run(
            command,
            cwd=mixed_ab.PROJECT_ROOT,
            env=environment,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"nsys stats failed for {report_name} with exit code "
                f"{completed.returncode}:\nstdout:\n{completed.stdout}\n"
                f"stderr:\n{completed.stderr}"
            )
        if not output_path.is_file():
            raise FileNotFoundError(
                f"nsys stats did not create the expected CSV: {output_path}"
            )
        outputs[report_name] = {
            "path": str(output_path),
            "command": command,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
    return {
        "sqlite": str(sqlite_path),
        "reports": outputs,
    }


def profile(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    started_at = datetime.now().astimezone().isoformat()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_directory = args.output_dir / f"{timestamp}_{args.policy}"
    run_directory.mkdir(parents=True)
    result_path = run_directory / "profile.json"
    log_path = run_directory / "server.log"
    profile_base = (run_directory / args.policy).resolve()
    report_path = Path(f"{profile_base}.nsys-rep")

    session_name = f"freetoken_{args.policy}_{os.getpid()}_{time.time_ns()}"
    port = mixed_ab.find_free_port_pair()
    base_url = f"http://127.0.0.1:{port}"
    server_command = mixed_ab.server_command(
        args, CONFIGURATION, args.policy, port
    )
    launch_command = [
        "nsys",
        "launch",
        f"--session-new={session_name}",
        "--trace=cuda,nvtx",
        "--wait=all",
        *server_command,
    ]
    start_command = [
        "nsys",
        "start",
        f"--session={session_name}",
        "--sample=none",
        "--cpuctxsw=none",
        f"--output={profile_base}",
        "--force-overwrite=true",
    ]
    stop_command = ["nsys", "stop", f"--session={session_name}"]
    cancel_command = ["nsys", "cancel", f"--session={session_name}"]
    environment = launch_environment()

    launch_process: subprocess.Popen[str] | None = None
    session_created = False
    collection_active = False
    collection_stopped = False
    warmup: dict[str, Any] | None = None
    measurement: dict[str, Any] | None = None
    batch_text: str | None = None
    control = {}

    print(
        f"START policy={args.policy} config={CONFIGURATION.name} "
        f"gpu=2 output={run_directory}",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            launch_process = subprocess.Popen(
                launch_command,
                cwd=mixed_ab.PROJECT_ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
            session_created = True
            mixed_ab.wait_until_serving(
                launch_process, base_url, log_path, args.timeout
            )

            warmup_offset = log_path.stat().st_size
            warmup = mixed_ab.run_workload(base_url, CONFIGURATION, args.timeout)
            mixed_ab.wait_for_batch_logs_to_settle(
                launch_process, log_path, warmup_offset
            )

            control["start"] = run_control_command(start_command, environment)
            collection_active = True
            measurement_offset = log_path.stat().st_size
            measurement = mixed_ab.run_workload(
                base_url, CONFIGURATION, args.timeout
            )
            mixed_ab.wait_for_batch_logs_to_settle(
                launch_process, log_path, measurement_offset
            )

            # End collection before parsing or serializing the measured result.
            control["stop"] = run_control_command(stop_command, environment)
            collection_active = False
            collection_stopped = True
            batch_text = mixed_ab.read_log_since(log_path, measurement_offset)
        finally:
            if session_created and not collection_stopped:
                if collection_active:
                    stopped = try_control_command(stop_command, environment)
                    if not stopped:
                        try_control_command(cancel_command, environment)
                else:
                    try_control_command(cancel_command, environment)
            terminate_launch_tree(launch_process)

    if warmup is None or measurement is None or batch_text is None:
        raise RuntimeError("profiling ended before the measured workload completed")
    if not report_path.is_file():
        raise FileNotFoundError(f"Nsight Systems report was not generated: {report_path}")

    batch_report = mixed_ab.parse_batch_logs(batch_text)
    result = {
        "started_at": started_at,
        "policy": args.policy,
        "gpu": "2",
        "model_mode": "nowag",
        "model": str(args.model.resolve()),
        "expert": str(args.expert.resolve()),
        "configuration": {
            "name": CONFIGURATION.name,
            "max_prefill_length": CONFIGURATION.max_prefill_length,
            "decode_requests": CONFIGURATION.decode_requests,
            "prefill_requests": CONFIGURATION.prefill_requests,
            "decode_tokens": CONFIGURATION.decode_tokens,
            "prefill_tokens": CONFIGURATION.prefill_tokens,
            "max_running_requests": CONFIGURATION.max_running_requests,
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": environment["CUDA_VISIBLE_DEVICES"],
            "PYTHONPATH": environment["PYTHONPATH"],
        },
        "session": session_name,
        "commands": {
            "server": server_command,
            "launch": launch_command,
            **control,
        },
        "warmup": summarize_workload(warmup),
        "measurement": summarize_workload(measurement),
        "batch_line_counts": batch_report["line_counts"],
        "batch_log_report": batch_report,
        "artifacts": {
            "result": str(result_path.resolve()),
            "report": str(report_path),
            "server_log": str(log_path.resolve()),
        },
        "stats": None,
    }
    write_json(result_path, result)
    try:
        result["stats"] = generate_stats(
            report_path, profile_base, environment
        )
    except Exception as exc:
        result["stats_error"] = str(exc)
        write_json(result_path, result)
        raise
    result["finished_at"] = datetime.now().astimezone().isoformat()
    write_json(result_path, result)
    return result_path, result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_inputs(args)
    result_path, result = profile(args)
    print(
        f"DONE policy={args.policy} elapsed={result['measurement']['elapsed_s']:.6f}s "
        f"report={result['artifacts']['report']} result={result_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
