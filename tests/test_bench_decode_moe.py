from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import signal
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "bench_decode_moe.py"
SPEC = importlib.util.spec_from_file_location("bench_decode_moe", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


def test_stop_server_waits_for_worker_after_frontend_exits() -> None:
    frontend = subprocess.Popen(
        [
            sys.executable,
            "-c",
            (
                "import subprocess, sys; "
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])"
            ),
        ],
        start_new_session=True,
    )
    process_group = frontend.pid
    try:
        frontend.wait(timeout=5)
        os.killpg(process_group, 0)

        benchmark.stop_server(frontend)

        with pytest.raises(ProcessLookupError):
            os.killpg(process_group, 0)
    finally:
        try:
            os.killpg(process_group, signal.SIGKILL)
        except ProcessLookupError:
            pass
