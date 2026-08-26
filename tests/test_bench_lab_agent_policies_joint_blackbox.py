"""Black-box contract tests for joint policy benchmark expansion and summaries."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_CLI = PROJECT_ROOT / "benchmarks" / "bench_lab_agent_policies.py"


def _gpu_free_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "HIP_VISIBLE_DEVICES": "",
            "ROCR_VISIBLE_DEVICES": "",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "WANDB_MODE": "offline",
        }
    )
    return env


def _run_cli(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(BENCHMARK_CLI), *arguments],
        cwd=PROJECT_ROOT,
        env=_gpu_free_environment(),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )


def _option_value(argv: list[str], option: str) -> str:
    if option in argv:
        option_index = argv.index(option)
        assert option_index + 1 < len(argv), f"{option} has no value in {argv!r}"
        return argv[option_index + 1]

    option_prefix = f"{option}="
    matches = [item.removeprefix(option_prefix) for item in argv if item.startswith(option_prefix)]
    assert len(matches) == 1, f"expected one {option} in {argv!r}"
    return matches[0]


@pytest.mark.parametrize(
    ("joint_arguments", "case_name"),
    [
        (
            ["--joint-groups", "1,2", "--joint-waves", "1,3"],
            "comma-separated",
        ),
        (
            ["--joint-groups", "1", "2", "--joint-waves", "1", "3"],
            "space-separated",
        ),
    ],
    ids=lambda value: value if isinstance(value, str) else None,
)
def test_dry_run_expands_joint_modes_in_stable_order(
    joint_arguments: list[str], case_name: str
) -> None:
    del case_name
    result = _run_cli(
        "--dry-run",
        "--ft-executable",
        "/bin/true",
        "--modes",
        "legacy",
        "mixed",
        *joint_arguments,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    commands = payload["commands"]
    assert [command["mode"] for command in commands] == [
        "legacy",
        "mixed",
        "joint_g1_wave1",
        "joint_g1_wave3",
        "joint_g2_wave1",
        "joint_g2_wave3",
    ]

    for command, expected_group, expected_wave in zip(
        commands[2:],
        ("1", "1", "2", "2"),
        ("1", "3", "1", "3"),
        strict=True,
    ):
        argv = command["argv"]
        assert isinstance(argv, list)
        assert all(isinstance(argument, str) for argument in argv)
        assert _option_value(argv, "--batching-policy") == "joint"
        assert _option_value(argv, "--prefill-layer-group-size") == expected_group
        assert _option_value(argv, "--prefill-wave-max-chunks") == expected_wave


@pytest.mark.parametrize(
    ("joint_arguments", "expected_parameter"),
    [
        (["--joint-groups", "1,2"], "joint-waves"),
        (["--joint-waves", "1,3"], "joint-groups"),
        (["--joint-groups", "0", "--joint-waves", "1"], "joint-groups"),
        (["--joint-groups", "-1", "--joint-waves", "1"], "joint-groups"),
        (["--joint-groups", "not-an-int", "--joint-waves", "1"], "joint-groups"),
        (["--joint-groups", "1", "--joint-waves", "0"], "joint-waves"),
        (["--joint-groups", "1", "--joint-waves", "-3"], "joint-waves"),
        (["--joint-groups", "1", "--joint-waves", "not-an-int"], "joint-waves"),
    ],
)
def test_invalid_joint_arguments_fail_before_executable_launch(
    tmp_path: Path,
    joint_arguments: list[str],
    expected_parameter: str,
) -> None:
    launch_marker = tmp_path / "ft-executable-was-launched"
    fake_executable = tmp_path / "fake-ft-executable"
    fake_executable.write_text(
        "#!/bin/sh\n"
        f"printf '%s\\n' invoked > {shlex.quote(str(launch_marker))}\n"
        "exit 73\n",
        encoding="utf-8",
    )
    fake_executable.chmod(0o755)

    result = _run_cli(
        "--ft-executable",
        str(fake_executable),
        "--modes",
        "legacy",
        "mixed",
        *joint_arguments,
    )

    assert result.returncode != 0
    diagnostic = re.sub(r"[-_\s]+", " ", f"{result.stdout}\n{result.stderr}".lower())
    assert expected_parameter.replace("-", " ") in diagnostic
    assert not launch_marker.exists(), "invalid joint configuration launched the executable"


def _record(turn_index: int, ttft_seconds: float | None) -> dict[str, object]:
    events = [] if ttft_seconds is None else [{"at_seconds": ttft_seconds}]
    return {
        "turn_index": turn_index,
        "ttft_seconds": ttft_seconds,
        "tpot_seconds": 0.1,
        "nonempty_text_events": events,
        "gap_interpretation": "no_gap",
        "measurement_failed": False,
        "output_mismatch": None,
    }


def test_summarize_mode_reports_overall_ttft_with_linear_percentiles() -> None:
    from benchmarks.bench_lab_agent_policies import summarize_mode

    records = [
        _record(turn_index=0, ttft_seconds=1.0),
        _record(turn_index=1, ttft_seconds=None),
        _record(turn_index=0, ttft_seconds=2.0),
        _record(turn_index=1, ttft_seconds=3.0),
        _record(turn_index=2, ttft_seconds=4.0),
    ]
    repetitions = [
        {
            "makespan_seconds": 10.0,
            "submitted_prompt_throughput_tokens_per_second": 11.0,
            "prompt_throughput_tokens_per_second": 12.0,
            "actual_new_prefill_throughput_tokens_per_second": 13.0,
            "decode_throughput_tokens_per_second": 14.0,
        }
    ]

    summary = summarize_mode(records, repetitions)

    ttft = summary["ttft_seconds"]
    assert "first_turn" in ttft
    assert "later_turns" in ttft
    assert ttft["overall"]["p50"] == pytest.approx(2.5)
    assert ttft["overall"]["p95"] == pytest.approx(3.85)
