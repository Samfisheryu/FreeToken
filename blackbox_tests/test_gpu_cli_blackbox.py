"""Black-box checks for the public ``--gpu`` CLI contract."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import textwrap

import pytest


@pytest.fixture(scope="session")
def cli() -> list[str]:
    command = shlex.split(os.environ.get("FT_CLI", "ft"))
    if not command or shutil.which(command[0]) is None:
        pytest.skip("FT_CLI is not available")
    return command


@pytest.fixture(scope="session")
def gpus() -> list[tuple[str, str]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi is None:
        pytest.skip("nvidia-smi is not available")
    result = subprocess.run(
        [nvidia_smi, "-L"], capture_output=True, text=True, timeout=10
    )
    if result.returncode:
        pytest.skip("no NVIDIA GPU is available")
    entries = re.findall(r"GPU (\d+):.*\(UUID: (GPU-[^)]+)\)", result.stdout)
    if not entries:
        pytest.skip("nvidia-smi reported no GPU")
    return entries


def _run(cli: list[str], *args: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(
        [*cli, *args], capture_output=True, text=True, timeout=timeout
    )


def _output(result: subprocess.CompletedProcess) -> str:
    return result.stdout + result.stderr


def _missing_model(tmp_path: Path) -> str:
    path = tmp_path / "model-that-does-not-exist"
    assert not path.exists()
    return str(path)


def _binding_python() -> str:
    executable = os.environ.get("FT_GPU_BINDING_PYTHON")
    if not executable or not Path(executable).is_file():
        pytest.skip("FT_GPU_BINDING_PYTHON is not configured")
    return executable


def _run_gpu_public_function(
    code: str,
    *args: str,
    env_updates: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env.update(env_updates or {})
    python_path = str(Path.cwd() / "python")
    env["PYTHONPATH"] = python_path + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [_binding_python(), "-c", textwrap.dedent(code), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=env,
    )


def test_serve_help_documents_gpu_and_tp_contract(cli):
    result = _run(cli, "serve", "--help")
    assert result.returncode == 0, _output(result)
    output = _output(result)
    normalized = " ".join(output.split())
    assert "--gpu GPU" in output
    assert "comma-separated" in output
    assert "entry i is TP rank i" in normalized
    assert "--tensor-parallel-size" in output


def test_serve_accepts_real_index_and_uuid_prefix_at_parse_layer(cli, gpus, tmp_path):
    index, uuid = gpus[0]
    missing_model = _missing_model(tmp_path)
    for gpu in (index, uuid[:12]):
        result = _run(
            cli,
            "serve",
            "--model-path",
            missing_model,
            "--gpu",
            gpu,
            "--tp-size",
            "1",
        )
        output = _output(result)
        assert result.returncode == 1, output
        assert missing_model in output
        assert "ft serve: error:" not in output


def test_serve_accepts_one_real_gpu_per_tp_rank(cli, gpus, tmp_path):
    if len(gpus) < 2:
        pytest.skip("two GPUs are required for the TP parsing check")
    missing_model = _missing_model(tmp_path)
    result = _run(
        cli,
        "serve",
        "--model-path",
        missing_model,
        "--gpu",
        f"{gpus[0][0]},{gpus[1][0]}",
        "--tp-size",
        "2",
    )
    output = _output(result)
    assert result.returncode == 1, output
    assert missing_model in output
    assert "ft serve: error:" not in output


@pytest.mark.parametrize(
    "gpu,tp_size",
    [("0", "2"), ("0,1", "1"), ("0", "0")],
)
def test_serve_rejects_gpu_tp_count_mismatch(cli, tmp_path, gpu, tp_size):
    result = _run(
        cli,
        "serve",
        "--model-path",
        _missing_model(tmp_path),
        "--gpu",
        gpu,
        "--tp-size",
        tp_size,
    )
    output = _output(result)
    assert result.returncode == 2, output
    assert "ft serve: error:" in output
    assert "--gpu" in output
    assert "--tensor-parallel-size" in output or "tensor parallel" in output


def test_serve_rejects_noninteger_tp_size(cli, tmp_path):
    result = _run(
        cli,
        "serve",
        "--model-path",
        _missing_model(tmp_path),
        "--gpu",
        "0",
        "--tp-size",
        "two",
    )
    output = _output(result)
    assert result.returncode == 2, output
    assert "invalid int value" in output


@pytest.mark.parametrize(
    "gpu,expected",
    [
        ("0,GPU-deadbeef", "all UUIDs or all indices"),
        ("", "at least one GPU"),
        ("999999", "GPU(s) on this machine"),
        ("GPU-not-real", "not found or not a unique prefix"),
        ("0,0", "same GPU appears twice"),
    ],
)
def test_public_gpu_selector_rejects_invalid_specs(gpu, expected):
    result = _run_gpu_public_function(
        """
        import sys
        from freetoken.gpu_select import parse_gpu_spec, resolve_gpu_uuids

        resolve_gpu_uuids(parse_gpu_spec(sys.argv[1]))
        """,
        gpu,
    )
    assert result.returncode != 0
    assert expected in _output(result)


def test_index_and_uuid_prefix_bind_the_same_physical_gpu(gpus):
    index, uuid = gpus[0]
    code = """
        import json
        import sys
        from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, gpu_identity

        assign_gpu(sys.argv[1])
        device = bind_assigned_gpu()
        print(json.dumps(gpu_identity(device.index)))
    """
    identities = []
    for spec in (index, uuid[:12]):
        result = _run_gpu_public_function(code, spec)
        assert result.returncode == 0, _output(result)
        identities.append(json.loads(result.stdout.strip().splitlines()[-1]))
    assert identities[0]["uuid"] == uuid
    assert identities[1]["uuid"] == uuid
    assert identities[0] == identities[1]


def test_index_is_relative_to_numeric_cuda_visible_devices(gpus):
    if len(gpus) < 2:
        pytest.skip("two GPUs are required for the CUDA_VISIBLE_DEVICES check")
    physical_index, uuid = gpus[1]
    result = _run_gpu_public_function(
        """
        import json
        from freetoken.gpu_select import assign_gpu, bind_assigned_gpu, gpu_identity

        assign_gpu("0")
        device = bind_assigned_gpu()
        print(json.dumps(gpu_identity(device.index)))
        """,
        env_updates={"CUDA_VISIBLE_DEVICES": physical_index},
    )
    assert result.returncode == 0, _output(result)
    identity = json.loads(result.stdout.strip().splitlines()[-1])
    assert identity["index"] == 0
    assert identity["uuid"] == uuid


@pytest.mark.parametrize(
    "command",
    [("checkpoint", "--help"), ("bench", "bw", "--help")],
)
def test_checkpoint_and_bench_help_publish_gpu_not_device(cli, command):
    result = _run(cli, *command)
    assert result.returncode == 0, _output(result)
    output = _output(result)
    assert "--gpu GPU" in output
    assert "--device" not in output


def test_checkpoint_rejects_removed_device_option(cli, tmp_path):
    result = _run(
        cli,
        "checkpoint",
        "--model",
        _missing_model(tmp_path),
        "--out",
        str(tmp_path / "out"),
        "--device",
        "cuda:0",
    )
    output = _output(result)
    assert result.returncode == 2, output
    assert "unrecognized arguments: --device cuda:0" in output


def test_bench_rejects_removed_device_option(cli):
    result = _run(cli, "bench", "bw", "--device", "cuda:0")
    output = _output(result)
    assert result.returncode == 2, output
    assert "unrecognized arguments: --device cuda:0" in output
