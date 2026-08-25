"""Black-box tests for FreeToken's sparse route-align dispatch contract."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import pytest
import torch


@dataclass
class _PublicModules:
    fused: Any
    cuda_ops: Any
    moe_ops: Any
    legacy_align: Any


@pytest.fixture
def public_modules() -> _PublicModules:
    return _PublicModules(
        fused=importlib.import_module("freetoken.moe.fused_nowag"),
        cuda_ops=importlib.import_module("nowag_vllm.cuda_ops"),
        moe_ops=importlib.import_module("nowag_vllm.moe_ops"),
        legacy_align=importlib.import_module("freetoken.kernel.triton.moe_align"),
    )


def _dummy_call_tensors(m: int, *, top_k: int = 8, num_experts: int = 4):
    hidden = 6
    intermediate = 6
    routes = m * top_k
    rows = routes * 16

    topk_ids = (
        torch.arange(routes, dtype=torch.int32).remainder(num_experts).view(m, top_k)
    )
    return {
        "args": (
            torch.zeros((m, hidden), dtype=torch.bfloat16),
            topk_ids,
            torch.full((m, top_k), 1.0 / top_k, dtype=torch.float32),
            torch.zeros((4096, 6), dtype=torch.bfloat16),
            torch.zeros((num_experts, 1, intermediate), dtype=torch.int32),
            torch.ones((num_experts, hidden), dtype=torch.bfloat16),
            torch.ones((num_experts, intermediate), dtype=torch.bfloat16),
            torch.zeros((num_experts, 1, intermediate), dtype=torch.int32),
            torch.ones((num_experts, hidden), dtype=torch.bfloat16),
            torch.ones((num_experts, intermediate), dtype=torch.bfloat16),
            torch.zeros((num_experts, 1, hidden), dtype=torch.int32),
            torch.ones((num_experts, intermediate), dtype=torch.bfloat16),
            torch.ones((num_experts, hidden), dtype=torch.bfloat16),
        ),
        "output": torch.empty((m, hidden), dtype=torch.bfloat16),
        "middle_workspace": torch.empty(
            (2 * rows, intermediate), dtype=torch.bfloat16
        ),
        "route_output_workspace": torch.empty(
            (routes, hidden), dtype=torch.bfloat16
        ),
        "topk_ids": topk_ids,
    }


def _install_runtime_stubs(
    monkeypatch: pytest.MonkeyPatch,
    modules: _PublicModules,
    *,
    capability: Callable[..., bool],
    sparse_align: Callable[..., Any],
    legacy_align: Callable[..., Any],
):
    captured: dict[str, Any] = {}

    def plugin_stub(*args, **kwargs):
        assert not args
        captured["align_routes"] = kwargs["align_routes"]
        return kwargs["output"]

    # Patch both public providers and the public wrapper's imported bindings. This
    # keeps the test independent of whether the wrapper uses a module-qualified
    # name or a directly imported public callable.
    monkeypatch.setattr(modules.moe_ops, "nowag_fused_moe", plugin_stub)
    monkeypatch.setattr(modules.fused, "nowag_fused_moe", plugin_stub, raising=False)

    monkeypatch.setattr(
        modules.cuda_ops, "has_moe_sparse_route_align", capability, raising=False
    )
    monkeypatch.setattr(
        modules.fused, "has_moe_sparse_route_align", capability, raising=False
    )
    monkeypatch.setattr(
        modules.cuda_ops, "moe_sparse_route_align", sparse_align, raising=False
    )
    monkeypatch.setattr(
        modules.fused, "moe_sparse_route_align", sparse_align, raising=False
    )
    monkeypatch.setattr(
        modules.legacy_align, "moe_align_block_size", legacy_align, raising=False
    )
    monkeypatch.setattr(
        modules.fused, "moe_align_block_size", legacy_align, raising=False
    )
    return captured


def _capture_align_callback(
    monkeypatch: pytest.MonkeyPatch,
    modules: _PublicModules,
    *,
    m: int,
    capability: Callable[..., bool],
    sparse_align: Callable[..., Any],
    legacy_align: Callable[..., Any],
):
    captured = _install_runtime_stubs(
        monkeypatch,
        modules,
        capability=capability,
        sparse_align=sparse_align,
        legacy_align=legacy_align,
    )
    tensors = _dummy_call_tensors(m)
    result = modules.fused.routed_experts_nowag(
        *tensors["args"],
        model_type="qwen3_5_moe",
        model_num_experts=4,
        gate_up_backend="auto",
        down_backend="auto",
        output=tensors["output"],
        middle_workspace=tensors["middle_workspace"],
        route_output_workspace=tensors["route_output_workspace"],
    )
    assert result is tensors["output"]
    return captured["align_routes"], tensors["topk_ids"]


def _recording_stub(result, calls: list[tuple[tuple[Any, ...], dict[str, Any]]]):
    def stub(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    return stub


def _assert_public_call(call, ids, block_size: int, num_experts: int):
    args, kwargs = call
    if args:
        assert len(args) == 3
        actual_ids, actual_block_size, actual_num_experts = args
    else:
        actual_ids = kwargs.get("topk_ids", kwargs.get("ids"))
        actual_block_size = kwargs.get("block_size")
        actual_num_experts = kwargs.get("num_experts")
    assert actual_ids is ids
    assert actual_block_size == block_size
    assert actual_num_experts == num_experts


def test_r256_uses_sparse_route_align_when_capability_present(
    monkeypatch, public_modules
):
    sparse_calls = []
    legacy_calls = []
    capability_calls = []
    expected = (
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    )

    def capability():
        capability_calls.append(True)
        return True

    callback, ids = _capture_align_callback(
        monkeypatch,
        public_modules,
        m=32,
        capability=capability,
        sparse_align=_recording_stub(expected, sparse_calls),
        legacy_align=_recording_stub(None, legacy_calls),
    )
    result = callback(ids, 16, 4)

    assert result is expected
    assert capability_calls == [True]
    assert len(sparse_calls) == 1
    assert not legacy_calls
    _assert_public_call(sparse_calls[0], ids, 16, 4)


def test_r264_delegates_to_legacy_even_when_capability_present(
    monkeypatch, public_modules
):
    sparse_calls = []
    legacy_calls = []
    capability_calls = []
    expected = (
        torch.tensor([4], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        torch.tensor([6], dtype=torch.int32),
    )

    def capability():
        capability_calls.append(True)
        return True

    callback, ids = _capture_align_callback(
        monkeypatch,
        public_modules,
        m=33,
        capability=capability,
        sparse_align=_recording_stub(None, sparse_calls),
        legacy_align=_recording_stub(expected, legacy_calls),
    )
    result = callback(ids, 32, 4)

    assert result is expected
    assert not capability_calls
    assert not sparse_calls
    assert len(legacy_calls) == 1
    _assert_public_call(legacy_calls[0], ids, 32, 4)


def test_capability_false_delegates_r256_to_legacy(monkeypatch, public_modules):
    sparse_calls = []
    legacy_calls = []
    capability_calls = []
    expected = (
        torch.tensor([7], dtype=torch.int32),
        torch.tensor([8], dtype=torch.int32),
        torch.tensor([9], dtype=torch.int32),
    )

    def capability():
        capability_calls.append(True)
        return False

    callback, ids = _capture_align_callback(
        monkeypatch,
        public_modules,
        m=32,
        capability=capability,
        sparse_align=_recording_stub(None, sparse_calls),
        legacy_align=_recording_stub(expected, legacy_calls),
    )
    result = callback(ids, 64, 4)

    assert result is expected
    assert capability_calls == [True]
    assert not sparse_calls
    assert len(legacy_calls) == 1
    _assert_public_call(legacy_calls[0], ids, 64, 4)


def test_sparse_route_align_error_propagates_unchanged(monkeypatch, public_modules):
    class SparseRouteError(RuntimeError):
        pass

    expected_error = SparseRouteError("public sparse-route failure")
    legacy_calls = []

    def sparse_align(*args, **kwargs):
        raise expected_error

    callback, ids = _capture_align_callback(
        monkeypatch,
        public_modules,
        m=32,
        capability=lambda: True,
        sparse_align=sparse_align,
        legacy_align=_recording_stub(None, legacy_calls),
    )

    with pytest.raises(SparseRouteError) as caught:
        callback(ids, 128, 4)
    assert caught.value is expected_error
    assert not legacy_calls
