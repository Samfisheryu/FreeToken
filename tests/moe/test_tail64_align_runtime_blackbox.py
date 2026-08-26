from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import Any

import pytest
import torch


@dataclass(frozen=True)
class _Modules:
    fused: Any
    align: Any
    moe_ops: Any
    tuning: Any


def _load_modules() -> _Modules:
    return _Modules(
        fused=importlib.import_module("freetoken.moe.fused_nowag"),
        align=importlib.import_module("freetoken.kernel.triton.moe_align"),
        moe_ops=importlib.import_module("nowag_vllm.moe_ops"),
        tuning=importlib.import_module("nowag_vllm.moe_tuning"),
    )


def _manual_tail64_plan(modules: _Modules):
    stage = modules.tuning.MoeCudaStageConfig(
        block_m=16,
        block_n=128,
        num_warps=8,
        num_stages=2,
        tasks_per_cta=1,
    )
    return modules.tuning.MoeCudaLaunchPlan(
        gate_up=stage,
        down=stage,
        source="manual",
        profile_name=None,
        adaptive_m_tiles=True,
        adaptive_residual_policy="tail64",
    )


def _call_routed(
    modules: _Modules,
    *,
    num_routes: int,
    middle_workspace: torch.Tensor | None,
) -> torch.Tensor:
    num_tokens = num_routes
    hidden_size = 6
    intermediate_size = 6
    num_experts = 4
    x = torch.zeros((num_tokens, hidden_size), dtype=torch.bfloat16)
    topk_ids = torch.zeros((num_tokens, 1), dtype=torch.int32)
    topk_weights = torch.ones((num_tokens, 1), dtype=torch.float32)
    codebook = torch.zeros((4096, 6), dtype=torch.bfloat16)
    gate_assignments = torch.zeros(
        (num_experts, 1, intermediate_size), dtype=torch.int32
    )
    gate_input_norm = torch.ones((num_experts, hidden_size), dtype=torch.bfloat16)
    gate_output_norm = torch.ones(
        (num_experts, intermediate_size), dtype=torch.bfloat16
    )
    up_assignments = torch.zeros(
        (num_experts, 1, intermediate_size), dtype=torch.int32
    )
    up_input_norm = torch.ones((num_experts, hidden_size), dtype=torch.bfloat16)
    up_output_norm = torch.ones(
        (num_experts, intermediate_size), dtype=torch.bfloat16
    )
    down_assignments = torch.zeros((num_experts, 1, hidden_size), dtype=torch.int32)
    down_input_norm = torch.ones(
        (num_experts, intermediate_size), dtype=torch.bfloat16
    )
    down_output_norm = torch.ones((num_experts, hidden_size), dtype=torch.bfloat16)
    output = torch.empty_like(x)
    routes = torch.empty((num_routes, hidden_size), dtype=torch.bfloat16)

    return modules.fused.routed_experts_nowag(
        x,
        topk_ids,
        topk_weights,
        codebook,
        gate_assignments,
        gate_input_norm,
        gate_output_norm,
        up_assignments,
        up_input_norm,
        up_output_norm,
        down_assignments,
        down_input_norm,
        down_output_norm,
        model_type="qwen3_5_moe",
        model_num_experts=num_experts,
        gate_up_backend="cuda_exact_k48",
        down_backend="cuda_exact_k48",
        cuda_launch_plan=_manual_tail64_plan(modules),
        output=output,
        middle_workspace=middle_workspace,
        route_output_workspace=routes,
    )


def _patch_public_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
    modules: _Modules,
    *,
    fused_impl,
    plain_impl,
    tail64_impl,
) -> None:
    monkeypatch.setattr(modules.fused, "nowag_fused_moe", fused_impl, raising=False)
    monkeypatch.setattr(modules.moe_ops, "nowag_fused_moe", fused_impl)
    monkeypatch.setattr(
        modules.fused, "moe_align_block_size", plain_impl, raising=False
    )
    monkeypatch.setattr(modules.align, "moe_align_block_size", plain_impl)
    monkeypatch.setattr(
        modules.fused,
        "moe_align_block_size_adaptive_tail64",
        tail64_impl,
        raising=False,
    )
    monkeypatch.setattr(
        modules.align, "moe_align_block_size_adaptive_tail64", tail64_impl
    )


def test_large_align_threshold_is_strictly_above_1024_routes() -> None:
    modules = _load_modules()

    assert modules.align.uses_large_moe_align(1024) is False
    assert modules.align.uses_large_moe_align(1025) is True


@pytest.mark.parametrize(
    ("num_routes", "expect_tail64", "expect_caller_owned"),
    [(1024, False, False), (1025, True, True)],
)
def test_manual_tail64_callback_and_alignment_storage_contract(
    monkeypatch: pytest.MonkeyPatch,
    num_routes: int,
    expect_tail64: bool,
    expect_caller_owned: bool,
) -> None:
    modules = _load_modules()
    captured: dict[str, Any] = {}
    plain_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    tail64_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def fake_fused(*args: Any, **kwargs: Any):
        captured.update(kwargs)
        return kwargs["output"]

    def fake_plain(*args: Any, **kwargs: Any):
        plain_calls.append((args, kwargs))
        return "plain"

    def fake_tail64(*args: Any, **kwargs: Any):
        tail64_calls.append((args, kwargs))
        return "tail64"

    _patch_public_entrypoints(
        monkeypatch,
        modules,
        fused_impl=fake_fused,
        plain_impl=fake_plain,
        tail64_impl=fake_tail64,
    )
    middle_workspace = (
        torch.empty((num_routes * 32 + 4096, 6), dtype=torch.bfloat16)
        if expect_caller_owned
        else None
    )

    result = _call_routed(
        modules,
        num_routes=num_routes,
        middle_workspace=middle_workspace,
    )

    assert result.shape == (num_routes, 6)
    assert captured["caller_owned_alignment_storage"] is expect_caller_owned
    ordinary_ids = torch.zeros((num_routes, 1), dtype=torch.int32)
    assert captured["align_routes"](ordinary_ids, 16, 4) == "plain"
    assert len(plain_calls) == 1
    plain_args, plain_kwargs = plain_calls[0]
    assert plain_args[0] is ordinary_ids
    assert plain_args[1:] == (16, 4)
    assert plain_kwargs == {"alignment_storage": None}

    tail64_callback = captured["align_routes_adaptive_tail64"]
    if expect_tail64:
        assert callable(tail64_callback)
        adaptive_args = tuple(object() for _ in range(6))
        assert tail64_callback(*adaptive_args) == "tail64"
        assert tail64_calls == [(adaptive_args, {})]
    else:
        assert tail64_callback is None
        assert tail64_calls == []


def test_tail64_callback_exception_propagates_to_public_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    modules = _load_modules()
    expected = RuntimeError("tail64 alignment failed")

    def fake_tail64(*args: Any, **kwargs: Any):
        raise expected

    def fake_fused(*args: Any, **kwargs: Any):
        adaptive_args = tuple(object() for _ in range(6))
        kwargs["align_routes_adaptive_tail64"](*adaptive_args)
        raise AssertionError("callback failure was swallowed")

    _patch_public_entrypoints(
        monkeypatch,
        modules,
        fused_impl=fake_fused,
        plain_impl=lambda *args, **kwargs: "plain",
        tail64_impl=fake_tail64,
    )
    middle_workspace = torch.empty((1025 * 32 + 4096, 6), dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="tail64 alignment failed") as caught:
        _call_routed(modules, num_routes=1025, middle_workspace=middle_workspace)

    assert caught.value is expected
