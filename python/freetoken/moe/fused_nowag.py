"""Routed experts backed by NoWAG assignments and normalizers."""

from __future__ import annotations

import os

import torch

from freetoken.moe.nowag import (
    NO_ACTIVATION_ROUNDING,
    RUNTIME_ASSIGNMENT_LAYOUT,
    get_nowag_model_rule,
)


def routed_experts_nowag(
    x: torch.Tensor,
    slots: torch.Tensor,
    topk_weights: torch.Tensor,
    codebook: torch.Tensor,
    gate_assignments: torch.Tensor,
    gate_input_norm: torch.Tensor,
    gate_output_norm: torch.Tensor,
    up_assignments: torch.Tensor,
    up_input_norm: torch.Tensor,
    up_output_norm: torch.Tensor,
    down_assignments: torch.Tensor,
    down_input_norm: torch.Tensor,
    down_output_norm: torch.Tensor,
    *,
    model_type: str | None,
    model_num_experts: int | None = None,
    swiglu_limit: float | None = None,
    gate_up_backend: str = "auto",
    down_backend: str = "auto",
    cuda_launch_plan: object | None = None,
    output: torch.Tensor | None = None,
    middle_workspace: torch.Tensor | None = None,
    route_output_workspace: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run one supported model's activation math over shared-codebook experts."""
    if codebook.ndim != 2:
        raise ValueError(f"NoWAG codebook must be [C, d], got {tuple(codebook.shape)}")
    rule = get_nowag_model_rule(model_type)
    backend_override = os.environ.get("FREETOKEN_NOWAG_BACKEND")
    if backend_override is not None:
        if backend_override not in ("triton", "auto"):
            raise ValueError(
                "FREETOKEN_NOWAG_BACKEND must be 'triton' or 'auto'"
            )
        gate_up_backend = backend_override
        down_backend = backend_override

    try:
        from nowag_vllm.moe_ops import nowag_fused_moe
    except ImportError as exc:
        raise RuntimeError("NoWAG serving requires the local nowag_vllm package") from exc

    from freetoken.kernel import moe_sum_reduce_triton
    # NoWAG's smaller expert rows can make the auto-sized slot cache exceed the
    # native sgl_kernel aligner's 1024-entry scan geometry. The in-tree Triton
    # aligner has the same contract without that limit.
    from freetoken.kernel.triton.moe_align import (
        moe_align_block_size as moe_align_block_size_triton,
    )

    gate_up_input_transform = None
    middle_transform = None
    if rule.requires_swiglu_limit:
        if swiglu_limit is None:
            raise ValueError(f"{rule.model_type} NoWAG requires swiglu_limit")
        from freetoken.kernel.triton.dsv4.fp8_linear import (
            act_quant_fp8_inplace,
            act_quant_fp8_roundtrip,
        )

        def round_gate_up_input(hidden: torch.Tensor) -> torch.Tensor:
            return act_quant_fp8_roundtrip(hidden, 128)

        def round_down_input(middle: torch.Tensor) -> torch.Tensor:
            return act_quant_fp8_inplace(middle, 128)

        gate_up_input_transform = round_gate_up_input
        middle_transform = round_down_input
    else:
        swiglu_limit = None

    if (
        rule.gate_up_input_rounding == NO_ACTIVATION_ROUNDING
    ) != (gate_up_input_transform is None):
        raise RuntimeError("NoWAG Gate/Up rounding rule has no matching transform")
    if (
        rule.down_input_rounding == NO_ACTIVATION_ROUNDING
    ) != (middle_transform is None):
        raise RuntimeError("NoWAG Down rounding rule has no matching transform")

    shared = codebook.unsqueeze(0)
    return nowag_fused_moe(
        hidden_states=x,
        gate_codebook=shared,
        gate_packed_assignments=gate_assignments,
        gate_input_norm=gate_input_norm,
        gate_output_norm=gate_output_norm,
        gate_in_features=x.shape[1],
        up_codebook=shared,
        up_packed_assignments=up_assignments,
        up_input_norm=up_input_norm,
        up_output_norm=up_output_norm,
        up_in_features=x.shape[1],
        down_codebook=shared,
        down_packed_assignments=down_assignments,
        down_input_norm=down_input_norm,
        down_output_norm=down_output_norm,
        down_in_features=gate_output_norm.shape[1],
        topk_weights=topk_weights,
        topk_ids=slots,
        model_num_experts=model_num_experts,
        group_size=codebook.shape[1],
        assignment_bits=12,
        assignment_layout=RUNTIME_ASSIGNMENT_LAYOUT,
        validate_route_ids=False,
        structural_down=True,
        gate_up_backend=gate_up_backend,
        down_backend=down_backend,
        cuda_launch_plan=cuda_launch_plan,
        output=output,
        middle_workspace=middle_workspace,
        route_output_workspace=route_output_workspace,
        swiglu_limit=swiglu_limit,
        gate_up_input_rounding=rule.gate_up_input_rounding,
        down_input_rounding=rule.down_input_rounding,
        down_norm_placement=rule.down_norm_placement,
        gate_up_input_transform=gate_up_input_transform,
        middle_transform=middle_transform,
        align_routes=moe_align_block_size_triton,
        sum_routes=moe_sum_reduce_triton,
    )


def routed_experts_nowag_dsv4(
    x: torch.Tensor,
    slots: torch.Tensor,
    topk_weights: torch.Tensor,
    codebook: torch.Tensor,
    gate_assignments: torch.Tensor,
    gate_input_norm: torch.Tensor,
    gate_output_norm: torch.Tensor,
    up_assignments: torch.Tensor,
    up_input_norm: torch.Tensor,
    up_output_norm: torch.Tensor,
    down_assignments: torch.Tensor,
    down_input_norm: torch.Tensor,
    down_output_norm: torch.Tensor,
    swiglu_limit: float,
) -> torch.Tensor:
    """Keep the public DSV4 entry point used by existing tests and callers."""
    return routed_experts_nowag(
        x,
        slots,
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
        model_type="deepseek_v4",
        swiglu_limit=swiglu_limit,
    )


__all__ = ["routed_experts_nowag", "routed_experts_nowag_dsv4"]
