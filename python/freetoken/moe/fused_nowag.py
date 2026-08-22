"""DeepSeek-V4 routed experts backed by NoWAG assignments and normalizers."""

from __future__ import annotations

import torch


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
    """Run DSV4's W4A8 activation sequence with NoWAG expert weights.

    ``slots`` addresses either the unified cache (decode) or one materialized
    expert layer (prefill). The shared codebook has no expert dimension and is
    viewed as one bank by the NoWAG kernel, so it remains a single GPU tensor.
    """
    if codebook.ndim != 2:
        raise ValueError(f"NoWAG codebook must be [C, d], got {tuple(codebook.shape)}")

    try:
        from nowag_vllm.moe_ops import nowag_fused_moe
    except ImportError as exc:
        raise RuntimeError(
            "DeepSeek-V4 NoWAG serving requires the local nowag_vllm package"
        ) from exc

    from freetoken.kernel import moe_sum_reduce_triton
    from freetoken.kernel.triton.dsv4.fp8_linear import (
        act_quant_fp8_inplace,
        act_quant_fp8_roundtrip,
    )
    from freetoken.moe.fused import moe_align_block_size

    rounded_x = act_quant_fp8_roundtrip(x, 128)

    def round_down_input(middle: torch.Tensor) -> torch.Tensor:
        return act_quant_fp8_inplace(middle, 128)

    shared = codebook.unsqueeze(0)
    return nowag_fused_moe(
        hidden_states=rounded_x,
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
        group_size=codebook.shape[1],
        assignment_bits=12,
        assignment_layout="row_major",
        validate_route_ids=False,
        structural_down=False,
        swiglu_limit=swiglu_limit,
        middle_transform=round_down_input,
        align_routes=moe_align_block_size,
        sum_routes=moe_sum_reduce_triton,
    )


__all__ = ["routed_experts_nowag_dsv4"]
