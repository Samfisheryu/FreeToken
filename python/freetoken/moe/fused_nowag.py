"""Routed experts backed by NoWAG assignments and normalizers."""

from __future__ import annotations

import os

import torch

from freetoken.moe.nowag import (
    NO_ACTIVATION_ROUNDING,
    RUNTIME_ASSIGNMENT_LAYOUT,
    get_nowag_model_rule,
)


def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
    return (
        left.untyped_storage().data_ptr()
        == right.untyped_storage().data_ptr()
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
        from nowag_vllm import cuda_ops as nowag_cuda_ops
        from nowag_vllm.moe_ops import nowag_fused_moe
    except ImportError as exc:
        raise RuntimeError("NoWAG serving requires the local nowag_vllm package") from exc

    from freetoken.kernel import moe_sum_reduce_triton
    # Larger route sets retain the in-tree Triton aligner, which has no native
    # sgl_kernel limit on the number of physical expert rows.
    from freetoken.kernel.triton.moe_align import (
        moe_align_block_size as moe_align_block_size_triton,
        moe_align_block_size_adaptive,
        moe_align_block_size_adaptive_tail64,
        uses_large_moe_align,
    )

    def align_nowag_routes(
        topk_ids: torch.Tensor,
        block_size: int,
        physical_expert_rows: int,
        *,
        alignment_storage: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if (
            alignment_storage is None
            and topk_ids.numel() <= 256
            and nowag_cuda_ops.has_moe_sparse_route_align()
        ):
            return nowag_cuda_ops.moe_sparse_route_align(
                topk_ids=topk_ids,
                block_size=block_size,
                num_experts=physical_expert_rows,
            )
        return moe_align_block_size_triton(
            topk_ids,
            block_size,
            physical_expert_rows,
            alignment_storage=alignment_storage,
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

        gate_roundtrip_output = output
        if gate_roundtrip_output is not None:
            if not isinstance(gate_roundtrip_output, torch.Tensor):
                raise ValueError("output must be a tensor")
            for live_name, live_tensor in (
                ("input", x),
                ("middle_workspace", middle_workspace),
                ("route_output_workspace", route_output_workspace),
            ):
                if (
                    isinstance(live_tensor, torch.Tensor)
                    and _shares_storage(gate_roundtrip_output, live_tensor)
                ):
                    raise ValueError(
                        "output reused for Gate/Up rounding must not share "
                        f"storage with {live_name}"
                    )

        def round_gate_up_input(hidden: torch.Tensor) -> torch.Tensor:
            return act_quant_fp8_roundtrip(
                hidden,
                128,
                output=gate_roundtrip_output,
            )

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
        align_routes=align_nowag_routes,
        align_routes_adaptive=(
            moe_align_block_size_adaptive
            if uses_large_moe_align(slots.numel())
            else None
        ),
        align_routes_adaptive_tail64=(
            moe_align_block_size_adaptive_tail64
            if uses_large_moe_align(slots.numel())
            else None
        ),
        caller_owned_alignment_storage=middle_workspace is not None,
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
