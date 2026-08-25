from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from freetoken.layers import (
    BaseOP,
    LinearColParallelMerged,
    LinearRowParallel,
    gelu_and_mul,
    gelu_tanh_and_mul,
    silu_and_mul,
)
from freetoken.utils import nvtx_annotate

if TYPE_CHECKING:
    import torch

    from .config import ModelConfig


class BaseLLMModel(ABC, BaseOP):
    supports_layer_group_prefill = False

    @abstractmethod
    def forward(self) -> torch.Tensor: ...


@dataclass
class LayerGroupState:
    hidden: torch.Tensor
    residual: torch.Tensor | None
    next_layer: int = 0


class ResidualLayerGroupCausalLM(BaseLLMModel):
    """Explicit opt-in for the common hidden/residual decoder-layer interface."""

    supports_layer_group_prefill = True

    @property
    def layer_group_num_layers(self) -> int:
        return len(self.model.layers.op_list)

    def begin_layer_group_prefill(self, input_ids: torch.Tensor) -> LayerGroupState:
        return LayerGroupState(
            hidden=self.model.embed_tokens.forward(input_ids),
            residual=None,
        )

    def advance_layer_group_prefill(
        self,
        state: LayerGroupState,
        end_layer: int,
    ) -> LayerGroupState:
        if not state.next_layer < end_layer <= self.layer_group_num_layers:
            raise ValueError(
                f"invalid layer-group range [{state.next_layer}, {end_layer}) for "
                f"{self.layer_group_num_layers} layers"
            )
        for layer_id in range(state.next_layer, end_layer):
            state.hidden, state.residual = self.model.layers.op_list[layer_id].forward(
                state.hidden, state.residual
            )
        state.next_layer = end_layer
        return state

    def finish_layer_group_prefill(
        self,
        state: LayerGroupState,
        output_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.next_layer != self.layer_group_num_layers:
            raise ValueError(
                "cannot finish layer-group prefill before every decoder layer ran"
            )
        if output_indices is None:
            hidden = self.model.norm.forward(state.hidden, state.residual)[0]
            return self.lm_head.forward(hidden)
        hidden = state.hidden[output_indices].contiguous()
        residual = (
            state.residual[output_indices].contiguous()
            if state.residual is not None
            else None
        )
        hidden = self.model.norm.forward(hidden, residual)[0]
        return self.lm_head.forward_selected(hidden)


class GatedMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate_up_proj = LinearColParallelMerged(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            has_bias=False,
        )

        fn_map = {"silu": silu_and_mul, "gelu": gelu_and_mul, "gelu_tanh": gelu_tanh_and_mul}
        act_fn = fn_map.get(config.hidden_act, None)
        if act_fn is None:
            raise ValueError(f"Unsupported activation function: {config.hidden_act}")
        self.act_fn = act_fn
        self.down_proj = LinearRowParallel(
            config.intermediate_size,
            config.hidden_size,
            has_bias=False,
        )

    @nvtx_annotate("MLP")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj.forward(x)
        del x
        y = self.act_fn(gate_up)
        del gate_up
        return self.down_proj.forward(y)


__all__ = [
    "BaseLLMModel",
    "GatedMLP",
    "LayerGroupState",
    "ResidualLayerGroupCausalLM",
]
