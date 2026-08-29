"""Layered execution adapter for DeepSeek-V4's distinct decode/ragged layouts."""

from __future__ import annotations

from dataclasses import dataclass

from freetoken.engine.layered_execution import (
    LayeredExecutionAdapter,
    LayeredGroupResult,
    LayeredGroupRun,
)
from freetoken.scheduler.forward import ForwardInput


@dataclass(frozen=True)
class _SeparateGroupStates:
    prefill: object
    decode: object | None


class DSV4LayeredExecutionAdapter(LayeredExecutionAdapter):
    """Run decode and ragged prefill inside one resident-group lifetime.

    DSV4 decode state is ``[B, 1, hc_mult, dim]`` and uses ``decode_step``;
    prefill state is ``[1, T, hc_mult, dim]`` and uses ragged attention. They
    share expert residency but cannot be concatenated into one attention batch.
    """

    def begin_group(
        self,
        group_input: ForwardInput,
        prefill_input: ForwardInput,
        prefill_state: object | None,
        decode_input: ForwardInput | None,
        start_stage: int,
    ) -> LayeredGroupRun:
        del group_input
        model_state = self._begin_prefill_state(
            prefill_input,
            prefill_state,
            start_stage,
        )
        decode_state = None
        if decode_input is not None:
            decode_state = (
                self._engine.begin_layer_group_prefill(decode_input.batch)
                if start_stage == 0
                else self._engine.begin_layer_group_decode(
                    decode_input.batch,
                    start_stage,
                )
            )
        return LayeredGroupRun(_SeparateGroupStates(model_state, decode_state))

    def advance_group(
        self,
        group_input: ForwardInput,
        prefill_input: ForwardInput,
        prefill_state: object | None,
        run: LayeredGroupRun,
        decode_input: ForwardInput | None,
        end_stage: int,
    ) -> LayeredGroupResult:
        del group_input
        states = run.combined_state
        if not isinstance(states, _SeparateGroupStates):
            raise TypeError("DSV4 layered group received incompatible state")

        decode_state = states.decode
        if decode_input is not None:
            if decode_state is None:
                raise RuntimeError("DSV4 layered decode state is missing")
            decode_state = self._engine.advance_layer_group_decode(
                decode_input.batch,
                decode_state,
                end_stage,
            )

        model_state = self._engine.advance_layer_group_prefill(
            prefill_input.batch,
            states.prefill,
            end_stage,
        )
        next_state, group_complete, wave_complete = self._complete_prefill_tile(
            prefill_state,
            model_state,
            end_stage,
        )
        return LayeredGroupResult(
            next_state,
            decode_state,
            group_complete,
            wave_complete,
        )


__all__ = ["DSV4LayeredExecutionAdapter"]
