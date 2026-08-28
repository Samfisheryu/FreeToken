from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from freetoken.core import Batch

from freetoken.scheduler.forward import ForwardInput

if TYPE_CHECKING:
    from freetoken.engine.engine import Engine


def _batch_view(batch: Batch, start: int, stop: int, decode_size: int) -> Batch:
    """Request-only view; allocation and accounting remain owned by Scheduler."""
    view = Batch(reqs=list(batch.reqs[start:stop]), decode_size=decode_size)
    view.log_new_tokens = batch.log_new_tokens
    view.log_cached_tokens = batch.log_cached_tokens
    view.prompt_admissions = list(batch.prompt_admissions)
    return view


def _sampling_view(args, start: int, stop: int):
    from freetoken.engine.sample import BatchSamplingArgs

    return BatchSamplingArgs(
        temperatures=(
            args.temperatures[start:stop]
            if args.temperatures is not None
            else None
        ),
        top_k=args.top_k[start:stop] if args.top_k is not None else None,
        top_p=args.top_p[start:stop] if args.top_p is not None else None,
    )


@dataclass(frozen=True)
class LayeredGroupResult:
    """Opaque model states after one mixed resident group."""

    prefill_state: object
    decode_state: object | None


@dataclass(frozen=True)
class LayeredGroupRun:
    """Opaque state ready to enter the active resident stage."""

    combined_state: object


class LayeredExecutionAdapter:
    """Model/backend-owned execution boundary for layered scheduling."""

    _REQUIRED_MODEL_METHODS = (
        "layer_group_num_layers",
        "layer_group_state_layer",
        "layer_group_merge_states",
        "layer_group_split_state",
        "begin_layer_group_prefill",
        "advance_layer_group_prefill",
        "finish_layer_group_prefill",
    )
    _RANGE_GRAPH_METHODS = (
        "create_layer_range_graph_inputs",
        "make_layer_range_graph_state",
        "stage_layer_range_graph_inputs",
        "finish_layer_range_graph_replay",
    )

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @classmethod
    def create(cls, engine: Engine) -> LayeredExecutionAdapter | None:
        model = engine.model
        factory = getattr(model, "create_layered_execution_adapter", None)
        if factory is not None:
            return factory(engine)
        if not all(hasattr(model, name) for name in cls._REQUIRED_MODEL_METHODS):
            return None
        return cls(engine)

    @property
    def num_stages(self) -> int:
        return self._engine.model.layer_group_num_layers

    @property
    def supports_range_graphs(self) -> bool:
        return self._engine.attn_backend.supports_layer_range_graphs and all(
            hasattr(self._engine.model, name)
            for name in self._RANGE_GRAPH_METHODS
        )

    def open_resident_wave(self):
        cache = self._engine.moe_offload_cache
        if cache is None:
            raise RuntimeError("layered execution requires an expert cache")
        return cache.open_resident_wave()

    def create_range_graph_inputs(self, seed_state: object):
        return self._engine.model.create_layer_range_graph_inputs(seed_state)

    def make_range_graph_state(self, inputs, start_stage: int, rows: int):
        return self._engine.model.make_layer_range_graph_state(
            inputs,
            start_stage,
            rows,
        )

    def stage_range_graph_inputs(
        self,
        inputs,
        state: object,
        rows: int,
        start_stage: int,
    ) -> None:
        self._engine.model.stage_layer_range_graph_inputs(
            inputs,
            state,
            rows,
            start_stage,
        )

    def finish_range_graph_replay(
        self,
        captured_state: object,
        rows: int,
        end_stage: int,
    ) -> object:
        return self._engine.model.finish_layer_range_graph_replay(
            captured_state,
            rows,
            end_stage,
        )

    def state_stage(self, state: object) -> int:
        return self._engine.model.layer_group_state_layer(state)

    def decode_view(
        self,
        mixed_input: ForwardInput,
        view: Batch | None = None,
    ) -> ForwardInput:
        mixed_batch = mixed_input.batch
        decode_size = mixed_batch.decode_size
        if view is None:
            view = _batch_view(mixed_batch, 0, decode_size, decode_size)
        decode_rows = sum(req.extend_len for req in view.reqs)
        if decode_rows != decode_size:
            raise RuntimeError("layered decode requires one query row per request")
        view.padded_reqs = list(view.reqs)
        view.positions = mixed_batch.positions[:decode_rows]
        if mixed_batch.out_loc is None:
            raise RuntimeError("layered source batch is missing output locations")
        view.out_loc = mixed_batch.out_loc[:decode_rows]
        view.active_table_idx = mixed_input.input_tuple[0][:decode_rows]
        view.input_ids = mixed_batch.input_ids[:decode_rows]
        metadata_ready = self._engine.prepare_execution_metadata_view(
            mixed_batch,
            view,
        )
        if not metadata_ready:
            raise RuntimeError("decode metadata view was not prepared")
        return ForwardInput(
            view,
            _sampling_view(mixed_input.sample_args, 0, decode_size),
            (
                mixed_input.input_tuple[0][:decode_rows],
                mixed_input.input_tuple[1][:decode_rows],
            ),
            (
                mixed_input.write_tuple[0][:decode_size],
                mixed_input.write_tuple[1][:decode_size],
            ),
        )

    def prefill_view(
        self,
        mixed_input: ForwardInput,
        view: Batch | None = None,
    ) -> tuple[ForwardInput, bool]:
        mixed_batch = mixed_input.batch
        request_start = mixed_batch.decode_size
        row_start = sum(req.extend_len for req in mixed_batch.decode_reqs)
        if view is None:
            view = _batch_view(
                mixed_batch,
                request_start,
                len(mixed_batch.reqs),
                0,
            )
        view.padded_reqs = list(view.reqs)
        view.positions = mixed_batch.positions[row_start:]
        if mixed_batch.out_loc is None:
            raise RuntimeError("layered source batch is missing output locations")
        view.out_loc = mixed_batch.out_loc[row_start:]
        view.input_ids = mixed_batch.input_ids[row_start:]
        metadata_ready = self._engine.prepare_execution_metadata_view(
            mixed_batch,
            view,
        )
        return (
            ForwardInput(
                view,
                _sampling_view(
                    mixed_input.sample_args,
                    request_start,
                    len(mixed_batch.reqs),
                ),
                (
                    mixed_input.input_tuple[0][row_start:],
                    mixed_input.input_tuple[1][row_start:],
                ),
                (
                    mixed_input.write_tuple[0][request_start:],
                    mixed_input.write_tuple[1][request_start:],
                ),
            ),
            metadata_ready,
        )

    def begin_group(
        self,
        group_input: ForwardInput,
        prefill_input: ForwardInput,
        prefill_state: object | None,
        decode_input: ForwardInput | None,
        start_stage: int,
    ) -> LayeredGroupRun:
        del prefill_input
        model = self._engine.model
        if start_stage == 0:
            if prefill_state is not None:
                raise RuntimeError("layered wave was embedded more than once")
            self._engine.restore_layered_linear_states(group_input.batch)
            combined_state = self._engine.begin_layer_group_prefill(group_input.batch)
        else:
            if prefill_state is None or self.state_stage(prefill_state) != start_stage:
                raise RuntimeError("layered prefill state is not at the active stage")
            if decode_input is None:
                combined_state = prefill_state
            else:
                decode_prefix = self._engine.begin_layer_group_decode(
                    decode_input.batch,
                    start_stage,
                )
                combined_state = model.layer_group_merge_states(
                    decode_prefix,
                    prefill_state,
                )

        return LayeredGroupRun(combined_state)

    def advance_group(
        self,
        group_input: ForwardInput,
        prefill_input: ForwardInput,
        run: LayeredGroupRun,
        decode_input: ForwardInput | None,
        end_stage: int,
    ) -> LayeredGroupResult:
        del prefill_input
        combined_state = self._engine.advance_layer_group_prefill(
            group_input.batch,
            run.combined_state,
            end_stage,
        )
        if decode_input is None:
            return LayeredGroupResult(combined_state, None)
        decode_rows = sum(req.extend_len for req in group_input.batch.decode_reqs)
        decode_state, prefill_state = self._engine.model.layer_group_split_state(
            combined_state,
            decode_rows,
        )
        return LayeredGroupResult(prefill_state, decode_state)

    def finish_decode(
        self,
        decode_input: ForwardInput,
        decode_state: object,
        resident_stage_end: int,
    ):
        if resident_stage_end < self.num_stages:
            decode_state = self._engine.advance_layer_group_decode(
                decode_input.batch,
                decode_state,
                self.num_stages,
            )
        return self._engine.finish_layer_group_prefill(
            decode_input.batch,
            decode_state,
            decode_input.sample_args,
        )

    def finish_prefill(
        self,
        prefill_input: ForwardInput,
        state: object,
        *,
        output_indices,
        request_indices,
    ):
        return self._engine.finish_layer_group_prefill(
            prefill_input.batch,
            state,
            prefill_input.sample_args,
            output_indices=output_indices,
            request_indices=request_indices,
        )


__all__ = [
    "LayeredExecutionAdapter",
    "LayeredGroupResult",
    "LayeredGroupRun",
]
