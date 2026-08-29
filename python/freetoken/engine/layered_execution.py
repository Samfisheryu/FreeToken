from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

import torch
from freetoken.core import Batch, Req

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
    group_complete: bool
    wave_complete: bool


@dataclass(frozen=True)
class LayeredGroupRun:
    """Opaque state ready to enter the active resident stage."""

    combined_state: object


@dataclass(frozen=True)
class _LayeredPrefillTile:
    """Allocation-free execution view of contiguous rows in one logical wave."""

    forward_input: ForwardInput
    terminal_rows: tuple[tuple[int, int], ...]


@dataclass
class _TiledPrefillState:
    """Adapter-owned stage x tile cursor and opaque per-tile model states."""

    source_input: ForwardInput
    tiles: tuple[_LayeredPrefillTile, ...]
    tile_states: list[object | None]
    prefill_execution: PrefillExecutionSession | None
    tile_index: int = 0
    stage_start: int = 0
    restored: bool = False


class PrefillExecutionSession(Protocol):
    """Opaque cache-owned resources for a physically tiled prefill."""

    def activate(self, reqs: list[Req]) -> None: ...

    def rewind(self) -> None: ...

    def close(self) -> None: ...

    def cancel(self) -> None: ...


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
        if isinstance(state, _TiledPrefillState):
            return state.stage_start
        return self._engine.model.layer_group_state_layer(state)

    @staticmethod
    def group_finishes_after_current_tile(state: object | None) -> bool:
        return (
            not isinstance(state, _TiledPrefillState)
            or state.tile_index + 1 == len(state.tiles)
        )

    def _begin_prefill_state(
        self,
        prefill_input: ForwardInput,
        prefill_state: object | None,
        start_stage: int,
        begin_input: ForwardInput | None = None,
    ) -> object:
        tiled = (
            prefill_state
            if isinstance(prefill_state, _TiledPrefillState)
            else None
        )
        model_state = (
            tiled.tile_states[tiled.tile_index]
            if tiled is not None
            else prefill_state
        )
        if start_stage == 0:
            if model_state is not None:
                raise RuntimeError("layered prefill tile was embedded more than once")
            if tiled is None:
                self._engine.restore_layered_linear_states(prefill_input.batch)
            elif not tiled.restored:
                self._engine.restore_layered_linear_states(tiled.source_input.batch)
                tiled.restored = True
            return self._engine.begin_layer_group_prefill(
                (begin_input or prefill_input).batch
            )
        if model_state is None or self._engine.model.layer_group_state_layer(
            model_state
        ) != start_stage:
            raise RuntimeError("layered prefill state is not at the active stage")
        return model_state

    def _complete_prefill_tile(
        self,
        prefill_state: object | None,
        model_state: object,
        end_stage: int,
    ) -> tuple[object, bool, bool]:
        if not isinstance(prefill_state, _TiledPrefillState):
            return model_state, True, end_stage == self.num_stages

        prefill_state.tile_states[prefill_state.tile_index] = model_state
        prefill_state.tile_index += 1
        group_complete = prefill_state.tile_index == len(prefill_state.tiles)
        if group_complete:
            prefill_state.tile_index = 0
            prefill_state.stage_start = end_stage
        return (
            prefill_state,
            group_complete,
            group_complete and end_stage == self.num_stages,
        )

    def initialize_prefill_state(
        self,
        prefill_input: ForwardInput,
        token_budget: int,
        prefill_execution: PrefillExecutionSession | None = None,
    ) -> object | None:
        """Freeze FIFO row tiles before the first resident group executes."""
        tiles = self.tile_prefill_input(prefill_input, token_budget)
        if not tiles:
            return None
        return _TiledPrefillState(
            source_input=prefill_input,
            tiles=tiles,
            tile_states=[None] * len(tiles),
            prefill_execution=prefill_execution,
        )

    @staticmethod
    def _prefill_rows(batch: Batch) -> int:
        return sum(req.extend_len for req in batch.prefill_reqs)

    def uses_prefill_tiles(
        self,
        batch: Batch,
        token_budget: int,
    ) -> bool:
        return self._prefill_rows(batch) > token_budget

    def prepare_metadata_view(self, source: Batch, target: Batch) -> bool:
        """Attach backend-owned metadata to an allocation-free batch view."""
        return self._engine.attn_backend.prepare_metadata_view(source, target)

    def capture_prefill_metadata_state(self, prefill_input: ForwardInput):
        del prefill_input
        return None

    def restore_prefill_metadata_state(
        self,
        group_input: ForwardInput,
        prefill_input: ForwardInput,
        state,
    ) -> None:
        del group_input, prefill_input
        if state is not None:
            raise RuntimeError("model adapter cannot restore prefill metadata state")

    def capture_stable_decode_state(self, batch: Batch) -> object | None:
        if (
            self._engine.linear_state_pool is not None
            or not self._engine.graph_runner.can_use_cuda_graph(batch)
        ):
            return None
        return self._engine.attn_backend.capture_stable_decode_state(batch)

    def restore_stable_decode_state(self, batch: Batch, state: object) -> bool:
        if self._engine.linear_state_pool is not None:
            return False
        return self._engine.attn_backend.restore_stable_decode_state(batch, state)

    def current_prefill_input(
        self,
        prefill_input: ForwardInput,
        prefill_state: object | None,
    ) -> ForwardInput:
        """Return the rows executed by the wave's current physical step."""
        if not isinstance(prefill_state, _TiledPrefillState):
            return prefill_input
        tile = prefill_state.tiles[prefill_state.tile_index]
        if prefill_state.prefill_execution is not None:
            prefill_state.prefill_execution.activate(tile.forward_input.batch.prefill_reqs)
        return tile.forward_input

    def tile_prefill_input(
        self,
        prefill_input: ForwardInput,
        max_rows: int,
    ) -> tuple[_LayeredPrefillTile, ...]:
        """Pack FIFO causal row views without allocating KV or advancing requests.

        A wave that fits ``max_rows`` returns no tiles so its existing execution
        path remains untouched.
        """
        if max_rows < 1:
            raise ValueError("layered prefill tile size must be positive")
        source = prefill_input.batch
        if source.decode_size != 0:
            raise ValueError("layered prefill tiling requires a prefill-only view")
        total_rows = int(source.input_ids.numel())
        if total_rows <= max_rows:
            return ()
        tiles: list[_LayeredPrefillTile] = []
        linear_progress: dict[int, tuple[int, int | None]] = {}
        tile_reqs: list[Req] = []
        tile_request_indices: list[int] = []
        tile_terminal_rows: list[tuple[int, int]] = []
        tile_rows = 0
        row_start = 0

        def finish_tile() -> None:
            nonlocal tile_reqs, tile_request_indices, tile_terminal_rows
            nonlocal tile_rows, row_start
            row_stop = row_start + tile_rows
            batch = Batch(reqs=tile_reqs, decode_size=0)
            batch.padded_reqs = list(tile_reqs)
            batch.positions = source.positions[row_start:row_stop]
            if source.out_loc is None:
                raise RuntimeError("layered prefill source is missing output locations")
            batch.out_loc = source.out_loc[row_start:row_stop]
            batch.input_ids = source.input_ids[row_start:row_stop]
            tile_input_mapping = (
                prefill_input.input_tuple[0][row_start:row_stop],
                prefill_input.input_tuple[1][row_start:row_stop],
            )
            self._engine.prepare_execution_metadata(
                batch,
                tile_input_mapping,
                linear_cache_is_hybrid=(
                    getattr(self._engine.config, "cache_type", "")
                    == "hybrid_radix"
                ),
            )
            for req_view, request_index in zip(
                tile_reqs,
                tile_request_indices,
                strict=True,
            ):
                linear_progress[request_index] = (
                    req_view.mamba_next_track_idx,
                    req_view.mamba_last_track_seqlen,
                )
                owner = source.reqs[request_index]
                owner.mamba_next_track_idx = req_view.mamba_next_track_idx
                owner.mamba_last_track_seqlen = req_view.mamba_last_track_seqlen
            request_start = tile_request_indices[0]
            request_stop = tile_request_indices[-1] + 1
            request_slice = slice(request_start, request_stop)
            tiles.append(
                _LayeredPrefillTile(
                    forward_input=ForwardInput(
                        batch,
                        _sampling_view(
                            prefill_input.sample_args,
                            request_start,
                            request_stop,
                        ),
                        (
                            prefill_input.input_tuple[0][row_start:row_stop],
                            prefill_input.input_tuple[1][row_start:row_stop],
                        ),
                        (
                            prefill_input.write_tuple[0][request_slice],
                            prefill_input.write_tuple[1][request_slice],
                        ),
                    ),
                    terminal_rows=tuple(tile_terminal_rows),
                )
            )
            row_start = row_stop
            tile_reqs = []
            tile_request_indices = []
            tile_terminal_rows = []
            tile_rows = 0

        for request_index, req in enumerate(source.reqs):
            request_row = 0
            while request_row < req.extend_len:
                request_remaining = req.extend_len - request_row
                take = min(request_remaining, max_rows - tile_rows)
                req_view = copy(req)
                progress = linear_progress.get(request_index)
                if progress is not None:
                    (
                        req_view.mamba_next_track_idx,
                        req_view.mamba_last_track_seqlen,
                    ) = progress
                req_view.cached_len = req.cached_len + request_row
                req_view.device_len = req_view.cached_len + take
                req_view.input_ids = req.input_ids[: req_view.device_len]
                tile_reqs.append(req_view)
                tile_request_indices.append(request_index)
                local_last_row = tile_rows + take - 1
                request_row += take
                tile_rows += take
                if request_row == req.extend_len:
                    tile_terminal_rows.append((request_index, local_last_row))
                if tile_rows == max_rows:
                    finish_tile()

        if tile_reqs:
            finish_tile()
        if row_start != total_rows:
            raise RuntimeError(
                f"layered prefill tiles cover {row_start} of {total_rows} rows"
            )
        return tuple(tiles)

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
        metadata_ready = self.prepare_metadata_view(
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
        *,
        prepare_metadata: bool = True,
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
        metadata_ready = False
        if prepare_metadata:
            metadata_ready = self.prepare_metadata_view(
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
        model = self._engine.model
        model_state = self._begin_prefill_state(
            prefill_input,
            prefill_state,
            start_stage,
            group_input if decode_input is not None else None,
        )
        if start_stage == 0:
            combined_state = model_state
        else:
            if decode_input is None:
                combined_state = model_state
            else:
                decode_prefix = self._engine.begin_layer_group_decode(
                    decode_input.batch,
                    start_stage,
                )
                combined_state = model.layer_group_merge_states(
                    decode_prefix,
                    model_state,
                )

        return LayeredGroupRun(combined_state)

    def advance_group(
        self,
        group_input: ForwardInput,
        prefill_input: ForwardInput,
        prefill_state: object | None,
        run: LayeredGroupRun,
        decode_input: ForwardInput | None,
        end_stage: int,
    ) -> LayeredGroupResult:
        combined_state = self._engine.advance_layer_group_prefill(
            group_input.batch,
            run.combined_state,
            end_stage,
        )
        decode_state = None
        tile_state = combined_state
        if decode_input is not None:
            decode_rows = sum(req.extend_len for req in group_input.batch.decode_reqs)
            decode_state, tile_state = self._engine.model.layer_group_split_state(
                combined_state,
                decode_rows,
            )
        next_state, group_complete, wave_complete = self._complete_prefill_tile(
            prefill_state,
            tile_state,
            end_stage,
        )
        return LayeredGroupResult(
            next_state,
            decode_state,
            group_complete,
            wave_complete,
        )

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

    def finish_prefill_wave(
        self,
        prefill_input: ForwardInput,
        state: object,
        request_indices: list[int],
    ):
        """Finish each selected logical request exactly once at its terminal tile."""
        if not isinstance(state, _TiledPrefillState):
            rows: list[int] = []
            row = 0
            selected = set(request_indices)
            for index, req in enumerate(prefill_input.batch.prefill_reqs):
                row += req.extend_len
                if index in selected:
                    rows.append(row - 1)
            output_indices = torch.tensor(
                rows,
                dtype=torch.int32,
                device=self._engine.device,
            )
            return [
                (
                    request_indices,
                    self.finish_prefill(
                        prefill_input,
                        state,
                        output_indices=output_indices,
                        request_indices=request_indices,
                    ),
                )
            ]

        selected = set(request_indices)
        outputs = []
        for tile, tile_state in zip(
            state.tiles,
            state.tile_states,
            strict=True,
        ):
            terminal = [
                (request_index, local_row)
                for request_index, local_row in tile.terminal_rows
                if request_index in selected
            ]
            if not terminal:
                continue
            if tile_state is None:
                raise RuntimeError("layered prefill terminal tile has no model state")
            tile_requests = [request_index for request_index, _ in terminal]
            output_indices = torch.tensor(
                [local_row for _, local_row in terminal],
                dtype=torch.int32,
                device=self._engine.device,
            )
            outputs.append(
                (
                    tile_requests,
                    self._engine.finish_layer_group_prefill(
                        prefill_input.batch,
                        tile_state,
                        prefill_input.sample_args,
                        output_indices=output_indices,
                        request_indices=tile_requests,
                    ),
                )
            )
        if {index for indices, _ in outputs for index in indices} != selected:
            raise RuntimeError("layered prefill did not preserve every terminal request")
        return outputs


__all__ = [
    "LayeredExecutionAdapter",
    "LayeredGroupResult",
    "LayeredGroupRun",
]
