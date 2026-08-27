from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from freetoken.core import Batch, Req
from freetoken.models.blocks import LayerGroupState
from freetoken.utils import init_logger

from .batch_composition import compose_mixed_batch
from .forward import ForwardData, ForwardInput
from .layer_group_execution import (
    batch_view,
    decode_view,
    merge_states,
    prefill_view,
    split_state,
)
from .prefill import ChunkedReq, PrefillManager
from .resident_wave import (
    ResidentFrontier,
    ResidentWaveAdmission,
    ResidentWaveState,
    abort_resident_member,
    admit_resident_frontiers,
    finish_resident_prefill,
    prepare_resident_frontier,
    resolve_group_zero_admission,
    schedule_resident_wave,
    write_and_filter,
)

if TYPE_CHECKING:
    from freetoken.engine import Engine

    from .decode import DecodeManager
    from .table import TableManager


logger = init_logger(__name__)


@dataclass
class _PipelineWave(ResidentWaveState):
    resident_groups: int = 0
    chunk_group_steps: int = 0
    frontier_group_forwards: int = 0
    iterations: int = 0
    decode_iterations: int = 0
    cross_group_prefetches: int = 0
    deferred_cross_group_prefetches: int = 0
    close_group_without_frontier: bool = False


class LayeredPipelineExecutor:
    """Replay every prompt chunk through one resident group before moving on."""

    def __init__(
        self,
        *,
        engine: Engine,
        prefill_manager: PrefillManager,
        decode_manager: DecodeManager,
        table_manager: TableManager,
        max_wave_chunks: int,
        chunks_per_iteration: int,
        prepare_batch: Callable[[Batch], ForwardInput],
        prepare_mixed_batch: Callable[[Batch, Batch], ForwardInput],
        build_execution_input: Callable[[Batch], ForwardInput],
        report_prompt_admissions: Callable[[Batch], None],
        restore_linear_states: Callable[[Batch], None],
        free_req_resources: Callable[[Req], None],
    ) -> None:
        self._engine = engine
        self._prefill_manager = prefill_manager
        self._decode_manager = decode_manager
        self._table_manager = table_manager
        self._max_wave_chunks = max_wave_chunks
        self._chunks_per_iteration = chunks_per_iteration
        self._prepare_batch = prepare_batch
        self._prepare_mixed_batch = prepare_mixed_batch
        self._build_execution_input = build_execution_input
        self._report_prompt_admissions = report_prompt_admissions
        self._restore_linear_states = restore_linear_states
        self._free_req_resources = free_req_resources
        self._wave: _PipelineWave | None = None
        self._staged_admission: ResidentWaveAdmission | None = None
        self._deferred_join_members: tuple[int, ...] | None = None
        self._decode_input: ForwardInput | None = None
        self._group_input: ForwardInput | None = None
        self._iteration_frontiers: list[ResidentFrontier] = []

    @property
    def active(self) -> bool:
        return self._wave is not None

    def schedule_first_batch(self, token_budget: int) -> Batch | None:
        decode_batch = self._decode_manager.schedule_next_batch()
        decode_reqs = list(decode_batch.reqs) if decode_batch is not None else []
        scheduled = schedule_resident_wave(
            decode_reqs,
            prefill_manager=self._prefill_manager,
            token_budget=token_budget,
            soft_chunk_cap=self._max_wave_chunks,
            max_frontier_chunks=self._chunks_per_iteration,
            deferred_join_members=self._deferred_join_members,
        )
        self._staged_admission = scheduled.admission
        self._deferred_join_members = scheduled.deferred_join_members
        return scheduled.batch

    def begin_wave(self, first_batch: Batch, token_budget: int) -> None:
        if self._wave is not None:
            raise RuntimeError("a layered pipeline wave is already active")
        admission = self._staged_admission
        self._staged_admission = None
        if admission is None or not first_batch.prefill_reqs:
            raise RuntimeError("layered pipeline wave has no staged prefill admission")

        prepared = self._prepare_batch(first_batch)
        self._report_prompt_admissions(first_batch)
        first_batch.input_ids = self._table_manager.token_pool[prepared.input_tuple]

        if first_batch.has_decode:
            first_prefill_batch = batch_view(
                first_batch, first_batch.decode_size, len(first_batch.reqs), 0
            )
            first_frontier = self._prepare_first_frontier_view(
                prepared, first_prefill_batch
            )
        else:
            first_prefill_batch = first_batch
            first_frontier = ResidentFrontier(forward_input=prepared)
            for req in first_prefill_batch.prefill_reqs:
                self._prefill_manager.reserve_layered_continuation(req)
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        group_size = moe_cache.effective_prefill_group_size
        if group_size < 1:
            raise RuntimeError("layered pipeline has no resident expert-layer capacity")
        wave = _PipelineWave(
            admission=admission,
            num_layers=self._engine.layer_group_num_layers,
            group_size=group_size,
            frontiers=[first_frontier],
            layer_prepares_at_start=moe_cache.prefill_layer_prepares,
        )
        admission.record_frontier(0, first_frontier)
        admission.refresh_members(self._prefill_manager)
        self._wave = wave
        self._iteration_frontiers = [first_frontier]
        self._iteration_frontiers.extend(
            admit_resident_frontiers(
                wave,
                prefill_manager=self._prefill_manager,
                token_budget=token_budget,
                prepare_frontier=self._prepare_frontier,
                max_chunks=(
                    self._chunks_per_iteration - first_frontier.chunk_count
                ),
            )
        )
        resolve_group_zero_admission(
            wave,
            prefill_manager=self._prefill_manager,
            has_decode=first_batch.has_decode,
        )
        self._decode_input = (
            decode_view(prepared, self._table_manager)
            if first_batch.has_decode
            else None
        )
        self._group_input = prepared

    def _prepare_frontier(self, batch: Batch) -> ResidentFrontier:
        return prepare_resident_frontier(
            batch,
            prepare_batch=self._prepare_batch,
            report_prompt_admissions=self._report_prompt_admissions,
            table_manager=self._table_manager,
            prefill_manager=self._prefill_manager,
        )

    def _prepare_first_frontier_view(
        self, mixed_input: ForwardInput, batch: Batch
    ) -> ResidentFrontier:
        prepared = prefill_view(mixed_input, self._table_manager, batch)
        for req in batch.prefill_reqs:
            self._prefill_manager.reserve_layered_continuation(req)
        return ResidentFrontier(
            forward_input=prepared,
            attention_metadata_ready=False,
        )

    def _ensure_chunk_attention_metadata(
        self, frontier: ResidentFrontier
    ) -> ForwardInput:
        if not frontier.attention_metadata_ready:
            frontier.forward_input = self._build_execution_input(
                frontier.forward_input.batch
            )
            frontier.attention_metadata_ready = True
        return frontier.forward_input

    def prepare_step(self, token_budget: int) -> None:
        wave = self._require_wave()
        if (
            self._group_input is not None
            or self._decode_input is not None
            or self._iteration_frontiers
        ):
            raise RuntimeError("a layered pipeline iteration is already staged")

        decode_batch = self._decode_manager.schedule_next_batch()
        decode_reqs = list(decode_batch.reqs) if decode_batch is not None else []
        if wave.current_layer == 0:
            self._prepare_group_zero_iteration(wave, decode_reqs, token_budget)
            return

        selected = self._select_frontiers(wave)
        if not selected:
            raise RuntimeError("layered pipeline replay has no frontier to advance")
        self._stage_existing_frontiers(selected, decode_reqs)

    def _select_frontiers(self, wave: _PipelineWave) -> list[ResidentFrontier]:
        selected: list[ResidentFrontier] = []
        chunks = 0
        for frontier in wave.frontiers[wave.next_frontier :]:
            if chunks + frontier.chunk_count > self._chunks_per_iteration:
                break
            selected.append(frontier)
            chunks += frontier.chunk_count
        return selected

    def _prepare_group_zero_iteration(
        self,
        wave: _PipelineWave,
        decode_reqs: list[Req],
        token_budget: int,
    ) -> None:
        wave.admission.refresh_members(self._prefill_manager)
        pending_uids = wave.admission.pending_member_uids(self._prefill_manager)
        if wave.awaiting_join_boundary and not pending_uids:
            wave.admission.freeze()
            wave.admission_complete = True
            wave.awaiting_join_boundary = False
            wave.close_group_without_frontier = True
            if decode_reqs:
                decode_batch = Batch(reqs=decode_reqs, decode_size=len(decode_reqs))
                self._decode_input = self._prepare_batch(decode_batch)
            return
        if pending_uids:
            wave.awaiting_join_boundary = False
        max_reqs = min(self._chunks_per_iteration, len(pending_uids))
        prefill_batch = self._prefill_manager.schedule_next_batch(
            token_budget * max_reqs,
            chunk_token_limit=token_budget,
            allowed_uids=pending_uids,
            max_reqs=max_reqs,
        )
        if prefill_batch is None:
            if not pending_uids:
                wave.admission.freeze()
                wave.admission_complete = True
                wave.close_group_without_frontier = True
            if decode_reqs:
                decode_batch = Batch(reqs=decode_reqs, decode_size=len(decode_reqs))
                self._decode_input = self._prepare_batch(decode_batch)
            return

        staged_batch = compose_mixed_batch(decode_reqs, prefill_batch)
        assert staged_batch is not None
        prepared = self._prepare_batch(staged_batch)
        self._report_prompt_admissions(staged_batch)
        staged_batch.input_ids = self._table_manager.token_pool[prepared.input_tuple]
        if decode_reqs:
            frontier_batch = batch_view(
                staged_batch,
                staged_batch.decode_size,
                len(staged_batch.reqs),
                0,
            )
            first = self._prepare_first_frontier_view(prepared, frontier_batch)
            allocation_batch = batch_view(
                staged_batch,
                0,
                staged_batch.decode_size,
                staged_batch.decode_size,
            )
            self._decode_input = decode_view(
                prepared, self._table_manager, allocation_batch
            )
        else:
            first = ResidentFrontier(forward_input=prepared)
            for req in prefill_batch.prefill_reqs:
                self._prefill_manager.reserve_layered_continuation(req)

        frontier_index = len(wave.frontiers)
        wave.frontiers.append(first)
        wave.admission.record_frontier(frontier_index, first)
        selected = [first]
        selected.extend(
            admit_resident_frontiers(
                wave,
                prefill_manager=self._prefill_manager,
                token_budget=token_budget,
                prepare_frontier=self._prepare_frontier,
                max_chunks=self._chunks_per_iteration - first.chunk_count,
            )
        )
        resolve_group_zero_admission(
            wave,
            prefill_manager=self._prefill_manager,
            has_decode=bool(decode_reqs),
        )
        self._iteration_frontiers = selected
        self._group_input = prepared

    def _stage_existing_frontiers(
        self,
        selected: list[ResidentFrontier],
        decode_reqs: list[Req],
    ) -> None:
        self._iteration_frontiers = selected
        first_prefill = selected[0].forward_input.batch
        for frontier in selected[1:]:
            self._ensure_chunk_attention_metadata(frontier)
        if not decode_reqs:
            first_input = self._ensure_chunk_attention_metadata(selected[0])
            first_input.batch.input_ids = self._table_manager.token_pool[
                first_input.input_tuple
            ]
            self._group_input = first_input
            return

        allocation_batch = Batch(reqs=decode_reqs, decode_size=len(decode_reqs))
        mixed_batch = compose_mixed_batch(decode_reqs, first_prefill)
        assert mixed_batch is not None
        self._group_input = self._prepare_mixed_batch(allocation_batch, mixed_batch)
        self._decode_input = decode_view(
            self._group_input, self._table_manager, allocation_batch
        )

    def advance_step(self) -> list[ForwardData]:
        wave = self._require_wave()
        selected = self._iteration_frontiers
        if not selected:
            if wave.close_group_without_frontier:
                return self._close_group_after_join_boundary(wave)
            return self._advance_decode_only_iteration(wave)
        group_input = self._group_input
        if group_input is None:
            raise RuntimeError("layered pipeline iteration was not prepared")
        start_layer = wave.current_layer
        end_layer = wave.current_group_end
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        if not wave.resident_group_active:
            moe_cache.begin_resident_prefill_group(start_layer, end_layer)
            wave.resident_group_active = True

        combined_state, decode_state = self._begin_iteration_state(
            wave, selected, group_input
        )
        next_frontier = wave.next_frontier + len(selected)
        group_complete = (
            wave.admission_complete and next_frontier == len(wave.frontiers)
        )
        next_group_prefetched = self._try_prefetch_next_group(
            wave, group_complete
        )
        decode_state = self._advance_selected_frontiers(
            wave,
            selected,
            group_input,
            combined_state,
            decode_state,
        )

        wave.next_frontier = next_frontier
        wave.chunk_group_steps += sum(frontier.chunk_count for frontier in selected)
        wave.frontier_group_forwards += len(selected)
        wave.iterations += 1
        if group_complete:
            self._complete_resident_group(wave, next_group_prefetched)

        outputs = self._finish_iteration_decode(wave, decode_state, end_layer)
        self._decode_input = None
        self._group_input = None
        self._iteration_frontiers = []
        if wave.done:
            outputs.extend(self._finish_wave(wave))
            self._log_completed_wave(wave)
            self._wave = None
        return outputs

    def _begin_iteration_state(
        self,
        wave: _PipelineWave,
        selected: list[ResidentFrontier],
        group_input: ForwardInput,
    ) -> tuple[LayerGroupState, LayerGroupState | None]:
        start_layer = wave.current_layer
        first = selected[0]
        decode_state: LayerGroupState | None = None
        if start_layer == 0:
            if first.state is not None:
                raise RuntimeError("group-zero chunk was already embedded")
            self._restore_linear_states(group_input.batch)
            combined_state = self._engine.begin_layer_group_prefill(group_input.batch)
        else:
            if first.state is None or first.state.next_layer != start_layer:
                raise RuntimeError("prefill state is not at the active pipeline group")
            if self._decode_input is not None:
                decode_input = self._decode_input
                decode_state = self._engine.begin_layer_group_decode(
                    decode_input.batch, start_layer
                )
                combined_state = merge_states(decode_state, first.state)
            else:
                combined_state = first.state

        for chunk in selected[1:]:
            if chunk.state is None:
                if start_layer != 0:
                    raise RuntimeError("a continuation chunk missed pipeline embedding")
                self._restore_linear_states(chunk.forward_input.batch)
                chunk.state = self._engine.begin_layer_group_prefill(
                    chunk.forward_input.batch
                )
        return combined_state, decode_state

    def _try_prefetch_next_group(
        self, wave: _PipelineWave, group_complete: bool
    ) -> bool:
        next_start = wave.current_group_end
        next_end = min(next_start + wave.group_size, wave.num_layers)
        if (
            not group_complete
            or next_start >= wave.num_layers
            or self._decode_input is None
        ):
            return False
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        prefetched = moe_cache.try_prefetch_next_resident_group(next_start, next_end)
        if prefetched:
            wave.cross_group_prefetches += 1
        else:
            wave.deferred_cross_group_prefetches += 1
        return prefetched

    def _advance_selected_frontiers(
        self,
        wave: _PipelineWave,
        selected: list[ResidentFrontier],
        group_input: ForwardInput,
        combined_state: LayerGroupState,
        decode_state: LayerGroupState | None,
    ) -> LayerGroupState | None:
        start_layer = wave.current_layer
        end_layer = wave.current_group_end
        first = selected[0]
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        try:
            combined_state = self._engine.advance_layer_group_prefill(
                group_input.batch, combined_state, end_layer
            )
            if group_input.batch.has_decode:
                decode_rows = sum(req.extend_len for req in group_input.batch.decode_reqs)
                decode_state, first.state = split_state(combined_state, decode_rows)
            else:
                first.state = combined_state

            for chunk in selected[1:]:
                assert chunk.state is not None
                if chunk.state.next_layer != start_layer:
                    raise RuntimeError("continuation state is not at the active pipeline group")
                chunk.state = self._engine.advance_layer_group_prefill(
                    chunk.forward_input.batch,
                    chunk.state,
                    end_layer,
                )
        except Exception:
            if wave.resident_group_active:
                moe_cache.end_prefill_group()
                wave.resident_group_active = False
            raise
        return decode_state

    def _complete_resident_group(
        self, wave: _PipelineWave, next_group_prefetched: bool
    ) -> None:
        start_layer = wave.current_layer
        next_start = wave.current_group_end
        next_end = min(next_start + wave.group_size, wave.num_layers)
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        moe_cache.end_prefill_group()
        wave.resident_group_active = False
        wave.resident_groups += 1
        wave.finish_group()
        wave.next_frontier = 0
        if start_layer == 0 and not wave.done:
            self._repack_frontiers_for_replay(wave)
        if next_group_prefetched:
            moe_cache.promote_prefetched_resident_group(next_start, next_end)
            wave.resident_group_active = True

    def _finish_iteration_decode(
        self,
        wave: _PipelineWave,
        decode_state: LayerGroupState | None,
        resident_group_end: int,
    ) -> list[ForwardData]:
        outputs: list[ForwardData] = []
        if self._decode_input is not None:
            if decode_state is None:
                raise RuntimeError("mixed pipeline group did not produce decode state")
            if resident_group_end < wave.num_layers:
                decode_state = self._engine.advance_layer_group_decode(
                    self._decode_input.batch,
                    decode_state,
                    wave.num_layers,
                )
            output = self._engine.finish_layer_group_prefill(
                self._decode_input.batch,
                decode_state,
                self._decode_input.sample_args,
            )
            write_and_filter(
                self._decode_input,
                output.next_tokens_gpu,
                self._table_manager,
                self._decode_manager,
            )
            outputs.append((self._decode_input, output))
            wave.decode_iterations += 1
        return outputs

    def _repack_frontiers_for_replay(self, wave: _PipelineWave) -> None:
        """Canonicalize late group-zero arrivals before replaying later groups.

        Group zero must respect when requests actually arrive.  Once membership is
        frozen, chunks from different requests are independent except for each
        request's own ordinal, so later groups can replay one ordinal-aligned ragged
        frontier instead of preserving arrival-shaped partial batches.
        """
        chunks_by_uid: dict[
            int,
            list[tuple[Req, torch.Tensor, torch.Tensor | None]],
        ] = {uid: [] for uid in wave.admission.members}
        next_layer: int | None = None
        for frontier in wave.frontiers:
            if frontier.forward_input.batch.decode_size != 0:
                raise RuntimeError("pipeline replay frontier unexpectedly contains decode")
            state = frontier.state
            if state is None:
                raise RuntimeError("pipeline replay frontier has no completed group-zero state")
            if next_layer is None:
                next_layer = state.next_layer
            elif state.next_layer != next_layer:
                raise RuntimeError("pipeline replay frontiers are at different layers")

            row = 0
            for req in frontier.forward_input.batch.prefill_reqs:
                end = row + req.extend_len
                residual = state.residual[row:end] if state.residual is not None else None
                chunks_by_uid[req.uid].append((req, state.hidden[row:end], residual))
                row = end
            if row != state.hidden.shape[0]:
                raise RuntimeError("pipeline replay state rows do not match its requests")

        ordinal_count = max((len(chunks) for chunks in chunks_by_uid.values()), default=0)
        if ordinal_count == 0:
            return
        if next_layer is None:
            raise RuntimeError("pipeline replay has no completed layer state")

        canonical_entries: list[
            list[tuple[Req, torch.Tensor, torch.Tensor | None]]
        ] = []
        for ordinal in range(ordinal_count):
            entries = [
                chunks_by_uid[uid][ordinal]
                for uid in wave.admission.members
                if ordinal < len(chunks_by_uid[uid])
            ]
            canonical_entries.extend(
                entries[offset : offset + self._chunks_per_iteration]
                for offset in range(0, len(entries), self._chunks_per_iteration)
            )
        if len(canonical_entries) >= len(wave.frontiers):
            return

        repacked: list[ResidentFrontier] = []
        for entries in canonical_entries:
            reqs = [req for req, _, _ in entries]
            batch = Batch(reqs=reqs, decode_size=0)
            forward_input = self._build_execution_input(batch)

            hidden_parts = [hidden for _, hidden, _ in entries]
            hidden = (
                hidden_parts[0]
                if len(hidden_parts) == 1
                else torch.cat(hidden_parts, dim=0)
            )
            residual_parts = [residual for _, _, residual in entries]
            if all(residual is None for residual in residual_parts):
                residual = None
            elif any(residual is None for residual in residual_parts):
                raise RuntimeError("pipeline replay residual states do not match")
            else:
                materialized = [
                    residual for residual in residual_parts if residual is not None
                ]
                residual = (
                    materialized[0]
                    if len(materialized) == 1
                    else torch.cat(materialized, dim=0)
                )
            repacked.append(
                ResidentFrontier(
                    forward_input=forward_input,
                    state=LayerGroupState(hidden, residual, next_layer),
                )
            )

        for member in wave.admission.members.values():
            member.terminal_frontier = None
            member.terminal_request_index = None
            member.terminal_output_row = None
        for frontier_index, frontier in enumerate(repacked):
            for request_index, (req, output_row) in enumerate(
                zip(frontier.forward_input.batch.prefill_reqs, frontier.last_rows())
            ):
                if isinstance(req, ChunkedReq):
                    continue
                member = wave.admission.members[req.uid]
                member.terminal_frontier = frontier_index
                member.terminal_request_index = request_index
                member.terminal_output_row = output_row

        wave.frontiers = repacked

    def _close_group_after_join_boundary(
        self, wave: _PipelineWave
    ) -> list[ForwardData]:
        """Freeze an idle group-zero join window after one final decode opportunity."""
        outputs = self._advance_decode_only_iteration(wave)
        start_layer = wave.current_layer
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        if not wave.resident_group_active:
            raise RuntimeError("layered pipeline join boundary lost its resident group")
        moe_cache.end_prefill_group()
        wave.resident_group_active = False
        wave.resident_groups += 1
        wave.finish_group()
        wave.next_frontier = 0
        wave.close_group_without_frontier = False
        if start_layer == 0 and not wave.done:
            self._repack_frontiers_for_replay(wave)
        if wave.done:
            outputs.extend(self._finish_wave(wave))
            self._log_completed_wave(wave)
            self._wave = None
        return outputs

    def _advance_decode_only_iteration(
        self, wave: _PipelineWave
    ) -> list[ForwardData]:
        decode_input = self._decode_input
        if decode_input is None:
            return []
        decode_input.batch.input_ids = self._table_manager.token_pool[
            decode_input.input_tuple
        ]
        state = self._engine.begin_layer_group_decode(
            decode_input.batch,
            wave.num_layers,
        )
        output = self._engine.finish_layer_group_prefill(
            decode_input.batch,
            state,
            decode_input.sample_args,
        )
        write_and_filter(
            decode_input,
            output.next_tokens_gpu,
            self._table_manager,
            self._decode_manager,
        )
        self._decode_input = None
        wave.iterations += 1
        wave.decode_iterations += 1
        return [(decode_input, output)]

    def _finish_wave(self, wave: _PipelineWave) -> list[ForwardData]:
        return finish_resident_prefill(
            wave,
            engine=self._engine,
            decode_manager=self._decode_manager,
            table_manager=self._table_manager,
            free_req_resources=self._free_req_resources,
        )

    def _log_completed_wave(self, wave: _PipelineWave) -> None:
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        layer_prepares = moe_cache.prefill_layer_prepares - wave.layer_prepares_at_start
        logger.info_rank0(
            "Layered pipeline wave complete: "
            f"chunks={wave.admission.total_chunks}, "
            f"wave_reqs={len(wave.admission.members)}, "
            f"frontier_batches={len(wave.frontiers)}, "
            f"resident_groups={wave.resident_groups}, "
            f"chunk_group_steps={wave.chunk_group_steps}, "
            f"frontier_group_forwards={wave.frontier_group_forwards}, "
            f"iterations={wave.iterations}, "
            f"decode_iterations={wave.decode_iterations}, "
            f"prefill_layer_prepares={layer_prepares}, "
            f"cross_group_prefetches={wave.cross_group_prefetches}, "
            "deferred_cross_group_prefetches="
            f"{wave.deferred_cross_group_prefetches}"
        )

    def abort(self, uid: int) -> Req | None:
        wave = self._wave
        if wave is None:
            return None
        return abort_resident_member(wave, uid)

    def _require_wave(self) -> _PipelineWave:
        if self._wave is None or self._wave.done:
            raise RuntimeError("no layered pipeline wave is active")
        return self._wave
__all__ = ["LayeredPipelineExecutor"]
