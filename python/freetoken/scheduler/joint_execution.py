from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from freetoken.core import Batch, Req
from freetoken.utils import init_logger

from .batch_composition import DecodeBatchSelector
from .forward import ForwardData, ForwardInput
from .prefill import PrefillManager
from .resident_wave import (
    ResidentFrontier,
    ResidentWaveAdmission,
    ResidentWaveState,
    abort_resident_member,
    admit_resident_frontiers,
    commit_resident_chunks,
    finish_resident_prefill,
    prepare_resident_frontier,
    resolve_group_zero_admission,
    schedule_resident_wave,
    write_and_filter,
)

if TYPE_CHECKING:
    from freetoken.engine import Engine
    from freetoken.models.blocks import LayerGroupState

    from .decode import DecodeManager
    from .table import TableManager


logger = init_logger(__name__)


class JointWaveExecutor:
    """Own the lifecycle of one group-resident mixed prefill wave.

    The scheduler remains responsible for receiving requests and choosing when
    to run. This executor owns joint admission, layer-group progression, output
    finalization, and abort marking.
    """

    def __init__(
        self,
        *,
        engine: Engine,
        prefill_manager: PrefillManager,
        decode_manager: DecodeManager,
        table_manager: TableManager,
        max_chunks: int,
        prepare_batch: Callable[[Batch], ForwardInput],
        report_prompt_admissions: Callable[[Batch], None],
        restore_linear_states: Callable[[Batch], None],
        free_req_resources: Callable[[Req], None],
    ) -> None:
        self._engine = engine
        self._prefill_manager = prefill_manager
        self._decode_manager = decode_manager
        self._table_manager = table_manager
        self._max_chunks = max_chunks
        self._prepare_batch = prepare_batch
        self._report_prompt_admissions = report_prompt_admissions
        self._restore_linear_states = restore_linear_states
        self._free_req_resources = free_req_resources
        self._decode_selector = DecodeBatchSelector()
        self._wave: ResidentWaveState | None = None
        self._staged_admission: ResidentWaveAdmission | None = None
        self._deferred_join_members: tuple[int, ...] | None = None

    @property
    def active(self) -> bool:
        return self._wave is not None

    def schedule_first_batch(self, token_budget: int) -> Batch | None:
        decode_batch = self._decode_manager.schedule_next_batch()
        decode_reqs, _ = self._decode_selector.select(decode_batch, token_budget)
        scheduled = schedule_resident_wave(
            decode_reqs,
            prefill_manager=self._prefill_manager,
            token_budget=token_budget,
            soft_chunk_cap=self._max_chunks,
            max_frontier_chunks=None,
            deferred_join_members=self._deferred_join_members,
        )
        self._staged_admission = scheduled.admission
        self._deferred_join_members = scheduled.deferred_join_members
        return scheduled.batch

    def begin_wave(self, first_batch: Batch, token_budget: int) -> None:
        """Prepare the currently visible complete requests for one resident wave."""
        if self._wave is not None:
            raise RuntimeError("a joint prefill wave is already active")
        admission = self._staged_admission
        self._staged_admission = None
        if admission is None or not first_batch.prefill_reqs:
            raise RuntimeError("joint wave has no staged prefill admission")

        first = self._prepare_frontier(first_batch)

        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        group_size = moe_cache.effective_prefill_group_size
        if group_size < 1:
            raise RuntimeError("joint batching has no resident expert-layer capacity")
        self._wave = ResidentWaveState(
            admission=admission,
            num_layers=self._engine.layer_group_num_layers,
            group_size=group_size,
            frontiers=[first],
            layer_prepares_at_start=moe_cache.prefill_layer_prepares,
        )
        admission.record_frontier(0, first)
        admit_resident_frontiers(
            self._wave,
            prefill_manager=self._prefill_manager,
            token_budget=token_budget,
            prepare_frontier=self._prepare_frontier,
        )
        resolve_group_zero_admission(
            self._wave,
            prefill_manager=self._prefill_manager,
            has_decode=first_batch.has_decode,
        )

    def _prepare_frontier(self, batch: Batch) -> ResidentFrontier:
        decode_rows = sum(req.extend_len for req in batch.decode_reqs)
        return prepare_resident_frontier(
            batch,
            prepare_batch=self._prepare_batch,
            report_prompt_admissions=self._report_prompt_admissions,
            table_manager=self._table_manager,
            prefill_manager=self._prefill_manager,
            prefill_row_offset=decode_rows,
        )

    def prepare_step(self, token_budget: int) -> None:
        wave = self._wave
        if wave is None or wave.current_layer != 0 or wave.admission_complete:
            return
        wave.admission.refresh_members(self._prefill_manager)
        pending_members = wave.admission.pending_member_uids(self._prefill_manager)
        if wave.awaiting_join_boundary and not pending_members:
            wave.admission.freeze()
            wave.admission_complete = True
            wave.awaiting_join_boundary = False
            return
        if pending_members:
            wave.awaiting_join_boundary = False
            admit_resident_frontiers(
                wave,
                prefill_manager=self._prefill_manager,
                token_budget=token_budget,
                prepare_frontier=self._prepare_frontier,
            )
        first_has_decode = wave.frontiers[0].forward_input.batch.has_decode
        resolve_group_zero_admission(
            wave,
            prefill_manager=self._prefill_manager,
            has_decode=first_has_decode,
        )

    def advance_step(self) -> list[ForwardData]:
        """Advance newly admitted frontiers, then close a frozen resident group."""
        wave = self._wave
        if wave is None or wave.done:
            raise RuntimeError("joint prefill wave has no group to advance")

        frontiers = wave.frontiers[wave.next_frontier :]
        if frontiers:
            self._advance_group_states(wave, frontiers)
            wave.next_frontier = len(wave.frontiers)
        if wave.current_layer == 0 and not wave.admission_complete:
            return []
        if not wave.resident_group_active:
            raise RuntimeError("joint resident group completed without active cache state")
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        moe_cache.end_prefill_group()
        wave.resident_group_active = False
        wave.finish_group()
        wave.next_frontier = 0
        if not wave.done:
            return []

        outputs = self._finish_wave(wave)
        self._log_completed_wave(wave)
        self._wave = None
        return outputs

    def _advance_group_states(
        self,
        wave: ResidentWaveState,
        frontiers: list[ResidentFrontier],
    ) -> None:
        start_layer = wave.current_layer
        end_layer = wave.current_group_end
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None

        if not wave.resident_group_active:
            moe_cache.begin_resident_prefill_group(start_layer, end_layer)
            wave.resident_group_active = True
        try:
            for frontier in frontiers:
                forward_input = frontier.forward_input
                if frontier.state is None:
                    if start_layer != 0:
                        raise RuntimeError(
                            "joint state reached a later group without embedding"
                        )
                    self._restore_linear_states(forward_input.batch)
                    frontier.state = self._engine.begin_layer_group_prefill(
                        forward_input.batch
                    )
                if frontier.state.next_layer != start_layer:
                    raise RuntimeError(
                        f"joint state is at layer {frontier.state.next_layer}, "
                        f"group starts at {start_layer}"
                    )
                frontier.state = self._engine.advance_layer_group_prefill(
                    forward_input.batch,
                    frontier.state,
                    end_layer,
                )
        except Exception:
            if wave.resident_group_active:
                moe_cache.end_prefill_group()
                wave.resident_group_active = False
            raise

    def _finish_wave(self, wave: ResidentWaveState) -> list[ForwardData]:
        """Commit every UID and publish decode plus all terminal prompt rows."""
        first = wave.frontiers[0]
        commit_resident_chunks(wave)

        outputs: list[ForwardData] = []
        first_input = first.forward_input
        if first_input.batch.has_decode:
            decode_end = first_input.batch.decode_size
            output = self._engine.finish_layer_group_prefill(
                first_input.batch,
                self._require_state(first),
                first_input.sample_args,
                request_slice=slice(0, decode_end),
            )
            decode_input = self._output_view(
                first_input,
                slice(0, decode_end),
                decode_size=decode_end,
            )
            write_and_filter(
                decode_input,
                output.next_tokens_gpu,
                self._table_manager,
                self._decode_manager,
            )
            outputs.append((decode_input, output))
        outputs.extend(
            finish_resident_prefill(
                wave,
                engine=self._engine,
                decode_manager=self._decode_manager,
                table_manager=self._table_manager,
                free_req_resources=self._free_req_resources,
                commit_chunks=False,
            )
        )
        return outputs

    @staticmethod
    def _require_state(frontier: ResidentFrontier) -> LayerGroupState:
        if frontier.state is None:
            raise RuntimeError("joint prefill frontier completed without model state")
        return frontier.state

    @staticmethod
    def _output_view(
        forward_input: ForwardInput,
        request_slice: slice,
        *,
        decode_size: int,
    ) -> ForwardInput:
        selected_reqs = forward_input.batch.reqs[request_slice]
        batch = Batch(reqs=list(selected_reqs), decode_size=decode_size)
        write_tuple = (
            forward_input.write_tuple[0][request_slice],
            forward_input.write_tuple[1][request_slice],
        )
        return ForwardInput(
            batch=batch,
            sample_args=forward_input.sample_args,
            input_tuple=forward_input.input_tuple,
            write_tuple=write_tuple,
        )

    def _log_completed_wave(self, wave: ResidentWaveState) -> None:
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        layer_prepares = (
            moe_cache.prefill_layer_prepares - wave.layer_prepares_at_start
        )
        groups = (wave.num_layers + wave.group_size - 1) // wave.group_size
        logger.info_rank0(
            "Joint wave complete: "
            f"chunks={wave.admission.total_chunks}, "
            f"wave_reqs={len(wave.admission.members)}, "
            f"frontier_batches={len(wave.frontiers)}, groups={groups}, "
            f"effective_group_size={wave.group_size}, "
            f"prefill_layer_prepares={layer_prepares}"
        )

    def abort(self, uid: int) -> Req | None:
        """Mark active joint work for cleanup after the final group drains."""
        wave = self._wave
        if wave is None:
            return None
        for frontier in wave.frontiers:
            for req in frontier.forward_input.batch.decode_reqs:
                if req.uid == uid:
                    req.aborted = True
                    return req
        return abort_resident_member(wave, uid)


__all__ = ["JointWaveExecutor"]
