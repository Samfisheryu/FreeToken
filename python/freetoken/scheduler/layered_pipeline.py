from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from freetoken.core import Batch, Req
from freetoken.utils import init_logger

from .batch_composition import compose_mixed_batch
from .forward import ForwardData, ForwardInput
from .prefill import ChunkedReq, PrefillManager
from .resident_wave import (
    ResidentWaveAdmission,
    abort_resident_admission,
    request_output_view,
    write_and_filter,
)

if TYPE_CHECKING:
    from freetoken.engine import Engine
    from freetoken.engine.layered_execution import PrefillExecutionSession

    from .decode import DecodeManager
    from .table import TableManager


logger = init_logger(__name__)


@dataclass
class _LayeredPipelineWave:
    """One ragged prompt batch advancing exactly one resident group per step."""

    admission: ResidentWaveAdmission
    prefill_input: ForwardInput
    cache_session: object
    prefill_execution: PrefillExecutionSession | None
    state: object | None = None
    current_stage: int = 0
    wave_complete: bool = False
    resident_groups: int = 0
    group_forwards: int = 0
    iterations: int = 0
    decode_iterations: int = 0

    @property
    def done(self) -> bool:
        return self.wave_complete


class LayeredPipelineExecutor:
    """Advance one complete ragged prefill wave by one layer group per iteration."""

    def __init__(
        self,
        *,
        engine: Engine,
        prefill_manager: PrefillManager,
        decode_manager: DecodeManager,
        table_manager: TableManager,
        max_wave_chunks: int,
        prepare_batch: Callable[[Batch], ForwardInput],
        prepare_allocation_only_batch: Callable[[Batch], ForwardInput],
        prepare_decode_batch: Callable[[Batch], ForwardInput],
        open_prefill_execution: Callable[[Batch], PrefillExecutionSession | None],
        report_prompt_admissions: Callable[[Batch], None],
        free_req_resources: Callable[[Req], None],
    ) -> None:
        self._engine = engine
        self._prefill_manager = prefill_manager
        self._decode_manager = decode_manager
        self._table_manager = table_manager
        self._max_wave_chunks = max_wave_chunks
        self._prepare_batch = prepare_batch
        self._prepare_allocation_only_batch = prepare_allocation_only_batch
        self._prepare_decode_batch = prepare_decode_batch
        self._open_prefill_execution = open_prefill_execution
        self._report_prompt_admissions = report_prompt_admissions
        self._free_req_resources = free_req_resources
        self._execution = engine.layered_execution_adapter
        self._wave: _LayeredPipelineWave | None = None
        self._staged_admission: ResidentWaveAdmission | None = None
        self._decode_input: ForwardInput | None = None
        self._group_input: ForwardInput | None = None
        self._current_prefill_input: ForwardInput | None = None

    @property
    def active(self) -> bool:
        return self._wave is not None

    def schedule_first_batch(self, token_budget: int) -> Batch | None:
        """Freeze one FIFO wave before its first group reaches the model."""
        decode_batch = self._decode_manager.schedule_next_batch()
        decode_reqs = list(decode_batch.reqs) if decode_batch is not None else []

        admission = ResidentWaveAdmission(self._max_wave_chunks, token_budget)
        admission.refresh_members(self._prefill_manager)
        prefill_batch = self._prefill_manager.schedule_full_prefill_batch(
            admission.uids,
            max_reqs=len(admission.members),
        )
        if prefill_batch is None:
            self._staged_admission = None
            return compose_mixed_batch(decode_reqs, None)

        admitted_uids = {req.uid for req in prefill_batch.prefill_reqs}
        admission.retain_uids(admitted_uids)
        admission.record_materialized_requests(prefill_batch.prefill_reqs)
        admission.freeze()
        self._staged_admission = admission
        return compose_mixed_batch(decode_reqs, prefill_batch)

    def begin_wave(self, first_batch: Batch, token_budget: int) -> None:
        if self._wave is not None:
            raise RuntimeError("a layered pipeline wave is already active")
        admission = self._staged_admission
        self._staged_admission = None
        if admission is None or not first_batch.prefill_reqs:
            raise RuntimeError("layered pipeline wave has no staged prefill admission")

        prefill_execution = self._open_prefill_execution(first_batch)
        try:
            uses_tiles = self._execution.uses_prefill_tiles(
                first_batch,
                token_budget,
            )
            allocation_only = uses_tiles or (
                first_batch.has_decode
                and self._execution.uses_separate_group_inputs
            )
            prepared = (
                self._prepare_allocation_only_batch(first_batch)
                if allocation_only
                else self._prepare_batch(first_batch)
            )
            self._report_prompt_admissions(first_batch)
            first_batch.input_ids = self._table_manager.token_pool[prepared.input_tuple]

            if first_batch.has_decode:
                prefill_input, prefill_metadata_ready = (
                    self._execution.prefill_view(
                        prepared,
                        prepare_metadata=not allocation_only,
                    )
                )
            else:
                prefill_input = prepared
                prefill_metadata_ready = True

            prefill_state = self._execution.initialize_prefill_state(
                prefill_input,
                token_budget,
                prefill_execution,
            )
            execution_prefill = self._execution.current_prefill_input(
                prefill_input,
                prefill_state,
            )
            if (
                execution_prefill is prefill_input
                and not prefill_metadata_ready
            ):
                self._execution.prepare_layered_prefill_input(prefill_input)
            self._current_prefill_input = execution_prefill
            if execution_prefill is prefill_input:
                if (
                    first_batch.has_decode
                    and self._execution.uses_separate_group_inputs
                ):
                    self._decode_input = self._execution.decode_view(
                        prepared,
                        source_metadata_ready=False,
                    )
                    self._group_input = execution_prefill
                else:
                    self._group_input = prepared
                    self._decode_input = (
                        self._execution.decode_view(prepared)
                        if first_batch.has_decode
                        else None
                    )
            elif first_batch.has_decode:
                self._decode_input = self._execution.decode_view(
                    prepared,
                    source_metadata_ready=False,
                )
                self._group_input = (
                    execution_prefill
                    if self._execution.uses_separate_group_inputs
                    else self._execution.compose_group_input(
                        self._decode_input,
                        execution_prefill,
                    )
                )
            else:
                self._group_input = execution_prefill
                self._decode_input = None

            for req in prefill_input.batch.prefill_reqs:
                self._prefill_manager.reserve_layered_continuation(req)

            cache_session = self._execution.open_resident_wave()
        except Exception:
            if prefill_execution is not None:
                prefill_execution.cancel()
            self._decode_input = None
            self._group_input = None
            self._current_prefill_input = None
            raise
        self._wave = _LayeredPipelineWave(
            admission=admission,
            prefill_input=prefill_input,
            cache_session=cache_session,
            prefill_execution=prefill_execution,
            state=prefill_state,
        )

    def prepare_step(self, token_budget: int) -> None:
        del token_budget
        wave = self._require_wave()
        if self._decode_input is not None or self._group_input is not None:
            raise RuntimeError("a layered pipeline iteration is already staged")

        execution_prefill = self._execution.current_prefill_input(
            wave.prefill_input,
            wave.state,
        )
        decode_batch = self._decode_manager.schedule_next_batch()
        if decode_batch is None:
            self._current_prefill_input = execution_prefill
            self._group_input = execution_prefill
            return

        self._current_prefill_input = execution_prefill
        decode_input = self._prepare_decode_batch(decode_batch)
        self._decode_input = decode_input
        self._group_input = (
            execution_prefill
            if self._execution.uses_separate_group_inputs
            else self._execution.compose_group_input(
                decode_input,
                execution_prefill,
            )
        )

    def advance_step(self) -> list[ForwardData]:
        wave = self._require_wave()
        group_input = self._group_input
        prefill_input = self._current_prefill_input
        if group_input is None or prefill_input is None:
            raise RuntimeError("layered pipeline iteration was not prepared")

        stage = wave.cache_session.begin(
            wave.current_stage,
            has_decode=self._decode_input is not None,
        )
        run = self._execution.begin_group(
            group_input,
            prefill_input,
            wave.state,
            self._decode_input,
            stage.start_layer,
        )
        if (
            self._decode_input is not None
            and self._execution.group_finishes_after_current_tile(wave.state)
            and wave.current_stage + 1 < wave.cache_session.stage_count
        ):
            wave.cache_session.hint_next(wave.current_stage + 1)
        try:
            result = self._execution.advance_group(
                group_input,
                prefill_input,
                wave.state,
                run,
                self._decode_input,
                stage.end_layer,
            )
        except Exception:
            wave.cache_session.cancel()
            if wave.prefill_execution is not None:
                wave.prefill_execution.cancel()
            raise

        wave.state = result.prefill_state
        wave.group_forwards += 1
        wave.iterations += 1
        wave.wave_complete = result.wave_complete
        if result.group_complete:
            next_layer = None if wave.wave_complete else stage.end_layer
            next_stage = wave.cache_session.complete(
                wave.current_stage,
                next_start_layer=next_layer,
            )
            wave.resident_groups += 1
            if not wave.wave_complete:
                if wave.prefill_execution is not None:
                    wave.prefill_execution.rewind()
                wave.current_stage = next_stage

        outputs = self._finish_iteration_decode(
            wave,
            result.decode_state,
            stage.end_layer,
        )
        self._decode_input = None
        self._group_input = None
        self._current_prefill_input = None
        if wave.done:
            outputs.extend(self._finish_wave(wave))
            if wave.prefill_execution is not None:
                wave.prefill_execution.close()
            wave.cache_session.close()
            self._log_completed_wave(wave)
            self._wave = None
        return outputs

    def _finish_iteration_decode(
        self,
        wave: _LayeredPipelineWave,
        decode_state: object | None,
        resident_group_end: int,
    ) -> list[ForwardData]:
        decode_input = self._decode_input
        if decode_input is None:
            return []
        if decode_state is None:
            raise RuntimeError("mixed layered pipeline group did not produce decode state")
        output = self._execution.finish_decode(
            decode_input,
            decode_state,
            resident_group_end,
        )
        write_and_filter(
            decode_input,
            output.next_tokens_gpu,
            self._table_manager,
            self._decode_manager,
        )
        wave.decode_iterations += 1
        return [(decode_input, output)]

    def _finish_wave(self, wave: _LayeredPipelineWave) -> list[ForwardData]:
        state = wave.state
        if (
            state is None
            or self._execution.state_stage(state) != self._execution.num_stages
        ):
            raise RuntimeError("layered pipeline wave completed without final model state")

        selected_requests: list[int] = []
        aborted_owners: list[Req] = []
        for request_index, req in enumerate(wave.prefill_input.batch.prefill_reqs):
            member = wave.admission.members[req.uid]
            if member.aborted or req.aborted:
                aborted_owners.append(req)
            elif isinstance(req, ChunkedReq):
                req.commit_prefill_kv()
            else:
                selected_requests.append(request_index)

        outputs: list[ForwardData] = []
        if selected_requests:
            for request_indices, output in self._execution.finish_prefill_wave(
                wave.prefill_input,
                state,
                selected_requests,
            ):
                output_input = request_output_view(
                    wave.prefill_input,
                    request_indices,
                )
                write_and_filter(
                    output_input,
                    output.next_tokens_gpu,
                    self._table_manager,
                    self._decode_manager,
                )
                outputs.append((output_input, output))

        if aborted_owners:
            self._engine.stream.synchronize()
            for owner in aborted_owners:
                self._free_req_resources(owner)
        wave.state = None
        return outputs

    def _log_completed_wave(self, wave: _LayeredPipelineWave) -> None:
        logger.info_rank0(
            "Layered pipeline wave complete: "
            f"reqs={len(wave.admission.members)}, "
            f"groups={wave.resident_groups}, "
            f"group_forwards={wave.group_forwards}, "
            f"iterations={wave.iterations}, "
            f"decode_iterations={wave.decode_iterations}, "
            f"prefill_layer_prepares={wave.cache_session.layer_prepares}"
        )

    def abort(self, uid: int) -> Req | None:
        wave = self._wave
        if wave is None:
            return None
        return abort_resident_admission(wave.admission, uid)

    def _require_wave(self) -> _LayeredPipelineWave:
        if self._wave is None or self._wave.done:
            raise RuntimeError("no layered pipeline wave is active")
        return self._wave


__all__ = ["LayeredPipelineExecutor"]
