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
    ResidentWaveAdmission,
    abort_resident_admission,
    request_output_view,
    write_and_filter,
)

if TYPE_CHECKING:
    from freetoken.engine import Engine

    from .decode import DecodeManager
    from .table import TableManager


logger = init_logger(__name__)


@dataclass
class _LayeredPipelineWave:
    """One ragged prompt batch advancing exactly one resident group per step."""

    admission: ResidentWaveAdmission
    prefill_input: ForwardInput
    attention_metadata_ready: bool
    num_layers: int
    group_size: int
    state: LayerGroupState | None = None
    current_layer: int = 0
    resident_group_active: bool = False
    resident_groups: int = 0
    group_forwards: int = 0
    iterations: int = 0
    decode_iterations: int = 0
    layer_prepares_at_start: int = 0

    @property
    def current_group_end(self) -> int:
        return min(self.current_layer + self.group_size, self.num_layers)

    @property
    def done(self) -> bool:
        return self.current_layer >= self.num_layers


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
        self._prepare_batch = prepare_batch
        self._prepare_mixed_batch = prepare_mixed_batch
        self._build_execution_input = build_execution_input
        self._report_prompt_admissions = report_prompt_admissions
        self._restore_linear_states = restore_linear_states
        self._free_req_resources = free_req_resources
        self._wave: _LayeredPipelineWave | None = None
        self._staged_admission: ResidentWaveAdmission | None = None
        self._decode_input: ForwardInput | None = None
        self._group_input: ForwardInput | None = None

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
        del token_budget
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
            prefill_batch = batch_view(
                first_batch,
                first_batch.decode_size,
                len(first_batch.reqs),
                0,
            )
            prefill_input = prefill_view(
                prepared,
                self._table_manager,
                prefill_batch,
            )
            attention_metadata_ready = False
            self._decode_input = decode_view(prepared, self._table_manager)
        else:
            prefill_input = prepared
            attention_metadata_ready = True
            self._decode_input = None

        for req in prefill_input.batch.prefill_reqs:
            self._prefill_manager.reserve_layered_continuation(req)

        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        group_size = moe_cache.effective_prefill_group_size
        if group_size < 1:
            raise RuntimeError("layered pipeline has no resident expert-layer capacity")
        self._wave = _LayeredPipelineWave(
            admission=admission,
            prefill_input=prefill_input,
            attention_metadata_ready=attention_metadata_ready,
            num_layers=self._engine.layer_group_num_layers,
            group_size=group_size,
            layer_prepares_at_start=moe_cache.prefill_layer_prepares,
        )
        self._group_input = prepared

    def prepare_step(self, token_budget: int) -> None:
        del token_budget
        wave = self._require_wave()
        if self._decode_input is not None or self._group_input is not None:
            raise RuntimeError("a layered pipeline iteration is already staged")

        decode_batch = self._decode_manager.schedule_next_batch()
        if decode_batch is None:
            if not wave.attention_metadata_ready:
                wave.prefill_input = self._build_execution_input(
                    wave.prefill_input.batch
                )
                wave.attention_metadata_ready = True
            self._group_input = wave.prefill_input
            return

        mixed_batch = compose_mixed_batch(
            list(decode_batch.reqs),
            wave.prefill_input.batch,
        )
        assert mixed_batch is not None
        self._group_input = self._prepare_mixed_batch(decode_batch, mixed_batch)
        self._decode_input = decode_view(
            self._group_input,
            self._table_manager,
            decode_batch,
        )

    def advance_step(self) -> list[ForwardData]:
        wave = self._require_wave()
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
            wave,
            group_input,
        )
        next_group_prefetched = self._try_prefetch_next_group(wave)
        try:
            combined_state = self._engine.advance_layer_group_prefill(
                group_input.batch,
                combined_state,
                end_layer,
            )
            if group_input.batch.has_decode:
                decode_rows = sum(
                    req.extend_len for req in group_input.batch.decode_reqs
                )
                decode_state, wave.state = split_state(combined_state, decode_rows)
            else:
                wave.state = combined_state
        except Exception:
            if wave.resident_group_active:
                moe_cache.end_prefill_group()
                wave.resident_group_active = False
            raise

        moe_cache.end_prefill_group()
        wave.resident_group_active = False
        wave.resident_groups += 1
        wave.group_forwards += 1
        wave.iterations += 1
        wave.current_layer = end_layer
        if next_group_prefetched:
            next_end = min(end_layer + wave.group_size, wave.num_layers)
            moe_cache.promote_prefetched_resident_group(end_layer, next_end)
            wave.resident_group_active = True

        outputs = self._finish_iteration_decode(wave, decode_state, end_layer)
        self._decode_input = None
        self._group_input = None
        if wave.done:
            outputs.extend(self._finish_wave(wave))
            self._log_completed_wave(wave)
            self._wave = None
        return outputs

    def _begin_iteration_state(
        self,
        wave: _LayeredPipelineWave,
        group_input: ForwardInput,
    ) -> tuple[LayerGroupState, LayerGroupState | None]:
        start_layer = wave.current_layer
        if start_layer == 0:
            if wave.state is not None:
                raise RuntimeError("layered pipeline wave was embedded more than once")
            self._restore_linear_states(group_input.batch)
            return self._engine.begin_layer_group_prefill(group_input.batch), None

        if wave.state is None or wave.state.next_layer != start_layer:
            raise RuntimeError("layered pipeline state is not at the active group")
        if self._decode_input is None:
            return wave.state, None
        decode_state = self._engine.begin_layer_group_decode(
            self._decode_input.batch,
            start_layer,
        )
        return merge_states(decode_state, wave.state), decode_state

    def _try_prefetch_next_group(self, wave: _LayeredPipelineWave) -> bool:
        next_start = wave.current_group_end
        next_end = min(next_start + wave.group_size, wave.num_layers)
        if next_start >= wave.num_layers or self._decode_input is None:
            return False
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        return moe_cache.try_prefetch_next_resident_group(next_start, next_end)

    def _finish_iteration_decode(
        self,
        wave: _LayeredPipelineWave,
        decode_state: LayerGroupState | None,
        resident_group_end: int,
    ) -> list[ForwardData]:
        decode_input = self._decode_input
        if decode_input is None:
            return []
        if decode_state is None:
            raise RuntimeError("mixed layered pipeline group did not produce decode state")
        if resident_group_end < wave.num_layers:
            decode_state = self._engine.advance_layer_group_decode(
                decode_input.batch,
                decode_state,
                wave.num_layers,
            )
        output = self._engine.finish_layer_group_prefill(
            decode_input.batch,
            decode_state,
            decode_input.sample_args,
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
        if state is None or state.next_layer != wave.num_layers:
            raise RuntimeError("layered pipeline wave completed without final model state")

        selected_requests: list[int] = []
        selected_rows: list[int] = []
        aborted_owners: list[Req] = []
        row = 0
        for request_index, req in enumerate(wave.prefill_input.batch.prefill_reqs):
            row += req.extend_len
            member = wave.admission.members[req.uid]
            if member.aborted or req.aborted:
                aborted_owners.append(req)
            elif isinstance(req, ChunkedReq):
                req.commit_prefill_kv()
            else:
                selected_requests.append(request_index)
                selected_rows.append(row - 1)

        outputs: list[ForwardData] = []
        if selected_requests:
            output_indices = torch.tensor(
                selected_rows,
                dtype=torch.int32,
                pin_memory=self._engine.device.type == "cuda",
            ).to(self._engine.device, non_blocking=True)
            output = self._engine.finish_layer_group_prefill(
                wave.prefill_input.batch,
                state,
                wave.prefill_input.sample_args,
                output_indices=output_indices,
                request_indices=selected_requests,
            )
            output_input = request_output_view(
                wave.prefill_input,
                selected_requests,
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
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        layer_prepares = (
            moe_cache.prefill_layer_prepares - wave.layer_prepares_at_start
        )
        logger.info_rank0(
            "Layered pipeline wave complete: "
            f"reqs={len(wave.admission.members)}, "
            f"groups={wave.resident_groups}, "
            f"group_forwards={wave.group_forwards}, "
            f"iterations={wave.iterations}, "
            f"decode_iterations={wave.decode_iterations}, "
            f"prefill_layer_prepares={layer_prepares}"
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
