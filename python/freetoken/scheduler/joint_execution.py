from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from freetoken.core import Batch, Req
from freetoken.utils import init_logger

from .forward import ForwardData, ForwardInput
from .joint_batch import JointBatchComposer
from .prefill import ChunkedReq, PrefillManager

if TYPE_CHECKING:
    import torch

    from freetoken.engine import Engine
    from freetoken.models.blocks import LayerGroupState

    from .decode import DecodeManager
    from .table import TableManager


logger = init_logger(__name__)


@dataclass
class _JointPrefillChunk:
    forward_input: ForwardInput
    state: LayerGroupState | None = None


@dataclass
class _JointPrefillWave:
    uid: int
    num_layers: int
    group_size: int
    chunks: list[_JointPrefillChunk]
    current_layer: int = 0
    layer_prepares_at_start: int = 0
    h2d_bytes_at_start: int = 0

    @property
    def current_group_end(self) -> int:
        return min(self.current_layer + self.group_size, self.num_layers)

    @property
    def done(self) -> bool:
        return self.current_layer >= self.num_layers

    def finish_group(self) -> None:
        if self.done:
            raise RuntimeError("joint prefill wave is already complete")
        self.current_layer = self.current_group_end


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
        self._composer = JointBatchComposer(
            prefill_manager=prefill_manager,
            decode_manager=decode_manager,
        )
        self._wave: _JointPrefillWave | None = None

    @property
    def active(self) -> bool:
        return self._wave is not None

    def schedule_first_batch(self, token_budget: int) -> Batch | None:
        return self._composer.schedule_first_batch(token_budget)

    def begin_wave(self, first_batch: Batch, token_budget: int) -> None:
        """Prepare one same-request wave, bounded by ``max_chunks``."""
        if self._wave is not None:
            raise RuntimeError("a joint prefill wave is already active")
        if len(first_batch.prefill_reqs) != 1:
            raise RuntimeError("a joint wave requires exactly one first prefill request")

        first = self._prepare_chunk(first_batch)
        uid = first_batch.prefill_reqs[0].uid
        chunks = [first]
        while (
            len(chunks) < self._max_chunks
            and self._prefill_manager.has_pending_uid(uid)
        ):
            batch = self._prefill_manager.schedule_next_batch(
                token_budget,
                allowed_uids={uid},
                max_reqs=1,
            )
            if batch is None:
                break
            chunks.append(self._prepare_chunk(batch))

        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        group_size = moe_cache.effective_prefill_group_size
        if group_size < 1:
            raise RuntimeError("joint batching has no resident expert-layer capacity")
        self._wave = _JointPrefillWave(
            uid=uid,
            num_layers=self._engine.layer_group_num_layers,
            group_size=group_size,
            chunks=chunks,
            layer_prepares_at_start=moe_cache.prefill_layer_prepares,
            h2d_bytes_at_start=moe_cache.prefill_h2d_bytes,
        )

    def _prepare_chunk(self, batch: Batch) -> _JointPrefillChunk:
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        batch.input_ids = self._table_manager.token_pool[forward_input.input_tuple]
        for req in batch.prefill_reqs:
            self._prefill_manager.reserve_layered_continuation(req)
        return _JointPrefillChunk(forward_input=forward_input)

    def advance_group(self) -> list[ForwardData]:
        """Advance every chunk through one resident group and finalize at wave end."""
        wave = self._wave
        if wave is None or wave.done:
            raise RuntimeError("joint prefill wave has no group to advance")

        self._advance_group_states(wave)
        wave.finish_group()
        if not wave.done:
            return []

        outputs = self._finish_wave(wave)
        self._log_completed_wave(wave)
        self._wave = None
        return outputs

    def _advance_group_states(self, wave: _JointPrefillWave) -> None:
        start_layer = wave.current_layer
        end_layer = wave.current_group_end
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None

        moe_cache.begin_resident_prefill_group(start_layer, end_layer)
        try:
            for chunk in wave.chunks:
                forward_input = chunk.forward_input
                if chunk.state is None:
                    if start_layer != 0:
                        raise RuntimeError(
                            "joint state reached a later group without embedding"
                        )
                    self._restore_linear_states(forward_input.batch)
                    chunk.state = self._engine.begin_layer_group_prefill(
                        forward_input.batch
                    )
                if chunk.state.next_layer != start_layer:
                    raise RuntimeError(
                        f"joint state is at layer {chunk.state.next_layer}, "
                        f"group starts at {start_layer}"
                    )
                chunk.state = self._engine.advance_layer_group_prefill(
                    forward_input.batch,
                    chunk.state,
                    end_layer,
                )
        finally:
            moe_cache.end_prefill_group()

    def _finish_wave(self, wave: _JointPrefillWave) -> list[ForwardData]:
        """Commit chunk boundaries and publish only requests that produce tokens."""
        first = wave.chunks[0]
        last = wave.chunks[-1]
        last_prompt_req = last.forward_input.batch.prefill_reqs[-1]
        prompt_is_final = not isinstance(last_prompt_req, ChunkedReq)

        for chunk in wave.chunks:
            for req in chunk.forward_input.batch.prefill_reqs:
                if isinstance(req, ChunkedReq):
                    req.commit_prefill_kv()

        outputs: list[ForwardData] = []
        if len(wave.chunks) == 1 and prompt_is_final:
            forward_input = first.forward_input
            output = self._engine.finish_layer_group_prefill(
                forward_input.batch,
                self._require_state(first),
                forward_input.sample_args,
            )
            self._write_and_filter(forward_input, output.next_tokens_gpu)
            outputs.append((forward_input, output))
        else:
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
                self._write_and_filter(decode_input, output.next_tokens_gpu)
                outputs.append((decode_input, output))

            if prompt_is_final:
                prompt_input = last.forward_input
                output = self._engine.finish_layer_group_prefill(
                    prompt_input.batch,
                    self._require_state(last),
                    prompt_input.sample_args,
                )
                self._write_and_filter(prompt_input, output.next_tokens_gpu)
                outputs.append((prompt_input, output))
            elif last_prompt_req.aborted:
                self._engine.stream.synchronize()
                self._free_req_resources(last_prompt_req)

        for chunk in wave.chunks:
            chunk.state = None
        return outputs

    @staticmethod
    def _require_state(chunk: _JointPrefillChunk) -> LayerGroupState:
        if chunk.state is None:
            raise RuntimeError("joint prefill chunk completed without model state")
        return chunk.state

    def _write_and_filter(
        self, forward_input: ForwardInput, next_tokens_gpu: torch.Tensor
    ) -> None:
        self._table_manager.token_pool[forward_input.write_tuple] = next_tokens_gpu
        self._decode_manager.filter_reqs(forward_input.batch.reqs)

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

    def _log_completed_wave(self, wave: _JointPrefillWave) -> None:
        moe_cache = self._engine.moe_offload_cache
        assert moe_cache is not None
        layer_prepares = (
            moe_cache.prefill_layer_prepares - wave.layer_prepares_at_start
        )
        h2d_bytes = moe_cache.prefill_h2d_bytes - wave.h2d_bytes_at_start
        groups = (wave.num_layers + wave.group_size - 1) // wave.group_size
        logger.info_rank0(
            "Joint wave complete: "
            f"chunks={len(wave.chunks)}, groups={groups}, "
            f"effective_group_size={wave.group_size}, "
            f"prefill_layer_prepares={layer_prepares}, "
            f"prefill_h2d_bytes={h2d_bytes}"
        )

    def abort(self, uid: int) -> Req | None:
        """Mark active joint work for cleanup after the final group drains."""
        wave = self._wave
        if wave is None:
            return None
        for chunk in wave.chunks:
            for req in chunk.forward_input.batch.decode_reqs:
                if req.uid == uid:
                    req.aborted = True
                    return req
        if wave.uid != uid:
            return None

        owner = wave.chunks[-1].forward_input.batch.prefill_reqs[-1]
        owner.aborted = True
        return owner


__all__ = ["JointWaveExecutor"]
