"""Layered execution metadata ownership for Qwen3.6 hybrid-linear models."""

from __future__ import annotations

from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch
from freetoken.engine.layered_execution import LayeredExecutionAdapter
from freetoken.scheduler.forward import ForwardInput


class Qwen3_5LayeredExecutionAdapter(LayeredExecutionAdapter):
    """Keep GDN cursor and metadata views behind the model execution boundary."""

    def prepare_metadata_view(self, source: Batch, target: Batch) -> bool:
        if self._engine.linear_state_pool is None:
            return super().prepare_metadata_view(source, target)
        source_metadata = source.fla_metadata
        if not isinstance(source_metadata, FLAMetadata):
            raise RuntimeError("source batch is missing linear attention metadata")
        if target.is_decode_only:
            if source_metadata.decode is None:
                raise RuntimeError("source batch has no linear decode metadata")
            target.linear_table_idx = source_metadata.decode.cache_indices
            target.fla_metadata = FLAMetadata(decode=source_metadata.decode)
        elif target.has_prefill:
            if source_metadata.prefill is None:
                raise RuntimeError("source batch has no linear prefill metadata")
            target.fla_metadata = FLAMetadata(prefill=source_metadata.prefill)
        return super().prepare_metadata_view(source, target)

    def capture_prefill_metadata_state(self, prefill_input: ForwardInput):
        if self._engine.linear_state_pool is None:
            return None
        return tuple(
            (req.mamba_next_track_idx, req.mamba_last_track_seqlen)
            for req in prefill_input.batch.prefill_reqs
        )

    def restore_prefill_metadata_state(
        self,
        group_input: ForwardInput,
        prefill_input: ForwardInput,
        state,
    ) -> None:
        if state is None:
            return
        if len(state) != len(prefill_input.batch.prefill_reqs):
            raise RuntimeError("layered prefill metadata snapshot has the wrong size")
        for req, (next_track, last_track) in zip(
            prefill_input.batch.prefill_reqs,
            state,
            strict=True,
        ):
            req.mamba_next_track_idx = next_track
            req.mamba_last_track_seqlen = last_track

        source_metadata = prefill_input.batch.fla_metadata
        mixed_metadata = group_input.batch.fla_metadata
        if not isinstance(source_metadata, FLAMetadata):
            raise RuntimeError("layered prefill tile is missing linear metadata")
        if not isinstance(mixed_metadata, FLAMetadata):
            raise RuntimeError("layered mixed batch is missing linear metadata")
        group_input.batch.fla_metadata = FLAMetadata(
            decode=mixed_metadata.decode,
            prefill=source_metadata.prefill,
        )

__all__ = ["Qwen3_5LayeredExecutionAdapter"]
