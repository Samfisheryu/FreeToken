"""Layered execution metadata ownership for Qwen3.6 hybrid-linear models."""

from __future__ import annotations

from freetoken.attention.linear import FLAMetadata
from freetoken.core import Batch
from freetoken.engine.layered_execution import LayeredExecutionAdapter


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

__all__ = ["Qwen3_5LayeredExecutionAdapter"]
