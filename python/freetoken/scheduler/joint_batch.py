from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.core import Batch

from .batch_composition import DecodeBatchSelector, compose_mixed_batch
from .decode import DecodeManager
from .prefill import PrefillManager


@dataclass
class JointBatchComposer:
    """Select one true mixed batch with decode tokens taking budget first."""

    prefill_manager: PrefillManager
    decode_manager: DecodeManager
    _decode_selector: DecodeBatchSelector = field(
        default_factory=DecodeBatchSelector, init=False
    )

    def schedule_first_batch(self, token_budget: int) -> Batch | None:
        decode_batch = self.decode_manager.schedule_next_batch()
        decode_reqs, decode_tokens = self._decode_selector.select(
            decode_batch, token_budget
        )
        prefill_batch = self.prefill_manager.schedule_next_batch(
            token_budget - decode_tokens,
            max_reqs=1,
        )
        return compose_mixed_batch(decode_reqs, prefill_batch)


__all__ = ["JointBatchComposer"]
