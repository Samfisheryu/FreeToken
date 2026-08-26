from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.core import Batch

from .batch_composition import DecodeBatchSelector, compose_mixed_batch
from .decode import DecodeManager
from .prefill import PrefillManager


@dataclass
class LegacyBatchComposer:
    """Original scheduler policy: run available prefill before decode."""

    prefill_manager: PrefillManager
    decode_manager: DecodeManager

    def schedule_next_batch(self, token_budget: int) -> Batch | None:
        return (
            self.prefill_manager.schedule_next_batch(token_budget)
            or self.decode_manager.schedule_next_batch()
        )


@dataclass
class MixedBatchComposer:
    """Compose one forward from the managers' existing runnable work.

    Decode requests consume the shared query-token budget first. PrefillManager
    receives the exact remainder and retains ownership of admission, prefix
    matching, and chunk continuations.
    """

    prefill_manager: PrefillManager
    decode_manager: DecodeManager
    _decode_selector: DecodeBatchSelector = field(
        default_factory=DecodeBatchSelector, init=False
    )

    def schedule_next_batch(self, token_budget: int) -> Batch | None:
        if self.prefill_manager.requires_exclusive_batch:
            prefill_batch = self.prefill_manager.schedule_next_batch(token_budget)
            if prefill_batch is not None:
                return prefill_batch
            decode_batch = self.decode_manager.schedule_next_batch()
            decode_reqs, _ = self._decode_selector.select(decode_batch, token_budget)
            return compose_mixed_batch(decode_reqs, None)

        decode_batch = self.decode_manager.schedule_next_batch()
        decode_reqs, decode_tokens = self._decode_selector.select(
            decode_batch, token_budget
        )
        prefill_batch = self.prefill_manager.schedule_next_batch(
            token_budget - decode_tokens
        )
        return compose_mixed_batch(decode_reqs, prefill_batch)


__all__ = ["LegacyBatchComposer", "MixedBatchComposer"]
