from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from freetoken.core import Batch, Req

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
    _decode_cursor: int = field(default=0, init=False)

    def schedule_next_batch(self, token_budget: int) -> Batch | None:
        if self.prefill_manager.requires_exclusive_batch:
            prefill_batch = self.prefill_manager.schedule_next_batch(token_budget)
            if prefill_batch is not None:
                return prefill_batch
            decode_batch = self.decode_manager.schedule_next_batch()
            decode_reqs, _ = self._select_decode_reqs(decode_batch, token_budget)
            return self._compose(decode_reqs, None)

        decode_batch = self.decode_manager.schedule_next_batch()
        decode_reqs, decode_tokens = self._select_decode_reqs(decode_batch, token_budget)
        prefill_batch = self.prefill_manager.schedule_next_batch(
            token_budget - decode_tokens
        )
        return self._compose(decode_reqs, prefill_batch)

    def _select_decode_reqs(
        self, decode_batch: Batch | None, token_budget: int
    ) -> tuple[List[Req], int]:
        if decode_batch is None or not decode_batch.reqs:
            return [], 0

        reqs = decode_batch.reqs
        total_tokens = sum(req.extend_len for req in reqs)
        if total_tokens <= token_budget:
            return list(reqs), total_tokens

        size = len(reqs)
        start = self._decode_cursor % size
        selected: List[Req] = []
        selected_tokens = 0
        for offset in range(size):
            req = reqs[(start + offset) % size]
            query_tokens = req.extend_len
            if selected_tokens + query_tokens <= token_budget:
                selected.append(req)
                selected_tokens += query_tokens
        self._decode_cursor = (start + 1) % size
        return selected, selected_tokens

    @staticmethod
    def _compose(decode_reqs: List[Req], prefill_batch: Batch | None) -> Batch | None:
        prefill_reqs = prefill_batch.reqs if prefill_batch is not None else []
        if not decode_reqs and not prefill_reqs:
            return None

        batch = Batch(
            reqs=[*decode_reqs, *prefill_reqs],
            decode_size=len(decode_reqs),
        )
        if prefill_batch is not None:
            batch.log_new_tokens = prefill_batch.log_new_tokens
            batch.log_cached_tokens = prefill_batch.log_cached_tokens
            batch.prompt_admissions = prefill_batch.prompt_admissions
        return batch


__all__ = ["LegacyBatchComposer", "MixedBatchComposer"]
