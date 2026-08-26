from __future__ import annotations

from dataclasses import dataclass

from freetoken.core import Batch, Req


@dataclass
class DecodeBatchSelector:
    """Fairly select decode requests that fit a shared query-token budget."""

    cursor: int = 0

    def select(
        self, decode_batch: Batch | None, token_budget: int
    ) -> tuple[list[Req], int]:
        if decode_batch is None or not decode_batch.reqs:
            return [], 0

        reqs = decode_batch.reqs
        total_tokens = sum(req.extend_len for req in reqs)
        if total_tokens <= token_budget:
            return list(reqs), total_tokens

        start = self.cursor % len(reqs)
        selected: list[Req] = []
        selected_tokens = 0
        for offset in range(len(reqs)):
            req = reqs[(start + offset) % len(reqs)]
            if selected_tokens + req.extend_len <= token_budget:
                selected.append(req)
                selected_tokens += req.extend_len
        self.cursor = (start + 1) % len(reqs)
        return selected, selected_tokens


def compose_mixed_batch(
    decode_reqs: list[Req], prefill_batch: Batch | None
) -> Batch | None:
    """Build one mixed batch while preserving prefill admission accounting."""
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


__all__ = ["DecodeBatchSelector", "compose_mixed_batch"]
