from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from freetoken.core import Batch, Req

from .decode import DecodeManager
from .prefill import PrefillManager

if TYPE_CHECKING:
    from .scheduler import ForwardInput


@dataclass
class JointBatchComposer:
    """Select one true mixed batch with decode tokens taking budget first."""

    prefill_manager: PrefillManager
    decode_manager: DecodeManager
    _decode_cursor: int = field(default=0, init=False)

    def schedule_first_batch(self, token_budget: int) -> Batch | None:
        decode_batch = self.decode_manager.schedule_next_batch()
        decode_reqs = self._select_decode_reqs(decode_batch, token_budget)
        decode_tokens = sum(req.extend_len for req in decode_reqs)
        prefill_batch = self.prefill_manager.schedule_next_batch(
            token_budget - decode_tokens,
            max_reqs=1,
        )
        return self._compose(decode_reqs, prefill_batch)

    def _select_decode_reqs(
        self, decode_batch: Batch | None, token_budget: int
    ) -> list[Req]:
        if decode_batch is None or not decode_batch.reqs:
            return []
        reqs = decode_batch.reqs
        if sum(req.extend_len for req in reqs) <= token_budget:
            return list(reqs)
        start = self._decode_cursor % len(reqs)
        selected: list[Req] = []
        used = 0
        for offset in range(len(reqs)):
            req = reqs[(start + offset) % len(reqs)]
            if used + req.extend_len <= token_budget:
                selected.append(req)
                used += req.extend_len
        self._decode_cursor = (start + 1) % len(reqs)
        return selected

    @staticmethod
    def _compose(decode_reqs: list[Req], prefill_batch: Batch | None) -> Batch | None:
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


@dataclass
class JointPrefillChunk:
    forward_input: ForwardInput
    state: Any | None = None


@dataclass
class JointPrefillWave:
    uid: int
    num_layers: int
    group_size: int
    chunks: list[JointPrefillChunk]
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


@dataclass
class JointExecutionStats:
    waves: int = 0
    group_steps: int = 0
    effective_group_size: int = 0

    def snapshot(
        self,
        *,
        prefill_layer_prepares: int,
        prefill_h2d_bytes: int,
    ) -> dict[str, int]:
        return {
            "waves": self.waves,
            "group_steps": self.group_steps,
            "effective_group_size": self.effective_group_size,
            "prefill_layer_prepares": prefill_layer_prepares,
            "prefill_h2d_bytes": prefill_h2d_bytes,
        }


__all__ = [
    "JointBatchComposer",
    "JointExecutionStats",
    "JointPrefillChunk",
    "JointPrefillWave",
]
