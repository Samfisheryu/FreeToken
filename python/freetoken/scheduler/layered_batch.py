from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from freetoken.core import Batch

from .decode import DecodeManager
from .prefill import PrefillManager

if TYPE_CHECKING:
    from .scheduler import ForwardInput


@dataclass(frozen=True)
class LayeredBatchPlan:
    """Two independent forwards selected for one scheduler round."""

    decode_batch: Batch | None
    prefill_batch: Batch | None

    def __post_init__(self) -> None:
        if self.decode_batch is None and self.prefill_batch is None:
            raise ValueError("LayeredBatchPlan requires at least one batch")


@dataclass
class LayeredBatchComposer:
    """Jointly select decode and prefill without concatenating their tokens."""

    prefill_manager: PrefillManager
    decode_manager: DecodeManager
    max_prefill_reqs: int | None = None

    def schedule_next_plan(self, token_budget: int) -> LayeredBatchPlan | None:
        decode_batch = self.decode_manager.schedule_next_batch()
        if self.max_prefill_reqs is None:
            prefill_batch = self.prefill_manager.schedule_next_batch(token_budget)
        else:
            prefill_batch = self.prefill_manager.schedule_next_batch(
                token_budget, max_reqs=self.max_prefill_reqs
            )
        if decode_batch is not None and not decode_batch.reqs:
            decode_batch = None
        if prefill_batch is not None and not prefill_batch.reqs:
            prefill_batch = None
        if decode_batch is None and prefill_batch is None:
            return None
        if decode_batch is not None and prefill_batch is not None:
            decode_ids = {id(req) for req in decode_batch.reqs}
            if any(id(req) in decode_ids for req in prefill_batch.reqs):
                raise ValueError(
                    "decode_batch and prefill_batch must not share request objects"
                )
        return LayeredBatchPlan(
            decode_batch=decode_batch,
            prefill_batch=prefill_batch,
        )


@dataclass
class LayeredExecutionStats:
    joint_rounds: int = 0
    decode_forwards: int = 0
    prefill_group_steps: int = 0
    decode_gpu_ms: float = 0.0
    prefill_gpu_ms: float = 0.0
    joint_wall_ms: float = 0.0

    def snapshot(self) -> dict[str, int | float]:
        return {
            "joint_rounds": self.joint_rounds,
            "decode_forwards": self.decode_forwards,
            "prefill_group_steps": self.prefill_group_steps,
            "decode_gpu_ms": self.decode_gpu_ms,
            "prefill_gpu_ms": self.prefill_gpu_ms,
            "joint_wall_ms": self.joint_wall_ms,
        }


@dataclass
class LayeredPrefillChunk:
    forward_input: ForwardInput
    # ``Req.device_len`` is advanced when the final layer samples.  Keep the
    # allocation boundary immutable so abort can return exactly the KV pages
    # allocated for this wave, including page padding but excluding output.
    allocated_device_len: int
    state: Any | None = None


@dataclass
class LayeredPrefillWave:
    uid: int
    num_layers: int
    group_size: int
    chunks: list[LayeredPrefillChunk] = field(default_factory=list)
    current_layer: int = 0
    chunk_cursor: int = 0
    admitting: bool = True

    @property
    def done(self) -> bool:
        return not self.admitting and self.current_layer >= self.num_layers

    @property
    def current_group_end(self) -> int:
        """Exclusive layer boundary for the scheduler round's active group."""
        return min(
            ((self.current_layer // self.group_size) + 1) * self.group_size,
            self.num_layers,
        )

    def add_chunk(self, chunk: LayeredPrefillChunk) -> None:
        if not self.admitting or self.current_layer != 0:
            raise RuntimeError("new chunks can only enter while layer 0 is active")
        self.chunks.append(chunk)

    def finish_admission(self) -> None:
        if not self.chunks:
            raise RuntimeError("cannot finish an empty prefill wave")
        self.admitting = False
        self.current_layer = 1
        self.chunk_cursor = 0

    def current_chunk(self) -> LayeredPrefillChunk:
        if self.admitting or self.done:
            raise RuntimeError("prefill wave has no replay chunk at this point")
        return self.chunks[self.chunk_cursor]

    def complete_replay_chunk(self) -> bool:
        """Advance the layer-major cursor; return True at a logical group boundary."""
        self.chunk_cursor += 1
        if self.chunk_cursor < len(self.chunks):
            return False
        self.chunk_cursor = 0
        self.current_layer += 1
        completed = self.current_layer
        return completed % self.group_size == 0 or completed == self.num_layers


__all__ = ["LayeredBatchComposer", "LayeredBatchPlan", "LayeredExecutionStats"]
