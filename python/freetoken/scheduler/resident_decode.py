from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from freetoken.core import Batch, Req

from .forward import ForwardInput, Indice2D

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnMetadata
    from freetoken.engine import Engine


@dataclass
class StableDecodeInput:
    reqs: tuple[Req, ...]
    table_indices: tuple[int, ...]
    cached_lens: tuple[int, ...]
    device_lens: tuple[int, ...]
    padded_size: int
    page_table_ptr: int
    token_pool_ptr: int
    positions: torch.Tensor
    input_tuple: Indice2D
    write_tuple: Indice2D
    attn_metadata: BaseAttnMetadata


def prepare_stable_decode(
    batch: Batch,
    previous: StableDecodeInput | None,
    *,
    engine: Engine,
    token_pool: torch.Tensor,
    prepare_resources: Callable[[Batch], None],
    build_forward_input: Callable[[Batch], ForwardInput],
) -> tuple[ForwardInput, StableDecodeInput | None]:
    """Allocate this decode step once and reuse graph metadata when rows stay stable."""
    if not batch.is_decode_only:
        raise ValueError("stable decode preparation requires a decode-only batch")

    prepare_resources(batch)
    reqs = tuple(batch.reqs)
    table_indices = tuple(req.table_idx for req in reqs)
    cached_lens = tuple(req.cached_len for req in reqs)
    device_lens = tuple(req.device_len for req in reqs)
    if (
        previous is not None
        and previous.reqs == reqs
        and previous.table_indices == table_indices
        and previous.padded_size == batch.padded_size
        and previous.page_table_ptr == engine.page_table.data_ptr()
        and previous.token_pool_ptr == token_pool.data_ptr()
        and all(
            current == prior + 1
            for current, prior in zip(cached_lens, previous.cached_lens)
        )
        and all(
            current == prior + 1
            for current, prior in zip(device_lens, previous.device_lens)
        )
    ):
        real_rows = slice(batch.size)
        previous.positions[real_rows].add_(1)
        previous.input_tuple[1][real_rows].add_(1)
        previous.write_tuple[1].add_(1)
        batch.positions = previous.positions
        batch.out_loc = engine.page_table[previous.input_tuple]
        batch.active_table_idx = previous.input_tuple[0].view(-1)
        batch.attn_metadata = previous.attn_metadata
        previous.cached_lens = cached_lens
        previous.device_lens = device_lens
        return (
            ForwardInput(
                batch=batch,
                sample_args=engine.sampler.prepare(batch),
                input_tuple=previous.input_tuple,
                write_tuple=previous.write_tuple,
            ),
            previous,
        )

    forward_input = build_forward_input(batch)

    # Stable reuse is valid only when Triton already points decode at graph-owned
    # metadata and no model-specific linear state must be restaged.
    from freetoken.attention.triton import TritonMetadata

    metadata = batch.attn_metadata
    if (
        engine.linear_state_pool is None
        and engine.graph_runner.can_use_cuda_graph(batch)
        and isinstance(metadata, TritonMetadata)
        and metadata.capture_staged
    ):
        batch.positions = metadata.q_positions
        current = StableDecodeInput(
            reqs=reqs,
            table_indices=table_indices,
            cached_lens=cached_lens,
            device_lens=device_lens,
            padded_size=batch.padded_size,
            page_table_ptr=engine.page_table.data_ptr(),
            token_pool_ptr=token_pool.data_ptr(),
            positions=batch.positions,
            input_tuple=forward_input.input_tuple,
            write_tuple=forward_input.write_tuple,
            attn_metadata=metadata,
        )
        return forward_input, current
    return forward_input, None


__all__ = ["StableDecodeInput", "prepare_stable_decode"]
