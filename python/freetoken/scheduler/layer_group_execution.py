from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from freetoken.core import Batch
from freetoken.models.blocks import LayerGroupState

from .forward import ForwardInput

if TYPE_CHECKING:
    from .table import TableManager


def batch_view(batch: Batch, start: int, stop: int, decode_size: int) -> Batch:
    """Build an allocation-free request view and preserve admission accounting."""
    view = Batch(reqs=list(batch.reqs[start:stop]), decode_size=decode_size)
    view.log_new_tokens = batch.log_new_tokens
    view.log_cached_tokens = batch.log_cached_tokens
    view.prompt_admissions = list(batch.prompt_admissions)
    return view


def decode_view(
    mixed_input: ForwardInput,
    table_manager: TableManager,
    view: Batch | None = None,
) -> ForwardInput:
    """View the leading decode rows of one prepared Triton mixed batch."""
    from freetoken.attention.triton import TritonMetadata
    from freetoken.engine.sample import BatchSamplingArgs

    mixed_batch = mixed_input.batch
    decode_size = mixed_batch.decode_size
    if view is None:
        view = batch_view(mixed_batch, 0, decode_size, decode_size)
    decode_rows = sum(req.extend_len for req in view.reqs)
    if decode_rows != decode_size:
        raise RuntimeError(
            "resident layer-group decode metadata requires one query row per request"
        )
    metadata = mixed_batch.attn_metadata
    if not isinstance(metadata, TritonMetadata):
        raise RuntimeError(
            "resident layer-group metadata views require Triton attention"
        )

    decode_kv_rows = sum(req.device_len for req in view.reqs)
    view.padded_reqs = list(view.reqs)
    view.positions = mixed_batch.positions[:decode_rows]
    assert mixed_batch.out_loc is not None
    view.out_loc = mixed_batch.out_loc[:decode_rows]
    view.active_table_idx = mixed_input.input_tuple[0][:decode_rows]
    view.attn_metadata = TritonMetadata(
        cu_seqlens_q_gpu=metadata.cu_seqlens_q_gpu[: decode_size + 1],
        indptr=metadata.indptr[: decode_size + 1],
        indices=metadata.indices[:decode_kv_rows],
        q_to_req=metadata.q_to_req[:decode_rows],
        q_positions=metadata.q_positions[:decode_rows],
        is_decode=True,
        prefix_lens=metadata.prefix_lens[:decode_size],
        max_q_len=1,
        swa_indices=(
            metadata.swa_indices[:decode_kv_rows]
            if metadata.swa_indices is not None
            else None
        ),
    )
    sample_args = BatchSamplingArgs(
        temperatures=(
            mixed_input.sample_args.temperatures[:decode_size]
            if mixed_input.sample_args.temperatures is not None
            else None
        ),
        top_k=(
            mixed_input.sample_args.top_k[:decode_size]
            if mixed_input.sample_args.top_k is not None
            else None
        ),
        top_p=(
            mixed_input.sample_args.top_p[:decode_size]
            if mixed_input.sample_args.top_p is not None
            else None
        ),
    )
    input_tuple = (
        mixed_input.input_tuple[0][:decode_rows],
        mixed_input.input_tuple[1][:decode_rows],
    )
    write_tuple = (
        mixed_input.write_tuple[0][:decode_size],
        mixed_input.write_tuple[1][:decode_size],
    )
    view.input_ids = table_manager.token_pool[input_tuple]
    return ForwardInput(view, sample_args, input_tuple, write_tuple)


def prefill_view(
    mixed_input: ForwardInput,
    table_manager: TableManager,
    view: Batch,
) -> ForwardInput:
    """View the prefill suffix without rebuilding attention metadata."""
    from freetoken.engine.sample import BatchSamplingArgs

    mixed_batch = mixed_input.batch
    request_start = mixed_batch.decode_size
    row_start = sum(req.extend_len for req in mixed_batch.decode_reqs)
    view.padded_reqs = list(view.reqs)
    view.positions = mixed_batch.positions[row_start:]
    assert mixed_batch.out_loc is not None
    view.out_loc = mixed_batch.out_loc[row_start:]
    input_tuple = (
        mixed_input.input_tuple[0][row_start:],
        mixed_input.input_tuple[1][row_start:],
    )
    write_tuple = (
        mixed_input.write_tuple[0][request_start:],
        mixed_input.write_tuple[1][request_start:],
    )
    sample_args = BatchSamplingArgs(
        temperatures=(
            mixed_input.sample_args.temperatures[request_start:]
            if mixed_input.sample_args.temperatures is not None
            else None
        ),
        top_k=(
            mixed_input.sample_args.top_k[request_start:]
            if mixed_input.sample_args.top_k is not None
            else None
        ),
        top_p=(
            mixed_input.sample_args.top_p[request_start:]
            if mixed_input.sample_args.top_p is not None
            else None
        ),
    )
    view.input_ids = table_manager.token_pool[input_tuple]
    return ForwardInput(view, sample_args, input_tuple, write_tuple)


def merge_states(
    decode: LayerGroupState, prefill: LayerGroupState
) -> LayerGroupState:
    if decode.next_layer != prefill.next_layer:
        raise RuntimeError("decode and prefill states are at different layers")
    if (decode.residual is None) != (prefill.residual is None):
        raise RuntimeError("decode and prefill residual states do not match")
    residual = (
        None
        if decode.residual is None
        else torch.cat((decode.residual, prefill.residual), dim=0)
    )
    return LayerGroupState(
        hidden=torch.cat((decode.hidden, prefill.hidden), dim=0),
        residual=residual,
        next_layer=decode.next_layer,
    )


def split_state(
    state: LayerGroupState, decode_rows: int
) -> tuple[LayerGroupState, LayerGroupState]:
    decode_residual = prefill_residual = None
    if state.residual is not None:
        decode_residual = state.residual[:decode_rows]
        prefill_residual = state.residual[decode_rows:]
    return (
        LayerGroupState(state.hidden[:decode_rows], decode_residual, state.next_layer),
        LayerGroupState(state.hidden[decode_rows:], prefill_residual, state.next_layer),
    )


__all__ = [
    "batch_view",
    "decode_view",
    "merge_states",
    "prefill_view",
    "split_state",
]
