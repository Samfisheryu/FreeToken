from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List

import torch
from freetoken.core import Batch, Req, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.models.blocks import LayerGroupState
from freetoken.utils import init_logger, mem_GB
from freetoken.utils.progress import emit_progress
from tqdm import tqdm

if TYPE_CHECKING:
    from freetoken.attention import BaseAttnBackend
    from freetoken.models import BaseLLMModel
    from freetoken.moe.offload_cache import OffloadMoeCache

logger = init_logger(__name__)


@dataclass
class _LayerRangeCapture:
    graph: torch.cuda.CUDAGraph
    output: LayerGroupState


@dataclass
class GraphCaptureBuffer:
    input_ids: torch.Tensor
    out_loc: torch.Tensor
    positions: torch.Tensor
    logits: torch.Tensor
    table_idx: torch.Tensor  # per-request slot id for GatedDeltaNet state gather/scatter
    # Decode GDN query indptr = arange(bs+1); a constant per captured bs, filled once.
    fla_cu_seqlens: torch.Tensor

    @classmethod
    def init(cls, bs: int, vocab_size: int, device: torch.device) -> GraphCaptureBuffer:
        return GraphCaptureBuffer(
            input_ids=torch.zeros(bs, dtype=torch.int32, device=device),
            out_loc=torch.zeros(bs, dtype=torch.int32, device=device),
            positions=torch.zeros(bs, dtype=torch.int32, device=device),
            logits=torch.empty(bs, vocab_size, dtype=torch.float32, device=device),
            table_idx=torch.zeros(bs, dtype=torch.int32, device=device),
            fla_cu_seqlens=torch.arange(bs + 1, dtype=torch.int32, device=device),
        )

    def set_batch(self, batch: Batch) -> None:
        from freetoken.attention.linear import FLAMetadata, FLAPathMetadata

        _slice = slice(batch.padded_size)
        bs = batch.padded_size
        batch.input_ids = self.input_ids[_slice]
        batch.out_loc = self.out_loc[_slice]
        batch.positions = self.positions[_slice]
        batch.linear_table_idx = self.table_idx[_slice]
        # Decode GDN metadata reads the persistent cu_seqlens (constant arange) and the
        # persistent table_idx slot map, so the captured kernels see stable addresses.
        batch.fla_metadata = FLAMetadata(
            decode=FLAPathMetadata(
                cu_seqlens=self.fla_cu_seqlens[: bs + 1],
                cache_indices=self.table_idx[_slice],
            )
        )

    def copy_from(self, batch: Batch) -> None:
        _slice = slice(batch.padded_size)
        self.input_ids[_slice] = batch.input_ids
        if batch.out_loc is not None:
            self.out_loc[_slice] = batch.out_loc
        self.positions[_slice] = batch.positions
        if batch.linear_table_idx is not None:
            self.table_idx[_slice] = batch.linear_table_idx


def _determine_cuda_graph_bs(
    cuda_graph_bs: List[int] | None,
    cuda_graph_max_bs: int | None,
    free_memory: int,
) -> List[int]:
    if cuda_graph_bs is not None:
        return cuda_graph_bs

    free_memory_gb = free_memory / (1 << 30)
    if cuda_graph_max_bs is None:
        if free_memory_gb > 80:  # H200
            cuda_graph_max_bs = 256
        else:
            cuda_graph_max_bs = 160

    if cuda_graph_max_bs < 1:
        return []

    candidates = [1, 2, 4] + list(range(8, cuda_graph_max_bs + 1, 8))
    return [bs for bs in candidates if bs <= cuda_graph_max_bs]


def get_free_memory(device: torch.device) -> int:
    return torch.cuda.mem_get_info(device)[0]


class GraphRunner:
    def __init__(
        self,
        stream: torch.cuda.Stream,
        device: torch.device,
        model: BaseLLMModel,
        attn_backend: BaseAttnBackend,
        cuda_graph_bs: List[int] | None,
        cuda_graph_max_bs: int | None,
        free_memory: int,
        max_seq_len: int,
        vocab_size: int,
        dummy_req: Req,
        moe_offload_cache: OffloadMoeCache | None = None,
    ) -> None:
        cuda_graph_bs = _determine_cuda_graph_bs(
            cuda_graph_bs=cuda_graph_bs,
            cuda_graph_max_bs=cuda_graph_max_bs,
            free_memory=free_memory,
        )
        self.attn_backend = attn_backend
        self.max_graph_bs = max(cuda_graph_bs) if cuda_graph_bs else 0
        self.graph_bs_list = sorted(cuda_graph_bs)
        self.dummy_req = dummy_req
        self.moe_offload_cache = moe_offload_cache
        self.stream = stream
        self.device = device
        self.layer_range_graph_map: Dict[
            tuple[int, int, int], _LayerRangeCapture
        ] = {}
        self.layer_range_group_ends: dict[int, int] = {}
        self.layer_range_batch_sizes: set[int] = set()
        self._layer_range_hidden_input: torch.Tensor | None = None
        self._layer_range_residual_input: torch.Tensor | None = None
        self._prepared_layer_range_batch: Batch | None = None
        self._capture_graphs(max_seq_len, vocab_size, model)

    def _reset_moe_offload_cache(self) -> None:
        if self.moe_offload_cache is not None:
            self.moe_offload_cache.reset()

    def _capture_graphs(self, max_seq_len: int, vocab_size: int, model: BaseLLMModel):
        # Mark the post-weights "warmup" phase for /health: this stretch (graph capture — or the
        # remaining readiness work when graphs are disabled) moves no bytes, so without this the
        # loader would sit at 100% (last byte bar) until the ready ack. total=0 ⇒ the desktop
        # reads it as an indeterminate phase and animates the bar. Must precede the
        # graphs-disabled early return so that config gets the phase too.
        emit_progress("Capturing CUDA graphs / warming up", 0, 0)
        self.graph_map: Dict[int, torch.cuda.CUDAGraph] = {}
        if self.max_graph_bs == 0:
            return logger.info_rank0("CUDA graph is disabled.")

        self.attn_backend.init_capture_graph(max_seq_len=max_seq_len, bs_list=self.graph_bs_list)

        torch.cuda.synchronize(self.device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)

        logger.info_rank0(f"Start capturing CUDA graphs with sizes: {self.graph_bs_list}")
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory before capturing CUDA graphs: {mem_GB(free_memory)}")

        self.buffer = GraphCaptureBuffer.init(self.max_graph_bs, vocab_size, self.device)
        self._reset_moe_offload_cache()

        pbar = tqdm(
            sorted(self.graph_bs_list, reverse=True),
            desc="Preparing for capturing CUDA graphs...",
            unit="batch",
            disable=not get_tp_info().is_primary(),  # disable for non-primary ranks
        )
        pool = None
        for bs in pbar:
            free_memory = get_free_memory(self.device)
            pbar.desc = f"Capturing graphs: bs = {bs:<3} | avail_mem = {mem_GB(free_memory)}"
            pbar.refresh()
            graph = torch.cuda.CUDAGraph()
            batch = Batch(reqs=[self.dummy_req] * bs, decode_size=bs)
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_capture(batch)
            self.buffer.set_batch(batch)
            # capture on the dummy linear-state slot so GatedDeltaNet gather/scatter
            # touches scratch (real slot indices are written by copy_from on replay). Hybrid-
            # radix decouples the GDN slot from table_idx -> use the GDN padding slot.
            dummy_slot = (self.dummy_req.linear_slot_idx
                          if self.dummy_req.linear_slot_idx is not None
                          else self.dummy_req.table_idx)
            self.buffer.table_idx[:bs].fill_(dummy_slot)
            with get_global_ctx().forward_batch(batch):
                self.buffer.logits[:bs] = model.forward()
                # Keep the offload cache warmed for capture. Resetting here forces
                # CUDA graph capture to replay cold-cache expert copies.
                with torch.cuda.graph(graph, pool=pool, stream=self.stream):
                    self.buffer.logits[:bs] = model.forward()
                self._reset_moe_offload_cache()
            if pool is None:
                pool = graph.pool()  # reuse cuda graph handle to reduce memory
            self.graph_map[bs] = graph

        assert pool is not None
        self._capture_layer_range_graphs(model)
        self._reset_moe_offload_cache()
        free_memory = get_free_memory(self.device)
        logger.info_rank0(f"Free GPU memory after capturing CUDA graphs: {mem_GB(free_memory)}")

    def _capture_layer_range_graphs(
        self,
        model: BaseLLMModel,
    ) -> None:
        """Capture exact-size decode graphs for the pipeline's fixed layer groups.

        Capturing one graph per group keeps startup graph work linear in model depth.
        The active resident group remains eager because its decode rows are merged with
        prefill rows; these graphs cover only the decode-only groups before and after it.
        """
        from freetoken.attention.triton import TritonAttentionBackend

        cache = self.moe_offload_cache
        if (
            cache is None
            or getattr(cache, "prefill_group_decode_reserve_layers", 0) == 0
            or not getattr(model, "supports_layer_group_prefill", False)
            or not isinstance(self.attn_backend, TritonAttentionBackend)
        ):
            return

        num_layers = model.layer_group_num_layers
        group_size = cache.effective_prefill_group_size
        if group_size < 1:
            return
        groups = [
            (start, min(start + group_size, num_layers))
            for start in range(0, num_layers, group_size)
        ]
        batch_sizes = self.graph_bs_list
        if not batch_sizes:
            return

        self.layer_range_batch_sizes = set(batch_sizes)
        self.layer_range_group_ends = dict(groups)
        logger.info_rank0(
            "Capturing layered-pipeline decode range graphs: "
            f"groups={groups}, batch_sizes={batch_sizes}"
        )

        # A shared, graph-external input state lets every group graph accept the
        # preceding group's dynamic output without capturing any Python state joins.
        seed_bs = max(batch_sizes)
        seed_batch = Batch(reqs=[self.dummy_req] * seed_bs, decode_size=seed_bs)
        seed_batch.padded_reqs = seed_batch.reqs
        self.attn_backend.prepare_for_layer_range_capture(seed_batch)
        self.buffer.set_batch(seed_batch)
        self._set_dummy_linear_slots(seed_bs)
        with get_global_ctx().forward_batch(seed_batch):
            seed = model.begin_layer_group_prefill(seed_batch.input_ids)
            seed = model.advance_layer_group_prefill(seed, groups[0][1])
        if seed.residual is None:
            raise RuntimeError(
                "layer-range graphs require residual state after the first decoder group"
            )
        self._layer_range_hidden_input = torch.empty_like(seed.hidden)
        self._layer_range_residual_input = torch.empty_like(seed.residual)
        self._layer_range_hidden_input.zero_()
        self._layer_range_residual_input.zero_()
        self._reset_moe_offload_cache()

        for bs in reversed(batch_sizes):
            # Whole-decode graphs can interleave arbitrarily with range replay,
            # and exact batch sizes can alternate between scheduler iterations.
            # Keep each size's monotonic group sequence in its own graph pool.
            range_pool: tuple[int, int] | None = None
            batch = Batch(reqs=[self.dummy_req] * bs, decode_size=bs)
            batch.padded_reqs = batch.reqs
            self.attn_backend.prepare_for_layer_range_capture(batch)
            self.buffer.set_batch(batch)
            self._set_dummy_linear_slots(bs)
            for start_layer, end_layer in groups:
                graph = torch.cuda.CUDAGraph()
                with get_global_ctx().forward_batch(batch):
                    if start_layer == 0:
                        warm = model.begin_layer_group_prefill(batch.input_ids)
                        model.advance_layer_group_prefill(warm, end_layer)
                        with torch.cuda.graph(
                            graph, pool=range_pool, stream=self.stream
                        ):
                            captured = model.begin_layer_group_prefill(batch.input_ids)
                            captured = model.advance_layer_group_prefill(
                                captured, end_layer
                            )
                    else:
                        assert self._layer_range_hidden_input is not None
                        assert self._layer_range_residual_input is not None
                        warm = LayerGroupState(
                            self._layer_range_hidden_input[:bs],
                            self._layer_range_residual_input[:bs],
                            start_layer,
                        )
                        model.advance_layer_group_prefill(warm, end_layer)
                        with torch.cuda.graph(
                            graph, pool=range_pool, stream=self.stream
                        ):
                            captured = LayerGroupState(
                                self._layer_range_hidden_input[:bs],
                                self._layer_range_residual_input[:bs],
                                start_layer,
                            )
                            captured = model.advance_layer_group_prefill(
                                captured, end_layer
                            )
                    self._reset_moe_offload_cache()
                if range_pool is None:
                    range_pool = graph.pool()
                self.layer_range_graph_map[(start_layer, end_layer, bs)] = (
                    _LayerRangeCapture(graph=graph, output=captured)
                )

    def _set_dummy_linear_slots(self, bs: int) -> None:
        dummy_slot = (
            self.dummy_req.linear_slot_idx
            if self.dummy_req.linear_slot_idx is not None
            else self.dummy_req.table_idx
        )
        self.buffer.table_idx[:bs].fill_(dummy_slot)

    def can_use_cuda_graph(self, batch: Batch) -> bool:
        return batch.is_decode_only and batch.size <= self.max_graph_bs

    def replay(self, batch: Batch) -> torch.Tensor:
        assert self.can_use_cuda_graph(batch)
        self.buffer.copy_from(batch)
        g = self.graph_map[batch.padded_size]
        self.attn_backend.prepare_for_replay(batch)
        g.replay()
        return self.buffer.logits[: batch.size]

    def has_layer_range_graphs_for(self, batch: Batch) -> bool:
        return (
            batch.is_decode_only
            and batch.size == batch.padded_size
            and batch.size in self.layer_range_batch_sizes
        )

    def layer_range_end(self, start_layer: int) -> int | None:
        return self.layer_range_group_ends.get(start_layer)

    def prepare_layer_range_replay(self, batch: Batch) -> None:
        if not self.has_layer_range_graphs_for(batch):
            raise ValueError("batch is not eligible for a layer-range graph")
        if batch is self._prepared_layer_range_batch:
            return
        self.buffer.copy_from(batch)
        from freetoken.attention.triton import TritonAttentionBackend

        assert isinstance(self.attn_backend, TritonAttentionBackend)
        self.attn_backend.prepare_for_layer_range_replay(batch)
        self._prepared_layer_range_batch = batch

    def replay_layer_range(
        self,
        batch: Batch,
        state: LayerGroupState | None,
        start_layer: int,
        end_layer: int,
    ) -> LayerGroupState:
        capture = self.layer_range_graph_map[(start_layer, end_layer, batch.size)]
        if start_layer == 0:
            if state is not None:
                raise ValueError("layer-zero range replay cannot accept decoder state")
        else:
            if state is None or state.next_layer != start_layer:
                raise ValueError(
                    f"layer-range replay expected state at layer {start_layer}"
                )
            if state.residual is None:
                raise ValueError("layer-range replay requires decoder residual state")
            assert self._layer_range_hidden_input is not None
            assert self._layer_range_residual_input is not None
            self._layer_range_hidden_input[: batch.size].copy_(state.hidden)
            self._layer_range_residual_input[: batch.size].copy_(state.residual)
        capture.graph.replay()
        output = capture.output
        return LayerGroupState(
            output.hidden[: batch.size],
            (
                output.residual[: batch.size]
                if output.residual is not None
                else None
            ),
            end_layer,
        )

    def pad_batch(self, batch: Batch) -> None:
        padded_size = (  # choose the first available batch size
            next(bs for bs in self.graph_bs_list if bs >= batch.size)
            if self.can_use_cuda_graph(batch)
            else batch.size
        )
        batch.padded_reqs = batch.reqs + [self.dummy_req] * (padded_size - batch.size)

    # NOTE: This must be called before freeing NCCL resources to prevent program hang
    def destroy_cuda_graphs(self) -> None:
        # Drop the CUDAGraph objects (and the shared mempool they hold) AND the static
        # GraphCaptureBuffer tensors ([max_bs, vocab] logits + input/out_loc/positions/...).
        # Dropping the references is the load-bearing step; without it a runtime rebuild's
        # free-before-alloc cannot reclaim this GPU memory. empty_cache() is left to the
        # caller / next capture (GraphRunner._capture_graphs already runs it).
        self.graph_map = {}
        self.layer_range_graph_map = {}
        self.layer_range_group_ends = {}
        self.layer_range_batch_sizes = set()
        self._layer_range_hidden_input = None
        self._layer_range_residual_input = None
        self._prepared_layer_range_batch = None
        self.buffer = None
        gc.collect()
