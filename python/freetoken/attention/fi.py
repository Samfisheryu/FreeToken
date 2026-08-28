from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Dict, List, Literal

import torch
from freetoken.core import Batch, get_global_ctx
from freetoken.distributed import get_tp_info
from freetoken.env import ENV
from freetoken.utils import div_even, init_logger

from .base import AttentionSpec, BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer import (
        BatchDecodeWithPagedKVCacheWrapper,
        BatchPrefillWithPagedKVCacheWrapper,
        CUDAGraphBatchDecodeWithPagedKVCacheWrapper,
    )
    from freetoken.models import ModelConfig


def _next_power_of_2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << math.ceil(math.log2(n))


logger = init_logger(__name__)


@dataclass
class FICaptureData(BaseCaptureData):
    @property
    def one_tensor(self) -> torch.Tensor:
        return self.seq_lens

    @property
    def indices(self) -> torch.Tensor:
        return self.page_table


@dataclass
class FIPathMetadata:
    """FlashInfer plan and tensors for one homogeneous attention path."""

    # fmt: off
    kind:                Literal["decode", "prefill"]
    cu_seqlens_q_cpu:   torch.Tensor  # on cpu
    cu_seqlens_k_cpu:   torch.Tensor  # on cpu
    indices:            torch.Tensor  # on backend device
    last_page_len_cpu:  torch.Tensor  # on cpu
    num_qo_heads:       int
    num_kv_heads:       int
    head_dim:           int
    page_size:          Literal[1] # currently only support page_size=1
    pos_encoding_mode:  str
    seq_lens_cpu:       torch.Tensor  # on cpu
    dtype:              torch.dtype
    num_query_tokens:   int
    wrapper:            BatchPrefillWithPagedKVCacheWrapper | BatchDecodeWithPagedKVCacheWrapper | CUDAGraphBatchDecodeWithPagedKVCacheWrapper
    initialized:        bool = False
    # fmt: on

    def __post_init__(self) -> None:
        assert self.page_size == 1, "Currently only page_size=1 is supported."
        assert (
            self.cu_seqlens_k_cpu.is_cpu
            and self.cu_seqlens_q_cpu.is_cpu
            and self.last_page_len_cpu.is_cpu
            and self.seq_lens_cpu.is_cpu
        )


@dataclass
class FIMetadata(BaseAttnMetadata):
    """Batch-level FlashInfer metadata with decode-first path views."""

    query_indptr: torch.Tensor
    decode: FIPathMetadata | None = None
    prefill: FIPathMetadata | None = None

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.query_indptr[1 : 1 + bs] - 1


class FlashInferBackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig) -> None:
        from flashinfer import (
            BatchDecodeWithPagedKVCacheWrapper,
            BatchPrefillWithPagedKVCacheWrapper,
        )

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.device = self.kvcache.device
        # fa2 split-KV prefill needs ``tmp_v <= qo_heads_local * padded_batch_size *
        # cta_tile_q * head_dim * 4`` bytes of scratch, where flashinfer's scheduler
        # caps ``padded_batch_size`` at ~``2 * SM / kv_heads_local`` and
        # ``cta_tile_q`` is 128 (64 at head_dim >= 256); when the cap can't be met
        # it disables split-KV and allocates no tmp_v at all. The original flat
        # 128 MiB overflowed on head_dim=256 extend-prefills (Qwen3.5/3.6 MoE) and
        # the flat 256 MiB on MiniMax-M3's 64-head dense layers (H100, 132 SMs:
        # 64 heads x ceil(2*132/4)=66 padded rows x 128 x 128 x 4 B = 264 MiB of
        # tmp_v, over the flat buffer). Derive the bound
        # from the model's TP-LOCAL geometry + this device's SM count, with slack
        # for the tmp_s/merge siblings, floored at the old 256 MiB -- geometries
        # that never exceeded the flat buffer (e.g. GLM-4.7's 96q/8kv) stay at it.
        tp_size = get_tp_info().size
        qo_local = div_even(config.num_qo_heads, tp_size)
        kv_local = div_even(config.num_kv_heads, tp_size, allow_replicate=True)
        sm_count = (
            torch.cuda.get_device_properties(self.device).multi_processor_count
            if self.device.type == "cuda"
            else 128
        )
        cta_tile_q = 64 if config.head_dim >= 256 else 128
        padded_batch = -(-2 * sm_count // max(1, kv_local))
        tmp_v_bound = qo_local * padded_batch * cta_tile_q * config.head_dim * 4
        workspace_bytes = max(256 * 1024 * 1024, tmp_v_bound + 32 * 1024 * 1024)
        self.float_workspace_buffer = torch.empty(
            workspace_bytes, dtype=torch.uint8, device=self.device
        )
        self.prefill_wrapper = BatchPrefillWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )
        self.decode_wrappers = BatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            use_tensor_cores=self.use_tensor_cores,
            kv_layout="NHD",
            backend="fa2",  # flashinfer fa3 is slow, use fa2 instead
        )

        # Eager prefill/decode plans must retain their own integer workspaces. CUDA-graph
        # decode wrappers reuse the eager decode workspace below; graph and eager decode are
        # mutually exclusive for a forward and the graph wrapper is replanned before replay.
        self.int_workspace_buffer = self.decode_wrappers._int_workspace_buffer

        # initialize some data members
        tp_size = get_tp_info().size
        self.qo_head_local = div_even(self.config.num_qo_heads, tp_size)
        self.kv_head_local = div_even(self.config.num_kv_heads, tp_size, allow_replicate=True)

        self.cached_ones_cpu: torch.Tensor = torch.tensor(
            [], dtype=torch.int32, pin_memory=self.device.type == "cuda"
        )
        # for cuda graph
        self.capture_bs: List[int] = []
        self.max_graph_bs = 0
        self.graph_wrappers: Dict[int, CUDAGraphBatchDecodeWithPagedKVCacheWrapper] = {}
        self.capture: FICaptureData | None = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize_path_once(self, metadata: FIPathMetadata) -> None:
        if metadata.initialized:
            return

        metadata.initialized = True
        # FlashInfer planning reuses a pinned host staging buffer and launches an
        # async H2D copy. Wait here before the next plan mutates that host buffer.
        self.last_event.synchronize()
        if metadata.kind == "decode":
            metadata.wrapper.plan(
                indptr=metadata.cu_seqlens_k_cpu,
                indices=metadata.indices,
                last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                data_type=metadata.dtype,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
            )
        else:
            metadata.wrapper.plan(
                qo_indptr=metadata.cu_seqlens_q_cpu,
                paged_kv_indptr=metadata.cu_seqlens_k_cpu,
                paged_kv_indices=metadata.indices,
                paged_kv_last_page_len=metadata.last_page_len_cpu,
                num_qo_heads=metadata.num_qo_heads,
                num_kv_heads=metadata.num_kv_heads,
                head_dim_qk=metadata.head_dim,
                page_size=metadata.page_size,
                pos_encoding_mode=metadata.pos_encoding_mode,
                seq_lens=metadata.seq_lens_cpu,
                q_data_type=metadata.dtype,
                kv_data_type=metadata.dtype,
                non_blocking=True,
                causal=True,
            )
        self.last_event.record()

    def _initialize_metadata_once(self, metadata: FIMetadata) -> None:
        if metadata.decode is not None:
            self._initialize_path_once(metadata.decode)
        if metadata.prefill is not None:
            self._initialize_path_once(metadata.prefill)

    def _get_ones_cpu(self, bs: int) -> torch.Tensor:
        if bs <= len(self.cached_ones_cpu):
            return self.cached_ones_cpu[:bs]
        # padding to next pow of 2
        next_len = _next_power_of_2(bs)
        self.cached_ones_cpu = torch.ones(
            next_len, dtype=torch.int32, pin_memory=self.device.type == "cuda"
        )
        return self.cached_ones_cpu[:bs]

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_id: int,
        batch: Batch,
        attn_spec: AttentionSpec | None = None,
    ) -> torch.Tensor:
        if attn_spec is not None:
            # This backend has no window/sinks/sm_scale plumbing; dropping the spec
            # silently would attend with the wrong scale or an unbounded window.
            raise ValueError("The fi backend does not support per-call AttentionSpec.")

        def _flatten_cache(cache: torch.Tensor) -> torch.Tensor:  # treat page = 1
            return cache.view(-1, 1, cache.shape[2], cache.shape[3])

        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        self._initialize_metadata_once(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        kv_cache = (self.kvcache.k_cache(layer_id), self.kvcache.v_cache(layer_id))
        kv_cache = (_flatten_cache(kv_cache[0]), _flatten_cache(kv_cache[1]))

        if metadata.decode is None:
            assert metadata.prefill is not None
            return metadata.prefill.wrapper.run(q=q, paged_kv_cache=kv_cache)
        if metadata.prefill is None:
            return metadata.decode.wrapper.run(q=q, paged_kv_cache=kv_cache)

        # Mixed batches are decode-first. FlashInfer supports writing directly into an
        # output view, so both specialized kernels preserve token order without a cat/copy.
        decode_tokens = metadata.decode.num_query_tokens
        output = torch.empty_like(q)
        metadata.decode.wrapper.run(
            q=q[:decode_tokens],
            paged_kv_cache=kv_cache,
            out=output[:decode_tokens],
        )
        metadata.prefill.wrapper.run(
            q=q[decode_tokens:],
            paged_kv_cache=kv_cache,
            out=output[decode_tokens:],
        )
        return output

    def _build_path_metadata(
        self,
        reqs,
        *,
        kind: Literal["decode", "prefill"],
        wrapper,
        page_table: torch.Tensor,
        cpu_kwargs: dict,
    ) -> FIPathMetadata:
        seqlens_q = [req.extend_len for req in reqs]
        seqlens_k = [req.device_len for req in reqs]
        cached_lens = [req.cached_len for req in reqs]
        seq_lens_cpu = torch.tensor(seqlens_k, **cpu_kwargs)
        cu_seqlens_k_cpu = torch.tensor([0, *seqlens_k], **cpu_kwargs).cumsum_(0)
        if kind == "decode":
            cu_seqlens_q_cpu = torch.arange(len(reqs) + 1, **cpu_kwargs)
        elif all(length == 0 for length in cached_lens):
            cu_seqlens_q_cpu = cu_seqlens_k_cpu
        else:
            cu_seqlens_q_cpu = torch.tensor([0, *seqlens_q], **cpu_kwargs).cumsum_(0)

        return FIPathMetadata(
            kind=kind,
            cu_seqlens_q_cpu=cu_seqlens_q_cpu,
            cu_seqlens_k_cpu=cu_seqlens_k_cpu,
            indices=torch.cat(
                [page_table[req.table_idx, : req.device_len] for req in reqs]
            ),
            last_page_len_cpu=self._get_ones_cpu(len(reqs)),
            num_qo_heads=self.qo_head_local,
            num_kv_heads=self.kv_head_local,
            head_dim=self.config.head_dim,
            page_size=1,
            pos_encoding_mode="NONE",
            seq_lens_cpu=seq_lens_cpu,
            dtype=self.kvcache.dtype,
            num_query_tokens=sum(seqlens_q),
            wrapper=wrapper,
        )

    def prepare_metadata(self, batch: Batch) -> None:
        page_table = get_global_ctx().page_table
        cpu_kwargs = {
            "device": "cpu",
            "dtype": torch.int32,
            "pin_memory": self.device.type == "cuda",
        }
        decode = None
        prefill = None
        if batch.has_decode:
            decode_reqs = batch.padded_reqs if batch.is_decode_only else batch.decode_reqs
            decode = self._build_path_metadata(
                decode_reqs,
                kind="decode",
                wrapper=self.decode_wrappers,
                page_table=page_table,
                cpu_kwargs=cpu_kwargs,
            )
        if batch.has_prefill:
            prefill = self._build_path_metadata(
                batch.prefill_reqs,
                kind="prefill",
                wrapper=self.prefill_wrapper,
                page_table=page_table,
                cpu_kwargs=cpu_kwargs,
            )

        if decode is not None and prefill is not None:
            query_lens = [req.extend_len for req in batch.reqs]
            query_indptr_cpu = torch.tensor(
                [0, *query_lens], **cpu_kwargs
            ).cumsum_(0)
        else:
            path = decode if decode is not None else prefill
            assert path is not None
            query_indptr_cpu = path.cu_seqlens_q_cpu

        batch.attn_metadata = FIMetadata(
            query_indptr=query_indptr_cpu.to(self.device, non_blocking=True),
            decode=decode,
            prefill=prefill,
        )

    def prepare_metadata_view(self, source: Batch, target: Batch) -> bool:
        metadata = source.attn_metadata
        if not isinstance(metadata, FIMetadata):
            raise TypeError("FlashInfer metadata view requires FlashInfer source metadata")
        if target.is_decode_only:
            if metadata.decode is None:
                raise TypeError("FlashInfer source metadata has no decode path")
            target.attn_metadata = FIMetadata(
                query_indptr=metadata.query_indptr[: target.decode_size + 1],
                decode=metadata.decode,
            )
            return True
        if metadata.prefill is None:
            return False
        request_start = source.decode_size
        row_start = sum(req.extend_len for req in source.decode_reqs)
        target.attn_metadata = FIMetadata(
            query_indptr=metadata.query_indptr[request_start:] - row_start,
            prefill=metadata.prefill,
        )
        return True

    def reset_capture(self) -> None:
        # Base clears the common capture scratch; additionally drop the per-bs decode graph
        # wrappers (their indptr/indices alias freed capture scratch). Preserves the
        # long-lived workspace buffers. Lets init_capture_graph re-run after a cache rebuild.
        super().reset_capture()
        self.graph_wrappers = {}

    def init_capture_graph(self, max_seq_len: int, bs_list: List[int]) -> None:
        assert self.capture is None, "Capture already initialized."
        max_bs = max(bs_list)
        capture = FICaptureData.create(max_bs, max_seq_len, self.kvcache.device)
        capture.page_table = capture.page_table.view(-1)  # use 1D as ragged indices
        self.max_graph_bs = max_bs
        self.capture = capture
        self.capture_bs = sorted(bs_list)

    @cached_property
    def use_tensor_cores(self) -> bool:
        if (overriden_value := ENV.FLASHINFER_USE_TENSOR_CORES.value) is not None:
            logger.warning(f"Overriding FlashInfer tensor core usage to {overriden_value}")
            return overriden_value
        GQA = self.config.num_qo_heads // self.config.num_kv_heads
        return GQA >= 4

    def prepare_for_capture(self, batch: Batch) -> None:
        from flashinfer import CUDAGraphBatchDecodeWithPagedKVCacheWrapper

        bs = batch.size
        assert bs in self.capture_bs and bs not in self.graph_wrappers and self.capture
        capture = self.capture
        self.graph_wrappers[bs] = CUDAGraphBatchDecodeWithPagedKVCacheWrapper(
            self.float_workspace_buffer,
            kv_layout="NHD",
            use_tensor_cores=self.use_tensor_cores,
            indptr_buffer=capture.cu_seqlens_k[: bs + 1],
            indices_buffer=capture.indices,
            last_page_len_buffer=capture.one_tensor[:bs],
        )
        self.graph_wrappers[bs]._backend = "fa2"
        self.graph_wrappers[bs]._int_workspace_buffer = self.int_workspace_buffer
        self.prepare_metadata(batch)
        metadata = batch.attn_metadata
        assert isinstance(metadata, FIMetadata)
        assert metadata.decode is not None and metadata.prefill is None
        metadata.decode.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)

    def prepare_for_replay(self, batch: Batch) -> None:
        metadata, bs = batch.attn_metadata, batch.padded_size
        assert isinstance(metadata, FIMetadata)
        assert metadata.decode is not None and metadata.prefill is None
        assert not metadata.decode.initialized
        assert self.capture is not None and bs in self.capture_bs
        metadata.decode.wrapper = self.graph_wrappers[bs]
        self._initialize_metadata_once(metadata)
