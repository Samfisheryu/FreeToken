from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Iterator, Sequence

import torch
from flashlib.kernels.slot_cache import N_STATS, Stat, lru_ensure

# Fuse the per-bank expert copies into a single multi-bank launch (one per copy_missing
# instead of one per bank). Set FREETOKEN_FUSED_COPY=0 to force the legacy per-bank path
# (kept for A/B profiling). Falls back to per-bank automatically if a bank's row bytes or
# base address are not 16-byte aligned.
_FUSED_COPY = os.getenv("FREETOKEN_FUSED_COPY", "1").strip().lower() not in {"0", "false", "no", "off"}

# cudaMemcpyBatchAsync silently degrades to a SYNCHRONOUS copy when a batch mixes
# large entries with sub-~256KB entries on registered host memory (H100 + CUDA 13.0,
# empirically bisected: a single 5-22KB entry beside one large entry blocks the
# calling thread for the full transfer; >=253KB entries never do). A synchronous
# call still moves bytes at full PCIe rate but stalls the host, which un-hides the
# GEMM under the copy in transition-zone workloads (gpt-oss 2048tok: -22% e2e).
# Banks whose rows are smaller than this ship as ONE whole-layer entry (their
# whole layer is tiny) and are excluded from the hit gather, so every per-run
# entry the batch sees is >= this size.
_SMALL_BANK_FEAT_BYTES = 256 * 1024

# Joint group admission temporarily makes its existing hits newer than every
# ordinary LRU page.  Each full-layer ensure replaces that layer's sentinel with
# a strictly newer finite epoch; end-of-group normalizes all pages to the final
# admission epoch.
_JOINT_PINNED_USAGE = (1 << 63) - 1

from freetoken.utils import init_logger

logger = init_logger(__name__)

# quant_format -> bank names, in registration order: the single place a format's bank
# layout is declared. The cache machinery (copy_missing, the prefill double buffers,
# bank_views) iterates banks in this order, the layers' kernel dispatch unpacks views
# in this order, and set_bank_sources validates against it.
_BANK_SCHEMAS: dict[str, tuple[str, ...]] = {
    # dense bf16 expert weights
    "bf16": ("gate_up", "down"),
    # DeepSeek-V3-style 128x128 block-fp8 experts (Qwen3.5-FP8): fp8-e4m3 weights +
    # bf16 per-block weight_scale_inv. gate_up [L*E, 2I, H] fp8 + gate_up_scale
    # [L*E, 2I//128, H//128] bf16; down [L*E, H, I] fp8 + down_scale [L*E, H//128, I//128].
    # Half the host/cache footprint of bf16; the grouped GEMM (kernel/triton/fp8_blockscale_moe)
    # reads the routed fp8 rows directly and dequantizes in the K-loop (no bf16 materialization).
    "fp8_block": ("gate_up", "gate_up_scale", "down", "down_scale"),
    # native GGUF Q4_0 experts: packed block bytes per output row, dequantized inside
    # the borrowed ggml MoE kernels. gate_up [L*E, 2I, H//32*18], down [L*E, H, I//32*18].
    "q4_0": ("gate_up", "down"),
    # native ModelOpt rows for the Triton inline-dequant kernels: packed e2m1 codes +
    # fp8-e4m3 per-16 block scales + per-output-row fp16 globals (w1/w3 carry distinct
    # globals, and folding them into the e4m3 block scales would underflow)
    "nvfp4": (
        "gate_up_packed",
        "gate_up_scale",
        "gate_up_global",
        "down_packed",
        "down_scale",
        "down_global",
    ),
    # pre-tiled layouts for the borrowed kernels; the globals are folded into the
    # block scales at repack time and collapse to [L*E] GPU-resident alpha vectors
    # (set_alphas), so they are not banks
    "nvfp4_marlin": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    "nvfp4_b12x": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # gpt-oss mxfp4, transposed split-K layout (N innermost): per-expert blocks_t
    # [K//2, N] (uint8), scales_t [K//32, N] (uint8 e8m0), bias [N]. No folded alphas
    # (scales are a bank); split-K GEMV decode + transposed _t grouped prefill.
    "mxfp4_triton": (
        "gate_up_blocks",
        "gate_up_scales",
        "gate_up_bias",
        "down_blocks",
        "down_scales",
        "down_bias",
    ),
    # DeepSeek-V4 FP4: packed e2m1 codes + e8m0 per-32 block scales, no global scale
    # (4 banks). Read by DeepSeek-V4's own DS-FP4 grouped GEMV kernels via bank_views().
    "ds_fp4": ("gate_up_packed", "gate_up_scale", "down_packed", "down_scale"),
    # NoWAG: three projections, each with word-major packed assignments [W, N]
    # and input/output normalizers. The model-wide shared codebook is installed
    # separately once.
    "nowag": (
        "gate_assignments",
        "gate_input_norm",
        "gate_output_norm",
        "up_assignments",
        "up_input_norm",
        "up_output_norm",
        "down_assignments",
        "down_input_norm",
        "down_output_norm",
    ),
}

# vLLM's marlin grouped-GEMM hands the full [cache_size] slot cache as its expert
# dimension; moe_align_block_size requires round_up(experts, 32) < 1024, i.e. <= 992.
MARLIN_MAX_CACHE_SIZE = 992


@dataclass(frozen=True)
class ResidentExpertStage:
    """Opaque execution stage backed by one stable resident expert working set."""

    index: int
    start_layer: int
    end_layer: int


class ResidentExpertSession:
    """Own one resident-wave cache lifecycle without exposing slots or CUDA events."""

    def __init__(
        self,
        cache: OffloadMoeCache,
        stages: tuple[ResidentExpertStage, ...],
    ) -> None:
        self._cache = cache
        self._stages = stages
        self._active_index: int | None = None
        self._prefetched_index: int | None = None
        self._closed = False
        self._prepares_at_start = cache.prefill_layer_prepares

    @property
    def stage_count(self) -> int:
        return len(self._stages)

    @property
    def layer_prepares(self) -> int:
        return self._cache.prefill_layer_prepares - self._prepares_at_start

    def stage(self, index: int) -> ResidentExpertStage:
        if self._closed:
            raise RuntimeError("resident expert session is closed")
        try:
            return self._stages[index]
        except IndexError as exc:
            raise ValueError(f"resident stage index {index} is out of range") from exc

    def begin(self, index: int) -> ResidentExpertStage:
        stage = self.stage(index)
        if self._active_index == index:
            return stage
        if self._active_index is not None:
            raise RuntimeError("another resident expert stage is still active")
        self._cache.begin_resident_prefill_group(
            stage.start_layer,
            stage.end_layer,
        )
        self._active_index = index
        return stage

    def hint_next(self, index: int) -> bool:
        if self._active_index is None:
            raise RuntimeError("next-stage hint requires an active resident stage")
        if index != self._active_index + 1:
            raise ValueError("resident next-stage hint must be consecutive")
        stage = self.stage(index)
        prefetched = self._cache.try_prefetch_next_resident_group(
            stage.start_layer,
            stage.end_layer,
        )
        if prefetched:
            self._prefetched_index = index
        return prefetched

    def complete(self, index: int) -> None:
        if self._active_index != index:
            raise RuntimeError("completed resident stage is not active")
        self._cache.end_prefill_group()
        self._active_index = None
        if self._prefetched_index is not None:
            next_stage = self.stage(self._prefetched_index)
            self._cache.promote_prefetched_resident_group(
                next_stage.start_layer,
                next_stage.end_layer,
            )
            self._active_index = self._prefetched_index
            self._prefetched_index = None

    def close(self) -> None:
        if self._active_index is not None or self._prefetched_index is not None:
            raise RuntimeError("cannot close a resident expert session with active work")
        self._closed = True

    def cancel(self) -> None:
        if self._closed:
            return
        if self._active_index is not None:
            self._cache.end_prefill_group()
            self._active_index = None
        if self._prefetched_index is not None:
            self._cache.cancel_prefetched_resident_group()
            self._prefetched_index = None
        self._closed = True


def plan_expert_cache_partition(
    total_slots: int,
    num_experts: int,
    prefill_buffers: int = 2,
) -> dict[str, int]:
    """Split one HBM expert-row budget into decode cache and prefill buffers."""
    if num_experts < 1:
        raise ValueError("num_experts must be >= 1")
    if prefill_buffers < 0:
        raise ValueError("prefill_buffers must be >= 0")
    prefill_slots = prefill_buffers * num_experts
    decode_slots = total_slots - prefill_slots
    if decode_slots < num_experts:
        if prefill_buffers == 2:
            raise ValueError(
                "layered batching requires at least 3 * num_experts expert slots: "
                f"got total_slots={total_slots}, num_experts={num_experts}"
            )
        raise ValueError(
            "expert cache partition must leave at least num_experts decode slots: "
            f"got total_slots={total_slots}, num_experts={num_experts}, "
            f"prefill_buffers={prefill_buffers}"
        )
    return {
        "total_slots": total_slots,
        "decode_slots": decode_slots,
        "prefill_buffer_slots": prefill_slots,
    }


@dataclass
class OffloadMoeCache:
    num_layers: int
    num_experts: int
    cache_size: int
    device: torch.device
    cache_policy: str = "lru"
    prefill_overlap: bool = False
    # Layered batching keeps full-layer prefill buffers outside the decode slot
    # cache while preserving ``cache_size`` as the total HBM expert-row budget.
    # Legacy/mixed retain their aliasing layout; joint instead uses the canonical
    # slot pool below and therefore leaves this false.
    separate_prefill_buffer: bool = False
    # Shared-pool group-resident batching requests this many consecutive expert layers.
    # All prefill and decode routes share one canonical slot pool; an admitted
    # group pins its full G*E working set in that pool, then unpins (without
    # discarding) it after the group's queued compute completes.  Zero selects
    # the ordinary prefill layouts.
    prefill_group_size: int = 0
    # Layered-pipeline keeps one full expert layer available for decode outside
    # the persistent resident group.  Joint has no such reserve because its
    # decode and prefill rows traverse the same active group together.
    prefill_group_decode_reserve_layers: int = 0
    # Prefill hit/miss split: experts already resident in the slot cache (slots
    # >= 2 * num_experts) are gathered device-side into the double buffer instead
    # of re-crossing PCIe; only the misses are H2D'd (one cudaMemcpyBatchAsync of
    # coalesced runs). Requires prefill_overlap, cache_size > 2 * num_experts and
    # the fused copy plan; silently falls back to the full-layer copy otherwise.
    prefill_hit_d2d: bool = False
    # "bf16" (default, dense expert weights) or one of the NVFP4 bank layouts:
    # "nvfp4" (native ModelOpt rows, FreeToken Triton kernels), "nvfp4_marlin"
    # (Marlin-tiled, vLLM W4A16 GEMM, sm_80-99) or "nvfp4_b12x" (flashinfer SM12x
    # W4A16); or "mxfp4_triton" (gpt-oss transposed split-K GEMV decode + _t grouped
    # prefill). The format names its bank layout (_BANK_SCHEMAS) and which kernels
    # may read the banks; the cache machinery itself is layout-agnostic.
    quant_format: str = "bf16"
    # Decode mode + bank layout; per-layer CPU routing is cpu_layer_ids. "gpu":
    # GPU-tiled banks, all decode on GPU (stream misses over PCIe into the slot
    # cache, GEMM on GPU). "cpu": native (CPU-readable) banks + a CPU executor;
    # decode computes experts on the CPU (the slot cache only backs the prefill
    # double buffer). "hybrid": native banks + a CPU executor + a full slot cache;
    # each layer fetches a capped subset of its misses over PCIe (``hybrid_max_fetch``
    # / ``hybrid_fetch_fraction`` below; the GPU computes those plus the hits) and the
    # CPU absorbs the overflow misses, then the partials merge. The CPU executor is
    # attached (set_cpu_executor) for cpu/hybrid, set whenever >=1 layer decodes on the CPU.
    decode_target: str = "gpu"
    # hybrid only: max experts fetched over PCIe per (layer, decode step); the rest
    # of that step's misses are computed on the CPU. 0 -> never fetch (CPU does every
    # miss, the GPU cache stays cold); large -> behaves like pure offload.
    hybrid_max_fetch: int = 1
    # hybrid only: when > 0, replaces the fixed cap with a per-step fraction -- fetch
    # ~fraction * misses experts over PCIe (rounded to whichever integer balances the
    # overlap best), the CPU computes the rest. The engine sets it to the benched
    # pcie_bw / cpu_bw ratio so the PCIe fetch and the CPU overflow GEMV take equal
    # time (perfect overlap): fetched : cpu = pcie : cpu - pcie.
    hybrid_fetch_fraction: float = 0.0

    def __post_init__(self) -> None:
        policy_ids = {"lru": 0}
        assert self.cache_policy in policy_ids
        assert self.decode_target in ("gpu", "cpu", "hybrid"), self.decode_target
        assert self.quant_format in _BANK_SCHEMAS, f"unknown quant_format {self.quant_format!r}"
        # Attached by the engine for decode_target == "cpu" (CpuMoeExecutor); None
        # for the GPU decode path.
        self.cpu_executor = None
        # MoE layer ids whose decode runs on the CPU executor; the rest use the GPU
        # offload/PCIe path. Set by the engine after construction (empty = all-GPU,
        # all layers = the plain --moe-backend cpu case).
        self.cpu_layer_ids: frozenset = frozenset()
        assert self.prefill_group_decode_reserve_layers >= 0, (
            "prefill_group_decode_reserve_layers must be >= 0"
        )
        assert not self.prefill_group_decode_reserve_layers or self.prefill_group_size, (
            "decode reserve applies only to resident prefill groups"
        )
        # Cache attachment replaces this constructor registration with the
        # actual per-layer working sets.  The rectangular canonical bank still
        # validates equal expert-row counts today; stage packing itself no
        # longer leaks that geometry to the scheduler.
        self._resident_working_set_rows = (self.num_experts,) * self.num_layers
        # num_experts floor + nvfp4_marlin slot cap, shared with the runtime-rebuild path.
        self.validate_rebuild(self.cache_size)
        assert not self.separate_prefill_buffer or self.prefill_overlap, (
            "separate_prefill_buffer requires prefill_overlap"
        )
        assert self.prefill_group_size >= 0, "prefill_group_size must be >= 0"
        assert not self.prefill_group_size or not self.separate_prefill_buffer, (
            "joint group residency uses the canonical expert pool, not a separate buffer"
        )
        overlap_floor = (
            (1 + self.prefill_group_decode_reserve_layers) * self.num_experts
            if self.prefill_group_size
            else 2 * self.num_experts
        )
        assert not self.prefill_overlap or self.cache_size >= overlap_floor, (
            "Prefill overlap does not fit its expert working set: "
            f"cache_size={self.cache_size}, required_slots={overlap_floor}"
        )
        self.cache_policy_id = policy_ids[self.cache_policy]
        self.slot_for_id = torch.full(
            (self.num_layers, self.num_experts),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        # Reverse map, in the flat id space flashlib's slot_cache works in:
        # id == layer_id * num_experts + expert, so one array replaces the (layer,
        # expert) pair and evicting a slot needs no decode.
        self.id_of_slot = torch.full(
            (self.decode_cache_size,),
            -1,
            dtype=torch.int32,
            device=self.device,
        )
        self.usage = torch.zeros(
            (self.decode_cache_size,), dtype=torch.int64, device=self.device
        )
        self.step = torch.zeros((), dtype=torch.int64, device=self.device)
        self.active_mask = torch.zeros((self.num_experts,), dtype=torch.int32, device=self.device)
        # flashlib lru_ensure accepts arbitrary query length K and requires both
        # output plans to hold min(K, C) rows.  C is therefore the fixed upper
        # bound even when the logical expert domain E is smaller.
        self._allocate_lru_plan_buffers()
        self._allocate_joint_group_mask_buffers()
        self._allocate_resident_prefetch_plan_buffers()
        self.num_indices = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # Joint reuses the ordinary LRU admission entry point with an immutable
        # logical-id query and a separate reusable physical-slot output.
        self._joint_expert_ids: torch.Tensor | None = None
        self._joint_admit_ids: torch.Tensor | None = None
        if self.prefill_group_size:
            self._joint_expert_ids = torch.arange(
                self.num_experts, dtype=torch.int32, device=self.device
            )
            self._joint_admit_ids = torch.empty_like(self._joint_expert_ids)
        # hybrid only: full missing count BEFORE the per-step fetch cap (num_indices holds
        # the capped count that copy_missing actually fetches). The difference is what the
        # CPU computes this step. Written by the hybrid ensure kernel.
        self.num_missing_full = torch.zeros((1,), dtype=torch.int64, device=self.device)
        # hybrid only: per-(layer, expert) last-active decode step (LRU on the expert), -1
        # if never active. The hybrid ensure kernel reads it to pick which capped misses to
        # fetch (most-recently active first) and bumps it for every active expert.
        self.expert_recency = torch.full(
            (self.num_layers, self.num_experts), -1, dtype=torch.int64, device=self.device
        )
        # Host source banks (one [num_experts, ...] tensor per layer, so layers can
        # carry independent host attributes -- see layer_residency) and their GPU
        # slot caches, keyed by the format's bank schema (attached by
        # set_bank_sources). The GPU slot cache stays one unified pool per bank.
        self.bank_schema = _BANK_SCHEMAS[self.quant_format]
        self.bank_sources: dict[str, list[torch.Tensor]] = {}
        self.bank_caches: dict[str, torch.Tensor] = {}
        # Per-layer host residency (HostResidency values). The GPU movement paths
        # (fused gather, prefill DMA) require "pinned"; other residency classes
        # are not supported here and are rejected by set_bank_sources.
        self.layer_residency: list[str] = []
        # marlin/b12x per-expert global scales ([L*E], GPU resident, see set_alphas).
        self.gate_up_alpha: torch.Tensor | None = None
        self.down_alpha: torch.Tensor | None = None
        # Joint cannot derive per-slot scales from the globally mutable inverse
        # map while later group layers are admitted on the copy stream.  Keep one
        # stable full-slot scale view per resident group position instead.
        self._joint_gate_up_alpha_slots: torch.Tensor | None = None
        self._joint_down_alpha_slots: torch.Tensor | None = None
        # One model-wide NoWAG codebook; it is not replicated per cache slot.  Keep
        # both views alive: GPU offload reads ``codebook`` while cpu/hybrid reads the
        # original host tensor directly from the persistent CPU worker pool.
        self.codebook: torch.Tensor | None = None
        self.host_codebook: torch.Tensor | None = None
        # Opt-in decode miss-rate instrumentation. Accumulated on-device (no per-step host
        # sync); read via ``decode_miss_stats``. Graph-safe: the ``+=`` is captured into the
        # decode graph and re-executes with each replay's REAL routing (record_decode_stats
        # must be enabled before capture — see engine graph setup). The only graph artifact
        # is a one-off warm-up increment at capture time (<0.1% over a session).
        self.collect_stats = False
        # [num_layers, N_STATS] -- ensure_experts passes lru_stats[layer_id] straight to
        # the kernel, which accumulates in the same launch. The stat_* tensors below stay
        # for the hybrid path, whose kernel is still ours.
        self.lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )
        self.stat_missing = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_active = torch.zeros((), dtype=torch.int64, device=self.device)
        self.stat_calls = torch.zeros((), dtype=torch.int64, device=self.device)
        # hybrid only: experts actually fetched over PCIe (<= stat_missing). The CPU
        # computes stat_missing - stat_fetched of them.
        self.stat_fetched = torch.zeros((), dtype=torch.int64, device=self.device)
        # Per-layer counterparts of the scalars above (indexed by MoE-layer id). Same
        # device-side accumulation (graph-safe: layer_id is a static index per graph node),
        # so one req's per-layer miss rate is readable via decode_miss_stats_per_layer().
        self.stat_missing_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_active_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_fetched_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        self.stat_steps_layer = torch.zeros(self.num_layers, dtype=torch.int64, device=self.device)
        # Opt-in decode routing histogram (per layer, per expert) for cache-skew
        # analysis. Accumulated in ``ensure_experts`` from the raw expert ids before the
        # kernel rewrites them to slots. Only accurate with CUDA graphs disabled (the
        # captured graph would not re-run this host-side scatter on replay).
        self.collect_decode_freq = False
        self.decode_freq = torch.zeros(
            (self.num_layers, self.num_experts), dtype=torch.int64, device=self.device
        )
        # (per-layer sources, cache) per bank, in schema order. Every piece of cache
        # machinery that moves bank bytes (copy_missing, the prefill double buffers,
        # bank_views) iterates this list, so the slot cache is bank-count agnostic.
        self.banks: list[tuple[list[torch.Tensor], torch.Tensor]] = []
        # Fused multi-bank copy descriptor (built by set_bank_sources/_build_copy_plan).
        # Source pointers are per layer (_copy_src_ptrs[layer_id] -> [num_banks] device
        # tensor); dst/feat are layer-invariant.
        self._copy_fused_ok = False
        self._copy_dst_ptrs: torch.Tensor | None = None
        self._copy_src_ptrs: list[torch.Tensor] | None = None
        self._copy_feat_bytes: torch.Tensor | None = None
        # The layer whose misses ensure_experts/materialize_layer staged last; consumed
        # by copy_missing to pick the per-layer source (part of the same pending-copy
        # state as evict_slots/src_indices/num_indices).
        self._pending_src_layer: int | None = None
        # Per-bank [2, num_experts, ...] prefill buffers.  In the legacy layout these
        # alias the slot cache's first 2E rows; with separate_prefill_buffer they are
        # independent allocations charged against the same total ``cache_size`` budget.
        self.prefill_bank_buffers: list[torch.Tensor] = []
        self.prefill_copy_stream: torch.cuda.Stream | None = None
        self.prefill_begin_event: torch.cuda.Event | None = None
        self.prefill_ready_events: list[torch.cuda.Event] = []
        self.prefill_hit_ready_events: list[torch.cuda.Event] = []
        self.prefill_release_events: list[torch.cuda.Event] = []
        self._prefill_buffer_layer: list[int | None] = []
        self._prefill_buffer_released: list[bool] = []
        self._prefill_buffer_has_release_event: list[bool] = []
        self._prefill_buffer_has_hit_ready_event: list[bool] = []
        self._prefill_group_active = False
        self._prefill_group_target_layer: int | None = None
        self._resident_group_range: tuple[int, int] | None = None
        self._resident_group_ready_events: list[torch.cuda.Event] = []
        self._resident_prefetch_range: tuple[int, int] | None = None
        self._resident_prefetch_ready_events: list[torch.cuda.Event] = []
        self._resident_alternate_ready_events: list[torch.cuda.Event] = []
        self._joint_group_release_event: torch.cuda.Event | None = None
        self._joint_group_has_release_event = False
        self._resident_prefetch_plan_ready_event: torch.cuda.Event | None = None
        # hit-D2D split state: pinned begin-of-chunk snapshot of slot_for_id (the
        # classification input; frozen for the chunk -- no decode runs inside one,
        # and buffer invalidation only clears slot < 2E entries, which classify as
        # miss regardless), the lazily resolved batch-memcpy entry point (False =
        # unavailable), and row counters for cache reports.
        self._prefill_slot_snapshot: torch.Tensor | None = None
        self._prefill_snapshot_np = None
        self._prefill_hit_d2d_active = False
        self._hit_d2d_fallback_logged = False
        self._batch_memcpy = None
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.prefill_layer_prepares = 0
        self.prefill_h2d_bytes = 0
        self._prefill_full_layer_bytes = 0
        self._expert_row_bytes = 0
        # Joint's total is statically E rows per admitted layer, so keep it on
        # the host.  Dynamic misses accumulate in lru_ensure's existing stats
        # output, avoiding a separate device add after every admission.
        self.joint_prefill_total_rows = 0
        self.joint_prefill_lru_stats = torch.zeros(
            (self.num_layers, N_STATS), dtype=torch.int64, device=self.device
        )

    def _allocate_lru_plan_buffers(self) -> None:
        self.evict_slots = torch.empty(
            (self.decode_cache_size,), dtype=torch.int32, device=self.device
        )
        self.src_indices = torch.empty(
            (self.decode_cache_size,), dtype=torch.int32, device=self.device
        )

    def _allocate_joint_group_mask_buffers(self) -> None:
        self._joint_group_lower_mask: torch.Tensor | None = None
        self._joint_group_upper_mask: torch.Tensor | None = None
        if self.prefill_group_size:
            self._joint_group_lower_mask = torch.empty(
                (self.decode_cache_size,), dtype=torch.bool, device=self.device
            )
            self._joint_group_upper_mask = torch.empty_like(
                self._joint_group_lower_mask
            )

    def _allocate_resident_prefetch_plan_buffers(self) -> None:
        """Allocate copy plans that never alias decode's pending LRU plan."""
        self._resident_prefetch_evict_slots: torch.Tensor | None = None
        self._resident_prefetch_src_indices: torch.Tensor | None = None
        self._resident_prefetch_num_indices: torch.Tensor | None = None
        if not self.prefill_group_size:
            return
        count = self.prefill_buffer_count
        self._resident_prefetch_evict_slots = torch.empty(
            (count, self.num_experts), dtype=torch.int32, device=self.device
        )
        self._resident_prefetch_src_indices = torch.empty_like(
            self._resident_prefetch_evict_slots
        )
        self._resident_prefetch_num_indices = torch.zeros(
            (count, 1), dtype=torch.int64, device=self.device
        )

    @property
    def effective_prefill_group_size(self) -> int:
        """Actual resident layer count after honoring the total HBM slot budget."""
        if self.prefill_group_size == 0:
            return 0
        stages = self._resident_stage_ranges()
        return max((end - start for start, end in stages), default=0)

    def register_resident_working_sets(self, rows_per_layer: Sequence[int]) -> None:
        """Register the model adapter's logical expert working set per stage."""
        rows = tuple(int(rows) for rows in rows_per_layer)
        if len(rows) != self.num_layers or any(rows < 1 for rows in rows):
            raise ValueError(
                "resident working sets must provide one positive row count per layer"
            )
        if any(rows != self.num_experts for rows in rows):
            raise ValueError(
                "the canonical expert bank currently requires equal per-layer row counts"
            )
        self._resident_working_set_rows = rows

    def _resident_stage_ranges(self) -> tuple[tuple[int, int], ...]:
        if self.prefill_group_size == 0:
            return ()
        reserve_rows = (
            self.prefill_group_decode_reserve_layers
            * max(self._resident_working_set_rows)
        )
        stage_capacity = self.decode_cache_size - reserve_rows
        ranges: list[tuple[int, int]] = []
        start = 0
        while start < self.num_layers:
            end = start
            used = 0
            while (
                end < self.num_layers
                and end - start < self.prefill_group_size
                and used + self._resident_working_set_rows[end] <= stage_capacity
            ):
                used += self._resident_working_set_rows[end]
                end += 1
            if end == start:
                break
            ranges.append((start, end))
            start = end
        return tuple(ranges)

    @property
    def prefill_buffer_count(self) -> int:
        if self.prefill_group_size:
            return self.effective_prefill_group_size
        return 2

    @property
    def prefill_buffer_slots(self) -> int:
        return (
            self.prefill_buffer_count * self.num_experts
            if self.separate_prefill_buffer
            else 0
        )

    @property
    def decode_cache_size(self) -> int:
        """Number of physical rows visible to decode's slot map and LRU."""
        return self.cache_size - self.prefill_buffer_slots

    def cache_partition(self) -> dict[str, int]:
        """Public HBM expert-row geometry for scheduler/status reporting."""
        return plan_expert_cache_partition(
            self.cache_size,
            self.num_experts,
            self.prefill_buffer_count if self.separate_prefill_buffer else 0,
        )

    def resident_stages(self) -> tuple[ResidentExpertStage, ...]:
        """Return the cache-owned execution stages for one resident wave.

        The cache registration already names the expert working set of every
        offloaded layer.  Keep the capacity arithmetic here so schedulers never
        infer it from a model's expert count.
        """
        ranges = self._resident_stage_ranges()
        if not ranges or ranges[-1][1] != self.num_layers:
            raise RuntimeError("resident expert stages do not fit in the cache")
        return tuple(
            ResidentExpertStage(index, start, end)
            for index, (start, end) in enumerate(ranges)
        )

    def open_resident_wave(self) -> ResidentExpertSession:
        """Open an opaque cache-residency lifecycle for one execution wave."""
        if self._prefill_group_active or self._resident_prefetch_range is not None:
            raise RuntimeError("another resident expert wave is already active")
        return ResidentExpertSession(self, self.resident_stages())

    def set_bank_sources(
        self,
        sources: dict[str, list[torch.Tensor]],
        layer_residency: list[str] | None = None,
    ) -> None:
        """Attach the host (CPU pinned) expert source banks and allocate a GPU slot
        cache per bank, following the format's bank schema.

        Every bank is a list of ``num_layers`` tensors, one ``[num_experts, ...]``
        per layer (independent allocations, so each layer can carry its own host
        attributes); each slot cache mirrors the bank's row shape and dtype as one
        unified GPU pool. The row layouts are produced by the weight loaders /
        repackers (see ``_BANK_SCHEMAS`` and :mod:`freetoken.moe.nvfp4_backends`)
        -- the cache machinery is layout-agnostic and just moves rows.

        ``layer_residency`` labels each layer with a ``HostResidency`` value
        (default: all pinned). Non-pinned layers have no device address, so the
        GPU movement paths cannot serve them and they are rejected here
        (platform-specific residency policies are not implemented).
        """
        from freetoken.moe.host_banks import HostResidency

        assert set(sources) == set(self.bank_schema), (
            f"banks {sorted(sources)} do not match the {self.quant_format!r} "
            f"schema {self.bank_schema}"
        )
        residency = layer_residency or [HostResidency.PINNED.value] * self.num_layers
        assert len(residency) == self.num_layers, (len(residency), self.num_layers)
        if any(r != HostResidency.PINNED.value for r in residency):
            raise NotImplementedError(
                "non-pinned host bank layers need platform-specific movement "
                "paths that are not implemented; only pinned layers are served"
            )
        self.layer_residency = list(residency)
        for name in self.bank_schema:
            per_layer = sources[name]
            assert len(per_layer) == self.num_layers, (name, len(per_layer))
            head = per_layer[0]
            for layer_id, source in enumerate(per_layer):
                assert source.is_contiguous(), f"bank {name!r} layer {layer_id} must be contiguous"
                assert source.size(0) == self.num_experts, (name, layer_id, source.shape)
                assert source.shape == head.shape and source.dtype == head.dtype, (
                    name, layer_id, source.shape, source.dtype,
                )
            self.bank_sources[name] = list(per_layer)
            self.bank_caches[name] = torch.empty(
                (self.decode_cache_size, *head.shape[1:]),
                dtype=head.dtype,
                device=self.device,
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._expert_row_bytes = sum(
            source[0][0].numel() * source[0].element_size()
            for source in self.bank_sources.values()
        )
        self._build_copy_plan()
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def _build_copy_plan(self) -> None:
        """Precompute the fused multi-bank copy descriptor (base addrs + per-row bytes).

        Built once here (and on :meth:`rebuild`, which reallocates the slot caches);
        the addresses are fixed for the cache's lifetime so the descriptor tensors are
        CUDA-graph safe. Disabled (-> per-bank fallback) if any bank's row bytes or base
        address is not 16-byte aligned, or via FREETOKEN_FUSED_COPY=0.
        """
        self._copy_fused_ok = False
        self._copy_dst_ptrs = None
        self._copy_src_ptrs = None
        self._copy_feat_bytes = None
        self._copy_dst_ptrs_host: list[int] = []
        self._copy_src_ptrs_host: list[list[int]] = []
        self._copy_feat_bytes_host: list[int] = []
        self._gather_bank_ids: list[int] = []
        self._gather_dst_ptrs: torch.Tensor | None = None
        self._gather_feat_bytes: torch.Tensor | None = None
        if not _FUSED_COPY or self.device.type != "cuda" or not self.banks:
            return
        from freetoken.kernel.pinned import device_ptr

        dst_ptrs, feats = [], []
        layer_src_ptrs = [[] for _ in range(self.num_layers)]
        for per_layer, cache in self.banks:
            feat = math.prod(per_layer[0].shape[1:]) * per_layer[0].element_size()
            if feat % 16 != 0 or cache.data_ptr() % 16 != 0:
                return  # leave fused disabled; copy_missing uses the per-bank path
            for layer_id, source in enumerate(per_layer):
                # The kernel dereferences these on the GPU, so store each host bank's
                # device alias (== data_ptr() under UVA identity; differs on
                # Windows/WDDM).
                src_dev = device_ptr(source)
                if src_dev % 16 != 0:
                    return
                layer_src_ptrs[layer_id].append(src_dev)
            dst_ptrs.append(cache.data_ptr())
            feats.append(feat)
        self._copy_dst_ptrs = torch.tensor(dst_ptrs, dtype=torch.int64, device=self.device)
        self._copy_src_ptrs = [
            torch.tensor(ptrs, dtype=torch.int64, device=self.device)
            for ptrs in layer_src_ptrs
        ]
        self._copy_feat_bytes = torch.tensor(feats, dtype=torch.int64, device=self.device)
        self._copy_dst_ptrs_host = dst_ptrs
        self._copy_src_ptrs_host = layer_src_ptrs
        self._copy_feat_bytes_host = feats
        # hit-D2D gather serves only the big banks; small banks are whole-layer
        # H2D entries (see _SMALL_BANK_FEAT_BYTES), so their rows never need D2D.
        self._gather_bank_ids = [i for i, f in enumerate(feats) if f >= _SMALL_BANK_FEAT_BYTES]
        if len(self._gather_bank_ids) == len(feats):
            self._gather_dst_ptrs = self._copy_dst_ptrs
            self._gather_feat_bytes = self._copy_feat_bytes
        elif self._gather_bank_ids:
            self._gather_dst_ptrs = self._copy_dst_ptrs[self._gather_bank_ids].contiguous()
            self._gather_feat_bytes = self._copy_feat_bytes[self._gather_bank_ids].contiguous()
        self._copy_fused_ok = True

    def validate_rebuild(self, cache_size: int) -> None:
        """Pure geometry validation of a rebuild target (no GPU side effects).

        Raises ``ValueError`` if ``cache_size`` is below the policy working-set floor
        or above the marlin slot cap. Called by :meth:`rebuild` and by the engine's
        pre-teardown check, so an invalid target rejects with the old cache intact.
        """
        partition = plan_expert_cache_partition(
            cache_size,
            self.num_experts,
            2 if self.separate_prefill_buffer else 0,
        )
        if self.prefill_group_size:
            required_layers = 1 + self.prefill_group_decode_reserve_layers
            if cache_size < required_layers * self.num_experts:
                if self.prefill_group_decode_reserve_layers:
                    raise ValueError(
                        "resident layered prefill requires at least two expert layers "
                        "of shared cache"
                    )
                raise ValueError(
                    "joint group batching requires at least num_experts expert slots: "
                    f"got total_slots={cache_size}, num_experts={self.num_experts}"
                )
        decode_size = partition["decode_slots"]
        if self.quant_format == "nvfp4_marlin" and decode_size > MARLIN_MAX_CACHE_SIZE:
            raise ValueError(
                f"decode expert cache size {decode_size} exceeds the marlin backend's slot limit of "
                f"{MARLIN_MAX_CACHE_SIZE} (vLLM moe_align_block_size caps padded experts at "
                "1024); reduce moe_cache_size or force --nvfp4-backend triton"
            )

    def rebuild(self, cache_size: int) -> None:
        """Resize the GPU slot cache + bookkeeping to ``cache_size`` IN PLACE.

        Keeps the CPU/pinned ``bank_sources`` and the GPU-resident alphas; never
        reloads banks. Tears down prefill-overlap buffers first (their views alias
        the old ``bank_caches``), frees the old GPU tensors, then reallocates. Slots
        cold-start after rebuild. Object identity is preserved so attached layers and
        ``ctx.moe_offload_cache`` stay valid.
        """
        assert self.bank_sources, "set_bank_sources must run before rebuild"
        self.validate_rebuild(cache_size)
        # 1. Tear down prefill-overlap (its buffer views alias the old bank_caches).
        self.prefill_bank_buffers = []
        self.prefill_copy_stream = None
        self.prefill_begin_event = None
        self.prefill_ready_events = []
        self.prefill_hit_ready_events = []
        self.prefill_release_events = []
        self._prefill_buffer_layer = []
        self._prefill_buffer_released = []
        self._prefill_buffer_has_release_event = []
        self._prefill_buffer_has_hit_ready_event = []
        self._prefill_group_active = False
        self._prefill_group_target_layer = None
        self._resident_group_range = None
        self._resident_group_ready_events = []
        self._resident_prefetch_range = None
        self._resident_prefetch_ready_events = []
        self._resident_alternate_ready_events = []
        self._joint_group_release_event = None
        self._joint_group_has_release_event = False
        self._resident_prefetch_plan_ready_event = None
        self._joint_gate_up_alpha_slots = None
        self._joint_down_alpha_slots = None
        # 2. Drop old GPU tensors (free-before-alloc).
        self.banks = []
        self.bank_caches = {}
        self.cache_size = cache_size
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        # 3. Reallocate the slot cache from the retained host sources.
        for name in self.bank_schema:
            head = self.bank_sources[name][0]
            self.bank_caches[name] = torch.empty(
                (self.decode_cache_size, *head.shape[1:]), dtype=head.dtype, device=self.device
            )
        self.banks = [(self.bank_sources[n], self.bank_caches[n]) for n in self.bank_schema]
        self._build_copy_plan()  # slot caches were reallocated -> refresh fused-copy addrs
        # 4. Reallocate cache_size-shaped bookkeeping; reset the slot map (cold start).
        self.slot_for_id.fill_(-1)
        self.id_of_slot = torch.full(
            (self.decode_cache_size,), -1, dtype=torch.int32, device=self.device
        )
        self.usage = torch.zeros(
            (self.decode_cache_size,), dtype=torch.int64, device=self.device
        )
        self._allocate_lru_plan_buffers()
        self._allocate_joint_group_mask_buffers()
        self._allocate_resident_prefetch_plan_buffers()
        self.step.zero_()
        self.active_mask.zero_()
        self.num_indices.zero_()
        self.num_missing_full.zero_()
        self.expert_recency.fill_(-1)
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()
        self.decode_freq.zero_()
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.prefill_layer_prepares = 0
        self.prefill_h2d_bytes = 0
        self.joint_prefill_total_rows = 0
        self.joint_prefill_lru_stats.zero_()
        self._hit_d2d_fallback_logged = False  # geometry changed; re-log if still unusable
        # 5. Re-evaluate prefill overlap against the new size.
        overlap_floor = (
            (1 + self.prefill_group_decode_reserve_layers) * self.num_experts
            if self.prefill_group_size
            else 2 * self.num_experts
        )
        if self.prefill_overlap and cache_size < overlap_floor:
            logger.warning(
                f"Disabling MoE prefill overlap on rebuild: cache_size {cache_size} "
                f"< required expert slots {overlap_floor}."
            )
            self.prefill_overlap = False
        if self.prefill_overlap:
            self._init_prefill_overlap_buffers()

    def set_alphas(
        self, gate_up_alpha: torch.Tensor | None, down_alpha: torch.Tensor | None
    ) -> None:
        """Attach the marlin/b12x per-expert global scales (``[L*E]``, GPU resident).

        These are kernel-preprocessed scalars, far too small to bother offloading;
        the forward path looks them up per slot with :meth:`alphas_for_slots` /
        :meth:`alphas_for_layer` (pure device-side lookups, CUDA-graph safe).
        ``(None, None)`` is a no-op so callers can pass a format's (possibly
        absent) alphas through unconditionally.
        """
        if gate_up_alpha is None and down_alpha is None:
            return
        assert gate_up_alpha is not None and down_alpha is not None
        total = self.num_layers * self.num_experts
        assert gate_up_alpha.shape == down_alpha.shape == (total,)
        self.gate_up_alpha = gate_up_alpha.to(self.device)
        self.down_alpha = down_alpha.to(self.device)
        self._init_joint_alpha_slots()

    def _init_joint_alpha_slots(self) -> None:
        """Allocate stable per-group-position slot scales when joint needs them."""
        if not self.prefill_group_size or self.gate_up_alpha is None:
            self._joint_gate_up_alpha_slots = None
            self._joint_down_alpha_slots = None
            return
        shape = (self.prefill_buffer_count, self.decode_cache_size)
        self._joint_gate_up_alpha_slots = torch.empty(
            shape, dtype=self.gate_up_alpha.dtype, device=self.device
        )
        self._joint_down_alpha_slots = torch.empty(
            shape, dtype=self.down_alpha.dtype, device=self.device
        )

    def set_codebook(self, codebook: torch.Tensor | None) -> None:
        """Install the model-wide NoWAG codebook on the host and cache device."""
        if codebook is None:
            return
        if codebook.ndim != 2:
            raise ValueError(
                f"NoWAG codebook must be [entries, group_size], got {tuple(codebook.shape)}"
            )
        self.host_codebook = codebook.contiguous()
        self.codebook = self.host_codebook.to(self.device).contiguous()

    def set_cpu_executor(self, executor) -> None:
        """Attach the CPU MoE executor (``decode_target`` in {"cpu", "hybrid"}).

        The executor owns the persistent worker pool, the pinned activation/result
        IO buffers, and the ``cudaLaunchHostFunc`` submit/sync plumbing. It reads
        experts straight from this cache's host ``bank_sources`` (no extra copy).
        """
        assert self.decode_target in ("cpu", "hybrid"), (
            "set_cpu_executor requires decode_target in {'cpu','hybrid'}"
        )
        self.cpu_executor = executor

    def is_cpu_layer(self, layer_id: int) -> bool:
        """Whether ``layer_id`` decodes on the CPU executor (vs the GPU offload path)."""
        return layer_id in self.cpu_layer_ids

    def alphas_for_slots(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Per-slot global scales for a decode call, or ``None`` when the format
        keeps no GPU-resident alphas (bf16 / triton-nvfp4). Slots of other layers
        yield garbage values, but only slots routed to -- and those belong to
        ``layer_id`` -- are ever read by the grouped GEMM."""
        if self.gate_up_alpha is None:
            return None
        idx = layer_id * self.num_experts + (
            self.id_of_slot.clamp(min=0).long() % self.num_experts
        )
        return self.gate_up_alpha[idx], self.down_alpha[idx]

    def alphas_for_resident_layer_slots(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Stable full-slot scales for one active joint resident layer."""
        if self.gate_up_alpha is None:
            return None
        if self._resident_group_range is None:
            raise RuntimeError("no joint resident expert group is active")
        start, end = self._resident_group_range
        if not start <= layer_id < end:
            raise RuntimeError(
                f"layer {layer_id} is outside resident group [{start}, {end})"
            )
        assert self._joint_gate_up_alpha_slots is not None
        assert self._joint_down_alpha_slots is not None
        buffer_id = layer_id - start
        return (
            self._joint_gate_up_alpha_slots[buffer_id],
            self._joint_down_alpha_slots[buffer_id],
        )

    def alphas_for_layer(self, layer_id: int) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Global scales for a full-layer prefill (overlap or materialize), where
        position == expert id (contiguous slices, no gather); ``None`` when the
        format keeps no GPU-resident alphas."""
        if self.gate_up_alpha is None:
            return None
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        return self.gate_up_alpha[lo:hi], self.down_alpha[lo:hi]

    def bank_views(self, n: int | None = None) -> tuple[torch.Tensor, ...]:
        """Per-bank cache views in registration order: the full ``[S]`` slot cache
        (decode), or its first ``n`` slots (materialized layer)."""
        assert self.banks, "set_bank_sources must register the banks first"
        if n is None:
            return tuple(cache for _, cache in self.banks)
        return tuple(cache[:n] for _, cache in self.banks)

    def _init_prefill_overlap_buffers(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        count = self.prefill_buffer_count
        self._prefill_buffer_layer = [None] * count
        self._prefill_buffer_released = [True] * count
        self._prefill_buffer_has_release_event = [False] * count
        self._prefill_buffer_has_hit_ready_event = [False] * count
        if self.prefill_group_size:
            # Joint has no second physical buffer.  These events publish pages
            # admitted into the canonical slot bank and protect the prior
            # group's queued GEMMs before any page can be reused.
            self._init_joint_alpha_slots()
            self.prefill_bank_buffers = []
            self._prefill_full_layer_bytes = self.num_experts * self._expert_row_bytes
            if self.device.type == "cuda":
                self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
                self.prefill_ready_events = [torch.cuda.Event() for _ in range(count)]
                self.prefill_hit_ready_events = []
                self.prefill_release_events = []
                self.prefill_begin_event = torch.cuda.Event()
                self._joint_group_release_event = torch.cuda.Event()
                self._resident_prefetch_plan_ready_event = torch.cuda.Event()
                self._resident_alternate_ready_events = [
                    torch.cuda.Event() for _ in range(count)
                ]
            return
        # One full expert layer per buffer, one tensor per registered bank.  Layered
        # serving uses disjoint allocations so decode residency remains stable; the
        # legacy prefill path retains its aliasing layout and memory behavior.
        if self.separate_prefill_buffer:
            self.prefill_bank_buffers = [
                torch.empty(
                    (count, self.num_experts, *cache.shape[1:]),
                    dtype=cache.dtype,
                    device=self.device,
                )
                for _, cache in self.banks
            ]
        else:
            self.prefill_bank_buffers = [
                cache[: count * self.num_experts].view(
                    count, self.num_experts, *cache.shape[1:]
                )
                for _, cache in self.banks
            ]
        self._prefill_full_layer_bytes = sum(
            buffer[0].numel() * buffer.element_size()
            for buffer in self.prefill_bank_buffers
        )

        # Raw bases used by the hit/miss split.  Source indices are decode-cache
        # relative in both layouts; destinations index the flattened 2E prefill rows.
        self._prefill_dst_ptrs_host = [
            buffer.data_ptr() for buffer in self.prefill_bank_buffers
        ]
        if self.device.type == "cuda" and self._copy_fused_ok:
            self._prefill_gather_dst_ptrs = torch.tensor(
                [self._prefill_dst_ptrs_host[i] for i in self._gather_bank_ids],
                dtype=torch.int64,
                device=self.device,
            )
        else:
            self._prefill_gather_dst_ptrs = None
        if self.device.type == "cuda":
            self.prefill_copy_stream = torch.cuda.Stream(device=self.device)
            self.prefill_ready_events = [torch.cuda.Event() for _ in range(count)]
            self.prefill_hit_ready_events = [torch.cuda.Event() for _ in range(count)]
            self.prefill_release_events = [torch.cuda.Event() for _ in range(count)]
            self.prefill_begin_event = torch.cuda.Event()
        if self.prefill_hit_d2d and self.device.type == "cuda":
            self._prefill_slot_snapshot = torch.empty(
                (self.num_layers, self.num_experts), dtype=torch.int32, pin_memory=True
            )
            self._prefill_snapshot_np = self._prefill_slot_snapshot.numpy()
            self._prefill_hit_dst = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_src = torch.empty(
                (self.num_experts,), dtype=torch.int32, device=self.device
            )
            self._prefill_hit_num = torch.zeros((1,), dtype=torch.int64, device=self.device)

    def _invalidate_prefill_buffer(self, buffer_id: int) -> None:
        if self.separate_prefill_buffer:
            return
        slot_start = buffer_id * self.num_experts
        slot_end = slot_start + self.num_experts
        old_ids = self.id_of_slot[slot_start:slot_end]
        self.slot_for_id.view(-1)[old_ids[old_ids >= 0].long()] = -1
        old_ids.fill_(-1)
        # usage=0 makes these slots the oldest, so the argmin(usage) victim selection in
        # ensure_experts evicts them first.
        self.usage[slot_start:slot_end].zero_()

    def begin_prefill(self) -> None:
        if not self.prefill_overlap:
            return
        if self._prefill_group_active:
            return
        self._begin_prefill_buffers()

    def begin_prefill_group(self) -> None:
        """Start one multi-chunk layer-ordered prefill wave."""
        if not self.prefill_overlap:
            return
        if self._prefill_group_active:
            raise RuntimeError("a layer-group prefill wave is already active")
        self._prefill_group_active = True
        self._prefill_group_target_layer = None
        self._resident_group_range = None
        self._begin_prefill_buffers()

    def begin_resident_prefill_group(self, start_layer: int, end_layer: int) -> None:
        """Admit ``[start_layer, end_layer)`` into the canonical expert pool.

        Resident hits keep their physical pages and are protected before the first
        miss is assigned.  Misses use only non-group empty/LRU pages, so every
        admitted page remains stable until :meth:`end_prefill_group` records
        completion of the group's queued compute.
        """
        self._activate_resident_prefill_group(start_layer, end_layer)

        if self.device.type == "cuda" and self.prefill_group_decode_reserve_layers:
            # A persistent pipeline group admits on the compute stream before
            # decode traverses layers outside the group.  Decode and admission
            # both mutate the canonical LRU maps, so only the resulting copies
            # may overlap on the copy stream.  Their plans are independent of
            # decode's shared staging buffers.
            assert self.prefill_copy_stream is not None
            assert self._resident_prefetch_plan_ready_event is not None
            assert self._resident_prefetch_evict_slots is not None
            assert self._resident_prefetch_src_indices is not None
            assert self._resident_prefetch_num_indices is not None
            assert self._joint_expert_ids is not None
            assert self._joint_admit_ids is not None

            current_stream = torch.cuda.current_stream(self.device)
            if self._joint_group_has_release_event:
                current_stream.wait_event(self._joint_group_release_event)
            self._pin_resident_group_pages(start_layer, end_layer)
            for buffer_id, layer_id in enumerate(range(start_layer, end_layer)):
                src_indices = self._resident_prefetch_src_indices[buffer_id]
                evict_slots = self._resident_prefetch_evict_slots[buffer_id]
                num_indices = self._resident_prefetch_num_indices[buffer_id]
                lru_ensure(
                    self._joint_expert_ids,
                    self.slot_for_id.view(-1),
                    self.id_of_slot,
                    self.usage,
                    self.step,
                    self._joint_admit_ids,
                    src_indices,
                    evict_slots,
                    num_indices,
                    stats=self.joint_prefill_lru_stats[layer_id],
                    id_base=layer_id * self.num_experts,
                )
                self.joint_prefill_total_rows += self.num_experts
                self._prepare_joint_slot_alphas(buffer_id, layer_id)
                self.prefill_layer_prepares += 1
                self._prefill_buffer_layer[buffer_id] = layer_id

            # lru_ensure gives newly admitted pages finite recency.  Restore the
            # persistent pin before any prefix/suffix decode LRU can run.
            self._pin_resident_group_pages(start_layer, end_layer)
            self._resident_prefetch_plan_ready_event.record(current_stream)
            self.prefill_copy_stream.wait_event(
                self._resident_prefetch_plan_ready_event
            )
            with torch.cuda.stream(self.prefill_copy_stream):
                for buffer_id, layer_id in enumerate(range(start_layer, end_layer)):
                    self._copy_missing_plan(
                        layer_id,
                        self._resident_prefetch_evict_slots[buffer_id],
                        self._resident_prefetch_src_indices[buffer_id],
                        self._resident_prefetch_num_indices[buffer_id],
                    )
                    self.prefill_ready_events[buffer_id].record(
                        self.prefill_copy_stream
                    )
            return

        def admit() -> None:
            # Protect every existing hit in Q before assigning the first miss.
            # The inverse map names the same canonical pages as flat logical ids,
            # so one range mask avoids compacting/converting Q's sparse slot map.
            self._pin_resident_group_pages(start_layer, end_layer)

            group_layers = end_layer - start_layer
            for buffer_id, layer_id in enumerate(range(start_layer, end_layer)):
                assert self._joint_expert_ids is not None
                assert self._joint_admit_ids is not None
                self._pending_src_layer = layer_id
                if self.device.type == "cuda":
                    # The immutable full logical layer is the query; physical
                    # slots are emitted separately for the canonical bank.
                    lru_ensure(
                        self._joint_expert_ids,
                        self.slot_for_id.view(-1),
                        self.id_of_slot,
                        self.usage,
                        self.step,
                        self._joint_admit_ids,
                        self.src_indices,
                        self.evict_slots,
                        self.num_indices,
                        stats=self.joint_prefill_lru_stats[layer_id],
                        id_base=layer_id * self.num_experts,
                    )
                else:
                    self._joint_admit_ids.copy_(self._joint_expert_ids)
                    self._ensure_joint_layer_cpu(layer_id, self._joint_admit_ids)
                    joint_stats = self.joint_prefill_lru_stats[layer_id]
                    joint_stats[Stat.ACTIVE] += self.num_experts
                    joint_stats[Stat.MISS] += self.num_indices[0]
                    joint_stats[Stat.CALLS] += 1

                self.joint_prefill_total_rows += self.num_experts
                if self.device.type == "cuda":
                    self.copy_missing()
                else:
                    self._copy_joint_missing_cpu(layer_id)
                self._prepare_joint_slot_alphas(buffer_id, layer_id)
                self.prefill_layer_prepares += 1
                self._prefill_buffer_layer[buffer_id] = layer_id
                if (
                    self.prefill_group_decode_reserve_layers
                    and buffer_id + 1 == group_layers
                ):
                    # Resident layered groups may span scheduler iterations.
                    # Hard-pin their newly admitted pages before later decode
                    # prefix/suffix work can mutate the non-group LRU pages.
                    self._pin_resident_group_pages(start_layer, end_layer)
                if self.prefill_ready_events:
                    self.prefill_ready_events[buffer_id].record(
                        self.prefill_copy_stream
                    )

        if self.prefill_copy_stream is None:
            admit()
            return

        current_stream = torch.cuda.current_stream(self.device)
        self.prefill_begin_event.record(current_stream)
        self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        if self._joint_group_has_release_event:
            self.prefill_copy_stream.wait_event(self._joint_group_release_event)
        with torch.cuda.stream(self.prefill_copy_stream):
            admit()

    def _activate_resident_prefill_group(
        self, start_layer: int, end_layer: int
    ) -> None:
        if not self.prefill_overlap:
            raise RuntimeError("resident prefill groups require prefill overlap")
        if self._prefill_group_active:
            raise RuntimeError("a layer-group prefill wave is already active")
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise ValueError(
                f"invalid resident layer group [{start_layer}, {end_layer})"
            )
        if end_layer - start_layer > self.effective_prefill_group_size:
            raise ValueError(
                f"resident group has {end_layer - start_layer} layers but only "
                f"{self.effective_prefill_group_size} fit in the canonical pool"
            )
        self._prefill_group_active = True
        self._prefill_group_target_layer = None
        self._resident_group_range = (start_layer, end_layer)
        self._resident_group_ready_events = self.prefill_ready_events
        self._prefill_buffer_layer = [None] * self.prefill_buffer_count
        self._prefill_buffer_released = [False] * self.prefill_buffer_count
        self._prefill_buffer_has_hit_ready_event = [False] * self.prefill_buffer_count

    def _pin_resident_group_pages(self, start_layer: int, end_layer: int) -> None:
        group_id_start = start_layer * self.num_experts
        group_id_end = end_layer * self.num_experts
        assert self._joint_group_lower_mask is not None
        assert self._joint_group_upper_mask is not None
        torch.ge(
            self.id_of_slot,
            group_id_start,
            out=self._joint_group_lower_mask,
        )
        torch.lt(
            self.id_of_slot,
            group_id_end,
            out=self._joint_group_upper_mask,
        )
        self._joint_group_lower_mask.logical_and_(self._joint_group_upper_mask)
        self.usage.masked_fill_(
            self._joint_group_lower_mask, _JOINT_PINNED_USAGE
        )

    def try_prefetch_next_resident_group(
        self, start_layer: int, end_layer: int
    ) -> bool:
        """Prefetch a next group without releasing the current resident group.

        Mapping and LRU admission run on the current compute stream.  Copy plans
        and ready events are disjoint from both decode and the active group, so
        current-group compute can overlap the next group's H2D copies.
        """
        if self.device.type != "cuda":
            return False
        if not self._prefill_group_active or self._resident_group_range is None:
            raise RuntimeError("next-group prefetch requires an active resident group")
        if self._resident_prefetch_range is not None:
            raise RuntimeError("a next resident group is already prefetched")
        if not 0 <= start_layer < end_layer <= self.num_layers:
            raise ValueError(
                f"invalid next resident layer group [{start_layer}, {end_layer})"
            )
        group_layers = end_layer - start_layer
        if group_layers > self.effective_prefill_group_size:
            raise ValueError(
                f"next resident group has {group_layers} layers but only "
                f"{self.effective_prefill_group_size} fit in the canonical pool"
        )
        active_start, active_end = self._resident_group_range
        active_layers = active_end - active_start
        protected_slots = (active_layers + group_layers) * self.num_experts
        if self.decode_cache_size - protected_slots < self.num_experts:
            return False

        assert self.prefill_copy_stream is not None
        assert self._resident_prefetch_plan_ready_event is not None
        assert self._resident_prefetch_evict_slots is not None
        assert self._resident_prefetch_src_indices is not None
        assert self._resident_prefetch_num_indices is not None
        assert self._joint_expert_ids is not None
        assert self._joint_admit_ids is not None

        current_stream = torch.cuda.current_stream(self.device)
        # Admission for a newly started current group runs on the copy stream.
        # Waiting its ready events here keeps canonical maps single-writer while
        # still allowing next-group copies to overlap the final current compute.
        for event in self._resident_group_ready_events[:active_layers]:
            current_stream.wait_event(event)

        ready_events = (
            self._resident_alternate_ready_events
            if self._resident_group_ready_events is self.prefill_ready_events
            else self.prefill_ready_events
        )
        if len(ready_events) < group_layers:
            raise RuntimeError("resident next-group ready events are not initialized")

        self._pin_resident_group_pages(start_layer, end_layer)
        for buffer_id, layer_id in enumerate(range(start_layer, end_layer)):
            src_indices = self._resident_prefetch_src_indices[buffer_id]
            evict_slots = self._resident_prefetch_evict_slots[buffer_id]
            num_indices = self._resident_prefetch_num_indices[buffer_id]
            lru_ensure(
                self._joint_expert_ids,
                self.slot_for_id.view(-1),
                self.id_of_slot,
                self.usage,
                self.step,
                self._joint_admit_ids,
                src_indices,
                evict_slots,
                num_indices,
                stats=self.joint_prefill_lru_stats[layer_id],
                id_base=layer_id * self.num_experts,
            )
            self.joint_prefill_total_rows += self.num_experts
            self.prefill_layer_prepares += 1

        # lru_ensure replaces the admission sentinel with a finite epoch.  Restore
        # a hard pin only after all mappings exist, before decode can mutate LRU.
        self._pin_resident_group_pages(start_layer, end_layer)
        self._resident_prefetch_plan_ready_event.record(current_stream)

        self.prefill_copy_stream.wait_event(
            self._resident_prefetch_plan_ready_event
        )
        with torch.cuda.stream(self.prefill_copy_stream):
            for buffer_id, layer_id in enumerate(range(start_layer, end_layer)):
                self._copy_missing_plan(
                    layer_id,
                    self._resident_prefetch_evict_slots[buffer_id],
                    self._resident_prefetch_src_indices[buffer_id],
                    self._resident_prefetch_num_indices[buffer_id],
                )
                ready_events[buffer_id].record(self.prefill_copy_stream)
        self._resident_prefetch_range = (start_layer, end_layer)
        self._resident_prefetch_ready_events = ready_events
        return True

    def promote_prefetched_resident_group(
        self, start_layer: int, end_layer: int
    ) -> None:
        """Make an already pinned/copied next group the active direct-map group."""
        if self._prefill_group_active:
            raise RuntimeError("current resident group must be released before promotion")
        if self._resident_prefetch_range != (start_layer, end_layer):
            raise RuntimeError(
                f"prefetched resident group is {self._resident_prefetch_range}, "
                f"not [{start_layer}, {end_layer})"
            )
        ready_events = self._resident_prefetch_ready_events
        self._activate_resident_prefill_group(start_layer, end_layer)
        self._resident_group_ready_events = ready_events
        self._prefill_buffer_layer[: end_layer - start_layer] = range(
            start_layer, end_layer
        )
        for buffer_id, layer_id in enumerate(range(start_layer, end_layer)):
            self._prepare_joint_slot_alphas(buffer_id, layer_id)
        self._resident_prefetch_range = None
        self._resident_prefetch_ready_events = []

    def cancel_prefetched_resident_group(self) -> None:
        """Release an exceptional-path prefetch without exposing its events."""
        if self._resident_prefetch_range is None:
            return
        current_stream = torch.cuda.current_stream(self.device)
        for event in self._resident_prefetch_ready_events:
            current_stream.wait_event(event)
        start_layer, end_layer = self._resident_prefetch_range
        group_slots = self.slot_for_id[start_layer:end_layer].reshape(-1)
        self.usage[group_slots] = self.step
        if self._joint_group_release_event is not None:
            self._joint_group_release_event.record(current_stream)
            self._joint_group_has_release_event = True
        self._resident_prefetch_range = None
        self._resident_prefetch_ready_events = []

    def _ensure_joint_layer_cpu(
        self, layer_id: int, expert_ids: torch.Tensor
    ) -> None:
        """Pure-Torch reference for joint admission on a CPU cache.

        CUDA always uses flashlib's original ``lru_ensure``.  This mirror keeps
        the public cache contract executable without introducing another kernel:
        empty slots first, then finite-usage LRU, with physical-slot tie breaks.
        """
        flat = expert_ids.reshape(-1)
        seen: list[int] = []
        for expert in flat.tolist():
            expert = int(expert)
            if not 0 <= expert < self.num_experts:
                raise ValueError(
                    f"expert id {expert} is outside [0, {self.num_experts})"
                )
            if expert not in seen:
                seen.append(expert)

        step = int(self.step.item()) + 1
        self.step.fill_(step)
        protected = {
            slot
            for slot, usage in enumerate(self.usage.tolist())
            if usage == _JOINT_PINNED_USAGE
        }
        missing: list[int] = []
        for expert in seen:
            slot = int(self.slot_for_id[layer_id, expert].item())
            if slot >= 0:
                self.usage[slot] = step
                protected.add(slot)
            else:
                missing.append(expert)

        usage = [int(value) for value in self.usage.tolist()]
        owners = [int(value) for value in self.id_of_slot.tolist()]
        for index, expert in enumerate(missing):
            candidates = [
                slot
                for slot in range(self.decode_cache_size)
                if slot not in protected
            ]
            if not candidates:
                raise RuntimeError("joint working set exceeds canonical expert pool")
            victim = min(
                candidates,
                key=lambda slot: (owners[slot] >= 0, usage[slot], slot),
            )
            old_id = owners[victim]
            if old_id >= 0:
                self.slot_for_id.view(-1)[old_id] = -1
            flat_id = layer_id * self.num_experts + expert
            self.id_of_slot[victim] = flat_id
            self.slot_for_id[layer_id, expert] = victim
            self.usage[victim] = step
            self.evict_slots[index] = victim
            self.src_indices[index] = expert
            owners[victim] = flat_id
            usage[victim] = step
            protected.add(victim)

        self.num_indices.fill_(len(missing))
        for index in range(flat.numel()):
            raw_id = int(flat[index].item())
            flat[index] = self.slot_for_id[layer_id, raw_id]

    def _copy_joint_missing_cpu(self, layer_id: int) -> None:
        """Copy staged joint misses with ordinary Torch CPU indexing."""
        count = int(self.num_indices.item())
        if count == 0:
            return
        dst = self.evict_slots[:count].long()
        src = self.src_indices[:count].long()
        for per_layer, cache in self.banks:
            cache.index_copy_(0, dst, per_layer[layer_id].index_select(0, src))

    def _prepare_joint_slot_alphas(self, buffer_id: int, layer_id: int) -> None:
        """Scatter one layer's immutable logical scales into its physical slots."""
        if self._joint_gate_up_alpha_slots is None:
            return
        assert self._joint_down_alpha_slots is not None
        assert self.gate_up_alpha is not None and self.down_alpha is not None
        slots = self.slot_for_id[layer_id]
        if self.device.type == "cuda":
            torch._assert_async(
                (slots >= 0).all(),
                "joint alpha mapping found a missing expert slot",
            )
        elif not bool((slots >= 0).all()):
            raise RuntimeError("joint alpha mapping found a missing expert slot")
        slots = slots.clamp_min(0).long()
        lo = layer_id * self.num_experts
        hi = lo + self.num_experts
        gate_up_slots = self._joint_gate_up_alpha_slots[buffer_id]
        down_slots = self._joint_down_alpha_slots[buffer_id]
        gate_up_slots.zero_()
        down_slots.zero_()
        gate_up_slots[slots] = self.gate_up_alpha[lo:hi]
        down_slots[slots] = self.down_alpha[lo:hi]

    def prepare_prefill_group_layer(self, layer_id: int) -> None:
        """Start this layer's expert copy before the paired decode forward."""
        if not self._prefill_group_active:
            raise RuntimeError("no layer-group prefill wave is active")
        buffer_id = layer_id % self.prefill_buffer_count
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            self._prefill_group_target_layer = layer_id
            return
        if (
            self._prefill_group_target_layer is not None
            and self._prefill_hit_d2d_active
            and self.prefill_copy_stream is not None
        ):
            # Decode may have changed the slot map since the prior layer.  Snapshot
            # only at this safe scheduler boundary; the hit gather is enqueued on
            # the current stream before decode mutates the LRU again.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()
        self._prefill_group_target_layer = layer_id
        self.prefetch_prefill_layer(layer_id)

    def end_prefill_group(self) -> None:
        if not self.prefill_overlap:
            return
        if not self._prefill_group_active:
            raise RuntimeError("no layer-group prefill wave is active")
        if self._resident_group_range is not None:
            if self.prefill_group_size:
                # Enqueued after every group GEMM on the compute stream.  Refresh
                # the complete working set before publishing its release so it
                # remains temporally local but becomes ordinarily evictable.
                start, end = self._resident_group_range
                group_slots = self.slot_for_id[start:end].reshape(-1)
                if self.device.type == "cuda":
                    self.usage[group_slots] = self.step
                else:
                    if not bool((group_slots >= 0).all()):
                        raise RuntimeError(
                            "joint group release found a missing expert mapping"
                        )
                    self.usage[group_slots.long()] = self.step
                if self._joint_group_release_event is not None:
                    self._joint_group_release_event.record(
                        torch.cuda.current_stream(self.device)
                    )
                    self._joint_group_has_release_event = True
                # Pages remain mapped with the group's final admission epoch.
                self._prefill_group_active = False
                self._prefill_group_target_layer = None
                self._resident_group_range = None
                self._resident_group_ready_events = []
                return
            current_stream = torch.cuda.current_stream(self.device)
            start, end = self._resident_group_range
            for layer_id in range(start, end):
                buffer_id = layer_id - start
                if self.prefill_release_events:
                    self.prefill_release_events[buffer_id].record(current_stream)
                    self._prefill_buffer_has_release_event[buffer_id] = True
                self._prefill_buffer_released[buffer_id] = True
        self._prefill_group_active = False
        self._prefill_group_target_layer = None
        self._resident_group_range = None
        self._resident_group_ready_events = []

    def _begin_prefill_buffers(self) -> None:
        count = self.prefill_buffer_count
        self._prefill_buffer_layer = [None] * count
        self._prefill_buffer_released = [True] * count
        self._prefill_buffer_has_hit_ready_event = [False] * count
        if self.prefill_copy_stream is not None:
            # Fence this prefill's copy-stream work behind everything already enqueued
            # on the compute stream. The release/ready events only order against the
            # *previous prefill*; under overlap scheduling a new prefill can be enqueued
            # while the preceding decode batch is still running, and that decode may
            # have loaded experts into the slots the buffers borrow -- without this
            # fence the first prefetch would stomp bytes a running GEMM is reading.
            self.prefill_begin_event.record(torch.cuda.current_stream(self.device))
            self.prefill_copy_stream.wait_event(self.prefill_begin_event)
        self._prefill_hit_d2d_active = self.prefill_hit_d2d and self._hit_d2d_usable()
        if self._prefill_hit_d2d_active:
            # The copy stream is fenced behind the previous decode, so the snapshot
            # observes its final slot map; one host sync per chunk, then per-layer
            # classification is pure host math.
            with torch.cuda.stream(self.prefill_copy_stream):
                self._prefill_slot_snapshot.copy_(self.slot_for_id, non_blocking=True)
            self.prefill_copy_stream.synchronize()

    def prefetch_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap or layer_id >= self.num_layers:
            return
        if layer_id < 0:
            raise ValueError(f"Invalid prefill layer id: {layer_id}")
        if self._resident_group_range is not None:
            start, end = self._resident_group_range
            if not start <= layer_id < end:
                return
            buffer_id = layer_id - start
        else:
            if (
                self._prefill_group_active
                and layer_id != self._prefill_group_target_layer
            ):
                return
            buffer_id = layer_id % self.prefill_buffer_count

        assert self.banks and self.prefill_bank_buffers
        if self._prefill_buffer_layer[buffer_id] == layer_id:
            return
        if self._prefill_buffer_layer[buffer_id] is not None:
            assert self._prefill_buffer_released[buffer_id], (
                "Prefill overlap buffer is being reused before release"
            )

        def copy() -> None:
            self._invalidate_prefill_buffer(buffer_id)
            for (per_layer, _), buffer in zip(self.banks, self.prefill_bank_buffers):
                buffer[buffer_id].copy_(per_layer[layer_id], non_blocking=True)

        self._prefill_buffer_has_hit_ready_event[buffer_id] = False
        self.prefill_layer_prepares += 1
        if self._prefill_hit_d2d_active:
            self._prefetch_split(layer_id, buffer_id)
        elif self.prefill_copy_stream is None:
            self.prefill_total_rows += self.num_experts
            self.prefill_h2d_bytes += self._prefill_full_layer_bytes
            copy()
        else:
            self.prefill_total_rows += self.num_experts
            self.prefill_h2d_bytes += self._prefill_full_layer_bytes
            with torch.cuda.stream(self.prefill_copy_stream):
                if self._prefill_buffer_has_release_event[buffer_id]:
                    self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
                copy()
                self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

        self._prefill_buffer_layer[buffer_id] = layer_id
        self._prefill_buffer_released[buffer_id] = False

    def _hit_d2d_usable(self) -> bool:
        """Whether the hit-D2D split can serve this prefill; logs the first fallback.

        The flag is an auto-fallback optional: any unusable condition must degrade
        to the legacy full-layer copy AND say so once in the server log, so a
        configuration that silently runs the legacy path is visible.
        """
        from freetoken.kernel.fast_index_copy import _skip_fast_index_copy_enabled

        if self._prefill_slot_snapshot is None or self.prefill_copy_stream is None:
            reason = "prefill overlap buffers are not initialized for this device"
        elif _skip_fast_index_copy_enabled():
            reason = "FREETOKEN_SKIP_FAST_INDEX_COPY is set (the hit gather would be a no-op)"
        elif not self._copy_fused_ok:
            reason = "the fused copy plan is unavailable (bank alignment or FREETOKEN_FUSED_COPY=0)"
        elif not self.separate_prefill_buffer and self.cache_size <= 2 * self.num_experts:
            reason = (
                f"cache_size {self.cache_size} leaves no hit region "
                f"(needs > {2 * self.num_experts} slots)"
            )
        elif not self._resolve_batch_memcpy():
            reason = "cudaMemcpyBatchAsync is unavailable"  # resolve logged the specifics
        else:
            return True
        if not self._hit_d2d_fallback_logged:
            logger.warning(
                f"MoE prefill hit-D2D requested but unavailable ({reason}); "
                "falling back to full-layer copies"
            )
            self._hit_d2d_fallback_logged = True
        return False

    def _resolve_batch_memcpy(self) -> bool:
        if self._batch_memcpy is None:
            try:
                from freetoken.kernel.batch_memcpy import load_batch_memcpy

                self._batch_memcpy = load_batch_memcpy()
            except Exception as exc:  # noqa: BLE001 -- any build/runtime gap => legacy path
                logger.warning(f"MoE prefill hit-D2D disabled ({exc}); using full-layer copies")
                self._batch_memcpy = False
        return self._batch_memcpy is not False

    def _prefetch_split(self, layer_id: int, buffer_id: int) -> None:
        """Hit/miss-split prefetch of one expert layer into the double buffer.

        Resident experts are gathered cache -> buffer on the CURRENT stream, fully
        device-side: a one-launch compaction reads the LIVE slot_for_id row into
        fixed-shape gather indices (no host round trip), then fast_index_copy_multi
        moves the rows. Serializing the gather before this layer's GEMMs costs its
        plain duration instead of nondeterministic SM contention. Misses cross
        PCIe as ONE cudaMemcpyBatchAsync of coalesced expert-id runs on the copy
        stream, under the existing release/ready event discipline; its host-built
        run list comes from the begin-of-chunk snapshot because the batch API
        takes HOST pointer arrays. Live-vs-snapshot cannot disagree: the only
        chunk-internal writer (buffer invalidation) rewrites slots already below
        the 2E threshold, and slots < 2E (including -1) are misses on both sides
        -- the buffers own those slots, so their bytes are volatile within the
        chunk. Hit and miss row sets are disjoint, so the streams need no
        ordering against each other.
        """
        import numpy as np

        from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit
        from freetoken.moe.offload_kernels import prefill_hit_compact

        E = self.num_experts
        snap = self._prefill_snapshot_np[layer_id]
        hit_floor = 0 if self.separate_prefill_buffer else 2 * E
        hit_mask = snap >= hit_floor
        self.prefill_hit_rows += int(hit_mask.sum())
        self.prefill_total_rows += E
        if self._gather_dst_ptrs is not None:
            current_stream = torch.cuda.current_stream(self.device)
            if self._prefill_buffer_has_release_event[buffer_id]:
                current_stream.wait_event(self.prefill_release_events[buffer_id])
            prefill_hit_compact(self, layer_id, buffer_id)
            # blocks_per_bank=64 vs the PCIe-tuned default of 8: HBM D2D needs the
            # wider grid (~22 GB/s per 1024-thread block on H100).
            fast_index_copy_multi_jit(
                self._prefill_gather_dst_ptrs,
                self._gather_dst_ptrs,
                self._gather_feat_bytes,
                self._prefill_hit_dst,
                self._prefill_hit_src,
                self._prefill_hit_num,
                blocks_per_bank=64,
            )
            self.prefill_hit_ready_events[buffer_id].record(current_stream)
            self._prefill_buffer_has_hit_ready_event[buffer_id] = True
        miss = np.nonzero(~hit_mask)[0]
        with torch.cuda.stream(self.prefill_copy_stream):
            if self._prefill_buffer_has_release_event[buffer_id]:
                self.prefill_copy_stream.wait_event(self.prefill_release_events[buffer_id])
            self._invalidate_prefill_buffer(buffer_id)
            if miss.size:
                run_starts = np.concatenate(([0], np.nonzero(np.diff(miss) != 1)[0] + 1))
                starts = miss[run_starts]
                lengths = np.diff(np.concatenate((run_starts, [miss.size])))
            dst, src, nbytes = [], [], []
            for b, feat in enumerate(self._copy_feat_bytes_host):
                if feat < _SMALL_BANK_FEAT_BYTES:
                    # Whole layer as one entry, EVEN with zero misses: it keeps every
                    # batch entry above the driver's async floor and covers the hit
                    # rows the gather skips for these banks.
                    dst.append(self._prefill_dst_ptrs_host[b] + buffer_id * E * feat)
                    src.append(self._copy_src_ptrs_host[layer_id][b])
                    nbytes.append(E * feat)
                elif miss.size:
                    dst.extend(
                        self._prefill_dst_ptrs_host[b] + (buffer_id * E + starts) * feat
                    )
                    src.extend(self._copy_src_ptrs_host[layer_id][b] + starts * feat)
                    nbytes.extend(lengths * feat)
            if dst:
                self.prefill_h2d_bytes += sum(int(size) for size in nbytes)
                self._batch_memcpy(
                    torch.tensor(dst, dtype=torch.int64),
                    torch.tensor(src, dtype=torch.int64),
                    torch.tensor(nbytes, dtype=torch.int64),
                    torch.cuda.current_stream(self.device).cuda_stream,
                )
            self.prefill_ready_events[buffer_id].record(self.prefill_copy_stream)

    def wait_prefill_layer(self, layer_id: int) -> tuple[torch.Tensor, ...]:
        """Ready bank views for ``layer_id`` in registration order.

        Ordinary streaming returns a contiguous ``[num_experts, ...]`` layer;
        joint returns the full canonical slot bank after its layer-ready event.
        """
        assert self.prefill_overlap
        if self.prefill_group_size and self._resident_group_range is not None:
            self._wait_resident_group_layer(layer_id)
            return self.bank_views()
        assert self.prefill_bank_buffers
        self.prefetch_prefill_layer(layer_id)
        if self._resident_group_range is not None:
            start, end = self._resident_group_range
            if not start <= layer_id < end:
                raise RuntimeError(
                    f"layer {layer_id} is outside resident group [{start}, {end})"
                )
            buffer_id = layer_id - start
        else:
            buffer_id = layer_id % self.prefill_buffer_count
        assert self._prefill_buffer_layer[buffer_id] == layer_id
        if self.prefill_ready_events:
            torch.cuda.current_stream(self.device).wait_event(self.prefill_ready_events[buffer_id])
        if self._prefill_buffer_has_hit_ready_event[buffer_id]:
            torch.cuda.current_stream(self.device).wait_event(
                self.prefill_hit_ready_events[buffer_id]
            )
        return tuple(buffer[buffer_id] for buffer in self.prefill_bank_buffers)

    def _wait_resident_group_layer(self, layer_id: int) -> None:
        if not self._prefill_group_active or self._resident_group_range is None:
            raise RuntimeError("no joint resident expert group is active")
        start, end = self._resident_group_range
        if not start <= layer_id < end:
            raise RuntimeError(
                f"layer {layer_id} is outside resident group [{start}, {end})"
            )
        buffer_id = layer_id - start
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            raise RuntimeError(f"joint resident layer {layer_id} was not admitted")
        if self._resident_group_ready_events:
            torch.cuda.current_stream(self.device).wait_event(
                self._resident_group_ready_events[buffer_id]
            )

    def has_resident_prefill_layer(self, layer_id: int) -> bool:
        """Whether joint currently protects ``layer_id`` in the canonical pool."""
        if not self.prefill_group_size or self._resident_group_range is None:
            return False
        start, end = self._resident_group_range
        return self._prefill_group_active and start <= layer_id < end

    def map_prefill_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Map raw expert ids in place to canonical physical slot ids.

        This public operation is valid only for a fully admitted layer in the
        active joint resident group.  Missing mappings are implementation errors;
        they never trigger a gather or a fallback prefill allocation.
        """
        self._wait_resident_group_layer(layer_id)
        if expert_ids.device.type == "cuda":
            mapped = torch.index_select(
                self.slot_for_id[layer_id],
                0,
                expert_ids.reshape(-1),
            )
            expert_ids.copy_(mapped.view_as(expert_ids))
            return

        raw_ids = expert_ids.long()
        in_range = (raw_ids >= 0) & (raw_ids < self.num_experts)
        if not bool(in_range.all()):
            raise ValueError(
                f"joint prefill expert ids must be in [0, {self.num_experts})"
            )
        safe_ids = raw_ids.clamp(0, self.num_experts - 1)
        mapped = self.slot_for_id[layer_id][safe_ids]
        if not bool((mapped >= 0).all()):
            raise RuntimeError(
                f"joint resident layer {layer_id} is missing an expert mapping"
            )
        expert_ids.copy_(mapped)

    def release_prefill_layer(self, layer_id: int) -> None:
        if not self.prefill_overlap:
            return
        if self._resident_group_range is not None:
            # A resident group is released atomically after every selected chunk
            # has traversed all of its layers.
            return
        buffer_id = layer_id % self.prefill_buffer_count
        if self._prefill_buffer_layer[buffer_id] != layer_id:
            return
        if self.prefill_release_events:
            self.prefill_release_events[buffer_id].record(torch.cuda.current_stream(self.device))
            self._prefill_buffer_has_release_event[buffer_id] = True
        self._prefill_buffer_released[buffer_id] = True

    def ensure_experts(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        from freetoken.moe.offload_kernels import ensure_experts

        if self.collect_decode_freq:
            # ``expert_ids`` still holds raw expert ids here (the kernel rewrites them to
            # slot ids in place), so snapshot the routing histogram before that happens.
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        ensure_experts(self, layer_id, expert_ids)

    def ensure_decode_experts(
        self, layer_id: int, expert_ids: torch.Tensor
    ) -> None:
        """Admit ordinary decode routes using causal layer-distance eviction."""
        if (
            not expert_ids.is_cuda
            or expert_ids.numel() * self.num_layers <= self.decode_cache_size
        ):
            self.ensure_experts(layer_id, expert_ids)
            return

        from freetoken.moe.offload_kernels import ensure_decode_experts

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        ensure_decode_experts(self, layer_id, expert_ids)

    def ensure_experts_hybrid(self, layer_id: int, expert_ids: torch.Tensor) -> None:
        """Capped-fetch LRU for the hybrid backend.

        Like :meth:`ensure_experts` but assigns slots to (and schedules copies for) at
        most ``hybrid_max_fetch`` -- or ``~hybrid_fetch_fraction * misses`` when the
        fraction is set -- of this step's missing experts; the overflow misses are
        left non-resident and ``expert_ids`` is rewritten to their cache slot (hit or
        freshly fetched) or ``-1`` (overflow -> compute on the CPU). ``num_indices`` holds
        the capped fetch count (for ``copy_missing``); ``num_missing_full`` the pre-cap
        miss count (for stats). All device-side / fixed-shape, so it is CUDA-graph safe."""
        from freetoken.moe.offload_kernels import ensure_experts_hybrid

        if self.collect_decode_freq:
            ids = expert_ids.reshape(-1).long()
            self.decode_freq[layer_id].scatter_add_(0, ids, torch.ones_like(ids))
        self._pending_src_layer = layer_id
        ensure_experts_hybrid(
            self, layer_id, expert_ids, self.hybrid_max_fetch, self.hybrid_fetch_fraction
        )

    def materialize_layer(self, layer_id: int) -> None:
        from freetoken.moe.offload_kernels import materialize_layer

        self._pending_src_layer = layer_id
        materialize_layer(self, layer_id)

    def reset(self) -> None:
        from freetoken.moe.offload_kernels import reset_cache

        reset_cache(self)
        self._prefill_group_active = False
        self._prefill_group_target_layer = None
        self._resident_group_range = None
        self._resident_group_ready_events = []
        self._resident_prefetch_range = None
        self._resident_prefetch_ready_events = []
        if self._resident_prefetch_num_indices is not None:
            self._resident_prefetch_num_indices.zero_()
        # Per-expert recency is not cache_size-shaped, so reset_cache leaves it alone; wipe
        # it here so a new sequence starts with cold hybrid fetch priorities.
        self.expert_recency.fill_(-1)

    def reset_stats(self) -> None:
        self.prefill_hit_rows = 0
        self.prefill_total_rows = 0
        self.prefill_layer_prepares = 0
        self.prefill_h2d_bytes = 0
        self.joint_prefill_total_rows = 0
        self.joint_prefill_lru_stats.zero_()
        self.lru_stats.zero_()
        self.stat_missing.zero_()
        self.stat_active.zero_()
        self.stat_calls.zero_()
        self.stat_fetched.zero_()
        self.stat_missing_layer.zero_()
        self.stat_active_layer.zero_()
        self.stat_fetched_layer.zero_()
        self.stat_steps_layer.zero_()

    def record_decode_stats(self, layer_id: int) -> None:
        """No-op: ``ensure_experts`` accumulates into ``lru_stats`` inside its own launch.

        Kept so the hybrid and non-hybrid call sites stay symmetric. The previous version
        was eight torch ops per layer per step, all captured into the decode graph.
        """

    def record_decode_stats_hybrid(self, layer_id: int) -> None:
        """Hybrid stats: full miss count (pre-cap), the PCIe-fetched count (capped), and
        the active count. The CPU computes (missing - fetched) experts. Device-side;
        accumulates both the scalar totals and the per-layer breakdown."""
        assert 0 <= layer_id < self.num_layers, f"layer_id {layer_id} out of range [0, {self.num_layers})"
        missing = self.num_missing_full.sum()
        fetched = self.num_indices.sum()
        active = self.active_mask.sum()
        self.stat_missing += missing
        self.stat_fetched += fetched
        self.stat_active += active
        self.stat_calls += 1
        self.stat_missing_layer[layer_id] += missing
        self.stat_fetched_layer[layer_id] += fetched
        self.stat_active_layer[layer_id] += active
        self.stat_steps_layer[layer_id] += 1

    def decode_miss_stats(self) -> dict:
        snapshot = self.cumulative_stats_snapshot()
        active = snapshot["decode_active_rows"]
        missing = snapshot["decode_missing_rows"]
        calls = snapshot["decode_layer_calls"]
        fetched = snapshot["decode_fetched_rows"]
        return {
            "layer_calls": calls,
            "active_per_layer": (active / calls) if calls else 0.0,
            "missing_per_layer": (missing / calls) if calls else 0.0,
            "miss_rate": (missing / active) if active else 0.0,
            # hybrid: how the misses split between PCIe fetch (GPU) and CPU compute.
            "fetched_per_layer": (fetched / calls) if calls else 0.0,
            "cpu_per_layer": ((missing - fetched) / calls) if calls else 0.0,
            "fetch_rate": (fetched / missing) if missing else 0.0,
            # Keep this long-standing public summary shape; the idle snapshot below exposes
            # exact integer totals without changing these derived fields.
            "prefill_hit_rows": snapshot["prefill_hit_rows"],
            "prefill_rows": snapshot["prefill_rows"],
            "prefill_layer_prepares": snapshot["prefill_layer_prepares"],
            "prefill_h2d_bytes": snapshot["prefill_h2d_bytes_total"],
        }

    def cumulative_stats_snapshot(self) -> dict[str, int]:
        """Read exact cumulative MoE counters at an explicit idle/statistics boundary."""
        if self.decode_target == "hybrid":
            active = int(self.stat_active.item())
            missing = int(self.stat_missing.item())
            calls = int(self.stat_calls.item())
        else:
            active, missing, calls = (int(x) for x in self.lru_stats.sum(0))
        fetched = int(self.stat_fetched.item())
        joint_miss_rows = int(
            self.joint_prefill_lru_stats[:, Stat.MISS].sum().item()
        )
        joint_rows = self.joint_prefill_total_rows
        joint_hits = joint_rows - joint_miss_rows
        return {
            "decode_active_rows": active,
            "decode_missing_rows": missing,
            "decode_layer_calls": calls,
            "decode_fetched_rows": fetched,
            "prefill_hit_rows": self.prefill_hit_rows + joint_hits,
            "prefill_rows": self.prefill_total_rows + joint_rows,
            "prefill_layer_prepares": self.prefill_layer_prepares,
            "prefill_h2d_bytes_total": (
                self.prefill_h2d_bytes
                + joint_miss_rows * self._expert_row_bytes
            ),
            "expert_row_bytes": self._expert_row_bytes,
        }

    def prefill_h2d_bytes_total(self) -> int:
        """Exact H2D bytes after explicitly synchronizing joint's row counter.

        Legacy/mixed/layered update ``prefill_h2d_bytes`` on the host.  Joint
        admission stays asynchronous and accumulates miss rows on device, so
        callers should use this method only at an explicit statistics boundary,
        never in the scheduler hot path.
        """
        joint_miss_rows = int(
            self.joint_prefill_lru_stats[:, Stat.MISS].sum().item()
        )
        return self.prefill_h2d_bytes + joint_miss_rows * self._expert_row_bytes

    def decode_miss_stats_per_layer(self) -> dict:
        """Per-MoE-layer realized decode stats for one (reset_stats-delimited) window.

        Requires ``collect_stats`` and the call sites passing ``layer_id``. Returns python
        lists indexed by MoE-layer id: missing/active experts per step and the realized
        miss_rate (missing/active) -- i.e. how cacheable each layer's routing actually was
        under the running LRU. Reads device tensors once (no per-step host sync)."""
        if self.decode_target == "hybrid":
            steps = self.stat_steps_layer.tolist()
            missing = self.stat_missing_layer.tolist()
            active = self.stat_active_layer.tolist()
        else:
            cols = self.lru_stats.t().tolist()
            active, missing, steps = cols[Stat.ACTIVE], cols[Stat.MISS], cols[Stat.CALLS]
        fetched = self.stat_fetched_layer.tolist()
        per_layer = []
        for L in range(self.num_layers):
            s, m, a, f = steps[L], missing[L], active[L], fetched[L]
            per_layer.append({
                "layer": L,
                "steps": s,
                "active_per_step": (a / s) if s else 0.0,
                "missing_per_step": (m / s) if s else 0.0,
                "miss_rate": (m / a) if a else 0.0,
                "fetched_per_step": (f / s) if s else 0.0,
            })
        return {"per_layer": per_layer}

    def decode_routing_stats(self) -> dict:
        """Per-layer decode routing concentration, for cache-skew analysis.

        Uses the histogram from ``collect_decode_freq``. The ``oracle_hit`` is the best a
        per-layer LRU holding ``cache_size/num_layers`` slots could achieve on the observed
        (stationary) routing distribution -- i.e. an upper bound on hit rate that depends
        purely on how skewed routing is, independent of any LRU/LFU dynamics.
        """
        freq = self.decode_freq.float()
        total = freq.sum(dim=1)
        valid = total > 0
        if int(valid.sum()) == 0:
            return {}
        slots_per_layer = self.decode_cache_size / self.num_layers
        C = max(1, int(round(slots_per_layer)))
        sorted_f, _ = torch.sort(freq, dim=1, descending=True)
        oracle_hit = (sorted_f[:, :C].sum(dim=1)[valid] / total[valid]).mean().item()
        ws = (freq > 0).sum(dim=1).float()
        cdf = torch.cumsum(sorted_f, dim=1) / total.clamp(min=1).unsqueeze(1)
        cover90 = ((cdf < 0.9).sum(dim=1).float() + 1)[valid]
        p = freq / total.clamp(min=1).unsqueeze(1)
        ent = -(p * p.clamp(min=1e-12).log()).sum(dim=1)[valid]
        norm_ent = (ent / torch.log(torch.tensor(float(self.num_experts)))).mean().item()
        return {
            "slots_per_layer": slots_per_layer,
            "working_set_mean": ws[valid].mean().item(),
            "working_set_max": int(ws[valid].max().item()),
            "experts_for_90pct": cover90.mean().item(),
            "oracle_hit_at_slots": oracle_hit,
            "norm_entropy": norm_ent,
        }

    def copy_missing(self) -> None:
        assert self.banks, "set_bank_sources must register the banks first"
        layer_id = self._pending_src_layer
        assert layer_id is not None, "no staged misses (ensure_experts/materialize_layer first)"
        self._copy_missing_plan(
            layer_id,
            self.evict_slots,
            self.src_indices,
            self.num_indices,
        )

    def _copy_missing_plan(
        self,
        layer_id: int,
        evict_slots: torch.Tensor,
        src_indices: torch.Tensor,
        num_indices: torch.Tensor,
    ) -> None:
        """Copy one explicit plan without touching decode's pending-plan state."""
        assert self.banks, "set_bank_sources must register the banks first"
        if self._copy_fused_ok:
            from freetoken.kernel.fast_index_copy import fast_index_copy_multi_jit

            # One launch copies the missing rows for every bank (instead of one launch per
            # bank). evict_slots/src_indices/num_indices are shared across banks;
            # src_indices holds layer-local expert rows, resolved against this layer's
            # source pointers (layer_id is a static int per captured graph node).
            fast_index_copy_multi_jit(
                self._copy_dst_ptrs,
                self._copy_src_ptrs[layer_id],
                self._copy_feat_bytes,
                evict_slots,
                src_indices,
                num_indices,
            )
            return

        from freetoken.kernel import fast_index_copy_jit

        for per_layer, cache in self.banks:
            fast_index_copy_jit(
                cache,
                evict_slots,
                per_layer[layer_id],
                src_indices,
                num_indices,
            )


def iter_offload_moe_layers(model) -> Iterator:
    from freetoken.layers import BaseOP, OffloadMoELayer

    # A model whose MoE blocks are bespoke nn.Modules (not OffloadMoELayer) declares its
    # offload layers explicitly via this hook (e.g. DeepSeek-V4-Flash); attach_offload_moe_cache
    # then sets .offload_cache on each yielded layer just like the OffloadMoELayer walk.
    hook = getattr(model, "_iter_offload_moe_layers", None)
    if hook is not None:
        yield from hook()
        return

    if isinstance(model, OffloadMoELayer):
        yield model

    if not isinstance(model, BaseOP):
        return

    for value in model.__dict__.values():
        if isinstance(value, BaseOP):
            yield from iter_offload_moe_layers(value)
        elif isinstance(value, (list, tuple)):
            for item in value:
                yield from iter_offload_moe_layers(item)


def attach_offload_moe_cache(model, cache: OffloadMoeCache) -> list:
    layers = list(iter_offload_moe_layers(model))
    for layer in layers:
        layer.offload_cache = cache
    return layers
