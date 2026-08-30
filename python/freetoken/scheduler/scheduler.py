from __future__ import annotations

import time
from typing import TYPE_CHECKING, List, NoReturn, Set, Tuple

import torch
from freetoken.core import Batch, Req
from freetoken.env import ENV
from freetoken.gpu_select import gpu_identity
from freetoken.message import (
    AbortBackendMsg,
    BaseBackendMsg,
    BatchBackendMsg,
    CacheRebuildBackendMsg,
    CacheRebuildResultMsg,
    DetokenizeMsg,
    ErrorReplyMsg,
    ExitMsg,
    PromptAdmittedMsg,
    UserMsg,
)
from freetoken.utils import (
    init_logger,
    load_eos_token_ids,
    load_tokenizer,
    load_toolcall_anchor_id,
)

from .cache import CacheManager
from .config import SchedulerConfig
from .decode import DecodeManager
from .forward import ForwardData, ForwardInput, Indice2D
from .io import SchedulerIOMixin
from .layered_batch import (
    LayeredBatchComposer,
    LayeredExecutionStats,
    LayeredPrefillChunk,
    LayeredPrefillWave,
)
from .joint_execution import JointWaveExecutor
from .layered_pipeline import LayeredPipelineExecutor
from .mixed_batch import LegacyBatchComposer, MixedBatchComposer
from .prefill import ChunkedReq, PrefillManager
from .resident_decode import StableDecodeInput, prepare_stable_decode
from .resident_wave import ResidentExecutor
from .status import SchedulerStatusReporter
from .table import TableManager

if TYPE_CHECKING:
    from freetoken.engine import ForwardOutput


logger = init_logger(__name__)


def _gib(n_bytes: int) -> str:
    return f"{n_bytes / (1 << 30):.2f} GiB"


class Scheduler(SchedulerIOMixin):
    def __init__(self, config: SchedulerConfig):
        from freetoken.engine import Engine

        self.engine = Engine(config)

        # use another stream to overlap metadata processing with computation
        self.device = self.engine.device
        self.stream = torch.cuda.Stream(device=self.device)
        self.engine_stream_ctx = torch.cuda.stream(self.engine.stream)
        self.layered_prefill_stream = (
            torch.cuda.Stream(device=self.device)
            if config.batching_policy == "layered"
            and config.prefill_execution == "concurrent"
            else None
        )
        torch.cuda.set_stream(self.stream)
        # sent on the readiness ack for /v1/stats gpus; a list so TP can add one entry per rank
        self.gpus = [gpu_identity(self.device.index)] if self.device.type == "cuda" else []

        # initialize other managers
        self.table_manager = TableManager(config.max_running_req, self.engine.page_table)
        # ONE cache manager for every model (ShadowRadix layering): the shared page table is the
        # virtual full-token coordinate; model-specific tiers ride the plug-ins -- DSV4's
        # window/cmp/idx shadows via swa_pool, Gemma's swa via swa_pool, GDN state via
        # linear_state_pool. No model supplies its own manager.
        self.cache_manager = CacheManager(
            self.engine.num_pages, config.page_size, self.engine.page_table, config.cache_type,
            linear_state_pool=self.engine.linear_state_pool,
            swa_pool=self.engine.kv_cache,
            sliding_window_size=next(
                (g.sliding_window for g in config.model_config.kv_cache_group_specs() if g.is_swa),
                None,
            ) or getattr(self.engine.kv_cache, "sliding_window_size", None),
        )
        self.decode_manager = DecodeManager(config.page_size)
        self.prefill_manager = PrefillManager(
            self.cache_manager, self.table_manager, self.decode_manager
        )
        self.layered_composer: LayeredBatchComposer | None = None
        self.layered_wave: LayeredPrefillWave | None = None
        self.layered_stats = LayeredExecutionStats()
        self.resident_executor: ResidentExecutor | None = None
        self.layered_pipeline_executor: LayeredPipelineExecutor | None = None
        self._resident_decode_input: StableDecodeInput | None = None
        if config.batching_policy == "legacy":
            composer_cls = LegacyBatchComposer
        elif config.batching_policy == "mixed":
            composer_cls = MixedBatchComposer
        elif config.batching_policy == "layered":
            composer_cls = None
            self.layered_composer = LayeredBatchComposer(
                prefill_manager=self.prefill_manager,
                decode_manager=self.decode_manager,
                max_prefill_reqs=1,
            )
        elif config.batching_policy == "joint":
            composer_cls = None
            self.resident_executor = JointWaveExecutor(
                engine=self.engine,
                prefill_manager=self.prefill_manager,
                decode_manager=self.decode_manager,
                table_manager=self.table_manager,
                max_chunks=config.prefill_wave_max_chunks,
                prepare_batch=self._prepare_batch,
                report_prompt_admissions=self._report_prompt_admissions,
                restore_linear_states=self._restore_linear_states,
                free_req_resources=self._free_req_resources,
            )
        elif config.batching_policy == "layered-pipeline":
            composer_cls = None
            self.layered_pipeline_executor = LayeredPipelineExecutor(
                engine=self.engine,
                prefill_manager=self.prefill_manager,
                decode_manager=self.decode_manager,
                table_manager=self.table_manager,
                max_wave_chunks=config.prefill_wave_max_chunks,
                prepare_batch=self._prepare_resident_batch,
                prepare_allocation_only_batch=(
                    self._prepare_resident_allocation_only_batch
                ),
                prepare_decode_batch=self._prepare_resident_group_decode_batch,
                open_prefill_execution=self.cache_manager.open_prefill_execution,
                report_prompt_admissions=self._report_prompt_admissions,
                free_req_resources=self._free_req_resources,
            )
            self.resident_executor = self.layered_pipeline_executor
        else:
            raise ValueError(f"Unknown batching policy: {config.batching_policy!r}")
        self.batch_composer = (
            composer_cls(
                prefill_manager=self.prefill_manager,
                decode_manager=self.decode_manager,
            )
            if composer_cls is not None
            else None
        )

        # some alias for easy access
        self.finished_reqs: Set[Req] = set()
        # Abort acknowledgements are a terminal accounting barrier. Queue them while processing
        # inbound control messages, then flush only AFTER _process_last_data publishes any
        # sampled replies from the prior overlapped forward.
        self._pending_abort_acks: Set[int] = set()
        # With multiple tokenizer workers, an AbortBackendMsg and its earlier UserMsg can arrive
        # through different PUSH producers and be observed out of order. Preserve a bounded
        # tombstone so an abort-before-admission request can never be resurrected after its
        # terminal accounting acknowledgement has already been published.
        self._abort_tombstones: dict[int, None] = {}
        self._forward_iter = 0  # global forward counter; drives the SWA proactive-eviction cadence
        # The launched-but-not-yet-drained batch (overlap): set at the top of each overlap_loop
        # iteration so the abort handler can tell whether a request's forward is still in flight
        # (mark it, defer the free to _process_last_data) or not (free immediately). Stays None
        # in normal_loop, where a batch launches and drains within one iteration.
        self._last_data: ForwardData | None = None
        # Resident policies keep decode-only output one stage behind the next graph launch.
        # Prefill output is drained immediately for client-visible TTFT.
        self._resident_last_outputs: list[ForwardData] = []
        # A received-but-not-yet-executed runtime cache rebuild (CacheRebuildBackendMsg),
        # run at the next idle safe point in overlap_loop. None when no rebuild is pending.
        self._pending_rebuild: CacheRebuildBackendMsg | None = None
        self.tokenizer = load_tokenizer(config.model_path)
        self.eos_token_ids = load_eos_token_ids(config.model_path, self.tokenizer)
        self.toolcall_anchor_id = None
        if config.special_token_ckpt and (
            self.cache_manager.is_hybrid or self.cache_manager.is_swa
        ):
            from freetoken.server.function_call_parser import toolcall_opener_for

            self.toolcall_anchor_id = load_toolcall_anchor_id(
                self.tokenizer,
                toolcall_opener_for(getattr(config, "tool_call_parser", "")),
            )
        self.token_pool = self.table_manager.token_pool
        self.config = config
        self._refresh_prefill_budget("startup")
        self.status_reporter = SchedulerStatusReporter(
            log=logger.info_rank0,
            decode_log_interval=config.decode_log_interval,
        )

        # Initialize the I/O mixin
        super().__init__(config, self.engine.tp_cpu_group)

    def run_when_idle(self) -> None:
        """Called when the scheduler is idle to perform background tasks."""
        logger.info_rank0("Scheduler is idle, waiting for new reqs...")
        self.cache_manager.check_integrity()
        moe_cache = self.engine.moe_offload_cache
        if moe_cache is not None and moe_cache.collect_stats:
            stats = moe_cache.cumulative_stats_snapshot()
            logger.info_rank0(
                "MoE cache stats snapshot: "
                + ", ".join(f"{name}={value}" for name, value in stats.items())
            )

    @torch.inference_mode()
    def rebuild_cache(
        self,
        *,
        moe_cache_size: int | None = None,
        num_pages: int | None = None,
        num_mamba_slots: int | None = None,
        num_swa_pages: int | None = None,
        _prefill_budget_event: str = "cache_rebuild",
    ) -> None:
        """Idle-only runtime cache rebuild: resize the MoE slot cache, KV pages, GDN (mamba) state
        pool, and/or the window pool (num_swa_pages), re-capture CUDA graphs, and re-thread the
        page managers (clearing the prefix cache on a KV/mamba/window resize). The caller MUST
        guarantee the scheduler is idle — no pending prefill, no running decode, no in-flight
        finished requests. All TP ranks must call this with identical arguments.
        """
        assert self.layered_wave is None, "rebuild requires no active layered prefill"
        resident_executor = getattr(self, "resident_executor", None)
        assert not (
            resident_executor is not None and resident_executor.active
        ), "rebuild requires no active resident prefill"
        assert not self.prefill_manager.runnable, "rebuild requires no pending prefill"
        assert not self.decode_manager.runnable, "rebuild requires no running decode"
        torch.cuda.synchronize(self.device)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()
        # The cached decode metadata can retain graph-capture tensor views. Drop it before
        # the idle rebuild destroys and recreates those buffers.
        self._resident_decode_input = None
        self.engine.rebuild_runtime_cache(
            moe_cache_size=moe_cache_size, num_pages=num_pages, num_mamba_slots=num_mamba_slots,
            num_swa_pages=num_swa_pages,
        )
        if num_pages is not None or num_mamba_slots is not None or num_swa_pages is not None:
            # Any of these resizes invalidates the prefix cache: a KV resize leaves stale page
            # indices, a mamba resize leaves stale GDN-snapshot slot ids, and a window-pool resize
            # (num_swa_pages) reallocates the SWA/window token pool, leaving stale slot ids in the
            # radix tree. Rebuild the prefix cache + reclaim the resized free-lists.
            self.cache_manager.rebuild(self.engine.num_pages, self.engine.page_table)
            if num_pages is not None:
                # token_pool is sized to the page table; only a KV-page resize reallocates it.
                # A mamba-only rebuild leaves the page table untouched, so skip this (else it
                # needlessly reallocates + zeros the whole GPU token_pool every mamba resize).
                self.table_manager.rebuild(self.engine.page_table)
                self.token_pool = self.table_manager.token_pool
            self.cache_manager.check_integrity()
        self._refresh_prefill_budget(_prefill_budget_event)
        if self.config.tp_info.size > 1:
            self.sync_all_ranks()

    def _refresh_prefill_budget(self, event: str) -> None:
        """Apply the current cache's physical prefill limit and publish it for pipeline runs."""
        requested = self.config.max_extend_tokens
        cache_limit = self.cache_manager.prefill_chunk_budget
        self.prefill_budget = min(requested, cache_limit) if cache_limit else requested
        if self.layered_pipeline_executor is not None:
            logger.info_rank0(
                "Layered pipeline iteration limit: "
                f"requested_tokens={requested}, effective_tokens={self.prefill_budget}, "
                f"event={event}"
            )

    def overlap_loop(self, last_data: ForwardData | None) -> ForwardData | None:
        """
        The main loop of overlapping scheduling and execution.

        It will overlap the execution of current batch and processing of last batch's results,
        which can effectively hide CPU latency and improve GPU utilization.
        """
        # Expose the un-drained batch to _process_one_msg (abort in-flight check). Assigning
        # before the message loop is what makes the check airtight: the batch launched later
        # this iteration can only be probed by messages of the NEXT iteration, which sees it here.
        self._last_data = last_data
        blocking = not (
            last_data is not None  # don't block if we have a batch to be processed
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to drain toward + execute
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Execute a queued cache rebuild once the scheduler is fully idle (the safe point):
        # no last batch to process, no pending prefill, no running decode. finished_reqs is
        # NOT a gate — those requests are already freed (no live GPU/page resources).
        if self._pending_rebuild is not None and last_data is None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        # Order this iteration's host->device token_pool copies (issued on ``self.stream``
        # during scheduling) after the previous batch's sampled-token writes (issued on the
        # engine stream in ``_forward``). Without this, a request that reuses a just-freed
        # table_idx can have its freshly copied prompt clobbered by the prior occupant's
        # still-pending output write -- corrupting tokens (e.g. dropping an image
        # placeholder, which the multimodal merge then rejects).
        self.stream.wait_stream(self.engine.stream)
        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            with self.engine_stream_ctx:  # run the batch in the engine's stream
                self.engine.stream.wait_stream(self.stream)
                # COW-restore GDN snapshots for prefix hits ON THE ENGINE STREAM, after the
                # cross-stream wait and before the forward reads the live slot (program order
                # vs the prior batch's snapshot writes). Doing this on self.stream would race.
                self._restore_linear_states(forward_input.batch)
                ongoing_data = (forward_input, self._forward(forward_input))

        # The drain issues GPU-visible writes to state the batch just launched still reads: the
        # page-table re-point and, for the paged-SWA pools, the full->swa (DSV4: full->window)
        # sentinel scatter. DSV4 stages the page table at replay time and translates
        # full_to_window INSIDE the captured graph, so an unordered drain can redirect an
        # in-flight forward. copy_done only covers batch N; order against N+1 explicitly.
        self.stream.wait_stream(self.engine.stream)
        self._process_last_data(last_data)
        self._flush_abort_acks()
        return ongoing_data

    def normal_loop(self) -> None:
        blocking = not (
            self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None  # a queued rebuild to execute at idle
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        # Non-overlap mode has no last_data to drain; execute a queued rebuild as soon as
        # the scheduler is idle (no pending prefill / running decode). Without this, a
        # rebuild in DISABLE_OVERLAP_SCHEDULING mode stays pending until the HTTP timeout.
        if self._pending_rebuild is not None and not (
            self.prefill_manager.runnable or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        forward_input = self._schedule_next_batch()
        ongoing_data = None
        if forward_input is not None:
            # already inside engine_stream_ctx (run_forever); restore on the engine stream
            self._restore_linear_states(forward_input.batch)
            ongoing_data = (forward_input, self._forward(forward_input))

        self._process_last_data(ongoing_data)
        self._flush_abort_acks()

    def layered_loop(self) -> None:
        """Run one decode, then finish the active prefill layer group."""
        blocking = not (
            self.layered_wave is not None
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        if self._pending_rebuild is not None and not (
            self.layered_wave is not None
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        self.stream.wait_stream(self.engine.stream)
        decode_input, prefill_chunk = self._schedule_layered_work()
        if decode_input is None and prefill_chunk is None:
            self._flush_abort_acks()
            return

        decode_started = decode_ended = None
        decode_data: ForwardData | None = None
        both = decode_input is not None and prefill_chunk is not None
        wall_started = time.perf_counter()

        # Metadata/token copies are issued on the scheduler stream.  Both compute
        # streams wait for them, but never wait for each other in the explicit
        # concurrent A/B mode.
        self.engine.stream.wait_stream(self.stream)
        if self.layered_prefill_stream is not None:
            self.layered_prefill_stream.wait_stream(self.stream)

        # Start the current expert-layer copy before decode.  The copy stream is
        # independent; prefill's MoE call waits on its ready event later.
        with self.engine_stream_ctx:
            if prefill_chunk is not None:
                moe_cache = self.engine.moe_offload_cache
                assert moe_cache is not None
                if not moe_cache._prefill_group_active:
                    moe_cache.begin_prefill_group()
                target_layer = (
                    0 if self.layered_wave is None else self.layered_wave.current_layer
                )
                moe_cache.prepare_prefill_group_layer(target_layer)

            if decode_input is not None:
                decode_started = torch.cuda.Event(enable_timing=True)
                decode_ended = torch.cuda.Event(enable_timing=True)
                decode_started.record(self.engine.stream)
                decode_data = (decode_input, self._forward(decode_input))
                decode_ended.record(self.engine.stream)

        prefill_data: list[ForwardData] = []
        prefill_started = prefill_ended = None
        if prefill_chunk is not None:
            wave = self.layered_wave
            assert wave is not None
            group_end = wave.current_group_end
            compute_stream = self.layered_prefill_stream or self.engine.stream
            prefill_started = torch.cuda.Event(enable_timing=True)
            prefill_ended = torch.cuda.Event(enable_timing=True)
            prefill_started.record(compute_stream)

            if self.layered_prefill_stream is not None and decode_ended is not None:
                # Launch one prefill token-layer before waiting for decode so the
                # explicit concurrent A/B path actually overlaps useful compute.
                produced, next_chunk, prefetched_from = self._run_layered_group_sweep(
                    prefill_chunk,
                    group_end,
                    compute_stream,
                    max_steps=1,
                    prefetch_next=False,
                )
                prefill_data.extend(produced)
                self._drain_layered_decode(decode_data, decode_started, decode_ended)
                decode_data = None
                if next_chunk is not None:
                    wave = self.layered_wave
                    assert wave is not None
                    self._prepare_layered_expert_layer(wave.current_layer)
                    produced, _, _ = self._run_layered_group_sweep(
                        next_chunk,
                        group_end,
                        compute_stream,
                        prefetched_from=prefetched_from,
                    )
                    prefill_data.extend(produced)
            else:
                # Serial compute keeps decode priority: publish its sampled token
                # before the potentially long group sweep.  The current layer's
                # H2D copy was already started above and overlapped the decode.
                self._drain_layered_decode(decode_data, decode_started, decode_ended)
                decode_data = None
                produced, _, _ = self._run_layered_group_sweep(
                    prefill_chunk,
                    group_end,
                    compute_stream,
                )
                prefill_data.extend(produced)
            prefill_ended.record(compute_stream)
        else:
            self._drain_layered_decode(decode_data, decode_started, decode_ended)
            decode_data = None

        if prefill_ended is not None:
            prefill_ended.synchronize()
            self.layered_stats.prefill_gpu_ms += prefill_started.elapsed_time(prefill_ended)

        if both:
            self.layered_stats.joint_rounds += 1
            self.layered_stats.joint_wall_ms += (time.perf_counter() - wall_started) * 1000.0

        for data in prefill_data:
            self._process_last_data(data)
        self._flush_abort_acks()

    def _drain_layered_decode(
        self,
        decode_data: ForwardData | None,
        started: torch.cuda.Event | None,
        ended: torch.cuda.Event | None,
    ) -> None:
        """Publish this round's decode result without waiting for prefill compute."""
        if ended is None:
            return
        ended.synchronize()
        assert started is not None
        self.layered_stats.decode_forwards += 1
        self.layered_stats.decode_gpu_ms += started.elapsed_time(ended)
        self._process_last_data(decode_data)

    def _run_layered_group_sweep(
        self,
        first_chunk: LayeredPrefillChunk,
        group_end: int,
        compute_stream: torch.cuda.Stream,
        *,
        max_steps: int | None = None,
        prefetched_from: int | None = None,
        prefetch_next: bool = True,
    ) -> tuple[list[ForwardData], LayeredPrefillChunk | None, int | None]:
        """Run layer-major token chunks through the current group.

        Serial mode stages the next expert layer before computing the current
        layer.  Concurrent mode stages it immediately after the first current-
        layer chunk is enqueued, so its copy can overlap the remaining prefill
        work without delaying the decode launch.
        """
        outputs: list[ForwardData] = []
        chunk: LayeredPrefillChunk | None = first_chunk
        steps = 0
        concurrent = compute_stream is self.layered_prefill_stream

        while chunk is not None:
            wave = self.layered_wave
            assert wave is not None
            layer = wave.current_layer
            can_prefetch_next = (
                layer + 1 < group_end
                and (
                    not wave.admitting
                    or not self.prefill_manager.has_pending_uid(wave.uid)
                )
            )
            if (
                not concurrent
                and prefetch_next
                and can_prefetch_next
                and prefetched_from != layer
            ):
                self._prepare_layered_expert_layer(layer + 1)
                prefetched_from = layer

            with torch.cuda.stream(compute_stream):
                data = self._run_layered_prefill_chunk(chunk)
            if data is not None:
                outputs.append(data)

            stop = self._complete_layered_prefill_step()
            if (
                concurrent
                and prefetch_next
                and can_prefetch_next
                and prefetched_from != layer
            ):
                self._prepare_layered_expert_layer(layer + 1)
                prefetched_from = layer

            steps += 1
            if stop:
                chunk = None
            else:
                wave = self.layered_wave
                assert wave is not None
                chunk = wave.current_chunk()

            if max_steps is not None and steps >= max_steps:
                return outputs, chunk, prefetched_from

        return outputs, None, prefetched_from

    def _prepare_layered_expert_layer(self, layer_id: int) -> None:
        moe_cache = self.engine.moe_offload_cache
        assert moe_cache is not None
        with self.engine_stream_ctx:
            moe_cache.prepare_prefill_group_layer(layer_id)

    def _schedule_layered_work(
        self,
    ) -> tuple[ForwardInput | None, LayeredPrefillChunk | None]:
        composer = self.layered_composer
        assert composer is not None

        decode_batch = None
        prefill_batch = None
        if self.layered_wave is None:
            plan = composer.schedule_next_plan(self.prefill_budget)
            if plan is not None:
                decode_batch = plan.decode_batch
                prefill_batch = plan.prefill_batch
        else:
            decode_batch = self.decode_manager.schedule_next_batch()
            wave = self.layered_wave
            if wave.admitting:
                prefill_batch = self.prefill_manager.schedule_next_batch(
                    self.prefill_budget,
                    allowed_uids={wave.uid},
                    max_reqs=1,
                )
                if (
                    prefill_batch is None
                    and not self.prefill_manager.has_pending_uid(wave.uid)
                ):
                    wave.finish_admission()

        decode_input = self._prepare_batch(decode_batch) if decode_batch is not None else None
        if decode_input is not None:
            self._report_prompt_admissions(decode_input.batch)

        if prefill_batch is not None:
            prefill_chunk = self._prepare_layered_prefill_chunk(prefill_batch)
        elif (
            self.layered_wave is not None
            and not self.layered_wave.admitting
            and not self.layered_wave.done
        ):
            prefill_chunk = self.layered_wave.current_chunk()
        else:
            prefill_chunk = None

        if self.layered_wave is not None and self.layered_wave.done:
            self._finish_layered_wave()
        return decode_input, prefill_chunk

    def _prepare_layered_prefill_chunk(self, batch: Batch) -> LayeredPrefillChunk:
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        batch.input_ids = self.token_pool[forward_input.input_tuple]

        if self.layered_wave is None:
            assert len(batch.reqs) == 1
            self.layered_wave = LayeredPrefillWave(
                uid=batch.reqs[0].uid,
                num_layers=self.engine.layer_group_num_layers,
                group_size=self.config.prefill_layer_group_size,
            )
        wave = self.layered_wave
        assert wave is not None
        if any(req.uid != wave.uid for req in batch.reqs):
            raise RuntimeError("a layered prefill wave cannot admit a different request")

        # Reserve only the next original-token boundary.  Request completion stays
        # untouched until this chunk reaches the final decoder layer.
        for req in batch.reqs:
            self.prefill_manager.reserve_layered_continuation(req)
        chunk = LayeredPrefillChunk(
            forward_input=forward_input,
            allocated_device_len=max(req.device_len for req in batch.reqs),
        )
        wave.add_chunk(chunk)
        return chunk

    def _run_layered_prefill_chunk(
        self, chunk: LayeredPrefillChunk
    ) -> ForwardData | None:
        wave = self.layered_wave
        assert wave is not None
        forward_input = chunk.forward_input
        batch = forward_input.batch
        target_layer = wave.current_layer
        if chunk.state is None:
            if target_layer != 0:
                raise RuntimeError("a prefill chunk reached a later layer without embedding")
            chunk.state = self.engine.begin_layer_group_prefill(batch)
        if chunk.state.next_layer != target_layer:
            raise RuntimeError(
                f"chunk is at layer {chunk.state.next_layer}, wave is at {target_layer}"
            )
        chunk.state = self.engine.advance_layer_group_prefill(
            batch, chunk.state, target_layer + 1
        )
        if chunk.state.next_layer != wave.num_layers:
            return None

        output = self.engine.finish_layer_group_prefill(
            batch, chunk.state, forward_input.sample_args
        )
        self.token_pool[forward_input.write_tuple] = output.next_tokens_gpu
        self.decode_manager.filter_reqs(batch.reqs)
        return forward_input, output

    def _complete_layered_prefill_step(self) -> bool:
        """Advance the token-layer cursor; return whether this round must stop."""
        wave = self.layered_wave
        assert wave is not None
        if wave.admitting:
            if not self.prefill_manager.has_pending_uid(wave.uid):
                wave.finish_admission()
                if 1 % wave.group_size == 0 or 1 == wave.num_layers:
                    self.layered_stats.prefill_group_steps += 1
                if wave.done:
                    self._finish_layered_wave()
                    return True
                return 1 % wave.group_size == 0
            # The admission cursor has not reached the prompt's last original
            # chunk.  Keep layer 0 resident and admit the next chunk next round.
            return True

        group_done = wave.complete_replay_chunk()
        if group_done:
            self.layered_stats.prefill_group_steps += 1
        if wave.done:
            self._finish_layered_wave()
            return True
        return group_done

    def _finish_layered_wave(self) -> None:
        wave = self.layered_wave
        if wave is None:
            return
        moe_cache = self.engine.moe_offload_cache
        assert moe_cache is not None
        if moe_cache._prefill_group_active:
            moe_cache.end_prefill_group()
        self.layered_wave = None

    def _abort_layered_wave(self, uid: int) -> Req | None:
        wave = self.layered_wave
        if wave is None or wave.uid != uid:
            return None
        reqs = [req for chunk in wave.chunks for req in chunk.forward_input.batch.reqs]
        owner = reqs[-1] if reqs else None
        assert owner is not None
        table_idx = owner.table_idx
        self.cache_manager.discard_incomplete_layered_wave(
            owner.cache_handle,
            table_idx,
            max(chunk.allocated_device_len for chunk in wave.chunks),
        )
        self.table_manager.free(table_idx)
        for req in reqs:
            req.table_idx = -1
        self._finish_layered_wave()
        return owner

    def resident_loop(self) -> None:
        """Advance either resident-wave policy and drain outputs one stage later."""
        executor = self.resident_executor
        assert executor is not None
        last_outputs = self._resident_last_outputs
        blocking = not (
            last_outputs
            or
            executor.active
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
            or self._pending_rebuild is not None
        )
        for msg in self.receive_msg(blocking=blocking):
            self._process_one_msg(msg)

        if self._pending_rebuild is not None and not last_outputs and not (
            executor.active
            or self.prefill_manager.runnable
            or self.decode_manager.runnable
        ):
            self._execute_pending_rebuild()

        self.stream.wait_stream(self.engine.stream)
        outputs: list[ForwardData] = []
        if not executor.active:
            batch = executor.schedule_first_batch(self.prefill_budget)
            if batch is not None and not batch.has_prefill:
                forward_input = self._prepare_resident_decode_batch(batch)
                self._report_prompt_admissions(batch)
                with self.engine_stream_ctx:
                    self.engine.stream.wait_stream(self.stream)
                    self._restore_linear_states(batch)
                    data = (forward_input, self._forward(forward_input))
                if self._resident_decode_input is not None:
                    # The engine has enqueued this graph and advanced request lengths. Reserve
                    # the next query pages now, before the scheduler stream is made to wait for
                    # the graph below, so disjoint future page-table writes overlap its compute.
                    self.cache_manager.reserve_next_decode(batch.reqs)
                outputs = [data]
            elif batch is not None:
                self._resident_decode_input = None
                executor.begin_wave(batch, self.prefill_budget)
        elif executor.active:
            executor.prepare_step(self.prefill_budget)

        if executor.active:
            self.engine.stream.wait_stream(self.stream)
            with self.engine_stream_ctx:
                outputs = executor.advance_step()
            if self.config.batching_policy == "layered-pipeline":
                for data in outputs:
                    output_batch = data[0].batch
                    if output_batch.is_decode_only:
                        # finish_decode has enqueued the sampled-token copy and advanced every
                        # request's lengths. Reserve a page-boundary-crossing next query now,
                        # on the scheduler stream, while the current group forward is still in
                        # flight. The following prepare_step consumes the reservation through
                        # the normal allocate_paged path, exactly like resident pure decode.
                        self.cache_manager.reserve_next_decode(
                            output_batch.decode_reqs
                        )

        # Any page-table/cache writes performed while draining the prior iteration must follow
        # the just-enqueued forward, which can still read those entries. This is the same
        # stream-ordering barrier used by overlap_loop; copy_done then normally completes while
        # the scheduler prepares and enqueues the next iteration instead of stalling the host.
        self.stream.wait_stream(self.engine.stream)
        ready_outputs = list(last_outputs)
        deferred_outputs: list[ForwardData] = []
        for data in outputs:
            if data[0].batch.has_prefill:
                # The prefill result is this request's first user-visible token. Publishing it
                # now preserves TTFT; only steady-state decode benefits from a one-stage drain.
                ready_outputs.append(data)
            else:
                deferred_outputs.append(data)
        self._process_last_outputs(ready_outputs)
        self._resident_last_outputs = deferred_outputs
        self._flush_abort_acks()

    @torch.inference_mode()
    def run_forever(self) -> NoReturn:
        # DSV4 (owned-KV) decode reads its per-token window/cmp/idx slot maps off the attention
        # backend's per-batch SNAPSHOT (staged in prepare_for_replay right before the replay, on
        # the same stream, like the generic out_loc copy_from), not the live slot maps -- so the
        # next batch's allocate_paged cannot corrupt the in-flight graph replay. DSV4 overlaps.
        if self.config.batching_policy == "layered":
            assert torch.cuda.current_stream() == self.stream
            while True:
                self.layered_loop()
        elif self.config.batching_policy in (
            "joint",
            "layered-pipeline",
        ):
            assert torch.cuda.current_stream() == self.stream
            while True:
                self.resident_loop()
        elif ENV.DISABLE_OVERLAP_SCHEDULING:
            with self.engine_stream_ctx:
                self.engine.stream.wait_stream(self.stream)
                while True:
                    self.normal_loop()
        else:
            assert torch.cuda.current_stream() == self.stream
            data = None
            while True:
                data = self.overlap_loop(data)

    def shutdown(self) -> None:
        torch.cuda.synchronize(self.device)
        self.sync_all_ranks()
        self.engine.shutdown()

    def _process_last_outputs(self, outputs: list[ForwardData]) -> None:
        """Drain one scheduler round without losing any newly-finished request.

        Resident execution can make several outputs ready together: the prior decode output,
        followed by one or more terminal prefill outputs. ``finished_reqs`` from the preceding
        round must suppress already-launched speculative tokens in every one of those outputs,
        while requests that finish in an earlier output of this round must remain visible to
        later outputs and to the next round. Commit the union only after the whole ordered group
        has drained; an empty group intentionally preserves the prior suppression set.
        """
        if not outputs:
            return

        suppressed_reqs = set(self.finished_reqs)
        round_finished_reqs: Set[Req] = set()
        for data in outputs:
            newly_finished = self._process_last_data(
                data,
                suppressed_finished_reqs=suppressed_reqs,
            )
            round_finished_reqs.update(newly_finished)
            suppressed_reqs.update(newly_finished)
        self.finished_reqs = round_finished_reqs

    def _process_last_data(
        self,
        last_data: ForwardData | None,
        *,
        suppressed_finished_reqs: Set[Req] | None = None,
    ) -> Set[Req]:
        if last_data is None:
            return set()

        commit_finished_reqs = suppressed_finished_reqs is None
        if suppressed_finished_reqs is None:
            suppressed_finished_reqs = self.finished_reqs

        batch, (_, next_tokens_cpu, copy_done) = last_data[0].batch, last_data[1]
        copy_done.synchronize()
        reply: List[DetokenizeMsg] = []
        new_finished_reqs: Set[Req] = set()
        with self.cache_manager.lazy_free_region():
            for i, req in enumerate(batch.reqs):
                if isinstance(req, ChunkedReq):
                    # Don't cache intermediate chunks; the full prompt is cached once when the
                    # final chunk is processed. Caching here snapshots a handle the next chunk
                    # already copied (overlap), so cache_req double-frees the prior chunk.
                    if req.aborted:
                        # Aborted mid-chunked-prefill while this chunk was in flight: the abort
                        # popped the pending continuation (no next chunk launches), and this
                        # drain point frees the chunk's pages/slots exactly once.
                        self._free_req_resources(req)
                    continue
                if req.aborted:
                    # Aborted while this final-chunk prefill / decode step was in flight: free
                    # here (the forward is drained) and finish the request. No DetokenizeMsg --
                    # the abort ack flushed after this method stays the uid's terminal reply.
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                    continue
                if req in suppressed_finished_reqs:
                    # Overlap scheduling launched one more decode step for a request that
                    # already terminated (filter_reqs keeps it while output budget remains,
                    # and the next batch is scheduled before this drain runs). Its resources
                    # are freed below/already; shipping this token would append past the
                    # client's terminal reply.
                    continue
                next_token = next_tokens_cpu[i]
                req.append_host(next_token.unsqueeze(0))
                next_token = int(next_token.item())
                # ``can_decode`` tracks launched forwards; overlap may already have advanced it
                # for the ongoing batch. Length is reached only when this drained token fills
                # the host-visible output budget. EOS and stop strings still win over length.
                hit_length = req.input_ids.numel() == req.max_device_len
                hit_eos = (
                    not req.sampling_params.ignore_eos and next_token in self.eos_token_ids
                )
                matched_stop = (
                    self._match_stop_str(req)
                    if not hit_eos and req.sampling_params.stop_strs
                    else None
                )
                finished = hit_length or hit_eos or matched_stop is not None
                finish_reason = (
                    ("stop" if (hit_eos or matched_stop is not None) else "length")
                    if finished
                    else None
                )
                if (
                    next_token == self.toolcall_anchor_id
                    and req.toolcall_anchor_len is None
                    and not finished
                ):
                    req.toolcall_anchor_len = req.input_ids.numel()
                reply.append(
                    DetokenizeMsg(
                        uid=req.uid,
                        next_token=next_token,
                        finished=finished,
                        finish_reason=finish_reason,
                        matched_stop=matched_stop,
                        stop_strs=req.sampling_params.stop_strs or None,
                    )
                )

                # NOTE: overlap scheduling may make the request freed twice, skip second free
                if finished and req not in suppressed_finished_reqs:
                    self.decode_manager.remove_req(req)
                    self._free_req_resources(req)
                    new_finished_reqs.add(req)
                elif i >= batch.decode_size and req.table_idx != -1:
                    # for prefill, non-chunk req, cache the prefix.
                    # Polymorphic: the DSV4 naive manager keeps the request's slots (no-op);
                    # the generic manager inserts the prefix into its radix/naive cache.
                    # table_idx == -1 is defense-in-depth: aborts mark in-flight requests
                    # instead of freeing them (handled above), so a freed request should
                    # never reach this commit -- but if a future path frees one early, skip
                    # rather than re-read the freed page-table row (and on hybrid, deref the
                    # None'd GDN ping-pong slots).
                    self.cache_manager.cache_req(req, finished=False)

        if commit_finished_reqs:
            self.finished_reqs = new_finished_reqs
        # Stamp each reply with the post-batch KV page occupancy so the frontend (shell
        # status bar) can show live KV usage without a separate query.
        used, total = self._kv_usage_pages()
        mamba_slots = self._mamba_slot_usage()
        swa_tokens = self._swa_token_usage()
        if reply:
            mem = self._gpu_mem_bytes()
            mamba_used, mamba_total = mamba_slots or (0, 0)
            swa_used, swa_total = swa_tokens or (0, 0)
            for m in reply:
                m.kv_used_pages = used
                m.kv_total_pages = total
                m.mamba_used_slots = mamba_used
                m.mamba_total_slots = mamba_total
                m.swa_used_tokens = swa_used
                m.swa_total_tokens = swa_total
                m.gpu_mem_bytes = mem
        self.status_reporter.report_batch(
            batch,
            running_reqs=len(self.decode_manager.running_reqs),
            queue_reqs=len(self.prefill_manager.pending_list),
            kv_used_pages=used,
            kv_total_pages=total,
            page_size=self.config.page_size,
            mamba_slots=mamba_slots,
            swa_tokens=swa_tokens,
        )
        self.send_result(reply)
        return new_finished_reqs

    def _match_stop_str(self, req: Req) -> str | None:
        """First stop string present in this request's generated tail, else None. Decodes
        only a short suffix (bounded by the longest stop string's char length, so a stop of
        N chars spans at most N tokens) to keep the per-step cost small."""
        stop_strs = req.sampling_params.stop_strs
        prompt_len = req.max_device_len - req.output_len
        if len(req.input_ids) <= prompt_len:
            return None
        max_chars = max(len(s) for s in stop_strs)
        tail_start = max(prompt_len, len(req.input_ids) - (max_chars + 1))
        tail = self.tokenizer.decode(req.input_ids[tail_start:].tolist())
        for s in stop_strs:
            if s in tail:
                return s
        return None

    def _kv_usage_pages(self) -> Tuple[int, int]:
        """(used_pages, total_pages) of the KV page pool.

        ``used`` follows SGLang's logging semantics: allocated pages that are not
        evictable (active requests + protected prefix cache). Evictable prefix-cache
        pages are available to future requests, so they are excluded from usage.
        Always the manager's own primary pool (for DSV4 the FULL cmp/idx tier); the
        window (swa) tier is reported separately by ``_swa_token_usage``.
        """
        return self.cache_manager.page_usage()

    def _mamba_slot_usage(self) -> Tuple[int, int] | None:
        """(used_slots, total_slots) of the GDN-state (mamba) pool for hybrid models, else None.

        Mirrors SGLang's mamba-pool semantics: ``total`` excludes the reserved padding
        sink (slot 0); ``used`` excludes free slots and evictable tree snapshots.
        """
        if not self.cache_manager.is_hybrid:
            return None
        total = self.cache_manager.linear_state_pool.num_slots - 1
        return total - self.cache_manager.mamba_available_size, total

    def _swa_token_usage(self) -> Tuple[int, int] | None:
        """(used_tokens, total_tokens) of the window (swa) pool for SWA models, else None.

        Mirrors the mamba accounting: ``total`` excludes the pool's reserved sentinel
        unit; ``used`` excludes free slots and evictable (unlocked) tree tokens.
        """
        cm = self.cache_manager
        if not cm.swa_paged:
            return None
        total = cm.swa_pool.swa_num_tokens - 1
        return total - cm.swa_available_size, total

    def _gpu_mem_bytes(self) -> int:
        """Bytes this engine process holds on the GPU (torch's reserved caching-allocator
        pool: weights + KV + MoE cache + graphs). 0 on CPU. Cheap, no device sync."""
        if self.device.type != "cuda":
            return 0
        return torch.cuda.memory_reserved(self.device)

    def _process_one_msg(self, msg: BaseBackendMsg) -> None:
        if isinstance(msg, BatchBackendMsg):
            for msg in msg.data:
                self._process_one_msg(msg)
        elif isinstance(msg, ExitMsg):
            raise KeyboardInterrupt
        elif isinstance(msg, UserMsg):
            logger.debug_rank0("Received user msg: %s", msg)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is not None and msg.uid in tombstones:
                tombstones.pop(msg.uid, None)
                logger.debug_rank0(
                    "Dropping request %d because its abort arrived before admission", msg.uid
                )
                return
            input_len, max_seq_len = len(msg.input_ids), self.engine.max_seq_len
            max_output_len = max_seq_len - input_len
            if max_output_len <= 0:
                logger.warning_rank0(
                    f"Input sequence length {input_len} exceeds {max_seq_len}, "
                    f"request {msg.uid} is dropped."
                )
                # Tell the client instead of dropping silently — otherwise its wait_for_ack
                # never sees a `finished` reply and hangs until the request times out.
                self.send_result(
                    [
                        ErrorReplyMsg(
                            uid=msg.uid,
                            # "prompt is too long: N tokens > M" is the phrasing Claude Code and
                            # OpenClaw match on; the Anthropic wire has no error code to read.
                            error=(
                                f"prompt is too long: {input_len} tokens > {max_seq_len} maximum "
                                f"(prompt + generation); shorten the prompt or increase the KV "
                                f"cache budget"
                            ),
                            # OpenAI's standard class for this, for clients that read a code.
                            code="context_length_exceeded",
                        )
                    ]
                )
                return
            if msg.sampling_params.max_tokens > max_output_len:
                msg.sampling_params.max_tokens = max_output_len
                logger.warning_rank0(
                    f"Adjust max_tokens to {max_output_len} for request {msg.uid}."
                )
            self.prefill_manager.add_one_req(msg)
        elif isinstance(msg, AbortBackendMsg):
            logger.debug_rank0("Aborting request %d", msg.uid)
            tombstones = getattr(self, "_abort_tombstones", None)
            if tombstones is None:
                tombstones = self._abort_tombstones = {}
            tombstones[msg.uid] = None
            # Unknown aborts normally consume their tombstone when the cross-worker UserMsg
            # catches up. Bound hostile/no-followup abort traffic without affecting realistic
            # in-flight concurrency.
            while len(tombstones) > 65_536:
                tombstones.pop(next(iter(tombstones)))
            pending_req = self.prefill_manager.abort_req(msg.uid)
            layered_req = self._abort_layered_wave(msg.uid)
            resident_executor = getattr(self, "resident_executor", None)
            resident_req = (
                resident_executor.abort(msg.uid)
                if resident_executor is not None
                else None
            )
            decode_req = self.decode_manager.abort_req(msg.uid)
            req_to_free = (
                None
                if resident_req is not None
                else layered_req or pending_req or decode_req
            )
            if req_to_free is not None:
                # SGLang-style abort: never free resources under an in-flight forward. If the
                # request is in the launched-but-not-drained batch (overlap), only mark it;
                # _process_last_data frees it this same iteration, after copy_done.synchronize()
                # -- so its KV pages / GDN slots are never recycled mid-write, and the
                # finished=False prefix-commit can't run on a freed request. A request with no
                # forward in flight (e.g. a decode req starved behind a long chunked prefill)
                # is freed immediately -- deferring would leak until its next batch, which
                # strict prefill-priority puts arbitrarily far away.
                inflight = (
                    self._last_data is not None
                    and req_to_free in self._last_data[0].batch.reqs
                ) or any(
                    req_to_free in data[0].batch.reqs
                    for data in getattr(self, "_resident_last_outputs", ())
                )
                if inflight:
                    req_to_free.aborted = True
                else:
                    self._free_req_resources(req_to_free)
            # Always acknowledge the abort, even when the request already left the manager,
            # but NOT yet: overlap_loop still has to publish the prior forward's sampled reply.
            # _flush_abort_acks runs after _process_last_data, making this a true terminal
            # accounting barrier for FrontendManager/prepare-stop.
            self._pending_abort_acks.add(msg.uid)
        elif isinstance(msg, CacheRebuildBackendMsg):
            # v1 scope: only if_idle, single-rank, non-owned-KV. drain mode and TP rebuild
            # need the drain-gate / all-rank failure-agreement machinery (deferred), so we
            # reject them cleanly rather than ship hang-prone half-wired paths.
            if not self.cache_manager.supports_runtime_rebuild:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "this model's cache does not support runtime rebuild"
                )
            elif msg.mode != "if_idle":
                self._reply_rebuild(
                    msg.request_id, "unsupported", f"mode {msg.mode!r} unsupported (use if_idle)"
                )
            elif self.config.tp_info.size > 1:
                self._reply_rebuild(
                    msg.request_id, "unsupported", "runtime rebuild unsupported under TP > 1"
                )
            elif (
                self.layered_wave is not None
                or (
                    self.resident_executor is not None
                    and self.resident_executor.active
                )
                or self.prefill_manager.runnable
                or self.decode_manager.runnable
            ):
                # if_idle: refuse rather than wait. (finished_reqs hold no resources — they
                # are already freed — so they do not block a rebuild.)
                self._reply_rebuild(msg.request_id, "busy")
            else:
                self._pending_rebuild = msg
        else:
            logger.error(f"Unknown message type: {type(msg)}")
            raise NotImplementedError

    def _restore_linear_states(self, batch) -> None:
        """COW-restore a hybrid prefix hit's GDN snapshot into its freshly-allocated live slot
        (first chunk only). MUST run on the ENGINE stream so it is program-ordered after the
        prior batch's snapshot writes and before this forward reads the live slot."""
        pool = self.engine.linear_state_pool
        if pool is None or not batch.has_prefill:
            return
        for req in batch.prefill_reqs:
            if req.mamba_restore_src is not None:
                pool.copy_from(req.mamba_restore_src, req.linear_slot_idx)
                req.mamba_restore_src = None  # consumed: restore exactly once

    def _free_req_resources(self, req: Req) -> None:
        # Idempotent: an EOS-finished request can stay in running_reqs (output budget left), so an
        # abort in the same overlap iteration races _process_last_data and would free it twice --
        # double-freeing its table_idx and (hybrid) GDN slots onto the free-list, handing the same
        # slots to two later requests. table_idx == -1 marks an already-freed request.
        if req.table_idx == -1:
            return
        # Polymorphic free: the DSV4 manager returns the request's window pages + cmp/idx blocks
        # to their tier free-lists; the generic manager frees its KV pages (it reads
        # page_table[req.table_idx], so free the table entry after).
        self.cache_manager.cache_req(req, finished=True)
        self.table_manager.free(req.table_idx)
        req.table_idx = -1

    def _reply_rebuild(self, request_id: str, status: str, error: str | None = None) -> None:
        # Single source of truth with the rollback snapshot (_current_cache_geometry): mamba is
        # usable slots (padding sink excluded, matching the status-bar gauge), and num_swa_pages
        # reports 0 unless the model actually has a window pool.
        geo = self._current_cache_geometry()
        self.send_result(
            [
                CacheRebuildResultMsg(
                    request_id=request_id,
                    status=status,
                    moe_cache_size=geo["moe_cache_size"] or 0,
                    num_pages=geo["num_pages"],
                    mamba_slots=geo["num_mamba_slots"] or 0,
                    num_swa_pages=geo["num_swa_pages"] or 0,
                    error=error,
                )
            ]
        )

    def _execute_pending_rebuild(self) -> None:
        from freetoken.engine.engine import CacheRebuildRejected

        msg = self._pending_rebuild
        assert msg is not None
        self._pending_rebuild = None
        requested = {
            "moe_cache_size": msg.moe_cache_size,
            "num_pages": msg.num_pages,
            "num_mamba_slots": msg.num_mamba_slots,
            "num_swa_pages": msg.num_swa_pages,
        }
        # Rollback target: the CURRENT (serving) sizes of ONLY the pools this request touches.
        # Passing the untouched pools too would trip rebuild_cache's KV/mamba/SWA gate and wipe
        # the prefix cache that a successful resize of just the requested pool preserves.
        snapshot = self._current_cache_geometry()
        prior = {k: snapshot[k] for k, v in requested.items() if v is not None}
        # Cleared here, set by engine.rebuild_runtime_cache at its point of no return — lets the
        # except below tell a pre-teardown failure (engine untouched) from a mid-teardown one.
        self.engine.rebuild_teardown_started = False
        try:
            self.rebuild_cache(**requested)
        except CacheRebuildRejected as e:
            # Rejected before any destructive free — old cache intact, keep serving.
            logger.warning(f"cache rebuild rejected: {e}")
            self._reply_rebuild(msg.request_id, "rejected", error=str(e))
            return
        except Exception as e:  # noqa: BLE001
            if not getattr(self.engine, "rebuild_teardown_started", True):
                # Failed before the destructive phase began: graphs and pools are untouched and
                # the engine is still serving. A destructive rollback would only add risk.
                logger.error(f"cache rebuild failed before teardown: {e!r} — old cache intact")
                self._reply_rebuild(msg.request_id, "rejected", error=repr(e))
                return
            if self.config.tp_info.size > 1:
                # A lone-rank failure cannot be rolled back symmetrically: rebuild_cache runs TP
                # barriers, and ranks that succeeded will not re-enter them — a solo rollback
                # would desync the group. Keep the latch-failed behavior for tp>1.
                logger.error(f"cache rebuild failed: {e!r} — tp>1, latching failed")
                self._reply_rebuild(msg.request_id, "failed", error=repr(e))
                return
            # The destructive phase failed — typically a CUDA OOM while reallocating a pool or
            # recapturing graphs. The graphs/pools are already torn down, so the engine cannot
            # serve as-is. Rather than latch "failed" (which forces a full process restart),
            # rebuild the touched pools back to the sizes that were serving a moment ago: they
            # fit before, so shrinking back frees the just-attempted allocation and restores
            # service. Only if the rollback ALSO fails is the engine genuinely wedged. (Post-OOM
            # CUDA state is not guaranteed sane — a rollback that succeeds here may still surface
            # a deferred fault on a later request; that residual risk is accepted over always
            # forcing a restart.)
            logger.error(f"cache rebuild failed: {e!r} — rolling back to the previous geometry")
            try:
                self.rebuild_cache(**prior, _prefill_budget_event="rollback")
            except Exception as e2:  # noqa: BLE001 — rollback failed too; genuinely unrecoverable
                logger.error(f"cache rebuild rollback failed: {e2!r} — server latched failed")
                self._reply_rebuild(
                    msg.request_id,
                    "failed",
                    error=f"{e!r}; rollback to the prior geometry also failed: {e2!r}",
                )
                return
            logger.warning("cache rebuild rolled back to the previous geometry — still serving")
            self._log_cache_geometry("Cache rolled back")
            self._reply_rebuild(
                msg.request_id, "rejected", error=f"rebuild failed and was rolled back: {e!r}"
            )
            return
        # Outside the try: an ack/send failure after a fully-applied rebuild must not be
        # mistaken for a rebuild failure and roll back the geometry the engine now serves.
        self._log_cache_geometry("Cache rebuilt")
        self._reply_rebuild(msg.request_id, "ok")

    def _current_cache_geometry(self) -> dict:
        """The pools' current (serving) sizes as rebuild_cache kwargs — the rollback snapshot and
        the single source for _reply_rebuild's readout. None for a pool this model lacks
        (rebuild_cache skips those; the reply maps them to the wire format's 0). num_swa_pages is
        the CONCRETE current window (usable pages) so a rollback restores it byte-for-byte,
        whether it was pinned or ratio-derived."""
        eng = self.engine
        config = self.config
        mc = config.model_config
        num_swa_pages = None
        if getattr(mc, "dsv4_args", None) is not None:
            sizes = getattr(eng.kv_cache, "sizes", None)
            if sizes is not None:  # usable window pages = physical n_win_pages minus the dummy page
                num_swa_pages = max(0, sizes.n_win_pages - 1)
        elif getattr(mc, "has_swa_attention", False) and (
            getattr(config, "cache_type", None) == "swa_radix"
        ):  # usable window tokens = pool tokens minus the slot-0 sentinel
            num_swa_pages = max(0, int(getattr(eng.kv_cache, "swa_num_tokens", 0) or 0) - 1)
        return dict(
            num_pages=eng.num_pages,
            moe_cache_size=eng.moe_offload_cache.cache_size if eng.moe_offload_cache is not None else None,
            num_mamba_slots=(eng.linear_state_pool.num_slots - 1) if eng.linear_state_pool is not None else None,
            num_swa_pages=num_swa_pages,
        )

    def _log_cache_geometry(self, event: str) -> None:
        """One-line readout of every pool's new size + VRAM after a rebuild changed them:
        full KV always; swa/mamba/MoE only for models with the pool. Byte figures are
        best-effort (0 when a unit cost cannot be measured) and must never block the reply."""
        from freetoken.kvcache.cache_status import compute_cache_pools, compute_cache_unit_bytes

        try:
            pools = compute_cache_pools(self.engine)
            unit = compute_cache_unit_bytes(self.engine)
            kv_tokens = pools["num_pages"] * pools["page_size"]
            parts = [
                f"KV {pools['num_pages']} pages"
                f" ({kv_tokens} tokens, {_gib(kv_tokens * unit['kv_bytes_per_token'])})"
            ]
            if pools["num_swa_pages"]:
                swa_tokens = pools["num_swa_pages"] * pools["swa_page_size"]
                parts.append(
                    f"swa {pools['num_swa_pages']} pages"
                    f" ({swa_tokens} tokens, {_gib(swa_tokens * unit['swa_bytes_per_token'])})"
                )
            if pools["num_mamba_slots"]:
                parts.append(
                    f"mamba {pools['num_mamba_slots']} slots"
                    f" ({_gib(pools['num_mamba_slots'] * unit['mamba_bytes_per_slot'])})"
                )
            moe = self.engine.moe_offload_cache
            if moe is not None:
                parts.append(
                    f"MoE cache {moe.cache_size}/{moe.num_layers * moe.num_experts}"
                    f" ({_gib(moe.cache_size * unit['moe_bytes_per_expert'])})"
                )
                if self.config.batching_policy == "joint":
                    parts.append(
                        f"joint group {moe.effective_prefill_group_size} layers "
                        f"({moe.decode_cache_size} shared expert slots)"
                    )
                elif self.config.batching_policy == "layered-pipeline":
                    parts.append(
                        f"layered pipeline group {moe.effective_prefill_group_size} layers "
                        f"({moe.decode_cache_size} shared expert slots)"
                    )
            logger.info_rank0(f"{event}: " + ", ".join(parts))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"could not log cache geometry: {e!r}")

    def _prepare_batch(
        self, batch: Batch, *, graph_pad: bool = True
    ) -> ForwardInput:
        self._prepare_batch_resources(batch, graph_pad=graph_pad)
        if batch.has_prefill:
            self._gather_multimodal(batch)
        return self._build_forward_input(batch)

    def _prepare_batch_resources(
        self,
        batch: Batch,
        *,
        graph_pad: bool,
    ) -> None:
        """Apply per-iteration request accounting and allocate its cache rows once."""
        if graph_pad:
            self.engine.graph_runner.pad_batch(batch)
        else:
            batch.padded_reqs = list(batch.reqs)
        self._forward_iter += 1
        if batch.has_decode:
            # Free each decoding request's now-out-of-window SWA slots BEFORE the alloc below,
            # so they can back the new token -- this is what bounds the per-request swa
            # footprint during decode. (no-op unless the model is SWA / paged swa pool.)
            self.cache_manager.maybe_free_swa_out_of_window(
                batch.decode_reqs, forward_iter=self._forward_iter)
            for req in batch.decode_reqs:
                req.decode_batch_idx += 1
        if batch.has_prefill:
            # Prefill sibling of the decode driver: free out-of-window swa BEFORE allocating
            # this chunk, so a chunked prompt longer than the swa pool never accumulates its
            # whole swa footprint (which would exhaust alloc_swa). No-op unless SWA/paged.
            self.cache_manager.free_swa_out_of_window_extend(batch.prefill_reqs)
        # Polymorphic page allocation: DSV4 allocates window pages + cmp/idx blocks into its
        # slot maps; the generic manager allocates KV pages into the page table.
        self.cache_manager.allocate_paged(batch.reqs)

    def _prepare_resident_group_decode_batch(
        self,
        batch: Batch,
    ) -> ForwardInput:
        """Allocate decode once, then build its resident-group input."""
        self._resident_decode_input = None
        self._prepare_batch_resources(batch, graph_pad=False)
        return self._build_layer_group_input(batch)

    def _prepare_resident_batch(self, batch: Batch) -> ForwardInput:
        """Prepare one resident-wave request batch without graph padding."""
        self._resident_decode_input = None
        return self._prepare_batch(batch, graph_pad=False)

    def _prepare_resident_allocation_only_batch(
        self,
        batch: Batch,
    ) -> ForwardInput:
        """Allocate a resident wave while leaving metadata to its adapter views."""
        self._resident_decode_input = None
        self._prepare_batch_resources(batch, graph_pad=False)
        if batch.has_prefill:
            self._gather_multimodal(batch)
        return self._build_forward_input(batch, prepare_metadata=False)

    def _prepare_resident_decode_batch(self, batch: Batch) -> ForwardInput:
        """Prepare pure decode once and reuse stable resident-policy metadata."""
        forward_input, self._resident_decode_input = prepare_stable_decode(
            batch,
            self._resident_decode_input,
            engine=self.engine,
            token_pool=self.token_pool,
            prepare_resources=lambda current: self._prepare_batch_resources(
                current,
                graph_pad=True,
            ),
            build_forward_input=self._build_forward_input,
        )
        return forward_input

    def _build_layer_group_input(self, batch: Batch) -> ForwardInput:
        """Build an eager execution view after its requests were allocated once."""
        batch.padded_reqs = list(batch.reqs)
        if batch.has_prefill:
            self._gather_multimodal(batch)
        forward_input = self._build_forward_input(batch)
        batch.input_ids = self.token_pool[forward_input.input_tuple]
        return forward_input

    def _build_forward_input(
        self,
        batch: Batch,
        *,
        prepare_metadata: bool = True,
    ) -> ForwardInput:
        batch.positions = _make_positions(batch, self.device)
        input_mapping = _make_input_tuple(batch, self.device)
        write_mapping = _make_write_tuple(batch, self.device)
        batch.out_loc = self.engine.page_table[input_mapping]
        if prepare_metadata:
            self.engine.prepare_execution_metadata(
                batch,
                input_mapping,
                linear_cache_is_hybrid=self.cache_manager.is_hybrid,
            )
        return ForwardInput(
            batch=batch,
            sample_args=self.engine.sampler.prepare(batch),
            input_tuple=input_mapping,
            write_tuple=write_mapping,
        )

    def _gather_multimodal(self, batch: Batch) -> None:
        """Concatenate prefill requests' vision soft tokens in request order so the
        model can scatter them at image-token positions. ``req.mm_embeds``
        is kept (not cleared) so the cache manager can recognize multimodal requests and
        keep them out of the shared prefix cache (image placeholders share a token id but
        carry per-image content)."""
        parts = [req.mm_embeds for req in batch.prefill_reqs if req.mm_embeds is not None]
        if parts:
            batch.mm_embeds = torch.cat(parts, dim=0)

    def _schedule_next_batch(self) -> ForwardInput | None:
        assert self.batch_composer is not None
        batch = self.batch_composer.schedule_next_batch(self.prefill_budget)
        if batch is None:
            return None
        forward_input = self._prepare_batch(batch)
        self._report_prompt_admissions(batch)
        return forward_input

    def _report_prompt_admissions(self, batch: Batch) -> None:
        """Publish first-prefill accounting only after batch preparation succeeded.

        ``send_result`` is rank-aware: TP rank 0 forwards the signal, other ranks are
        no-ops. The offline handler explicitly ignores this online-accounting message.
        """
        if not batch.prompt_admissions:
            return
        self.send_result(
            [
                PromptAdmittedMsg(uid=uid, prompt_tokens=prompt_tokens, cached_tokens=cached_tokens)
                for uid, prompt_tokens, cached_tokens in batch.prompt_admissions
            ]
        )

    def _flush_abort_acks(self) -> None:
        pending = getattr(self, "_pending_abort_acks", None)
        if not pending:
            return
        uids = sorted(pending)
        pending.clear()
        self.send_result([ErrorReplyMsg(uid=uid, error="request aborted") for uid in uids])

    def _forward(self, forward_input: ForwardInput) -> ForwardOutput:
        batch, sample_args, input_mapping, output_mapping = forward_input
        batch.input_ids = self.token_pool[input_mapping]
        if self.toolcall_anchor_id is not None and batch.has_decode:
            self.cache_manager.snapshot_toolcall_anchor(batch.decode_reqs)
        forward_output = self.engine.forward_batch(batch, sample_args)
        self.token_pool[output_mapping] = forward_output.next_tokens_gpu
        self.decode_manager.filter_reqs(forward_input.batch.reqs)
        return forward_output


def _make_positions(batch: Batch, device: torch.device) -> torch.Tensor:
    needed_size = sum(r.extend_len for r in batch.padded_reqs)
    indices_host = torch.empty(needed_size, dtype=torch.int32, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        torch.arange(
            req.cached_len,
            req.device_len,
            dtype=torch.int32,
            out=indices_host[offset : offset + length],
        )
        offset += length
    return indices_host.to(device, non_blocking=True)


def _make_input_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_host = torch.empty(len(batch.positions), dtype=torch.int64, pin_memory=True)
    offset = 0
    for req in batch.padded_reqs:
        length = req.extend_len
        mapping_host[offset : offset + length].fill_(req.table_idx)
        offset += length
    return mapping_host.to(device, non_blocking=True), batch.positions.to(torch.int64)


def _make_write_tuple(batch: Batch, device: torch.device) -> Indice2D:
    mapping_list = [req.table_idx for req in batch.reqs]
    mapping_host = torch.tensor(mapping_list, dtype=torch.int64, pin_memory=True)
    write_list = [(req.device_len if req.can_decode else -1) for req in batch.reqs]
    write_host = torch.tensor(write_list, dtype=torch.int64, pin_memory=True)
    return mapping_host.to(device, non_blocking=True), write_host.to(device, non_blocking=True)
