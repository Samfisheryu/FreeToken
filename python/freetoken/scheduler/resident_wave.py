from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

import torch
from freetoken.core import Batch, Req

from .batch_composition import compose_mixed_batch
from .forward import ForwardData, ForwardInput
from .prefill import ChunkedReq, PrefillManager

if TYPE_CHECKING:
    from freetoken.engine import Engine
    from freetoken.models.blocks import LayerGroupState

    from .decode import DecodeManager
    from .table import TableManager


@dataclass
class ResidentFrontier:
    """One ragged batch containing at most one next chunk per wave member."""

    forward_input: ForwardInput
    state: LayerGroupState | None = None
    attention_metadata_ready: bool = True
    prefill_row_offset: int = 0

    @property
    def chunk_count(self) -> int:
        return len(self.forward_input.batch.prefill_reqs)

    def last_rows(self) -> list[int]:
        rows: list[int] = []
        offset = self.prefill_row_offset
        for req in self.forward_input.batch.prefill_reqs:
            offset += req.extend_len
            rows.append(offset - 1)
        return rows


@dataclass
class ResidentWaveMember:
    uid: int
    planned_chunks: int
    admitted_chunks: int = 0
    latest_req: Req | None = None
    terminal_frontier: int | None = None
    terminal_request_index: int | None = None
    terminal_output_row: int | None = None
    aborted: bool = False

    @property
    def terminal(self) -> bool:
        return self.terminal_frontier is not None


@dataclass
class ResidentWaveAdmission:
    """FIFO membership and complete-request accounting for one resident wave.

    ``soft_chunk_cap`` applies to the sum of complete requests admitted after
    the first member.  A first request larger than the cap remains intact and
    makes the wave exclusive.
    """

    soft_chunk_cap: int
    chunk_token_limit: int
    members: dict[int, ResidentWaveMember] = field(default_factory=dict)
    total_chunks: int = 0
    frozen: bool = False
    exclusive: bool = False

    @property
    def uids(self) -> set[int]:
        return set(self.members)

    @property
    def reserved_chunks(self) -> int:
        return sum(member.planned_chunks for member in self.members.values())

    @property
    def has_membership_capacity(self) -> bool:
        return not self.exclusive and self.reserved_chunks < self.soft_chunk_cap

    def refresh_members(self, prefill_manager: PrefillManager) -> list[int]:
        """Refresh exact continuations, then admit complete FIFO requests that fit."""
        if self.frozen:
            return []

        candidates = prefill_manager.pending_wave_candidates(self.chunk_token_limit)
        remaining_by_uid = {uid: chunks for uid, chunks, _ in candidates}
        for member in self.members.values():
            member.planned_chunks = member.admitted_chunks + remaining_by_uid.get(
                member.uid, 0
            )

        added: list[int] = []
        for uid, chunks, multimodal in candidates:
            if uid in self.members:
                continue
            if self.exclusive:
                break
            if not self.members:
                self.members[uid] = ResidentWaveMember(uid, chunks)
                added.append(uid)
                self.exclusive = multimodal or chunks > self.soft_chunk_cap
                if self.exclusive:
                    break
                continue
            if multimodal or self.reserved_chunks + chunks > self.soft_chunk_cap:
                break
            self.members[uid] = ResidentWaveMember(uid, chunks)
            added.append(uid)
        return added

    def pending_member_uids(self, prefill_manager: PrefillManager) -> set[int]:
        pending = {
            uid
            for uid, _, _ in prefill_manager.pending_wave_candidates(
                self.chunk_token_limit
            )
        }
        return self.uids & pending

    def has_unadmitted_pending(self, prefill_manager: PrefillManager) -> bool:
        return any(
            uid not in self.members
            for uid, _, _ in prefill_manager.pending_wave_candidates(
                self.chunk_token_limit
            )
        )

    def record_frontier(self, frontier_index: int, frontier: ResidentFrontier) -> None:
        rows = frontier.last_rows()
        for request_index, (req, output_row) in enumerate(
            zip(frontier.forward_input.batch.prefill_reqs, rows)
        ):
            member = self.members.get(req.uid)
            if member is None:
                raise RuntimeError(f"resident frontier contains unadmitted uid {req.uid}")
            member.admitted_chunks += 1
            member.latest_req = req
            self.total_chunks += 1
            if not isinstance(req, ChunkedReq):
                member.terminal_frontier = frontier_index
                member.terminal_request_index = (
                    frontier.forward_input.batch.decode_size + request_index
                )
                member.terminal_output_row = output_row

    def freeze(self) -> None:
        self.frozen = True


@dataclass
class ResidentWaveState:
    """Execution-independent state shared by resident-wave policies."""

    admission: ResidentWaveAdmission
    num_layers: int
    group_size: int
    frontiers: list[ResidentFrontier]
    current_layer: int = 0
    next_frontier: int = 0
    admission_complete: bool = False
    resident_group_active: bool = False
    awaiting_join_boundary: bool = False
    layer_prepares_at_start: int = 0

    @property
    def current_group_end(self) -> int:
        return min(self.current_layer + self.group_size, self.num_layers)

    @property
    def done(self) -> bool:
        return self.current_layer >= self.num_layers

    def finish_group(self) -> None:
        if self.done:
            raise RuntimeError("resident wave is already complete")
        self.current_layer = self.current_group_end


@dataclass(frozen=True)
class ResidentSchedule:
    batch: Batch | None
    admission: ResidentWaveAdmission | None
    deferred_join_members: tuple[int, ...] | None


class ResidentExecutor(Protocol):
    """Scheduler-facing contract shared by resident-wave policies."""

    @property
    def active(self) -> bool: ...

    def schedule_first_batch(self, token_budget: int) -> Batch | None: ...

    def begin_wave(self, first_batch: Batch, token_budget: int) -> None: ...

    def prepare_step(self, token_budget: int) -> None: ...

    def advance_step(self) -> list[ForwardData]: ...

    def abort(self, uid: int) -> Req | None: ...


def schedule_resident_wave(
    decode_reqs: list[Req],
    *,
    prefill_manager: PrefillManager,
    token_budget: int,
    soft_chunk_cap: int,
    max_frontier_chunks: int | None,
    deferred_join_members: tuple[int, ...] | None,
) -> ResidentSchedule:
    """Select the first frontier while preserving whole-request FIFO membership."""
    admission = ResidentWaveAdmission(soft_chunk_cap, token_budget)
    admission.refresh_members(prefill_manager)
    member_snapshot = tuple(admission.members)
    if (
        decode_reqs
        and member_snapshot
        and admission.has_membership_capacity
        and not admission.has_unadmitted_pending(prefill_manager)
        and deferred_join_members != member_snapshot
    ):
        return ResidentSchedule(
            compose_mixed_batch(decode_reqs, None),
            None,
            member_snapshot,
        )

    pending_uids = admission.pending_member_uids(prefill_manager)
    max_reqs = len(pending_uids)
    if max_frontier_chunks is not None:
        max_reqs = min(max_reqs, max_frontier_chunks)
    prefill_batch = prefill_manager.schedule_next_batch(
        token_budget * max_reqs,
        chunk_token_limit=token_budget,
        allowed_uids=pending_uids,
        max_reqs=max_reqs,
    )
    if prefill_batch is not None or not member_snapshot or not decode_reqs:
        deferred_join_members = None
    return ResidentSchedule(
        compose_mixed_batch(decode_reqs, prefill_batch),
        admission if prefill_batch is not None else None,
        deferred_join_members,
    )


def prepare_resident_frontier(
    batch: Batch,
    *,
    prepare_batch: Callable[[Batch], ForwardInput],
    report_prompt_admissions: Callable[[Batch], None],
    table_manager: TableManager,
    prefill_manager: PrefillManager,
    prefill_row_offset: int = 0,
) -> ResidentFrontier:
    """Allocate one frontier exactly once, then advance its chunk cursors."""
    forward_input = prepare_batch(batch)
    report_prompt_admissions(batch)
    batch.input_ids = table_manager.token_pool[forward_input.input_tuple]
    for req in batch.prefill_reqs:
        prefill_manager.reserve_layered_continuation(req)
    return ResidentFrontier(
        forward_input=forward_input,
        prefill_row_offset=prefill_row_offset,
    )


def admit_resident_frontiers(
    wave: ResidentWaveState,
    *,
    prefill_manager: PrefillManager,
    token_budget: int,
    prepare_frontier: Callable[[Batch], ResidentFrontier],
    max_chunks: int | None = None,
) -> list[ResidentFrontier]:
    """Admit FIFO member frontiers, with at most one chunk per UID per pass."""
    admitted: list[ResidentFrontier] = []
    remaining = max_chunks
    while remaining is None or remaining > 0:
        wave.admission.refresh_members(prefill_manager)
        pending_uids = wave.admission.pending_member_uids(prefill_manager)
        if not pending_uids:
            break
        max_reqs = len(pending_uids)
        if remaining is not None:
            max_reqs = min(max_reqs, remaining)
        batch = prefill_manager.schedule_next_batch(
            token_budget * max_reqs,
            chunk_token_limit=token_budget,
            allowed_uids=pending_uids,
            max_reqs=max_reqs,
        )
        if batch is None:
            break
        frontier = prepare_frontier(batch)
        frontier_index = len(wave.frontiers)
        wave.frontiers.append(frontier)
        wave.admission.record_frontier(frontier_index, frontier)
        admitted.append(frontier)
        if remaining is not None:
            remaining -= frontier.chunk_count
    wave.admission.refresh_members(prefill_manager)
    return admitted


def resolve_group_zero_admission(
    wave: ResidentWaveState,
    *,
    prefill_manager: PrefillManager,
    has_decode: bool,
) -> None:
    """Freeze membership, or leave one scheduler boundary for later arrivals."""
    pending_members = wave.admission.pending_member_uids(prefill_manager)
    if pending_members:
        wave.awaiting_join_boundary = False
        wave.admission_complete = False
        return
    if (
        has_decode
        and wave.admission.has_membership_capacity
        and not wave.admission.has_unadmitted_pending(prefill_manager)
    ):
        wave.awaiting_join_boundary = True
        wave.admission_complete = False
        return
    wave.admission.freeze()
    wave.admission_complete = True
    wave.awaiting_join_boundary = False


def finish_resident_prefill(
    wave: ResidentWaveState,
    *,
    engine: Engine,
    decode_manager: DecodeManager,
    table_manager: TableManager,
    free_req_resources: Callable[[Req], None],
    commit_chunks: bool = True,
) -> list[ForwardData]:
    """Commit every chunk and publish each member's terminal prompt row once."""
    if commit_chunks:
        commit_resident_chunks(wave)

    terminals: dict[int, list[tuple[int, int]]] = {}
    aborted_owners: list[Req] = []
    for member in wave.admission.members.values():
        owner = member.latest_req
        if owner is None:
            continue
        if member.aborted or owner.aborted:
            aborted_owners.append(owner)
            continue
        if not member.terminal:
            raise RuntimeError(
                f"resident wave completed before uid {member.uid} reached its tail"
            )
        assert member.terminal_frontier is not None
        assert member.terminal_request_index is not None
        assert member.terminal_output_row is not None
        terminals.setdefault(member.terminal_frontier, []).append(
            (member.terminal_request_index, member.terminal_output_row)
        )

    outputs: list[ForwardData] = []
    for frontier_index, selected in sorted(terminals.items()):
        frontier = wave.frontiers[frontier_index]
        if frontier.state is None:
            raise RuntimeError("resident wave completed without final model state")
        request_indices = [request_index for request_index, _ in selected]
        output_indices = torch.tensor(
            [output_row for _, output_row in selected],
            dtype=torch.int32,
            pin_memory=engine.device.type == "cuda",
        ).to(engine.device, non_blocking=True)
        output = engine.finish_layer_group_prefill(
            frontier.forward_input.batch,
            frontier.state,
            frontier.forward_input.sample_args,
            output_indices=output_indices,
            request_indices=request_indices,
        )
        output_input = request_output_view(frontier.forward_input, request_indices)
        write_and_filter(
            output_input,
            output.next_tokens_gpu,
            table_manager,
            decode_manager,
        )
        outputs.append((output_input, output))

    if aborted_owners:
        engine.stream.synchronize()
        for owner in aborted_owners:
            free_req_resources(owner)
    for frontier in wave.frontiers:
        frontier.state = None
    return outputs


def commit_resident_chunks(wave: ResidentWaveState) -> None:
    for frontier in wave.frontiers:
        for req in frontier.forward_input.batch.prefill_reqs:
            if isinstance(req, ChunkedReq):
                req.commit_prefill_kv()


def abort_resident_member(wave: ResidentWaveState, uid: int) -> Req | None:
    member = wave.admission.members.get(uid)
    if member is None:
        return None
    owner = member.latest_req
    if owner is None:
        del wave.admission.members[uid]
        return None
    member.aborted = True
    owner.aborted = True
    return owner


def write_and_filter(
    forward_input: ForwardInput,
    next_tokens_gpu: torch.Tensor,
    table_manager: TableManager,
    decode_manager: DecodeManager,
) -> None:
    table_manager.token_pool[forward_input.write_tuple] = next_tokens_gpu
    decode_manager.filter_reqs(forward_input.batch.reqs)


def request_output_view(
    forward_input: ForwardInput, request_indices: list[int]
) -> ForwardInput:
    """Build the request-aligned output view after logits were selected by row."""
    from freetoken.engine.sample import BatchSamplingArgs

    batch = Batch(
        reqs=[forward_input.batch.reqs[index] for index in request_indices],
        decode_size=0,
    )
    args = forward_input.sample_args
    selected_args = BatchSamplingArgs(
        temperatures=(
            args.temperatures[request_indices]
            if args.temperatures is not None
            else None
        ),
        top_k=(args.top_k[request_indices] if args.top_k is not None else None),
        top_p=(args.top_p[request_indices] if args.top_p is not None else None),
    )
    write_tuple = (
        forward_input.write_tuple[0][request_indices],
        forward_input.write_tuple[1][request_indices],
    )
    return ForwardInput(
        batch=batch,
        sample_args=selected_args,
        input_tuple=forward_input.input_tuple,
        write_tuple=write_tuple,
    )


__all__ = [
    "ResidentFrontier",
    "ResidentExecutor",
    "ResidentWaveAdmission",
    "ResidentWaveMember",
    "ResidentSchedule",
    "ResidentWaveState",
    "abort_resident_member",
    "admit_resident_frontiers",
    "commit_resident_chunks",
    "finish_resident_prefill",
    "prepare_resident_frontier",
    "request_output_view",
    "resolve_group_zero_admission",
    "schedule_resident_wave",
    "write_and_filter",
]
