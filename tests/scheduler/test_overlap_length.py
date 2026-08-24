"""CPU tests for sampled-token settlement under overlap scheduling."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.message import DetokenizeMsg
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.scheduler import Scheduler


def _scheduler(*, eos_token_ids=()):
    sent = []
    freed = []
    decode_manager = DecodeManager(page_size=1)
    cache_manager = SimpleNamespace(
        lazy_free_region=lambda: nullcontext(),
        cache_req=lambda _req, *, finished: None,
    )
    scheduler = SimpleNamespace(
        cache_manager=cache_manager,
        decode_manager=decode_manager,
        prefill_manager=SimpleNamespace(pending_list=[]),
        finished_reqs=set(),
        eos_token_ids=set(eos_token_ids),
        toolcall_anchor_id=None,
        config=SimpleNamespace(page_size=1),
        status_reporter=SimpleNamespace(report_batch=lambda *_, **__: None),
        send_result=sent.extend,
        _kv_usage_pages=lambda: (0, 32),
        _mamba_slot_usage=lambda: None,
        _swa_token_usage=lambda: None,
        _gpu_mem_bytes=lambda: 0,
        _match_stop_str=lambda _req: None,
    )

    def free_req(req):
        freed.append(req)
        req.table_idx = -1

    scheduler._free_req_resources = free_req
    return scheduler, decode_manager, sent, freed


def _request(*, max_tokens=3, stop_strs=None):
    return Req(
        input_ids=torch.tensor([10, 11], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=max_tokens,
        uid=7,
        sampling_params=SamplingParams(
            max_tokens=max_tokens,
            stop_strs=[] if stop_strs is None else stop_strs,
        ),
        cache_handle=SimpleNamespace(),
    )


def _launch(batch, decode_manager):
    """Model Engine.forward_batch: advance launched state before settling last_data."""
    for req in batch.reqs:
        req.complete_one()
    decode_manager.filter_reqs(batch.reqs)


def _drain(scheduler, batch, token):
    last_data = (
        SimpleNamespace(batch=batch),
        (
            None,
            torch.tensor([token], dtype=torch.int32),
            SimpleNamespace(synchronize=lambda: None),
        ),
    )
    Scheduler._process_last_data(scheduler, last_data)


def _token_replies(sent):
    return [message for message in sent if isinstance(message, DetokenizeMsg)]


def test_overlap_length_keeps_the_last_launched_token():
    scheduler, decode_manager, sent, freed = _scheduler()
    req = _request(max_tokens=3)

    last_batch = Batch(reqs=[req], phase="prefill")
    _launch(last_batch, decode_manager)

    # Each overlap iteration launches ongoing first, then settles last_batch.
    ongoing_batch = decode_manager.schedule_next_batch()
    assert ongoing_batch is not None
    _launch(ongoing_batch, decode_manager)
    _drain(scheduler, last_batch, 101)

    last_batch = ongoing_batch
    ongoing_batch = decode_manager.schedule_next_batch()
    assert ongoing_batch is not None
    _launch(ongoing_batch, decode_manager)
    _drain(scheduler, last_batch, 102)

    # All three forwards are now launched, but only two sampled tokens are settled.
    assert not req.can_decode
    assert req.input_ids.tolist() == [10, 11, 101, 102]
    assert req.table_idx != -1
    assert not _token_replies(sent)[-1].finished

    _drain(scheduler, ongoing_batch, 103)

    replies = _token_replies(sent)
    assert [message.next_token for message in replies] == [101, 102, 103]
    assert [message.finished for message in replies] == [False, False, True]
    assert replies[-1].finish_reason == "length"
    assert req.input_ids.tolist() == [10, 11, 101, 102, 103]
    assert freed == [req]


def test_non_overlap_length_still_returns_exact_budget():
    scheduler, decode_manager, sent, freed = _scheduler()
    req = _request(max_tokens=3)

    batch = Batch(reqs=[req], phase="prefill")
    for token in (101, 102, 103):
        _launch(batch, decode_manager)
        _drain(scheduler, batch, token)
        batch = decode_manager.schedule_next_batch()

    replies = _token_replies(sent)
    assert [message.next_token for message in replies] == [101, 102, 103]
    assert [message.finished for message in replies] == [False, False, True]
    assert replies[-1].finish_reason == "length"
    assert batch is None
    assert freed == [req]


def test_overlap_eos_still_wins_and_drops_speculative_next_token():
    scheduler, decode_manager, sent, freed = _scheduler(eos_token_ids={42})
    req = _request(max_tokens=3)

    last_batch = Batch(reqs=[req], phase="prefill")
    _launch(last_batch, decode_manager)
    ongoing_batch = decode_manager.schedule_next_batch()
    assert ongoing_batch is not None
    _launch(ongoing_batch, decode_manager)

    _drain(scheduler, last_batch, 42)
    _drain(scheduler, ongoing_batch, 99)

    replies = _token_replies(sent)
    assert len(replies) == 1
    assert replies[0].next_token == 42
    assert replies[0].finished and replies[0].finish_reason == "stop"
    assert req.input_ids.tolist() == [10, 11, 42]
    assert freed == [req]


def test_stop_string_still_wins_when_it_meets_the_length_limit():
    scheduler, decode_manager, sent, freed = _scheduler()
    req = _request(max_tokens=1, stop_strs=["END"])
    scheduler._match_stop_str = lambda _req: "END"
    batch = Batch(reqs=[req], phase="prefill")

    _launch(batch, decode_manager)
    _drain(scheduler, batch, 77)

    reply = _token_replies(sent)[0]
    assert reply.finished and reply.finish_reason == "stop"
    assert reply.matched_stop == "END"
    assert freed == [req]
