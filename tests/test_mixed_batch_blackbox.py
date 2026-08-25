import pytest

from freetoken.core import Batch
from freetoken.scheduler.mixed_batch import MixedBatchComposer


class FakeRequest:
    def __init__(self, uid, extend_len):
        self.uid = uid
        self.extend_len = extend_len

    def __repr__(self):
        return f"FakeRequest(uid={self.uid!r}, extend_len={self.extend_len})"


class QueueDecodeManager:
    def __init__(self, *batches):
        self._batches = list(batches)
        self.calls = 0

    def schedule_next_batch(self):
        self.calls += 1
        if not self._batches:
            return None
        return self._batches.pop(0)


class StickyDecodeManager:
    def __init__(self, batch):
        self.batch = batch
        self.calls = 0

    def schedule_next_batch(self):
        self.calls += 1
        return self.batch


class QueuePrefillManager:
    def __init__(self, *batches, requires_exclusive_batch=False):
        self._batches = list(batches)
        self.budgets = []
        self.calls = 0
        self.requires_exclusive_batch = requires_exclusive_batch

    def schedule_next_batch(self, prefill_budget):
        self.calls += 1
        self.budgets.append(prefill_budget)
        if prefill_budget <= 0 or not self._batches:
            return None
        return self._batches.pop(0)


def make_decode_batch(*reqs):
    return Batch(reqs=list(reqs), decode_size=len(reqs))


def make_prefill_batch(*reqs):
    return Batch(reqs=list(reqs), decode_size=0)


def assert_public_batch_state(batch, decode_reqs, prefill_reqs):
    assert list(batch.decode_reqs) == decode_reqs
    assert list(batch.prefill_reqs) == prefill_reqs
    assert batch.has_decode is bool(decode_reqs)
    assert batch.has_prefill is bool(prefill_reqs)
    assert batch.is_mixed is bool(decode_reqs and prefill_reqs)
    assert batch.is_decode_only is bool(decode_reqs and not prefill_reqs)
    assert batch.uses_extend_path is bool(prefill_reqs)
    assert batch.size == len(decode_reqs) + len(prefill_reqs)


def test_empty_queues_return_none_and_offer_the_full_budget_to_prefill():
    prefill_manager = QueuePrefillManager()
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=QueueDecodeManager(),
    )

    assert composer.schedule_next_batch(token_budget=7) is None
    assert prefill_manager.budgets == [7]


def test_pure_decode_batch_has_decode_only_public_state():
    decode_reqs = [FakeRequest("d1", 1), FakeRequest("d2", 1)]
    composer = MixedBatchComposer(
        prefill_manager=QueuePrefillManager(),
        decode_manager=QueueDecodeManager(make_decode_batch(*decode_reqs)),
    )

    batch = composer.schedule_next_batch(token_budget=8)

    assert batch is not None
    assert_public_batch_state(batch, decode_reqs, [])


def test_pure_prefill_batch_has_extend_path_public_state():
    prefill_reqs = [FakeRequest("p1", 3), FakeRequest("p2", 2)]
    prefill_manager = QueuePrefillManager(make_prefill_batch(*prefill_reqs))
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=QueueDecodeManager(),
    )

    batch = composer.schedule_next_batch(token_budget=5)

    assert batch is not None
    assert prefill_manager.budgets == [5]
    assert_public_batch_state(batch, [], prefill_reqs)


def test_mixed_batch_keeps_decode_before_prefill_and_uses_remaining_budget():
    decode_reqs = [FakeRequest("d1", 1), FakeRequest("d2", 1)]
    prefill_reqs = [FakeRequest("p1", 4)]
    decode_manager = QueueDecodeManager(make_decode_batch(*decode_reqs))
    prefill_manager = QueuePrefillManager(make_prefill_batch(*prefill_reqs))
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
    )

    batch = composer.schedule_next_batch(token_budget=6)

    assert batch is not None
    assert decode_manager.calls == 1
    assert prefill_manager.calls == 1
    assert prefill_manager.budgets == [4]
    assert_public_batch_state(batch, decode_reqs, prefill_reqs)


def test_budget_equal_to_decode_count_defers_prefill_to_next_round():
    decode_reqs = [FakeRequest("d1", 1), FakeRequest("d2", 1), FakeRequest("d3", 1)]
    prefill_req = FakeRequest("p1", 3)
    prefill_manager = QueuePrefillManager(make_prefill_batch(prefill_req))
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=QueueDecodeManager(make_decode_batch(*decode_reqs)),
    )

    first = composer.schedule_next_batch(token_budget=3)
    second = composer.schedule_next_batch(token_budget=3)

    assert first is not None
    assert second is not None
    assert prefill_manager.budgets == [0, 3]
    assert_public_batch_state(first, decode_reqs, [])
    assert_public_batch_state(second, [], [prefill_req])


def test_budget_smaller_than_decode_count_selects_a_bounded_decode_subset():
    decode_reqs = [FakeRequest(f"d{index}", 1) for index in range(6)]
    prefill_manager = QueuePrefillManager()
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=StickyDecodeManager(make_decode_batch(*decode_reqs)),
    )

    batch = composer.schedule_next_batch(token_budget=2)

    assert batch is not None
    selected = list(batch.decode_reqs)
    assert 0 < len(selected) < len(decode_reqs)
    assert all(req in decode_reqs for req in selected)
    assert sum(req.extend_len for req in selected) <= 2
    assert prefill_manager.budgets == [0]
    assert_public_batch_state(batch, selected, [])


def test_decode_extend_lengths_are_summed_before_offering_prefill_budget():
    decode_reqs = [FakeRequest("d1", 1), FakeRequest("d2", 2), FakeRequest("d3", 1)]
    prefill_req = FakeRequest("p1", 3)
    prefill_manager = QueuePrefillManager(make_prefill_batch(prefill_req))
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=QueueDecodeManager(make_decode_batch(*decode_reqs)),
    )

    batch = composer.schedule_next_batch(token_budget=7)

    assert batch is not None
    assert prefill_manager.budgets == [3]
    assert_public_batch_state(batch, decode_reqs, [prefill_req])


def test_consecutive_chunks_are_composed_only_in_the_round_that_returns_them():
    decode_one = FakeRequest("d1", 1)
    decode_two = FakeRequest("d2", 1)
    chunk_one = FakeRequest("shared-prompt", 3)
    chunk_two = FakeRequest("shared-prompt", 3)
    prefill_manager = QueuePrefillManager(
        make_prefill_batch(chunk_one),
        make_prefill_batch(chunk_two),
    )
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=QueueDecodeManager(
            make_decode_batch(decode_one),
            make_decode_batch(decode_two),
        ),
    )

    first = composer.schedule_next_batch(token_budget=4)
    second = composer.schedule_next_batch(token_budget=4)

    assert first is not None
    assert second is not None
    assert prefill_manager.budgets == [3, 3]
    assert_public_batch_state(first, [decode_one], [chunk_one])
    assert_public_batch_state(second, [decode_two], [chunk_two])


@pytest.mark.parametrize(
    ("log_new_tokens", "log_cached_tokens", "prompt_admissions"),
    [
        (0, 0, []),
        (13, 8, [("p1", 2), ("p2", 3)]),
    ],
    ids=["zero-counters-and-empty-admissions", "multiple-admissions"],
)
def test_mixed_batch_preserves_prefill_metadata_unchanged(
    log_new_tokens,
    log_cached_tokens,
    prompt_admissions,
):
    decode_req = FakeRequest("d1", 1)
    prefill_req = FakeRequest("p1", 2)
    prefill_batch = make_prefill_batch(prefill_req)
    prefill_batch.log_new_tokens = log_new_tokens
    prefill_batch.log_cached_tokens = log_cached_tokens
    prefill_batch.prompt_admissions = prompt_admissions
    composer = MixedBatchComposer(
        prefill_manager=QueuePrefillManager(prefill_batch),
        decode_manager=QueueDecodeManager(make_decode_batch(decode_req)),
    )

    batch = composer.schedule_next_batch(token_budget=3)

    assert batch is not None
    assert batch.log_new_tokens == log_new_tokens
    assert batch.log_cached_tokens == log_cached_tokens
    assert batch.prompt_admissions == prompt_admissions
    assert_public_batch_state(batch, [decode_req], [prefill_req])


def test_exclusive_prefill_gets_full_budget_without_calling_decode_manager():
    decode_req = FakeRequest("d1", 1)
    prefill_req = FakeRequest("multimodal-prompt", 9)
    decode_manager = QueueDecodeManager(make_decode_batch(decode_req))
    prefill_manager = QueuePrefillManager(
        make_prefill_batch(prefill_req),
        requires_exclusive_batch=True,
    )
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
    )

    batch = composer.schedule_next_batch(token_budget=9)

    assert batch is not None
    assert prefill_manager.budgets == [9]
    assert prefill_manager.calls == 1
    assert decode_manager.calls == 0
    assert_public_batch_state(batch, [], [prefill_req])


def test_prefill_admission_none_still_returns_the_decode_batch():
    decode_req = FakeRequest("d1", 1)
    decode_manager = QueueDecodeManager(make_decode_batch(decode_req))
    prefill_manager = QueuePrefillManager(None)
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
    )

    batch = composer.schedule_next_batch(token_budget=4)

    assert batch is not None
    assert decode_manager.calls == 1
    assert prefill_manager.calls == 1
    assert prefill_manager.budgets == [3]
    assert_public_batch_state(batch, [decode_req], [])


def test_failed_exclusive_prefill_returns_a_budgeted_decode_subset():
    decode_reqs = [
        FakeRequest("d1", 1),
        FakeRequest("d2", 9),
        FakeRequest("d3", 1),
    ]
    decode_manager = QueueDecodeManager(make_decode_batch(*decode_reqs))
    prefill_manager = QueuePrefillManager(
        None,
        requires_exclusive_batch=True,
    )
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
    )

    batch = composer.schedule_next_batch(token_budget=3)

    assert batch is not None
    assert prefill_manager.calls == 1
    assert prefill_manager.budgets == [3]
    assert decode_manager.calls == 1
    selected = list(batch.decode_reqs)
    assert selected
    assert decode_reqs[1] not in selected
    assert all(req in decode_reqs for req in selected)
    assert sum(req.extend_len for req in selected) <= 3
    assert_public_batch_state(batch, selected, [])


def test_zero_budget_schedules_no_work_and_preserves_requests_for_next_round():
    decode_req = FakeRequest("d1", 1)
    prefill_req = FakeRequest("p1", 1)
    prefill_manager = QueuePrefillManager(make_prefill_batch(prefill_req))
    decode_manager = StickyDecodeManager(make_decode_batch(decode_req))
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
    )

    empty = composer.schedule_next_batch(token_budget=0)
    next_batch = composer.schedule_next_batch(token_budget=2)

    assert empty is None
    assert next_batch is not None
    assert decode_manager.calls == 2
    assert prefill_manager.calls == 2
    assert prefill_manager.budgets == [0, 1]
    assert_public_batch_state(next_batch, [decode_req], [prefill_req])


def test_zero_budget_schedules_neither_exclusive_prefill_nor_decode():
    prefill_req = FakeRequest("exclusive", 1)
    prefill_manager = QueuePrefillManager(
        make_prefill_batch(prefill_req),
        requires_exclusive_batch=True,
    )
    decode_manager = QueueDecodeManager()
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=decode_manager,
    )

    batch = composer.schedule_next_batch(token_budget=0)

    assert batch is None
    assert decode_manager.calls == 1
    assert prefill_manager.calls == 1
    assert prefill_manager.budgets == [0]


def test_over_budget_decode_selection_rotates_until_every_request_runs():
    decode_reqs = [FakeRequest(f"d{index}", 1) for index in range(5)]
    decode_manager = StickyDecodeManager(make_decode_batch(*decode_reqs))
    composer = MixedBatchComposer(
        prefill_manager=QueuePrefillManager(),
        decode_manager=decode_manager,
    )

    selected_uids = set()
    for _ in range(len(decode_reqs)):
        batch = composer.schedule_next_batch(token_budget=2)
        assert batch is not None
        assert sum(req.extend_len for req in batch.decode_reqs) <= 2
        selected_uids.update(req.uid for req in batch.decode_reqs)

    assert selected_uids == {req.uid for req in decode_reqs}
    assert decode_manager.calls == len(decode_reqs)


def test_oversized_decode_is_skipped_when_other_runnable_requests_fit():
    oversized = FakeRequest("too-large", 5)
    small_one = FakeRequest("small-1", 1)
    small_two = FakeRequest("small-2", 2)
    composer = MixedBatchComposer(
        prefill_manager=QueuePrefillManager(),
        decode_manager=StickyDecodeManager(
            make_decode_batch(oversized, small_one, small_two)
        ),
    )

    batch = composer.schedule_next_batch(token_budget=3)

    assert batch is not None
    selected = list(batch.decode_reqs)
    assert selected
    assert oversized not in selected
    assert all(req in (small_one, small_two) for req in selected)
    assert sum(req.extend_len for req in selected) <= 3


def test_when_no_decode_fits_prefill_receives_the_whole_budget():
    oversized_decode = FakeRequest("too-large", 5)
    prefill_req = FakeRequest("p1", 3)
    prefill_manager = QueuePrefillManager(make_prefill_batch(prefill_req))
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=StickyDecodeManager(make_decode_batch(oversized_decode)),
    )

    batch = composer.schedule_next_batch(token_budget=3)

    assert batch is not None
    assert prefill_manager.budgets == [3]
    assert_public_batch_state(batch, [], [prefill_req])


def test_composition_does_not_mutate_inputs_and_returns_a_new_batch():
    decode_req = FakeRequest("d1", 1)
    prefill_req = FakeRequest("p1", 2)
    decode_batch = make_decode_batch(decode_req)
    prefill_batch = make_prefill_batch(prefill_req)
    decode_batch.log_new_tokens = 4
    decode_batch.log_cached_tokens = 5
    decode_batch.prompt_admissions = [("decode-input", 1)]
    prefill_batch.log_new_tokens = 6
    prefill_batch.log_cached_tokens = 7
    prefill_batch.prompt_admissions = [("prefill-input", 2)]
    composer = MixedBatchComposer(
        prefill_manager=QueuePrefillManager(prefill_batch),
        decode_manager=QueueDecodeManager(decode_batch),
    )

    output = composer.schedule_next_batch(token_budget=3)

    assert output is not None
    assert output is not decode_batch
    assert output is not prefill_batch
    assert list(decode_batch.reqs) == [decode_req]
    assert decode_batch.decode_size == 1
    assert decode_batch.log_new_tokens == 4
    assert decode_batch.log_cached_tokens == 5
    assert decode_batch.prompt_admissions == [("decode-input", 1)]
    assert list(prefill_batch.reqs) == [prefill_req]
    assert prefill_batch.decode_size == 0
    assert prefill_batch.log_new_tokens == 6
    assert prefill_batch.log_cached_tokens == 7
    assert prefill_batch.prompt_admissions == [("prefill-input", 2)]
    assert_public_batch_state(output, [decode_req], [prefill_req])


def test_metadata_from_consecutive_rounds_does_not_leak_between_batches():
    first_req = FakeRequest("p1", 2)
    second_req = FakeRequest("p2", 2)
    first_input = make_prefill_batch(first_req)
    first_input.log_new_tokens = 2
    first_input.log_cached_tokens = 0
    first_input.prompt_admissions = [("p1-a", 1), ("p1-b", 1)]
    second_input = make_prefill_batch(second_req)
    second_input.log_new_tokens = 0
    second_input.log_cached_tokens = 2
    second_input.prompt_admissions = []
    composer = MixedBatchComposer(
        prefill_manager=QueuePrefillManager(first_input, second_input),
        decode_manager=QueueDecodeManager(),
    )

    first = composer.schedule_next_batch(token_budget=2)
    second = composer.schedule_next_batch(token_budget=2)

    assert first is not None
    assert second is not None
    assert first.log_new_tokens == 2
    assert first.log_cached_tokens == 0
    assert first.prompt_admissions == [("p1-a", 1), ("p1-b", 1)]
    assert second.log_new_tokens == 0
    assert second.log_cached_tokens == 2
    assert second.prompt_admissions == []


def test_chunk_can_resume_after_an_intermediate_prefill_admission_none():
    decode_reqs = [FakeRequest(f"d{index}", 1) for index in range(3)]
    first_chunk = FakeRequest("shared-prompt", 3)
    second_chunk = FakeRequest("shared-prompt", 3)
    prefill_manager = QueuePrefillManager(
        make_prefill_batch(first_chunk),
        None,
        make_prefill_batch(second_chunk),
    )
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=QueueDecodeManager(
            *(make_decode_batch(req) for req in decode_reqs)
        ),
    )

    first = composer.schedule_next_batch(token_budget=4)
    middle = composer.schedule_next_batch(token_budget=4)
    last = composer.schedule_next_batch(token_budget=4)

    assert first is not None
    assert middle is not None
    assert last is not None
    assert prefill_manager.budgets == [3, 3, 3]
    assert_public_batch_state(first, [decode_reqs[0]], [first_chunk])
    assert_public_batch_state(middle, [decode_reqs[1]], [])
    assert_public_batch_state(last, [decode_reqs[2]], [second_chunk])


def test_each_forward_restarts_from_the_full_query_token_budget():
    first_decode = FakeRequest("d1", 4)
    second_decode = FakeRequest("d2", 1)
    first_prefill = FakeRequest("p1", 1)
    second_prefill = FakeRequest("p2", 4)
    prefill_manager = QueuePrefillManager(
        make_prefill_batch(first_prefill),
        make_prefill_batch(second_prefill),
    )
    composer = MixedBatchComposer(
        prefill_manager=prefill_manager,
        decode_manager=QueueDecodeManager(
            make_decode_batch(first_decode),
            make_decode_batch(second_decode),
        ),
    )

    first = composer.schedule_next_batch(token_budget=5)
    second = composer.schedule_next_batch(token_budget=5)

    assert first is not None
    assert second is not None
    assert prefill_manager.budgets == [1, 4]
    assert_public_batch_state(first, [first_decode], [first_prefill])
    assert_public_batch_state(second, [second_decode], [second_prefill])
