from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from freetoken.moe.offload_cache import OffloadMoeCache


BANK_NAMES = ("gate_up", "down")
PREFILL_STAT_KEYS = {
    "layer_calls",
    "active_per_layer",
    "missing_per_layer",
    "miss_rate",
    "fetched_per_layer",
    "cpu_per_layer",
    "fetch_rate",
    "prefill_hit_rows",
    "prefill_rows",
    "prefill_layer_prepares",
    "prefill_h2d_bytes",
}


def _make_cache(
    *,
    num_layers: int,
    num_experts: int,
    cache_size: int,
    prefill_group_size: int,
    device: torch.device | None = None,
) -> OffloadMoeCache:
    return OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=device if device is not None else torch.device("cpu"),
        cache_policy="lru",
        prefill_overlap=True,
        separate_prefill_buffer=False,
        prefill_group_size=prefill_group_size,
        quant_format="bf16",
    )


def _make_sources(
    num_layers: int, num_experts: int, *, pinned: bool = False
) -> dict[str, list[torch.Tensor]]:
    gate_up: list[torch.Tensor] = []
    down: list[torch.Tensor] = []
    for layer_id in range(num_layers):
        logical_base = layer_id * num_experts
        layer_gate_up = torch.stack(
            [
                torch.full(
                    (2, 4),
                    logical_base + expert_id,
                    dtype=torch.bfloat16,
                )
                for expert_id in range(num_experts)
            ]
        ).contiguous()
        layer_down = torch.stack(
            [
                torch.full(
                    (4, 2),
                    128 + logical_base + expert_id,
                    dtype=torch.bfloat16,
                )
                for expert_id in range(num_experts)
            ]
        ).contiguous()
        if pinned:
            layer_gate_up = layer_gate_up.pin_memory()
            layer_down = layer_down.pin_memory()
        gate_up.append(layer_gate_up)
        down.append(layer_down)
    return {"gate_up": gate_up, "down": down}


def _source_bytes_for_layers(
    sources: dict[str, list[torch.Tensor]], layer_ids: range
) -> int:
    return sum(
        sources[bank_name][layer_id].numel()
        * sources[bank_name][layer_id].element_size()
        for bank_name in BANK_NAMES
        for layer_id in layer_ids
    )


def _assert_canonical_partition(cache: OffloadMoeCache, cache_size: int) -> None:
    assert cache.decode_cache_size == cache_size
    assert cache.prefill_buffer_slots == 0
    assert cache.cache_partition() == {
        "total_slots": cache_size,
        "decode_slots": cache_size,
        "prefill_buffer_slots": 0,
    }


def _wait_prefill_layers(
    cache: OffloadMoeCache, start: int, end: int
) -> None:
    for layer_id in range(start, end):
        resident_banks = cache.wait_prefill_layer(layer_id)
        assert isinstance(resident_banks, tuple)


def _cuda_device_event_count(profile_result: object) -> int:
    return sum(
        event.device_type == torch.autograd.DeviceType.CUDA
        for event in profile_result.events()
    )


def _top_k_ids() -> torch.Tensor:
    return (
        torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.int32)
        .repeat(16, 1)
        .contiguous()
    )


def _group_slots(
    cache: OffloadMoeCache,
    *,
    num_experts: int,
    start: int,
    end: int,
) -> dict[tuple[int, int], int]:
    slots: dict[tuple[int, int], int] = {}
    for layer_id in range(start, end):
        for expert_id in range(num_experts):
            slot = int(cache.slot_for_id[layer_id, expert_id])
            logical_id = layer_id * num_experts + expert_id
            assert 0 <= slot < cache.decode_cache_size
            assert int(cache.id_of_slot[slot]) == logical_id
            slots[(layer_id, expert_id)] = slot

    assert len(set(slots.values())) == (end - start) * num_experts
    return slots


def _map_prefill_experts(
    cache: OffloadMoeCache, layer_id: int, expert_ids: torch.Tensor
) -> None:
    """Exercise the public in-place raw expert-ID to physical-slot mapping."""
    original_storage = expert_ids.data_ptr()
    result = cache.map_prefill_experts(layer_id, expert_ids)
    assert result is None
    assert expert_ids.data_ptr() == original_storage


def _assert_bank_rows(
    cache: OffloadMoeCache,
    sources: dict[str, list[torch.Tensor]],
    layer_id: int,
    source_ids: torch.Tensor,
    physical_slots: torch.Tensor,
) -> None:
    for bank_name in BANK_NAMES:
        bank = cache.bank_caches[bank_name]
        row_shape = sources[bank_name][layer_id].shape[1:]
        bank_indices = physical_slots.reshape(-1).to(
            device=bank.device, dtype=torch.long
        )
        source_indices = source_ids.reshape(-1).to(device="cpu", dtype=torch.long)
        actual = bank.index_select(0, bank_indices).reshape(
            *source_ids.shape, *row_shape
        )
        expected = sources[bank_name][layer_id].index_select(
            0, source_indices
        ).reshape(*source_ids.shape, *row_shape)
        torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize(
    ("num_layers", "num_experts", "cache_size", "requested", "expected"),
    [
        (5, 2, 10, 3, 3),  # requested group size is limiting
        (3, 2, 8, 9, 3),  # layer count is limiting
        (6, 3, 7, 9, 2),  # pool capacity is limiting
    ],
)
def test_joint_geometry_uses_the_whole_canonical_pool_and_caps_group_size(
    num_layers: int,
    num_experts: int,
    cache_size: int,
    requested: int,
    expected: int,
) -> None:
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        prefill_group_size=requested,
    )

    assert cache.effective_prefill_group_size == expected
    _assert_canonical_partition(cache, cache_size)


def test_joint_runs_at_one_layer_capacity_after_oversized_request_is_capped() -> None:
    num_layers = 4
    num_experts = 3
    cache_size = num_experts
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        prefill_group_size=99,
    )
    sources = _make_sources(num_layers, num_experts)
    cache.set_bank_sources(sources)

    assert cache.effective_prefill_group_size == 1
    cache.begin_resident_prefill_group(2, 3)

    assert len(
        _group_slots(cache, num_experts=num_experts, start=2, end=3)
    ) == num_experts
    cache.end_prefill_group()


def test_group_admission_is_complete_bijective_and_maps_directly_into_banks() -> None:
    num_layers = 4
    num_experts = 2
    cache_size = 5
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        prefill_group_size=2,
    )
    sources = _make_sources(num_layers, num_experts)
    cache.set_bank_sources(sources)

    cache.begin_resident_prefill_group(1, 3)
    slots = _group_slots(cache, num_experts=num_experts, start=1, end=3)

    assert all(bank.shape[0] == cache_size for bank in cache.bank_caches.values())
    for layer_id in range(1, 3):
        raw_expert_ids = torch.tensor([1, 0, 1], dtype=torch.int32)
        expected_slots = torch.tensor(
            [slots[(layer_id, int(expert_id))] for expert_id in raw_expert_ids],
            dtype=torch.int32,
        )
        source_ids = raw_expert_ids.clone()

        _map_prefill_experts(cache, layer_id, raw_expert_ids)

        assert torch.equal(raw_expert_ids, expected_slots)
        _assert_bank_rows(cache, sources, layer_id, source_ids, raw_expert_ids)
    cache.end_prefill_group()


def test_end_only_unprotects_and_repeat_or_overlap_fetches_only_misses() -> None:
    num_layers = 4
    num_experts = 2
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=6,
        prefill_group_size=2,
    )
    sources = _make_sources(num_layers, num_experts)
    cache.set_bank_sources(sources)

    before_first = cache.prefill_h2d_bytes_total()
    cache.begin_resident_prefill_group(0, 2)
    after_first = cache.prefill_h2d_bytes_total()
    first_slots = cache.slot_for_id.clone()
    assert after_first - before_first == _source_bytes_for_layers(
        sources, range(0, 2)
    )

    cache.end_prefill_group()
    assert torch.equal(cache.slot_for_id, first_slots)

    cache.begin_resident_prefill_group(0, 2)
    assert cache.prefill_h2d_bytes_total() == after_first
    assert torch.equal(cache.slot_for_id, first_slots)
    cache.end_prefill_group()

    before_overlap = cache.prefill_h2d_bytes_total()
    cache.begin_resident_prefill_group(1, 3)
    assert (
        cache.prefill_h2d_bytes_total() - before_overlap
        == _source_bytes_for_layers(sources, range(2, 3))
    )
    _group_slots(cache, num_experts=num_experts, start=1, end=3)
    cache.end_prefill_group()


def test_group_admission_protects_old_working_set_hits_from_lru_eviction() -> None:
    num_layers = 4
    num_experts = 2
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=4,
        prefill_group_size=2,
    )
    cache.set_bank_sources(_make_sources(num_layers, num_experts))

    cache.begin_resident_prefill_group(2, 3)
    cache.end_prefill_group()

    cache.begin_resident_prefill_group(3, 4)
    cache.end_prefill_group()
    old_hit_slots = {
        expert_id: int(cache.slot_for_id[2, expert_id])
        for expert_id in range(num_experts)
    }

    # Layer 2 is older than the subsequently admitted layer 3 under ordinary
    # LRU. The next group needs layer 1 misses while retaining layer 2 hits.
    cache.begin_resident_prefill_group(1, 3)

    admitted = _group_slots(cache, num_experts=num_experts, start=1, end=3)
    assert {
        expert_id: admitted[(2, expert_id)] for expert_id in range(num_experts)
    } == old_hit_slots
    assert {int(logical_id) for logical_id in cache.id_of_slot.tolist()} == {
        layer_id * num_experts + expert_id
        for layer_id in range(1, 3)
        for expert_id in range(num_experts)
    }
    cache.end_prefill_group()


def test_resident_layer_alphas_follow_rearranged_canonical_slots() -> None:
    num_layers = 4
    num_experts = 2
    cache_size = 5
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        prefill_group_size=2,
    )
    cache.set_bank_sources(_make_sources(num_layers, num_experts))
    gate_up_alpha = torch.arange(
        num_layers * num_experts, dtype=torch.float32
    ).add_(0.25)
    down_alpha = torch.arange(
        num_layers * num_experts, dtype=torch.float32
    ).add_(64.5)
    cache.set_alphas(gate_up_alpha, down_alpha)

    cache.begin_resident_prefill_group(0, 2)
    cache.end_prefill_group()
    cache.begin_resident_prefill_group(1, 3)

    slots = _group_slots(cache, num_experts=num_experts, start=1, end=3)
    layer_two_slots = [
        slots[(2, expert_id)] for expert_id in range(num_experts)
    ]
    assert layer_two_slots != list(
        range(layer_two_slots[0], layer_two_slots[0] + num_experts)
    ), "resident alpha lookup must support non-contiguous or reordered slots"

    for layer_id in range(1, 3):
        resident_alphas = cache.alphas_for_resident_layer_slots(layer_id)
        assert resident_alphas is not None
        resident_gate_up, resident_down = resident_alphas
        assert resident_gate_up.shape == (cache_size,)
        assert resident_down.shape == (cache_size,)

        for expert_id in range(num_experts):
            physical_slot = slots[(layer_id, expert_id)]
            logical_id = layer_id * num_experts + expert_id
            torch.testing.assert_close(
                resident_gate_up[physical_slot], gate_up_alpha[logical_id]
            )
            torch.testing.assert_close(
                resident_down[physical_slot], down_alpha[logical_id]
            )
    cache.end_prefill_group()


def test_cpu_mapping_reports_invalid_ids_and_inactive_or_outside_layers() -> None:
    num_layers = 4
    num_experts = 2
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=5,
        prefill_group_size=2,
    )
    cache.set_bank_sources(_make_sources(num_layers, num_experts))

    with pytest.raises(
        RuntimeError, match="no joint resident expert group is active"
    ):
        cache.map_prefill_experts(
            0, torch.tensor([[1, 0]], dtype=torch.int32).contiguous()
        )

    cache.begin_resident_prefill_group(0, 2)
    with pytest.raises(ValueError, match=r"must be in \[0, 2\)"):
        cache.map_prefill_experts(
            0, torch.tensor([[0, -1], [2, 1]], dtype=torch.int32).contiguous()
        )
    with pytest.raises(RuntimeError, match="outside resident group"):
        cache.map_prefill_experts(
            2, torch.tensor([[1, 0]], dtype=torch.int32).contiguous()
        )
    cache.end_prefill_group()


def test_joint_prefill_stats_are_exact_across_cold_repeat_overlap_and_reset() -> None:
    num_layers = 4
    num_experts = 2
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=6,
        prefill_group_size=2,
    )
    sources = _make_sources(num_layers, num_experts)
    cache.set_bank_sources(sources)

    def assert_stats(
        *, rows: int, hits: int, prepares: int, h2d_bytes: int
    ) -> None:
        stats = cache.decode_miss_stats()
        assert set(stats) == PREFILL_STAT_KEYS
        assert stats["layer_calls"] == 0
        for name in (
            "active_per_layer",
            "missing_per_layer",
            "miss_rate",
            "fetched_per_layer",
            "cpu_per_layer",
            "fetch_rate",
        ):
            assert stats[name] == 0.0
        assert stats["prefill_rows"] == rows
        assert stats["prefill_hit_rows"] == hits
        assert stats["prefill_layer_prepares"] == prepares
        assert stats["prefill_h2d_bytes"] == h2d_bytes
        assert cache.prefill_h2d_bytes_total() == h2d_bytes

    assert_stats(rows=0, hits=0, prepares=0, h2d_bytes=0)

    cold_bytes = _source_bytes_for_layers(sources, range(0, 2))
    cache.begin_resident_prefill_group(0, 2)
    cache.end_prefill_group()
    assert_stats(rows=4, hits=0, prepares=2, h2d_bytes=cold_bytes)

    cache.begin_resident_prefill_group(0, 2)
    cache.end_prefill_group()
    assert_stats(rows=8, hits=4, prepares=4, h2d_bytes=cold_bytes)

    overlap_bytes = cold_bytes + _source_bytes_for_layers(sources, range(2, 3))
    cache.begin_resident_prefill_group(1, 3)
    cache.end_prefill_group()
    assert_stats(rows=12, hits=6, prepares=6, h2d_bytes=overlap_bytes)

    cache.reset_stats()
    assert_stats(rows=0, hits=0, prepares=0, h2d_bytes=0)


def test_ended_group_pages_rejoin_lru_without_losing_recent_locality() -> None:
    num_layers = 4
    num_experts = 2
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=6,
        prefill_group_size=2,
    )
    cache.set_bank_sources(_make_sources(num_layers, num_experts))

    cache.begin_resident_prefill_group(0, 2)
    cache.end_prefill_group()
    layer_zero_slots = {
        expert_id: int(cache.slot_for_id[0, expert_id])
        for expert_id in range(num_experts)
    }

    # A hit-only group refreshes layer 0 after release, making layer 1 older.
    cache.begin_resident_prefill_group(0, 1)
    cache.end_prefill_group()
    cache.begin_resident_prefill_group(2, 3)
    cache.end_prefill_group()

    cache.begin_resident_prefill_group(3, 4)
    _group_slots(cache, num_experts=num_experts, start=3, end=4)
    assert {
        expert_id: int(cache.slot_for_id[0, expert_id])
        for expert_id in range(num_experts)
    } == layer_zero_slots
    expected_resident_ids = {
        layer_id * num_experts + expert_id
        for layer_id in (0, 2, 3)
        for expert_id in range(num_experts)
    }
    assert {int(logical_id) for logical_id in cache.id_of_slot.tolist()} == (
        expected_resident_ids
    )

    mappings_before_end = cache.slot_for_id.clone()
    cache.end_prefill_group()
    assert torch.equal(cache.slot_for_id, mappings_before_end)


def test_cuda_joint_admission_returns_before_prior_compute_work_finishes() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for asynchronous admission behavior")

    device = torch.device("cuda", torch.cuda.current_device())
    num_layers = 4
    num_experts = 2
    cache_size = 5
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        prefill_group_size=2,
        device=device,
    )
    sources = _make_sources(num_layers, num_experts, pinned=True)
    cache.set_bank_sources(sources)

    compute_stream = torch.cuda.Stream(device=device)

    # Warm the public lifecycle, then reset its logical mapping state.
    with torch.cuda.stream(compute_stream):
        cache.begin_resident_prefill_group(0, 2)
        _wait_prefill_layers(cache, 0, 2)
        cache.end_prefill_group()
    torch.cuda.synchronize(device)
    cache.reset()
    torch.cuda.synchronize(device)

    blocker_done = torch.cuda.Event()
    clock_rate_khz = torch.cuda.get_device_properties(device).clock_rate
    half_second_cycles = max(100_000_000, int(clock_rate_khz * 1_000 * 0.5))

    with torch.cuda.stream(compute_stream):
        # Keep prior device work observably incomplete when begin returns.
        for _ in range(4):
            torch.cuda._sleep(half_second_cycles)
        blocker_done.record()
        cache.begin_resident_prefill_group(0, 2)

    assert not blocker_done.query(), (
        "begin_resident_prefill_group waited on earlier compute-stream work "
        "instead of returning after enqueue"
    )

    def consume_and_finish_group(
        start: int, end: int, *, check_mapping_event_limit: bool = False
    ) -> None:
        source_ids = _top_k_ids()
        with torch.cuda.stream(compute_stream):
            _wait_prefill_layers(cache, start, end)

            slots = _group_slots(
                cache,
                num_experts=num_experts,
                start=start,
                end=end,
            )
            for layer_id in range(start, end):
                expected_slots_cpu = torch.empty_like(source_ids)
                for expert_id in range(num_experts):
                    expected_slots_cpu[source_ids == expert_id] = slots[
                        (layer_id, expert_id)
                    ]
                routed_experts = source_ids.to(device)
                expected_slots = expected_slots_cpu.to(device)

                if check_mapping_event_limit and layer_id == start:
                    torch.cuda.synchronize(device)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as mapping_profile:
                        _map_prefill_experts(cache, layer_id, routed_experts)
                        torch.cuda.synchronize(device)
                    device_events = _cuda_device_event_count(mapping_profile)
                    assert 1 <= device_events <= 4, (
                        "hot mapping emitted "
                        f"{device_events} CUDA device events; expected 1..4"
                    )
                else:
                    _map_prefill_experts(cache, layer_id, routed_experts)

                assert routed_experts.is_contiguous()
                assert routed_experts.dtype == torch.int32
                assert routed_experts.shape == (64, 2)
                assert torch.equal(routed_experts, expected_slots)
                _assert_bank_rows(
                    cache, sources, layer_id, source_ids, routed_experts
                )

            cache.end_prefill_group()

        torch.cuda.synchronize(device)

    consume_and_finish_group(0, 2)
    first_slots = cache.slot_for_id.clone()

    with torch.cuda.stream(compute_stream):
        cache.begin_resident_prefill_group(0, 2)
    consume_and_finish_group(0, 2, check_mapping_event_limit=True)
    assert torch.equal(cache.slot_for_id, first_slots)

    with torch.cuda.stream(compute_stream):
        cache.begin_resident_prefill_group(1, 3)
    consume_and_finish_group(1, 3)


def test_cuda_fully_resident_joint_lifecycle_has_bounded_device_activity() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for lifecycle device-event profiling")

    device = torch.device("cuda", torch.cuda.current_device())
    num_layers = 4
    num_experts = 8
    cache_size = 24
    cache = _make_cache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        prefill_group_size=2,
        device=device,
    )
    sources = _make_sources(num_layers, num_experts, pinned=True)
    cache.set_bank_sources(sources)
    compute_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(compute_stream):
        cache.begin_resident_prefill_group(0, 2)
        _wait_prefill_layers(cache, 0, 2)
        cache.end_prefill_group()
    torch.cuda.synchronize(device)
    warm_slots = cache.slot_for_id.clone()
    torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as lifecycle_profile:
        with torch.cuda.stream(compute_stream):
            cache.begin_resident_prefill_group(0, 2)
            _wait_prefill_layers(cache, 0, 2)
            cache.end_prefill_group()
        torch.cuda.synchronize(device)

    device_events = _cuda_device_event_count(lifecycle_profile)
    assert 1 <= device_events <= 10, (
        "hot fully-resident lifecycle emitted "
        f"{device_events} CUDA device events; expected 1..10"
    )

    assert torch.equal(cache.slot_for_id, warm_slots)
    slots = _group_slots(
        cache,
        num_experts=num_experts,
        start=0,
        end=2,
    )
    source_ids = torch.arange(num_experts, dtype=torch.int32)
    for layer_id in range(2):
        physical_slots = torch.tensor(
            [slots[(layer_id, expert_id)] for expert_id in range(num_experts)],
            dtype=torch.int32,
        )
        _assert_bank_rows(cache, sources, layer_id, source_ids, physical_slots)


def test_zero_group_legacy_and_mixed_cache_geometry_remains_unchanged() -> None:
    # Legacy and mixed scheduling share this cache configuration; their
    # distinction is outside OffloadMoeCache.
    cache = _make_cache(
        num_layers=4,
        num_experts=2,
        cache_size=8,
        prefill_group_size=0,
    )

    assert cache.effective_prefill_group_size == 0
    _assert_canonical_partition(cache, 8)
