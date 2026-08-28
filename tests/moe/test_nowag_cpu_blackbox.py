"""Black-box tests for the public NoWAG CPU MoE and FP8 contracts."""

import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="NoWAG decode returns CUDA tensors"
)

_BATCH_VARIANTS = ((1, 0), (2, 1), (3, 0), (4, 1), (5, 0), (7, 1), (8, 0))
_COMPACT_RULES = (
    pytest.param("qwen3_5_moe", 14, 8, 8, None, id="qwen35-k8-d6-tail"),
    pytest.param("deepseek_v4", 128, 128, 6, 0.75, id="dsv4-k6-d6-tail"),
)
_HYBRID_RULES = (
    pytest.param("qwen3_5_moe", 128, 128, 8, None, id="qwen35-k8-128"),
    pytest.param("deepseek_v4", 128, 128, 6, 0.75, id="dsv4-k6-128"),
)
_REAL_RULES = (
    pytest.param("qwen3_5_moe", 2048, 512, 8, None, id="qwen35-real-shape"),
    pytest.param("deepseek_v4", 4096, 2048, 6, 0.75, id="dsv4-real-shape"),
)


def _pack_12bit(logical_ids: torch.Tensor) -> torch.Tensor:
    """Pack public [expert, output, D6-group] IDs as [E, uint32-word, N]."""
    experts, outputs, groups = logical_ids.shape
    word_count = (groups * 12 + 31) // 32
    bit = torch.arange(groups, dtype=torch.int64) * 12
    word = bit // 32
    shift = bit % 32
    spill = shift + 12 > 32
    words = torch.zeros((experts, outputs, word_count), dtype=torch.int64)

    # One vectorized output-row pack per expert keeps real-shape fixtures practical.
    for expert in range(experts):
        ids = logical_ids[expert].to(torch.int64)
        words[expert].scatter_add_(
            1,
            word.expand(outputs, -1),
            (ids << shift) & 0xFFFFFFFF,
        )
        if bool(spill.any()):
            words[expert].scatter_add_(
                1,
                (word[spill] + 1).expand(outputs, -1),
                ids[:, spill] >> (32 - shift[spill]),
            )

    words = torch.where(words >= 2**31, words - 2**32, words)
    return words.to(torch.int32).permute(0, 2, 1).contiguous()


def _logical_ids(
    experts: int, outputs: int, groups: int, seed: int, variant: int
) -> torch.Tensor:
    expert = torch.arange(experts, dtype=torch.int64)[:, None, None]
    output = torch.arange(outputs, dtype=torch.int64)[None, :, None]
    group = torch.arange(groups, dtype=torch.int64)[None, None, :]
    ids = (
        seed
        + (719 + 18 * variant) * expert
        + (43 + 4 * variant) * output
        + (997 - 10 * variant) * group
        + (3 + 2 * variant) * output * group
    ) % 4096
    if groups >= 3:
        # Group 2 begins at bit 24; 0xABC therefore straddles uint32 words.
        ids[0, 0, :3] = torch.tensor([0x123, 0x456, 0xABC])
    if groups >= 4:
        ids[0, 0, 3] = 0xDEF
    return ids.to(torch.int32).contiguous()


def _norm(experts: int, width: int, phase: float, variant: int) -> torch.Tensor:
    position = torch.arange(experts * width, dtype=torch.float32).reshape(experts, width)
    values = 0.85 + (0.17 + 0.02 * variant) * torch.sin(
        (0.113 + 0.006 * variant) * position + phase
    )
    values += 0.06 * torch.cos(0.037 * position - phase - 0.2 * variant)
    return values.to(torch.bfloat16).contiguous()


def _make_case(
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    variant: int,
    *,
    experts: int | None = None,
):
    experts = experts or max(top_k, 8)
    row = torch.arange(4096, dtype=torch.int64)[:, None]
    lane = torch.arange(6, dtype=torch.int64)[None, :]
    codebook_values = (
        (37 + 2 * variant) * row
        + (613 - 8 * variant) * lane
        + (3 + 2 * variant) * row * lane
    ) % 4093 - 2046
    codebook = (codebook_values.float() / 4096).to(torch.bfloat16).contiguous()

    logical = {
        "gate": _logical_ids(
            experts, intermediate_size, (hidden_size + 5) // 6, 0x011, variant
        ),
        "up": _logical_ids(
            experts, intermediate_size, (hidden_size + 5) // 6, 0x5A3, variant
        ),
        "down": _logical_ids(
            experts, hidden_size, (intermediate_size + 5) // 6, 0xA71, variant
        ),
    }
    bank_sources = {
        "gate_assignments": [_pack_12bit(logical["gate"])],
        "up_assignments": [_pack_12bit(logical["up"])],
        "down_assignments": [_pack_12bit(logical["down"])],
        "gate_input_norm": [_norm(experts, hidden_size, 0.1, variant)],
        "up_input_norm": [_norm(experts, hidden_size, 0.7, variant)],
        "down_input_norm": [_norm(experts, intermediate_size, 1.3, variant)],
        "gate_output_norm": [_norm(experts, intermediate_size, 1.9, variant)],
        "up_output_norm": [_norm(experts, intermediate_size, 2.5, variant)],
        "down_output_norm": [_norm(experts, hidden_size, 3.1, variant)],
    }
    cache = SimpleNamespace(
        quant_format="nowag",
        num_layers=1,
        num_experts=experts,
        bank_sources=bank_sources,
        host_codebook=codebook,
    )
    return SimpleNamespace(
        cache=cache,
        logical=logical,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        variant=variant,
    )


def _fp8_roundtrip(
    values: torch.Tensor, *, output: torch.Tensor | None = None
) -> torch.Tensor:
    from freetoken.kernel.triton.dsv4.fp8_linear import act_quant_fp8_roundtrip

    return act_quant_fp8_roundtrip(values.contiguous(), block=128, output=output)


def _project(case, name, expert, inputs, codebook, device_banks):
    ids = case.logical[name][expert].to(device=inputs.device, dtype=torch.long)
    # A partial final D6 group contributes only its leading K % 6 weights.
    dense_weight = codebook[ids].reshape(ids.shape[0], -1)[:, : inputs.shape[-1]]
    input_norm = device_banks[f"{name}_input_norm"][expert]
    output_norm = device_banks[f"{name}_output_norm"][expert]

    normalized = (inputs.float() * input_norm.float()).to(torch.bfloat16)
    products = normalized.float()[:, None, :] * dense_weight.float()[None, :, :]
    accumulated = products.sum(dim=-1, dtype=torch.float32)
    return (accumulated * output_norm.float()).to(torch.bfloat16)


def _reference(case, hidden, topk_weights, topk_ids, model_type, swiglu_limit=None):
    codebook = case.cache.host_codebook.to(device=hidden.device)
    device_banks = {
        key: layers[0].to(device=hidden.device)
        for key, layers in case.cache.bank_sources.items()
        if key.endswith("_norm")
    }
    prepared_hidden = (
        _fp8_roundtrip(hidden) if model_type == "deepseek_v4" else hidden
    )
    result = torch.zeros(
        (hidden.shape[0], case.hidden_size), dtype=torch.float32, device=hidden.device
    )

    # Group routes by expert. This is still a direct dense definition, while avoiding
    # rebuilding a real-size expert matrix once per routed token.
    for expert in range(case.cache.num_experts):
        positions = torch.nonzero(
            (topk_ids == expert) & (topk_weights != 0), as_tuple=False
        )
        if positions.numel() == 0:
            continue
        tokens = positions[:, 0]
        inputs = prepared_hidden.index_select(0, tokens)
        gate = _project(case, "gate", expert, inputs, codebook, device_banks)
        up = _project(case, "up", expert, inputs, codebook, device_banks)
        if model_type == "deepseek_v4":
            gate = gate.float().clamp(max=swiglu_limit)
            up = up.float().clamp(min=-swiglu_limit, max=swiglu_limit)

        middle = (F.silu(gate.float()) * up.float()).to(torch.bfloat16)
        if model_type == "deepseek_v4":
            middle = _fp8_roundtrip(middle)
        down = _project(case, "down", expert, middle, codebook, device_banks)
        route_weights = topk_weights[positions[:, 0], positions[:, 1]].float()
        result.index_add_(0, tokens, down.float() * route_weights[:, None])
    return result.to(torch.bfloat16)


def _executor(
    case,
    *,
    model_type,
    batch_size=8,
    apply_router_weight_on_input=False,
    limit=None,
):
    from freetoken.moe.cpu_executor import CpuMoeExecutor

    kwargs = dict(
        top_k=case.top_k,
        activation="silu",
        apply_router_weight_on_input=apply_router_weight_on_input,
        num_threads=4,
        max_tokens=max(8, batch_size),
        device=torch.device("cuda"),
        swiglu_alpha=1.0,
        nowag_model_type=model_type,
    )
    if limit is not None:
        kwargs["swiglu_limit"] = limit
    return CpuMoeExecutor(case.cache, **kwargs)


def _decode_inputs(case, batch_size: int):
    position = torch.arange(
        batch_size * case.hidden_size, dtype=torch.float32
    ).reshape(batch_size, case.hidden_size)
    hidden = 0.9 * torch.sin((0.131 + 0.004 * case.variant) * position)
    hidden += 0.35 * torch.cos(0.037 * position + 0.2 * case.variant)
    hidden += 0.04 * (position.remainder(7 + case.variant) - 3)

    token = torch.arange(batch_size, dtype=torch.int64)[:, None]
    route = torch.arange(case.top_k, dtype=torch.int64)[None, :]
    topk_ids = (3 * token + route + case.variant) % case.cache.num_experts
    weights = 0.2 + ((5 * token + 7 * route + case.variant) % 13).float() / 13
    weights[:, 1] = 0.0
    weights /= weights.sum(dim=1, keepdim=True)
    # A nonzero padded route verifies that -1 is skipped rather than treated as a slot.
    topk_ids[-1, -1] = -1
    return (
        hidden.to(device="cuda", dtype=torch.bfloat16).contiguous(),
        weights.to(device="cuda").contiguous(),
        topk_ids.to(device="cuda", dtype=torch.int32).contiguous(),
    )


def _assert_decode_matches(actual, expected, *, real_shape=False):
    assert actual.is_cuda
    assert actual.is_contiguous()
    assert actual.dtype == torch.bfloat16
    assert actual.shape == expected.shape
    # Independent CPU/GPU FP32 reductions may order additions differently. The
    # wider bound covers real H/I reductions while remaining only a few BF16 ulps.
    rtol, atol = ((0.035, 0.06) if real_shape else (0.02, 0.02))
    torch.testing.assert_close(actual.float(), expected.float(), rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "batch_size,variant",
    _BATCH_VARIANTS,
    ids=lambda value: f"v{value}" if isinstance(value, int) else None,
)
@pytest.mark.parametrize("model_type,h_size,i_size,top_k,limit", _COMPACT_RULES)
def test_compact_cpu_decode_matrix(
    model_type, h_size, i_size, top_k, limit, batch_size, variant
):
    case = _make_case(h_size, i_size, top_k, variant)
    hidden, topk_weights, topk_ids = _decode_inputs(case, batch_size)
    expected = _reference(
        case, hidden, topk_weights, topk_ids, model_type, swiglu_limit=limit
    )

    executor = _executor(
        case, model_type=model_type, batch_size=batch_size, limit=limit
    )
    actual = executor.decode(0, hidden, topk_weights, topk_ids)
    torch.cuda.synchronize()

    _assert_decode_matches(actual, expected)


@pytest.mark.parametrize("model_type,h_size,i_size,top_k,limit", _COMPACT_RULES)
def test_decode_submit_sync_matches_dense_reference(
    model_type, h_size, i_size, top_k, limit
):
    case = _make_case(h_size, i_size, top_k, variant=1)
    hidden, topk_weights, topk_ids = _decode_inputs(case, batch_size=3)
    expected = _reference(
        case, hidden, topk_weights, topk_ids, model_type, swiglu_limit=limit
    )

    executor = _executor(case, model_type=model_type, batch_size=3, limit=limit)
    pending = executor.decode_submit(0, hidden, topk_weights, topk_ids)
    actual = executor.decode_sync(pending)
    torch.cuda.synchronize()

    _assert_decode_matches(actual, expected)


@pytest.mark.skipif(
    os.environ.get("FREETOKEN_RUN_LARGE_NOWAG") != "1",
    reason="set FREETOKEN_RUN_LARGE_NOWAG=1 for synthetic real-shape coverage",
)
@pytest.mark.parametrize("model_type,h_size,i_size,top_k,limit", _REAL_RULES)
def test_real_shape_cpu_decode(
    model_type, h_size, i_size, top_k, limit
):
    case = _make_case(h_size, i_size, top_k, variant=1, experts=top_k)
    hidden, topk_weights, topk_ids = _decode_inputs(case, batch_size=1)
    expected = _reference(
        case, hidden, topk_weights, topk_ids, model_type, swiglu_limit=limit
    )

    executor = _executor(case, model_type=model_type, batch_size=1, limit=limit)
    actual = executor.decode(0, hidden, topk_weights, topk_ids)
    torch.cuda.synchronize()

    _assert_decode_matches(actual, expected, real_shape=True)


@pytest.mark.parametrize("batch_size", (1, 3, 8))
def test_fp8_roundtrip_fresh_and_preallocated_output(batch_size):
    position = torch.arange(
        batch_size * 256, dtype=torch.float32, device="cuda"
    ).reshape(batch_size, 256)
    values = (
        1.1 * torch.sin(0.071 * position)
        + 0.4 * torch.cos(0.023 * position)
        + 0.03 * (position.remainder(11) - 5)
    ).to(torch.bfloat16).contiguous()
    original = values.clone()

    fresh = _fp8_roundtrip(values)
    output = torch.empty_like(values)
    reused = _fp8_roundtrip(values, output=output)
    torch.cuda.synchronize()

    assert fresh.is_contiguous()
    assert fresh.data_ptr() != values.data_ptr()
    assert reused is output
    assert reused.data_ptr() == output.data_ptr()
    assert torch.equal(reused, fresh)
    assert torch.equal(values, original)


@pytest.mark.parametrize("bad_output", ("shape", "dtype", "device"))
def test_fp8_roundtrip_rejects_invalid_output(bad_output):
    values = torch.linspace(-1.25, 1.5, 256, device="cuda").reshape(2, 128)
    values = values.to(torch.bfloat16).contiguous()
    if bad_output == "shape":
        output = torch.empty((1, 256), dtype=values.dtype, device=values.device)
    elif bad_output == "dtype":
        output = torch.empty_like(values, dtype=torch.float32)
    else:
        output = torch.empty_like(values, device="cpu")

    with pytest.raises((ValueError, RuntimeError, TypeError)):
        _fp8_roundtrip(values, output=output)
        torch.cuda.synchronize()


def _assert_synchronous_rejection(action):
    with pytest.raises(Exception) as captured:
        action()
    assert str(captured.value).strip()


def test_deepseek_v4_requires_swiglu_limit():
    case = _make_case(128, 128, top_k=6, variant=0)
    _assert_synchronous_rejection(lambda: _executor(case, model_type="deepseek_v4"))


def test_nowag_rejects_router_weight_on_input():
    case = _make_case(14, 8, top_k=8, variant=0)
    _assert_synchronous_rejection(
        lambda: _executor(
            case,
            model_type="qwen3_5_moe",
            apply_router_weight_on_input=True,
        )
    )


def test_nowag_rejects_unknown_model_rule():
    case = _make_case(14, 8, top_k=8, variant=0)
    _assert_synchronous_rejection(
        lambda: _executor(case, model_type="not_a_nowag_model")
    )


def test_cpu_executor_rejects_non_nowag_cache_format():
    case = _make_case(14, 8, top_k=8, variant=0)
    case.cache.quant_format = "not_nowag"
    _assert_synchronous_rejection(
        lambda: _executor(case, model_type="qwen3_5_moe")
    )


def _make_hybrid_cache(case, model_type, limit, max_fetch):
    from freetoken.moe.offload_cache import OffloadMoeCache

    case.cache.bank_sources = {
        name: [tensor.pin_memory() for tensor in layers]
        for name, layers in case.cache.bank_sources.items()
    }
    case.cache.host_codebook = case.cache.host_codebook.pin_memory()
    executor = _executor(case, model_type=model_type, limit=limit)
    cache = OffloadMoeCache(
        num_layers=1,
        num_experts=case.cache.num_experts,
        cache_size=case.cache.num_experts,
        device=torch.device("cuda"),
        quant_format="nowag",
        decode_target="hybrid",
        hybrid_max_fetch=max_fetch,
        hybrid_fetch_fraction=0.0,
    )
    cache.set_bank_sources(case.cache.bank_sources)
    cache.set_codebook(case.cache.host_codebook)
    cache.set_alphas(None, None)
    cache.set_cpu_executor(executor)
    return cache, executor


def _warm_hybrid(cache, expert_ids):
    warm_ids = expert_ids.clone()
    cache.ensure_experts_hybrid(0, warm_ids)
    cache.copy_missing()
    torch.cuda.synchronize()


@pytest.mark.parametrize("batch_size", (3, 8))
@pytest.mark.parametrize("scenario", ("full_hit", "partial_miss", "full_miss"))
@pytest.mark.parametrize("model_type,h_size,i_size,top_k,limit", _HYBRID_RULES)
def test_hybrid_cache_split_and_public_stats(
    model_type, h_size, i_size, top_k, limit, scenario, batch_size
):
    case = _make_case(h_size, i_size, top_k, variant=batch_size % 2)
    original_ids = torch.arange(top_k, dtype=torch.int32, device="cuda")
    original_ids = original_ids.expand(batch_size, -1).contiguous()

    if scenario == "full_hit":
        max_fetch = top_k
    elif scenario == "partial_miss":
        max_fetch = 1
    else:
        max_fetch = 0
    cache, executor = _make_hybrid_cache(case, model_type, limit, max_fetch)

    if scenario == "full_hit":
        _warm_hybrid(cache, original_ids)
    elif scenario == "partial_miss":
        _warm_hybrid(cache, original_ids[:, :1])
    cache.reset_stats()

    mapped_ids = original_ids.clone()
    cache.ensure_experts_hybrid(0, mapped_ids)
    cache.record_decode_stats_hybrid(0)
    cache.copy_missing()
    torch.cuda.synchronize()

    if scenario == "full_hit":
        missing, fetched, cpu = 0, 0, 0
        resident = top_k
    elif scenario == "partial_miss":
        missing, fetched, cpu = top_k - 1, 1, top_k - 2
        resident = 2
    else:
        missing, fetched, cpu = top_k, 0, top_k
        resident = 0

    resident_experts = torch.unique(original_ids[mapped_ids >= 0]).numel()
    cpu_experts = torch.unique(original_ids[mapped_ids == -1]).numel()
    assert bool((mapped_ids >= -1).all())
    assert resident_experts == resident
    assert cpu_experts == cpu

    stats = cache.decode_miss_stats()
    assert stats["layer_calls"] == 1
    assert stats["active_per_layer"] == pytest.approx(top_k)
    assert stats["missing_per_layer"] == pytest.approx(missing)
    assert stats["fetched_per_layer"] == pytest.approx(fetched)
    assert stats["cpu_per_layer"] == pytest.approx(cpu)
    assert stats["miss_rate"] == pytest.approx(missing / top_k)
    expected_fetch_rate = fetched / missing if missing else 0.0
    assert stats["fetch_rate"] == pytest.approx(expected_fetch_rate)

    layer_stats = cache.decode_miss_stats_per_layer()["per_layer"]
    assert len(layer_stats) == 1
    layer = layer_stats[0]
    assert layer["layer"] == 0
    assert layer["steps"] == 1
    assert layer["active_per_step"] == pytest.approx(top_k)
    assert layer["missing_per_step"] == pytest.approx(missing)
    assert layer["miss_rate"] == pytest.approx(missing / top_k)
    assert layer["fetched_per_step"] == pytest.approx(fetched)


def test_hybrid_cache_rejects_unknown_quant_format():
    from freetoken.moe.offload_cache import OffloadMoeCache

    _assert_synchronous_rejection(
        lambda: OffloadMoeCache(
            num_layers=1,
            num_experts=8,
            cache_size=8,
            device=torch.device("cuda"),
            quant_format="not_a_quant_format",
            decode_target="hybrid",
        )
    )
