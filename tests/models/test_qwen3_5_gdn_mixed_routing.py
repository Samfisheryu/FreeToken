from types import MethodType, SimpleNamespace

import torch

import freetoken.models.qwen3_5_moe.gdn as gdn_module
from freetoken.core import Batch
from freetoken.models.qwen3_5_moe.gdn import Qwen3_5GatedDeltaNet


class _Projection:
    def forward(self, hidden_states):
        return hidden_states.new_zeros((hidden_states.shape[0], 7))


class _Norm:
    def forward(self, core_out, _z):
        return core_out


class _Identity:
    def forward(self, value):
        return value


def test_mixed_gdn_routes_and_merges_decode_first(monkeypatch):
    op = object.__new__(Qwen3_5GatedDeltaNet)
    op._fp8 = False
    op.in_proj = _Projection()
    op._in_proj_split = [3, 2, 1, 1]
    op.conv_dim = 3
    op.value_dim = 2
    op.num_v_heads = 1
    op.head_v_dim = 2
    op.norm = _Norm()
    op.out_proj = _Identity()
    op.layer_id = 0

    calls = []

    def run_decode(self, conv_in, a, b, pool, li, fla, dtype):
        calls.append(("decode", conv_in.shape[0], fla))
        return conv_in.new_ones((conv_in.shape[0], 1, 2))

    def run_prefill(self, conv_in, a, b, pool, li, fla, dtype):
        calls.append(("prefill", conv_in.shape[0], fla))
        return conv_in.new_full((conv_in.shape[0], 1, 2), 2)

    op._run_decode = MethodType(run_decode, op)
    op._run_prefill = MethodType(run_prefill, op)

    batch = Batch(reqs=[object()] * 5, decode_size=2)
    decode_metadata = object()
    prefill_metadata = object()
    batch.fla_metadata = SimpleNamespace(
        decode=decode_metadata,
        prefill=prefill_metadata,
    )
    pool = SimpleNamespace(local_index=lambda _layer_id: 0)
    monkeypatch.setattr(
        gdn_module,
        "get_global_ctx",
        lambda: SimpleNamespace(batch=batch, linear_state_pool=pool),
    )

    output = op.forward(torch.zeros(5, 4))

    assert calls == [
        ("decode", 2, decode_metadata),
        ("prefill", 3, prefill_metadata),
    ]
    torch.testing.assert_close(output[:2], torch.ones(2, 2))
    torch.testing.assert_close(output[2:], torch.full((3, 2), 2.0))
