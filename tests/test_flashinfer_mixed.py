import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import freetoken.attention.fi as fi_module
from freetoken.attention.fi import FIMetadata, FlashInferBackend
from freetoken.core import Batch


class _Event:
    def __init__(self):
        self.records = 0
        self.synchronizes = 0

    def record(self):
        self.records += 1

    def synchronize(self):
        self.synchronizes += 1


class _Wrapper:
    def __init__(self, name, delta, order):
        self.name = name
        self.delta = delta
        self.order = order
        self.plans = []
        self.runs = []

    def plan(self, **kwargs):
        self.order.append(f"plan-{self.name}")
        self.plans.append(kwargs)

    def run(self, *, q, paged_kv_cache, out=None):
        self.order.append(f"run-{self.name}")
        self.runs.append((q.clone(), out))
        result = q + self.delta
        if out is None:
            return result
        out.copy_(result)
        return out


class _KVCache:
    def __init__(self, order):
        self.dtype = torch.float32
        self.order = order
        self.cache = torch.zeros(16, 1, 1, 2)
        self.stores = []

    def store_kv(self, k, v, out_loc, layer_id):
        self.order.append("store")
        self.stores.append((k, v, out_loc, layer_id))

    def k_cache(self, _layer_id):
        return self.cache

    def v_cache(self, _layer_id):
        return self.cache


def _req(*, extend_len, cached_len, table_idx):
    return SimpleNamespace(
        extend_len=extend_len,
        cached_len=cached_len,
        device_len=cached_len + extend_len,
        table_idx=table_idx,
    )


def _backend():
    order = []
    backend = object.__new__(FlashInferBackend)
    backend.device = torch.device("cpu")
    backend.config = SimpleNamespace(head_dim=2)
    backend.qo_head_local = 1
    backend.kv_head_local = 1
    backend.cached_ones_cpu = torch.tensor([], dtype=torch.int32)
    backend.decode_wrappers = _Wrapper("decode", 10, order)
    backend.prefill_wrapper = _Wrapper("prefill", 20, order)
    backend.kvcache = _KVCache(order)
    backend.last_event = _Event()
    backend.capture = None
    backend.capture_bs = []
    backend.graph_wrappers = {}
    return backend, order


def _prepare(monkeypatch, backend, batch):
    batch.padded_reqs = batch.reqs
    page_table = torch.arange(4 * 16, dtype=torch.int32).reshape(4, 16)
    monkeypatch.setattr(
        fi_module,
        "get_global_ctx",
        lambda: SimpleNamespace(page_table=page_table),
    )
    backend.prepare_metadata(batch)
    assert isinstance(batch.attn_metadata, FIMetadata)
    return batch.attn_metadata


def test_eager_wrappers_keep_separate_integer_workspaces(monkeypatch):
    class NativeWrapper:
        def __init__(self, float_workspace, *args, **kwargs):
            self.float_workspace = float_workspace
            self._int_workspace_buffer = object()

    flashinfer = ModuleType("flashinfer")
    flashinfer.BatchDecodeWithPagedKVCacheWrapper = NativeWrapper
    flashinfer.BatchPrefillWithPagedKVCacheWrapper = NativeWrapper
    monkeypatch.setitem(sys.modules, "flashinfer", flashinfer)
    monkeypatch.setattr(
        fi_module,
        "get_global_ctx",
        lambda: SimpleNamespace(kv_cache=SimpleNamespace(device=torch.device("cpu"))),
    )
    monkeypatch.setattr(fi_module, "get_tp_info", lambda: SimpleNamespace(size=1))
    real_empty = torch.empty
    monkeypatch.setattr(
        fi_module.torch,
        "empty",
        lambda *args, **kwargs: real_empty(1, dtype=torch.uint8),
    )
    monkeypatch.setattr(fi_module.torch.cuda, "Event", _Event)
    config = SimpleNamespace(num_qo_heads=8, num_kv_heads=2, head_dim=128)

    backend = FlashInferBackend(config)

    assert backend.prefill_wrapper.float_workspace is backend.float_workspace_buffer
    assert backend.decode_wrappers.float_workspace is backend.float_workspace_buffer
    assert (
        backend.prefill_wrapper._int_workspace_buffer
        is not backend.decode_wrappers._int_workspace_buffer
    )
    assert backend.int_workspace_buffer is backend.decode_wrappers._int_workspace_buffer


def test_mixed_metadata_partitions_and_rebases(monkeypatch):
    backend, _ = _backend()
    batch = Batch(
        reqs=[
            _req(extend_len=1, cached_len=3, table_idx=0),
            _req(extend_len=1, cached_len=5, table_idx=1),
            _req(extend_len=5, cached_len=2, table_idx=2),
            _req(extend_len=3, cached_len=0, table_idx=3),
        ],
        decode_size=2,
    )

    metadata = _prepare(monkeypatch, backend, batch)

    assert metadata.query_indptr.tolist() == [0, 1, 2, 7, 10]
    assert metadata.get_last_indices(4).tolist() == [0, 1, 6, 9]
    assert metadata.decode is not None
    assert metadata.decode.cu_seqlens_q_cpu.tolist() == [0, 1, 2]
    assert metadata.decode.cu_seqlens_k_cpu.tolist() == [0, 4, 10]
    assert metadata.decode.seq_lens_cpu.tolist() == [4, 6]
    assert metadata.decode.indices.tolist() == [0, 1, 2, 3, *range(16, 22)]
    assert metadata.prefill is not None
    assert metadata.prefill.cu_seqlens_q_cpu.tolist() == [0, 5, 8]
    assert metadata.prefill.cu_seqlens_k_cpu.tolist() == [0, 7, 10]
    assert metadata.prefill.seq_lens_cpu.tolist() == [7, 3]
    assert metadata.prefill.indices.tolist() == [*range(32, 39), *range(48, 51)]


def test_mixed_forward_runs_two_wrappers_into_original_order(monkeypatch):
    backend, order = _backend()
    batch = Batch(
        reqs=[
            _req(extend_len=1, cached_len=3, table_idx=0),
            _req(extend_len=1, cached_len=5, table_idx=1),
            _req(extend_len=3, cached_len=2, table_idx=2),
        ],
        decode_size=2,
    )
    _prepare(monkeypatch, backend, batch)
    batch.out_loc = torch.arange(5)
    q = torch.arange(10, dtype=torch.float32).reshape(5, 1, 2)

    output = backend.forward(q, q, q, layer_id=4, batch=batch)

    assert order == ["plan-decode", "plan-prefill", "store", "run-decode", "run-prefill"]
    torch.testing.assert_close(output[:2], q[:2] + 10)
    torch.testing.assert_close(output[2:], q[2:] + 20)
    torch.testing.assert_close(backend.decode_wrappers.runs[0][0], q[:2])
    torch.testing.assert_close(backend.prefill_wrapper.runs[0][0], q[2:])
    assert backend.decode_wrappers.runs[0][1] is not None
    assert backend.prefill_wrapper.runs[0][1] is not None
    assert len(backend.kvcache.stores) == 1


@pytest.mark.parametrize("decode_size, wrapper_name", [(1, "decode"), (0, "prefill")])
def test_pure_forward_keeps_single_wrapper_fast_path(monkeypatch, decode_size, wrapper_name):
    backend, order = _backend()
    req = _req(
        extend_len=1 if decode_size else 4,
        cached_len=3 if decode_size else 0,
        table_idx=0,
    )
    batch = Batch(reqs=[req], decode_size=decode_size)
    _prepare(monkeypatch, backend, batch)
    batch.out_loc = torch.arange(req.extend_len)
    q = torch.arange(req.extend_len * 2, dtype=torch.float32).reshape(-1, 1, 2)

    output = backend.forward(q, q, q, layer_id=0, batch=batch)

    wrapper = backend.decode_wrappers if decode_size else backend.prefill_wrapper
    assert order == [f"plan-{wrapper_name}", "store", f"run-{wrapper_name}"]
    assert wrapper.runs[0][1] is None
    torch.testing.assert_close(output, q + wrapper.delta)


def test_decode_graph_replay_rebinds_only_decode_path(monkeypatch):
    backend, order = _backend()
    batch = Batch(
        reqs=[
            _req(extend_len=1, cached_len=3, table_idx=0),
            _req(extend_len=1, cached_len=5, table_idx=1),
        ],
        decode_size=2,
    )
    metadata = _prepare(monkeypatch, backend, batch)
    graph_wrapper = _Wrapper("graph", 30, order)
    backend.capture = object()
    backend.capture_bs = [2]
    backend.graph_wrappers = {2: graph_wrapper}

    backend.prepare_for_replay(batch)

    assert metadata.decode is not None
    assert metadata.decode.wrapper is graph_wrapper
    assert metadata.decode.initialized
    assert metadata.prefill is None
    assert order == ["plan-graph"]
