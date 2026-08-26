from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import pytest


_ISOLATED_MODULE_ROOTS = ("freetoken", "nowag_vllm", "torch", "triton")


def _is_isolated_module(name: str) -> bool:
    return any(name == root or name.startswith(f"{root}.") for root in _ISOLATED_MODULE_ROOTS)


@pytest.fixture(autouse=True)
def _restore_import_state():
    before = {
        name: module
        for name, module in sys.modules.items()
        if _is_isolated_module(name)
    }
    yield
    for name in tuple(sys.modules):
        if _is_isolated_module(name) and name not in before:
            sys.modules.pop(name, None)
    sys.modules.update(before)


class _Dummy:
    def __init__(self, name: str = "dummy") -> None:
        self.name = name

    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return self

    def __getattr__(self, name: str):
        return _Dummy(f"{self.name}.{name}")

    def __getitem__(self, key):
        return self

    def __iter__(self):
        return iter(())

    def __or__(self, other):
        return self

    def __ror__(self, other):
        return self

    def __mro_entries__(self, bases):
        return ()


class _StubModule(types.ModuleType):
    def __getattr__(self, name: str):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _Dummy(f"{self.__name__}.{name}")
        setattr(self, name, value)
        return value


class _FakeKernel:
    def __init__(self, function) -> None:
        self.function = function

    def __getitem__(self, grid):
        return lambda *args, **kwargs: None


def _namespace(name: str, path: Path) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    return module


def _load_align_module(monkeypatch):
    package = Path(__file__).resolve().parents[2] / "python" / "freetoken"
    for name in tuple(sys.modules):
        if name == "freetoken" or name.startswith("freetoken."):
            monkeypatch.delitem(sys.modules, name, raising=False)

    monkeypatch.setitem(sys.modules, "freetoken", _namespace("freetoken", package))
    kernel = _StubModule("freetoken.kernel")
    kernel.__path__ = [str(package / "kernel")]
    monkeypatch.setitem(sys.modules, "freetoken.kernel", kernel)
    monkeypatch.setitem(
        sys.modules,
        "freetoken.kernel.triton",
        _namespace("freetoken.kernel.triton", package / "kernel" / "triton"),
    )

    fake_torch = _StubModule("torch")
    fake_torch.Tensor = type("Tensor", (), {})
    fake_torch.empty = lambda shape, dtype=None, device=None: _FakeTensor(
        tuple(shape), dtype, device=device
    )
    fake_torch.zeros = lambda shape, dtype=None, device=None: _FakeTensor(
        tuple(shape), dtype, device=device
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    fake_triton = _StubModule("triton")

    def decorator(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return _FakeKernel(args[0])
        return lambda function: _FakeKernel(function)

    fake_triton.jit = decorator
    monkeypatch.setitem(sys.modules, "triton", fake_triton)
    monkeypatch.setitem(sys.modules, "triton.language", _StubModule("triton.language"))
    return importlib.import_module("freetoken.kernel.triton.moe_align")


def test_uses_large_moe_align_public_boundary(monkeypatch) -> None:
    align = _load_align_module(monkeypatch)
    assert align.uses_large_moe_align(1024) is False
    assert align.uses_large_moe_align(1025) is True


class _FakeTensor:
    def __init__(self, shape: tuple[int, ...], dtype, device="cuda:0") -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.ndim = len(shape)
        self.is_cuda = True

    def numel(self) -> int:
        result = 1
        for extent in self.shape:
            result *= extent
        return result

    def unsqueeze(self, dim: int):
        shape = list(self.shape)
        if dim < 0:
            dim += len(shape) + 1
        shape.insert(dim, 1)
        return _FakeTensor(tuple(shape), self.dtype)

    def is_contiguous(self) -> bool:
        return True


def _load_routed_module(monkeypatch):
    align = _load_align_module(monkeypatch)
    package = Path(__file__).resolve().parents[2] / "python" / "freetoken"
    monkeypatch.setitem(
        sys.modules,
        "freetoken.moe",
        _namespace("freetoken.moe", package / "moe"),
    )
    nowag = types.ModuleType("nowag_vllm")
    nowag.__path__ = []
    cuda_ops = _StubModule("nowag_vllm.cuda_ops")
    nowag_ops = types.ModuleType("nowag_vllm.moe_ops")
    nowag.cuda_ops = cuda_ops
    nowag.moe_ops = nowag_ops
    monkeypatch.setitem(sys.modules, "nowag_vllm", nowag)
    monkeypatch.setitem(sys.modules, "nowag_vllm.cuda_ops", cuda_ops)
    monkeypatch.setitem(sys.modules, "nowag_vllm.moe_ops", nowag_ops)
    return align, nowag_ops, importlib.import_module("freetoken.moe.fused_nowag")


def _routed_args(num_tokens: int):
    fake_torch = sys.modules["torch"]
    bf16 = fake_torch.bfloat16
    int32 = fake_torch.int32
    float32 = fake_torch.float32
    top_k = 8
    hidden = 2048
    intermediate = 512
    experts = 256
    routes = num_tokens * top_k
    tensor = lambda shape, dtype=bf16: _FakeTensor(shape, dtype)
    positional = (
        tensor((num_tokens, hidden)),
        tensor((num_tokens, top_k), int32),
        tensor((num_tokens, top_k), float32),
        tensor((4096, 6)),
        tensor((experts, 128, intermediate), int32),
        tensor((experts, hidden)),
        tensor((experts, intermediate)),
        tensor((experts, 128, intermediate), int32),
        tensor((experts, hidden)),
        tensor((experts, intermediate)),
        tensor((experts, 32, hidden), int32),
        tensor((experts, intermediate)),
        tensor((experts, hidden)),
    )
    keywords = {
        "model_type": "qwen3_5_moe",
        "model_num_experts": experts,
        "swiglu_limit": None,
        "gate_up_backend": "cuda_exact_k48",
        "down_backend": "cuda_exact_k48",
        "cuda_launch_plan": object(),
        "output": tensor((num_tokens, hidden)),
        "middle_workspace": tensor((16000, intermediate)),
        "route_output_workspace": tensor((routes, hidden)),
    }
    return positional, keywords


@pytest.mark.parametrize(
    ("num_tokens", "expect_adaptive"), ((128, False), (129, True))
)
def test_routed_experts_selects_adaptive_callback_only_above_1024_routes(
    monkeypatch, num_tokens: int, expect_adaptive: bool
) -> None:
    align, nowag_ops, routed = _load_routed_module(monkeypatch)
    plain_result = (object(), object(), object())
    adaptive_result = (object(), object(), object())
    plain_calls = []
    adaptive_calls = []

    def plain(*args, **kwargs):
        plain_calls.append((args, kwargs))
        return plain_result

    def adaptive(*args, **kwargs):
        adaptive_calls.append((args, kwargs))
        return adaptive_result

    align.moe_align_block_size = plain
    align.moe_align_block_size_adaptive = adaptive
    captured = {}
    result_marker = object()

    def capture_nowag(**kwargs):
        captured.update(kwargs)
        captured["builder_selection"] = (
            "adaptive_callback"
            if kwargs["align_routes_adaptive"] is not None
            else "plugin_builder"
        )
        return result_marker

    nowag_ops.nowag_fused_moe = capture_nowag
    positional, keywords = _routed_args(num_tokens)
    assert routed.routed_experts_nowag(*positional, **keywords) is result_marker

    ordinary_args = (positional[1], 16, 256)
    assert captured["align_routes"](*ordinary_args) is plain_result
    assert plain_calls == [(ordinary_args, {"alignment_storage": None})]
    if expect_adaptive:
        assert captured["builder_selection"] == "adaptive_callback"
        callback = captured["align_routes_adaptive"]
        assert callback is adaptive
        adaptive_args = tuple(object() for _ in range(6))
        assert callback(*adaptive_args) is adaptive_result
        assert adaptive_calls == [(adaptive_args, {})]
    else:
        assert captured["builder_selection"] == "plugin_builder"
        assert captured["align_routes_adaptive"] is None
        assert adaptive_calls == []


def test_adaptive_callback_error_propagates_through_routed_experts(monkeypatch) -> None:
    align, nowag_ops, routed = _load_routed_module(monkeypatch)
    expected = RuntimeError("adaptive builder failed")

    def fail_adaptive(*args, **kwargs):
        raise expected

    align.moe_align_block_size_adaptive = fail_adaptive

    def invoke_callback(**kwargs):
        callback = kwargs["align_routes_adaptive"]
        return callback(*(object() for _ in range(6)))

    nowag_ops.nowag_fused_moe = invoke_callback
    positional, keywords = _routed_args(129)
    with pytest.raises(RuntimeError) as raised:
        routed.routed_experts_nowag(*positional, **keywords)
    assert raised.value is expected


def test_routed_experts_propagates_plugin_error(monkeypatch) -> None:
    _, nowag_ops, routed = _load_routed_module(monkeypatch)
    expected = RuntimeError("plugin failed")

    def fail_nowag(**kwargs):
        raise expected

    nowag_ops.nowag_fused_moe = fail_nowag
    positional, keywords = _routed_args(129)
    with pytest.raises(RuntimeError) as raised:
        routed.routed_experts_nowag(*positional, **keywords)
    assert raised.value is expected
