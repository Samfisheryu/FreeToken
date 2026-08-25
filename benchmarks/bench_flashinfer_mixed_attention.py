#!/usr/bin/env python3
"""Compare serial and fused FlashInfer mixed-attention execution.

This is an independent kernel benchmark; it does not import FreeToken.  It models
the decode-first mixed layout used by the scheduler and compares:

* ``serial``: ``BatchDecodeWithPagedKVCacheWrapper`` followed by
  ``BatchPrefillWithPagedKVCacheWrapper``, writing into two slices of one output;
* ``pod``: one ``BatchPODWithPagedKVCacheWrapper.run`` launch, followed by the
  decode-first output copies required by the caller.

The plan-once measurement times attention only.  The serving-step measurement
includes planning, 28 repeated layer runs, and required output handling.  The
same synthetic BF16 queries and paged KV cache are used by both paths and their
outputs are checked before timing results are reported.

The script never chooses a physical GPU itself.  Select one explicitly, e.g.::

    CUDA_VISIBLE_DEVICES=2 python benchmarks/bench_flashinfer_mixed_attention.py

When more than one device is visible, also pass ``--confirm-logical-device-0``;
the benchmark always uses logical ``cuda:0``.  Its own live CUDA tensors are
capped at 2 GiB, with one fixed 256 MiB FlashInfer float workspace at a time.
FlashInfer/PyTorch module and allocator overhead is outside that accounting.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any, Callable


MIB = 1024**2
GIB = 1024**3
WORKSPACE_BYTES = 256 * MIB
WRAPPER_INT_WORKSPACE_BYTES = 16 * MIB
MAX_BENCHMARK_TENSOR_BYTES = 2 * GIB

DTYPE_NAME = "bfloat16"
KV_LAYOUT = "NHD"
PAGE_SIZE = 1
NUM_QO_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
DEFAULT_LAYERS = 28


@dataclass(frozen=True)
class Configuration:
    name: str
    decode_requests: int
    prefill_query_lens: tuple[int, int]

    @property
    def prefill_query_tokens(self) -> int:
        return sum(self.prefill_query_lens)


CONFIGURATIONS = {
    configuration.name: configuration
    for configuration in (
        Configuration("p64_d6_p2", 6, (29, 29)),
        Configuration("p256_d12_p2", 12, (122, 122)),
    )
}


@dataclass
class PathInputs:
    query_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    query_indptr_cpu: Any
    kv_indptr_cpu: Any
    kv_indices: Any
    last_page_len_cpu: Any
    seq_lens_cpu: Any


@dataclass
class MixedInputs:
    configuration: Configuration
    q_decode: Any
    q_prefill: Any
    kv_cache: tuple[Any, Any]
    decode: PathInputs
    prefill: PathInputs
    estimated_peak_bytes: int

    @property
    def decode_tokens(self) -> int:
        return self.configuration.decode_requests

    @property
    def query_tokens(self) -> int:
        return self.decode_tokens + self.configuration.prefill_query_tokens


def parse_args(argv: list[str] | None = None) -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=tuple(CONFIGURATIONS),
        default=tuple(CONFIGURATIONS),
    )
    parser.add_argument(
        "--decode-kv-len",
        type=int,
        default=1024,
        help="KV tokens per decode request, including its current query token",
    )
    parser.add_argument(
        "--prefill-cached-kv-len",
        type=int,
        default=1024,
        help="cached KV tokens before each prefill chunk",
    )
    parser.add_argument("--layers", type=int, default=DEFAULT_LAYERS)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--kernel-repeats", type=int, default=100)
    parser.add_argument("--step-repeats", type=int, default=20)
    parser.add_argument("--percentile", type=float, default=95.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rtol", type=float, default=3e-2)
    parser.add_argument("--atol", type=float, default=3e-2)
    parser.add_argument(
        "--confirm-logical-device-0",
        action="store_true",
        help=(
            "confirm use of logical cuda:0 when CUDA_VISIBLE_DEVICES exposes "
            "more than one device"
        ),
    )
    parser.add_argument("--json", type=Path, help="also write the final JSON result here")
    args = parser.parse_args(argv)

    positive = {
        "--decode-kv-len": args.decode_kv_len,
        "--layers": args.layers,
        "--kernel-repeats": args.kernel_repeats,
        "--step-repeats": args.step_repeats,
    }
    for option, value in positive.items():
        if value <= 0:
            parser.error(f"{option} must be positive")
    if args.prefill_cached_kv_len < 0:
        parser.error("--prefill-cached-kv-len must be non-negative")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not 0.0 <= args.percentile <= 100.0:
        parser.error("--percentile must be between 0 and 100")
    if args.rtol < 0.0 or args.atol < 0.0:
        parser.error("--rtol and --atol must be non-negative")
    return parser, args


def require_explicit_device(parser: argparse.ArgumentParser, args: argparse.Namespace) -> str:
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not raw:
        parser.error(
            "CUDA_VISIBLE_DEVICES must explicitly select the target GPU "
            "(for example CUDA_VISIBLE_DEVICES=2)"
        )
    visible = tuple(item.strip() for item in raw.split(",") if item.strip())
    if len(visible) != 1 and not args.confirm_logical_device_0:
        parser.error(
            "CUDA_VISIBLE_DEVICES exposes more than one device; narrow it to one, "
            "or pass --confirm-logical-device-0 to use its first entry"
        )
    return raw


def pinned_int32(torch: Any, values: list[int]) -> Any:
    return torch.tensor(values, dtype=torch.int32, device="cpu", pin_memory=True)


def indptr(torch: Any, lengths: tuple[int, ...]) -> Any:
    result = pinned_int32(torch, [0, *lengths])
    return result.cumsum_(0)


def build_path_inputs(
    torch: Any,
    device: Any,
    query_lens: tuple[int, ...],
    kv_lens: tuple[int, ...],
    first_page: int,
) -> PathInputs:
    page_counts = tuple((length + PAGE_SIZE - 1) // PAGE_SIZE for length in kv_lens)
    page_count = sum(page_counts)
    last_page_lens = tuple(
        (length - 1) % PAGE_SIZE + 1 for length in kv_lens
    )
    return PathInputs(
        query_lens=query_lens,
        kv_lens=kv_lens,
        query_indptr_cpu=indptr(torch, query_lens),
        kv_indptr_cpu=indptr(torch, page_counts),
        kv_indices=torch.arange(
            first_page,
            first_page + page_count,
            dtype=torch.int32,
            device=device,
        ),
        last_page_len_cpu=pinned_int32(torch, list(last_page_lens)),
        seq_lens_cpu=pinned_int32(torch, list(kv_lens)),
    )


def estimate_peak_bytes(total_pages: int, total_query_tokens: int) -> int:
    element_bytes = 2  # BF16
    cache_bytes = (
        2 * total_pages * PAGE_SIZE * NUM_KV_HEADS * HEAD_DIM * element_bytes
    )
    query_bytes = total_query_tokens * NUM_QO_HEADS * HEAD_DIM * element_bytes
    # POD returns two output tensors, serving-step assembly needs one output, and
    # validation retains one serial reference while POD is live.
    output_bytes = 3 * query_bytes
    page_metadata_bytes = 2 * total_pages * 4
    return (
        cache_bytes
        + query_bytes
        + output_bytes
        + page_metadata_bytes
        + WORKSPACE_BYTES
        + WRAPPER_INT_WORKSPACE_BYTES
    )


def make_inputs(
    torch: Any,
    device: Any,
    configuration: Configuration,
    decode_kv_len: int,
    prefill_cached_kv_len: int,
    seed: int,
) -> MixedInputs:
    decode_query_lens = (1,) * configuration.decode_requests
    decode_kv_lens = (decode_kv_len,) * configuration.decode_requests
    prefill_kv_lens = tuple(
        prefill_cached_kv_len + query_len
        for query_len in configuration.prefill_query_lens
    )
    decode_pages = sum(decode_kv_lens)
    prefill_pages = sum(prefill_kv_lens)
    total_pages = decode_pages + prefill_pages
    total_query_tokens = configuration.decode_requests + configuration.prefill_query_tokens
    estimated_peak_bytes = estimate_peak_bytes(total_pages, total_query_tokens)
    if estimated_peak_bytes > MAX_BENCHMARK_TENSOR_BYTES:
        raise ValueError(
            "requested KV lengths exceed the 2 GiB benchmark-owned CUDA tensor cap: "
            f"estimated {estimated_peak_bytes / MIB:.1f} MiB"
        )

    decode = build_path_inputs(
        torch, device, decode_query_lens, decode_kv_lens, first_page=0
    )
    prefill = build_path_inputs(
        torch,
        device,
        configuration.prefill_query_lens,
        prefill_kv_lens,
        first_page=decode_pages,
    )

    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    dtype = torch.bfloat16
    q_decode = torch.randn(
        configuration.decode_requests,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    q_prefill = torch.randn(
        configuration.prefill_query_tokens,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    k_cache = torch.randn(
        total_pages,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    v_cache = torch.randn(
        total_pages,
        PAGE_SIZE,
        NUM_KV_HEADS,
        HEAD_DIM,
        dtype=dtype,
        device=device,
        generator=generator,
    )
    return MixedInputs(
        configuration=configuration,
        q_decode=q_decode,
        q_prefill=q_prefill,
        kv_cache=(k_cache, v_cache),
        decode=decode,
        prefill=prefill,
        estimated_peak_bytes=estimated_peak_bytes,
    )


def plan_serial(torch: Any, decode_wrapper: Any, prefill_wrapper: Any, inputs: MixedInputs) -> None:
    decode = inputs.decode
    decode_wrapper.plan(
        decode.kv_indptr_cpu,
        decode.kv_indices,
        decode.last_page_len_cpu,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        seq_lens=decode.seq_lens_cpu,
        non_blocking=True,
    )
    prefill = inputs.prefill
    prefill_wrapper.plan(
        prefill.query_indptr_cpu,
        prefill.kv_indptr_cpu,
        prefill.kv_indices,
        prefill.last_page_len_cpu,
        NUM_QO_HEADS,
        NUM_KV_HEADS,
        HEAD_DIM,
        PAGE_SIZE,
        causal=True,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        seq_lens=prefill.seq_lens_cpu,
        non_blocking=True,
    )


def plan_pod(torch: Any, pod_wrapper: Any, inputs: MixedInputs) -> None:
    prefill = inputs.prefill
    decode = inputs.decode
    pod_wrapper.plan(
        prefill.query_indptr_cpu,
        prefill.kv_indptr_cpu,
        prefill.kv_indices,
        prefill.last_page_len_cpu,
        decode.query_indptr_cpu,
        decode.kv_indptr_cpu,
        decode.kv_indices,
        decode.last_page_len_cpu,
        num_qo_heads=NUM_QO_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        head_dim=HEAD_DIM,
        page_size=PAGE_SIZE,
        pos_encoding_mode="NONE",
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        non_blocking=True,
    )


def percentile(values: list[float], requested: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * requested / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize(samples_ms: list[float], requested_percentile: float) -> dict[str, Any]:
    percentile_name = f"p{requested_percentile:g}_ms"
    return {
        "samples": len(samples_ms),
        "median_ms": statistics.median(samples_ms),
        percentile_name: percentile(samples_ms, requested_percentile),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def cuda_event_samples(
    torch: Any,
    run: Callable[[], Any],
    warmup: int,
    repeats: int,
) -> list[float]:
    retained = None
    for _ in range(warmup):
        retained = run()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        retained = run()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    del retained
    return samples


def wall_clock_samples(
    torch: Any,
    run: Callable[[], Any],
    warmup: int,
    repeats: int,
) -> list[float]:
    retained = None
    for _ in range(warmup):
        retained = run()
        torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        retained = run()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000.0)
    del retained
    return samples


def benchmark_serial(
    torch: Any,
    flashinfer: Any,
    inputs: MixedInputs,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], tuple[Any, Any]]:
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=inputs.q_decode.device)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace,
        kv_layout=KV_LAYOUT,
        use_tensor_cores=False,
        backend="fa2",
    )
    prefill_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace,
        kv_layout=KV_LAYOUT,
        backend="fa2",
    )
    output = torch.empty(
        inputs.query_tokens,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=inputs.q_decode.device,
    )

    def plan() -> None:
        plan_serial(torch, decode_wrapper, prefill_wrapper, inputs)

    def run_attention() -> Any:
        decode_wrapper.run(
            q=inputs.q_decode,
            paged_kv_cache=inputs.kv_cache,
            out=output[: inputs.decode_tokens],
        )
        prefill_wrapper.run(
            q=inputs.q_prefill,
            paged_kv_cache=inputs.kv_cache,
            out=output[inputs.decode_tokens :],
        )
        return output

    plan()
    torch.cuda.synchronize()
    run_attention()
    torch.cuda.synchronize()
    reference = (
        output[: inputs.decode_tokens].clone(),
        output[inputs.decode_tokens :].clone(),
    )

    kernel_samples = cuda_event_samples(
        torch, run_attention, args.warmup, args.kernel_repeats
    )

    def serving_step() -> Any:
        plan()
        for _ in range(args.layers):
            run_attention()
        return output

    serving_samples = wall_clock_samples(
        torch, serving_step, args.warmup, args.step_repeats
    )
    result = {
        "plan_once_attention_only": summarize(kernel_samples, args.percentile),
        "serving_step_plan_plus_layers": summarize(serving_samples, args.percentile),
    }
    return result, reference


def validation_metrics(torch: Any, actual: Any, expected: Any) -> dict[str, float]:
    difference = (actual.float() - expected.float()).abs()
    relative = difference / expected.float().abs().clamp_min(1e-3)
    return {
        "max_abs": difference.max().item(),
        "max_rel_with_1e-3_floor": relative.max().item(),
    }


def benchmark_pod(
    torch: Any,
    flashinfer: Any,
    inputs: MixedInputs,
    reference: tuple[Any, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    workspace = torch.empty(WORKSPACE_BYTES, dtype=torch.uint8, device=inputs.q_decode.device)
    pod_wrapper = flashinfer.BatchPODWithPagedKVCacheWrapper(
        workspace, kv_layout=KV_LAYOUT
    )
    assembled = torch.empty(
        inputs.query_tokens,
        NUM_QO_HEADS,
        HEAD_DIM,
        dtype=torch.bfloat16,
        device=inputs.q_decode.device,
    )

    def plan() -> None:
        plan_pod(torch, pod_wrapper, inputs)

    def run_attention() -> tuple[Any, Any]:
        return pod_wrapper.run(
            inputs.q_prefill,
            inputs.kv_cache,
            inputs.q_decode,
            inputs.kv_cache,
            causal_p=True,
        )

    def run_and_assemble() -> Any:
        output_prefill, output_decode = run_attention()
        assembled[: inputs.decode_tokens].copy_(output_decode)
        assembled[inputs.decode_tokens :].copy_(output_prefill)
        return assembled

    try:
        plan()
        torch.cuda.synchronize()
        output_prefill, output_decode = run_attention()
        torch.cuda.synchronize()
    except Exception:
        print(
            "BatchPODWithPagedKVCacheWrapper failed; the original FlashInfer "
            "0.6.x API/JIT/architecture error follows.",
            file=sys.stderr,
        )
        raise

    validation = {
        "decode": validation_metrics(torch, output_decode, reference[0]),
        "prefill": validation_metrics(torch, output_prefill, reference[1]),
        "rtol": args.rtol,
        "atol": args.atol,
    }
    torch.testing.assert_close(
        output_decode, reference[0], rtol=args.rtol, atol=args.atol
    )
    torch.testing.assert_close(
        output_prefill, reference[1], rtol=args.rtol, atol=args.atol
    )
    del output_prefill, output_decode

    kernel_samples = cuda_event_samples(
        torch, run_attention, args.warmup, args.kernel_repeats
    )

    def serving_step() -> Any:
        plan()
        for _ in range(args.layers):
            run_and_assemble()
        return assembled

    serving_samples = wall_clock_samples(
        torch, serving_step, args.warmup, args.step_repeats
    )
    result = {
        "plan_once_attention_only": summarize(kernel_samples, args.percentile),
        "serving_step_plan_plus_layers_and_decode_first_copies": summarize(
            serving_samples, args.percentile
        ),
    }
    return result, validation


def run_configuration(
    torch: Any,
    flashinfer: Any,
    configuration: Configuration,
    args: argparse.Namespace,
    device: Any,
) -> dict[str, Any]:
    inputs = make_inputs(
        torch,
        device,
        configuration,
        args.decode_kv_len,
        args.prefill_cached_kv_len,
        args.seed,
    )
    print(
        f"[{configuration.name}] serial path: compile/warm/measure",
        file=sys.stderr,
        flush=True,
    )
    serial, reference = benchmark_serial(torch, flashinfer, inputs, args)
    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"[{configuration.name}] POD path: validate/compile/warm/measure",
        file=sys.stderr,
        flush=True,
    )
    pod, validation = benchmark_pod(torch, flashinfer, inputs, reference, args)
    serial_kernel = serial["plan_once_attention_only"]["median_ms"]
    pod_kernel = pod["plan_once_attention_only"]["median_ms"]
    serial_step = serial["serving_step_plan_plus_layers"]["median_ms"]
    pod_step = pod[
        "serving_step_plan_plus_layers_and_decode_first_copies"
    ]["median_ms"]
    return {
        "configuration": configuration.name,
        "decode_requests": configuration.decode_requests,
        "prefill_requests": len(configuration.prefill_query_lens),
        "prefill_query_lens": configuration.prefill_query_lens,
        "total_query_tokens": inputs.query_tokens,
        "decode_kv_len": args.decode_kv_len,
        "prefill_cached_kv_len": args.prefill_cached_kv_len,
        "prefill_kv_lens": inputs.prefill.kv_lens,
        "estimated_peak_benchmark_tensors_mib": inputs.estimated_peak_bytes / MIB,
        "validation": validation,
        "serial": serial,
        "pod": pod,
        "serial_over_pod_speedup": {
            "plan_once_attention_only": serial_kernel / pod_kernel,
            "serving_step": serial_step / pod_step,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser, args = parse_args(argv)
    visible_devices = require_explicit_device(parser, args)

    import torch
    import flashinfer

    flashinfer_version = importlib.metadata.version("flashinfer-python")
    if not flashinfer_version.startswith("0.6."):
        raise RuntimeError(
            "this benchmark targets the FlashInfer 0.6.x BatchPOD API; "
            f"installed version is {flashinfer_version}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available under the selected CUDA_VISIBLE_DEVICES")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no logical CUDA device is visible")

    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    result = {
        "flashinfer_version": flashinfer_version,
        "gpu": {
            "cuda_visible_devices": visible_devices,
            "logical_device": "cuda:0",
            "name": properties.name,
            "compute_capability": f"{properties.major}.{properties.minor}",
        },
        "attention_shape": {
            "dtype": DTYPE_NAME,
            "kv_layout": KV_LAYOUT,
            "page_size": PAGE_SIZE,
            "num_qo_heads": NUM_QO_HEADS,
            "num_kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "layers_per_serving_step": args.layers,
        },
        "timing": {
            "plan_once_attention_only": "CUDA events; plan performed once before timing",
            "serving_step": (
                "host wall clock with CUDA synchronization; includes plan, all layer "
                "runs, and required output handling"
            ),
            "warmup": args.warmup,
            "kernel_repeats": args.kernel_repeats,
            "step_repeats": args.step_repeats,
            "reported_percentile": args.percentile,
        },
        "memory": {
            "float_workspace_mib": WORKSPACE_BYTES / MIB,
            "hard_cap_benchmark_owned_cuda_tensors_mib": (
                MAX_BENCHMARK_TENSOR_BYTES / MIB
            ),
            "note": (
                "serial and POD wrappers are benchmarked sequentially; FlashInfer/PyTorch "
                "module and allocator overhead is excluded from the cap"
            ),
        },
        "configurations": [],
    }

    for index, name in enumerate(args.configs):
        configuration = CONFIGURATIONS[name]
        result["configurations"].append(
            run_configuration(torch, flashinfer, configuration, args, device)
        )
        if index + 1 < len(args.configs):
            gc.collect()
            torch.cuda.empty_cache()

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
