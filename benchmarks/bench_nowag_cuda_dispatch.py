#!/usr/bin/env python3
"""Profile NoWAG Triton and Exact-K48 through FreeToken's MoE entry point."""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import torch

from freetoken.moe.fused_nowag import routed_experts_nowag
from freetoken.moe.nowag import get_nowag_model_rule
from nowag_vllm.moe_ops import required_structural_middle_rows
from nowag_vllm.moe_tuning import MoeCudaLaunchPlan, MoeCudaStageConfig


Runner = Callable[[], torch.Tensor]


def _csv_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty integer list")
    return result


def _csv_strings(value: str) -> list[str]:
    result = [item for item in value.split(",") if item]
    if not result:
        raise argparse.ArgumentTypeError("expected a non-empty value list")
    return result


def _packed_words(width: int, group_size: int, assignment_bits: int) -> int:
    return math.ceil(math.ceil(width / group_size) * assignment_bits / 32)


def _plan_label(plan: MoeCudaLaunchPlan) -> str:
    gate = plan.gate_up
    down = plan.down
    return (
        f"exact_gbm{gate.block_m}_gbn{gate.block_n}_"
        f"dbm{down.block_m}_dbn{down.block_n}"
    )


def _candidate_plans(intermediate_size: int, hidden_size: int):
    plans: dict[str, MoeCudaLaunchPlan] = {}
    for down_bm in (64, 128):
        for gate_bm in (32, 64, 128):
            if down_bm % gate_bm:
                continue
            for gate_bn in (64, 128):
                if intermediate_size % gate_bn:
                    continue
                for down_bn in (64, 128):
                    if hidden_size % down_bn:
                        continue
                    plan = MoeCudaLaunchPlan(
                        gate_up=MoeCudaStageConfig(
                            block_m=gate_bm,
                            block_n=gate_bn,
                            num_warps=gate_bn // 16,
                            num_stages=2,
                        ),
                        down=MoeCudaStageConfig(
                            block_m=down_bm,
                            block_n=down_bn,
                            num_warps=down_bn // 16,
                            num_stages=2,
                        ),
                        source="manual",
                        profile_name=None,
                    )
                    plans[_plan_label(plan)] = plan
    if not plans:
        raise ValueError("the requested shape has no Exact-K48 tile candidates")
    return plans


def _measure_interleaved(
    runners: dict[str, Runner],
    *,
    warmup: int,
    samples: int,
    order_seed: int,
) -> dict[str, float]:
    labels = list(runners)
    rng = random.Random(order_seed)
    for _ in range(warmup):
        rng.shuffle(labels)
        for label in labels:
            runners[label]()
    torch.cuda.synchronize()

    if samples % 4:
        raise ValueError("samples must be divisible by four for ABBA timing")

    events = {label: [] for label in labels}
    for _ in range(samples // 4):
        rng.shuffle(labels)
        forward = labels.copy()
        reverse = list(reversed(forward))
        for order in (forward, reverse, reverse, forward):
            for label in order:
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                runners[label]()
                end.record()
                events[label].append((start, end))
    torch.cuda.synchronize()
    return {
        label: statistics.median(
            start.elapsed_time(end) for start, end in pairs
        )
        for label, pairs in events.items()
    }


def _relative_rmse(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = actual.float() - expected.float()
    scale = expected.float().square().mean().sqrt().clamp_min(1e-12)
    return float((difference.square().mean().sqrt() / scale).item())


def _recommendations(rows, tokens, plan_names):
    recommendations = []
    for num_tokens in tokens:
        selected = [row for row in rows if row["tokens"] == num_tokens]
        candidates = []
        for name in plan_names:
            if any(name not in row["timings_ms"] for row in selected):
                continue
            normalized = []
            speedups = []
            for row in selected:
                timings = row["timings_ms"]
                normalized.append(timings[name] / min(timings.values()))
                speedups.append(timings["triton"] / timings[name])
            candidates.append(
                (
                    statistics.geometric_mean(normalized),
                    max(normalized),
                    name,
                    min(speedups),
                    statistics.geometric_mean(speedups),
                )
            )
        if not candidates:
            continue
        geomean_normalized, worst_normalized, name, minimum, geomean = min(
            candidates
        )
        recommendations.append(
            {
                "tokens": num_tokens,
                "plan": name,
                "geomean_normalized_to_case_best": geomean_normalized,
                "worst_normalized_to_case_best": worst_normalized,
                "min_speedup_over_triton": minimum,
                "geomean_speedup_over_triton": geomean,
                "exact_faster_in_every_measured_case": minimum > 1.0,
            }
        )
    return recommendations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--seeds", type=_csv_ints, default=_csv_ints("7,19,31"))
    parser.add_argument("--tokens", type=_csv_ints, default=_csv_ints("1,2,4,8,16"))
    parser.add_argument(
        "--routings",
        type=_csv_strings,
        default=_csv_strings("balanced,hotspot"),
    )
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument(
        "--bank-rows",
        type=int,
        help="Physical expert rows in the GPU bank; defaults to --experts",
    )
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=2048)
    parser.add_argument("--intermediate-size", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--rmse-tolerance", type=float, default=1e-2)
    parser.add_argument(
        "--activation-mode",
        choices=("qwen", "dsv4"),
        default="qwen",
        help="Expert activation math; CUDA backend selection remains shape/profile based",
    )
    parser.add_argument("--swiglu-limit", type=float, default=10.0)
    parser.add_argument(
        "--check-auto",
        action="store_true",
        help="also validate and time the profile-selected auto backend",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if set(args.routings) - {"balanced", "hotspot"}:
        raise ValueError("routings must contain only balanced and hotspot")
    if args.warmup < 0 or args.samples <= 0:
        raise ValueError("warmup must be non-negative and samples must be positive")
    if args.samples % 4:
        raise ValueError("samples must be divisible by four for ABBA timing")
    if args.experts < args.top_k:
        raise ValueError("experts must be at least top-k")
    bank_rows = args.bank_rows or args.experts
    if bank_rows < args.experts:
        raise ValueError("bank-rows must be at least experts")

    started = time.perf_counter()
    torch.manual_seed(20260824)
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16
    group_size = 6
    assignment_bits = 12
    codebook_size = 4096
    model_experts = args.experts
    model_type = (
        "deepseek_v4" if args.activation_mode == "dsv4" else "qwen3_5_moe"
    )
    activation_rule = get_nowag_model_rule(model_type)
    hidden_size = args.hidden_size
    intermediate_size = args.intermediate_size
    top_k = args.top_k

    codebook = (torch.randn(codebook_size, group_size, device=device) * 0.05).to(
        dtype
    )

    def assignments(out_features: int, in_features: int) -> torch.Tensor:
        words = _packed_words(in_features, group_size, assignment_bits)
        return torch.randint(
            -(1 << 31),
            (1 << 31) - 1,
            (bank_rows, words, out_features),
            dtype=torch.int32,
            device=device,
        )

    def normalizer(width: int) -> torch.Tensor:
        return (0.9 + 0.2 * torch.rand(bank_rows, width, device=device)).to(dtype)

    gate_assignments = assignments(intermediate_size, hidden_size)
    up_assignments = assignments(intermediate_size, hidden_size)
    down_assignments = assignments(hidden_size, intermediate_size)
    gate_input_norm = normalizer(hidden_size)
    gate_output_norm = normalizer(intermediate_size)
    up_input_norm = normalizer(hidden_size)
    up_output_norm = normalizer(intermediate_size)
    down_input_norm = normalizer(intermediate_size)
    down_output_norm = normalizer(hidden_size)
    plans = _candidate_plans(intermediate_size, hidden_size)
    max_down_bm = max(plan.down.block_m for plan in plans.values())
    rows = []

    for seed in args.seeds:
        torch.manual_seed(seed)
        for num_tokens in args.tokens:
            for routing in args.routings:
                hidden_states = torch.randn(
                    num_tokens, hidden_size, dtype=dtype, device=device
                )
                route_count = num_tokens * top_k
                if routing == "balanced":
                    stride = max(1, bank_rows // route_count)
                    flat_ids = (
                        torch.arange(route_count, dtype=torch.int32, device=device)
                        * stride
                        + seed
                    ).remainder(bank_rows)
                else:
                    hot_experts = min(bank_rows, max(top_k, 16))
                    flat_ids = (
                        torch.arange(route_count, dtype=torch.int32, device=device)
                        + seed
                    ).remainder(hot_experts)
                topk_ids = flat_ids.view(num_tokens, top_k)
                topk_weights = torch.rand(num_tokens, top_k, device=device)
                topk_weights /= topk_weights.sum(dim=1, keepdim=True)
                capacity = required_structural_middle_rows(
                    route_count, bank_rows, max_down_bm
                )
                output = torch.empty_like(hidden_states)
                middle = torch.empty(
                    2 * capacity, intermediate_size, dtype=dtype, device=device
                )
                route_output = torch.empty(
                    route_count, hidden_size, dtype=dtype, device=device
                )
                common = dict(
                    x=hidden_states,
                    slots=topk_ids,
                    topk_weights=topk_weights,
                    codebook=codebook,
                    gate_assignments=gate_assignments,
                    gate_input_norm=gate_input_norm,
                    gate_output_norm=gate_output_norm,
                    up_assignments=up_assignments,
                    up_input_norm=up_input_norm,
                    up_output_norm=up_output_norm,
                    down_assignments=down_assignments,
                    down_input_norm=down_input_norm,
                    down_output_norm=down_output_norm,
                    model_type=model_type,
                    model_num_experts=model_experts,
                    swiglu_limit=(
                        args.swiglu_limit if args.activation_mode == "dsv4" else None
                    ),
                    output=output,
                    middle_workspace=middle,
                    route_output_workspace=route_output,
                )

                def triton_run():
                    return routed_experts_nowag(
                        **common,
                        gate_up_backend="triton",
                        down_backend="triton",
                    )

                reference = triton_run().clone()
                torch.cuda.synchronize()
                runners: dict[str, Runner] = {"triton": triton_run}
                relative_rmse = {"triton": 0.0}
                failures = {}
                if args.check_auto:

                    def auto_run():
                        return routed_experts_nowag(**common)

                    try:
                        actual = auto_run().clone()
                        torch.cuda.synchronize()
                        error = _relative_rmse(actual, reference)
                        if not math.isfinite(error) or error > args.rmse_tolerance:
                            raise AssertionError(f"relative_rmse={error}")
                        relative_rmse["auto"] = error
                        runners["auto"] = auto_run
                    except Exception as exc:
                        failures["auto"] = f"{type(exc).__name__}: {exc}"
                for name, plan in plans.items():

                    def exact_run(plan=plan):
                        return routed_experts_nowag(
                            **common,
                            gate_up_backend="cuda_exact_k48",
                            down_backend="cuda_exact_k48",
                            cuda_launch_plan=plan,
                        )

                    try:
                        actual = exact_run().clone()
                        torch.cuda.synchronize()
                        error = _relative_rmse(actual, reference)
                        if not math.isfinite(error) or error > args.rmse_tolerance:
                            raise AssertionError(f"relative_rmse={error}")
                        relative_rmse[name] = error
                        runners[name] = exact_run
                    except Exception as exc:
                        failures[name] = f"{type(exc).__name__}: {exc}"

                timings = _measure_interleaved(
                    runners,
                    warmup=args.warmup,
                    samples=args.samples,
                    order_seed=seed * 1_000_003 + num_tokens * 101 + len(rows),
                )
                best = min(timings, key=timings.get)
                row = {
                    "seed": seed,
                    "tokens": num_tokens,
                    "routing": routing,
                    "timings_ms": timings,
                    "relative_rmse_vs_triton": relative_rmse,
                    "failures": failures,
                    "best": best,
                    "best_ms": timings[best],
                    "best_speedup_over_triton": timings["triton"] / timings[best],
                }
                rows.append(row)
                print(
                    f"seed={seed} M={num_tokens} {routing}: {best} "
                    f"{timings[best]:.4f} ms, Triton {timings['triton']:.4f} ms",
                    flush=True,
                )

    properties = torch.cuda.get_device_properties(device)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]
    payload = {
        "schema_version": 1,
        "command": command,
        "elapsed_seconds": time.perf_counter() - started,
        "gpu": {
            "name": properties.name,
            "compute_capability": list(torch.cuda.get_device_capability(device)),
            "total_memory_bytes": properties.total_memory,
        },
        "entry": "freetoken.moe.fused_nowag.routed_experts_nowag",
        "routing_primitives": [
            "freetoken.kernel.triton.moe_align.moe_align_block_size",
            "freetoken.kernel.moe_sum_reduce_triton",
        ],
        "shape": {
            "model_num_experts": model_experts,
            "physical_bank_rows": bank_rows,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "top_k": top_k,
            "dtype": "bfloat16",
            "group_size": group_size,
            "assignment_bits": assignment_bits,
            "assignment_layout": "word_major",
            "activation_mode": args.activation_mode,
            "activation_math": asdict(activation_rule),
        },
        "settings": {
            "seeds": args.seeds,
            "tokens": args.tokens,
            "routings": args.routings,
            "warmup_rounds": args.warmup,
            "samples_per_candidate": args.samples,
            "relative_rmse_tolerance": args.rmse_tolerance,
            "check_auto": args.check_auto,
            "timing": (
                "randomized multi-candidate ABBA: forward, reverse, reverse, "
                "forward"
            ),
        },
        "plans": {
            name: {
                "gate_up": asdict(plan.gate_up),
                "down": asdict(plan.down),
            }
            for name, plan in plans.items()
        },
        "recommendations": _recommendations(rows, args.tokens, plans),
        "results": rows,
    }
    args.result_json.parent.mkdir(parents=True, exist_ok=True)
    args.result_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.result_json}")


if __name__ == "__main__":
    main()
