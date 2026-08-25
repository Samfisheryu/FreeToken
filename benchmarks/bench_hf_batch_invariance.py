#!/usr/bin/env python3
"""Compare greedy generation across Hugging Face batching paths.

The continuous-batching paths use Transformers' public ``generate_batch`` and
``continuous_batching_context_manager`` APIs.  They are not local scheduler
emulations.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import ContinuousBatchingConfig, GenerationConfig


OFFICIAL_REFERENCES = [
    "https://huggingface.co/docs/transformers/continuous_batching",
    "https://huggingface.co/docs/transformers/continuous_batching_architecture",
    "https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/generation/continuous_batching/continuous_api.py",
    "https://github.com/huggingface/transformers/blob/v5.15.1/src/transformers/generation/configuration_utils.py",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--cache-dir", default="/data1/lmcache_kv/hf-cache")
    parser.add_argument("--output", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=256)
    parser.add_argument("--cb-memory-percent", type=float, default=0.40)
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args()
    if args.repeats < 3:
        parser.error("--repeats must be at least 3")
    return args


def prompts() -> dict[str, str]:
    long_context = " ".join(
        f"Record {index} says that the blue instrument was stored in cabinet {index % 9}."
        for index in range(96)
    )
    return {
        "short": "Complete the factual sentence: The capital of France is",
        "medium": (
            "Continue this sequence with a concise explanation: 2, 3, 5, 7, 11, 13. "
            "The rule behind the sequence is"
        ),
        "long": f"Read these records and state where the blue instrument appears most often. {long_context}\nAnswer:",
    }


def decode(tokenizer: Any, token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def score_metadata(step_scores: tuple[torch.Tensor, ...], generated: torch.Tensor) -> dict[str, list[Any]]:
    selected_logprobs: list[float] = []
    top1_top2_margins: list[float] = []
    top1_ids: list[int] = []
    top2_ids: list[int] = []
    for step, scores in enumerate(step_scores):
        row = scores.float()
        top_values, top_ids = torch.topk(row, k=2, dim=-1)
        selected = generated[:, step]
        selected_lp = torch.log_softmax(row, dim=-1).gather(1, selected[:, None]).squeeze(1)
        selected_logprobs.append(float(selected_lp.item()))
        top1_top2_margins.append(float((top_values[:, 0] - top_values[:, 1]).item()))
        top1_ids.append(int(top_ids[:, 0].item()))
        top2_ids.append(int(top_ids[:, 1].item()))
    return {
        "selected_logprobs": selected_logprobs,
        "top1_top2_margins": top1_top2_margins,
        "top1_ids": top1_ids,
        "top2_ids": top2_ids,
    }


def batched_score_metadata(
    step_scores: tuple[torch.Tensor, ...], generated: torch.Tensor, row_index: int
) -> dict[str, list[Any]]:
    row_scores = tuple(scores[row_index : row_index + 1] for scores in step_scores)
    return score_metadata(row_scores, generated[row_index : row_index + 1])


def ordinary_output(
    tokenizer: Any,
    generated: torch.Tensor,
    metadata: dict[str, list[Any]],
) -> dict[str, Any]:
    token_ids = [int(token) for token in generated.tolist()]
    return {
        "tokens": token_ids,
        "text": decode(tokenizer, token_ids),
        **metadata,
    }


@torch.inference_mode()
def run_sequential(
    model: Any,
    tokenizer: Any,
    prompt_ids: dict[str, list[int]],
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    outputs: dict[str, Any] = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    for name, ids in prompt_ids.items():
        input_ids = torch.tensor([ids], dtype=torch.long, device=model.device)
        attention_mask = torch.ones_like(input_ids)
        result = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
        )
        generated = result.sequences[0, input_ids.shape[1] :]
        metadata = score_metadata(result.scores, generated[None, :])
        outputs[name] = ordinary_output(tokenizer, generated, metadata)
    torch.cuda.synchronize()
    return {"elapsed_seconds": time.perf_counter() - start, "outputs": outputs}


@torch.inference_mode()
def run_static_batch(
    model: Any,
    tokenizer: Any,
    prompt_ids: dict[str, list[int]],
    generation_config: GenerationConfig,
) -> dict[str, Any]:
    names = list(prompt_ids)
    encoded = tokenizer.pad(
        {"input_ids": [prompt_ids[name] for name in names]},
        padding=True,
        return_tensors="pt",
    ).to(model.device)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = model.generate(
        **encoded,
        generation_config=generation_config,
        return_dict_in_generate=True,
        output_scores=True,
    )
    torch.cuda.synchronize()
    generated = result.sequences[:, encoded.input_ids.shape[1] :]
    outputs = {
        name: ordinary_output(
            tokenizer,
            generated[index],
            batched_score_metadata(result.scores, generated, index),
        )
        for index, name in enumerate(names)
    }
    return {"elapsed_seconds": time.perf_counter() - start, "outputs": outputs}


def make_cb_config(args: argparse.Namespace, force_varlen: bool) -> ContinuousBatchingConfig:
    kwargs: dict[str, Any] = {
        "max_batch_tokens": args.max_batch_tokens,
        "max_memory_percent": args.cb_memory_percent,
        "return_logprobs": True,
    }
    if force_varlen:
        kwargs["max_blocks_per_request"] = 0
    return ContinuousBatchingConfig(**kwargs)


def cb_output(tokenizer: Any, result: Any) -> dict[str, Any]:
    if result.error is not None:
        raise RuntimeError(f"continuous batching request {result.request_id} failed: {result.error}")
    token_ids = [int(token) for token in result.generated_tokens]
    return {
        "tokens": token_ids,
        "text": decode(tokenizer, token_ids),
        "selected_logprobs": [float(value) for value in result.logprobs],
        "top1_top2_margins": None,
        "top1_ids": None,
        "top2_ids": None,
        "status": result.status.name,
    }


@torch.inference_mode()
def run_generate_batch(
    model: Any,
    tokenizer: Any,
    prompt_ids: dict[str, list[int]],
    generation_config: GenerationConfig,
    args: argparse.Namespace,
    force_varlen: bool,
) -> dict[str, Any]:
    names = list(prompt_ids)
    cb_config = make_cb_config(args, force_varlen)
    torch.cuda.synchronize()
    start = time.perf_counter()
    results = model.generate_batch(
        inputs=[prompt_ids[name] for name in names],
        generation_config=generation_config,
        continuous_batching_config=cb_config,
        progress_bar=False,
        warmup=False,
    )
    torch.cuda.synchronize()
    outputs = {
        name: cb_output(tokenizer, results[f"req_{index}"])
        for index, name in enumerate(names)
    }
    return {"elapsed_seconds": time.perf_counter() - start, "outputs": outputs}


@torch.inference_mode()
def run_delayed_arrival(
    model: Any,
    tokenizer: Any,
    prompt_ids: dict[str, list[int]],
    generation_config: GenerationConfig,
    args: argparse.Namespace,
) -> dict[str, Any]:
    cb_config = make_cb_config(args, force_varlen=False)
    finished: dict[str, Any] = {}
    torch.cuda.synchronize()
    start = time.perf_counter()
    with model.continuous_batching_context_manager(
        generation_config=generation_config,
        continuous_batching_config=cb_config,
        warmup=False,
        timeout=args.timeout,
    ) as manager:
        manager.add_request(
            input_ids=prompt_ids["short"],
            request_id="early",
            max_new_tokens=args.max_new_tokens,
            streaming=True,
            record_timestamps=True,
            eos_token_id=-1,
        )
        first = manager.get_result(timeout=args.timeout)
        if first is None:
            raise TimeoutError("timed out before the early request produced its first token")
        if first.request_id != "early" or first.is_finished():
            raise RuntimeError("early request finished before the delayed request could be submitted")
        early_tokens_before_late_submit = len(first.generated_tokens)
        manager.add_request(
            input_ids=prompt_ids["long"],
            request_id="late",
            max_new_tokens=args.max_new_tokens,
            streaming=False,
            record_timestamps=True,
            eos_token_id=-1,
        )
        deadline = time.monotonic() + args.timeout
        while len(finished) < 2:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for delayed results; finished={list(finished)}")
            result = manager.get_result(timeout=min(1.0, remaining))
            if result is None:
                if not manager.is_running():
                    raise RuntimeError("continuous batching manager stopped before both requests finished")
                continue
            if result.is_finished():
                finished[result.request_id] = result
        resolved_config = {
            name: getattr(manager.continuous_batching_config, name)
            for name in (
                "block_size",
                "num_blocks",
                "max_batch_tokens",
                "max_memory_percent",
                "max_blocks_per_request",
                "use_async_batching",
                "use_cuda_graph",
                "scheduler_type",
                "return_logprobs",
            )
        }
    torch.cuda.synchronize()
    return {
        "elapsed_seconds": time.perf_counter() - start,
        "early_tokens_before_late_submit": early_tokens_before_late_submit,
        "resolved_continuous_batching_config": resolved_config,
        "outputs": {
            "short": cb_output(tokenizer, finished["early"]),
            "long": cb_output(tokenizer, finished["late"]),
        },
    }


def item_at(values: Any, index: int) -> Any:
    if values is None or index >= len(values):
        return None
    return values[index]


def compare_outputs(tokenizer: Any, left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_tokens = left["tokens"]
    right_tokens = right["tokens"]
    limit = min(len(left_tokens), len(right_tokens))
    index = next((i for i in range(limit) if left_tokens[i] != right_tokens[i]), None)
    if index is None and len(left_tokens) != len(right_tokens):
        index = limit
    divergence = None
    if index is not None:
        left_token = item_at(left_tokens, index)
        right_token = item_at(right_tokens, index)
        divergence = {
            "index": index,
            "left_token_id": left_token,
            "right_token_id": right_token,
            "left_token_text": None if left_token is None else decode(tokenizer, [left_token]),
            "right_token_text": None if right_token is None else decode(tokenizer, [right_token]),
            "left_selected_logprob": item_at(left.get("selected_logprobs"), index),
            "right_selected_logprob": item_at(right.get("selected_logprobs"), index),
            "left_top1_top2_margin": item_at(left.get("top1_top2_margins"), index),
            "right_top1_top2_margin": item_at(right.get("top1_top2_margins"), index),
        }
    return {
        "tokens_exact": left_tokens == right_tokens,
        "text_exact": left["text"] == right["text"],
        "first_divergence": divergence,
    }


def summarize(tokenizer: Any, strategies: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    within: dict[str, Any] = {}
    for strategy, trials in strategies.items():
        names = trials[0]["outputs"].keys()
        comparisons = []
        for trial_index in range(1, len(trials)):
            for name in names:
                comparisons.append(
                    {
                        "trial": trial_index,
                        "prompt": name,
                        **compare_outputs(
                            tokenizer,
                            trials[0]["outputs"][name],
                            trials[trial_index]["outputs"][name],
                        ),
                    }
                )
        within[strategy] = {
            "all_tokens_exact": all(item["tokens_exact"] for item in comparisons),
            "all_text_exact": all(item["text_exact"] for item in comparisons),
            "comparisons": comparisons,
        }

    baseline = strategies["sequential"][0]["outputs"]
    cross: dict[str, Any] = {}
    for strategy, trials in strategies.items():
        if strategy == "sequential":
            continue
        comparisons = []
        for trial_index, trial in enumerate(trials):
            for name, output in trial["outputs"].items():
                comparisons.append(
                    {
                        "trial": trial_index,
                        "prompt": name,
                        **compare_outputs(tokenizer, baseline[name], output),
                    }
                )
        cross[strategy] = {
            "all_tokens_exact_vs_sequential_trial_0": all(item["tokens_exact"] for item in comparisons),
            "all_text_exact_vs_sequential_trial_0": all(item["text_exact"] for item in comparisons),
            "comparisons": comparisons,
        }
    return {"within_strategy": within, "cross_strategy": cross}


def main() -> None:
    args = parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "1":
        raise RuntimeError("this experiment must run with CUDA_VISIBLE_DEVICES=1")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("expected exactly one visible CUDA device (physical GPU 1)")

    torch.manual_seed(0)
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=args.cache_dir)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).to("cuda:0").eval()

    prompt_text = prompts()
    prompt_ids = {
        name: tokenizer.encode(text, add_special_tokens=True)
        for name, text in prompt_text.items()
    }
    generation_config = GenerationConfig(
        do_sample=False,
        max_new_tokens=args.max_new_tokens,
        eos_token_id=-1,
        pad_token_id=tokenizer.pad_token_id,
    )

    strategies: dict[str, list[dict[str, Any]]] = {
        "sequential": [],
        "static_batch": [],
        "continuous_batching_auto_path": [],
        "continuous_batching_forced_varlen": [],
        "continuous_batching_delayed_arrival": [],
    }
    for repeat in range(args.repeats):
        print(f"repeat {repeat + 1}/{args.repeats}: sequential", flush=True)
        strategies["sequential"].append(
            run_sequential(model, tokenizer, prompt_ids, generation_config)
        )
        print(f"repeat {repeat + 1}/{args.repeats}: static batch", flush=True)
        strategies["static_batch"].append(
            run_static_batch(model, tokenizer, prompt_ids, generation_config)
        )
        print(f"repeat {repeat + 1}/{args.repeats}: continuous batching auto path", flush=True)
        strategies["continuous_batching_auto_path"].append(
            run_generate_batch(
                model,
                tokenizer,
                prompt_ids,
                generation_config,
                args,
                force_varlen=False,
            )
        )
        print(f"repeat {repeat + 1}/{args.repeats}: continuous batching forced varlen", flush=True)
        strategies["continuous_batching_forced_varlen"].append(
            run_generate_batch(
                model,
                tokenizer,
                prompt_ids,
                generation_config,
                args,
                force_varlen=True,
            )
        )
        print(f"repeat {repeat + 1}/{args.repeats}: delayed arrival", flush=True)
        strategies["continuous_batching_delayed_arrival"].append(
            run_delayed_arrival(model, tokenizer, prompt_ids, generation_config, args)
        )

    report = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "logical_device": 0,
            "gpu": torch.cuda.get_device_name(0),
            "model": args.model,
            "model_name_or_path": model.name_or_path,
            "dtype": str(model.dtype),
            "attention_implementation_outside_cb": model.config._attn_implementation,
        },
        "official_references": OFFICIAL_REFERENCES,
        "experiment": {
            "repeats": args.repeats,
            "greedy_equivalent": "do_sample=False (temperature=0, top_p=1 semantics)",
            "max_new_tokens": args.max_new_tokens,
            "eos_disabled": True,
            "max_batch_tokens": args.max_batch_tokens,
            "cb_memory_percent": args.cb_memory_percent,
            "prompt_token_lengths": {name: len(ids) for name, ids in prompt_ids.items()},
            "prompts": prompt_text,
            "cb_margin_note": (
                "Transformers continuous batching returns each selected token's logprob, not the full top-2 "
                "distribution; its top1_top2_margins are therefore null."
            ),
        },
        "strategies": strategies,
    }
    report["summary"] = summarize(tokenizer, strategies)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)
    print(f"wrote {output}", flush=True)


if __name__ == "__main__":
    main()
