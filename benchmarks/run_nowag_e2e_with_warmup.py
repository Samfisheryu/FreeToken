#!/usr/bin/env python3
"""Run the NoWAG AIME driver after an unmeasured concurrent warm-up batch."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import ModuleType


def load_driver(path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location("nowag_aime_driver", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load evaluation driver: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-script", type=Path, required=True)
    parser.add_argument("--warmup-requests", type=int, default=8)
    parser.add_argument("--warmup-max-tokens", type=int, default=8)
    wrapper_args, driver_args = parser.parse_known_args()
    if driver_args[:1] == ["--"]:
        driver_args = driver_args[1:]
    if wrapper_args.warmup_requests <= 0:
        raise ValueError("--warmup-requests must be positive")
    if wrapper_args.warmup_max_tokens <= 0:
        raise ValueError("--warmup-max-tokens must be positive")

    driver = load_driver(wrapper_args.eval_script.resolve())
    run_formal_batch = driver.run_pending_concurrent

    def run_after_warmup(
        args,
        variant,
        origin,
        model_id,
        rows,
        pending,
        variant_report,
        report,
        output_path,
    ):
        count = min(wrapper_args.warmup_requests, len(pending))
        warmup_rows = []
        for index, row in enumerate(pending[:count]):
            warmup_row = dict(row)
            # Diverge before the real problem so the formal request cannot reuse
            # the complete warm-up prompt through the radix cache.
            warmup_row["problem"] = f"Kernel warm-up request {index}.\n{row['problem']}"
            warmup_rows.append(warmup_row)
        warmup_args = copy.copy(args)
        warmup_args.max_tokens = wrapper_args.warmup_max_tokens
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=count) as executor:
            futures = [
                executor.submit(driver._stream_one, warmup_args, origin, model_id, row)
                for row in warmup_rows
            ]
            for future in futures:
                future.result()
        warmup_wall_seconds = time.perf_counter() - started
        variant_report["warmup"] = {
            "counted_in_performance": False,
            "requests": count,
            "max_tokens": warmup_args.max_tokens,
            "wall_seconds": warmup_wall_seconds,
            "prompt": "selected problem with a distinct leading warm-up line",
        }
        driver.write_report(output_path, report)
        print(
            f"[{variant}] unmeasured warm-up complete: requests={count} "
            f"max_tokens={warmup_args.max_tokens} "
            f"wall_seconds={warmup_wall_seconds:.3f}",
            flush=True,
        )
        return run_formal_batch(
            args,
            variant,
            origin,
            model_id,
            rows,
            pending,
            variant_report,
            report,
            output_path,
        )

    driver.run_pending_concurrent = run_after_warmup
    return driver.main(driver_args)


if __name__ == "__main__":
    raise SystemExit(main())
