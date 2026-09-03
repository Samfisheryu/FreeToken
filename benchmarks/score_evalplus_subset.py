#!/usr/bin/env python3
"""Score a submitted HumanEval+/MBPP+ subset with EvalPlus 0.3.1.

This helper is launched by ``bench_mixed_task_goodput.py`` after the measured
HTTP workload has fully drained.  The caller supplies complete Python
solutions and task ids; this process loads the fixed official datasets and
uses EvalPlus's own base+plus execution and special-oracle behavior.
"""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
from typing import Any


INPUT_SCHEMA = "freetoken.evalplus_subset_input.v1"
OUTPUT_SCHEMA = "freetoken.evalplus_subset_result.v1"
PUBLIC_CONTRACT = r"""
Input JSON contract:
  {
    "schema": "freetoken.evalplus_subset_input.v1",
    "items": [
      {
        "record_id": "non-empty caller-unique string",
        "dataset": "humaneval | mbpp",
        "task_id": "official id from that fixed dataset",
        "solution": "non-empty complete Python source string"
      }
    ]
  }
``solution`` is executable source, not a Markdown fence and not a function-body
fragment.  HumanEval ids belong to HumanEval+ v0.1.10; MBPP ids belong to MBPP+
v0.2.0.  The runner sends exactly these four item fields after removing the one
required outer ```python fence from a model response.

Output JSON contract:
  {
    "schema": "freetoken.evalplus_subset_result.v1",
    "evalplus_version": "0.3.1",
    "dataset_versions": {
      "humaneval": "v0.1.10",
      "mbpp": "v0.2.0"
    },
    "items": [
      {
        "record_id": "copied input record_id",
        "task_id": "copied official task id",
        "dataset": "humaneval | mbpp",
        "base_status": "pass | fail | timeout",
        "plus_status": "pass | fail | timeout",
        "passed": true
      }
    ]
  }
``passed`` is true only when both statuses are ``pass``.  Item order matches
input order.  On success the helper atomically writes --output and exits 0.
Invalid JSON/schema/fields/task ids, an EvalPlus version other than 0.3.1,
official-data load failures, verifier failures, or an unwritable output produce
a non-zero exit and no valid output contract.  Stdout/stderr are diagnostics,
not a JSON API.  HUMANEVAL_OVERRIDE_PATH and MBPP_OVERRIDE_PATH may point to the
fixed uncompressed official JSONL files; otherwise EvalPlus uses its own cache.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PUBLIC_CONTRACT,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def expected_outputs(dataset: str, problem: dict[str, Any]) -> dict[str, Any]:
    from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
    from evalplus.gen.util import trusted_exec

    output_not_none = (
        dataset == "mbpp" and problem["entry_point"] in MBPP_OUTPUT_NOT_NONE_TASKS
    )
    reference_code = problem["prompt"] + problem["canonical_solution"]
    base, base_time = trusted_exec(
        reference_code,
        problem["base_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )
    plus, plus_time = trusted_exec(
        reference_code,
        problem["plus_input"],
        problem["entry_point"],
        record_time=True,
        output_not_none=output_not_none,
    )
    return {
        "base": base,
        "base_time": base_time,
        "plus": plus,
        "plus_time": plus_time,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        evalplus_version = metadata.version("evalplus")
    except metadata.PackageNotFoundError as exc:
        raise RuntimeError("EvalPlus 0.3.1 is not installed in this Python environment") from exc
    if evalplus_version != "0.3.1":
        raise RuntimeError(f"EvalPlus 0.3.1 is required; found {evalplus_version}")
    payload = load_json(args.input.expanduser().resolve())
    if payload.get("schema") != INPUT_SCHEMA or not isinstance(payload.get("items"), list):
        raise ValueError("unsupported EvalPlus subset input schema")

    from evalplus.data import get_human_eval_plus, get_mbpp_plus
    from evalplus.evaluate import check_correctness

    problems = {
        "humaneval": get_human_eval_plus(version="v0.1.10"),
        "mbpp": get_mbpp_plus(version="v0.2.0"),
    }
    expected_cache: dict[tuple[str, str], dict[str, Any]] = {}
    results: list[dict[str, Any]] = []
    for item in payload["items"]:
        if not isinstance(item, dict):
            raise ValueError("EvalPlus subset item is not an object")
        record_id = item.get("record_id")
        dataset = item.get("dataset")
        task_id = item.get("task_id")
        solution = item.get("solution")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("EvalPlus subset item has no record_id")
        if dataset not in problems:
            raise ValueError(f"unsupported EvalPlus dataset: {dataset!r}")
        if not isinstance(task_id, str) or task_id not in problems[dataset]:
            raise ValueError(f"unknown EvalPlus task_id: {task_id!r}")
        if not isinstance(solution, str) or not solution.strip():
            raise ValueError(f"EvalPlus item {record_id} has no solution")
        problem = problems[dataset][task_id]
        key = (dataset, task_id)
        if key not in expected_cache:
            expected_cache[key] = expected_outputs(dataset, problem)
        checked = check_correctness(
            dataset,
            0,
            problem,
            solution,
            expected_cache[key],
            base_only=False,
            fast_check=True,
            identifier=record_id,
        )
        base_status = checked["base"][0]
        plus_status = checked["plus"][0]
        results.append(
            {
                "record_id": record_id,
                "task_id": task_id,
                "dataset": dataset,
                "base_status": base_status,
                "plus_status": plus_status,
                "passed": base_status == plus_status == "pass",
            }
        )

    output = {
        "schema": OUTPUT_SCHEMA,
        "evalplus_version": evalplus_version,
        "dataset_versions": {"humaneval": "v0.1.10", "mbpp": "v0.2.0"},
        "items": results,
    }
    target = args.output.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
