#!/usr/bin/env python3
"""Score a fixed LiveCodeBench v5/v6 subset with the official hidden tests.

The scorer is an offline process used only after serving measurement has fully
drained.  It imports the official LiveCodeBench evaluator from the exact pinned
checkout and applies its generic-chat code extraction plus code-generation test
runner.  Judge wall time is therefore outside serving throughput.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


INPUT_SCHEMA = "freetoken.livecodebench_subset_input.v1"
OUTPUT_SCHEMA = "freetoken.livecodebench_subset_result.v1"
DATASET_REPOSITORY = "livecodebench/code_generation_lite"
DATASET_REVISION = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
RUNNER_REVISION = "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24"
RUNNER_VERSION = "0.1.0"
DEFAULT_SOURCE_CACHE = Path("/data2/servebig-envs/aime_lcb_sources_v1")
SOURCE_FILES = {
    "lcb-test5.jsonl": {"increment": "v5", "expected_rows": 167},
    "lcb-test6.jsonl": {"increment": "v6", "expected_rows": 175},
}
LCB_FIELDS = {
    "question_title",
    "question_content",
    "platform",
    "question_id",
    "contest_id",
    "contest_date",
    "starter_code",
    "difficulty",
    "public_test_cases",
    "private_test_cases",
    "metadata",
}
PUBLIC_CONTRACT = r"""
Input JSON schema ``freetoken.livecodebench_subset_input.v1``:
  {
    "schema": "freetoken.livecodebench_subset_input.v1",
    "dataset_revision": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    "runner_revision": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
    "items": [
      {
        "record_id": "unique non-empty caller id",
        "question_id": "official LiveCodeBench question_id",
        "source_file": "lcb-test5.jsonl | lcb-test6.jsonl",
        "model_output": "complete assistant content, possibly including reasoning"
      }
    ]
  }

The scorer uses LiveCodeBench's OpenAI-chat extraction rule: the code between
the final pair of Markdown fences is graded.  Missing/empty fenced code is an
ordinary ``no_code`` failure and is not sent to the execution harness.

Output JSON schema ``freetoken.livecodebench_subset_result.v1``:
  {
    "schema": "freetoken.livecodebench_subset_result.v1",
    "dataset": {"repository": "...", "revision": "..."},
    "runner": {"revision": "...", "package_version": "0.1.0"},
    "items": [
      {
        "record_id": "copied caller id",
        "question_id": "copied official id",
        "source_file": "copied fixed source file",
        "extraction_status": "code | no_code",
        "status": "pass | fail | timeout | no_code",
        "official_error_code": "integer or null",
        "passed": true
      }
    ]
  }

Output item order exactly matches input order.  A model compile error, runtime
error, wrong answer, or test timeout is a valid failed item.  Bad input, an
unknown/duplicate id, missing or changed source data, a checkout not at the
pinned runner commit, import/process/evaluator infrastructure failure, or an
unwritable/existing output path exits non-zero without publishing a new output.
Success atomically publishes --output and exits zero.  Stdout contains counts
only and never includes model output or hidden tests.
"""


_WORKER_ROOT: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PUBLIC_CONTRACT,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lcb-root",
        type=Path,
        required=True,
        help=f"official LiveCodeBench checkout at {RUNNER_REVISION}",
    )
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--test-timeout", type=int, default=6)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.test_timeout < 1:
        parser.error("--test-timeout must be positive")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def git_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_lcb_root(root: Path) -> None:
    if not (root / "lcb_runner" / "evaluation" / "compute_code_generation_metrics.py").is_file():
        raise FileNotFoundError(f"not a LiveCodeBench checkout: {root}")
    observed = git_commit(root)
    if observed != RUNNER_REVISION:
        raise ValueError(
            f"LiveCodeBench checkout must be {RUNNER_REVISION}; found {observed}"
        )


def validate_input(payload: dict[str, Any]) -> list[dict[str, str]]:
    if payload.get("schema") != INPUT_SCHEMA:
        raise ValueError("unsupported LiveCodeBench scorer input schema")
    if payload.get("dataset_revision") != DATASET_REVISION:
        raise ValueError("input dataset_revision differs from the fixed contract")
    if payload.get("runner_revision") != RUNNER_REVISION:
        raise ValueError("input runner_revision differs from the fixed contract")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("input items must be a list")
    items: list[dict[str, str]] = []
    record_ids: set[str] = set()
    question_ids: set[str] = set()
    required = {"record_id", "question_id", "source_file", "model_output"}
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict) or set(item) != required:
            raise ValueError(f"item {index} fields differ from the scorer contract")
        if not all(isinstance(item[field], str) for field in required):
            raise ValueError(f"item {index} fields must all be strings")
        record_id = item["record_id"]
        question_id = item["question_id"]
        source_file = item["source_file"]
        if not record_id or record_id in record_ids:
            raise ValueError(f"item {index} has empty/duplicate record_id")
        if not question_id or question_id in question_ids:
            raise ValueError(f"item {index} has empty/duplicate question_id")
        if source_file not in SOURCE_FILES:
            raise ValueError(f"item {index} has unsupported source_file {source_file!r}")
        record_ids.add(record_id)
        question_ids.add(question_id)
        items.append({field: item[field] for field in required})
    return items


def worker_init(root: str) -> None:
    global _WORKER_ROOT
    _WORKER_ROOT = root
    if root not in sys.path:
        sys.path.insert(0, root)


def official_grade(sample: dict[str, str], code: str, timeout: int) -> dict[str, Any]:
    if _WORKER_ROOT is None:
        raise RuntimeError("LiveCodeBench worker was not initialized")
    from lcb_runner.evaluation.compute_code_generation_metrics import (
        evaluate_generations_by_problem,
    )

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        results, metadata_rows = evaluate_generations_by_problem(
            ([code], sample, False, timeout)
        )
    if (
        not isinstance(results, list)
        or len(results) != 1
        or not isinstance(results[0], list)
        or not isinstance(metadata_rows, list)
        or len(metadata_rows) != 1
        or not isinstance(metadata_rows[0], dict)
    ):
        raise RuntimeError("official LiveCodeBench evaluator returned an invalid result")
    per_test = results[0]
    metadata = metadata_rows[0]
    passed = bool(per_test) and all(value == True for value in per_test)  # noqa: E712
    error_code = metadata.get("error_code")
    if error_code is not None and not isinstance(error_code, int):
        error_code = None
    if error_code == -5:
        raise RuntimeError("official LiveCodeBench evaluator reported TestRunnerError")
    status = "pass" if passed else ("timeout" if error_code == -3 else "fail")
    return {
        "status": status,
        "official_error_code": error_code,
        "passed": passed,
    }


def import_official_types(root: Path) -> tuple[Any, Any, Any]:
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    from lcb_runner.benchmarks.code_generation import CodeGenerationProblem
    from lcb_runner.lm_styles import LMStyle
    from lcb_runner.utils.extraction_utils import extract_code

    return CodeGenerationProblem, LMStyle, extract_code


def complete_futures(
    pending: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]],
    results: dict[str, dict[str, Any]],
    *, wait_all: bool,
) -> None:
    if not pending:
        return
    done, _ = concurrent.futures.wait(
        pending,
        return_when=(
            concurrent.futures.ALL_COMPLETED
            if wait_all
            else concurrent.futures.FIRST_COMPLETED
        ),
    )
    for future in done:
        base = pending.pop(future)
        verdict = future.result()
        results[base["record_id"]] = {**base, **verdict}


def score(
    *, items: list[dict[str, str]], root: Path, source_cache: Path, workers: int, timeout: int
) -> list[dict[str, Any]]:
    CodeGenerationProblem, LMStyle, extract_code = import_official_types(root)
    by_file: dict[str, dict[str, dict[str, str]]] = {name: {} for name in SOURCE_FILES}
    results: dict[str, dict[str, Any]] = {}
    for item in items:
        code = extract_code(item["model_output"], LMStyle.OpenAIChat).strip()
        base = {
            "record_id": item["record_id"],
            "question_id": item["question_id"],
            "source_file": item["source_file"],
        }
        if not code:
            results[item["record_id"]] = {
                **base,
                "extraction_status": "no_code",
                "status": "no_code",
                "official_error_code": None,
                "passed": False,
            }
        else:
            by_file[item["source_file"]][item["question_id"]] = {
                **item,
                "extracted_code": code,
            }

    pending: dict[concurrent.futures.Future[dict[str, Any]], dict[str, Any]] = {}
    found: set[str] = set()
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=workers,
        initializer=worker_init,
        initargs=(str(root),),
    ) as executor:
        for source_file, source_contract in SOURCE_FILES.items():
            targets = by_file[source_file]
            path = source_cache / source_file
            if targets and not path.is_file():
                raise FileNotFoundError(path)
            if not targets:
                continue
            row_count = 0
            with path.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    row_count += 1
                    row = json.loads(line)
                    if not isinstance(row, dict) or set(row) != LCB_FIELDS:
                        raise ValueError(f"{path}:{line_number} fields changed")
                    question_id = row.get("question_id")
                    if question_id not in targets:
                        continue
                    if question_id in found:
                        raise ValueError(f"duplicate selected question_id in sources: {question_id}")
                    found.add(question_id)
                    item = targets[question_id]
                    problem = CodeGenerationProblem(**row)
                    sample = problem.get_evaluation_sample()
                    base = {
                        "record_id": item["record_id"],
                        "question_id": question_id,
                        "source_file": source_file,
                        "extraction_status": "code",
                    }
                    future = executor.submit(
                        official_grade, sample, item["extracted_code"], timeout
                    )
                    pending[future] = base
                    if len(pending) >= workers * 2:
                        complete_futures(pending, results, wait_all=False)
            if row_count != source_contract["expected_rows"]:
                raise ValueError(
                    f"{path} has {row_count} rows; expected {source_contract['expected_rows']}"
                )
        complete_futures(pending, results, wait_all=True)

    expected_found = {
        item["question_id"]
        for item in items
        if item["record_id"] not in results
        or results[item["record_id"]]["extraction_status"] == "code"
    }
    if found != expected_found:
        missing = sorted(expected_found.difference(found))
        raise ValueError(f"official source data omitted selected question ids: {missing}")
    if set(results) != {item["record_id"] for item in items}:
        raise RuntimeError("scorer did not produce exactly one result per input item")
    return [results[item["record_id"]] for item in items]


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing scorer output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    root = args.lcb_root.expanduser().resolve()
    source_cache = args.source_cache.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to replace existing scorer output: {output_path}")
    validate_lcb_root(root)
    payload = load_json(input_path)
    items = validate_input(payload)
    scored = score(
        items=items,
        root=root,
        source_cache=source_cache,
        workers=args.workers,
        timeout=args.test_timeout,
    )
    output = {
        "schema": OUTPUT_SCHEMA,
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "revision": DATASET_REVISION,
        },
        "runner": {
            "repository": "https://github.com/LiveCodeBench/LiveCodeBench.git",
            "revision": RUNNER_REVISION,
            "package_version": RUNNER_VERSION,
            "scenario": "codegeneration",
            "metric": "pass@1",
            "test_timeout_seconds": args.test_timeout,
            "workers": args.workers,
        },
        "items": scored,
    }
    write_json_atomic(output_path, output)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "item_count": len(scored),
                "passed": sum(bool(item["passed"]) for item in scored),
                "model_output_in_stdout": False,
                "hidden_tests_in_stdout": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
