#!/usr/bin/env python3
"""Score generated patches with the pinned SWE-bench Verified v5 harness.

The helper is intentionally offline.  ``bench_long_mixed_task_goodput.py``
launches it only after the online requests have drained and the goodput
denominator is closed.  Docker pulls, patch application, tests, and report
generation therefore do not enter online time.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any


INPUT_SCHEMA = "freetoken.swebench_verified_subset_input.v2"
OUTPUT_SCHEMA = "freetoken.swebench_verified_subset_result.v2"
EVALUATOR_VERSION = "5.0.1"
EVALUATOR_REVISION = "87ab1f6ced28f75ba73ca899dc759b019310944a"
DATASET_REVISION = "78f471bf655a3137b2e8a75af1501690ec009ec3"
MODEL_NAME = "freetoken_long_goodput_v2"
PUBLIC_CONTRACT = r"""
Input JSON contract:
  {
    "schema": "freetoken.swebench_verified_subset_input.v2",
    "items": [
      {
        "record_id": "non-empty caller-unique string",
        "instance_id": "official selected SWE-bench Verified id",
        "model_patch": "non-empty single unified-diff string"
      }
    ]
  }
The item object has exactly those three fields.  Item record_id and instance_id
values are unique.  --verified-parquet must be the fixed local Parquet from
SWE-bench/SWE-bench_Verified@78f471bf655a3137b2e8a75af1501690ec009ec3.
--swebench-root must be an official git checkout whose HEAD is exactly
87ab1f6ced28f75ba73ca899dc759b019310944a (tag v5.0.1).  The helper passes each
patch to that checkout's official Docker harness; each selected dataset row's
literal `image` field is the container reference.

Output JSON contract:
  {
    "schema": "freetoken.swebench_verified_subset_result.v2",
    "evaluator": {
      "repository": "SWE-bench/SWE-bench",
      "version": "5.0.1",
      "revision": "87ab1f6ced28f75ba73ca899dc759b019310944a"
    },
    "dataset": {
      "repository": "SWE-bench/SWE-bench_Verified",
      "revision": "78f471bf655a3137b2e8a75af1501690ec009ec3"
    },
    "items": [
      {
        "record_id": "copied input record_id",
        "instance_id": "copied official instance id",
        "status": "resolved | unresolved | evaluation_error",
        "resolved": true,
        "report_available": true
      }
    ]
  }
Item order matches input order. `resolved` is true only for status=resolved.
An official per-instance error or absent report becomes evaluation_error and
resolved=false.  Invalid JSON/schema/fields, duplicate or unknown ids, a wrong
git revision/version, invalid fixed Parquet, harness launch failure, non-zero
harness exit, or unwritable output exits non-zero and produces no valid output
contract.  On success --output is atomically replaced and the process exits 0.
Stdout/stderr are diagnostics, not a JSON API.
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=PUBLIC_CONTRACT,
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verified-parquet", type=Path, required=True)
    parser.add_argument("--swebench-root", type=Path, required=True)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--instance-timeout", type=int, default=1800)
    args = parser.parse_args(argv)
    if args.max_workers < 1:
        parser.error("--max-workers must be positive")
    if args.instance_timeout < 1:
        parser.error("--instance-timeout must be positive")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def evaluator_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def validate_evaluator(root: Path) -> None:
    if not root.is_dir():
        raise FileNotFoundError(f"SWE-bench checkout not found: {root}")
    revision = evaluator_head(root)
    if revision != EVALUATOR_REVISION:
        raise RuntimeError(
            f"SWE-bench checkout must be {EVALUATOR_REVISION}; found {revision}"
        )
    sys.path.insert(0, str(root))
    import swebench

    version = getattr(swebench, "__version__", None)
    if version != EVALUATOR_VERSION:
        raise RuntimeError(f"SWE-bench {EVALUATOR_VERSION} is required; found {version!r}")


def validate_items(payload: dict[str, Any]) -> list[dict[str, str]]:
    if set(payload) != {"schema", "items"} or payload.get("schema") != INPUT_SCHEMA:
        raise ValueError("unsupported SWE-bench subset input schema")
    raw_items = payload.get("items")
    if not isinstance(raw_items, list):
        raise ValueError("input items must be a list")
    items: list[dict[str, str]] = []
    record_ids: set[str] = set()
    instance_ids: set[str] = set()
    for index, item in enumerate(raw_items):
        if not isinstance(item, dict) or set(item) != {"record_id", "instance_id", "model_patch"}:
            raise ValueError(f"item {index} must have exactly record_id/instance_id/model_patch")
        record_id = item["record_id"]
        instance_id = item["instance_id"]
        patch = item["model_patch"]
        if not isinstance(record_id, str) or not record_id:
            raise ValueError(f"item {index} has invalid record_id")
        if not isinstance(instance_id, str) or not instance_id:
            raise ValueError(f"item {index} has invalid instance_id")
        if not isinstance(patch, str) or not patch.strip():
            raise ValueError(f"item {index} has empty model_patch")
        if record_id in record_ids or instance_id in instance_ids:
            raise ValueError(f"item {index} duplicates record_id or instance_id")
        record_ids.add(record_id)
        instance_ids.add(instance_id)
        items.append(
            {"record_id": record_id, "instance_id": instance_id, "model_patch": patch}
        )
    return items


def validate_dataset(path: Path, wanted: set[str]) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"fixed Verified Parquet not found: {path}")
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("SWE-bench subset scoring requires pyarrow") from exc
    table = parquet.read_table(
        path,
        columns=[
            "instance_id",
            "image",
            "repo",
            "version",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
            "log_parser",
            "eval_type",
            "eval_script",
        ],
    )
    if table.num_rows != 500:
        raise ValueError(f"fixed Verified Parquet must contain 500 rows; found {table.num_rows}")
    rows = table.to_pylist()
    by_id = {row["instance_id"]: row for row in rows}
    if len(by_id) != 500:
        raise ValueError("fixed Verified Parquet instance ids must be unique")
    missing = sorted(wanted - set(by_id))
    if missing:
        raise ValueError("unknown selected Verified instance ids: " + ", ".join(missing))
    for instance_id in wanted:
        row = by_id[instance_id]
        required = (
            "image",
            "repo",
            "version",
            "log_parser",
            "eval_type",
            "eval_script",
        )
        if any(not isinstance(row[field], str) or not row[field] for field in required):
            raise ValueError(f"Verified evaluator metadata is incomplete for {instance_id}")
        if not row["image"].startswith("swebench/sweb.eval.x86_64.") or not row["image"].endswith(":latest"):
            raise ValueError(f"unexpected fixed container reference for {instance_id}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().absolute()
    verified_path = args.verified_parquet.expanduser().resolve()
    swebench_root = args.swebench_root.expanduser().resolve()
    payload = load_json(input_path)
    items = validate_items(payload)
    validate_evaluator(swebench_root)
    validate_dataset(verified_path, {item["instance_id"] for item in items})

    if not items:
        result_items: list[dict[str, Any]] = []
    else:
        with tempfile.TemporaryDirectory(prefix="freetoken-swebench-v2-") as temporary:
            root = Path(temporary)
            predictions_path = root / "predictions.json"
            predictions = [
                {
                    "instance_id": item["instance_id"],
                    "model_patch": item["model_patch"],
                    "model_name_or_path": MODEL_NAME,
                }
                for item in items
            ]
            predictions_path.write_text(
                json.dumps(predictions, ensure_ascii=False) + "\n", encoding="utf-8"
            )
            run_id = "long_goodput_v2"
            environment = os.environ.copy()
            existing = environment.get("PYTHONPATH")
            environment["PYTHONPATH"] = os.pathsep.join(
                [str(swebench_root), *([existing] if existing else [])]
            )
            command = [
                sys.executable,
                "-m",
                "swebench.harness.run_evaluation",
                "--dataset_name",
                str(verified_path),
                "--split",
                "test",
                "--instance_ids",
                *[item["instance_id"] for item in items],
                "--predictions_path",
                str(predictions_path),
                "--max_workers",
                str(args.max_workers),
                "--timeout",
                str(args.instance_timeout),
                "--run_id",
                run_id,
                "--report_dir",
                str(root / "reports"),
            ]
            completed = subprocess.run(
                command,
                cwd=root,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"official SWE-bench harness exited {completed.returncode}; output tail:\n"
                    + completed.stdout[-12000:]
                )
            result_items = []
            for item in items:
                report_path = (
                    root
                    / "logs"
                    / "run_evaluation"
                    / run_id
                    / MODEL_NAME
                    / item["instance_id"]
                    / "report.json"
                )
                available = report_path.is_file()
                resolved = False
                if available:
                    report = load_json(report_path)
                    instance_report = report.get(item["instance_id"])
                    resolved = bool(
                        isinstance(instance_report, dict)
                        and instance_report.get("resolved") is True
                    )
                result_items.append(
                    {
                        "record_id": item["record_id"],
                        "instance_id": item["instance_id"],
                        "status": (
                            "resolved" if resolved else "unresolved" if available else "evaluation_error"
                        ),
                        "resolved": resolved,
                        "report_available": available,
                    }
                )

    result = {
        "schema": OUTPUT_SCHEMA,
        "evaluator": {
            "repository": "SWE-bench/SWE-bench",
            "version": EVALUATOR_VERSION,
            "revision": EVALUATOR_REVISION,
        },
        "dataset": {
            "repository": "SWE-bench/SWE-bench_Verified",
            "revision": DATASET_REVISION,
        },
        "items": result_items,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_output.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary_output.replace(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
