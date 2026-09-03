#!/usr/bin/env python3
"""Build the frozen 50-task AIME25 + LiveCodeBench hard-mixed workload.

The builder caches fixed-revision official source files, validates their full
public schema and frozen counts, then atomically publishes disjoint warmup,
development, and final JSONL streams.  LiveCodeBench private tests remain only
in the external source cache; they are never copied into prompts or workload
rows.  ``--plan-only`` performs no download and no filesystem write.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import urllib.request
from typing import Any, Iterable, Iterator


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "workloads" / "aime25_lcb_goodput_v1.json"
DEFAULT_SOURCE_CACHE = Path("/data2/servebig-envs/aime_lcb_sources_v1")
DEFAULT_OUTPUT = Path(
    "/data1/lmcache_kv/goodput_campaign/aime25_lcb50_familythink_v1"
)
SPEC_SCHEMA = "freetoken.aime_lcb50_familythink_goodput_spec.v1"
MANIFEST_SCHEMA = "freetoken.aime_lcb50_familythink_goodput_manifest.v1"
TASK_SCHEMA = "freetoken.aime_lcb50_familythink_goodput_task.v1"
FORBIDDEN_SCHEMA = "freetoken.aime_lcb50_familythink_training_forbidden_texts.v1"
FAMILIES = ("aime", "code")
THINKING_BY_FAMILY = {"aime": True, "code": False}
SPLITS = ("warmup", "dev", "final")
BOXED_INTEGER_RE = re.compile(r"\\boxed\s*\{\s*([+-]?\d+)\s*\}")
INTEGER_RE = re.compile(r"[+-]?\d+\Z")
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
FROZEN_REVISIONS = {
    "aime25": "563bb8404243c5f09de6ec262f2db674fe5bce9b",
    "aime24": "83a7f387baaa524a8bda0022eac0541582297103",
    "amc23": "80815d37005feb82cd7f8fbc6901d5d3eff43057",
    "livecodebench": "0fe84c3912ea0c4d4a78037083943e8f0c4dd505",
    "evaluator": "28fef95ea8c9f7a547c8329f2cd3d32b92c1fa24",
}
FROZEN_SOURCES = {
    "aime25": {
        "repository": "math-ai/aime25",
        "revision": FROZEN_REVISIONS["aime25"],
        "file": "test.jsonl",
        "cache_file": "aime25-test.jsonl",
        "url": (
            "https://huggingface.co/datasets/math-ai/aime25/resolve/"
            f"{FROZEN_REVISIONS['aime25']}/test.jsonl"
        ),
        "format": "jsonl",
        "expected_rows": 30,
        "expected_fields": ["answer", "id", "problem"],
        "license": "Apache-2.0",
    },
    "aime24": {
        "repository": "math-ai/aime24",
        "revision": FROZEN_REVISIONS["aime24"],
        "file": "test-00000-of-00001.parquet",
        "cache_file": "aime24-test.parquet",
        "url": (
            "https://huggingface.co/datasets/math-ai/aime24/resolve/"
            f"{FROZEN_REVISIONS['aime24']}/test-00000-of-00001.parquet"
        ),
        "format": "parquet",
        "expected_rows": 30,
        "expected_fields": ["id", "problem", "solution", "url"],
        "license": "Apache-2.0",
    },
    "amc23": {
        "repository": "math-ai/amc23",
        "revision": FROZEN_REVISIONS["amc23"],
        "file": "test-00000-of-00001.parquet",
        "cache_file": "amc23-test.parquet",
        "url": (
            "https://huggingface.co/datasets/math-ai/amc23/resolve/"
            f"{FROZEN_REVISIONS['amc23']}/test-00000-of-00001.parquet"
        ),
        "format": "parquet",
        "expected_rows": 40,
        "expected_fields": ["answer", "id", "question", "url"],
        "license": "Apache-2.0",
    },
    "livecodebench": {
        "repository": "livecodebench/code_generation_lite",
        "revision": FROZEN_REVISIONS["livecodebench"],
        "format": "jsonl",
        "expected_fields": [
            "contest_date",
            "contest_id",
            "difficulty",
            "metadata",
            "platform",
            "private_test_cases",
            "public_test_cases",
            "question_content",
            "question_id",
            "question_title",
            "starter_code",
        ],
        "increments": {
            "v4": {
                "file": "test4.jsonl",
                "cache_file": "lcb-test4.jsonl",
                "url": (
                    "https://huggingface.co/datasets/livecodebench/"
                    "code_generation_lite/resolve/"
                    f"{FROZEN_REVISIONS['livecodebench']}/test4.jsonl"
                ),
                "expected_rows": 101,
            },
            "v5": {
                "file": "test5.jsonl",
                "cache_file": "lcb-test5.jsonl",
                "url": (
                    "https://huggingface.co/datasets/livecodebench/"
                    "code_generation_lite/resolve/"
                    f"{FROZEN_REVISIONS['livecodebench']}/test5.jsonl"
                ),
                "expected_rows": 167,
                "expected_difficulty_counts": {"easy": 41, "medium": 52, "hard": 74},
            },
            "v6": {
                "file": "test6.jsonl",
                "cache_file": "lcb-test6.jsonl",
                "url": (
                    "https://huggingface.co/datasets/livecodebench/"
                    "code_generation_lite/resolve/"
                    f"{FROZEN_REVISIONS['livecodebench']}/test6.jsonl"
                ),
                "expected_rows": 175,
                "expected_difficulty_counts": {"easy": 43, "medium": 52, "hard": 80},
            },
        },
        "license": "CC",
        "evaluator": {
            "repository": "https://github.com/LiveCodeBench/LiveCodeBench.git",
            "revision": FROZEN_REVISIONS["evaluator"],
            "package_version": "0.1.0",
            "scenario": "codegeneration",
            "metric": "pass@1",
            "default_test_timeout_seconds": 6,
        },
    },
}
FROZEN_STREAMS = {
    "warmup": {
        "aime_source": "amc23",
        "aime_count": 1,
        "aime_selection": "first row in frozen source order",
        "code_increment": "v4",
        "code_difficulties": ["medium", "hard"],
        "code_count": 1,
        "code_selection": "first medium-or-hard row in frozen source order",
        "total": 2,
    },
    "dev": {
        "aime_source": "aime24",
        "aime_count": 30,
        "code_increment": "v5",
        "code_difficulties": ["medium", "hard"],
        "code_count": 126,
        "code_difficulty_counts": {"medium": 52, "hard": 74},
        "total": 156,
    },
    "final": {
        "aime_source": "aime25",
        "aime_count": 30,
        "code_increment": "v6",
        "code_difficulties": ["medium", "hard"],
        "code_count": 20,
        "code_difficulty_counts": {"medium": 10, "hard": 10},
        "code_selection": (
            "first ten medium rows and first ten hard rows, independently, "
            "in frozen source order"
        ),
        "total": 50,
    },
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--download-timeout", type=float, default=120.0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print frozen sources/counts/outputs without downloading or writing",
    )
    args = parser.parse_args(argv)
    if args.download_timeout <= 0:
        parser.error("--download-timeout must be positive")
    args.spec = args.spec.expanduser().resolve()
    try:
        args.frozen_spec = load_spec(args.spec)
    except (OSError, ValueError) as exc:
        parser.error(f"invalid frozen workload spec: {exc}")
    return args


def load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict) or value.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"unsupported hard-mixed workload spec: {path}")
    validate_frozen_spec(value)
    return value


def validate_frozen_spec(spec: dict[str, Any]) -> None:
    sources = spec.get("sources")
    streams = spec.get("streams")
    traffic = spec.get("traffic")
    policy = spec.get("request_policy")
    if not all(isinstance(value, dict) for value in (sources, streams, traffic, policy)):
        raise ValueError("spec must contain sources, streams, traffic, and request_policy objects")
    if sources != FROZEN_SOURCES:
        raise ValueError(
            "sources differs from the frozen identifiers, revisions, schemas, or counts"
        )
    if streams != FROZEN_STREAMS:
        raise ValueError("streams differs from the frozen sources, selections, or counts")
    if (
        traffic.get("user_count"),
        traffic.get("think_time_seconds"),
        traffic.get("one_request_in_flight_per_user"),
        traffic.get("fixed_queue_once"),
        traffic.get("queue_wrap"),
        traffic.get("dev_submission_window_seconds"),
        traffic.get("final_submission_window_seconds"),
        traffic.get("warmup_task_count"),
        traffic.get("warmup_drain_required"),
    ) != (20, 0, True, True, False, 300, None, 2, True):
        raise ValueError("traffic differs from the frozen 20-user closed-loop contract")
    if "thinking" in policy or (
        policy.get("thinking_by_family"),
        policy.get("n"),
        policy.get("temperature"),
        policy.get("top_p"),
        policy.get("top_k"),
        policy.get("max_running_requests"),
        policy.get("max_sequence_length"),
        policy.get("max_new_tokens"),
    ) != (
        THINKING_BY_FAMILY,
        1,
        0.0,
        1.0,
        -1,
        20,
        65536,
        {
            "aime": {"default": 49152, "minimum": 1, "maximum": 49152},
            "code": {"default": 32768, "minimum": 1, "maximum": 32768},
        },
    ):
        raise ValueError("request_policy family thinking, decoding, or sequence length changed")


def plan_payload(
    spec_path: Path, spec: dict[str, Any], source_cache: Path, output: Path
) -> dict[str, Any]:
    sources = spec["sources"]
    planned_sources = []
    for name in ("aime25", "aime24", "amc23"):
        row = sources[name]
        planned_sources.append(
            {
                "name": name,
                "repository": row["repository"],
                "revision": row["revision"],
                "file": row["file"],
                "cache_path": str(source_cache / row["cache_file"]),
                "expected_rows": row["expected_rows"],
            }
        )
    lcb = sources["livecodebench"]
    for increment in ("v4", "v5", "v6"):
        row = lcb["increments"][increment]
        planned_sources.append(
            {
                "name": f"livecodebench-{increment}",
                "repository": lcb["repository"],
                "revision": lcb["revision"],
                "file": row["file"],
                "cache_path": str(source_cache / row["cache_file"]),
                "expected_rows": row["expected_rows"],
            }
        )
    return {
        "plan_only": True,
        "schema": MANIFEST_SCHEMA,
        "spec": str(spec_path),
        "source_cache": str(source_cache),
        "output_dir": str(output),
        "sources": planned_sources,
        "streams": spec["streams"],
        "request_policy": spec["request_policy"],
        "published_files": [
            "manifest.json",
            "warmup.jsonl",
            "dev.jsonl",
            "final.jsonl",
            "training_forbidden_texts.jsonl",
        ],
        "downloads_performed": False,
        "writes_performed": False,
    }


def cache_source(cache: Path, row: dict[str, Any], timeout: float) -> Path:
    target = cache / str(row["cache_file"])
    if target.is_file():
        return target
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / f".{target.name}.{os.getpid()}.partial"
    if temporary.exists():
        temporary.unlink()
    request = urllib.request.Request(
        str(row["url"]), headers={"User-Agent": "FreeToken-aime-lcb-goodput/1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            with temporary.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def read_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for ordinal, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{ordinal + 1} is not a JSON object")
            yield ordinal, row


def read_parquet(
    path: Path,
    expected_fields: set[str],
    loaded_fields: set[str] | None = None,
) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("building AIME/AMC sources requires pyarrow") from exc
    parquet_file = parquet.ParquetFile(path)
    if set(parquet_file.schema_arrow.names) != expected_fields:
        raise ValueError(
            f"{path} fields changed: "
            f"{sorted(parquet_file.schema_arrow.names)} != {sorted(expected_fields)}"
        )
    selected_fields = loaded_fields if loaded_fields is not None else expected_fields
    if not selected_fields <= expected_fields:
        raise ValueError(f"{path} requested fields outside its frozen schema")
    table = parquet_file.read(columns=sorted(selected_fields))
    rows = table.to_pylist()
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} did not decode to object rows")
    return rows


def parse_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} is not an integer answer")
    if isinstance(value, int):
        return value
    text = str(value).strip()
    if INTEGER_RE.fullmatch(text):
        return int(text)
    matches = BOXED_INTEGER_RE.findall(text)
    if matches:
        return int(matches[-1])
    raise ValueError(f"{label} is not an integer or boxed integer")


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split()).strip()


def math_user_prompt(problem: str, spec: dict[str, Any]) -> str:
    return problem.rstrip() + "\n\n" + spec["request_policy"]["aime_answer_instruction"]


def code_user_prompt(question: str, starter_code: str) -> str:
    prompt = "### Question:\n" + question + "\n\n"
    if starter_code:
        prompt += (
            "### Format: You will use the following starter code to write the solution "
            "to the problem and enclose your code within delimiters.\n"
            f"```python\n{starter_code}\n```\n\n"
        )
    else:
        prompt += (
            "### Format: Read the inputs from stdin solve the problem and write the answer "
            "to stdout (do not directly test on the sample inputs). Enclose your code within "
            "delimiters as follows. Ensure that when the python program runs, it reads the "
            "inputs, runs the algorithm and writes output to STDOUT.\n"
            "```python\n# YOUR CODE HERE\n```\n\n"
        )
    return prompt + "### Answer: (use the provided format with backticks)\n\n"


def load_math_source(
    name: str, path: Path, source: dict[str, Any], spec: dict[str, Any]
) -> list[dict[str, Any]]:
    expected_fields = set(source["expected_fields"])
    if source["format"] == "jsonl":
        rows = []
        for ordinal, row in read_jsonl(path):
            if set(row) != expected_fields:
                raise ValueError(f"{name} row {ordinal} fields changed")
            rows.append(row)
    else:
        loaded_fields = expected_fields - {"answer"} if name == "amc23" else expected_fields
        rows = read_parquet(path, expected_fields, loaded_fields)
    if len(rows) != int(source["expected_rows"]):
        raise ValueError(f"{name} has {len(rows)} rows; expected {source['expected_rows']}")
    tasks: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for ordinal, row in enumerate(rows):
        problem_field = "question" if name == "amc23" else "problem"
        problem = row.get(problem_field)
        if not isinstance(problem, str) or not problem.strip():
            raise ValueError(f"{name} row {ordinal} has no problem text")
        source_id = str(row.get("id"))
        if not source_id or source_id in seen_ids:
            raise ValueError(f"{name} row {ordinal} has invalid/duplicate id {source_id!r}")
        seen_ids.add(source_id)
        task = {
            "source_name": name,
            "source_id": source_id,
            "source_ordinal": ordinal,
            "problem_text": problem.strip(),
            "task_text": math_user_prompt(problem, spec),
        }
        if name != "amc23":
            answer_field = "solution" if name == "aime24" else "answer"
            answer = parse_integer(row.get(answer_field), f"{name} row {ordinal} answer")
            if not 0 <= answer <= 999:
                raise ValueError(f"{name} row {ordinal} answer is outside 000..999")
            task["answer"] = answer
        tasks.append(task)
    return tasks


def load_lcb_increment(
    increment: str, path: Path, lcb: dict[str, Any]
) -> tuple[list[dict[str, Any]], Counter[str]]:
    source = lcb["increments"][increment]
    expected_fields = set(lcb["expected_fields"])
    if expected_fields != LCB_FIELDS:
        raise ValueError("LiveCodeBench expected_fields differs from the v1 contract")
    candidates: list[dict[str, Any]] = []
    difficulty_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    for ordinal, row in read_jsonl(path):
        if set(row) != expected_fields:
            raise ValueError(f"LiveCodeBench {increment} row {ordinal} fields changed")
        question_id = row.get("question_id")
        question = row.get("question_content")
        starter = row.get("starter_code")
        difficulty = row.get("difficulty")
        if not isinstance(question_id, str) or not question_id or question_id in seen_ids:
            raise ValueError(f"LiveCodeBench {increment} row {ordinal} has invalid/duplicate id")
        if not isinstance(question, str) or not question.strip() or not isinstance(starter, str):
            raise ValueError(f"LiveCodeBench {increment} row {ordinal} has invalid prompt fields")
        if difficulty not in {"easy", "medium", "hard"}:
            raise ValueError(f"LiveCodeBench {increment} row {ordinal} has invalid difficulty")
        for field in (
            "question_title",
            "platform",
            "contest_id",
            "contest_date",
            "public_test_cases",
            "private_test_cases",
            "metadata",
        ):
            if not isinstance(row[field], str):
                raise ValueError(f"LiveCodeBench {increment} row {ordinal} field {field} is not text")
        try:
            public_tests = json.loads(row["public_test_cases"])
            metadata = json.loads(row["metadata"])
        except json.JSONDecodeError as exc:
            raise ValueError(f"LiveCodeBench {increment} row {ordinal} public metadata is invalid") from exc
        if not isinstance(public_tests, list) or not isinstance(metadata, dict):
            raise ValueError(f"LiveCodeBench {increment} row {ordinal} public metadata has wrong type")
        if not row["private_test_cases"]:
            raise ValueError(f"LiveCodeBench {increment} row {ordinal} has empty hidden tests")
        seen_ids.add(question_id)
        difficulty_counts[difficulty] += 1
        candidates.append(
            {
                "increment": increment,
                "source_file": source["cache_file"],
                "source_ordinal": ordinal,
                "question_id": question_id,
                "question_content": question.strip(),
                "starter_code": starter,
                "difficulty": difficulty,
                "task_text": code_user_prompt(question, starter),
            }
        )
    if len(candidates) != int(source["expected_rows"]):
        raise ValueError(
            f"LiveCodeBench {increment} has {len(candidates)} rows; expected {source['expected_rows']}"
        )
    expected_counts = source.get("expected_difficulty_counts")
    if expected_counts is not None and dict(difficulty_counts) != expected_counts:
        raise ValueError(
            f"LiveCodeBench {increment} difficulty counts changed: "
            f"{dict(difficulty_counts)} != {expected_counts}"
        )
    return candidates, difficulty_counts


def first_lcb_by_difficulty(
    rows: list[dict[str, Any]], counts: dict[str, int]
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    observed = Counter({difficulty: 0 for difficulty in counts})
    for row in rows:
        difficulty = row["difficulty"]
        if difficulty in counts and observed[difficulty] < counts[difficulty]:
            selected.append(row)
            observed[difficulty] += 1
    if dict(observed) != counts:
        raise ValueError(
            "LiveCodeBench source cannot satisfy frozen difficulty selection: "
            f"{dict(observed)} != {counts}"
        )
    return selected


def task_from_math(row: dict[str, Any], split: str, spec: dict[str, Any]) -> dict[str, Any]:
    name = row["source_name"]
    source = spec["sources"][name]
    scored = split != "warmup"
    if scored:
        reference = {"kind": "aime_integer", "answer": row["answer"]}
    else:
        if name != "amc23" or "answer" in row:
            raise ValueError("warmup math must be answer-free AMC23")
        reference = {"kind": "unscored"}
    return {
        "schema": TASK_SCHEMA,
        "task_id": f"{name}/{row['source_id']}",
        "split": split,
        "family": "aime",
        "source": source["repository"],
        "source_revision": source["revision"],
        "source_file": source["cache_file"],
        "source_ordinal": row["source_ordinal"],
        "difficulty": "competition_math",
        "scored": scored,
        "problem_text": row["problem_text"],
        "task_text": row["task_text"],
        "reference": reference,
    }


def task_from_code(row: dict[str, Any], split: str, spec: dict[str, Any]) -> dict[str, Any]:
    lcb = spec["sources"]["livecodebench"]
    scored = split != "warmup"
    reference = (
        {
            "kind": "livecodebench",
            "question_id": row["question_id"],
            "source_file": row["source_file"],
            "difficulty": row["difficulty"],
        }
        if scored
        else {"kind": "unscored"}
    )
    return {
        "schema": TASK_SCHEMA,
        "task_id": f"lcb/{row['question_id']}",
        "split": split,
        "family": "code",
        "source": lcb["repository"],
        "source_revision": lcb["revision"],
        "source_file": row["source_file"],
        "source_increment": row["increment"],
        "source_ordinal": row["source_ordinal"],
        "difficulty": row["difficulty"],
        "scored": scored,
        "problem_text": row["question_content"],
        "task_text": row["task_text"],
        "reference": reference,
    }


def interleave(tasks_aime: list[dict[str, Any]], tasks_code: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = len(tasks_aime) + len(tasks_code)
    if not tasks_aime or not tasks_code:
        raise ValueError("each measured stream must contain AIME and code tasks")
    aime_positions = {(index * total) // len(tasks_aime) for index in range(len(tasks_aime))}
    if len(aime_positions) != len(tasks_aime):
        raise ValueError("AIME spacing produced duplicate positions")
    result: list[dict[str, Any]] = []
    aime_index = 0
    code_index = 0
    for position in range(total):
        if position in aime_positions:
            result.append(tasks_aime[aime_index])
            aime_index += 1
        else:
            result.append(tasks_code[code_index])
            code_index += 1
    if aime_index != len(tasks_aime) or code_index != len(tasks_code):
        raise ValueError("interleave did not consume both task families exactly once")
    return result


def assignments(tasks: list[dict[str, Any]], user_count: int) -> list[dict[str, Any]]:
    users = [
        {"user_id": f"user-{index:02d}", "user_index": index, "task_ids": []}
        for index in range(user_count)
    ]
    for position, task in enumerate(tasks):
        users[position % user_count]["task_ids"].append(task["task_id"])
    assigned = [task_id for user in users for task_id in user["task_ids"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != {
        task["task_id"] for task in tasks
    }:
        raise ValueError("assignment did not consume each task exactly once")
    return users


def selection_provenance(
    streams: dict[str, list[dict[str, Any]]], spec: dict[str, Any]
) -> dict[str, Any]:
    selected_ordinals: dict[str, Any] = {}
    for split in SPLITS:
        tasks = streams[split]
        selected_ordinals[split] = {
            "aime": [
                task["source_ordinal"] for task in tasks if task["family"] == "aime"
            ],
            "code_by_difficulty": {
                difficulty: [
                    task["source_ordinal"]
                    for task in tasks
                    if task["family"] == "code" and task["difficulty"] == difficulty
                ]
                for difficulty in ("medium", "hard")
            },
        }
    return {"rules": spec["streams"], "selected_source_ordinals": selected_ordinals}


def validate_global_disjointness(streams: dict[str, list[dict[str, Any]]]) -> None:
    ids: dict[str, str] = {}
    texts: dict[str, str] = {}
    for split in SPLITS:
        for task in streams[split]:
            task_id = task["task_id"]
            if task_id in ids:
                raise ValueError(f"task id overlaps {ids[task_id]} and {split}: {task_id}")
            ids[task_id] = split
            normalized = normalize_text(task["problem_text"])
            if not normalized:
                raise ValueError(f"task has empty normalized problem: {task_id}")
            if normalized in texts:
                raise ValueError(f"duplicate problem text: {texts[normalized]} and {task_id}")
            texts[normalized] = task_id


def validate_stream_counts(
    streams: dict[str, list[dict[str, Any]]], spec: dict[str, Any]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for split in SPLITS:
        tasks = streams[split]
        expected = spec["streams"][split]
        families = Counter(task["family"] for task in tasks)
        if (
            families["aime"],
            families["code"],
            len(tasks),
        ) != (expected["aime_count"], expected["code_count"], expected["total"]):
            raise ValueError(f"{split} family/task counts differ from the frozen contract")
        difficulties = Counter(
            task["difficulty"] for task in tasks if task["family"] == "code"
        )
        expected_difficulties = expected.get("code_difficulty_counts")
        if expected_difficulties is not None and dict(difficulties) != expected_difficulties:
            raise ValueError(
                f"{split} code difficulty counts changed: {dict(difficulties)}"
            )
        result[split] = {
            "total": len(tasks),
            "families": dict(families),
            "code_difficulties": dict(difficulties),
        }
    return result


def forbidden_rows(
    streams: dict[str, list[dict[str, Any]]], spec: dict[str, Any]
) -> Iterator[dict[str, str]]:
    policy = spec["request_policy"]
    for split in SPLITS:
        for task in streams[split]:
            common = {"split": split, "family": task["family"], "task_id": task["task_id"]}
            yield {**common, "kind": "problem", "text": task["problem_text"]}
            system = (
                policy["aime_system_prompt"]
                if task["family"] == "aime"
                else policy["code_system_prompt"]
            )
            final_text = task["task_text"] if not system else system + "\n\n" + task["task_text"]
            yield {**common, "kind": "final_prompt", "text": final_text}


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def build(
    *, spec_path: Path, spec: dict[str, Any], source_cache: Path, output: Path, timeout: float
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to replace existing workload directory: {output}")
    sources = spec["sources"]
    paths = {
        name: cache_source(source_cache, sources[name], timeout)
        for name in ("aime25", "aime24", "amc23")
    }
    lcb = sources["livecodebench"]
    lcb_paths = {
        increment: cache_source(source_cache, lcb["increments"][increment], timeout)
        for increment in ("v4", "v5", "v6")
    }
    math = {
        name: load_math_source(name, paths[name], sources[name], spec)
        for name in ("aime25", "aime24", "amc23")
    }
    lcb_rows: dict[str, list[dict[str, Any]]] = {}
    lcb_source_counts: dict[str, Any] = {}
    all_lcb_ids: dict[str, str] = {}
    for increment in ("v4", "v5", "v6"):
        rows, counts = load_lcb_increment(increment, lcb_paths[increment], lcb)
        for row in rows:
            question_id = row["question_id"]
            if question_id in all_lcb_ids:
                raise ValueError(
                    f"LiveCodeBench id overlaps {all_lcb_ids[question_id]} and {increment}: "
                    f"{question_id}"
                )
            all_lcb_ids[question_id] = increment
        lcb_rows[increment] = rows
        lcb_source_counts[increment] = {
            "total": len(rows),
            "difficulties": dict(counts),
        }

    warm_math = math["amc23"][:1]
    warm_code_candidates = [
        row for row in lcb_rows["v4"] if row["difficulty"] in {"medium", "hard"}
    ]
    if not warm_code_candidates:
        raise ValueError("LiveCodeBench v4 has no medium/hard warmup candidate")
    warm_aime_tasks = [task_from_math(row, "warmup", spec) for row in warm_math]
    warm_code_tasks = [
        task_from_code(row, "warmup", spec) for row in warm_code_candidates[:1]
    ]
    warmup = [warm_aime_tasks[0], warm_code_tasks[0]]

    dev_aime = [task_from_math(row, "dev", spec) for row in math["aime24"]]
    dev_code = [
        task_from_code(row, "dev", spec)
        for row in lcb_rows["v5"]
        if row["difficulty"] in {"medium", "hard"}
    ]
    final_aime = [task_from_math(row, "final", spec) for row in math["aime25"]]
    final_code_rows = first_lcb_by_difficulty(
        lcb_rows["v6"], FROZEN_STREAMS["final"]["code_difficulty_counts"]
    )
    final_code = [task_from_code(row, "final", spec) for row in final_code_rows]
    streams = {
        "warmup": warmup,
        "dev": interleave(dev_aime, dev_code),
        "final": interleave(final_aime, final_code),
    }
    validate_global_disjointness(streams)
    task_counts = validate_stream_counts(streams, spec)
    split_assignments = {
        split: assignments(streams[split], 2 if split == "warmup" else 20)
        for split in SPLITS
    }
    final_queue_lengths = Counter(
        len(user["task_ids"]) for user in split_assignments["final"]
    )
    if final_queue_lengths != Counter({2: 10, 3: 10}):
        raise ValueError("final assignment must give ten users two tasks and ten users three")

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for split in SPLITS:
            write_jsonl(staging / f"{split}.jsonl", streams[split])
        write_jsonl(
            staging / "training_forbidden_texts.jsonl", forbidden_rows(streams, spec)
        )
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "name": spec["name"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "spec": str(spec_path),
            "normalization": spec["normalization"],
            "sources": sources,
            "source_cache": str(source_cache),
            "source_files": {
                "aime25": sources["aime25"]["cache_file"],
                "aime24": sources["aime24"]["cache_file"],
                "amc23": sources["amc23"]["cache_file"],
                "livecodebench": {
                    increment: lcb["increments"][increment]["cache_file"]
                    for increment in ("v4", "v5", "v6")
                },
            },
            "source_validation": {
                "aime25_rows": len(math["aime25"]),
                "aime24_rows": len(math["aime24"]),
                "amc23_rows": len(math["amc23"]),
                "amc23_answer_values_loaded_or_validated": False,
                "livecodebench": lcb_source_counts,
                "all_task_ids_unique": True,
                "all_normalized_problem_texts_unique": True,
                "streams_pairwise_disjoint": True,
            },
            "task_files": {split: f"{split}.jsonl" for split in SPLITS},
            "task_counts": task_counts,
            "assignments": split_assignments,
            "selection": selection_provenance(streams, spec),
            "traffic": spec["traffic"],
            "request_policy": spec["request_policy"],
            "metrics": spec["metrics"],
            "training_forbidden_texts": {
                "schema": FORBIDDEN_SCHEMA,
                "path": "training_forbidden_texts.jsonl",
                "contains_answers_or_hidden_tests": False,
                "rows_per_task": 2,
            },
        }
        write_json(staging / "manifest.json", manifest)
        staging.replace(output)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = args.spec
    source_cache = args.source_cache.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    spec = args.frozen_spec
    if args.plan_only:
        print(
            json.dumps(
                plan_payload(spec_path, spec, source_cache, output),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    manifest = build(
        spec_path=spec_path,
        spec=spec,
        source_cache=source_cache,
        output=output,
        timeout=args.download_timeout,
    )
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "manifest": str(output / "manifest.json"),
                "task_counts": manifest["task_counts"],
                "source_cache": str(source_cache),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
