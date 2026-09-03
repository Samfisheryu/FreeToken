#!/usr/bin/env python3
"""Download fixed official sources and build the mixed task-goodput streams.

The output is immutable benchmark input, not training data.  It contains one
warmup stream plus disjoint development and final streams.  A separate public
training-isolation JSONL intentionally omits all reference solutions and
answers.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import io
import json
import math
from pathlib import Path
import re
import unicodedata
import urllib.request
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "workloads" / "mixed_task_goodput_v1.json"
DEFAULT_OUTPUT = Path("/data1/lmcache_kv/goodput_campaign/mixed_workload_sources_v1")
MANIFEST_SCHEMA = "freetoken.mixed_task_goodput_manifest.v1"
TASK_SCHEMA = "freetoken.mixed_task_goodput_task.v1"
FORBIDDEN_SCHEMA = "freetoken.training_forbidden_texts.v1"
FAMILIES = ("numeric", "code", "knowledge")
EXPECTED_SOURCE_FIELDS = {
    "numeric": {"input", "code", "target"},
    "code": {
        "task_id",
        "prompt",
        "entry_point",
        "contract",
        "canonical_solution",
        "base_input",
        "plus_input",
        "atol",
    },
    "knowledge": {"question", "subject", "choices", "answer"},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-cache",
        type=Path,
        help="optional directory containing already downloaded, decompressed source files",
    )
    parser.add_argument(
        "--download-timeout", type=float, default=120.0, help="seconds per source file"
    )
    args = parser.parse_args(argv)
    if args.download_timeout <= 0:
        parser.error("--download-timeout must be positive")
    return args


def load_spec(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    if not isinstance(spec, dict) or spec.get("schema") != "freetoken.mixed_task_goodput_spec.v1":
        raise ValueError(f"unsupported workload spec: {path}")
    return spec


def download(url: str, timeout: float) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "FreeToken-mixed-goodput/1"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def source_bytes(
    *, cache: Path | None, filename: str, url: str, timeout: float, gzip_source: bool = False
) -> bytes:
    cached = cache / filename if cache is not None else None
    if cached is not None and cached.is_file():
        return cached.read_bytes()
    raw = download(url, timeout)
    return gzip.decompress(raw) if gzip_source else raw


def write_bytes(path: Path, data: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    temporary.replace(path)


def read_jsonl_bytes(raw: bytes, source: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{source} line {line_number} is not a JSON object")
        rows.append(value)
    return rows


def require_fields(row: dict[str, Any], required: set[str], source: str) -> None:
    missing = sorted(required.difference(row))
    if missing:
        raise ValueError(f"{source} fields differ from the frozen contract; missing {missing}")


def normalize_text(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(value.split()).strip()


def evenly_spaced(rows: list[dict[str, Any]], count: int, label: str) -> list[dict[str, Any]]:
    if len(rows) < count:
        raise ValueError(f"{label} has {len(rows)} candidates but needs {count}")
    indices = [(index * len(rows)) // count for index in range(count)]
    if len(set(indices)) != count:
        raise ValueError(f"{label} cannot select {count} unique evenly-spaced rows")
    return [rows[index] for index in indices]


def decimal_reference(value: Any, source: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source} target must be int or float, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{source} target must be finite")
    if number.is_integer():
        return str(int(number))
    return repr(number)


def code_sort_key(row: dict[str, Any]) -> int:
    task_id = row.get("task_id")
    if not isinstance(task_id, str) or not re.fullmatch(r"(?:HumanEval|Mbpp)/\d+", task_id):
        raise ValueError(f"unexpected EvalPlus task_id: {task_id!r}")
    return int(task_id.rsplit("/", 1)[1])


def numeric_task(row: dict[str, Any], ordinal: int) -> dict[str, Any]:
    require_fields(row, EXPECTED_SOURCE_FIELDS["numeric"], f"gsm-hard row {ordinal}")
    problem = row["input"]
    if not isinstance(problem, str) or not problem.strip():
        raise ValueError(f"gsm-hard row {ordinal} has no input")
    return {
        "schema": TASK_SCHEMA,
        "task_id": f"gsm-hard/{ordinal}",
        "family": "numeric",
        "source": "reasoning-machines/gsm-hard",
        "source_split": "train",
        "source_ordinal": ordinal,
        "task_text": problem,
        "reference": {"kind": "decimal", "value": decimal_reference(row["target"], f"gsm-hard row {ordinal}")},
    }


def code_task(row: dict[str, Any], dataset: str, ordinal: int) -> dict[str, Any]:
    require_fields(row, EXPECTED_SOURCE_FIELDS["code"], f"EvalPlus {dataset} row {ordinal}")
    for field in ("task_id", "prompt", "entry_point", "contract", "canonical_solution"):
        if not isinstance(row[field], str):
            raise ValueError(f"EvalPlus {dataset} row {ordinal} field {field} must be text")
    if not isinstance(row["base_input"], list):
        raise ValueError(f"EvalPlus {dataset} row {ordinal} base_input must be a list")
    plus_input = row["plus_input"]
    valid_empty_mbpp_793 = (
        dataset == "mbpp" and row["task_id"] == "Mbpp/793" and plus_input == {}
    )
    if not isinstance(plus_input, list) and not valid_empty_mbpp_793:
        raise ValueError(
            f"EvalPlus {dataset} row {ordinal} plus_input must be a list; only "
            "Mbpp/793 may use the official empty-dict zero-test representation"
        )
    return {
        "schema": TASK_SCHEMA,
        "task_id": str(row["task_id"]),
        "family": "code",
        "source": f"EvalPlus/{dataset}",
        "source_split": "test",
        "source_ordinal": ordinal,
        "task_text": row["prompt"],
        "reference": {
            "kind": "evalplus",
            "dataset": dataset,
            "entry_point": row["entry_point"],
            "contract": row["contract"],
            "canonical_solution": row["canonical_solution"],
            "base_input": row["base_input"],
            "plus_input": row["plus_input"],
            "atol": row["atol"],
        },
    }


def knowledge_task(row: dict[str, Any], split: str, ordinal: int) -> dict[str, Any]:
    require_fields(row, EXPECTED_SOURCE_FIELDS["knowledge"], f"MMLU {split} row {ordinal}")
    question, subject, choices, answer = (
        row["question"], row["subject"], row["choices"], row["answer"]
    )
    if not isinstance(question, str) or not isinstance(subject, str):
        raise ValueError(f"MMLU {split} row {ordinal} question/subject must be text")
    if not isinstance(choices, list) or len(choices) != 4 or not all(isinstance(x, str) for x in choices):
        raise ValueError(f"MMLU {split} row {ordinal} choices must contain four strings")
    if isinstance(answer, bool) or not isinstance(answer, int) or answer not in range(4):
        raise ValueError(f"MMLU {split} row {ordinal} answer must be int64 in 0..3")
    letters = "ABCD"
    text = question.rstrip() + "\n" + "\n".join(
        f"{letter}. {choice}" for letter, choice in zip(letters, choices)
    )
    return {
        "schema": TASK_SCHEMA,
        "task_id": f"mmlu/{split}/{subject}/{ordinal}",
        "family": "knowledge",
        "source": "cais/mmlu",
        "source_split": split,
        "source_ordinal": ordinal,
        "subject": subject,
        "task_text": text,
        "reference": {"kind": "choice", "value": letters[answer]},
    }


def read_parquet(raw: bytes, source: str) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("building the MMLU stream requires pyarrow") from exc
    table = parquet.read_table(io.BytesIO(raw))
    rows = table.to_pylist()
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{source} did not decode to object rows")
    return rows


def round_robin(
    rows: list[dict[str, Any]], count: int, excluded_ids: set[str], label: str
) -> list[dict[str, Any]]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["task_id"] not in excluded_ids:
            by_subject[str(row["subject"])].append(row)
    selected: list[dict[str, Any]] = []
    depth = 0
    subjects = sorted(by_subject)
    while len(selected) < count:
        progressed = False
        for subject in subjects:
            subject_rows = by_subject[subject]
            if depth < len(subject_rows):
                selected.append(subject_rows[depth])
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            raise ValueError(f"{label} exhausted after {len(selected)} rows")
        depth += 1
    return selected


def ensure_unique(tasks: Iterable[dict[str, Any]]) -> None:
    ids: set[str] = set()
    texts: dict[str, str] = {}
    for task in tasks:
        task_id = task["task_id"]
        if task_id in ids:
            raise ValueError(f"task appears in more than one stream: {task_id}")
        ids.add(task_id)
        normalized = normalize_text(task["task_text"])
        if not normalized:
            raise ValueError(f"task has empty normalized text: {task_id}")
        if normalized in texts:
            raise ValueError(f"normalized duplicate task text: {texts[normalized]} and {task_id}")
        texts[normalized] = task_id


def assign_stream(
    pools: dict[str, list[dict[str, Any]]], turns_per_user: int
) -> list[dict[str, Any]]:
    positions = {family: 0 for family in FAMILIES}
    users = [
        {"user_id": f"user-{user_index:02d}", "user_index": user_index, "task_ids": []}
        for user_index in range(20)
    ]
    for turn_index in range(turns_per_user):
        for user_index in range(20):
            family = FAMILIES[(user_index + turn_index) % len(FAMILIES)]
            position = positions[family]
            if position >= len(pools[family]):
                raise ValueError(f"{family} pool exhausted during assignment")
            users[user_index]["task_ids"].append(pools[family][position]["task_id"])
            positions[family] += 1
    expected = {family: len(pools[family]) for family in FAMILIES}
    if positions != expected:
        raise ValueError(f"stream assignment did not consume each family exactly once: {positions} != {expected}")
    return users


def warmup_assignments(pools: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    positions = {family: 0 for family in FAMILIES}
    users: list[dict[str, Any]] = []
    for user_index in range(20):
        family = FAMILIES[user_index % len(FAMILIES)]
        task = pools[family][positions[family]]
        positions[family] += 1
        users.append(
            {
                "user_id": f"user-{user_index:02d}",
                "user_index": user_index,
                "task_ids": [task["task_id"]],
            }
        )
    if positions != {"numeric": 7, "code": 7, "knowledge": 6}:
        raise ValueError(f"warmup family allocation changed: {positions}")
    return users


def final_user_prompt(task: dict[str, Any], spec: dict[str, Any]) -> str:
    instruction = spec["request_policy"]["families"][task["family"]]["instruction"]
    return task["task_text"].rstrip() + "\n\n" + instruction


def forbidden_rows(
    split: str, tasks: Iterable[dict[str, Any]], spec: dict[str, Any]
) -> Iterable[dict[str, str]]:
    system_prompt = spec["request_policy"]["system_prompt"]
    for task in tasks:
        common = {"family": task["family"], "task_id": task["task_id"], "split": split}
        yield {**common, "kind": "problem", "text": task["task_text"]}
        yield {
            **common,
            "kind": "final_prompt",
            "text": system_prompt + "\n\n" + final_user_prompt(task, spec),
        }
        if task["family"] == "code":
            reference = task["reference"]
            yield {**common, "kind": "entry_point", "text": reference["entry_point"]}
            if reference["contract"]:
                yield {**common, "kind": "contract", "text": reference["contract"]}
            for suite in ("base_input", "plus_input"):
                for public_test in reference[suite]:
                    yield {
                        **common,
                        "kind": "public_test",
                        "text": json.dumps(public_test, ensure_ascii=False, separators=(",", ":")),
                    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = args.spec.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty frozen workload directory: {output}"
        )
    output.mkdir(parents=True, exist_ok=True)
    sources_dir = output / "sources"
    sources_dir.mkdir()
    spec = load_spec(spec_path)
    source_spec = spec["sources"]
    source_cache = args.source_cache.expanduser().resolve() if args.source_cache else None

    gsm_raw = source_bytes(
        cache=source_cache,
        filename="gsmhardv2.jsonl",
        url=source_spec["numeric"]["url"],
        timeout=args.download_timeout,
    )
    write_bytes(sources_dir / "gsmhardv2.jsonl", gsm_raw)
    gsm_rows = read_jsonl_bytes(gsm_raw, "gsm-hard")
    if len(gsm_rows) != 1319:
        raise ValueError(f"gsm-hard row count changed: expected 1319, got {len(gsm_rows)}")
    numeric = [numeric_task(row, ordinal) for ordinal, row in enumerate(gsm_rows)]
    numeric_dev_candidates = [row for row in numeric if row["source_ordinal"] % 4 == 0]
    numeric_final_candidates = [row for row in numeric if row["source_ordinal"] % 4 != 0]
    numeric_dev = evenly_spaced(numeric_dev_candidates, 80, "numeric dev")
    numeric_final = evenly_spaced(numeric_final_candidates, 140, "numeric final")
    numeric_used = {row["task_id"] for row in numeric_dev + numeric_final}
    numeric_warmup = evenly_spaced(
        [row for row in numeric if row["task_id"] not in numeric_used], 7, "numeric warmup"
    )

    code_sets: dict[str, list[dict[str, Any]]] = {}
    for dataset, filename in (
        ("humaneval", "HumanEvalPlus-v0.1.10.jsonl"),
        ("mbpp", "MbppPlus-v0.2.0.jsonl"),
    ):
        dataset_spec = source_spec["code"]["datasets"][dataset]
        raw = source_bytes(
            cache=source_cache,
            filename=filename,
            url=dataset_spec["url"],
            timeout=args.download_timeout,
            gzip_source=True,
        )
        write_bytes(sources_dir / filename, raw)
        rows = read_jsonl_bytes(raw, f"EvalPlus {dataset}")
        if len(rows) != int(dataset_spec["rows"]):
            raise ValueError(
                f"EvalPlus {dataset} row count changed: expected {dataset_spec['rows']}, got {len(rows)}"
            )
        rows.sort(key=code_sort_key)
        code_sets[dataset] = [code_task(row, dataset, ordinal) for ordinal, row in enumerate(rows)]

    code_quotas = spec["selection"]["code_quotas"]
    code_selected: dict[str, list[dict[str, Any]]] = {split: [] for split in ("warmup", "dev", "final")}
    for dataset, rows in code_sets.items():
        dev_candidates = [row for ordinal, row in enumerate(rows) if ordinal % 4 == 0]
        final_candidates = [row for ordinal, row in enumerate(rows) if ordinal % 4 != 0]
        dev = evenly_spaced(dev_candidates, int(code_quotas["dev"][dataset]), f"code {dataset} dev")
        final = evenly_spaced(final_candidates, int(code_quotas["final"][dataset]), f"code {dataset} final")
        used = {row["task_id"] for row in dev + final}
        warmup = evenly_spaced(
            [row for row in rows if row["task_id"] not in used],
            int(code_quotas["warmup"][dataset]),
            f"code {dataset} warmup",
        )
        code_selected["dev"].extend(dev)
        code_selected["final"].extend(final)
        code_selected["warmup"].extend(warmup)

    mmlu_validation_raw = source_bytes(
        cache=source_cache,
        filename="mmlu_validation.parquet",
        url=source_spec["knowledge"]["validation_url"],
        timeout=args.download_timeout,
    )
    mmlu_test_raw = source_bytes(
        cache=source_cache,
        filename="mmlu_test.parquet",
        url=source_spec["knowledge"]["test_url"],
        timeout=args.download_timeout,
    )
    write_bytes(sources_dir / "mmlu_validation.parquet", mmlu_validation_raw)
    write_bytes(sources_dir / "mmlu_test.parquet", mmlu_test_raw)
    mmlu_validation_rows = read_parquet(mmlu_validation_raw, "MMLU validation")
    mmlu_test_rows = read_parquet(mmlu_test_raw, "MMLU test")
    if len(mmlu_validation_rows) != 1531 or len(mmlu_test_rows) != 14042:
        raise ValueError(
            "MMLU row counts changed: expected validation=1531/test=14042, got "
            f"{len(mmlu_validation_rows)}/{len(mmlu_test_rows)}"
        )
    excluded_subjects = set(source_spec["knowledge"]["excluded_subjects"])
    validation = [
        knowledge_task(row, "validation", ordinal)
        for ordinal, row in enumerate(mmlu_validation_rows)
        if row.get("subject") not in excluded_subjects
    ]
    final_knowledge_pool = [
        knowledge_task(row, "test", ordinal)
        for ordinal, row in enumerate(mmlu_test_rows)
        if row.get("subject") not in excluded_subjects
    ]
    subjects = sorted({row["subject"] for row in validation})
    if len(subjects) != 52:
        raise ValueError(f"MMLU non-mathematics subject count changed: expected 52, got {len(subjects)}")
    knowledge_dev = round_robin(validation, 80, set(), "knowledge dev")
    knowledge_warmup = round_robin(
        validation, 6, {row["task_id"] for row in knowledge_dev}, "knowledge warmup"
    )
    knowledge_final = round_robin(final_knowledge_pool, 140, set(), "knowledge final")

    pools = {
        "warmup": {
            "numeric": numeric_warmup,
            "code": code_selected["warmup"],
            "knowledge": knowledge_warmup,
        },
        "dev": {
            "numeric": numeric_dev,
            "code": code_selected["dev"],
            "knowledge": knowledge_dev,
        },
        "final": {
            "numeric": numeric_final,
            "code": code_selected["final"],
            "knowledge": knowledge_final,
        },
    }
    all_tasks = [task for split in pools.values() for family in FAMILIES for task in split[family]]
    ensure_unique(all_tasks)
    split_tasks = {
        split: [task for family in FAMILIES for task in pools[split][family]]
        for split in ("warmup", "dev", "final")
    }
    for split, tasks in split_tasks.items():
        write_jsonl(output / f"{split}.jsonl", tasks)

    assignments = {
        "warmup": warmup_assignments(pools["warmup"]),
        "dev": assign_stream(pools["dev"], 12),
        "final": assign_stream(pools["final"], 21),
    }
    forbidden = [
        row
        for split in ("warmup", "dev", "final")
        for row in forbidden_rows(split, split_tasks[split], spec)
    ]
    for row in forbidden:
        if set(row) != {"family", "task_id", "split", "kind", "text"}:
            raise ValueError("training-forbidden row schema changed")
    write_jsonl(output / "training_forbidden_texts.jsonl", forbidden)

    manifest = {
        "schema": MANIFEST_SCHEMA,
        "name": spec["name"],
        "spec_path": str(spec_path),
        "sources": source_spec,
        "source_files": {
            "gsm_hard": "sources/gsmhardv2.jsonl",
            "humaneval_plus": "sources/HumanEvalPlus-v0.1.10.jsonl",
            "mbpp_plus": "sources/MbppPlus-v0.2.0.jsonl",
            "mmlu_validation": "sources/mmlu_validation.parquet",
            "mmlu_test": "sources/mmlu_test.parquet",
        },
        "training_forbidden_texts": {
            "file": "training_forbidden_texts.jsonl",
            "schema": FORBIDDEN_SCHEMA,
            "normalization": "nfkc_casefold_unicode_whitespace_v1",
        },
        "traffic": spec["traffic"],
        "request_policy": spec["request_policy"],
        "selection": spec["selection"],
        "task_files": {split: f"{split}.jsonl" for split in ("warmup", "dev", "final")},
        "task_counts": {
            split: {
                "total": len(split_tasks[split]),
                **{
                    family: sum(task["family"] == family for task in split_tasks[split])
                    for family in FAMILIES
                },
            }
            for split in ("warmup", "dev", "final")
        },
        "assignments": assignments,
    }
    write_json(output / "manifest.json", manifest)
    print(
        json.dumps(
            {
                "manifest": str(output / "manifest.json"),
                "task_counts": manifest["task_counts"],
                "training_forbidden_rows": len(forbidden),
                "answers_printed": False,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
