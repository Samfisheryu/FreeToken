#!/usr/bin/env python3
"""Build the frozen long-context mixed task-goodput v2 workload.

The builder downloads only fixed-revision official source files into a reusable
source cache.  Publication is fresh and atomic: an existing output directory is
never updated, and a partially built directory is never renamed into place.
Gold answers, programs, patches, tests, and supporting facts are used only for
offline scoring metadata; none enter a model prompt or the training-isolation
text file.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata
import urllib.request
from typing import Any, Callable, Iterable, Iterator


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "workloads" / "long_mixed_task_goodput_v2.json"
DEFAULT_SOURCE_CACHE = Path(
    "/data1/lmcache_kv/goodput_campaign/long_mixed_workload_v2_sources"
)
DEFAULT_OUTPUT = Path("/data1/lmcache_kv/goodput_campaign/long_mixed_workload_v2")
SPEC_SCHEMA = "freetoken.long_mixed_task_goodput_spec.v2"
MANIFEST_SCHEMA = "freetoken.long_mixed_task_goodput_manifest.v2"
TASK_SCHEMA = "freetoken.long_mixed_task_goodput_task.v2"
FORBIDDEN_SCHEMA = "freetoken.long_training_forbidden_texts.v2"
FAMILIES = ("numeric", "code", "knowledge")
TARGETS = (8192, 16384, 32768)
SYSTEM_PROMPT = "Follow the requested output format exactly."
SWE_EVALUATOR_REVISION = "87ab1f6ced28f75ba73ca899dc759b019310944a"
SWE_EVALUATOR_VERSION = "5.0.1"
FINQA_EVALUATOR_REVISION = "0f16e2867befa6840783e58be38c9efb9229d742"
CODE_REPOSITORY_DOMAIN = "Code Repository Understanding"
CODE_REPOSITORY_SUBDOMAIN = "Code repo QA"
TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)
FILE_MARKER_RE = re.compile(
    r"(?m)^\[(start|end) of ([^\]\r\n]+)\][ \t]*\r?$"
)

DOCFINQA_FIELDS = {"Context", "Question", "Program", "Answer"}
BM25_FIELDS = {
    "instance_id",
    "text",
    "repo",
    "base_commit",
    "problem_statement",
    "hints_text",
    "created_at",
    "patch",
    "test_patch",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "environment_setup_commit",
}
VERIFIED_FIELDS = {
    "base_commit",
    "created_at",
    "difficulty",
    "environment_setup_commit",
    "eval_type",
    "image",
    "instance_id",
    "log_parser",
    "repo",
    "version",
    "patch",
    "test_patch",
    "eval_script",
    "problem_statement",
    "hints_text",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
}
VERIFIED_EVAL_FIELDS = {
    "instance_id",
    "image",
    "repo",
    "version",
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "log_parser",
    "eval_type",
    "eval_script",
}
LONGBENCH_FIELDS = {
    "_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
    "context",
}


@dataclass
class Candidate:
    family: str
    task_id: str
    source: str
    source_revision: str
    source_split: str
    source_ordinal: int
    question: str
    context: str
    reference: dict[str, Any]
    options: tuple[str, str, str, str] | None = None
    context_kind: str = "document"
    cache: dict[int, "BuiltTask | None"] = field(default_factory=dict)


@dataclass
class BuiltTask:
    candidate: Candidate
    target: int
    user_prompt: str
    selected_context: str
    prompt_tokens: int
    retrieval: dict[str, Any]

    def row(self, split: str) -> dict[str, Any]:
        candidate = self.candidate
        return {
            "schema": TASK_SCHEMA,
            "task_id": candidate.task_id,
            "family": candidate.family,
            "split": split,
            "source": candidate.source,
            "source_revision": candidate.source_revision,
            "source_split": candidate.source_split,
            "source_ordinal": candidate.source_ordinal,
            "length_bucket_tokens": self.target,
            "prompt_tokens": self.prompt_tokens,
            "task_text": self.user_prompt,
            "retrieval": self.retrieval,
            "reference": candidate.reference,
        }


class PromptBuilder:
    def __init__(self, tokenizer: Any, spec: dict[str, Any]) -> None:
        self.tokenizer = tokenizer
        self.spec = spec
        retrieval = spec["length_buckets"]["retrieval"]
        self.maximum_block_tokens = int(retrieval["maximum_block_tokens"])
        self.k1 = float(retrieval["k1"])
        self.b = float(retrieval["b"])

    def encode(self, text: str) -> list[int]:
        return list(self.tokenizer.encode(text, add_special_tokens=False))

    def messages(self, user_prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

    def prompt_tokens(self, user_prompt: str) -> int:
        ids = self.tokenizer.apply_chat_template(
            self.messages(user_prompt),
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if isinstance(ids, Mapping):
            if "input_ids" not in ids:
                raise ValueError("tokenizer chat template result lacks input_ids")
            ids = ids["input_ids"]
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if ids and isinstance(ids[0], list):
            ids = ids[0]
        return len(ids)

    def render(self, candidate: Candidate, context: str) -> str:
        instruction = self.spec["families"][candidate.family]["instruction"]
        if candidate.family == "numeric":
            return (
                "Financial document:\n"
                + context.strip()
                + "\n\nQuestion:\n"
                + candidate.question.strip()
                + "\n\n"
                + instruction
            )
        if candidate.family == "code":
            return (
                "Repository excerpts:\n"
                + context.strip()
                + "\n\nIssue:\n"
                + candidate.question.strip()
                + "\n\n"
                + instruction
            )
        assert candidate.options is not None
        choices = "\n".join(
            f"{letter}. {choice.strip()}"
            for letter, choice in zip("ABCD", candidate.options)
        )
        return (
            "Context:\n"
            + context.strip()
            + "\n\nQuestion:\n"
            + candidate.question.strip()
            + "\n"
            + choices
            + "\n\n"
            + instruction
        )

    def build(self, candidate: Candidate, target: int) -> BuiltTask | None:
        if target in candidate.cache:
            return candidate.cache[target]
        lower = math.ceil(target * float(self.spec["length_buckets"]["minimum_fraction"]))
        full_prompt = self.render(candidate, candidate.context)
        raw_prompt_tokens = self.prompt_tokens(full_prompt)
        if raw_prompt_tokens <= target:
            if raw_prompt_tokens < lower:
                candidate.cache[target] = None
                return None
            built = BuiltTask(
                candidate=candidate,
                target=target,
                user_prompt=full_prompt,
                selected_context=candidate.context.strip(),
                prompt_tokens=raw_prompt_tokens,
                retrieval={
                    "mode": "full_source_context",
                    "query_source": "question_only",
                    "raw_prompt_tokens": raw_prompt_tokens,
                    "selected_prompt_tokens": raw_prompt_tokens,
                    "minimum_prompt_tokens": lower,
                    "maximum_prompt_tokens": target,
                    "padding_or_repetition": False,
                },
            )
            candidate.cache[target] = built
            return built

        blocks = self.source_blocks(candidate)
        if not blocks:
            candidate.cache[target] = None
            return None
        ranked = self.bm25_rank(candidate.question, blocks)
        empty_tokens = self.prompt_tokens(self.render(candidate, ""))
        lengths = {index: len(self.encode(text)) for index, text in enumerate(blocks)}
        estimated_budget = max(0, target - empty_tokens - len(blocks))
        selected_ranked: list[int] = []
        estimated = 0
        for index in ranked:
            length = lengths[index] + 1
            if estimated + length <= estimated_budget:
                selected_ranked.append(index)
                estimated += length

        def materialize(indices: Iterable[int]) -> tuple[str, int]:
            ordered = sorted(indices)
            context = "\n\n".join(blocks[index].strip() for index in ordered).strip()
            return context, self.prompt_tokens(self.render(candidate, context))

        selected_context, selected_tokens = materialize(selected_ranked)
        while selected_ranked and selected_tokens > target:
            selected_ranked.pop()
            selected_context, selected_tokens = materialize(selected_ranked)

        selected_set = set(selected_ranked)
        if selected_tokens < lower:
            for index in ranked:
                if index in selected_set:
                    continue
                trial = [*selected_ranked, index]
                trial_context, trial_tokens = materialize(trial)
                if trial_tokens <= target:
                    selected_ranked = trial
                    selected_set.add(index)
                    selected_context, selected_tokens = trial_context, trial_tokens
                if selected_tokens >= lower:
                    break
        if not selected_ranked or not lower <= selected_tokens <= target:
            candidate.cache[target] = None
            return None
        chosen = sorted(selected_ranked)
        user_prompt = self.render(candidate, selected_context)
        built = BuiltTask(
            candidate=candidate,
            target=target,
            user_prompt=user_prompt,
            selected_context=selected_context,
            prompt_tokens=selected_tokens,
            retrieval={
                "mode": "question_only_bm25",
                "algorithm": "Okapi BM25",
                "algorithm_version": "question_only_original_blocks_v1",
                "k1": self.k1,
                "b": self.b,
                "query_source": "question_only",
                "source_block_count": len(blocks),
                "selected_block_count": len(chosen),
                "selected_block_indices": chosen,
                "selected_blocks_restored_to_source_order": True,
                "maximum_source_block_tokens": self.maximum_block_tokens,
                "raw_prompt_tokens": raw_prompt_tokens,
                "selected_prompt_tokens": selected_tokens,
                "minimum_prompt_tokens": lower,
                "maximum_prompt_tokens": target,
                "padding_or_repetition": False,
            },
        )
        candidate.cache[target] = built
        return built

    def source_blocks(self, candidate: Candidate) -> list[str]:
        text = candidate.context.replace("\r\n", "\n").replace("\r", "\n")
        if candidate.context_kind == "repository":
            initial = [
                block
                for _, block in repository_file_blocks(text, candidate.task_id)
            ]
        else:
            initial = split_paragraphs(text)
        blocks: list[str] = []
        for value in initial:
            blocks.extend(self.split_token_blocks(value))
        unique: list[str] = []
        seen: set[str] = set()
        for block in blocks:
            block = block.strip()
            if not block or block in seen:
                continue
            seen.add(block)
            unique.append(block)
        return unique

    def split_token_blocks(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        encoded = self.tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        if not isinstance(encoded, Mapping):
            raise ValueError("tokenizer must return input_ids and offset_mapping")
        ids = encoded.get("input_ids")
        offsets = encoded.get("offset_mapping")
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        if hasattr(offsets, "tolist"):
            offsets = offsets.tolist()
        if not isinstance(ids, list) or not isinstance(offsets, list) or len(ids) != len(offsets):
            raise ValueError("tokenizer returned invalid input_ids/offset_mapping")
        if len(ids) <= self.maximum_block_tokens:
            return [text]
        normalized_offsets: list[tuple[int, int]] = []
        prefix_ends: list[int] = []
        furthest_end = 0
        for value in offsets:
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError("tokenizer returned invalid offset_mapping entry")
            start, end = value
            if (
                not isinstance(start, int)
                or not isinstance(end, int)
                or start < 0
                or end < start
                or end > len(text)
            ):
                raise ValueError("tokenizer offset_mapping contains an invalid source span")
            normalized_offsets.append((start, end))
            furthest_end = max(furthest_end, end)
            prefix_ends.append(furthest_end)
        pieces: list[str] = []
        token_start = 0
        character_start = 0
        while token_start < len(ids):
            token_end = min(token_start + self.maximum_block_tokens, len(ids))
            while True:
                character_end = (
                    len(text)
                    if token_end == len(ids)
                    else max(prefix_ends[token_end - 1], normalized_offsets[token_end][0])
                )
                piece = text[character_start:character_end].strip()
                if piece and len(self.encode(piece)) <= self.maximum_block_tokens:
                    break
                token_end -= 1
                if token_end <= token_start:
                    raise ValueError("tokenizer offsets cannot form a non-empty source block")
            pieces.append(piece)
            token_start = token_end
            character_start = character_end
        return pieces

    def bm25_rank(self, question: str, blocks: list[str]) -> list[int]:
        query = lexical_tokens(question)
        documents = [lexical_tokens(block) for block in blocks]
        average_length = sum(len(document) for document in documents) / max(1, len(documents))
        document_frequency: Counter[str] = Counter()
        for document in documents:
            document_frequency.update(set(document))
        scores: list[tuple[float, int]] = []
        count = len(documents)
        for index, document in enumerate(documents):
            frequencies = Counter(document)
            length = len(document)
            score = 0.0
            for term in query:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                frequency_docs = document_frequency[term]
                inverse = math.log(1.0 + (count - frequency_docs + 0.5) / (frequency_docs + 0.5))
                norm = self.k1 * (
                    1.0 - self.b + self.b * length / max(1.0, average_length)
                )
                score += inverse * frequency * (self.k1 + 1.0) / (frequency + norm)
            scores.append((score, index))
        return [index for _, index in sorted(scores, key=lambda item: (-item[0], item[1]))]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--source-cache", type=Path, default=DEFAULT_SOURCE_CACHE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--tokenizer",
        help="Qwen tokenizer path/id; default is frozen in the spec",
    )
    parser.add_argument("--download-timeout", type=float, default=600.0)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="print fixed sources, traffic, bucket quotas, and ceilings without downloads",
    )
    args = parser.parse_args(argv)
    if args.download_timeout <= 0:
        parser.error("--download-timeout must be positive")
    return args


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def load_spec(path: Path) -> dict[str, Any]:
    spec = load_json(path)
    if spec.get("schema") != SPEC_SCHEMA:
        raise ValueError(f"unsupported long mixed workload spec: {path}")
    validate_frozen_spec(spec)
    return spec


def validate_frozen_spec(spec: dict[str, Any]) -> None:
    if spec.get("system_prompt") != SYSTEM_PROMPT:
        raise ValueError("the frozen common system prompt changed")
    if spec.get("max_sequence_length") != 40960:
        raise ValueError("max_sequence_length must remain 40960")
    traffic = spec.get("traffic")
    expected_traffic = {
        "user_count": 20,
        "first_submission_stagger_seconds": 0.5,
        "think_time_seconds": 2.0,
        "hard_request_timeout_seconds": 210,
    }
    if not isinstance(traffic, dict):
        raise ValueError("spec traffic object is missing")
    for field, expected in expected_traffic.items():
        if traffic.get(field) != expected:
            raise ValueError(f"traffic.{field} must remain {expected}")
    expected_streams = {
        "dev": (120, 12, 80, {"8192": 32, "16384": 32, "32768": 16}),
        "final": (180, 51, 340, {"8192": 136, "16384": 136, "32768": 68}),
    }
    for split, expected in expected_streams.items():
        row = traffic.get(split)
        actual = (
            row.get("submission_window_seconds") if isinstance(row, dict) else None,
            row.get("turns_per_user") if isinstance(row, dict) else None,
            row.get("tasks_per_family") if isinstance(row, dict) else None,
            row.get("length_bucket_counts_per_family") if isinstance(row, dict) else None,
        )
        if actual != expected:
            raise ValueError(f"traffic.{split} differs from the frozen contract")
    policies = spec.get("families")
    for family, cap, slo in (
        ("numeric", 128, 90),
        ("code", 2048, 180),
        ("knowledge", 32, 60),
    ):
        row = policies.get(family) if isinstance(policies, dict) else None
        if not isinstance(row, dict) or (row.get("max_tokens_cap"), row.get("slo_seconds")) != (cap, slo):
            raise ValueError(f"{family} cap/SLO differs from the frozen contract")
    if tuple(spec.get("length_buckets", {}).get("targets", [])) != TARGETS:
        raise ValueError("length bucket targets must remain 8192/16384/32768")
    if spec.get("evaluators", {}).get("swebench", {}).get("revision") != SWE_EVALUATOR_REVISION:
        raise ValueError("SWE-bench evaluator revision changed")
    if spec.get("evaluators", {}).get("finqa_numeric", {}).get("revision") != FINQA_EVALUATOR_REVISION:
        raise ValueError("FinQA evaluator revision changed")


def hf_url(repository: str, revision: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{path}"


def github_raw(repository: str, revision: str, path: str) -> str:
    return f"https://raw.githubusercontent.com/{repository}/{revision}/{path}"


def source_plan(spec: dict[str, Any]) -> dict[str, tuple[str, str]]:
    sources = spec["sources"]
    evaluators = spec["evaluators"]
    return {
        "docfinqa_validation.json": (
            hf_url(sources["docfinqa"]["repository"], sources["docfinqa"]["revision"], "dev.json"),
            "DocFinQA validation",
        ),
        "docfinqa_test.json": (
            hf_url(sources["docfinqa"]["repository"], sources["docfinqa"]["revision"], "test.json"),
            "DocFinQA test",
        ),
        "swe_bm25_test.parquet": (
            hf_url(sources["swe_bm25"]["repository"], sources["swe_bm25"]["revision"], "data/test-00000-of-00001.parquet"),
            "SWE-bench BM25 test",
        ),
        "swe_verified_test.parquet": (
            hf_url(sources["swe_verified"]["repository"], sources["swe_verified"]["revision"], "data/test-00000-of-00001.parquet"),
            "SWE-bench Verified test",
        ),
        "swe_verified_eval.yaml": (
            hf_url(sources["swe_verified"]["repository"], sources["swe_verified"]["revision"], "eval.yaml"),
            "SWE-bench Verified evaluation metadata",
        ),
        "longbench_v2_data.json": (
            hf_url(sources["longbench_v2"]["repository"], sources["longbench_v2"]["revision"], "data.json"),
            "LongBench-v2 train",
        ),
        "finqa_evaluate.py": (
            github_raw(evaluators["finqa_numeric"]["repository"], evaluators["finqa_numeric"]["revision"], evaluators["finqa_numeric"]["path"]),
            "official FinQA evaluator",
        ),
        "swebench_init.py": (
            github_raw(evaluators["swebench"]["repository"], evaluators["swebench"]["revision"], "swebench/__init__.py"),
            "official SWE-bench version declaration",
        ),
        "swebench_harness_utils.py": (
            github_raw(evaluators["swebench"]["repository"], evaluators["swebench"]["revision"], "swebench/harness/utils.py"),
            "official SWE-bench local dataset/TestSpec loader",
        ),
    }


def theoretical_contract(spec: dict[str, Any]) -> dict[str, Any]:
    traffic = spec["traffic"]
    result: dict[str, Any] = {}
    for split in ("dev", "final"):
        stream = traffic[split]
        total = int(stream["tasks_per_family"]) * 3
        window = float(stream["submission_window_seconds"])
        turns = int(stream["turns_per_user"])
        zero_service_last_submission = (
            (traffic["user_count"] - 1) * traffic["first_submission_stagger_seconds"]
            + (turns - 1) * traffic["think_time_seconds"]
        )
        result[split] = {
            "frozen_task_count": total,
            "submission_window_seconds": window,
            "all_correct_slo_goodput_ceiling_per_second": total / window,
            "all_correct_slo_goodput_ceiling_per_hour": total / window * 3600.0,
            "zero_service_last_submission_offset_seconds": zero_service_last_submission,
            "no_wrap_queue_note": (
                "A measured run is invalid if a per-user frozen queue is exhausted "
                "while another submission would still fall before window close."
            ),
        }
    return result


def plan_payload(args: argparse.Namespace, spec: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "plan-only",
        "spec": str(args.spec.expanduser().resolve()),
        "source_cache": str(args.source_cache.expanduser().absolute()),
        "output_dir": str(args.output_dir.expanduser().absolute()),
        "tokenizer": args.tokenizer or spec["tokenizer"]["default_path"],
        "sources": [
            {"cache_file": name, "url": url, "purpose": purpose}
            for name, (url, purpose) in source_plan(spec).items()
        ],
        "traffic": spec["traffic"],
        "length_buckets": spec["length_buckets"],
        "theoretical": theoretical_contract(spec),
        "downloads_started": False,
        "output_written": False,
    }


def download_cached(cache: Path, name: str, url: str, timeout: float) -> Path:
    target = cache / name
    if target.is_file() and target.stat().st_size > 0:
        return target
    cache.mkdir(parents=True, exist_ok=True)
    temporary = cache / f".{name}.part-{os.getpid()}"
    request = urllib.request.Request(url, headers={"User-Agent": "FreeToken-long-goodput-v2"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        if temporary.stat().st_size == 0:
            raise ValueError(f"downloaded empty source: {url}")
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def iter_json_array(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as handle:
        buffer = ""
        position = 0
        eof = False

        def refill() -> None:
            nonlocal buffer, position, eof
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(1024 * 1024)
            if chunk:
                buffer += chunk
            else:
                eof = True

        refill()
        while position < len(buffer) and buffer[position].isspace():
            position += 1
        if position >= len(buffer):
            refill()
        if position >= len(buffer) or buffer[position] != "[":
            raise ValueError(f"expected top-level JSON array: {path}")
        position += 1
        first = True
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    break
                refill()
            if position < len(buffer) and buffer[position] == "]":
                position += 1
                break
            if not first:
                if position >= len(buffer):
                    refill()
                if position >= len(buffer) or buffer[position] != ",":
                    raise ValueError(f"expected comma in JSON array: {path}")
                position += 1
                while True:
                    while position < len(buffer) and buffer[position].isspace():
                        position += 1
                    if position < len(buffer) or eof:
                        break
                    refill()
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    position = end
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise ValueError(f"invalid/truncated JSON array: {path}")
                    refill()
            if not isinstance(value, dict):
                raise ValueError(f"source array contains non-object row: {path}")
            yield value
            first = False
            if position > 8 * 1024 * 1024:
                refill()
        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                raise ValueError(f"unexpected data after JSON array: {path}")
            if eof:
                break
            refill()


def read_parquet(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("building v2 requires pyarrow") from exc
    table = parquet.read_table(path)
    rows = table.to_pylist()
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Parquet source contains non-object rows: {path}")
    return list(table.schema.names), rows


def exact_fields(row: dict[str, Any], expected: set[str], label: str) -> None:
    if set(row) != expected:
        missing = sorted(expected - set(row))
        extra = sorted(set(row) - expected)
        raise ValueError(f"{label} schema differs; missing={missing}, extra={extra}")


def require_text(row: dict[str, Any], field_name: str, label: str) -> str:
    value = row.get(field_name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label}.{field_name} must be non-empty text")
    return value


def validate_evaluator_sources(paths: dict[str, Path]) -> None:
    finqa = paths["finqa_evaluate.py"].read_text(encoding="utf-8")
    for marker in (
        'text = text.replace(",", "")',
        'text = text.replace("%", "")',
        "num = num / 100.0",
        "this_res = round(this_res, 5)",
    ):
        if marker not in finqa:
            raise ValueError(f"fixed FinQA evaluator lacks required normalization marker: {marker}")
    swe_init = paths["swebench_init.py"].read_text(encoding="utf-8")
    if SWE_EVALUATOR_VERSION not in swe_init:
        raise ValueError("fixed SWE-bench evaluator does not declare version 5.0.1")
    swe_utils = paths["swebench_harness_utils.py"].read_text(encoding="utf-8")
    for marker in ('name.endswith(".parquet")', 'image=instance["image"]', "eval_script"):
        if marker not in swe_utils:
            raise ValueError(f"fixed SWE-bench evaluator lacks required v5 marker: {marker}")
    if not paths["swe_verified_eval.yaml"].read_text(encoding="utf-8").strip():
        raise ValueError("fixed SWE-bench Verified eval.yaml is empty")


def lexical_tokens(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return TOKEN_RE.findall(normalized)


def normalize_forbidden_text(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(text.split()).strip()


def split_paragraphs(text: str) -> list[str]:
    pieces = [piece.strip() for piece in re.split(r"\n[ \t]*\n+", text) if piece.strip()]
    return pieces or ([text.strip()] if text.strip() else [])


def finqa_number(value: str) -> float:
    text = value.replace("$", "").strip().split("(", 1)[0].strip().replace(",", "")
    percent = "%" in text
    if percent:
        text = text.replace("%", "")
    number = float(text)
    if not math.isfinite(number):
        raise ValueError(f"non-finite FinQA answer: {value!r}")
    if percent:
        number /= 100.0
    return round(number, 5)


def numeric_candidates(path: Path, split: str, revision: str) -> Iterator[Candidate]:
    count = 0
    for ordinal, row in enumerate(iter_json_array(path)):
        exact_fields(row, DOCFINQA_FIELDS, f"DocFinQA {split} row {ordinal}")
        context = require_text(row, "Context", f"DocFinQA {split} row {ordinal}")
        question = require_text(row, "Question", f"DocFinQA {split} row {ordinal}")
        if not isinstance(row["Program"], str):
            raise ValueError(f"DocFinQA {split} row {ordinal}.Program must be text")
        answer = require_text(row, "Answer", f"DocFinQA {split} row {ordinal}")
        try:
            normalized = finqa_number(answer)
        except (ValueError, OverflowError):
            continue
        count += 1
        yield Candidate(
            family="numeric",
            task_id=f"docfinqa/{split}/{ordinal}",
            source="kensho/DocFinQA",
            source_revision=revision,
            source_split=split,
            source_ordinal=ordinal,
            question=question,
            context=context,
            reference={
                "kind": "finqa_numeric",
                "official_answer": answer,
                "normalized_value": normalized,
                "normalization": "official_str_to_num_then_python_round_5_v1",
            },
        )
    expected = 780 if split == "validation" else 922
    if ordinal + 1 != expected:
        raise ValueError(f"DocFinQA {split} row count is {ordinal + 1}, expected {expected}")
    if count == 0:
        raise ValueError(f"DocFinQA {split} has no numeric answers")


def repository_file_blocks(text: str, instance_id: str) -> list[tuple[str, str]]:
    blocks: list[tuple[str, str]] = []
    opened: tuple[str, re.Match[str]] | None = None
    for marker in FILE_MARKER_RE.finditer(text):
        kind, path = marker.groups()
        if kind == "start":
            if opened is not None:
                raise ValueError(f"BM25 {instance_id} has nested repository path markers")
            opened = (path, marker)
            continue
        if opened is None:
            raise ValueError(f"BM25 {instance_id} has an unmatched repository end marker")
        start_path, start_marker = opened
        if path != start_path:
            raise ValueError(f"BM25 {instance_id} repository start/end paths differ")
        blocks.append((path, text[start_marker.start():marker.end()].strip()))
        opened = None
    if opened is not None:
        raise ValueError(f"BM25 {instance_id} has an unmatched repository start marker")
    if not blocks:
        raise ValueError(f"BM25 {instance_id} has no repository path blocks")
    return blocks


def extract_repository_context(text: str, instance_id: str) -> str:
    return "\n\n".join(
        block for _, block in repository_file_blocks(text, instance_id)
    )


def swe_candidates(
    bm25_path: Path, verified_path: Path, bm_revision: str, verified_revision: str
) -> list[Candidate]:
    bm_schema, bm_rows = read_parquet(bm25_path)
    verified_schema, verified_rows = read_parquet(verified_path)
    if set(bm_schema) != BM25_FIELDS:
        raise ValueError("SWE BM25 Parquet schema differs from the fixed 13-string-field contract")
    if set(verified_schema) != VERIFIED_FIELDS:
        raise ValueError("SWE-bench Verified Parquet schema differs from the fixed v5 contract")
    if len(bm_rows) != 2294 or len(verified_rows) != 500:
        raise ValueError(f"SWE source counts changed: BM25={len(bm_rows)}, Verified={len(verified_rows)}")
    bm_by_id: dict[str, dict[str, Any]] = {}
    for ordinal, row in enumerate(bm_rows):
        exact_fields(row, BM25_FIELDS, f"SWE BM25 row {ordinal}")
        instance_id = require_text(row, "instance_id", f"SWE BM25 row {ordinal}")
        if instance_id in bm_by_id:
            raise ValueError(f"duplicate SWE BM25 instance_id: {instance_id}")
        if not all(isinstance(value, str) for value in row.values()):
            raise ValueError(f"SWE BM25 {instance_id} has a non-string field")
        bm_by_id[instance_id] = row
    verified_by_id: dict[str, dict[str, Any]] = {}
    for ordinal, row in enumerate(verified_rows):
        exact_fields(row, VERIFIED_FIELDS, f"SWE Verified row {ordinal}")
        instance_id = require_text(row, "instance_id", f"SWE Verified row {ordinal}")
        if instance_id in verified_by_id:
            raise ValueError(f"duplicate SWE Verified instance_id: {instance_id}")
        for field_name in VERIFIED_EVAL_FIELDS - {"FAIL_TO_PASS", "PASS_TO_PASS"}:
            require_text(row, field_name, f"SWE Verified {instance_id}")
        for field_name in ("FAIL_TO_PASS", "PASS_TO_PASS"):
            if not isinstance(row[field_name], list) or not all(isinstance(item, str) for item in row[field_name]):
                raise ValueError(f"SWE Verified {instance_id}.{field_name} must be list<string>")
        image = row["image"]
        if re.fullmatch(r"swebench/sweb\.eval\.x86_64\..+:latest", image) is None:
            raise ValueError(f"SWE Verified {instance_id} has unexpected image reference: {image!r}")
        verified_by_id[instance_id] = row
    intersection = set(bm_by_id).intersection(verified_by_id)
    if len(intersection) != 500 or set(verified_by_id) != intersection:
        raise ValueError(
            "fixed SWE join changed: expected all 500 Verified ids in the BM25 test split"
        )
    candidates: list[Candidate] = []
    for ordinal, instance_id in enumerate(sorted(intersection)):
        bm25 = bm_by_id[instance_id]
        verified = verified_by_id[instance_id]
        if bm25["problem_statement"] != verified["problem_statement"]:
            raise ValueError(f"SWE problem_statement differs across sources for {instance_id}")
        question = require_text(verified, "problem_statement", f"SWE Verified {instance_id}")
        context = extract_repository_context(
            require_text(bm25, "text", f"SWE BM25 {instance_id}"), instance_id
        )
        candidates.append(
            Candidate(
                family="code",
                task_id=f"swebench/{instance_id}",
                source="SWE-bench BM25 40K intersect Verified",
                source_revision=f"{bm_revision}+{verified_revision}",
                source_split="test",
                source_ordinal=ordinal,
                question=question,
                context=context,
                context_kind="repository",
                reference={"kind": "swebench_verified", "instance_id": instance_id},
            )
        )
    return candidates


def longbench_candidates(path: Path, revision: str) -> list[Candidate]:
    rows: list[dict[str, Any]] = []
    code_rows = 0
    for ordinal, row in enumerate(iter_json_array(path)):
        exact_fields(row, LONGBENCH_FIELDS, f"LongBench-v2 row {ordinal}")
        if not all(isinstance(value, str) for value in row.values()):
            raise ValueError(f"LongBench-v2 row {ordinal} has a non-string field")
        if row["domain"] == CODE_REPOSITORY_DOMAIN:
            code_rows += 1
            if row["sub_domain"] != CODE_REPOSITORY_SUBDOMAIN:
                raise ValueError("LongBench-v2 code domain/sub-domain alias changed")
            continue
        if row["answer"] not in "ABCD" or len(row["answer"]) != 1:
            raise ValueError(f"LongBench-v2 row {ordinal} answer must be A-D")
        rows.append({"ordinal": ordinal, **row})
    if ordinal + 1 != 503:
        raise ValueError(f"LongBench-v2 row count is {ordinal + 1}, expected 503")
    if code_rows != 50 or len(rows) != 453:
        raise ValueError(
            f"LongBench-v2 exclusion changed: code={code_rows}, remaining={len(rows)}"
        )
    ids = [require_text(row, "_id", "LongBench-v2") for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("LongBench-v2 _id values must be unique")
    candidates: list[Candidate] = []
    for row in sorted(rows, key=lambda item: item["_id"]):
        ordinal = int(row["ordinal"])
        task_id = row["_id"]
        candidates.append(
            Candidate(
                family="knowledge",
                task_id=f"longbench-v2/{task_id}",
                source="zai-org/LongBench-v2",
                source_revision=revision,
                source_split="train",
                source_ordinal=ordinal,
                question=require_text(row, "question", f"LongBench-v2 {task_id}"),
                context=require_text(row, "context", f"LongBench-v2 {task_id}"),
                options=tuple(require_text(row, f"choice_{letter}", f"LongBench-v2 {task_id}") for letter in "ABCD"),
                reference={"kind": "choice", "value": row["answer"]},
            )
        )
    return candidates


def select_bucketed(
    candidates: Iterable[Candidate],
    counts: dict[int, int],
    prompts: PromptBuilder,
    label: str,
    reserved_questions: set[str] | None = None,
) -> dict[int, list[BuiltTask]]:
    selected = {target: [] for target in TARGETS}
    priority = tuple(sorted(TARGETS, reverse=True))
    seen = 0
    for candidate in candidates:
        seen += 1
        question_key = normalize_forbidden_text(candidate.question)
        if reserved_questions is not None and question_key in reserved_questions:
            continue
        for target in priority:
            if len(selected[target]) >= counts.get(target, 0):
                continue
            built = prompts.build(candidate, target)
            if built is not None:
                selected[target].append(built)
                if reserved_questions is not None:
                    reserved_questions.add(question_key)
                break
        if all(len(selected[target]) == counts.get(target, 0) for target in TARGETS):
            break
    missing = {
        target: counts.get(target, 0) - len(selected[target])
        for target in TARGETS
        if len(selected[target]) != counts.get(target, 0)
    }
    if missing:
        raise ValueError(f"{label} lacks eligible real-text prompts after {seen} candidates: {missing}")
    return selected


def add_counts(left: dict[int, int], right: dict[int, int]) -> dict[int, int]:
    return {target: int(left.get(target, 0)) + int(right.get(target, 0)) for target in TARGETS}


def split_selected(
    selected: dict[int, list[BuiltTask]], first_counts: dict[int, int]
) -> tuple[dict[int, list[BuiltTask]], dict[int, list[BuiltTask]]]:
    first: dict[int, list[BuiltTask]] = {}
    second: dict[int, list[BuiltTask]] = {}
    for target in TARGETS:
        boundary = first_counts.get(target, 0)
        first[target] = selected[target][:boundary]
        second[target] = selected[target][boundary:]
    return first, second


def counts_from_spec(spec: dict[str, Any], split: str) -> dict[int, int]:
    raw = spec["traffic"][split]["length_bucket_counts_per_family"]
    return {int(target): int(count) for target, count in raw.items()}


def warmup_counts(family: str) -> dict[int, int]:
    return {
        "numeric": {8192: 3, 16384: 2, 32768: 2},
        "code": {8192: 2, 16384: 3, 32768: 2},
        "knowledge": {8192: 2, 16384: 2, 32768: 2},
    }[family]


def balanced_bucket_sequence(counts: dict[int, int]) -> list[int]:
    remaining = dict(counts)
    result: list[int] = []
    while sum(remaining.values()):
        for target in TARGETS:
            if remaining.get(target, 0):
                result.append(target)
                remaining[target] -= 1
    return result


def assign_tasks(
    split: str,
    by_family: dict[str, dict[int, list[BuiltTask]]],
    turns: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    sequences = {
        family: balanced_bucket_sequence(
            {target: len(by_family[family][target]) for target in TARGETS}
        )
        for family in FAMILIES
    }
    indices = {family: 0 for family in FAMILIES}
    queues = {
        family: {target: list(by_family[family][target]) for target in TARGETS}
        for family in FAMILIES
    }
    users = [{"user_index": user, "task_ids": []} for user in range(20)]
    rows: list[dict[str, Any]] = []
    for turn in range(turns):
        for user in range(20):
            family = FAMILIES[(user + turn) % 3]
            occurrence = indices[family]
            if occurrence >= len(sequences[family]):
                raise ValueError(f"{split} has too few {family} tasks for assignment")
            target = sequences[family][occurrence]
            indices[family] += 1
            built = queues[family][target].pop(0)
            row = built.row(split)
            users[user]["task_ids"].append(row["task_id"])
            rows.append(row)
    if any(queue for family in FAMILIES for queue in queues[family].values()):
        raise ValueError(f"{split} assignment did not consume every selected task")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError(f"{split} assignment contains duplicate task ids")
    return users, rows


def forbidden_rows(tasks_by_split: dict[str, list[dict[str, Any]]], built_lookup: dict[str, BuiltTask]) -> Iterator[dict[str, Any]]:
    for split in ("warmup", "dev", "final"):
        for task in tasks_by_split[split]:
            built = built_lookup[task["task_id"]]
            candidate = built.candidate
            for kind, text in (
                ("question", candidate.question),
                ("context", built.selected_context),
                ("final_prompt", built.user_prompt),
            ):
                yield {
                    "family": candidate.family,
                    "task_id": candidate.task_id,
                    "split": split,
                    "kind": kind,
                    "text": text,
                }


def write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec_path = args.spec.expanduser().resolve()
    spec = load_spec(spec_path)
    if args.plan_only:
        print(json.dumps(plan_payload(args, spec), indent=2, ensure_ascii=False))
        return 0

    output = args.output_dir.expanduser().absolute()
    cache = args.source_cache.expanduser().absolute()
    if output.exists():
        raise FileExistsError(
            f"fresh atomic publication requires a nonexistent output directory: {output}"
        )
    source_paths = {
        name: download_cached(cache, name, url, args.download_timeout)
        for name, (url, _) in source_plan(spec).items()
    }
    validate_evaluator_sources(source_paths)

    tokenizer_name = args.tokenizer or spec["tokenizer"]["default_path"]
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("building v2 requires transformers") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("the frozen Qwen tokenizer must expose a chat_template")
    prompts = PromptBuilder(tokenizer, spec)

    sources = spec["sources"]
    dev_counts = counts_from_spec(spec, "dev")
    final_counts = counts_from_spec(spec, "final")
    selected_by_split: dict[str, dict[str, dict[int, list[BuiltTask]]]] = {
        split: {} for split in ("warmup", "dev", "final")
    }

    numeric_combined = select_bucketed(
        numeric_candidates(
            source_paths["docfinqa_validation.json"],
            "validation",
            sources["docfinqa"]["revision"],
        ),
        add_counts(dev_counts, warmup_counts("numeric")),
        prompts,
        "DocFinQA validation dev+warmup",
    )
    numeric_dev, numeric_warmup = split_selected(numeric_combined, dev_counts)
    selected_by_split["dev"]["numeric"] = numeric_dev
    selected_by_split["warmup"]["numeric"] = numeric_warmup
    selected_by_split["final"]["numeric"] = select_bucketed(
        numeric_candidates(
            source_paths["docfinqa_test.json"],
            "test",
            sources["docfinqa"]["revision"],
        ),
        final_counts,
        prompts,
        "DocFinQA test final",
    )

    swe = swe_candidates(
        source_paths["swe_bm25_test.parquet"],
        source_paths["swe_verified_test.parquet"],
        sources["swe_bm25"]["revision"],
        sources["swe_verified"]["revision"],
    )
    swe_dev_pool = [candidate for index, candidate in enumerate(swe) if index % 5 == 0]
    swe_final_pool = [candidate for index, candidate in enumerate(swe) if index % 5 != 0]
    swe_combined = select_bucketed(
        swe_dev_pool,
        add_counts(dev_counts, warmup_counts("code")),
        prompts,
        "SWE-bench dev+warmup partition",
    )
    swe_dev, swe_warmup = split_selected(swe_combined, dev_counts)
    selected_by_split["dev"]["code"] = swe_dev
    selected_by_split["warmup"]["code"] = swe_warmup
    selected_by_split["final"]["code"] = select_bucketed(
        swe_final_pool, final_counts, prompts, "SWE-bench final partition"
    )

    longbench = longbench_candidates(
        source_paths["longbench_v2_data.json"], sources["longbench_v2"]["revision"]
    )
    long_dev_pool = [candidate for index, candidate in enumerate(longbench) if index % 5 == 0]
    long_final_pool = [candidate for index, candidate in enumerate(longbench) if index % 5 != 0]
    reserved_longbench_questions: set[str] = set()
    selected_by_split["warmup"]["knowledge"] = select_bucketed(
        long_dev_pool,
        warmup_counts("knowledge"),
        prompts,
        "LongBench-v2 warmup partition",
        reserved_longbench_questions,
    )
    selected_by_split["dev"]["knowledge"] = select_bucketed(
        long_dev_pool,
        dev_counts,
        prompts,
        "LongBench-v2 dev partition",
        reserved_longbench_questions,
    )
    selected_by_split["final"]["knowledge"] = select_bucketed(
        long_final_pool,
        final_counts,
        prompts,
        "LongBench-v2 final partition",
        reserved_longbench_questions,
    )

    assignments: dict[str, list[dict[str, Any]]] = {}
    task_rows: dict[str, list[dict[str, Any]]] = {}
    assignments["warmup"], task_rows["warmup"] = assign_tasks(
        "warmup", selected_by_split["warmup"], 1
    )
    assignments["dev"], task_rows["dev"] = assign_tasks(
        "dev", selected_by_split["dev"], spec["traffic"]["dev"]["turns_per_user"]
    )
    assignments["final"], task_rows["final"] = assign_tasks(
        "final", selected_by_split["final"], spec["traffic"]["final"]["turns_per_user"]
    )
    all_ids = [row["task_id"] for rows in task_rows.values() for row in rows]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("warmup/dev/final task ids overlap")
    built_lookup = {
        built.candidate.task_id: built
        for split in selected_by_split.values()
        for family in split.values()
        for bucket in family.values()
        for built in bucket
    }

    counts = {
        split: {
            family: {
                str(target): sum(
                    row["family"] == family and row["length_bucket_tokens"] == target
                    for row in rows
                )
                for target in TARGETS
            }
            for family in FAMILIES
        }
        for split, rows in task_rows.items()
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "spec": {"file": spec_path.name, "schema": SPEC_SCHEMA},
        "sources": spec["sources"],
        "source_cache": {
            "directory": str(cache),
            "files": {name: name for name in source_paths},
        },
        "evaluators": spec["evaluators"],
        "tokenizer": {
            "identifier": tokenizer_name,
            "class": type(tokenizer).__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", tokenizer_name)),
            "add_generation_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "counting": "apply_chat_template(tokenize=True)",
        },
        "length_buckets": spec["length_buckets"],
        "traffic": spec["traffic"],
        "request_policy": {
            "system_prompt": SYSTEM_PROMPT,
            "greedy": True,
            "enable_thinking": False,
            "max_sequence_length": 40960,
            "families": spec["families"],
        },
        "selection": {
            "docfinqa": "validation supplies dev+warmup; test supplies final; source order",
            "swe": "sorted Verified-intersection id ordinal modulo 5: 0 dev+warmup, 1-4 final",
            "longbench_v2": (
                "exclude the exact 50-row code-repository domain, sort by _id, then "
                "ordinal modulo 5: 0 dev+warmup, 1-4 final; reserve normalized "
                "Question values in warmup, dev, final priority order"
            ),
            "retrieval_query": "question only",
            "retrieval_exclusions": [
                "DocFinQA Program",
                "all answers",
                "gold patches",
                "test patches",
                "public tests",
                "supporting facts",
                "SWE hints_text",
            ],
        },
        "task_files": {
            "warmup": "tasks_warmup.jsonl",
            "dev": "tasks_dev.jsonl",
            "final": "tasks_final.jsonl",
        },
        "assignments": assignments,
        "counts": counts,
        "training_forbidden_texts": {
            "file": "training_forbidden_texts.jsonl",
            "schema": FORBIDDEN_SCHEMA,
            "normalization": "nfkc_casefold_unicode_whitespace_v1",
            "substring_filter_kinds": ["question", "context", "final_prompt"],
            "row_fields": ["family", "task_id", "split", "kind", "text"],
            "excluded_reference_material": [
                "DocFinQA Program",
                "answers",
                "gold patches",
                "test patches",
                "public tests",
                "supporting facts",
            ],
        },
        "theoretical": theoretical_contract(spec),
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.build-", dir=output.parent))
    try:
        shutil.copy2(spec_path, temporary / spec_path.name)
        for split, rows in task_rows.items():
            write_jsonl(temporary / f"tasks_{split}.jsonl", rows)
        write_jsonl(
            temporary / "training_forbidden_texts.jsonl",
            forbidden_rows(task_rows, built_lookup),
        )
        write_json(temporary / "manifest.json", manifest)
        temporary.replace(output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "manifest": str(output / "manifest.json"),
                "task_counts": {split: len(rows) for split, rows in task_rows.items()},
                "counts": counts,
                "atomic_publication": True,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
