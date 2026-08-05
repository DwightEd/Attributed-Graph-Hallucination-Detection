"""Strict label-blind RAGTruth metadata loading for CEPT experiments."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RAGTruthExample:
    response_id: str
    source_id: str
    split: str
    task_type: str
    prompt: str
    source_info: object
    response: str
    generator_model: str | None

    def source_record(self) -> dict[str, object]:
        """Return only fields permitted to enter layout/teacher construction."""

        return {
            "source_id": self.source_id,
            "task_type": self.task_type,
            "prompt": self.prompt,
            "source_info": self.source_info,
        }


def _read_jsonl(path: str | Path) -> list[Mapping[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"RAGTruth JSONL file is absent: {source}")
    records: list[Mapping[str, object]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {source}:{line_number}") from error
            if not isinstance(value, Mapping):
                raise TypeError(f"RAGTruth row must be an object: {source}:{line_number}")
            records.append(value)
    if not records:
        raise ValueError(f"RAGTruth JSONL file is empty: {source}")
    return records


def _nonempty(record: Mapping[str, object], key: str, *, context: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{context} requires non-empty {key}")
    return value.strip() if key in {"id", "source_id", "split"} else value


def load_ragtruth_examples(
    response_path: str | Path,
    source_path: str | Path,
    *,
    split: str,
    response_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[RAGTruthExample]:
    """Join official metadata while physically projecting all labels away."""

    requested_split = str(split).strip().casefold()
    if requested_split not in {"train", "test"}:
        raise ValueError("RAGTruth split must be train or test")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    sources: dict[str, Mapping[str, object]] = {}
    for record in _read_jsonl(source_path):
        source_id = _nonempty(record, "source_id", context="RAGTruth source")
        if source_id in sources:
            raise ValueError(f"duplicate RAGTruth source_id: {source_id}")
        for required in ("prompt", "task_type", "source_info"):
            if required not in record:
                raise ValueError(f"RAGTruth source {source_id} is missing {required}")
        _nonempty(record, "prompt", context=f"RAGTruth source {source_id}")
        _nonempty(record, "task_type", context=f"RAGTruth source {source_id}")
        sources[source_id] = record

    responses: dict[str, Mapping[str, object]] = {}
    source_splits: dict[str, set[str]] = {}
    response_order: list[str] = []
    for record in _read_jsonl(response_path):
        response_id = _nonempty(record, "id", context="RAGTruth response")
        if response_id in responses:
            raise ValueError(f"duplicate RAGTruth response id: {response_id}")
        source_id = _nonempty(
            record, "source_id", context=f"RAGTruth response {response_id}"
        )
        row_split = _nonempty(
            record, "split", context=f"RAGTruth response {response_id}"
        ).casefold()
        if row_split not in {"train", "test"}:
            raise ValueError(f"RAGTruth response {response_id} has invalid split")
        _nonempty(record, "response", context=f"RAGTruth response {response_id}")
        if source_id not in sources:
            raise ValueError(
                f"RAGTruth response {response_id} references absent source {source_id}"
            )
        source_splits.setdefault(source_id, set()).add(row_split)
        responses[response_id] = record
        response_order.append(response_id)
    leakage = sorted(
        source_id for source_id, values in source_splits.items() if len(values) != 1
    )
    if leakage:
        raise ValueError(
            "RAGTruth source IDs cross official splits: " + ", ".join(leakage[:5])
        )

    if response_ids is None:
        selected_ids = [
            response_id
            for response_id in response_order
            if str(responses[response_id]["split"]).casefold() == requested_split
        ]
    else:
        selected_ids = [str(value).strip() for value in response_ids]
        if not selected_ids or any(not value for value in selected_ids):
            raise ValueError("response_ids must contain non-empty IDs")
        if len(set(selected_ids)) != len(selected_ids):
            raise ValueError("response_ids must be unique")
        absent = [value for value in selected_ids if value not in responses]
        if absent:
            raise ValueError("requested RAGTruth responses are absent: " + ", ".join(absent))
        wrong_split = [
            value
            for value in selected_ids
            if str(responses[value]["split"]).casefold() != requested_split
        ]
        if wrong_split:
            raise ValueError(
                "requested RAGTruth responses have the wrong official split: "
                + ", ".join(wrong_split)
            )
    if limit is not None:
        selected_ids = selected_ids[:limit]
    if not selected_ids:
        raise ValueError(f"no RAGTruth examples selected from {requested_split}")

    examples: list[RAGTruthExample] = []
    for response_id in selected_ids:
        response_record = responses[response_id]
        source_id = str(response_record["source_id"]).strip()
        source_record = sources[source_id]
        model = response_record.get("model")
        examples.append(
            RAGTruthExample(
                response_id=response_id,
                source_id=source_id,
                split=requested_split,
                task_type=str(source_record["task_type"]),
                prompt=str(source_record["prompt"]),
                source_info=source_record["source_info"],
                response=str(response_record["response"]),
                generator_model=None if model is None else str(model),
            )
        )
    return examples


def select_balanced_pilot(
    examples: Sequence[RAGTruthExample], *, limit: int, seed: int
) -> list[RAGTruthExample]:
    """Stable label-free round-robin selection over task/generator strata."""

    if limit <= 0:
        raise ValueError("pilot limit must be positive")
    strata: dict[tuple[str, str], list[RAGTruthExample]] = {}
    for example in examples:
        key = (
            example.task_type.casefold(),
            (example.generator_model or "unknown").casefold(),
        )
        strata.setdefault(key, []).append(example)
    if not strata:
        raise ValueError("pilot selection received no examples")

    def rank(example: RAGTruthExample) -> str:
        return hashlib.sha256(
            f"{seed}\0{example.source_id}\0{example.response_id}".encode()
        ).hexdigest()

    queues = {
        key: sorted(values, key=rank) for key, values in sorted(strata.items())
    }
    offsets = {key: 0 for key in queues}
    used_sources: set[str] = set()
    selected: list[RAGTruthExample] = []
    while len(selected) < min(limit, len(examples)):
        progress = False
        for key, queue in queues.items():
            while offsets[key] < len(queue):
                candidate = queue[offsets[key]]
                offsets[key] += 1
                if candidate.source_id in used_sources:
                    continue
                selected.append(candidate)
                used_sources.add(candidate.source_id)
                progress = True
                break
            if len(selected) == limit:
                break
        if not progress:
            break
    if not selected:
        raise ValueError("pilot selection found no source-unique examples")
    return selected


__all__ = [
    "RAGTruthExample",
    "load_ragtruth_examples",
    "select_balanced_pilot",
]
