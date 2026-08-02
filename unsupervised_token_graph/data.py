"""Dataset adapters that keep graph inputs separate from evaluation labels."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class TokenGraphExample:
    """One label-free passage/question/answer sequence."""

    example_id: str
    pair_id: str
    dataset: str
    passage: str
    question: str
    answer: str
    text: str
    segment_char_spans: Mapping[str, tuple[int, int]]
    metadata: Mapping[str, Any] = field(default_factory=dict)


def _stable_id(*parts: str) -> str:
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:20]


def compose_example(
    passage: str,
    question: str,
    answer: str,
    *,
    example_id: str,
    pair_id: str | None = None,
    dataset: str = "unknown",
    metadata: Mapping[str, Any] | None = None,
) -> TokenGraphExample:
    """Concatenate all three segments and retain exact content boundaries."""

    values = {
        "passage": str(passage),
        "question": str(question),
        "answer": str(answer),
    }
    parts: list[str] = []
    spans: dict[str, tuple[int, int]] = {}
    for index, (name, value) in enumerate(values.items()):
        if index:
            parts.append("\n\n")
        parts.append(f"{name.title()}:\n")
        start = sum(len(part) for part in parts)
        parts.append(value)
        spans[name] = (start, start + len(value))

    return TokenGraphExample(
        example_id=str(example_id),
        pair_id=str(pair_id or example_id),
        dataset=str(dataset),
        passage=values["passage"],
        question=values["question"],
        answer=values["answer"],
        text="".join(parts),
        segment_char_spans=spans,
        metadata=dict(metadata or {}),
    )


def _read_records(path: str | Path) -> list[dict[str, Any]]:
    text = Path(path).read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        value = json.loads(text)
        if not isinstance(value, list):
            raise ValueError(f"Expected a JSON array in {path}")
        return [dict(record) for record in value]
    return [dict(json.loads(line)) for line in text.splitlines() if line.strip()]


def load_halueval_qa(
    path: str | Path,
) -> tuple[list[TokenGraphExample], dict[str, int]]:
    """Expand each HaluEval-QA pair; return labels in a separate mapping."""

    examples: list[TokenGraphExample] = []
    evaluation_labels: dict[str, int] = {}
    for row_index, row in enumerate(_read_records(path)):
        passage = str(row["knowledge"])
        question = str(row["question"])
        pair_id = f"halueval-{_stable_id(str(row_index), passage, question)}"
        candidates = (
            (str(row["right_answer"]), 0),
            (str(row["hallucinated_answer"]), 1),
        )
        pair_examples: list[TokenGraphExample] = []
        for answer, label in candidates:
            example_id = f"halueval-{_stable_id(pair_id, answer)}"
            pair_examples.append(
                compose_example(
                    passage,
                    question,
                    answer,
                    example_id=example_id,
                    pair_id=pair_id,
                    dataset="halueval_qa",
                )
            )
            evaluation_labels[example_id] = label
        examples.extend(sorted(pair_examples, key=lambda item: item.example_id))
    return examples, evaluation_labels


_BOOL_ANSWER_RE = re.compile(r"^\s*(yes|no)\b", re.IGNORECASE)


def parse_bool_answer(answer: str) -> bool:
    """Parse a leading Yes/No without treating the gold False class as error."""

    match = _BOOL_ANSWER_RE.search(str(answer))
    if not match:
        raise ValueError(f"Prediction is not a leading Yes/No answer: {answer!r}")
    return match.group(1).casefold() == "yes"


def _record_id(record: Mapping[str, Any], index: int) -> str:
    return str(record.get("id", index))


def load_boolq_predictions(
    dataset_path: str | Path,
    predictions_path: str | Path,
) -> tuple[list[TokenGraphExample], dict[str, int]]:
    """Join generated BoolQ answers and derive correctness only for evaluation."""

    dataset_rows = _read_records(dataset_path)
    predictions = {
        _record_id(row, index): row
        for index, row in enumerate(_read_records(predictions_path))
    }
    examples: list[TokenGraphExample] = []
    evaluation_labels: dict[str, int] = {}
    for index, row in enumerate(dataset_rows):
        example_id = _record_id(row, index)
        if example_id not in predictions:
            raise ValueError(f"Missing BoolQ prediction for {example_id!r}")
        model_answer = str(predictions[example_id]["model_answer"])
        predicted_bool = parse_bool_answer(model_answer)
        gold_bool = bool(row["answer"])
        examples.append(
            compose_example(
                str(row["passage"]),
                str(row["question"]),
                model_answer,
                example_id=example_id,
                pair_id=example_id,
                dataset="boolq",
                metadata={"title": str(row.get("title", ""))},
            )
        )
        evaluation_labels[example_id] = int(predicted_bool != gold_bool)
    return examples, evaluation_labels


_FORBIDDEN_EVALUATION_KEYS = {
    "candidate",
    "candidate_role",
    "candidate_type",
    "correct_candidate",
    "gold_answer",
    "is_correct",
    "is_hallucinated",
    "label",
    "labels",
    "target",
    "y",
    "y_token",
}


def _remove_evaluation_fields(value):
    if isinstance(value, Mapping):
        return {
            key: _remove_evaluation_fields(nested)
            for key, nested in value.items()
            if key.casefold() not in _FORBIDDEN_EVALUATION_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [_remove_evaluation_fields(nested) for nested in value]
    return value


def write_prepared_dataset(
    examples: list[TokenGraphExample],
    evaluation_labels: Mapping[str, int],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write label-free model inputs and a physically separate label sidecar."""

    output_directory = Path(output_dir)
    output_directory.mkdir(parents=True, exist_ok=True)
    example_ids = {example.example_id for example in examples}
    if set(evaluation_labels) != example_ids:
        raise ValueError("evaluation labels must match the prepared example ids")
    example_path = output_directory / "examples.jsonl"
    label_path = output_directory / "evaluation_labels.jsonl"
    example_lines = [
        json.dumps(
            _remove_evaluation_fields(asdict(example)),
            ensure_ascii=False,
            sort_keys=True,
        )
        for example in examples
    ]
    label_lines = [
        json.dumps(
            {"example_id": example.example_id, "label": int(evaluation_labels[example.example_id])},
            ensure_ascii=False,
            sort_keys=True,
        )
        for example in examples
    ]
    example_path.write_text("\n".join(example_lines) + "\n", encoding="utf-8")
    label_path.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
    return {"examples": example_path, "evaluation_labels": label_path}


def deterministic_split(
    examples: list[TokenGraphExample],
    *,
    train_fraction: float,
    seed: int,
) -> tuple[list[TokenGraphExample], list[TokenGraphExample]]:
    """Split by pair id so paired candidates can never cross partitions."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between zero and one")
    pair_ids = sorted({example.pair_id for example in examples})
    if len(pair_ids) < 2:
        raise ValueError("at least two pair groups are required")
    ordered_pairs = sorted(
        pair_ids,
        key=lambda pair_id: _stable_id(str(seed), pair_id),
    )
    train_count = min(
        len(pair_ids) - 1,
        max(1, round(len(pair_ids) * train_fraction)),
    )
    train_pairs = set(ordered_pairs[:train_count])
    train = [example for example in examples if example.pair_id in train_pairs]
    test = [example for example in examples if example.pair_id not in train_pairs]
    return train, test
