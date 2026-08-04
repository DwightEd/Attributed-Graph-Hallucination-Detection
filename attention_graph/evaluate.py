"""Held-out evaluation and generic token-to-sentence aggregation.

Nothing in the training or graph-preparation path imports labels.  The sole
label-reading entry point is :func:`load_evaluation_labels`, which explicitly
requests ``y_token`` from the formal cache after predictions have been frozen.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .data import load_attention_record


_SENTENCE_CLOSERS = frozenset('"\'”’)]}')
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}
)


@dataclass(frozen=True)
class EvaluationLabels:
    """Held-out labels indexed independently of graph artifacts."""

    response_labels: Mapping[tuple[str, str], int]
    token_labels: Mapping[tuple[str, str, int], int]


def _binary_array(values: Sequence[object], *, name: str) -> np.ndarray:
    if len(values) == 0:
        raise ValueError(f"{name} is empty")
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite binary values") from error
    if array.ndim != 1 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a one-dimensional binary sequence")
    try:
        finite = np.isfinite(array.astype(np.float64))
        binary = (array == 0) | (array == 1)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite binary values") from error
    if not bool(np.all(finite & binary)):
        raise ValueError(f"{name} must contain only finite values in {{0, 1}}")
    return array.astype(np.int8)


def _score_array(values: Sequence[object], *, name: str) -> np.ndarray:
    if len(values) == 0:
        raise ValueError(f"{name} is empty")
    try:
        scores = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain finite scalar scores") from error
    if scores.ndim != 1 or not bool(np.isfinite(scores).all()):
        raise ValueError(f"{name} must contain finite scalar scores")
    return scores


def evaluate_binary_scores(
    labels: Sequence[object],
    scores: Sequence[object],
) -> dict[str, float | int]:
    """Compute rank metrics, prevalence baseline, and orientation-free forms.

    Orientation-free metrics evaluate both ``score`` and ``-score``.  They are
    diagnostic separability measures only; they do not use class prevalence to
    choose an orientation and therefore do not assume hallucinations are rare.
    """

    label_array = _binary_array(labels, name="labels")
    score_array = _score_array(scores, name="scores")
    if len(label_array) != len(score_array):
        raise ValueError("labels and scores must have equal length")
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("both classes are required for AUROC/AP evaluation")
    from sklearn.metrics import average_precision_score, roc_auc_score

    prevalence = float(label_array.mean())
    auroc = float(roc_auc_score(label_array, score_array))
    reverse_auroc = float(roc_auc_score(label_array, -score_array))
    average_precision = float(average_precision_score(label_array, score_array))
    reverse_average_precision = float(
        average_precision_score(label_array, -score_array)
    )
    orientation_free_ap = max(average_precision, reverse_average_precision)
    return {
        "samples": int(len(label_array)),
        "positive_count": int(label_array.sum()),
        "positive_fraction": prevalence,
        "average_precision_random_baseline": prevalence,
        "auroc": auroc,
        "average_precision": average_precision,
        "average_precision_lift": average_precision - prevalence,
        "orientation_free_auroc": max(auroc, reverse_auroc),
        "orientation_free_average_precision": orientation_free_ap,
        "orientation_free_average_precision_lift": orientation_free_ap - prevalence,
    }


def _validate_cached_labels(value: object, *, context: str) -> torch.Tensor:
    labels = torch.as_tensor(value).detach().cpu().flatten()
    if not labels.numel():
        raise ValueError(f"{context} is empty")
    if labels.is_complex() or (
        labels.is_floating_point() and not bool(torch.isfinite(labels).all())
    ):
        raise ValueError(f"{context} must contain finite binary labels")
    if bool((~((labels == 0) | (labels == 1))).any()):
        raise ValueError(f"{context} must contain only labels in {{0, 1}}")
    return labels.long()


def load_evaluation_labels(
    attention_paths: Sequence[str | Path],
    *,
    mmap: bool = True,
) -> EvaluationLabels:
    """Read ``y_token`` only for held-out evaluation, never graph training."""

    if not attention_paths:
        raise ValueError("evaluation attention paths are empty")
    response_labels: dict[tuple[str, str], int] = {}
    token_labels: dict[tuple[str, str, int], int] = {}
    for raw_path in attention_paths:
        path = Path(raw_path).expanduser().resolve()
        sample = load_attention_record(
            path,
            device="cpu",
            mmap=mmap,
            include_labels=True,
        )
        if "y_token" not in sample:
            raise ValueError(f"held-out y_token labels are absent: {path}")
        identity = (str(sample["source_id"]), str(sample["sample_id"]))
        if identity in response_labels:
            raise ValueError(f"duplicate evaluation response identity: {identity}")
        labels = _validate_cached_labels(sample["y_token"], context=f"y_token in {path}")
        response_idx = int(sample["response_idx"])
        token_count = int(torch.as_tensor(sample["token_ids"]).numel())
        if labels.numel() != token_count:
            raise ValueError(f"y_token length does not match token_ids: {path}")
        if bool(labels[:response_idx].any()):
            raise ValueError(f"y_token marks prompt tokens as hallucinated: {path}")
        response = labels[response_idx:]
        if not response.numel():
            raise ValueError(f"evaluation response has no tokens: {path}")
        response_labels[identity] = int(response.max().item())
        for token_idx in range(response_idx, token_count):
            token_labels[identity + (token_idx,)] = int(labels[token_idx].item())
    return EvaluationLabels(
        response_labels=response_labels,
        token_labels=token_labels,
    )


def _record_identity(record: Mapping[str, object]) -> tuple[str, str]:
    source_id = str(record.get("source_id", "")).strip()
    sample_id = str(record.get("sample_id", record.get("response_id", ""))).strip()
    response_id = str(record.get("response_id", sample_id)).strip()
    if not source_id or not sample_id:
        raise ValueError("prediction records require source_id and sample_id/response_id")
    if response_id and response_id != sample_id:
        raise ValueError("prediction sample_id and response_id disagree")
    return source_id, sample_id


def _probability(value: object, *, context: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite probability") from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{context} must be a finite probability in [0, 1]")
    return probability


def evaluate_predictions(
    response_records: Sequence[Mapping[str, object]],
    token_records: Sequence[Mapping[str, object]],
    labels: EvaluationLabels,
    *,
    response_score_field: str = "hallucination_probability",
    token_score_field: str = "score",
) -> dict[str, object]:
    """Join frozen response/token probabilities to exact held-out identities."""

    if not response_records:
        raise ValueError("response prediction records are empty")
    if not token_records:
        raise ValueError("token prediction records are empty")
    response_scores: dict[tuple[str, str], float] = {}
    for record in response_records:
        identity = _record_identity(record)
        if identity in response_scores:
            raise ValueError(f"duplicate response prediction identity: {identity}")
        if response_score_field not in record:
            raise ValueError(f"response record is missing {response_score_field}")
        response_scores[identity] = _probability(
            record[response_score_field], context=f"response score {identity}"
        )
    token_scores: dict[tuple[str, str, int], float] = {}
    for record in token_records:
        identity = _record_identity(record)
        if "token_idx" not in record or token_score_field not in record:
            raise ValueError("token records require token_idx and a score")
        key = identity + (int(record["token_idx"]),)
        if key in token_scores:
            raise ValueError(f"duplicate token prediction identity: {key}")
        token_scores[key] = _probability(
            record[token_score_field], context=f"token score {key}"
        )
    if set(response_scores) != set(labels.response_labels):
        missing = set(labels.response_labels).difference(response_scores)
        extra = set(response_scores).difference(labels.response_labels)
        raise ValueError(
            "response prediction/label alignment failed: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if set(token_scores) != set(labels.token_labels):
        missing = set(labels.token_labels).difference(token_scores)
        extra = set(token_scores).difference(labels.token_labels)
        raise ValueError(
            "token prediction/label alignment failed: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    response_keys = sorted(response_scores)
    token_keys = sorted(token_scores)
    return {
        "artifact_type": "attention_graph_posthoc_evaluation_v1",
        "labels_read_during": "evaluation_only",
        "response": evaluate_binary_scores(
            [labels.response_labels[key] for key in response_keys],
            [response_scores[key] for key in response_keys],
        ),
        "token": evaluate_binary_scores(
            [labels.token_labels[key] for key in token_keys],
            [token_scores[key] for key in token_keys],
        ),
    }


def evaluate_predictions_from_attention(
    response_records: Sequence[Mapping[str, object]],
    token_records: Sequence[Mapping[str, object]],
    attention_paths: Sequence[str | Path],
    *,
    mmap: bool = True,
    response_score_field: str = "hallucination_probability",
    token_score_field: str = "score",
) -> dict[str, object]:
    """Explicit post-hoc stage that loads labels and evaluates frozen scores."""

    labels = load_evaluation_labels(attention_paths, mmap=mmap)
    return evaluate_predictions(
        response_records,
        token_records,
        labels,
        response_score_field=response_score_field,
        token_score_field=token_score_field,
    )


def evaluate_sentence_predictions(
    sentence_records: Sequence[Mapping[str, object]],
    labels: EvaluationLabels,
    *,
    score_field: str = "score",
) -> dict[str, object]:
    """Evaluate frozen sentence scores using any-positive token membership."""

    if not sentence_records:
        raise ValueError("sentence prediction records are empty")
    pooling = {str(record.get("pooling", "")) for record in sentence_records}
    if pooling != {"mean"}:
        raise ValueError("sentence predictions must use one declared mean pooling policy")
    used_tokens: set[tuple[str, str, int]] = set()
    seen_sentences: set[tuple[str, str, int]] = set()
    sentence_labels: list[int] = []
    sentence_scores: list[float] = []
    for record in sentence_records:
        identity = _record_identity(record)
        if "sentence_idx" not in record or "token_indices" not in record:
            raise ValueError("sentence records require sentence_idx and token_indices")
        sentence_key = identity + (int(record["sentence_idx"]),)
        if sentence_key in seen_sentences:
            raise ValueError(f"duplicate sentence prediction identity: {sentence_key}")
        seen_sentences.add(sentence_key)
        token_indices = [int(value) for value in record["token_indices"]]
        if not token_indices or len(token_indices) != len(set(token_indices)):
            raise ValueError("sentence token membership must be non-empty and unique")
        token_keys = [identity + (token_idx,) for token_idx in token_indices]
        if any(key not in labels.token_labels for key in token_keys):
            raise ValueError(f"sentence/token label alignment failed: {sentence_key}")
        duplicate_tokens = used_tokens.intersection(token_keys)
        if duplicate_tokens:
            raise ValueError("sentence token coverage overlaps across sentences")
        used_tokens.update(token_keys)
        if score_field not in record:
            raise ValueError(f"sentence record is missing {score_field}")
        sentence_scores.append(
            _probability(record[score_field], context=f"sentence score {sentence_key}")
        )
        sentence_labels.append(
            max(int(labels.token_labels[key]) for key in token_keys)
        )
    if used_tokens != set(labels.token_labels):
        missing = set(labels.token_labels).difference(used_tokens)
        extra = used_tokens.difference(labels.token_labels)
        raise ValueError(
            "sentence token coverage differs from held-out labels: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    return {
        "pooling": "mean",
        "label_rule": "sentence_positive_if_any_member_token_is_positive",
        "metrics": evaluate_binary_scores(sentence_labels, sentence_scores),
    }


def _is_sentence_period(text: str, index: int) -> bool:
    if text[index] != ".":
        return True
    if (
        0 < index < len(text) - 1
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    ):
        return False
    start = index - 1
    while start >= 0 and (text[start].isalpha() or text[start] == "."):
        start -= 1
    word = text[start + 1 : index].casefold()
    return not (word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()))


def sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Split non-empty text into contiguous sentence character spans."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("text must be a non-empty string")
    spans: list[tuple[int, int]] = []
    start = 0
    index = 0
    while index < len(text):
        character = text[index]
        end: int | None = None
        if character == "\n":
            end = index + 1
        elif character in ".!?。！？" and _is_sentence_period(text, index):
            end = index + 1
            while end < len(text) and text[end] in _SENTENCE_CLOSERS:
                end += 1
            if end < len(text) and not text[end].isspace():
                end = None
        if end is not None and text[start:end].strip():
            spans.append((start, end))
            start = end
            index = end
            continue
        index += 1
    if text[start:].strip():
        spans.append((start, len(text)))
    if not spans:
        raise ValueError("text produced no sentence spans")
    return spans


def _validated_offsets(
    token_offsets: Mapping[int, tuple[int, int]],
    *,
    text_length: int,
) -> list[tuple[int, int, int]]:
    if not token_offsets:
        raise ValueError("token offsets are empty")
    output: list[tuple[int, int, int]] = []
    seen_indices: set[int] = set()
    for raw_index, raw_offset in token_offsets.items():
        token_idx = int(raw_index)
        if token_idx in seen_indices:
            raise ValueError(f"duplicate token index after normalization: {token_idx}")
        seen_indices.add(token_idx)
        if len(raw_offset) != 2:
            raise ValueError(f"token offset must be a pair: {raw_offset}")
        left, right = map(int, raw_offset)
        if not 0 <= left < right <= text_length:
            raise ValueError(f"token offset is outside text: {(left, right)}")
        output.append((token_idx, left, right))
    output.sort(key=lambda item: (item[1], item[2], item[0]))
    for previous, current in zip(output, output[1:]):
        if current[1] < previous[2]:
            raise ValueError("token character offsets overlap")
    return output


def map_token_offsets_to_sentences(
    text: str,
    token_offsets: Mapping[int, tuple[int, int]],
) -> list[list[int]]:
    """Assign each token to the sentence with greatest character overlap."""

    spans = sentence_char_spans(text)
    sentence_tokens: list[list[int]] = [[] for _ in spans]
    for token_idx, left, right in _validated_offsets(
        token_offsets, text_length=len(text)
    ):
        overlaps = [
            max(0, min(right, sentence_right) - max(left, sentence_left))
            for sentence_left, sentence_right in spans
        ]
        best = max(overlaps, default=0)
        if best <= 0:
            raise ValueError(f"token offset does not overlap a sentence: {(left, right)}")
        sentence_tokens[overlaps.index(best)].append(token_idx)
    if any(not indices for indices in sentence_tokens):
        raise ValueError("at least one sentence has no aligned tokens")
    return sentence_tokens


def aggregate_sentence_probabilities(
    text: str,
    token_probabilities: Mapping[int, float],
    token_offsets: Mapping[int, tuple[int, int]],
) -> list[dict[str, object]]:
    """Aggregate token probabilities by arithmetic mean for each sentence."""

    probability_keys = {int(key) for key in token_probabilities}
    offset_keys = {int(key) for key in token_offsets}
    if probability_keys != offset_keys:
        raise ValueError("token probability/offset coverage differs")
    probabilities = {
        int(key): _probability(value, context=f"token probability {key}")
        for key, value in token_probabilities.items()
    }
    spans = sentence_char_spans(text)
    membership = map_token_offsets_to_sentences(text, token_offsets)
    output: list[dict[str, object]] = []
    for sentence_idx, ((left, right), token_indices) in enumerate(
        zip(spans, membership)
    ):
        values = [probabilities[index] for index in token_indices]
        output.append(
            {
                "sentence_idx": sentence_idx,
                "char_start": left,
                "char_end": right,
                "text": text[left:right].strip(),
                "token_indices": token_indices,
                "token_count": len(token_indices),
                "pooling": "mean",
                "score": float(sum(values) / len(values)),
            }
        )
    return output


__all__ = [
    "EvaluationLabels",
    "aggregate_sentence_probabilities",
    "evaluate_binary_scores",
    "evaluate_predictions",
    "evaluate_predictions_from_attention",
    "evaluate_sentence_predictions",
    "load_evaluation_labels",
    "map_token_offsets_to_sentences",
    "sentence_char_spans",
]
