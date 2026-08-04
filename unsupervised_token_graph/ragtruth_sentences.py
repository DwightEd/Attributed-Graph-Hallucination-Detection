"""Label-free RAGTruth token-to-sentence score aggregation and evaluation."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from .ragtruth_data import atomic_json, atomic_jsonl, load_compact_manifest
from .ragtruth_graph import load_attention_sample

_SYSTEM_PROMPT = "You are a helpful assistant."
_CLOSERS = frozenset('"\'”’)]}')
_ABBREVIATIONS = frozenset(
    {"dr", "mr", "mrs", "ms", "prof", "sr", "jr", "st", "vs", "etc", "e.g", "i.e"}
)


def _is_sentence_period(text: str, index: int) -> bool:
    if text[index] != ".":
        return True
    if 0 < index < len(text) - 1 and text[index - 1].isdigit() and text[index + 1].isdigit():
        return False
    start = index - 1
    while start >= 0 and (text[start].isalpha() or text[start] == "."):
        start -= 1
    word = text[start + 1:index].casefold()
    if word in _ABBREVIATIONS or (len(word) == 1 and word.isalpha()):
        return False
    return True


def sentence_char_spans(text: str) -> list[tuple[int, int]]:
    """Return contiguous English sentence spans, retaining inter-sentence space."""

    if not isinstance(text, str) or not text.strip():
        raise ValueError("response text must be a non-empty string")
    spans: list[tuple[int, int]] = []
    start, index = 0, 0
    while index < len(text):
        character = text[index]
        end: int | None = None
        if character == "\n":
            end = index + 1
        elif character in ".!?" and _is_sentence_period(text, index):
            end = index + 1
            while end < len(text) and text[end] in _CLOSERS:
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
        raise ValueError("response text produced no sentence spans")
    return spans


def _identity(record: Mapping[str, object]) -> tuple[str, str]:
    return str(record["source_id"]), str(record["sample_id"])


def _assign_token_to_sentence(
    offset: tuple[int, int], spans: Sequence[tuple[int, int]], text_length: int,
) -> int:
    start, end = map(int, offset)
    if not 0 <= start < end <= text_length:
        raise ValueError(f"invalid response token offset: {(start, end)}")
    overlaps = [max(0, min(end, right) - max(start, left)) for left, right in spans]
    best = max(overlaps, default=0)
    if best <= 0:
        raise ValueError(f"response token offset is outside sentence coverage: {(start, end)}")
    return overlaps.index(best)


def aggregate_sentence_token_scores(
    score_records: Sequence[Mapping[str, object]],
    token_offsets: Mapping[tuple[str, str], Mapping[int, tuple[int, int]]],
    response_texts: Mapping[tuple[str, str], str],
    *,
    top_fraction: float = 0.20,
) -> list[dict[str, object]]:
    """Pool frozen token scores into label-free sentence anomaly scores."""

    if not 0.0 < top_fraction <= 1.0:
        raise ValueError("top_fraction must be in (0, 1]")
    grouped_scores: dict[tuple[str, str], dict[int, float]] = {}
    for record in score_records:
        identity = _identity(record)
        token_idx = int(record["token_idx"])
        score = float(record["score"])
        if not math.isfinite(score):
            raise ValueError("token scores must be finite")
        if token_idx in grouped_scores.setdefault(identity, {}):
            raise ValueError(f"duplicate token score: {identity + (token_idx,)}")
        grouped_scores[identity][token_idx] = score
    if not grouped_scores or set(grouped_scores) != set(token_offsets) or set(grouped_scores) != set(response_texts):
        raise ValueError("sample coverage differs across scores, offsets, and responses")

    output: list[dict[str, object]] = []
    for identity in sorted(grouped_scores):
        scores = grouped_scores[identity]
        offsets = {int(index): tuple(value) for index, value in token_offsets[identity].items()}
        if set(scores) != set(offsets):
            raise ValueError(f"response token coverage mismatch for {identity}")
        text = response_texts[identity]
        spans = sentence_char_spans(text)
        sentence_tokens: list[list[int]] = [[] for _ in spans]
        for token_idx in sorted(offsets):
            sentence_idx = _assign_token_to_sentence(offsets[token_idx], spans, len(text))
            sentence_tokens[sentence_idx].append(token_idx)
        for sentence_idx, ((char_start, char_end), token_indices) in enumerate(
            zip(spans, sentence_tokens)
        ):
            if not token_indices:
                raise ValueError(f"sentence has no aligned response tokens: {identity + (sentence_idx,)}")
            values = sorted((scores[index] for index in token_indices), reverse=True)
            top_count = max(1, math.ceil(len(values) * top_fraction))
            output.append(
                {
                    "source_id": identity[0],
                    "sample_id": identity[1],
                    "sentence_idx": sentence_idx,
                    "char_start": char_start,
                    "char_end": char_end,
                    "text": text[char_start:char_end].strip(),
                    "token_indices": token_indices,
                    "token_count": len(token_indices),
                    "pooling": f"top_{top_fraction:.6g}_mean",
                    "score": float(sum(values[:top_count]) / top_count),
                    "max_score": float(values[0]),
                    "mean_score": float(sum(values) / len(values)),
                }
            )
    return output


def evaluate_sentence_score_records(
    sentence_records: Sequence[Mapping[str, object]],
    token_labels: Mapping[tuple[str, str, int], int],
) -> dict[str, float | int]:
    """Join sentence membership to held-out token labels after score freezing."""

    labels, scores = [], []
    pooling_values = {
        str(record.get("pooling", "unspecified")) for record in sentence_records
    }
    if len(pooling_values) != 1:
        raise ValueError("sentence score file mixes pooling policies")
    seen: set[tuple[str, str, int]] = set()
    used_tokens: set[tuple[str, str, int]] = set()
    for record in sentence_records:
        identity = _identity(record)
        sentence_idx = int(record["sentence_idx"])
        sentence_key = identity + (sentence_idx,)
        if sentence_key in seen:
            raise ValueError(f"duplicate sentence score: {sentence_key}")
        seen.add(sentence_key)
        indices = [int(value) for value in record["token_indices"]]
        if not indices or len(indices) != len(set(indices)):
            raise ValueError(f"sentence token membership is empty or duplicated: {sentence_key}")
        keys = [(identity[0], identity[1], index) for index in indices]
        duplicated = used_tokens.intersection(keys)
        if duplicated:
            raise ValueError(f"tokens belong to multiple sentences: {sorted(duplicated)[:4]}")
        used_tokens.update(keys)
        missing = [key for key in keys if key not in token_labels]
        if missing:
            raise ValueError(f"sentence token labels are missing: {missing[:4]}")
        labels.append(max(int(token_labels[key]) for key in keys))
        scores.append(float(record["score"]))
    if used_tokens != set(token_labels):
        raise ValueError("sentence/token label coverage mismatch")
    label_array = np.asarray(labels, dtype=np.int8)
    score_array = np.asarray(scores, dtype=np.float64)
    if set(label_array.tolist()) != {0, 1}:
        raise ValueError("both clean and hallucinated held-out sentences are required")
    if not np.isfinite(score_array).all():
        raise ValueError("sentence scores must be finite")
    from sklearn.metrics import average_precision_score, roc_auc_score

    prevalence = float(label_array.mean())
    auprc = float(average_precision_score(label_array, score_array))
    return {
        "pooling": next(iter(pooling_values)),
        "sentence_count": len(labels),
        "positive_count": int(label_array.sum()),
        "prevalence": prevalence,
        "auroc": float(roc_auc_score(label_array, score_array)),
        "auprc": auprc,
        "auprc_lift": float(auprc / prevalence),
    }


def _read_unique_jsonl(
    path: str | Path, key: str, fields: Sequence[str],
) -> dict[str, dict[str, object]]:
    output: dict[str, dict[str, object]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = dict(json.loads(line))
        missing = [field for field in fields if field not in raw]
        if missing:
            raise ValueError(f"metadata row in {path} is missing fields: {missing}")
        record = {field: raw[field] for field in fields}
        identity = str(record[key])
        if identity in output:
            raise ValueError(f"duplicate {key} in {path}: {identity}")
        output[identity] = record
    if not output:
        raise ValueError(f"metadata file is empty: {path}")
    return output


def _token_ids_list(value: object) -> list[int]:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.flatten().tolist()]
    if value and isinstance(value, list) and isinstance(value[0], list):
        value = value[0]
    return [int(item) for item in value]


def reconstruct_response_offsets(
    tokenizer: object,
    *,
    prompt: str,
    response: str,
    expected_token_ids: Sequence[int],
    expected_response_idx: int,
) -> dict[int, tuple[int, int]]:
    """Recreate extraction offsets and fail closed on tokenizer/cache drift."""

    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    encoding = tokenizer(
        rendered_prompt + response,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = _token_ids_list(encoding["input_ids"])
    expected = [int(value) for value in expected_token_ids]
    if token_ids != expected:
        raise RuntimeError("tokenizer/cache token_ids mismatch; use the extraction model tokenizer")
    offsets = [tuple(map(int, value)) for value in encoding["offset_mapping"]]
    boundary = len(rendered_prompt)
    response_offsets: dict[int, tuple[int, int]] = {}
    for token_idx, (start, end) in enumerate(offsets):
        if start < boundary < end:
            raise RuntimeError("token crosses the cached prompt/response boundary")
        if end <= boundary:
            continue
        if start < boundary:
            raise RuntimeError("invalid cached prompt/response boundary")
        response_offsets[token_idx] = (start - boundary, end - boundary)
    if not response_offsets or min(response_offsets) != int(expected_response_idx):
        raise RuntimeError("tokenizer/cache response_idx mismatch")
    if sorted(response_offsets) != list(range(int(expected_response_idx), len(expected))):
        raise RuntimeError("response token offsets are not a contiguous suffix")
    return response_offsets


def write_sentence_score_file(
    score_path: str | Path,
    attention_dir: str | Path,
    graph_dir: str | Path,
    response_path: str | Path,
    source_path: str | Path,
    tokenizer_path: str | Path,
    output_path: str | Path,
    *,
    top_fraction: float = 0.20,
) -> dict[str, object]:
    """Reconstruct exact response offsets and persist label-free sentence scores."""

    scores = [
        dict(json.loads(line))
        for line in Path(score_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not scores:
        raise ValueError(f"score file is empty: {score_path}")
    identities = {_identity(record) for record in scores}
    manifest = load_compact_manifest(graph_dir)
    by_identity = {_identity(record): record for record in manifest}
    if len(by_identity) != len(manifest):
        raise RuntimeError("compact manifest contains duplicate sample identities")
    if not identities.issubset(by_identity):
        raise ValueError("compact manifest is missing scored sample identities")
    responses = _read_unique_jsonl(
        response_path, "id", ("id", "source_id", "response", "split")
    )
    sources = _read_unique_jsonl(source_path, "source_id", ("source_id", "prompt"))
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError("transformers is required for sentence offset reconstruction") from error
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), use_fast=True)
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError("tqdm is required for sentence score reconstruction") from error
    attention_root = Path(attention_dir).resolve()
    token_offsets: dict[tuple[str, str], dict[int, tuple[int, int]]] = {}
    response_texts: dict[tuple[str, str], str] = {}
    for identity in tqdm(
        sorted(identities), desc="align test tokens -> sentences", unit="response"
    ):
        record = by_identity[identity]
        attention_path = (attention_root / str(record["source_file"])).resolve()
        if attention_root not in attention_path.parents or not attention_path.is_file():
            raise RuntimeError(f"unsafe or missing source attention file: {attention_path}")
        sample = load_attention_sample(
            attention_path, device="cpu", mmap=True, include_labels=False
        )
        response_record = responses.get(identity[1])
        if response_record is None or str(response_record.get("source_id")) != identity[0]:
            raise ValueError(f"RAGTruth response metadata mismatch for {identity}")
        if str(response_record["split"]).casefold() != str(
            record.get("dataset_split", "")
        ).casefold():
            raise ValueError(f"RAGTruth official split metadata mismatch for {identity}")
        source_record = sources.get(identity[0])
        if source_record is None:
            raise ValueError(f"RAGTruth source metadata is absent for {identity[0]}")
        response_text = str(response_record["response"])
        token_offsets[identity] = reconstruct_response_offsets(
            tokenizer,
            prompt=str(source_record["prompt"]),
            response=response_text,
            expected_token_ids=torch.as_tensor(sample["token_ids"]).tolist(),
            expected_response_idx=int(sample["response_idx"]),
        )
        response_texts[identity] = response_text
        del sample
    sentence_records = aggregate_sentence_token_scores(
        scores, token_offsets, response_texts, top_fraction=top_fraction
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_jsonl(output, sentence_records)
    summary = {
        "schema_version": "ragtruth_sentence_scores_v1",
        "state": "complete",
        "label_free": True,
        "responses": len(identities),
        "sentences": len(sentence_records),
        "top_fraction": top_fraction,
        "scores": str(output),
    }
    atomic_json(output.with_suffix(".summary.json"), summary)
    return summary
