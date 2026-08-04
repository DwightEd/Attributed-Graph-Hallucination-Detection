"""RAGTruth metadata and tokenizer adapter for label-free sentence scores."""

from __future__ import annotations

import json
import math
import os
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch

from .data import load_attention_record
from .evaluate import aggregate_sentence_probabilities


EXTRACTION_SYSTEM_PROMPT = "You are a helpful assistant."


def _read_jsonl_index(
    path: str | Path,
    *,
    identity_field: str,
    required_fields: Sequence[str],
) -> dict[str, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"RAGTruth metadata file is absent: {source}")
    output: dict[str, dict[str, object]] = {}
    for line_number, line in enumerate(
        source.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(
                f"invalid JSON in {source} at line {line_number}"
            ) from error
        if not isinstance(raw, Mapping):
            raise TypeError(f"RAGTruth metadata row must be an object: {source}:{line_number}")
        missing = [field for field in required_fields if field not in raw]
        if missing:
            raise ValueError(
                f"RAGTruth metadata row is missing {missing}: {source}:{line_number}"
            )
        identity = str(raw[identity_field]).strip()
        if not identity:
            raise ValueError(
                f"RAGTruth {identity_field} must be non-empty: {source}:{line_number}"
            )
        if identity in output:
            raise ValueError(f"duplicate RAGTruth {identity_field}: {identity}")
        output[identity] = {field: raw[field] for field in required_fields}
    if not output:
        raise ValueError(f"RAGTruth metadata file is empty: {source}")
    return output


def load_ragtruth_metadata(
    response_path: str | Path,
    source_path: str | Path,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Load strict response/source indexes from the official JSONL files."""

    responses = _read_jsonl_index(
        response_path,
        identity_field="id",
        required_fields=("id", "source_id", "response", "split"),
    )
    sources = _read_jsonl_index(
        source_path,
        identity_field="source_id",
        required_fields=("source_id", "prompt"),
    )
    for response_id, record in responses.items():
        source_id = str(record["source_id"]).strip()
        if not source_id or source_id not in sources:
            raise ValueError(
                f"RAGTruth response {response_id} references missing source {source_id!r}"
            )
        if not isinstance(record["response"], str) or not str(record["response"]).strip():
            raise ValueError(f"RAGTruth response text is empty: {response_id}")
        split = str(record["split"]).strip().casefold()
        if split not in {"train", "test"}:
            raise ValueError(f"RAGTruth response has unsupported split: {response_id}")
        record["source_id"] = source_id
        record["split"] = split
    for source_id, record in sources.items():
        if not isinstance(record["prompt"], str) or not str(record["prompt"]).strip():
            raise ValueError(f"RAGTruth prompt is empty: {source_id}")
    return responses, sources


def _flat_token_ids(value: object) -> list[int]:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise ValueError("tokenizer input_ids must describe one sequence")
    return [int(item) for item in tensor.tolist()]


def _offset_pairs(value: object) -> list[tuple[int, int]]:
    tensor = torch.as_tensor(value)
    if tensor.ndim == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 2 or tensor.shape[1] != 2:
        raise ValueError("tokenizer offset_mapping must have shape [tokens, 2]")
    return [tuple(map(int, pair)) for pair in tensor.tolist()]


def reconstruct_response_offsets(
    tokenizer: object,
    *,
    prompt: str,
    response: str,
    expected_token_ids: Sequence[int] | torch.Tensor,
    expected_response_idx: int,
    system_prompt: str = EXTRACTION_SYSTEM_PROMPT,
) -> dict[int, tuple[int, int]]:
    """Replay extraction tokenization and fail closed on any cache drift."""

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not isinstance(response, str) or not response.strip():
        raise ValueError("response must be a non-empty string")
    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError("system_prompt must be a non-empty string")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise RuntimeError("tokenizer chat template did not return rendered text")
    combined = rendered_prompt + response
    encoding = tokenizer(
        combined,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoding, Mapping):
        raise TypeError("tokenizer output must be a mapping")
    if "input_ids" not in encoding or "offset_mapping" not in encoding:
        raise ValueError("tokenizer output lacks input_ids or offset_mapping")
    observed_ids = _flat_token_ids(encoding["input_ids"])
    expected_ids = _flat_token_ids(expected_token_ids)
    if observed_ids != expected_ids:
        raise RuntimeError(
            "tokenizer/cache token_ids mismatch; use the exact extraction-model tokenizer"
        )
    offsets = _offset_pairs(encoding["offset_mapping"])
    if len(offsets) != len(observed_ids):
        raise RuntimeError("tokenizer ids and offsets have different lengths")
    response_idx = int(expected_response_idx)
    if not 0 < response_idx < len(expected_ids):
        raise ValueError("expected_response_idx must split prompt and response tokens")
    boundary = len(rendered_prompt)
    response_offsets: dict[int, tuple[int, int]] = {}
    for token_idx, (left, right) in enumerate(offsets):
        if not 0 <= left <= right <= len(combined):
            raise RuntimeError("tokenizer produced an invalid character offset")
        if left < boundary < right:
            raise RuntimeError("a token crosses the cached prompt/response boundary")
        if right <= boundary:
            continue
        if left < boundary or left == right:
            raise RuntimeError("tokenizer produced an invalid response-token offset")
        response_offsets[token_idx] = (left - boundary, right - boundary)
    expected_indices = list(range(response_idx, len(expected_ids)))
    if sorted(response_offsets) != expected_indices:
        raise RuntimeError("tokenizer/cache response_idx mismatch")
    if not response_offsets:
        raise RuntimeError("tokenizer produced no response token offsets")
    if max(right for _, right in response_offsets.values()) > len(response):
        raise RuntimeError("response offsets exceed response text")
    return response_offsets


def _prediction_identity(record: Mapping[str, object]) -> tuple[str, str]:
    source_id = str(record.get("source_id", "")).strip()
    sample_id = str(record.get("sample_id", record.get("response_id", ""))).strip()
    response_id = str(record.get("response_id", sample_id)).strip()
    if not source_id or not sample_id:
        raise ValueError("token score requires source_id and sample_id/response_id")
    if response_id and response_id != sample_id:
        raise ValueError("token score sample_id and response_id disagree")
    return source_id, sample_id


def _probability(value: object, *, context: str) -> float:
    try:
        probability = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite probability") from error
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise ValueError(f"{context} must be a probability in [0, 1]")
    return probability


def build_ragtruth_sentence_records(
    token_records: Sequence[Mapping[str, object]],
    attention_paths: Sequence[str | Path],
    *,
    response_path: str | Path,
    source_path: str | Path,
    tokenizer: object,
    token_score_field: str = "score",
    mmap: bool = True,
) -> list[dict[str, object]]:
    """Align frozen token probabilities and return label-free sentence rows."""

    if not token_records:
        raise ValueError("token score records are empty")
    if not attention_paths:
        raise ValueError("attention paths are empty")
    responses, sources = load_ragtruth_metadata(response_path, source_path)
    scores: dict[tuple[str, str], dict[int, float]] = {}
    for record in token_records:
        identity = _prediction_identity(record)
        if "token_idx" not in record or token_score_field not in record:
            raise ValueError("token score record lacks token_idx or score")
        token_idx = int(record["token_idx"])
        sample_scores = scores.setdefault(identity, {})
        if token_idx in sample_scores:
            raise ValueError(f"duplicate token score identity: {identity + (token_idx,)}")
        sample_scores[token_idx] = _probability(
            record[token_score_field], context=f"token score {identity + (token_idx,)}"
        )

    samples: dict[tuple[str, str], Mapping[str, object]] = {}
    for raw_path in attention_paths:
        sample = load_attention_record(
            Path(raw_path).expanduser().resolve(),
            device="cpu",
            mmap=mmap,
            include_labels=False,
        )
        identity = (str(sample["source_id"]), str(sample["sample_id"]))
        if identity in samples:
            raise ValueError(f"duplicate attention sample identity: {identity}")
        if str(sample.get("response_id", identity[1])) != identity[1]:
            raise ValueError(f"attention response_id/sample_id mismatch: {identity}")
        samples[identity] = sample
    if set(samples) != set(scores):
        missing = set(samples).difference(scores)
        extra = set(scores).difference(samples)
        raise ValueError(
            "attention/token-score sample alignment failed: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    output: list[dict[str, object]] = []
    for identity in sorted(samples):
        source_id, response_id = identity
        sample = samples[identity]
        response_record = responses.get(response_id)
        if response_record is None:
            raise ValueError(f"RAGTruth response metadata is absent: {response_id}")
        if str(response_record["source_id"]) != source_id:
            raise ValueError(f"RAGTruth response/source mismatch: {identity}")
        source_record = sources.get(source_id)
        if source_record is None:
            raise ValueError(f"RAGTruth source metadata is absent: {source_id}")
        dataset_split = str(sample.get("dataset_split", "")).strip().casefold()
        if dataset_split != str(response_record["split"]):
            raise ValueError(f"RAGTruth official split mismatch: {identity}")
        response_text = str(response_record["response"])
        offsets = reconstruct_response_offsets(
            tokenizer,
            prompt=str(source_record["prompt"]),
            response=response_text,
            expected_token_ids=torch.as_tensor(sample["token_ids"]),
            expected_response_idx=int(sample["response_idx"]),
        )
        if set(offsets) != set(scores[identity]):
            raise ValueError(f"token score/offset coverage mismatch: {identity}")
        sentence_rows = aggregate_sentence_probabilities(
            response_text,
            scores[identity],
            offsets,
        )
        for row in sentence_rows:
            output.append(
                {
                    "source_id": source_id,
                    "sample_id": response_id,
                    "response_id": response_id,
                    "dataset_split": dataset_split,
                    **row,
                }
            )
    if not output:
        raise RuntimeError("sentence alignment produced no records")
    return output


def write_ragtruth_sentence_records(
    records: Sequence[Mapping[str, object]],
    output_path: str | Path,
) -> Path:
    """Atomically persist label-free sentence records as JSONL."""

    if not records:
        raise ValueError("sentence records are empty")
    serializable: list[dict[str, object]] = []
    seen: set[tuple[str, str, int]] = set()
    for raw in records:
        record = dict(raw)
        forbidden = [
            key
            for key in record
            if "label" in str(key).casefold() or "target" in str(key).casefold()
        ]
        if forbidden:
            raise ValueError("sentence records must remain label-free")
        key = (
            str(record["source_id"]),
            str(record["sample_id"]),
            int(record["sentence_idx"]),
        )
        if key in seen:
            raise ValueError(f"duplicate sentence record: {key}")
        seen.add(key)
        _probability(record["score"], context=f"sentence score {key}")
        serializable.append(record)
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            "".join(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                for record in serializable
            ),
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def prepare_ragtruth_sentence_scores(
    token_records: Sequence[Mapping[str, object]],
    attention_paths: Sequence[str | Path],
    *,
    response_path: str | Path,
    source_path: str | Path,
    tokenizer: object,
    output_path: str | Path,
    token_score_field: str = "score",
    mmap: bool = True,
) -> list[dict[str, object]]:
    """CLI-friendly build-and-write entry point."""

    records = build_ragtruth_sentence_records(
        token_records,
        attention_paths,
        response_path=response_path,
        source_path=source_path,
        tokenizer=tokenizer,
        token_score_field=token_score_field,
        mmap=mmap,
    )
    write_ragtruth_sentence_records(records, output_path)
    return records


__all__ = [
    "EXTRACTION_SYSTEM_PROMPT",
    "build_ragtruth_sentence_records",
    "load_ragtruth_metadata",
    "prepare_ragtruth_sentence_scores",
    "reconstruct_response_offsets",
    "write_ragtruth_sentence_records",
]
