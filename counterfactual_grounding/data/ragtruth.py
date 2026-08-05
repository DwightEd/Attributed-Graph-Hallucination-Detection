"""Deterministic, label-blind RAGTruth evidence/query/response layouts.

The adapter deliberately derives evidence spans from RAGTruth's structured
``source_info`` field.  It never consults response-level or token-level
hallucination annotations.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

from .graph import Segment

SYSTEM_PROMPT = "You are a helpful assistant."


@dataclass(frozen=True)
class EvidenceChunk:
    """One independently addressable evidence span in the rendered prompt."""

    chunk_id: int
    text: str
    char_start: int
    char_end: int
    token_positions: torch.Tensor


@dataclass(frozen=True)
class RagTruthTokenLayout:
    """Exact token replay together with a complete E/Q/R token partition."""

    rendered_text: str
    input_ids: torch.Tensor
    segment_ids: torch.Tensor
    response_idx: int
    evidence_token_positions: torch.Tensor
    evidence_chunks: tuple[EvidenceChunk, ...]


def _reject_labels(record: Mapping[str, object]) -> None:
    forbidden = [str(key) for key in record if "label" in str(key).casefold()]
    if forbidden:
        raise ValueError(
            "RAGTruth layout construction is label-blind; forbidden fields: "
            + ", ".join(sorted(forbidden))
        )


def _require_unique(text: str, needle: str, *, description: str) -> tuple[int, int]:
    if not needle:
        raise ValueError(f"{description} must not be empty")
    start = text.find(needle)
    if start < 0:
        raise ValueError(f"{description} is absent from the RAGTruth prompt")
    if text.find(needle, start + 1) >= 0:
        raise ValueError(f"{description} is not unique in the RAGTruth prompt")
    return start, start + len(needle)


def _qa_passage_texts(source_info: Mapping[str, object]) -> list[str]:
    question = source_info.get("question")
    passages = source_info.get("passages")
    if not isinstance(question, str) or not question:
        raise ValueError("QA source_info requires a non-empty question")
    if isinstance(passages, Sequence) and not isinstance(passages, str):
        chunks = [str(item).strip() for item in passages if str(item).strip()]
    elif isinstance(passages, str):
        # Official RAGTruth joins numbered passages with a blank line.  Split
        # only at a blank line followed by the next numbered passage so a
        # passage may itself contain ordinary newlines.
        chunks = [
            item.strip()
            for item in re.split(
                r"\n\s*\n(?=\s*passage\s+\d+\s*:)",
                passages.strip(),
                flags=re.IGNORECASE,
            )
            if item.strip()
        ]
    else:
        raise TypeError("QA source_info requires passages")
    if not chunks:
        raise ValueError("QA source_info contains no evidence passages")
    return chunks


def _data2txt_span(prompt: str, source_info: object) -> tuple[int, int]:
    marker = "Structured data:"
    marker_start, marker_end = _require_unique(
        prompt, marker, description="Data2txt Structured data marker"
    )
    del marker_start
    endings: list[tuple[int, str]] = []
    for ending in ("Overview:", "output:"):
        position = prompt.find(ending, marker_end)
        if position >= 0:
            endings.append((position, ending))
    if not endings:
        raise ValueError("Data2txt prompt has no Overview:/output: terminator")
    end, _ = min(endings)
    start = marker_end
    while start < end and prompt[start].isspace():
        start += 1
    while end > start and prompt[end - 1].isspace():
        end -= 1
    literal = prompt[start:end]
    try:
        parsed = ast.literal_eval(literal)
    except (SyntaxError, ValueError) as error:
        raise ValueError("Data2txt structured evidence is not a Python literal") from error
    if parsed != source_info:
        raise ValueError(
            "Data2txt structured evidence is not equivalent to source_info"
        )
    return start, end


def _evidence_spans(
    record: Mapping[str, object], prompt: str
) -> list[tuple[int, int]]:
    task_type = str(record.get("task_type", "")).casefold()
    source_info = record.get("source_info")
    if task_type == "summary":
        if not isinstance(source_info, str):
            raise ValueError("Summary source_info must be text")
        return [
            _require_unique(prompt, source_info, description="Summary source_info")
        ]
    if task_type == "qa":
        if not isinstance(source_info, Mapping):
            raise ValueError("QA source_info must be a mapping")
        question = str(source_info.get("question", ""))
        passage_spans = [
            _require_unique(prompt, passage, description="QA passage")
            for passage in _qa_passage_texts(source_info)
        ]
        # A valid question can be quoted again inside a passage.  Only the
        # question-template slot before the first passage must be unique.
        first_passage_start = min(start for start, _ in passage_spans)
        _require_unique(
            prompt[:first_passage_start], question, description="QA question slot"
        )
        return passage_spans
    if task_type == "data2txt":
        return [_data2txt_span(prompt, source_info)]
    raise ValueError(f"unsupported RAGTruth task_type: {record.get('task_type')!r}")


def _one_dimensional_list(value: object, *, name: str) -> list[Any]:
    if hasattr(value, "tolist"):
        value = value.tolist()  # type: ignore[union-attr]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"tokenizer {name} must be a sequence")
    values = list(value)
    if len(values) == 1 and isinstance(values[0], Sequence):
        first = list(values[0])
        is_batch = name == "input_ids" or (
            name == "offset_mapping"
            and bool(first)
            and isinstance(first[0], Sequence)
        )
        if is_batch:
            values = first
    return values


def _render_prompt(tokenizer: object, prompt: str) -> str:
    apply_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_template):
        return prompt
    rendered = apply_template(
        [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered, str):
        raise TypeError("tokenizer chat template did not return rendered text")
    return rendered


def build_ragtruth_layout(
    source_record: Mapping[str, object],
    response_text: str,
    tokenizer: object,
) -> RagTruthTokenLayout:
    """Replay one sample and assign every token to E, Q, or R.

    For a real Hugging Face tokenizer, the original Llama chat template is
    rendered first.  Minimal test tokenizers without ``apply_chat_template``
    operate on the raw prompt.  In either case no truncation is permitted.
    """

    _reject_labels(source_record)
    prompt_value = source_record.get("prompt")
    if not isinstance(prompt_value, str) or not prompt_value:
        raise ValueError("RAGTruth source record requires a non-empty prompt")
    if not isinstance(response_text, str) or not response_text:
        raise ValueError("RAGTruth response must be non-empty text")

    prompt_spans = _evidence_spans(source_record, prompt_value)
    rendered_prompt = _render_prompt(tokenizer, prompt_value)
    prompt_start, _ = _require_unique(
        rendered_prompt, prompt_value, description="raw RAGTruth prompt"
    )
    rendered_spans = [
        (prompt_start + start, prompt_start + end) for start, end in prompt_spans
    ]
    rendered_text = rendered_prompt + response_text
    encoding = tokenizer(
        rendered_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoding, Mapping) or "offset_mapping" not in encoding:
        raise ValueError("an offset-aware tokenizer is required")
    raw_ids = _one_dimensional_list(encoding.get("input_ids"), name="input_ids")
    raw_offsets = _one_dimensional_list(
        encoding["offset_mapping"], name="offset_mapping"
    )
    input_ids = torch.tensor([int(item) for item in raw_ids], dtype=torch.long)
    offsets = [tuple(int(part) for part in pair) for pair in raw_offsets]
    if any(len(pair) != 2 for pair in offsets):
        raise ValueError("each tokenizer offset must contain start and end")
    if len(offsets) != input_ids.numel():
        raise ValueError("token ids and offsets must align one-to-one")

    prompt_boundary = len(rendered_prompt)
    response_positions: list[int] = []
    for position, (start, end) in enumerate(offsets):
        if start < prompt_boundary < end:
            raise ValueError(
                "a token crosses the prompt/response boundary; exact replay failed"
            )
        if end > prompt_boundary:
            if start < prompt_boundary:
                raise ValueError("invalid response token offset")
            response_positions.append(position)
    if not response_positions:
        raise ValueError("the response produced no aligned tokens")
    response_idx = response_positions[0]
    if response_positions != list(range(response_idx, len(offsets))):
        raise ValueError("response tokens must form a contiguous suffix")
    if not 0 < response_idx < input_ids.numel():
        raise ValueError("response_idx must split a non-empty prompt and response")

    segment_ids = torch.full(
        (input_ids.numel(),), int(Segment.QUERY), dtype=torch.int8
    )
    segment_ids[response_idx:] = int(Segment.RESPONSE)
    chunk_positions: list[list[int]] = [[] for _ in rendered_spans]
    for position, (token_start, token_end) in enumerate(offsets[:response_idx]):
        if token_end <= token_start:
            continue
        for chunk_id, (span_start, span_end) in enumerate(rendered_spans):
            if token_start < span_end and token_end > span_start:
                chunk_positions[chunk_id].append(position)
                segment_ids[position] = int(Segment.EVIDENCE)

    chunks: list[EvidenceChunk] = []
    for chunk_id, ((char_start, char_end), positions) in enumerate(
        zip(rendered_spans, chunk_positions, strict=True)
    ):
        if not positions:
            raise ValueError(f"evidence chunk {chunk_id} produced no tokens")
        chunks.append(
            EvidenceChunk(
                chunk_id=chunk_id,
                text=rendered_text[char_start:char_end],
                char_start=char_start,
                char_end=char_end,
                token_positions=torch.tensor(positions, dtype=torch.long),
            )
        )
    evidence_positions = torch.nonzero(
        segment_ids == int(Segment.EVIDENCE), as_tuple=False
    ).flatten()
    return RagTruthTokenLayout(
        rendered_text=rendered_text,
        input_ids=input_ids,
        segment_ids=segment_ids,
        response_idx=response_idx,
        evidence_token_positions=evidence_positions,
        evidence_chunks=tuple(chunks),
    )


__all__ = [
    "SYSTEM_PROMPT",
    "EvidenceChunk",
    "RagTruthTokenLayout",
    "build_ragtruth_layout",
]
