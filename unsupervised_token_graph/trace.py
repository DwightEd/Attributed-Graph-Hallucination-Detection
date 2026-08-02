"""Single-tokenization segment alignment for model traces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence


SEGMENT_ID = {"passage": 1, "question": 2, "answer": 3}


def assign_segment_ids(
    offset_mapping: Sequence[tuple[int, int]],
    segment_char_spans: Mapping[str, tuple[int, int]],
    special_tokens_mask: Sequence[int | bool],
) -> list[int]:
    """Assign final-tokenization offsets to template/passage/question/answer."""

    if len(offset_mapping) != len(special_tokens_mask):
        raise ValueError("offset_mapping and special_tokens_mask lengths differ")
    missing = set(SEGMENT_ID).difference(segment_char_spans)
    if missing:
        raise ValueError(f"Missing character spans: {sorted(missing)}")

    segment_ids: list[int] = []
    for (token_start, token_end), is_special in zip(
        offset_mapping, special_tokens_mask
    ):
        if is_special or token_end <= token_start:
            segment_ids.append(0)
            continue
        best_segment = 0
        best_overlap = 0
        for name, identifier in SEGMENT_ID.items():
            span_start, span_end = segment_char_spans[name]
            overlap = max(
                0,
                min(token_end, span_end) - max(token_start, span_start),
            )
            if overlap > best_overlap:
                best_segment = identifier
                best_overlap = overlap
        segment_ids.append(best_segment)
    return segment_ids
