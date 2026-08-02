"""Attention-derived observables for token-graph pattern analysis."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


SEGMENT_NAMES = ("passage", "question", "answer")


def _as_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _validate_token_spans(
    spans: Mapping[str, tuple[int, int]],
    sequence_length: int,
) -> None:
    missing = set(SEGMENT_NAMES).difference(spans)
    if missing:
        raise ValueError(f"Missing token spans: {sorted(missing)}")
    previous_end = 0
    for name in SEGMENT_NAMES:
        start, end = spans[name]
        if not 0 <= start < end <= sequence_length:
            raise ValueError(f"Invalid {name} token span {(start, end)}")
        if start < previous_end:
            raise ValueError("Token spans must be ordered and non-overlapping")
        previous_end = end


def summarize_attention_trace(
    attention,
    segment_token_spans: Mapping[str, tuple[int, int]],
    *,
    edge_threshold: float = 0.05,
) -> dict[str, np.ndarray | float]:
    """Summarize how answer tokens route attention across input segments.

    The input shape is ``(layers, heads, query_tokens, key_tokens)``. Labels
    are deliberately absent from this interface.
    """

    values = _as_numpy(attention)
    if values.ndim != 4 or values.shape[-1] != values.shape[-2]:
        raise ValueError(
            "attention must have shape (layers, heads, tokens, tokens)"
        )
    _validate_token_spans(segment_token_spans, values.shape[-1])
    answer_start, answer_end = segment_token_spans["answer"]
    answer_rows = values[:, :, answer_start:answer_end, :]

    masses: dict[str, np.ndarray] = {}
    for name in SEGMENT_NAMES:
        start, end = segment_token_spans[name]
        masses[name] = answer_rows[..., start:end].sum(axis=-1).mean(axis=-1)

    covered_mass = sum(masses.values())
    safe_covered_mass = np.where(covered_mass > 0, covered_mass, 1.0)
    normalized_rows = answer_rows / np.clip(
        answer_rows.sum(axis=-1, keepdims=True), 1e-12, None
    )
    entropy = -np.sum(
        np.where(
            normalized_rows > 0,
            normalized_rows * np.log(np.clip(normalized_rows, 1e-12, None)),
            0.0,
        ),
        axis=-1,
    )
    entropy /= max(np.log(values.shape[-1]), 1.0)

    causal_slots = 0
    active_slots = 0
    edge_presence = values.max(axis=(0, 1)) > edge_threshold
    for query in range(answer_start, answer_end):
        causal_slots += query
        active_slots += int(edge_presence[query, :query].sum())

    passage_ratio = masses["passage"] / safe_covered_mass
    question_ratio = masses["question"] / safe_covered_mass
    answer_ratio = masses["answer"] / safe_covered_mass
    return {
        "answer_to_passage_mass": masses["passage"],
        "answer_to_question_mass": masses["question"],
        "answer_to_answer_mass": masses["answer"],
        "answer_to_passage_ratio": passage_ratio,
        "answer_to_question_ratio": question_ratio,
        "answer_self_reliance": answer_ratio,
        "answer_attention_entropy": entropy.mean(axis=-1),
        "passage_head_disagreement": float(passage_ratio.std(axis=1).mean()),
        "passage_layer_drift": float(
            passage_ratio[-1].mean() - passage_ratio[len(passage_ratio) // 2].mean()
        ),
        "answer_edge_density": float(active_slots / max(causal_slots, 1)),
    }
