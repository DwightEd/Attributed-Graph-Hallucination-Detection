"""Attention-derived observables for token-graph pattern analysis."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch


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
    edge_presence=None,
) -> dict[str, np.ndarray | float | torch.Tensor]:
    """Summarize how answer tokens route attention across input segments.

    The input shape is ``(layers, heads, query_tokens, key_tokens)``. Labels
    are deliberately absent from this interface.
    """

    if isinstance(attention, torch.Tensor):
        return _summarize_attention_tensor(
            attention,
            segment_token_spans,
            edge_threshold=edge_threshold,
            edge_presence=edge_presence,
        )

    values = _as_numpy(attention)
    if values.ndim != 4 or values.shape[-1] != values.shape[-2]:
        raise ValueError(
            "attention must have shape (layers, heads, tokens, tokens)"
        )
    _validate_token_spans(segment_token_spans, values.shape[-1])
    answer_start, answer_end = segment_token_spans["answer"]
    answer_rows = values[:, :, answer_start:answer_end, :]
    query_positions = np.arange(answer_start, answer_end).reshape(1, 1, -1, 1)
    key_positions = np.arange(values.shape[-1]).reshape(1, 1, 1, -1)
    causal_answer_rows = np.where(key_positions < query_positions, answer_rows, 0.0)

    masses: dict[str, np.ndarray] = {}
    for name in ("passage", "question"):
        start, end = segment_token_spans[name]
        masses[name] = causal_answer_rows[..., start:end].sum(axis=-1).mean(axis=-1)
    masses["answer"] = causal_answer_rows[
        ..., answer_start:answer_end
    ].sum(axis=-1).mean(axis=-1)

    covered_mass = sum(masses.values())
    safe_covered_mass = np.where(covered_mass > 0, covered_mass, 1.0)
    normalized_rows = causal_answer_rows / np.clip(
        causal_answer_rows.sum(axis=-1, keepdims=True), 1e-12, None
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
    if edge_presence is None:
        edge_presence = values.max(axis=(0, 1)) > edge_threshold
    else:
        edge_presence = np.asarray(edge_presence, dtype=bool)
        if edge_presence.shape != values.shape[-2:]:
            raise ValueError("edge_presence must have shape (tokens, tokens)")
    for query in range(answer_start, answer_end):
        causal_slots += query
        active_slots += int(edge_presence[query, :query].sum())

    passage_ratio = masses["passage"] / safe_covered_mass
    question_ratio = masses["question"] / safe_covered_mass
    answer_ratio = masses["answer"] / safe_covered_mass
    passage_length = segment_token_spans["passage"][1] - segment_token_spans["passage"][0]
    question_length = segment_token_spans["question"][1] - segment_token_spans["question"][0]
    answer_length = answer_end - answer_start
    mean_prior_slots = max((answer_length - 1) / 2.0, 1.0)
    token_densities = {
        "passage": masses["passage"] / passage_length,
        "question": masses["question"] / question_length,
        "answer": masses["answer"] / mean_prior_slots,
    }
    density_total = sum(token_densities.values())
    safe_density_total = np.where(density_total > 0, density_total, 1.0)
    return {
        "answer_to_passage_mass": masses["passage"],
        "answer_to_question_mass": masses["question"],
        "answer_to_answer_mass": masses["answer"],
        "answer_to_prior_answer_mass": masses["answer"],
        "answer_to_passage_ratio": passage_ratio,
        "answer_to_question_ratio": question_ratio,
        "answer_self_reliance": answer_ratio,
        "answer_prior_reliance": answer_ratio,
        "answer_to_passage_token_normalized": (
            token_densities["passage"] / safe_density_total
        ),
        "answer_to_question_token_normalized": (
            token_densities["question"] / safe_density_total
        ),
        "answer_to_prior_answer_token_normalized": (
            token_densities["answer"] / safe_density_total
        ),
        "answer_attention_entropy": entropy.mean(axis=-1),
        "passage_head_disagreement": float(passage_ratio.std(axis=1).mean()),
        "passage_layer_drift": float(
            passage_ratio[-1].mean() - passage_ratio[len(passage_ratio) // 2].mean()
        ),
        "answer_edge_density": float(active_slots / max(causal_slots, 1)),
    }


def _summarize_attention_tensor(
    attention: torch.Tensor,
    segment_token_spans: Mapping[str, tuple[int, int]],
    *,
    edge_threshold: float,
    edge_presence,
) -> dict[str, torch.Tensor]:
    """Run the dense attention reductions without leaving the tensor device."""

    values = attention.detach()
    if values.ndim != 4 or values.shape[-1] != values.shape[-2]:
        raise ValueError(
            "attention must have shape (layers, heads, tokens, tokens)"
        )
    _validate_token_spans(segment_token_spans, values.shape[-1])
    device = values.device
    answer_start, answer_end = segment_token_spans["answer"]

    # Only answer-query rows need float32 arithmetic. Keeping the full dense
    # attention tensor in its model dtype avoids a second multi-GiB allocation.
    answer_rows = values[:, :, answer_start:answer_end, :].to(torch.float32)
    query_positions = torch.arange(
        answer_start, answer_end, device=device
    ).reshape(1, 1, -1, 1)
    key_positions = torch.arange(values.shape[-1], device=device).reshape(
        1, 1, 1, -1
    )
    causal_answer_rows = torch.where(
        key_positions < query_positions,
        answer_rows,
        torch.zeros((), dtype=answer_rows.dtype, device=device),
    )

    masses: dict[str, torch.Tensor] = {}
    for name in ("passage", "question"):
        start, end = segment_token_spans[name]
        masses[name] = causal_answer_rows[..., start:end].sum(dim=-1).mean(dim=-1)
    masses["answer"] = causal_answer_rows[
        ..., answer_start:answer_end
    ].sum(dim=-1).mean(dim=-1)

    covered_mass = masses["passage"] + masses["question"] + masses["answer"]
    safe_covered_mass = torch.where(
        covered_mass > 0,
        covered_mass,
        torch.ones_like(covered_mass),
    )
    normalized_rows = causal_answer_rows / causal_answer_rows.sum(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    entropy = -torch.where(
        normalized_rows > 0,
        normalized_rows * normalized_rows.clamp_min(1e-12).log(),
        torch.zeros_like(normalized_rows),
    ).sum(dim=-1)
    entropy = entropy / max(float(np.log(values.shape[-1])), 1.0)

    if edge_presence is None:
        edge_presence = values.amax(dim=(0, 1)).to(torch.float32) > float(
            edge_threshold
        )
    else:
        edge_presence = torch.as_tensor(
            edge_presence, dtype=torch.bool, device=device
        ).detach()
        if edge_presence.shape != values.shape[-2:]:
            raise ValueError("edge_presence must have shape (tokens, tokens)")
    causal_presence = torch.tril(edge_presence, diagonal=-1)
    active_slots = causal_presence[answer_start:answer_end].sum(dtype=torch.float32)
    causal_slots = sum(range(answer_start, answer_end))

    passage_ratio = masses["passage"] / safe_covered_mass
    question_ratio = masses["question"] / safe_covered_mass
    answer_ratio = masses["answer"] / safe_covered_mass
    passage_length = (
        segment_token_spans["passage"][1] - segment_token_spans["passage"][0]
    )
    question_length = (
        segment_token_spans["question"][1] - segment_token_spans["question"][0]
    )
    answer_length = answer_end - answer_start
    mean_prior_slots = max((answer_length - 1) / 2.0, 1.0)
    token_densities = {
        "passage": masses["passage"] / passage_length,
        "question": masses["question"] / question_length,
        "answer": masses["answer"] / mean_prior_slots,
    }
    density_total = (
        token_densities["passage"]
        + token_densities["question"]
        + token_densities["answer"]
    )
    safe_density_total = torch.where(
        density_total > 0,
        density_total,
        torch.ones_like(density_total),
    )
    return {
        "answer_to_passage_mass": masses["passage"],
        "answer_to_question_mass": masses["question"],
        "answer_to_answer_mass": masses["answer"],
        "answer_to_prior_answer_mass": masses["answer"],
        "answer_to_passage_ratio": passage_ratio,
        "answer_to_question_ratio": question_ratio,
        "answer_self_reliance": answer_ratio,
        "answer_prior_reliance": answer_ratio,
        "answer_to_passage_token_normalized": (
            token_densities["passage"] / safe_density_total
        ),
        "answer_to_question_token_normalized": (
            token_densities["question"] / safe_density_total
        ),
        "answer_to_prior_answer_token_normalized": (
            token_densities["answer"] / safe_density_total
        ),
        "answer_attention_entropy": entropy.mean(dim=-1),
        "passage_head_disagreement": passage_ratio.std(
            dim=1, correction=0
        ).mean(),
        "passage_layer_drift": (
            passage_ratio[-1].mean()
            - passage_ratio[len(passage_ratio) // 2].mean()
        ),
        "answer_edge_density": active_slots / max(causal_slots, 1),
    }
