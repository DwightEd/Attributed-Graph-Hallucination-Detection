"""Strict, label-blind contracts for token-aligned evidence interventions."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from counterfactual_grounding.data.ragtruth import RagTruthTokenLayout


@dataclass(frozen=True)
class CounterfactualAudit:
    """Alignment facts established before any model intervention is run."""

    token_count: int
    response_idx: int
    changed_positions: torch.Tensor
    predictor_positions: torch.Tensor
    response_token_ids: torch.Tensor


class CounterfactualGenerationError(ValueError):
    """No registered minimal edit satisfies the fixed-token contract."""


@dataclass(frozen=True)
class EqualTokenCounterfactual:
    factual_text: str
    counterfactual_text: str
    factual_input_ids: torch.Tensor
    counterfactual_input_ids: torch.Tensor
    changed_positions: torch.Tensor
    changed_char_span: tuple[int, int]
    original_text: str
    replacement_text: str
    audit: CounterfactualAudit


def _reject_labels(record: Mapping[str, object]) -> None:
    forbidden = [
        str(key)
        for key in record
        if "label" in str(key).casefold() or str(key).casefold() == "y_token"
    ]
    if forbidden:
        raise ValueError(
            "counterfactual teacher records must remain label-blind; forbidden "
            f"fields: {', '.join(sorted(forbidden))}"
        )


def _long_vector(record: Mapping[str, object], name: str) -> torch.Tensor:
    if name not in record:
        raise ValueError(f"counterfactual record is missing {name}")
    value = torch.as_tensor(record[name]).detach().cpu().to(torch.long)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return value


def validate_counterfactual_pair(
    factual: Mapping[str, object],
    counterfactual: Mapping[str, object],
) -> CounterfactualAudit:
    """Require an equal-length, evidence-only edit with an unchanged response.

    This function intentionally knows nothing about hallucination labels.  It
    establishes the fixed-position contract needed by RoPE and K/V mediation.
    """

    _reject_labels(factual)
    _reject_labels(counterfactual)
    factual_ids = _long_vector(factual, "input_ids")
    counterfactual_ids = _long_vector(counterfactual, "input_ids")
    if factual_ids.shape != counterfactual_ids.shape:
        raise ValueError("factual and counterfactual inputs require equal token length")
    token_count = int(factual_ids.numel())
    factual_response_idx = int(factual.get("response_idx", -1))
    counterfactual_response_idx = int(counterfactual.get("response_idx", -1))
    if factual_response_idx != counterfactual_response_idx:
        raise ValueError("counterfactual response_idx must preserve predictor positions")
    if not 0 < factual_response_idx < token_count:
        raise ValueError("response_idx must split a non-empty prompt and response")

    factual_evidence = _long_vector(factual, "evidence_token_positions")
    counterfactual_evidence = _long_vector(
        counterfactual, "evidence_token_positions"
    )
    if not torch.equal(factual_evidence, counterfactual_evidence):
        raise ValueError("counterfactual evidence positions must remain unchanged")
    if factual_evidence.numel() == 0:
        raise ValueError("at least one evidence token position is required")
    if bool(
        ((factual_evidence < 0) | (factual_evidence >= factual_response_idx)).any()
    ):
        raise ValueError("evidence token positions must lie inside the prompt")
    if torch.unique(factual_evidence).numel() != factual_evidence.numel():
        raise ValueError("evidence token positions must be unique")

    changed = torch.nonzero(factual_ids != counterfactual_ids, as_tuple=False).flatten()
    if changed.numel() == 0:
        raise ValueError("counterfactual must change at least one evidence token")
    allowed = torch.zeros(token_count, dtype=torch.bool)
    allowed[factual_evidence] = True
    if bool((~allowed[changed]).any()):
        raise ValueError("every changed position must be registered as evidence")
    if not torch.equal(
        factual_ids[factual_response_idx:], counterfactual_ids[factual_response_idx:]
    ):
        raise ValueError("counterfactual intervention must preserve response token ids")

    predictor_positions = torch.arange(
        factual_response_idx - 1, token_count - 1, dtype=torch.long
    )
    return CounterfactualAudit(
        token_count=token_count,
        response_idx=factual_response_idx,
        changed_positions=changed,
        predictor_positions=predictor_positions,
        response_token_ids=factual_ids[factual_response_idx:].clone(),
    )


def _token_ids(tokenizer: object, text: str) -> torch.Tensor:
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    if not isinstance(encoding, Mapping) or "input_ids" not in encoding:
        raise CounterfactualGenerationError("tokenizer did not return input_ids")
    value = encoding["input_ids"]
    tensor = torch.as_tensor(value, dtype=torch.long)
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.ndim != 1:
        raise CounterfactualGenerationError(
            "tokenizer input_ids must describe exactly one sequence"
        )
    return tensor.detach().cpu()


def generate_equal_token_counterfactual(
    layout: RagTruthTokenLayout,
    tokenizer: object,
) -> EqualTokenCounterfactual:
    """Find the first one-digit evidence edit preserving every token position.

    This conservative Gate-1 adapter changes neither entity type nor surface
    width.  A sample is marked teacher-unavailable when no digit edit survives
    exact whole-sequence token replay; it is never repaired with padding,
    truncation, or a cross-sample donor.
    """

    return generate_equal_token_counterfactuals(
        layout, tokenizer, max_candidates=1
    )[0]


def generate_equal_token_counterfactuals(
    layout: RagTruthTokenLayout,
    tokenizer: object,
    *,
    max_candidates: int = 2,
    max_attempts: int = 512,
) -> tuple[EqualTokenCounterfactual, ...]:
    """Generate several independently audited minimal numeric candidates."""

    if max_candidates <= 0 or max_attempts <= 0:
        raise ValueError("max_candidates and max_attempts must be positive")
    factual_text = layout.rendered_text
    replayed_factual = _token_ids(tokenizer, factual_text)
    factual_ids = layout.input_ids.detach().cpu().long().flatten()
    if not torch.equal(replayed_factual, factual_ids):
        raise CounterfactualGenerationError(
            "factual tokenizer replay does not match the registered layout"
        )
    digit_positions = [
        position
        for chunk in layout.evidence_chunks
        for position in range(chunk.char_start, chunk.char_end)
        if factual_text[position].isdigit()
    ]
    if not digit_positions:
        raise CounterfactualGenerationError(
            "evidence contains no numeric digit counterfactual candidate"
        )

    factual_record = {
        "input_ids": factual_ids,
        "response_idx": layout.response_idx,
        "evidence_token_positions": layout.evidence_token_positions,
    }
    results: list[EqualTokenCounterfactual] = []
    attempts = 0
    # Vary distinct evidence characters before trying a more distant value at
    # the same character.  This makes a two-candidate stability audit less
    # likely to be two near-duplicates of one field.
    for offset in range(1, 10):
        for char_position in digit_positions:
            attempts += 1
            if attempts > max_attempts:
                if results:
                    return tuple(results)
                raise CounterfactualGenerationError(
                    "numeric candidate search exceeded the registered attempt limit"
                )
            original = factual_text[char_position]
            replacement = str((int(original) + offset) % 10)
            candidate_text = (
                factual_text[:char_position]
                + replacement
                + factual_text[char_position + 1 :]
            )
            candidate_ids = _token_ids(tokenizer, candidate_text)
            if candidate_ids.shape != factual_ids.shape:
                continue
            try:
                audit = validate_counterfactual_pair(
                    factual_record,
                    {
                        "input_ids": candidate_ids,
                        "response_idx": layout.response_idx,
                        "evidence_token_positions": layout.evidence_token_positions,
                    },
                )
            except ValueError:
                continue
            results.append(
                EqualTokenCounterfactual(
                    factual_text=factual_text,
                    counterfactual_text=candidate_text,
                    factual_input_ids=factual_ids.clone(),
                    counterfactual_input_ids=candidate_ids,
                    changed_positions=audit.changed_positions.clone(),
                    changed_char_span=(char_position, char_position + 1),
                    original_text=original,
                    replacement_text=replacement,
                    audit=audit,
                )
            )
            if len(results) == max_candidates:
                return tuple(results)
    if results:
        return tuple(results)
    raise CounterfactualGenerationError(
        "no numeric candidate preserved equal-token length, response suffix, "
        "and evidence-only token changes"
    )


__all__ = [
    "CounterfactualAudit",
    "CounterfactualGenerationError",
    "EqualTokenCounterfactual",
    "generate_equal_token_counterfactual",
    "generate_equal_token_counterfactuals",
    "validate_counterfactual_pair",
]
