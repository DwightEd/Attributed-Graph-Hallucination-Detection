"""Audit clean and hallucinated RAGTruth samples by concrete error pattern."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from typing import Any


PATTERN_PRIORITY = (
    "source_attribution",
    "entity_attribute_binding",
    "numeric_temporal",
    "polarity_negation",
    "epistemic_inference",
    "relation_predicate",
    "unsupported_addition",
    "unclassified",
)

MONTHS = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)

SOURCE_ATTRIBUTION_RE = re.compile(
    r"\b(passage|source|document|article|table|row|review)\s*(?:no\.?\s*)?\d+\b",
    re.IGNORECASE,
)
NUMERIC_TEMPORAL_RE = re.compile(
    rf"(?:[$€£%]|\b\d[\d,.]*\b|\b(?:{MONTHS})\b|"
    r"\b(?:day|days|week|weeks|month|months|year|years|age|aged|"
    r"percent|percentage|million|billion)\b)",
    re.IGNORECASE,
)
POLARITY_RE = re.compile(
    r"\b(?:no|not|never|none|without|cannot|can't|unable|"
    r"available|unavailable|free|higher|lower|more|less|yes)\b",
    re.IGNORECASE,
)
EPISTEMIC_RE = re.compile(
    r"\b(?:likely|unlikely|probably|possibly|may|might|suggest(?:s|ed)?|"
    r"believ(?:e|ed)|implied?|unclear|exact|conclud(?:e|ed)|"
    r"reported?|estimated?|alleged(?:ly)?)\b",
    re.IGNORECASE,
)
ENTITY_BINDING_RE = re.compile(
    r"\b(?:net worth|attribute(?:d)? to|assigned to|belongs? to|"
    r"refers? to|instead of|whose|subject|entity|person|name)\b",
    re.IGNORECASE,
)


def _label_axes(label_type: str) -> tuple[str, str]:
    normalized = label_type.casefold()
    support_relation = (
        "baseless" if "baseless" in normalized else
        "conflict" if "conflict" in normalized else
        "unknown"
    )
    severity = (
        "evident" if "evident" in normalized else
        "subtle" if "subtle" in normalized else
        "unknown"
    )
    return support_relation, severity


def classify_error_pattern(label: Mapping[str, Any]) -> dict[str, Any]:
    """Map one annotation to orthogonal support, severity, and mechanism axes.

    The mechanism tags are deterministic heuristics for auditing. They are not
    intended to replace RAGTruth's human labels or to be used as train labels.
    """

    label_type = str(label.get("label_type") or "")
    text = str(label.get("text") or "")
    meta = str(label.get("meta") or "")
    searchable = f"{text}\n{meta}"
    support_relation, severity = _label_axes(label_type)
    tags: set[str] = set()

    if support_relation == "baseless":
        tags.add("unsupported_addition")
    if support_relation == "conflict":
        tags.add("relation_predicate")
    if SOURCE_ATTRIBUTION_RE.search(searchable):
        tags.add("source_attribution")
    if ENTITY_BINDING_RE.search(searchable):
        tags.add("entity_attribute_binding")
    if NUMERIC_TEMPORAL_RE.search(searchable):
        tags.add("numeric_temporal")
    if POLARITY_RE.search(searchable):
        tags.add("polarity_negation")
    if EPISTEMIC_RE.search(searchable):
        tags.add("epistemic_inference")
    if not tags:
        tags.add("unclassified")

    ordered_tags = [name for name in PATTERN_PRIORITY if name in tags]
    return {
        "support_relation": support_relation,
        "severity": severity,
        "primary_pattern": ordered_tags[0],
        "pattern_tags": ordered_tags,
    }


def merge_intervals(
    intervals: Iterable[tuple[int, int]],
    *,
    upper_bound: int | None = None,
) -> list[tuple[int, int]]:
    """Clamp, discard empty ranges, and merge overlapping intervals."""

    normalized = []
    for start, end in intervals:
        start = max(0, int(start))
        end = int(end)
        if upper_bound is not None:
            start = min(start, upper_bound)
            end = min(end, upper_bound)
        if end > start:
            normalized.append((start, end))

    merged: list[tuple[int, int]] = []
    for start, end in sorted(normalized):
        if not merged or start > merged[-1][1]:
            merged.append((start, end))
        else:
            previous_start, previous_end = merged[-1]
            merged[-1] = (previous_start, max(previous_end, end))
    return merged
