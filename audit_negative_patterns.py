"""Audit clean and hallucinated RAGTruth samples by concrete error pattern."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any


KNOWN_LABEL_TYPES = {
    "Evident Baseless Info",
    "Evident Conflict",
    "Subtle Baseless Info",
    "Subtle Conflict",
}

PATTERN_GUIDANCE = {
    "source_attribution": (
        "引用了错误 passage/source",
        "typed source→claim 边或边重构",
    ),
    "entity_attribute_binding": (
        "实体与属性/数值错误绑定",
        "entity–slot–value 异构边重构",
    ),
    "numeric_temporal": (
        "数字、日期、时长或数量变化",
        "数值节点属性重构与 source-copy 对齐",
    ),
    "polarity_negation": (
        "否定、可用性、大小方向等极性反转",
        "带符号/关系类型的兼容性重构",
    ),
    "epistemic_inference": (
        "可能性、确定性或推断强度发生变化",
        "claim modality 属性重构",
    ),
    "relation_predicate": (
        "证据存在但谓词或关系不一致",
        "relation-aware edge compatibility",
    ),
    "unsupported_addition": (
        "生成了来源中没有的信息",
        "source-support deficit 与 masked claim reconstruction",
    ),
    "unclassified": (
        "现有启发式未覆盖",
        "人工复核后扩展规则",
    ),
}

SAMPLE_FIELDS = (
    "response_id",
    "source_id",
    "task",
    "model",
    "split",
    "quality",
    "sample_class",
    "response_characters",
    "annotation_count",
    "hallucinated_union_characters",
    "hallucinated_character_rate",
    "label_types",
    "primary_patterns",
)

SPAN_FIELDS = (
    "response_id",
    "source_id",
    "task",
    "model",
    "split",
    "label_type",
    "support_relation",
    "severity",
    "primary_pattern",
    "pattern_tags",
    "start",
    "end",
    "valid_bounds",
    "span_text_matches",
    "span_text",
    "actual_text",
    "meta",
    "context",
)

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


@dataclass(frozen=True)
class AuditResult:
    report: dict[str, Any]
    sample_rows: list[dict[str, Any]]
    span_rows: list[dict[str, Any]]


def _new_group() -> dict[str, int]:
    return {
        "responses": 0,
        "clean_responses": 0,
        "hallucinated_responses": 0,
        "annotated_spans": 0,
        "multi_span_hallucinated_responses": 0,
        "response_characters": 0,
        "hallucinated_union_characters": 0,
    }


def _update_group(
    group: dict[str, int],
    *,
    hallucinated: bool,
    span_count: int,
    response_characters: int,
    hallucinated_characters: int,
) -> None:
    group["responses"] += 1
    class_key = "hallucinated_responses" if hallucinated else "clean_responses"
    group[class_key] += 1
    group["annotated_spans"] += span_count
    group["response_characters"] += response_characters
    group["hallucinated_union_characters"] += hallucinated_characters
    if span_count > 1:
        group["multi_span_hallucinated_responses"] += 1


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _finalize_group(group: Mapping[str, int]) -> dict[str, int | float]:
    finalized: dict[str, int | float] = dict(group)
    finalized["hallucinated_response_rate"] = _rate(
        group["hallucinated_responses"], group["responses"]
    )
    finalized["hallucinated_character_rate"] = _rate(
        group["hallucinated_union_characters"], group["response_characters"]
    )
    finalized["multi_span_share_of_hallucinated"] = _rate(
        group["multi_span_hallucinated_responses"],
        group["hallucinated_responses"],
    )
    return finalized


def _load_sources(
    source_path: Path,
) -> tuple[dict[str, dict[str, Any]], int]:
    sources: dict[str, dict[str, Any]] = {}
    malformed_rows = 0
    with source_path.open("r", encoding="utf-8") as source_file:
        for line in source_file:
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError:
                malformed_rows += 1
                continue
            source_id = str(source.get("source_id") or "")
            if source_id:
                sources[source_id] = source
            else:
                malformed_rows += 1
    return sources, malformed_rows


def _label_bounds(label: Mapping[str, Any]) -> tuple[int, int] | None:
    try:
        return int(label["start"]), int(label["end"])
    except (KeyError, TypeError, ValueError):
        return None


def _context(response: str, start: int, end: int, radius: int = 90) -> str:
    left = max(0, start - radius)
    right = min(len(response), end + radius)
    return response[left:right].replace("\n", " ")


def audit_dataset(
    responses_path: str | Path,
    sources_path: str | Path,
    *,
    max_examples: int = 5,
) -> AuditResult:
    """Stream RAGTruth JSONL files and return sample/span-level audit records."""

    responses_path = Path(responses_path)
    sources_path = Path(sources_path)
    sources, malformed_source_rows = _load_sources(sources_path)
    quality = Counter({"malformed_source_rows": malformed_source_rows})
    total = _new_group()
    groups: dict[str, defaultdict[str, dict[str, int]]] = {
        "task": defaultdict(_new_group),
        "model": defaultdict(_new_group),
        "split": defaultdict(_new_group),
    }
    label_types: Counter[str] = Counter()
    primary_patterns: Counter[str] = Counter()
    pattern_tags: Counter[str] = Counter()
    task_patterns: defaultdict[str, Counter[str]] = defaultdict(Counter)
    examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    clean_examples: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_rows: list[dict[str, Any]] = []
    span_rows: list[dict[str, Any]] = []

    with responses_path.open("r", encoding="utf-8") as response_file:
        for line in response_file:
            if not line.strip():
                continue
            try:
                response_record = json.loads(line)
            except json.JSONDecodeError:
                quality["malformed_response_rows"] += 1
                continue

            response_id = str(response_record.get("id") or "")
            source_id = str(response_record.get("source_id") or "")
            source = sources.get(source_id)
            if source is None:
                quality["missing_source_responses"] += 1
            task = str((source or {}).get("task_type") or "__missing__")
            model = str(response_record.get("model") or "__missing__")
            split = str(response_record.get("split") or "__missing__")
            response = str(response_record.get("response") or "")
            labels = response_record.get("labels") or []
            if not isinstance(labels, list):
                quality["non_list_label_fields"] += 1
                labels = []

            valid_intervals: list[tuple[int, int]] = []
            sample_primary_patterns: set[str] = set()
            sample_label_types: set[str] = set()
            for label in labels:
                if not isinstance(label, Mapping):
                    quality["malformed_labels"] += 1
                    continue
                label_type = str(label.get("label_type") or "")
                classification = classify_error_pattern(label)
                bounds = _label_bounds(label)
                start, end = bounds if bounds is not None else (-1, -1)
                valid_bounds = (
                    bounds is not None
                    and 0 <= start < end <= len(response)
                )
                if not valid_bounds:
                    quality["invalid_span_bounds"] += 1
                else:
                    valid_intervals.append((start, end))

                expected_text = str(label.get("text") or "")
                actual_text = response[start:end] if valid_bounds else ""
                text_matches = actual_text == expected_text
                if not text_matches:
                    quality["span_text_mismatches"] += 1
                if label_type not in KNOWN_LABEL_TYPES:
                    quality["unknown_label_types"] += 1

                label_types[label_type or "__missing__"] += 1
                primary = classification["primary_pattern"]
                primary_patterns[primary] += 1
                sample_primary_patterns.add(primary)
                sample_label_types.add(label_type or "__missing__")
                task_patterns[task][primary] += 1
                for tag in classification["pattern_tags"]:
                    pattern_tags[tag] += 1

                span_row = {
                    "response_id": response_id,
                    "source_id": source_id,
                    "task": task,
                    "model": model,
                    "split": split,
                    "label_type": label_type,
                    **classification,
                    "start": start,
                    "end": end,
                    "valid_bounds": valid_bounds,
                    "span_text_matches": text_matches,
                    "span_text": expected_text,
                    "actual_text": actual_text,
                    "meta": str(label.get("meta") or "").replace("\n", " "),
                    "context": _context(response, max(0, start), max(0, end)),
                }
                span_rows.append(span_row)
                if len(examples[primary]) < max_examples:
                    examples[primary].append(span_row.copy())

            merged = merge_intervals(valid_intervals, upper_bound=len(response))
            hallucinated_characters = sum(end - start for start, end in merged)
            raw_valid_characters = sum(end - start for start, end in valid_intervals)
            if raw_valid_characters > hallucinated_characters:
                quality["overlapping_span_responses"] += 1
            hallucinated = bool(labels)
            span_count = len(labels)
            group_values = {
                "hallucinated": hallucinated,
                "span_count": span_count,
                "response_characters": len(response),
                "hallucinated_characters": hallucinated_characters,
            }
            _update_group(total, **group_values)
            _update_group(groups["task"][task], **group_values)
            _update_group(groups["model"][model], **group_values)
            _update_group(groups["split"][split], **group_values)

            sample_row = {
                "response_id": response_id,
                "source_id": source_id,
                "task": task,
                "model": model,
                "split": split,
                "quality": str(response_record.get("quality") or ""),
                "sample_class": "hallucinated" if hallucinated else "clean",
                "response_characters": len(response),
                "annotation_count": span_count,
                "hallucinated_union_characters": hallucinated_characters,
                "hallucinated_character_rate": _rate(
                    hallucinated_characters, len(response)
                ),
                "label_types": sorted(sample_label_types),
                "primary_patterns": sorted(sample_primary_patterns),
            }
            sample_rows.append(sample_row)
            if not hallucinated and len(clean_examples[task]) < max_examples:
                clean_examples[task].append(
                    {
                        "response_id": response_id,
                        "source_id": source_id,
                        "model": model,
                        "response_preview": response[:240].replace("\n", " "),
                    }
                )

    breakdown_groups = {
        axis: {
            key: _finalize_group(value)
            for key, value in sorted(axis_groups.items())
        }
        for axis, axis_groups in groups.items()
    }
    report = {
        "schema_version": 1,
        "inputs": {
            "responses": str(responses_path),
            "sources": str(sources_path),
        },
        "summary": _finalize_group(total),
        "data_quality": dict(sorted(quality.items())),
        "breakdowns": {
            **breakdown_groups,
            "label_type": dict(sorted(label_types.items())),
            "primary_pattern": dict(sorted(primary_patterns.items())),
            "pattern_tag": dict(sorted(pattern_tags.items())),
            "task_primary_pattern": {
                task: dict(sorted(counts.items()))
                for task, counts in sorted(task_patterns.items())
            },
        },
        "examples": {
            "by_primary_pattern": dict(sorted(examples.items())),
            "clean_by_task": dict(sorted(clean_examples.items())),
        },
    }
    return AuditResult(report=report, sample_rows=sample_rows, span_rows=span_rows)


def _markdown_cell(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _percent(value: float | int) -> str:
    return f"{float(value) * 100:.2f}%"


def render_markdown_report(report: Mapping[str, Any]) -> str:
    """Render a human-readable audit report from the JSON-compatible report."""

    summary = report["summary"]
    breakdowns = report["breakdowns"]
    lines = [
        "# RAGTruth 正负样本错误模式审计",
        "",
        "> `clean` 表示没有人工标注 hallucination span；`hallucinated` 表示至少有一个标注 span。"
        "错误机制是用于研究审计的可解释启发式标签，不应作为训练标签。",
        "",
        "## 总体分布",
        "",
        "| 指标 | 数值 |",
        "|---|---:|",
        f"| 回答总数 | {summary['responses']} |",
        f"| clean 回答 | {summary['clean_responses']} |",
        f"| hallucinated 回答 | {summary['hallucinated_responses']} |",
        f"| 回答级幻觉比例 | {_percent(summary['hallucinated_response_rate'])} |",
        f"| 标注 span 数 | {summary['annotated_spans']} |",
        f"| 错误字符覆盖率 | {_percent(summary['hallucinated_character_rate'])} |",
        f"| 多错误回答占 hallucinated 比例 | "
        f"{_percent(summary['multi_span_share_of_hallucinated'])} |",
        "",
        "## 按任务分布",
        "",
        "| 任务 | 回答数 | clean | hallucinated | 回答级比例 | 错误字符覆盖率 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for task, values in breakdowns["task"].items():
        lines.append(
            f"| {_markdown_cell(task)} | {values['responses']} | "
            f"{values['clean_responses']} | {values['hallucinated_responses']} | "
            f"{_percent(values['hallucinated_response_rate'])} | "
            f"{_percent(values['hallucinated_character_rate'])} |"
        )

    lines.extend(
        [
            "",
            "## 官方标注类型",
            "",
            "| 标签 | span 数 |",
            "|---|---:|",
        ]
    )
    for label_type, count in breakdowns["label_type"].items():
        lines.append(f"| {_markdown_cell(label_type)} | {count} |")

    lines.extend(
        [
            "",
            "## 具体错误机制",
            "",
            "| 主模式 | span 数 | 审计含义 | 对应图建模假设 |",
            "|---|---:|---|---|",
        ]
    )
    for pattern, count in sorted(
        breakdowns["primary_pattern"].items(),
        key=lambda item: (-item[1], item[0]),
    ):
        meaning, graph_hypothesis = PATTERN_GUIDANCE.get(
            pattern, PATTERN_GUIDANCE["unclassified"]
        )
        lines.append(
            f"| `{pattern}` | {count} | {meaning} | {graph_hypothesis} |"
        )

    lines.extend(["", "## 数据质量检查", "", "| 检查项 | 数量 |", "|---|---:|"])
    quality = report.get("data_quality", {})
    if quality:
        for name, count in quality.items():
            lines.append(f"| `{name}` | {count} |")
    else:
        lines.append("| 未发现结构问题 | 0 |")

    lines.extend(["", "## 代表性负样本", ""])
    pattern_examples = report["examples"]["by_primary_pattern"]
    for pattern, examples in pattern_examples.items():
        lines.extend([f"### `{pattern}`", ""])
        for example in examples:
            span = _markdown_cell(example.get("span_text", ""))
            meta = _markdown_cell(example.get("meta", ""))
            task = _markdown_cell(example.get("task", ""))
            model = _markdown_cell(example.get("model", ""))
            lines.append(f"- `{task}` / `{model}` — **{span}**：{meta}")
        lines.append("")

    lines.extend(
        [
            "## 使用边界",
            "",
            "- 官方 `label_type` 用于评价和审计，不进入无监督训练。",
            "- `primary_pattern` 是互斥的便捷汇总；`pattern_tags` 保留一个 span 的多种机制。",
            "- 启发式规则只能用于发现数据模式和制定消融实验，不能替代人工事实核验。",
            "- 高回答级污染与低 span 覆盖率并存，因此优先做 token/claim 局部异常检测。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Iterable[str],
) -> None:
    fieldnames = tuple(fieldnames)
    with path.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {
                field: " | ".join(str(item) for item in value)
                if isinstance(value, list)
                else value
                for field, value in row.items()
            }
            writer.writerow(normalized)


def write_audit_outputs(
    result: AuditResult,
    output_directory: str | Path,
) -> dict[str, Path]:
    """Write machine-readable and human-readable audit artifacts."""

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": output_directory / "audit_report.json",
        "markdown": output_directory / "audit_report.md",
        "samples_csv": output_directory / "sample_audit.csv",
        "spans_csv": output_directory / "span_audit.csv",
    }
    paths["json"].write_text(
        json.dumps(result.report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    paths["markdown"].write_text(
        render_markdown_report(result.report),
        encoding="utf-8",
    )
    _write_csv(paths["samples_csv"], result.sample_rows, SAMPLE_FIELDS)
    _write_csv(paths["spans_csv"], result.span_rows, SPAN_FIELDS)
    return paths


def build_argument_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parent
    dataset_root = project_root / "Datasets" / "RAGTruth"
    parser = argparse.ArgumentParser(
        description=(
            "Audit clean and hallucinated RAGTruth responses, concrete error "
            "mechanisms, span integrity, and task/model distributions."
        )
    )
    parser.add_argument(
        "--responses",
        type=Path,
        default=dataset_root / "response.jsonl",
        help="Path to response.jsonl.",
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=dataset_root / "source_info.jsonl",
        help="Path to source_info.jsonl.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "audit_outputs" / "ragtruth_patterns",
        help="Directory for JSON, Markdown, and CSV outputs.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=5,
        help="Maximum representative examples stored for each pattern/task.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be non-negative")
    result = audit_dataset(
        args.responses,
        args.sources,
        max_examples=args.max_examples,
    )
    paths = write_audit_outputs(result, args.output_dir)
    summary = result.report["summary"]
    print(
        f"Audited {summary['responses']} responses and "
        f"{summary['annotated_spans']} annotated spans."
    )
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
