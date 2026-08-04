"""HaluEval data boundary for the attention-only grounding-flow method."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from attention_graph.data import PreparedGraphRecord, load_graph
from attention_graph.graph import AttentionGraph, GraphBuildConfig
from attention_graph.halueval import (
    discover_legacy_halueval_records,
    prepare_legacy_halueval_graphs,
    split_halueval_pairs,
)


@dataclass(frozen=True)
class FlowPreparedRecord:
    response_id: str
    pair_id: str
    partition: str
    graph_record: PreparedGraphRecord
    legacy_graph_path: Path
    segment_ids: torch.Tensor
    token_ids: torch.Tensor

    @property
    def response_tokens(self) -> int:
        return int(self.graph_record.num_response_nodes)


class FlowDataset(Sequence[tuple[AttentionGraph, torch.Tensor, FlowPreparedRecord]]):
    """Mmap prepared attention graphs while keeping segment ids structural."""

    def __init__(self, records: Sequence[FlowPreparedRecord]) -> None:
        self.records = tuple(records)
        self._validated: set[Path] = set()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self, index: int | slice
    ) -> tuple[AttentionGraph, torch.Tensor, FlowPreparedRecord] | list[tuple[AttentionGraph, torch.Tensor, FlowPreparedRecord]]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        record = self.records[index]
        path = record.graph_record.graph_path.resolve()
        graph = load_graph(
            path,
            device="cpu",
            mmap=True,
            validate=path not in self._validated,
        )
        if not torch.equal(graph.token_ids.cpu(), record.token_ids):
            raise ValueError("prepared graph token identity disagrees with legacy segments")
        response_positions = torch.nonzero(
            record.segment_ids == 3, as_tuple=False
        ).flatten()
        if (
            not len(response_positions)
            or int(response_positions[0]) != graph.response_idx
            or len(record.segment_ids) != graph.num_nodes
        ):
            raise ValueError("prepared graph response boundary disagrees with legacy segments")
        self._validated.add(path)
        return graph, record.segment_ids.clone(), record


def _torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - older supported torch
        return torch.load(path, map_location="cpu", weights_only=True)


def _value(record: Mapping[str, object] | object, name: str) -> object:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _label_keys(value: object, prefix: str = "") -> list[str]:
    if not isinstance(value, Mapping):
        return []
    found: list[str] = []
    for key, child in value.items():
        name = str(key).casefold()
        location = f"{prefix}.{key}" if prefix else str(key)
        if "label" in name or name in {"y", "y_token", "direction_score"}:
            found.append(location)
        found.extend(_label_keys(child, location))
    return found


def read_label_free_examples(path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    """Return response->pair and response->prompt-group without reading labels."""

    source = Path(path).expanduser().resolve()
    pairs: dict[str, str] = {}
    prompt_groups: dict[str, str] = {}
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read HaluEval examples: {source}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid examples JSONL row {line_number}") from error
        if not isinstance(row, Mapping):
            raise ValueError("HaluEval example rows must be JSON objects")
        forbidden = _label_keys(row)
        if forbidden:
            raise ValueError("grounding-flow preparation is label-blind")
        response_id = str(row.get("response_id", row.get("example_id", ""))).strip()
        pair_id = str(row.get("pair_id", "")).strip()
        if not response_id or not pair_id or response_id in pairs:
            raise ValueError("examples require unique response_id and non-empty pair_id")
        evidence = row.get("knowledge", row.get("passage", row.get("prompt", "")))
        question = row.get("question", "")
        if not str(evidence).strip() and not str(question).strip():
            raise ValueError("examples require knowledge/passage/prompt or question content")
        payload = json.dumps(
            [str(evidence), str(question)], ensure_ascii=False, separators=(",", ":")
        )
        pairs[response_id] = pair_id
        prompt_groups[response_id] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if not pairs:
        raise ValueError("HaluEval examples are empty")
    return pairs, prompt_groups


def _complete_pair_selection(
    records: Sequence[Mapping[str, object] | object], limit_pairs: int | None
) -> list[Mapping[str, object] | object]:
    grouped: dict[str, list[Mapping[str, object] | object]] = defaultdict(list)
    for record in records:
        grouped[str(_value(record, "pair_id"))].append(record)
    if any(len(candidates) != 2 for candidates in grouped.values()):
        raise ValueError("HaluEval requires exactly two candidates per pair")
    complete = [
        pair_id
        for pair_id, candidates in grouped.items()
        if all(
            (candidate.get("trace_path") if isinstance(candidate, Mapping) else getattr(candidate, "trace_path"))
            is not None
            for candidate in candidates
        )
    ]
    if limit_pairs is None and len(complete) != len(grouped):
        raise ValueError("complete graph and trace artifacts are required for every pair")
    selected_ids = sorted(complete)
    if limit_pairs is not None:
        if limit_pairs < 3:
            raise ValueError("limit_pairs must leave at least three complete pairs")
        selected_ids = selected_ids[:limit_pairs]
    if len(selected_ids) < 3:
        raise ValueError("grounding-flow splitting requires at least three complete pairs")
    return [record for pair_id in selected_ids for record in grouped[pair_id]]


def _structure_from_legacy(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    loaded = _torch_load(path)
    if not isinstance(loaded, Mapping):
        raise ValueError(f"legacy graph is not a mapping: {path}")
    # Deliberate allowlist: segment ids are structural source types.  No x,
    # hidden state, logit, entropy, label, or feature record is returned.
    segments = torch.as_tensor(loaded.get("segment_ids"), dtype=torch.long).flatten()
    token_ids = torch.as_tensor(loaded.get("token_ids"), dtype=torch.long).flatten()
    answer = torch.as_tensor(loaded.get("answer_mask", segments == 3), dtype=torch.bool).flatten()
    if (
        not len(segments)
        or token_ids.shape != segments.shape
        or answer.shape != segments.shape
        or not torch.equal(answer, segments == 3)
    ):
        raise ValueError("legacy segment_ids/answer_mask are inconsistent")
    first = (
        int(torch.nonzero(answer, as_tuple=False)[0].item())
        if bool(answer.any())
        else -1
    )
    if first <= 0 or not bool(answer[first:].all()) or bool(answer[:first].any()):
        raise ValueError("legacy response must be one non-empty contiguous suffix")
    return segments, token_ids


def prepare_halueval_flow_records(
    *,
    extraction_dir: str | Path,
    examples_path: str | Path,
    output_dir: str | Path,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 42,
    limit_pairs: int | None = None,
    expected_candidates: int | None = None,
    group_by_prompt: bool = True,
    require_complete_cache: bool = False,
    conversion_device: str | torch.device | None = None,
    conversion_chunk_edges: int = 8_192,
    query_block: int = 32,
    resume: bool = True,
    progress_callback: Callable[[str, int, int], None] | None = None,
) -> tuple[dict[str, list[FlowPreparedRecord]], dict[str, object]]:
    """Adapt pure-attention legacy data without entering either MAE pipeline."""

    example_pairs, prompt_groups = read_label_free_examples(examples_path)
    discovered = discover_legacy_halueval_records(
        extraction_dir,
        progress_callback=(
            None
            if progress_callback is None
            else lambda current, total: progress_callback("legacy_discovery", current, total)
        ),
    )
    discovered_ids = {str(_value(record, "response_id")) for record in discovered}
    if len(discovered_ids) != len(discovered):
        raise ValueError("legacy manifest contains duplicate response identities")
    if expected_candidates is not None:
        if expected_candidates < 1:
            raise ValueError("expected_candidates must be positive or None")
        if len(discovered_ids) != expected_candidates:
            raise ValueError(
                "expected candidate count does not match extraction manifest: "
                f"expected={expected_candidates} manifest={len(discovered_ids)} "
                f"examples_available={len(example_pairs)}"
            )
    if not discovered_ids.issubset(example_pairs):
        raise ValueError("legacy manifest contains responses absent from examples")
    if any(
        example_pairs[str(_value(record, "response_id"))]
        != str(_value(record, "pair_id"))
        for record in discovered
    ):
        raise ValueError("legacy manifest pair identity disagrees with examples")
    examples_cover_manifest = discovered_ids.issubset(example_pairs)
    manifest_covers_all_examples = discovered_ids == set(example_pairs)
    complete = all(str(_value(record, "artifact_status")) == "full" for record in discovered)
    if require_complete_cache and (not examples_cover_manifest or not complete):
        raise ValueError(
            "complete graph-and-trace artifacts and example coverage are required "
            "for every manifest candidate"
        )
    selected = _complete_pair_selection(discovered, limit_pairs)
    split_input: list[Mapping[str, object] | object]
    if group_by_prompt:
        split_input = [
            {
                **(dict(record) if isinstance(record, Mapping) else vars(record)),
                "group_id": prompt_groups[str(_value(record, "response_id"))],
            }
            for record in selected
        ]
    else:
        split_input = list(selected)
    partitions_any = split_halueval_pairs(
        split_input,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        group_by_prompt=group_by_prompt,
    )
    if any(not partitions_any[name] for name in ("train", "validation", "test")):
        raise ValueError("HaluEval split produced an empty partition")
    partition_by_response = {
        str(_value(record, "response_id")): partition
        for partition, records in partitions_any.items()
        for record in records
    }
    cache_assignment = {
        response_id: "test" if partition == "test" else "train"
        for response_id, partition in partition_by_response.items()
    }
    prepared = prepare_legacy_halueval_graphs(
        selected,
        output_dir=Path(output_dir).expanduser().resolve() / "prepared",
        config=GraphBuildConfig(
            selection="threshold",
            threshold=None,
            max_edges_per_target=None,
            query_block=query_block,
        ),
        dataset_split_by_response=cache_assignment,
        conversion_device=conversion_device,
        conversion_chunk_edges=conversion_chunk_edges,
        build_device=("cpu" if conversion_device is None else conversion_device),
        resume=resume,
        progress_callback=progress_callback,
    )
    prepared_by_response = {record.response_id: record for record in prepared}
    legacy_by_response = {
        str(_value(record, "response_id")): Path(_value(record, "graph_path")).resolve()
        for record in selected
    }
    structure_by_response = {
        response_id: _structure_from_legacy(path)
        for response_id, path in legacy_by_response.items()
    }
    result: dict[str, list[FlowPreparedRecord]] = {
        "train": [], "validation": [], "test": []
    }
    for partition, records in partitions_any.items():
        for record in records:
            response_id = str(_value(record, "response_id"))
            result[partition].append(
                FlowPreparedRecord(
                    response_id=response_id,
                    pair_id=str(_value(record, "pair_id")),
                    partition=partition,
                    graph_record=prepared_by_response[response_id],
                    legacy_graph_path=legacy_by_response[response_id],
                    segment_ids=structure_by_response[response_id][0],
                    token_ids=structure_by_response[response_id][1],
                )
            )
    metadata: dict[str, object] = {
        "schema": "halueval-grounding-flow-prepared-v1",
        "scope": (
            "legacy_cache_complete"
            if complete and examples_cover_manifest and limit_pairs is None
            else "legacy_cache_partial_pilot"
        ),
        "counts": {name: len(records) for name, records in result.items()},
        "inventory": {
            "expected_candidates": expected_candidates,
            "example_candidates": len(example_pairs),
            "manifest_candidates": len(discovered_ids),
            "selected_candidates": len(selected),
            "examples_cover_manifest": examples_cover_manifest,
            "manifest_covers_all_examples": manifest_covers_all_examples,
            "all_artifacts_full": complete,
        },
        "pair_ids": {
            name: sorted({record.pair_id for record in records})
            for name, records in result.items()
        },
        "split": {
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "group_by_prompt": group_by_prompt,
        },
        "input_protocol": {
            "mode": "legacy_tau_censored_attention_only",
            "legacy_tau_values": sorted(
                {
                    float(_value(record, "legacy_tau"))
                    for record in selected
                    if _value(record, "legacy_tau") is not None
                }
            ),
            "selection": "all_retained_threshold_support",
            "max_edges_per_target": None,
            "missing_attention_policy": "explicit_unknown_mass",
            "segment_role": "structural_source_type_only",
        },
    }
    return result, metadata


__all__ = [
    "FlowDataset",
    "FlowPreparedRecord",
    "prepare_halueval_flow_records",
    "read_label_free_examples",
]
