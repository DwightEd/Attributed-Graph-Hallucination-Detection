"""Label-free adapter for persisted HaluEval ``token_graph_v2`` artifacts.

The old artifacts retain ordered attention channels but also contain features
that are not valid inputs to the attention-only graph model.  This module
explicitly reconstructs the formal sparse cache from the retained attention
fields, and reads response labels only in its evaluation helpers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from . import data
from .data import PreparedGraphRecord
from .evaluate import evaluate_binary_scores
from .graph import FORMAL_CACHE_SCHEMA, GraphBuildConfig, build_attention_graph


_LABEL_MARKERS = ("label",)


def _reject_label_keys(*records: Mapping[str, object]) -> None:
    """Keep labels out of graph conversion, including legacy side fields."""

    forbidden: list[str] = []
    def visit(value: object, prefix: str = "") -> None:
        if not isinstance(value, Mapping):
            return
        for key, child in value.items():
            name = str(key).casefold()
            location = f"{prefix}.{key}" if prefix else str(key)
            if name in {"y", "y_token", "hallucination_labels"} or any(
                marker in name for marker in _LABEL_MARKERS
            ):
                forbidden.append(location)
            visit(child, location)

    for record in records:
        visit(record)
    if forbidden:
        raise ValueError("legacy graph conversion is label-blind: " + ", ".join(sorted(forbidden)))


def _as_vector(value: object, *, name: str, dtype: torch.dtype) -> torch.Tensor:
    vector = torch.as_tensor(value, dtype=dtype).flatten()
    if vector.ndim != 1 or not vector.numel():
        raise ValueError(f"{name} must be a non-empty vector")
    return vector


def _identity(record: Mapping[str, object], name: str) -> str:
    value = str(record.get(name, "")).strip()
    if not value:
        raise ValueError(f"legacy artifact is missing {name}")
    return value


def _same_identity(legacy: Mapping[str, object], trace: Mapping[str, object], name: str) -> str:
    left, right = _identity(legacy, name), _identity(trace, name)
    if left != right:
        raise ValueError(f"legacy graph/trace {name} identity mismatch")
    return left


def _legacy_floor(legacy: Mapping[str, object], trace: Mapping[str, object]) -> float:
    graph_config = legacy.get("graph_config", {})
    if not isinstance(graph_config, Mapping):
        raise ValueError("legacy graph_config must be a mapping")
    raw_floor = graph_config.get("tau", trace.get("edge_threshold"))
    try:
        floor = float(raw_floor)
    except (TypeError, ValueError) as error:
        raise ValueError("legacy tau/edge_threshold is required") from error
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("legacy tau must be finite in (0, 1]")
    if "edge_threshold" in trace:
        try:
            trace_floor = float(trace["edge_threshold"])
        except (TypeError, ValueError) as error:
            raise ValueError("trace edge_threshold must be numeric") from error
        if not math.isfinite(trace_floor) or trace_floor != floor:
            raise ValueError("legacy tau and trace edge_threshold disagree")
    return floor


def legacy_graph_to_formal_attention_cache(
    legacy_graph: Mapping[str, object],
    trace_metadata: Mapping[str, object],
    *,
    dataset_split: str = "train",
    storage_dtype: torch.dtype = torch.float16,
    conversion_device: str | torch.device | None = None,
    conversion_chunk_edges: int = 8_192,
) -> dict[str, object]:
    """Adapt one legacy graph to the strict, label-free formal cache schema.

    Dense attention is deliberately not required.  The retained ordered edge
    channels are losslessly re-expressed as response-attention CSR entries
    accepted by :mod:`attention_graph.data`.
    """

    if not isinstance(legacy_graph, Mapping) or not isinstance(trace_metadata, Mapping):
        raise ValueError("legacy graph and trace metadata must be mappings")
    _reject_label_keys(legacy_graph, trace_metadata)
    if legacy_graph.get("schema_version") != "token_graph_v2":
        raise ValueError("only token_graph_v2 legacy graphs are supported")

    response_id = _same_identity(legacy_graph, trace_metadata, "example_id")
    pair_id = _same_identity(legacy_graph, trace_metadata, "pair_id")
    fingerprint = _same_identity(legacy_graph, trace_metadata, "extraction_fingerprint")
    token_ids = _as_vector(legacy_graph.get("token_ids"), name="token_ids", dtype=torch.long)
    trace_ids = _as_vector(trace_metadata.get("input_ids"), name="trace input_ids", dtype=torch.long)
    segment_ids = _as_vector(legacy_graph.get("segment_ids"), name="segment_ids", dtype=torch.long)
    trace_segments = _as_vector(trace_metadata.get("segment_ids"), name="trace segment_ids", dtype=torch.long)
    if not torch.equal(token_ids, trace_ids) or not torch.equal(segment_ids, trace_segments):
        raise ValueError("legacy graph and trace token identity mismatch")
    token_count = int(token_ids.numel())
    if segment_ids.numel() != token_count:
        raise ValueError("segment_ids must align with token_ids")

    answer_raw = legacy_graph.get("answer_mask", segment_ids == 3)
    answer_mask = torch.as_tensor(answer_raw, dtype=torch.bool).flatten()
    if answer_mask.numel() != token_count or not bool(answer_mask.any()):
        raise ValueError("answer_mask must mark a non-empty answer suffix")
    response_idx = int(torch.nonzero(answer_mask, as_tuple=False)[0].item())
    expected_mask = torch.arange(token_count) >= response_idx
    if not torch.equal(answer_mask, expected_mask):
        raise ValueError("answer tokens must be a contiguous suffix")
    if not torch.equal(answer_mask, segment_ids == 3):
        raise ValueError("answer_mask and segment_ids disagree about the answer suffix")
    if response_idx == 0:
        raise ValueError("answer suffix must leave a non-empty prompt")

    shape = tuple(int(item) for item in trace_metadata.get("attention_shape", ()))
    if len(shape) != 4 or shape[2:] != (token_count, token_count) or min(shape[:2]) < 1:
        raise ValueError("trace attention_shape must be [layers, heads, tokens, tokens]")
    layers, heads = shape[:2]
    channels = layers * heads
    slices = legacy_graph.get("x_view_slices")
    if not isinstance(slices, Mapping) or "attention_diagonal" not in slices:
        raise ValueError("legacy x_view_slices must provide attention_diagonal")
    try:
        diagonal_start, diagonal_end = (int(item) for item in slices["attention_diagonal"])
    except (TypeError, ValueError) as error:
        raise ValueError("attention_diagonal slice must be a pair") from error
    if (diagonal_start, diagonal_end) != (0, channels):
        raise ValueError("attention_diagonal must be the ordered leading attention channels")
    x = torch.as_tensor(legacy_graph.get("x"), dtype=torch.float32)
    if x.ndim != 2 or x.shape[0] != token_count or x.shape[1] < channels:
        raise ValueError("legacy x does not contain ordered attention diagonal channels")
    diagonal = x[:, :channels]
    if not bool(torch.isfinite(diagonal).all()) or bool(((diagonal < 0) | (diagonal > 1)).any()):
        raise ValueError("legacy attention diagonal must be finite in [0, 1]")

    edge_index = torch.as_tensor(legacy_graph.get("edge_index"), dtype=torch.long)
    # Keep mmap-backed edge channels on CPU.  Converting the whole tensor to
    # float32 would eagerly materialize an E x C copy before chunking starts.
    edge_attr = torch.as_tensor(legacy_graph.get("edge_attr"))
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("legacy edge_index must have shape [2, edges]")
    if edge_attr.ndim != 2 or edge_attr.shape != (edge_index.shape[1], channels):
        raise ValueError("legacy edge_attr must have one ordered channel vector per edge")
    if not edge_attr.is_floating_point():
        raise ValueError("legacy edge attention must be floating point")
    floor = _legacy_floor(legacy_graph, trace_metadata)
    source, target = edge_index

    # The formal cache is CSR ordered by flattened (layer, head) then response
    # query.  Only old attention fields are copied; hidden/logprob/entropy stay
    # outside this allowlisted reconstruction.
    if storage_dtype not in {torch.float16, torch.float32}:
        raise ValueError("legacy formal cache storage dtype must be float16 or float32")
    if not isinstance(conversion_chunk_edges, int) or conversion_chunk_edges < 1:
        raise ValueError("legacy conversion_chunk_edges must be a positive integer")
    if conversion_device is None:
        conversion_device = "cuda" if torch.cuda.is_available() else "cpu"
    work_device = torch.device(conversion_device)
    if work_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("legacy conversion device requests CUDA, but CUDA is unavailable")

    response_tokens = token_count - response_idx
    row_total = channels * response_tokens
    edge_count = int(edge_index.shape[1])
    # A legacy token pair represents one ordered vector of channels.  Check
    # this before channel censoring: disjoint nonzero channels would otherwise
    # evade the later CSR-entry duplicate test.
    pair_keys = target.to(work_device) * token_count + source.to(work_device)
    if torch.unique(pair_keys).numel() != edge_count:
        raise ValueError("legacy graph has duplicate attention edges")
    del pair_keys
    # Retain only compact COO payload per edge chunk on the conversion device.
    # A single final device-side sort creates CSR order without synchronizing
    # every active row into Python or keeping large chunk payloads in CPU RAM.
    row_parts: list[torch.Tensor] = []
    column_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for start in range(0, edge_count, conversion_chunk_edges):
        stop = min(start + conversion_chunk_edges, edge_count)
        source_cpu, target_cpu = source[start:stop], target[start:stop]
        if (
            bool((source_cpu < 0).any())
            or bool((target_cpu < 0).any())
            or bool((target_cpu >= token_count).any())
            or bool((source_cpu >= target_cpu).any())
        ):
            raise ValueError("legacy attention edges violate causal token ordering")
        # The source artifacts are CPU mmap tensors.  Only the active edge
        # chunk crosses to GPU, so an idle GPU accelerates conversion without
        # making CPU RAM scale with channels.
        attr_chunk = edge_attr[start:stop].to(work_device, dtype=torch.float32)
        if not bool(torch.isfinite(attr_chunk).all()) or bool(
            ((attr_chunk < 0) | (attr_chunk > 1)).any()
        ):
            raise ValueError("legacy edge attention must be finite in [0, 1]")
        if bool(((attr_chunk != 0) & (attr_chunk <= floor)).any()):
            raise ValueError("retained legacy attention must be above the tau censoring floor")

        edge_ids, channel_ids = torch.nonzero(attr_chunk, as_tuple=True)
        if not edge_ids.numel():
            continue
        source_chunk = source_cpu.to(work_device)
        target_chunk = target_cpu.to(work_device)
        response_entries = target_chunk[edge_ids] >= response_idx
        edge_ids, channel_ids = edge_ids[response_entries], channel_ids[response_entries]
        if not edge_ids.numel():
            continue
        rows = channel_ids * response_tokens + (target_chunk[edge_ids] - response_idx)
        columns = source_chunk[edge_ids]
        row_parts.append(rows)
        column_parts.append(columns.to(dtype=torch.int32))
        value_parts.append(attr_chunk[edge_ids, channel_ids].to(storage_dtype))

    if row_parts:
        rows = torch.cat(row_parts)
        columns = torch.cat(column_parts)
        values = torch.cat(value_parts)
        del row_parts, column_parts, value_parts

        row_counts = torch.bincount(rows, minlength=row_total)
        sort_key = rows * token_count + columns
        del rows
        order = torch.argsort(sort_key, stable=True)
        sorted_key = sort_key[order]
        if sorted_key.numel() > 1 and bool((sorted_key[1:] == sorted_key[:-1]).any()):
            raise ValueError("legacy graph has duplicate attention edges")
        del sort_key, sorted_key
        columns, values = columns[order], values[order]
        del order
        row_ptr = torch.cat(
            (torch.zeros(1, dtype=torch.int64, device=work_device), torch.cumsum(row_counts, dim=0))
        )
        columns, values, row_ptr = columns.cpu(), values.cpu(), row_ptr.cpu()
    else:
        row_ptr = torch.zeros(row_total + 1, dtype=torch.int64)
        columns = torch.empty(0, dtype=torch.int32)
        values = torch.empty(0, dtype=storage_dtype)
    sample = {
        "attention_cache_schema": FORMAL_CACHE_SCHEMA,
        "attention_cache_fingerprint": fingerprint,
        "cache_dtype": "float16" if storage_dtype == torch.float16 else "float32",
        "input_policy": "legacy_token_graph_v2_attention_only",
        "was_truncated": False,
        "split": str(dataset_split).strip().casefold(),
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "attention_diagonal": diagonal.T.reshape(layers, heads, token_count).to(
            device="cpu", dtype=storage_dtype
        ),
        "response_idx": response_idx,
        "response_row_ptr": row_ptr.to(torch.int64),
        "response_column_indices": columns.to(torch.int32),
        "response_values": values.to(storage_dtype),
        "attention_floor": floor,
        "token_ids": token_ids.cpu(),
        "source_id": pair_id,
        "response_id": response_id,
        "sample_id": response_id,
    }
    if sample["split"] not in {"train", "test"}:
        raise ValueError("legacy formal cache split must be train or test")
    return sample


def legacy_graph_to_attention_graph(
    legacy_graph: Mapping[str, object],
    trace_metadata: Mapping[str, object],
    config: GraphBuildConfig | None = None,
    *,
    device: str | torch.device | None = None,
):
    """Convert a label-free old token graph into an ``AttentionGraph``."""

    formal = legacy_graph_to_formal_attention_cache(
        legacy_graph,
        trace_metadata,
        storage_dtype=torch.float32,
        conversion_device=device,
    )
    build_config = GraphBuildConfig() if config is None else config
    if build_config.threshold is not None and build_config.threshold < float(
        formal["attention_floor"]
    ):
        raise ValueError("requested threshold cannot be lower than legacy tau floor")
    return build_attention_graph(formal, build_config, device=device)


def _torch_load(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # pragma: no cover - compatibility with older torch
        return torch.load(path, map_location="cpu", weights_only=True)


def discover_legacy_halueval_records(root: str | Path) -> list[dict[str, object]]:
    """Discover exactly the graph files named by an extraction manifest."""

    root_path = Path(root).expanduser().resolve()
    manifest_path = root_path / "extraction_manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read extraction manifest: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("extraction manifest must be an object")
    graph_files = manifest.get("graph_files")
    example_ids = manifest.get("example_ids")
    if not isinstance(graph_files, list) or not isinstance(example_ids, list):
        raise ValueError("extraction manifest must list graph_files and example_ids")
    if any(not isinstance(value, str) for value in graph_files + example_ids):
        raise ValueError("manifest graph_files and example_ids must be string lists")
    if len(graph_files) != len(example_ids):
        raise ValueError("manifest graph_files and example_ids must have the same length")
    state = manifest.get("state", "partial")
    if not isinstance(state, str) or not state.strip():
        raise ValueError("manifest state must be a non-empty string")
    state = state.strip().casefold()

    def artifact_dir(field: str, default_name: str) -> Path:
        preferred = root_path / default_name
        if preferred.is_dir():
            return preferred
        raw = Path(str(manifest.get(field, preferred))).expanduser()
        return raw if raw.is_absolute() else root_path / raw

    def artifact_path(directory: Path, name: object) -> Path:
        relative = Path(str(name))
        if not str(name).strip() or relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest graph_files must be non-empty relative paths")
        # A manifest may name files as ``graphs/a.pt`` while root/graphs is
        # already the selected directory; do not form graphs/graphs/a.pt.
        root_relative = root_path / relative
        if relative.parts and relative.parts[0] == directory.name and root_relative.is_file():
            candidate = root_relative
        elif relative.parts and relative.parts[0] in {"graphs", "traces"}:
            candidate = directory.joinpath(*relative.parts[1:])
        else:
            candidate = directory / relative
        return candidate

    graph_dir = artifact_dir("graph_dir", "graphs")
    trace_dir = artifact_dir("trace_dir", "traces")
    normalized_ids: list[str] = []
    seen_files: set[str] = set()
    for graph_file, example_id in zip(graph_files, example_ids):
        relative = Path(str(graph_file))
        file_key = relative.as_posix()
        identifier = str(example_id).strip()
        if file_key in seen_files or not identifier or identifier in normalized_ids:
            raise ValueError("manifest graph_files and example_ids must be unique")
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("manifest graph_files must be relative paths")
        seen_files.add(file_key)
        normalized_ids.append(identifier)
    records: list[dict[str, object]] = []
    seen_graph_paths: set[Path] = set()
    for name, expected_response_id in zip(graph_files, normalized_ids):
        graph_path = artifact_path(graph_dir, name)
        if not graph_path.is_file():
            raise ValueError(f"manifest graph file is missing: {graph_path}")
        resolved_graph_path = graph_path.resolve()
        if resolved_graph_path in seen_graph_paths:
            raise ValueError("manifest graph_files must be unique")
        seen_graph_paths.add(resolved_graph_path)
        graph = _torch_load(graph_path)
        if not isinstance(graph, Mapping):
            raise ValueError(f"legacy graph is not a mapping: {graph_path}")
        response_id = _identity(graph, "example_id")
        pair_id = _identity(graph, "pair_id")
        graph_config = graph.get("graph_config", {})
        legacy_tau = graph_config.get("tau") if isinstance(graph_config, Mapping) else None
        if response_id != expected_response_id:
            raise ValueError("manifest example_ids and graph example_id identity mismatch")
        trace_path = artifact_path(trace_dir, name)
        trace_exists = trace_path.is_file()
        records.append(
            {
                "response_id": response_id,
                "pair_id": pair_id,
                "graph_path": graph_path,
                "trace_path": trace_path if trace_exists else None,
                "artifact_status": "full" if state == "complete" and trace_exists else "partial",
                "manifest_state": state,
                "legacy_tau": legacy_tau,
                "extractor_model_id": graph.get("extractor_model_id"),
                "extraction_fingerprint": graph.get("extraction_fingerprint"),
            }
        )
    return records


def _record_value(record: Mapping[str, object] | object, name: str) -> object:
    if isinstance(record, Mapping):
        return record[name]
    return getattr(record, name)


def split_halueval_pairs(
    records: Sequence[Mapping[str, object] | object],
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 0,
    group_by_prompt: bool = False,
) -> dict[str, list[Mapping[str, object] | object]]:
    """Split whole two-candidate pairs with a stable, process-independent hash.

    The default deliberately reproduces the prior pair-id cohort.  Opt-in
    prompt grouping assigns every pair with the same prepared source prompt to
    one partition, so duplicated knowledge/question content cannot leak.
    """

    if not records:
        raise ValueError("HaluEval records are empty")
    if not (0.0 <= validation_fraction < 1.0 and 0.0 <= test_fraction < 1.0):
        raise ValueError("split fractions must be in [0, 1)")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation and test fractions must leave training pairs")
    by_pair: dict[str, list[Mapping[str, object] | object]] = defaultdict(list)
    seen_response: set[str] = set()
    for record in records:
        response_id = str(_record_value(record, "response_id")).strip()
        pair_id = str(_record_value(record, "pair_id")).strip()
        if not response_id or not pair_id or response_id in seen_response:
            raise ValueError("records require unique response_id and pair_id")
        seen_response.add(response_id)
        by_pair[pair_id].append(record)
    if any(len(candidates) != 2 for candidates in by_pair.values()):
        raise ValueError("each HaluEval pair must contain exactly two complete candidates")
    if len(by_pair) < 3:
        raise ValueError("HaluEval splitting requires at least three complete pairs")
    pair_ids = sorted(
        by_pair,
        key=lambda pair_id: hashlib.sha256(f"{seed}\x1f{pair_id}".encode("utf-8")).hexdigest(),
    )
    if group_by_prompt:
        by_group: dict[str, list[str]] = defaultdict(list)
        for pair_id, candidates in by_pair.items():
            values = {str(_record_value(candidate, "group_id")).strip() for candidate in candidates}
            if len(values) != 1 or not next(iter(values)):
                raise ValueError("prompt-group split requires one non-empty group_id per pair")
            by_group[next(iter(values))].append(pair_id)
        if len(by_group) < 3:
            raise ValueError("prompt-group split requires at least three distinct prompt groups")
        ordered_groups = sorted(
            by_group,
            key=lambda group_id: hashlib.sha256(f"{seed}\x1f{group_id}".encode("utf-8")).hexdigest(),
        )
        # Select entire prompt groups in stable-hash order.  This preserves
        # the requested held-out proportions as closely as possible without
        # allowing a duplicate prompt to cross a partition.
        total_pairs = len(pair_ids)
        test_target = 0 if test_fraction == 0.0 else max(1, round(total_pairs * test_fraction))
        validation_target = 0 if validation_fraction == 0.0 else max(1, round(total_pairs * validation_fraction))
        test_groups: list[str] = []
        validation_groups: list[str] = []
        test_pairs = validation_pairs = 0
        for index, group_id in enumerate(ordered_groups):
            group_size = len(by_group[group_id])
            remaining_groups = len(ordered_groups) - index - 1
            if test_pairs < test_target and remaining_groups >= 2:
                test_groups.append(group_id)
                test_pairs += group_size
            elif validation_pairs < validation_target and remaining_groups >= 1:
                validation_groups.append(group_id)
                validation_pairs += group_size
        test_ids = {pair_id for group_id in test_groups for pair_id in by_group[group_id]}
        validation_ids = {
            pair_id for group_id in validation_groups for pair_id in by_group[group_id]
        }
        if (
            (test_fraction > 0.0 and not test_ids)
            or (validation_fraction > 0.0 and not validation_ids)
            or len(test_ids | validation_ids) == total_pairs
        ):
            raise ValueError("prompt-group split cannot produce non-empty train, validation, and test partitions")
        result: dict[str, list[Mapping[str, object] | object]] = {"train": [], "validation": [], "test": []}
        for pair_id in pair_ids:
            bucket = "test" if pair_id in test_ids else "validation" if pair_id in validation_ids else "train"
            result[bucket].extend(sorted(by_pair[pair_id], key=lambda item: str(_record_value(item, "response_id"))))
        return result
    total = len(pair_ids)
    test_count = 0 if test_fraction == 0.0 else max(1, round(total * test_fraction))
    validation_count = (
        0
        if validation_fraction == 0.0
        else min(total - test_count - 1, max(1, round(total * validation_fraction)))
    )
    test_count = min(test_count, total - validation_count - 1)
    test_ids = set(pair_ids[:test_count])
    validation_ids = set(pair_ids[test_count : test_count + validation_count])
    result: dict[str, list[Mapping[str, object] | object]] = {"train": [], "validation": [], "test": []}
    for pair_id in pair_ids:
        bucket = "test" if pair_id in test_ids else "validation" if pair_id in validation_ids else "train"
        result[bucket].extend(sorted(by_pair[pair_id], key=lambda item: str(_record_value(item, "response_id"))))
    return result


def load_halueval_response_labels(path: str | Path) -> dict[str, int]:
    """Load the explicit held-out response-label sidecar, never graph files."""

    labels: dict[str, int] = {}
    source = Path(path).expanduser()
    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read response-label sidecar: {source}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSONL label row {line_number}") from error
        if not isinstance(row, Mapping):
            raise ValueError("response-label rows must be objects")
        identifiers = [str(row[key]).strip() for key in ("response_id", "example_id") if key in row]
        if len(identifiers) != 1 or not identifiers[0]:
            raise ValueError("each label row requires exactly one response_id or example_id")
        if "label" not in row or isinstance(row["label"], bool) or row["label"] not in (0, 1):
            raise ValueError("response labels must be unique finite binary values")
        response_id = identifiers[0]
        if response_id in labels:
            raise ValueError(f"response labels must be unique: {response_id}")
        labels[response_id] = int(row["label"])
    if not labels:
        raise ValueError("response-label sidecar is empty")
    return labels


def _prediction_scores(predictions: Sequence[Mapping[str, object]]) -> dict[str, float]:
    if not predictions:
        raise ValueError("prediction records are empty")
    scores: dict[str, float] = {}
    for record in predictions:
        if not isinstance(record, Mapping):
            raise ValueError("prediction records must be mappings")
        identifier = str(record.get("response_id", record.get("example_id", ""))).strip()
        if not identifier or identifier in scores:
            raise ValueError("prediction records require unique response_id")
        has_score = "score" in record
        has_probability = "hallucination_probability" in record
        if not has_score and not has_probability:
            raise ValueError("prediction records require score or hallucination_probability")
        def numeric(value: object) -> float:
            try:
                result = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("prediction score must be finite") from error
            if not math.isfinite(result):
                raise ValueError("prediction score must be finite")
            return result

        score = numeric(record["score"]) if has_score else numeric(record["hallucination_probability"])
        if has_score and has_probability and score != numeric(record["hallucination_probability"]):
            raise ValueError("prediction score and hallucination_probability disagree")
        scores[identifier] = score
    return scores


def _paired_summary(
    labels: Mapping[str, int], scores: Mapping[str, float], pairs: Mapping[str, str]
) -> tuple[float, float, list[float]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for response_id, pair_id in pairs.items():
        grouped[str(pair_id)].append(response_id)
    pair_accuracy: list[float] = []
    margins: list[float] = []
    for pair_id, response_ids in grouped.items():
        if len(response_ids) != 2:
            raise ValueError(f"pair {pair_id} must contain exactly two responses")
        positives = [response_id for response_id in response_ids if labels[response_id] == 1]
        negatives = [response_id for response_id in response_ids if labels[response_id] == 0]
        if len(positives) != 1 or len(negatives) != 1:
            raise ValueError(f"pair {pair_id} must contain one binary label of each class")
        margin = scores[positives[0]] - scores[negatives[0]]
        margins.append(margin)
        pair_accuracy.append(1.0 if margin > 0.0 else 0.5 if margin == 0.0 else 0.0)
    return float(np.mean(pair_accuracy)), float(np.mean(margins)), pair_accuracy


def evaluate_halueval_predictions(
    predictions: Sequence[Mapping[str, object]],
    labels: Mapping[str, object],
    pair_by_response: Mapping[str, object],
    *,
    response_length_by_id: Mapping[str, object] | None = None,
    bootstrap_samples: int = 1000,
    seed: int = 0,
) -> dict[str, object]:
    """Exact-join held-out HaluEval response scores and report paired ranking."""

    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    scores = _prediction_scores(predictions)
    normalized_labels: dict[str, int] = {}
    for response_id, value in labels.items():
        key = str(response_id)
        if key in normalized_labels or isinstance(value, bool) or value not in (0, 1):
            raise ValueError("labels must be unique binary values")
        normalized_labels[key] = int(value)
    if not isinstance(pair_by_response, Mapping):
        raise ValueError("pair_by_response must be a mapping")
    normalized_pairs: dict[str, str] = {}
    for response_id, pair_id in pair_by_response.items():
        response_key = str(response_id).strip()
        pair_key = str(pair_id).strip()
        if not response_key or response_key in normalized_pairs:
            raise ValueError("pair_by_response response_id keys must be unique after normalization")
        if not pair_key:
            raise ValueError("pair_by_response pair_id must be non-empty")
        normalized_pairs[response_key] = pair_key
    if set(scores) != set(normalized_labels) or set(scores) != set(normalized_pairs):
        raise ValueError("prediction/label/pair response_id join has missing or extra records")
    response_ids = sorted(scores)
    metrics: dict[str, object] = evaluate_binary_scores(
        [normalized_labels[response_id] for response_id in response_ids],
        [scores[response_id] for response_id in response_ids],
    )
    paired_accuracy, paired_margin, pair_outcomes = _paired_summary(normalized_labels, scores, normalized_pairs)
    generator = np.random.default_rng(seed)
    outcome_array = np.asarray(pair_outcomes, dtype=np.float64)
    bootstrap = np.asarray(
        [float(generator.choice(outcome_array, size=len(outcome_array), replace=True).mean()) for _ in range(bootstrap_samples)]
    )
    low, high = np.quantile(bootstrap, (0.025, 0.975)).tolist()
    metrics.update(
        {
            "paired_accuracy": paired_accuracy,
            "paired_margin": paired_margin,
            "margin": paired_margin,
            "paired_bootstrap_ci": {
                "low": float(low),
                "high": float(high),
                "samples": int(bootstrap_samples),
                "seed": int(seed),
            },
        }
    )
    if response_length_by_id is not None:
        lengths = {str(response_id): value for response_id, value in response_length_by_id.items()}
        if set(lengths) != set(scores):
            raise ValueError("response length response_id join has missing or extra records")
        length_scores: dict[str, float] = {}
        for response_id, value in lengths.items():
            try:
                length = float(value)
            except (TypeError, ValueError) as error:
                raise ValueError("response lengths must be finite") from error
            if not math.isfinite(length) or length < 0:
                raise ValueError("response lengths must be finite non-negative values")
            length_scores[response_id] = length
        baseline: dict[str, object] = evaluate_binary_scores(
            [normalized_labels[response_id] for response_id in response_ids],
            [length_scores[response_id] for response_id in response_ids],
        )
        length_accuracy, length_margin, _ = _paired_summary(normalized_labels, length_scores, normalized_pairs)
        baseline.update({"paired_accuracy": length_accuracy, "paired_margin": length_margin, "margin": length_margin})
        metrics["length_only"] = baseline
        metrics["length_only_diagnostic"] = baseline
    return metrics


def _atomic_torch_save(path: Path, value: Mapping[str, object]) -> None:
    """Write an adapted cache atomically so resume never sees a partial file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _legacy_cache_name(record: Mapping[str, object] | object) -> str:
    response_id = str(_record_value(record, "response_id")).strip()
    if not response_id:
        raise ValueError("legacy cache requires a non-empty response_id")
    return "attention_" + hashlib.sha256(response_id.encode("utf-8")).hexdigest() + ".pt"


def _reusable_formal_cache(cache_path: Path, expected: Mapping[str, object]) -> bool:
    """Accept resume data only when its immutable response identity still agrees."""

    try:
        cached = _torch_load(cache_path)
    except (OSError, RuntimeError, TypeError, ValueError, KeyError):
        return False
    if not isinstance(cached, Mapping):
        return False
    try:
        return (
            str(cached["response_id"]) == str(expected["response_id"])
            and str(cached["source_id"]) == str(expected["source_id"])
            and str(cached["split"]) == str(expected["split"])
            and str(cached["attention_cache_fingerprint"])
            == str(expected["attention_cache_fingerprint"])
            and float(cached["attention_floor"]) == float(expected["attention_floor"])
        )
    except (KeyError, TypeError, ValueError):
        return False


def _formal_cache_identity_from_legacy(
    legacy_graph: Mapping[str, object],
    trace_metadata: Mapping[str, object],
    *,
    record: Mapping[str, object] | object,
    dataset_split: str,
) -> dict[str, object]:
    """Derive the cache identity without materializing sparse attention tensors."""

    response_id = _same_identity(legacy_graph, trace_metadata, "example_id")
    source_id = _same_identity(legacy_graph, trace_metadata, "pair_id")
    fingerprint = _same_identity(legacy_graph, trace_metadata, "extraction_fingerprint")
    if response_id != str(_record_value(record, "response_id")).strip():
        raise ValueError("legacy record and graph response_id identity mismatch")
    if source_id != str(_record_value(record, "pair_id")).strip():
        raise ValueError("legacy record and graph pair_id identity mismatch")
    return {
        "response_id": response_id,
        "source_id": source_id,
        "split": dataset_split,
        "attention_cache_fingerprint": fingerprint,
        "attention_floor": _legacy_floor(legacy_graph, trace_metadata),
    }


def _remove_stale_response_caches(cache_dir: Path, *, response_id: str, keep: Path) -> None:
    """Migrate pre-identity cache names without retaining stale duplicate responses."""

    if not cache_dir.is_dir():
        return
    for candidate in cache_dir.glob("attention_*.pt"):
        if candidate == keep:
            continue
        try:
            cached = _torch_load(candidate)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            continue
        if isinstance(cached, Mapping) and str(cached.get("response_id", "")) == response_id:
            candidate.unlink()


def prepare_legacy_halueval_graphs(
    records: Sequence[Mapping[str, object] | object],
    *,
    output_dir: str | Path,
    config: GraphBuildConfig,
    dataset_split_by_response: Mapping[str, str] | None = None,
    conversion_device: str | torch.device | None = None,
    conversion_chunk_edges: int = 8_192,
    build_device: str | torch.device = "cpu",
    resume: bool = True,
    mmap: bool = True,
) -> list[PreparedGraphRecord]:
    """Convert complete legacy pairs and prepare standard mmap graph artifacts.

    Train and validation candidates deliberately share the formal ``train``
    cache partition; only the held-out pair partition is written under
    ``test``.  Labels are neither accepted nor read by this conversion path.
    """

    if not records:
        raise ValueError("legacy HaluEval records are empty")
    response_ids = [str(_record_value(record, "response_id")).strip() for record in records]
    if not all(response_ids) or len(set(response_ids)) != len(response_ids):
        raise ValueError("legacy HaluEval records require unique response_id values")
    assignments = (
        {response_id: "train" for response_id in response_ids}
        if dataset_split_by_response is None
        else {str(key).strip(): str(value).strip().casefold() for key, value in dataset_split_by_response.items()}
    )
    if set(assignments) != set(response_ids) or any(
        value not in {"train", "test"} for value in assignments.values()
    ):
        raise ValueError("legacy response split assignment must exactly map every response to train or test")

    destination = Path(output_dir).expanduser().resolve()
    cache_root = destination / "adapted_cache"
    for record, response_id in zip(records, response_ids):
        trace_value = _record_value(record, "trace_path")
        if trace_value is None:
            raise ValueError("legacy graph preparation requires a complete graph and trace for every response")
        graph_path = Path(_record_value(record, "graph_path")).expanduser().resolve()
        trace_path = Path(trace_value).expanduser().resolve()
        legacy_graph, trace = _torch_load(graph_path), _torch_load(trace_path)
        if not isinstance(legacy_graph, Mapping) or not isinstance(trace, Mapping):
            raise ValueError("legacy graph and trace artifacts must contain mappings")
        split = assignments[response_id]
        expected_identity = _formal_cache_identity_from_legacy(
            legacy_graph, trace, record=record, dataset_split=split
        )
        cache_path = cache_root / split / _legacy_cache_name(record)
        _remove_stale_response_caches(cache_path.parent, response_id=response_id, keep=cache_path)
        if resume and cache_path.is_file() and _reusable_formal_cache(cache_path, expected_identity):
            continue
        formal = legacy_graph_to_formal_attention_cache(
            legacy_graph,
            trace,
            dataset_split=split,
            conversion_device=(build_device if conversion_device is None else conversion_device),
            conversion_chunk_edges=conversion_chunk_edges,
        )
        if str(formal["response_id"]) != response_id:
            raise ValueError("legacy record and graph response_id identity mismatch")
        _atomic_torch_save(cache_path, formal)

    prepared = data.prepare_graphs(
        cache_root=cache_root,
        output_dir=destination / "graphs",
        config=config,
        splits=tuple(sorted(set(assignments.values()))),
        build_device=build_device,
        resume=resume,
        mmap=mmap,
    )
    by_response = {record.response_id: record for record in prepared}
    if set(by_response) != set(response_ids):
        raise RuntimeError("prepared legacy graph response IDs do not match the requested records")
    return [by_response[response_id] for response_id in response_ids]


__all__ = [
    "discover_legacy_halueval_records",
    "evaluate_halueval_predictions",
    "legacy_graph_to_formal_attention_cache",
    "legacy_graph_to_attention_graph",
    "load_halueval_response_labels",
    "prepare_legacy_halueval_graphs",
    "split_halueval_pairs",
]
