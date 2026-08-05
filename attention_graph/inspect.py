"""Inspect reusable label-free attention graph artifacts.

The inspector intentionally uses :func:`attention_graph.data.load_graph` so
the same schema and integrity checks apply to research code and diagnostics.
It never reads evaluation labels or the Grounding Flow detector outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import uuid
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .data import load_graph
from .graph import AttentionGraph

GRAPH_STRUCTURE_SCHEMA = "attention-graph-structure-v1"
RUN_INSPECTION_SCHEMA = "attention-graph-run-inspection-v1"
_STATISTICS_SAMPLE_LIMIT = 100_000
_INDEX_NUMERIC_FIELDS = (
    "num_nodes",
    "num_response_nodes",
    "num_edges",
    "num_rp_edges",
    "num_rr_edges",
    "num_traces",
)
_TENSOR_FIELDS = (
    "node_attr",
    "node_context",
    "response_mask",
    "edge_index",
    "edge_type",
    "edge_score",
    "trace_edge_id",
    "trace_channel",
    "trace_value",
    "token_ids",
)


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be a positive integer") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _json_index_path(run_dir: str | Path) -> tuple[Path, Path]:
    root = Path(run_dir).expanduser().resolve()
    candidates = (
        root / "prepared" / "graphs" / "artifact_index.json",
        root / "graphs" / "artifact_index.json",
        root / "artifact_index.json",
        root / "prepared" / "graphs" / "index.json",
        root / "graphs" / "index.json",
        root / "index.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return root, candidate
    raise FileNotFoundError(
        "prepared graph index is absent; expected one of: "
        + ", ".join(str(path) for path in candidates)
    )


def _read_graph_index(
    run_dir: str | Path,
) -> tuple[Path, Path, list[dict[str, object]]]:
    root, index_path = _json_index_path(run_dir)
    try:
        loaded = json.loads(index_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid prepared graph index: {index_path}") from error
    if not isinstance(loaded, list) or not loaded:
        raise ValueError("prepared graph index must contain a non-empty JSON list")
    records: list[dict[str, object]] = []
    response_ids: list[str] = []
    for offset, raw in enumerate(loaded):
        if not isinstance(raw, Mapping):
            raise TypeError(f"prepared graph index row {offset} must be an object")
        record = {str(key): value for key, value in raw.items()}
        response_id = str(record.get("response_id", "")).strip()
        if not response_id:
            raise ValueError(f"prepared graph index row {offset} lacks response_id")
        response_ids.append(response_id)
        records.append(record)
    duplicates = sorted(
        response_id for response_id, count in Counter(response_ids).items() if count > 1
    )
    if duplicates:
        raise ValueError(
            "prepared graph index contains duplicate response_id values: "
            + ", ".join(duplicates[:10])
        )
    return root, index_path, records


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _select_record(
    records: Sequence[Mapping[str, object]], response_id: str | None
) -> Mapping[str, object]:
    requested = None if response_id is None else str(response_id).strip()
    if requested == "":
        raise ValueError("response_id must be non-empty when provided")
    selected = (
        records[0]
        if requested is None
        else next(
            (record for record in records if str(record["response_id"]) == requested),
            None,
        )
    )
    if selected is None:
        raise ValueError(
            f"response_id is absent from prepared graph index: {requested}"
        )
    return selected


def _resolve_indexed_path(index_path: Path, record: Mapping[str, object]) -> Path:
    raw_path = str(record.get("graph_path", "")).strip()
    if not raw_path:
        raise ValueError("prepared graph index row lacks graph_path")
    stored = Path(raw_path).expanduser()
    graph_root = index_path.parent.resolve()
    split = str(record.get("dataset_split", "")).strip().casefold()
    local = graph_root / split / stored.name
    if split in {"train", "test"} and local.is_file():
        resolved_local = local.resolve()
        if not _is_within(resolved_local, graph_root):
            raise ValueError(
                f"indexed graph resolves outside prepared graph directory: {local}"
            )
        return resolved_local

    stored_candidate = (
        stored.resolve() if stored.is_absolute() else (graph_root / stored).resolve()
    )
    escaped_existing_path = stored_candidate.is_file() and not _is_within(
        stored_candidate, graph_root
    )
    if stored_candidate.is_file() and not escaped_existing_path:
        return stored_candidate

    matches = sorted(
        resolved
        for match in graph_root.rglob(stored.name)
        if match.is_file() and _is_within((resolved := match.resolve()), graph_root)
    )
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"copied run contains multiple graph files named {stored.name}: "
            + ", ".join(str(path) for path in matches[:10])
        )
    if escaped_existing_path:
        raise ValueError(
            "indexed graph resolves outside prepared graph directory: "
            f"{stored_candidate}"
        )
    raise FileNotFoundError(
        f"prepared graph file is absent: stored={stored} local_fallback={local}"
    )


def resolve_graph(
    run_dir: str | Path,
    *,
    response_id: str | None = None,
    graph_path: str | Path | None = None,
) -> Path:
    """Resolve one graph from a run, with a portable copied-run fallback."""

    if graph_path is not None:
        if response_id is not None:
            raise ValueError("response_id and graph_path are mutually exclusive")
        direct = Path(graph_path).expanduser().resolve()
        if not direct.is_file():
            raise FileNotFoundError(f"prepared graph file is absent: {direct}")
        return direct

    _, index_path, records = _read_graph_index(run_dir)
    selected = _select_record(records, response_id)
    return _resolve_indexed_path(index_path, selected)


def _tensor_schema(graph: AttentionGraph) -> dict[str, dict[str, object]]:
    return {
        name: {
            "shape": list(getattr(graph, name).shape),
            "dtype": str(getattr(graph, name).dtype).removeprefix("torch."),
            "device": str(getattr(graph, name).device),
        }
        for name in _TENSOR_FIELDS
    }


def _tensor_statistics(value: torch.Tensor) -> dict[str, object]:
    flat = value.detach().flatten()
    count = flat.numel()
    if not count:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "maximum": None,
            "median_method": "empty",
            "median_sample_count": 0,
        }
    if not bool(torch.isfinite(flat).all()):
        raise ValueError("graph inspection requires finite attention values")
    if count <= _STATISTICS_SAMPLE_LIMIT:
        median_sample = flat
        median_method = "exact"
    else:
        sample_indices = torch.linspace(
            0,
            count - 1,
            steps=_STATISTICS_SAMPLE_LIMIT,
            dtype=torch.long,
            device=flat.device,
        )
        median_sample = flat[sample_indices]
        median_method = "deterministic_systematic_sample"
    if not (median_sample.is_floating_point() or median_sample.is_complex()):
        median_sample = median_sample.float()
    return {
        "count": count,
        "minimum": float(flat.min().item()),
        "median": float(torch.quantile(median_sample, 0.5).item()),
        "mean": float(flat.mean(dtype=torch.float64).item()),
        "maximum": float(flat.max().item()),
        "median_method": median_method,
        "median_sample_count": median_sample.numel(),
    }


def _edge_trace_aggregates(
    graph: AttentionGraph,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    edge_count = graph.num_edges
    trace_edge = graph.trace_edge_id.detach().cpu().long()
    trace_value = graph.trace_value.detach().cpu().float()
    counts = torch.bincount(trace_edge, minlength=edge_count)
    sums = torch.zeros(edge_count, dtype=torch.float32)
    maxima = torch.zeros(edge_count, dtype=torch.float32)
    if len(trace_edge):
        sums.index_add_(0, trace_edge, trace_value)
        maxima.fill_(-torch.inf)
        maxima.scatter_reduce_(
            0, trace_edge, trace_value, reduce="amax", include_self=True
        )
        maxima[maxima == -torch.inf] = 0.0
    starts = torch.cumsum(counts, dim=0) - counts
    return counts, starts, sums, maxima


def _top_indices(value: torch.Tensor, *, limit: int) -> list[int]:
    selected_count = min(limit, value.numel())
    if selected_count == 0:
        return []
    selected = torch.topk(
        value, k=selected_count, largest=True, sorted=True
    ).indices.tolist()
    return sorted(selected, key=lambda item: (-float(value[item]), item))


def _top_edge_rows(graph: AttentionGraph, *, limit: int) -> list[dict[str, object]]:
    source = graph.edge_index[0].detach().cpu().long()
    target = graph.edge_index[1].detach().cpu().long()
    relation = graph.edge_type.detach().cpu().long()
    scores = graph.edge_score.detach().cpu().float()
    token_ids = graph.token_ids.detach().cpu().long()
    trace_channel = graph.trace_channel.detach().cpu().long()
    trace_value = graph.trace_value.detach().cpu().float()
    counts, starts, sums, maxima = _edge_trace_aggregates(graph)
    ordered = _top_indices(scores, limit=limit)
    rows: list[dict[str, object]] = []
    for edge_id in ordered:
        trace_start = int(starts[edge_id])
        trace_end = trace_start + int(counts[edge_id])
        local_channel_order = _top_indices(trace_value[trace_start:trace_end], limit=5)
        channel_order = [trace_start + offset for offset in local_channel_order]
        source_id = int(source[edge_id])
        target_id = int(target[edge_id])
        relation_name = "RP" if int(relation[edge_id]) == 0 else "RR"
        rows.append(
            {
                "edge_id": edge_id,
                "relation": relation_name,
                "source": source_id,
                "target": target_id,
                "source_role": "prompt"
                if source_id < graph.response_idx
                else "response",
                "target_response_offset": target_id - graph.response_idx,
                "causal_lag": target_id - source_id,
                "source_token_id": int(token_ids[source_id]),
                "target_token_id": int(token_ids[target_id]),
                "edge_score": float(scores[edge_id]),
                "observed_channels": int(counts[edge_id]),
                "observed_channel_fraction": float(
                    counts[edge_id] / graph.num_channels
                ),
                "retained_attention_sum": float(sums[edge_id]),
                "maximum_attention": float(maxima[edge_id]),
                "strongest_channels": [
                    {
                        "channel": int(trace_channel[offset]),
                        "layer": int(trace_channel[offset]) // graph.num_heads,
                        "head": int(trace_channel[offset]) % graph.num_heads,
                        "attention": float(trace_value[offset]),
                    }
                    for offset in channel_order
                ],
            }
        )
    return rows


def _per_target_rows(graph: AttentionGraph, *, limit: int) -> list[dict[str, object]]:
    source = graph.edge_index[0].detach().cpu().long()
    target = graph.edge_index[1].detach().cpu().long()
    relation = graph.edge_type.detach().cpu().long()
    scores = graph.edge_score.detach().cpu().float()
    token_ids = graph.token_ids.detach().cpu().long()
    node_attr = graph.node_attr.detach().cpu().float()
    response_nodes = graph.num_nodes - graph.response_idx
    local_target = target - graph.response_idx
    relation_group = local_target * 2 + relation
    typed_counts = torch.bincount(relation_group, minlength=response_nodes * 2).reshape(
        response_nodes, 2
    )
    typed_mass = torch.zeros(response_nodes * 2, dtype=torch.float32)
    typed_mass.index_add_(0, relation_group, scores)
    typed_mass = typed_mass.reshape(response_nodes, 2)
    earliest = torch.full((response_nodes,), graph.num_nodes, dtype=torch.long)
    latest = torch.full((response_nodes,), -1, dtype=torch.long)
    earliest.scatter_reduce_(0, local_target, source, reduce="amin", include_self=True)
    latest.scatter_reduce_(0, local_target, source, reduce="amax", include_self=True)

    rows: list[dict[str, object]] = []
    for response_offset in range(min(response_nodes, limit)):
        target_id = graph.response_idx + response_offset
        rp_count = int(typed_counts[response_offset, 0])
        rr_count = int(typed_counts[response_offset, 1])
        rows.append(
            {
                "target": target_id,
                "response_offset": response_offset,
                "token_id": int(token_ids[target_id]),
                "historical_edges": rp_count + rr_count,
                "rp_edges": rp_count,
                "rr_edges": rr_count,
                "rp_retained_attention_lower_bound": float(
                    typed_mass[response_offset, 0]
                ),
                "rr_retained_attention_lower_bound": float(
                    typed_mass[response_offset, 1]
                ),
                "historical_retained_attention_lower_bound": float(
                    typed_mass[response_offset].sum()
                ),
                "self_attention_mean": float(node_attr[target_id].mean()),
                "earliest_source": int(earliest[response_offset])
                if rp_count + rr_count
                else None,
                "latest_source": int(latest[response_offset])
                if rp_count + rr_count
                else None,
            }
        )
    return rows


def _top_channel_rows(graph: AttentionGraph, *, limit: int) -> list[dict[str, object]]:
    channel = graph.trace_channel.detach().cpu().long()
    attention = graph.trace_value.detach().cpu().float()
    counts = torch.bincount(channel, minlength=graph.num_channels)
    sums = torch.zeros(graph.num_channels, dtype=torch.float32)
    maxima = torch.zeros(graph.num_channels, dtype=torch.float32)
    if len(channel):
        sums.index_add_(0, channel, attention)
        maxima.fill_(-torch.inf)
        maxima.scatter_reduce_(0, channel, attention, reduce="amax", include_self=True)
        maxima[maxima == -torch.inf] = 0.0
    ordered = _top_indices(sums, limit=limit)
    return [
        {
            "channel": channel_id,
            "layer": channel_id // graph.num_heads,
            "head": channel_id % graph.num_heads,
            "trace_entries": int(counts[channel_id]),
            "retained_attention_sum": float(sums[channel_id]),
            "mean_observed_attention": (
                float(sums[channel_id] / counts[channel_id])
                if int(counts[channel_id])
                else None
            ),
            "maximum_attention": float(maxima[channel_id]),
        }
        for channel_id in ordered
    ]


def summarize_graph(
    graph: AttentionGraph,
    *,
    top_edges: int = 20,
    max_targets: int = 64,
) -> dict[str, object]:
    """Return a JSON-safe structural report for one prepared graph."""

    if top_edges < 1 or max_targets < 1:
        raise ValueError("top_edges and max_targets must be positive")
    prompt_nodes = graph.response_idx
    response_nodes = graph.num_nodes - graph.response_idx
    rp_edges = int((graph.edge_type == 0).sum())
    rr_edges = int((graph.edge_type == 1).sum())
    possible_rp = prompt_nodes * response_nodes
    possible_rr = response_nodes * (response_nodes - 1) // 2
    possible = possible_rp + possible_rr
    return {
        "schema": GRAPH_STRUCTURE_SCHEMA,
        "identity": {
            "source_id": graph.source_id,
            "response_id": graph.response_id,
            "sample_id": graph.sample_id,
            "response_idx": graph.response_idx,
            "attention_floor": graph.attention_floor,
        },
        "build_config": asdict(graph.build_config),
        "censorship": {
            "cache_censored": True,
            "attention_floor": graph.attention_floor,
            "edge_score_definition": (
                "sum of retained trace attention divided by all layer-head "
                "channels; below-floor attention remains unknown, so this is "
                "a lower bound"
            ),
        },
        "dimensions": {
            "nodes": graph.num_nodes,
            "prompt_nodes": prompt_nodes,
            "response_nodes": response_nodes,
            "edges": graph.num_edges,
            "rp_edges": rp_edges,
            "rr_edges": rr_edges,
            "layers": graph.num_layers,
            "heads": graph.num_heads,
            "channels": graph.num_channels,
            "traces": int(graph.trace_value.numel()),
        },
        "tensor_schema": _tensor_schema(graph),
        "relation_counts": {"RP": rp_edges, "RR": rr_edges},
        "topology": {
            "possible_causal_edges": possible,
            "possible_rp_edges": possible_rp,
            "possible_rr_edges": possible_rr,
            "selected_edge_density": float(graph.num_edges / possible)
            if possible
            else 0.0,
            "rp_density": float(rp_edges / possible_rp) if possible_rp else 0.0,
            "rr_density": float(rr_edges / possible_rr) if possible_rr else 0.0,
        },
        "attention_statistics": {
            "self_attention_diagonal": _tensor_statistics(graph.node_attr),
            "retained_attention_lower_bound_per_channel": _tensor_statistics(
                graph.edge_score
            ),
            "observed_trace_attention": _tensor_statistics(graph.trace_value),
        },
        "per_target": _per_target_rows(graph, limit=max_targets),
        "per_target_truncated": response_nodes > max_targets,
        "top_edges": _top_edge_rows(graph, limit=top_edges),
        "top_channels": _top_channel_rows(
            graph, limit=min(top_edges, graph.num_channels)
        ),
    }


def _number_summary(
    records: Sequence[Mapping[str, object]], field: str
) -> dict[str, object]:
    values: list[float] = []
    for offset, record in enumerate(records):
        if field not in record:
            raise ValueError(f"prepared graph index row {offset} lacks {field}")
        raw = record[field]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise TypeError(f"prepared graph index {field} must be numeric")
        value = float(raw)
        if not math.isfinite(value) or value < 0:
            raise ValueError(
                f"prepared graph index {field} must be finite and non-negative"
            )
        values.append(value)
    ordered = sorted(values)
    middle = len(ordered) // 2
    median = (
        ordered[middle]
        if len(ordered) % 2
        else (ordered[middle - 1] + ordered[middle]) / 2.0
    )
    return {
        "total": int(sum(values)),
        "minimum": ordered[0],
        "median": median,
        "mean": float(sum(values) / len(values)),
        "maximum": ordered[-1],
    }


def inspect_run(
    run_dir: str | Path,
    *,
    response_id: str | None = None,
    top_edges: int = 20,
    max_targets: int = 64,
) -> dict[str, object]:
    """Inspect an entire prepared run and one selected reusable graph."""

    root, index_path, records = _read_graph_index(run_dir)
    selected_record = _select_record(records, response_id)
    selected_path = _resolve_indexed_path(index_path, selected_record)
    graph = load_graph(selected_path, device="cpu", mmap=True, validate=True)
    expected_response_id = str(selected_record["response_id"])
    if graph.response_id != expected_response_id:
        raise ValueError(
            "prepared graph identity conflicts with its index: "
            f"index={expected_response_id} graph={graph.response_id}"
        )
    split_counts = dict(
        sorted(
            Counter(
                str(record.get("split", record.get("dataset_split", "unknown")))
                for record in records
            ).items()
        )
    )
    return {
        "schema": RUN_INSPECTION_SCHEMA,
        "run_dir": str(root),
        "graph_index": str(index_path),
        "reusable": {
            "selected_graph_label_free": True,
            "selected_graph_schema_validated": True,
            "loader": "attention_graph.data.load_graph",
            "mmap_compatible": True,
            "independent_of_detector": True,
        },
        "inventory": {
            "indexed_graphs": len(records),
            "split_counts": split_counts,
            "distributions": {
                field: _number_summary(records, field)
                for field in _INDEX_NUMERIC_FIELDS
            },
        },
        "selection": {
            "requested_response_id": response_id,
            "selected_response_id": graph.response_id,
            "graph_path": str(selected_path),
            "policy": "response_id"
            if response_id is not None
            else "first_index_record",
        },
        "selected_graph": summarize_graph(
            graph, top_edges=top_edges, max_targets=max_targets
        ),
    }


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect reusable label-free attention graph artifacts"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run-dir", type=Path, help="Grounding Flow output directory")
    source.add_argument("--graph", type=Path, help="one prepared *.graph.pt artifact")
    parser.add_argument("--response-id", help="select one response from --run-dir")
    parser.add_argument("--top-edges", type=_positive_int, default=20)
    parser.add_argument("--max-targets", type=_positive_int, default=64)
    parser.add_argument("--output", type=Path, help="optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.graph is not None:
        if args.response_id is not None:
            raise ValueError("--response-id requires --run-dir")
        graph = load_graph(args.graph, device="cpu", mmap=True, validate=True)
        report = summarize_graph(
            graph, top_edges=args.top_edges, max_targets=args.max_targets
        )
    else:
        report = inspect_run(
            args.run_dir,
            response_id=args.response_id,
            top_edges=args.top_edges,
            max_targets=args.max_targets,
        )
    if args.output is not None:
        _atomic_json(args.output, report)
    print(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GRAPH_STRUCTURE_SCHEMA",
    "RUN_INSPECTION_SCHEMA",
    "build_parser",
    "inspect_run",
    "main",
    "resolve_graph",
    "summarize_graph",
]
