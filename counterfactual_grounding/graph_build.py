"""Build the reusable prediction-event graph store from a legacy cache."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from attention_graph.data import (
    audit_attention_cache,
    discover_attention_cache,
    load_attention_record,
)

from .artifacts import atomic_json, atomic_jsonl, file_sha256
from .data.dataset import load_ragtruth_examples
from .data.graph import build_prediction_event_graph
from .data.ragtruth import build_ragtruth_layout
from .data.store import adapt_legacy_cache_to_layout, save_prediction_event_graph


def build_legacy_prediction_graph_store(
    *,
    cache_root: Path,
    responses: Path,
    sources: Path,
    tokenizer_source: Path,
    output_dir: Path,
    split: str,
    limit: int | None,
) -> dict[str, object]:
    """Adapt each cache once; runtime graph views are never saved as copies."""

    from transformers import AutoTokenizer

    if split not in {"train", "test"}:
        raise ValueError("graph split must be train or test")
    if limit is not None and limit <= 0:
        raise ValueError("graph limit must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        use_fast=True,
        local_files_only=True,
        trust_remote_code=False,
    )
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise RuntimeError("canonical graph replay requires a fast tokenizer")
    cache_audit = audit_attention_cache(cache_root, splits=(split,))[split]
    cache_records = discover_attention_cache(cache_root, splits=(split,))
    discovered_cache_files = len(cache_records)
    if discovered_cache_files == 0:
        raise FileNotFoundError(
            f"no formal attention-cache records discovered for split {split!r}: "
            f"{cache_root}"
        )
    if limit is not None:
        cache_records = cache_records[:limit]
    examples = load_ragtruth_examples(
        responses, sources, split=split
    )
    by_id = {example.response_id: example for example in examples}

    root = output_dir.expanduser().resolve()
    graph_dir = root / "graphs" / split
    index_rows: list[dict[str, object]] = []
    task_counts: Counter[str] = Counter()
    available_events = 0
    total_events = 0
    for position, cache in enumerate(cache_records, start=1):
        cache_path = cache.path
        record = load_attention_record(
            cache_path, device="cpu", mmap=True, include_labels=False
        )
        record_split = str(record.get("dataset_split", "")).strip().casefold()
        if record_split != cache.dataset_split or record_split != split:
            raise ValueError(
                "attention cache split metadata disagrees with its directory/request: "
                f"record={record_split!r} directory={cache.dataset_split!r} "
                f"requested={split!r} path={cache_path}"
            )
        response_id = str(record["response_id"])
        if response_id not in by_id:
            raise ValueError(
                f"attention cache response is absent from RAGTruth {split}: {response_id}"
            )
        example = by_id[response_id]
        layout = build_ragtruth_layout(
            example.source_record(), example.response, tokenizer
        )
        cache_sha256 = file_sha256(cache_path)
        adapted = adapt_legacy_cache_to_layout(
            record, layout, cache_sha256=cache_sha256
        )
        graph = build_prediction_event_graph(adapted)
        stored = save_prediction_event_graph(graph, graph_dir)
        available_events += stored.available_events
        total_events += stored.total_events
        task_counts[example.task_type] += 1
        index_rows.append(
            {
                "schema": "cept-canonical-graph-index-v1",
                "source_id": stored.source_id,
                "response_id": stored.response_id,
                "official_split": stored.dataset_split,
                "task_type": example.task_type,
                "generator_model": example.generator_model or "unknown",
                "cache_path": str(cache_path.resolve()),
                "cache_sha256": cache_sha256,
                "graph_path": str(stored.path),
                "graph_sha256": stored.sha256,
                "available_prediction_rows": stored.available_events,
                "total_prediction_events": stored.total_events,
                "legacy_shift_coverage": (
                    stored.available_events / stored.total_events
                ),
            }
        )
        print(
            f"{{\"event\":\"canonical_graph\",\"current\":{position},"
            f"\"total\":{len(cache_records)}}}",
            flush=True,
        )
    atomic_jsonl(root / f"{split}.index.jsonl", index_rows)
    cache_inventory_complete = bool(cache_audit.get("complete", False))
    graph_inventory_complete = cache_inventory_complete and limit is None
    manifest = {
        "schema": "cept-canonical-graph-store-v1",
        "state": (
            "complete" if graph_inventory_complete else "complete_partial_inventory"
        ),
        "build_state": "complete",
        "cache_inventory_complete": cache_inventory_complete,
        "graph_inventory_complete": graph_inventory_complete,
        "selection_scope": (
            "full_cache_inventory" if limit is None else "limited_prefix"
        ),
        "requested_limit": limit,
        "discovered_cache_files": discovered_cache_files,
        "selected_cache_files": len(cache_records),
        "cache_manifest_state": cache_audit.get("manifest_state"),
        "split": split,
        "graphs": len(index_rows),
        "task_counts": dict(task_counts),
        "prediction_row_coverage": available_events / total_events,
        "cache_root": str(cache_root.expanduser().resolve()),
        "responses_sha256": file_sha256(responses),
        "sources_sha256": file_sha256(sources),
        "label_policy": "canonical graph and index contain no evaluation labels",
        "legacy_adapter_warning": (
            "The old cache lacks predictor row response_idx-1. The first response "
            "event remains in every graph with row_available=false and unknown_mass=1."
        ),
        "graph_schema": "cept-prediction-event-graph-v2",
    }
    atomic_json(root / f"{split}.manifest.json", manifest)
    return manifest


__all__ = ["build_legacy_prediction_graph_store"]
