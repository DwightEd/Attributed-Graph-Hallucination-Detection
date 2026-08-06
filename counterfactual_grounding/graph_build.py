"""Build the reusable prediction-event graph store from a legacy cache."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping
from pathlib import Path

from attention_graph.data import (
    audit_attention_cache,
    discover_attention_cache,
    load_attention_record,
)

from .artifacts import (
    atomic_json,
    atomic_jsonl,
    canonical_hash,
    file_sha256,
    source_inventory_signature,
)
from .data.dataset import load_ragtruth_examples
from .data.graph import build_prediction_event_graph
from .data.ragtruth import build_ragtruth_layout
from .data.store import adapt_legacy_cache_to_layout, save_prediction_event_graph
from .observer_runtime import observer_runtime_from_cache_manifest
from .teacher.counterfactuals import (
    CounterfactualGenerationError,
    generate_equal_token_counterfactuals,
)


def _declared_cache_model_signature(
    manifest: Mapping[str, object],
    *,
    model_source: Path,
    current_signature: str,
) -> tuple[str | None, str]:
    """Verify extractor-recorded model bytes against the current source.

    New caches may carry the canonical source signature directly.  The legacy
    RAGTruth extractor already recorded a complete root-file SHA-256 inventory;
    exact filename and byte-hash equality is also sufficient evidence and does
    not require re-extracting an otherwise immutable cache.
    """

    value = manifest.get("model_source_signature")
    if value is None and isinstance(manifest.get("provenance"), Mapping):
        value = manifest["provenance"].get("model_source_signature")
    if value is not None:
        signature = str(value).strip()
        if len(signature) == 64:
            signature = "sha256:" + signature
        if not signature.startswith("sha256:") or len(signature) != 71:
            raise ValueError("attention cache model_source_signature is malformed")
        return signature, "explicit_model_source_signature"

    spec = manifest.get("attention_cache_spec")
    declared = spec.get("model_files_sha256") if isinstance(spec, Mapping) else None
    if not isinstance(declared, Mapping) or not declared:
        return None, "absent"
    declared_hashes = {str(name): str(digest) for name, digest in declared.items()}
    if any(
        not name
        or Path(name).name != name
        or len(digest) != 64
        for name, digest in declared_hashes.items()
    ):
        raise ValueError("legacy attention cache model-file inventory is malformed")
    current_files = {
        item.name: item
        for item in model_source.iterdir()
        if item.is_file()
    }
    if set(current_files) != set(declared_hashes):
        raise RuntimeError(
            "legacy attention cache model-file inventory differs from the current "
            "model/tokenizer source"
        )
    for name, expected in declared_hashes.items():
        if file_sha256(current_files[name]) != expected:
            raise RuntimeError(
                "legacy attention cache model-file bytes differ from the current "
                f"model/tokenizer source: {name}"
            )
    return current_signature, "legacy_model_files_sha256_exact_inventory"


def _load_cache_manifest(cache_audit: Mapping[str, object]) -> Mapping[str, object]:
    raw_path = cache_audit.get("manifest_path")
    if not raw_path:
        raise ValueError(
            "attention cache audit has no extractor manifest; observer runtime and "
            "cache content identity cannot be verified"
        )
    path = Path(str(raw_path)).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"attention cache manifest is absent: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, Mapping):
        raise TypeError("attention cache manifest must be an object")
    return manifest


def _verify_cache_content_inventory(
    manifest: Mapping[str, object], cache_paths: list[Path]
) -> tuple[dict[str, str], str]:
    declared = manifest.get("cache_files_sha256")
    if not isinstance(declared, Mapping) or not declared:
        raise ValueError(
            "attention cache manifest has no non-empty cache_files_sha256 inventory"
        )
    declared_hashes = {str(name): str(digest).casefold() for name, digest in declared.items()}
    if any(
        not name
        or Path(name).name != name
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        for name, digest in declared_hashes.items()
    ):
        raise ValueError("attention cache cache_files_sha256 inventory is malformed")
    paths_by_name = {path.name: path for path in cache_paths}
    if len(paths_by_name) != len(cache_paths) or set(paths_by_name) != set(
        declared_hashes
    ):
        raise RuntimeError(
            "attention cache cache_files_sha256 membership differs from discovered "
            "cache files"
        )
    observed_hashes: dict[str, str] = {}
    for name, path in paths_by_name.items():
        observed = file_sha256(path)
        if observed != declared_hashes[name]:
            raise RuntimeError(
                "attention cache-file bytes differ from extractor manifest: "
                f"{path}"
            )
        observed_hashes[name] = observed
    signature = "sha256:" + canonical_hash(
        {"cache_files_sha256": observed_hashes}
    )
    return observed_hashes, signature


def build_legacy_prediction_graph_store(
    *,
    cache_root: Path,
    responses: Path,
    sources: Path,
    tokenizer_source: Path,
    output_dir: Path,
    split: str,
    limit: int | None,
    model_source_signature: str | None = None,
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
    actual_model_source_signature = "sha256:" + source_inventory_signature(
        tokenizer_source
    )
    if model_source_signature is None:
        model_source_signature = actual_model_source_signature
    if (
        not model_source_signature.startswith("sha256:")
        or len(model_source_signature) != 71
    ):
        raise ValueError("model_source_signature must be sha256:<64 hex characters>")
    if model_source_signature != actual_model_source_signature:
        raise RuntimeError(
            "supplied model_source_signature disagrees with the actual model/"
            "tokenizer source bytes"
        )
    cache_audit = audit_attention_cache(cache_root, splits=(split,))[split]
    cache_manifest = _load_cache_manifest(cache_audit)
    observer_runtime, observer_runtime_signature = (
        observer_runtime_from_cache_manifest(cache_manifest)
    )
    cache_model_source_signature, cache_model_identity_evidence = (
        _declared_cache_model_signature(
            cache_manifest,
            model_source=tokenizer_source,
            current_signature=model_source_signature,
        )
    )
    if (
        cache_model_source_signature is not None
        and cache_model_source_signature != model_source_signature
    ):
        raise RuntimeError(
            "attention cache was extracted from a different model source than "
            "the requested teacher/tokenizer source"
        )
    if cache_model_source_signature != model_source_signature:
        cache_model_identity_status = "unverified_legacy_manifest"
    elif cache_model_identity_evidence == "explicit_model_source_signature":
        cache_model_identity_status = "verified_extractor_manifest"
    else:
        cache_model_identity_status = "verified_legacy_content_manifest"
    cache_records = discover_attention_cache(cache_root, splits=(split,))
    discovered_cache_files = len(cache_records)
    if discovered_cache_files == 0:
        raise FileNotFoundError(
            f"no formal attention-cache records discovered for split {split!r}: "
            f"{cache_root}"
        )
    verified_cache_hashes, cache_content_inventory_signature = (
        _verify_cache_content_inventory(
            cache_manifest, [record.path for record in cache_records]
        )
    )
    if limit is not None:
        cache_records = cache_records[:limit]
    examples = load_ragtruth_examples(responses, sources, split=split)
    by_id = {example.response_id: example for example in examples}

    root = output_dir.expanduser().resolve()
    graph_dir = root / "graphs" / split
    index_rows: list[dict[str, object]] = []
    task_counts: Counter[str] = Counter()
    available_events = 0
    total_events = 0
    deployment_seed_available = 0
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
        try:
            deployment_seed_positions = tuple(
                int(value)
                for value in generate_equal_token_counterfactuals(
                    layout, tokenizer, max_candidates=1
                )[0].changed_positions.tolist()
            )
        except CounterfactualGenerationError:
            deployment_seed_positions = ()
        deployment_seed_available += int(bool(deployment_seed_positions))
        cache_sha256 = verified_cache_hashes[cache_path.name]
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
                "schema": "cept-canonical-graph-index-v2",
                "source_id": stored.source_id,
                "response_id": stored.response_id,
                "official_split": stored.dataset_split,
                "task_type": example.task_type,
                "generator_model": example.generator_model or "unknown",
                "cache_path": str(cache_path.resolve()),
                "cache_sha256": cache_sha256,
                "extractor_declared_cache_sha256": cache_sha256,
                "cache_content_identity_status": (
                    "verified_extractor_cache_files_sha256"
                ),
                "cache_content_identity_evidence": (
                    "cache_files_sha256_exact_bytes"
                ),
                "cache_content_inventory_signature": (
                    cache_content_inventory_signature
                ),
                "graph_path": str(stored.path),
                "graph_sha256": stored.sha256,
                "model_source_signature": model_source_signature,
                "cache_model_source_signature": cache_model_source_signature,
                "cache_model_identity_status": cache_model_identity_status,
                "cache_model_identity_evidence": cache_model_identity_evidence,
                "observer_runtime": observer_runtime,
                "observer_runtime_signature": observer_runtime_signature,
                "deployment_seed_protocol": (
                    "numeric_digit_surface_preserving_v1_first_candidate"
                ),
                "deployment_seed_positions": list(deployment_seed_positions),
                "available_prediction_rows": stored.available_events,
                "total_prediction_events": stored.total_events,
                "legacy_shift_coverage": (
                    stored.available_events / stored.total_events
                ),
            }
        )
        print(
            f'{{"event":"canonical_graph","current":{position},'
            f'"total":{len(cache_records)}}}',
            flush=True,
        )
    atomic_jsonl(root / f"{split}.index.jsonl", index_rows)
    cache_inventory_complete = bool(cache_audit.get("complete", False))
    graph_inventory_complete = cache_inventory_complete and limit is None
    manifest = {
        "schema": "cept-canonical-graph-store-v2",
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
        "deployment_seed_available_graphs": deployment_seed_available,
        "deployment_seed_coverage": deployment_seed_available / len(index_rows),
        "deployment_seed_protocol": (
            "numeric_digit_surface_preserving_v1_first_candidate"
        ),
        "cache_root": str(cache_root.expanduser().resolve()),
        "responses_sha256": file_sha256(responses),
        "sources_sha256": file_sha256(sources),
        "model_source_signature": model_source_signature,
        "cache_model_source_signature": cache_model_source_signature,
        "cache_model_identity_status": cache_model_identity_status,
        "cache_model_identity_evidence": cache_model_identity_evidence,
        "cache_content_identity_status": (
            "verified_extractor_cache_files_sha256"
        ),
        "cache_content_identity_evidence": "cache_files_sha256_exact_bytes",
        "cache_content_inventory_signature": cache_content_inventory_signature,
        "observer_runtime": observer_runtime,
        "observer_runtime_signature": observer_runtime_signature,
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
