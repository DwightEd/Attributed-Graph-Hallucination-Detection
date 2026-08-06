"""Orchestration for cache validation, topology extraction, training and scoring."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Iterable

from .config import CorruptionConfig, TopologyConfig
from .contracts import AttentionStore, load_store
from .corruptions import CorruptedAttentionStore
from .model import RankerConfig, fit_ranker, load_checkpoint, save_checkpoint, score_signatures
from .signature import TopologySignature, extract_signature


def discover_samples(root: str | Path, *, recursive: bool = True) -> list[Path]:
    directory = Path(root)
    if not directory.is_dir():
        raise ValueError(f"attention directory does not exist: {directory}")
    paths = sorted(directory.rglob("*.pt") if recursive else directory.glob("*.pt"))
    if not paths:
        raise ValueError(f"no .pt attention samples found in {directory}")
    return paths


def _write_jsonl(records: Iterable[dict[str, object]], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _store_contract(store: AttentionStore) -> tuple[int, int]:
    return store.layers, store.heads


def validate_directory(
    input_dir: str | Path,
    *,
    recursive: bool = True,
    limit: int | None = None,
) -> dict[str, object]:
    """Validate persisted graph compatibility without reading any labels."""

    paths = discover_samples(input_dir, recursive=recursive)
    selected = paths if limit is None else paths[: max(1, int(limit))]
    stores = [load_store(path) for path in selected]
    contracts = {_store_contract(store) for store in stores}
    if len(contracts) != 1:
        raise ValueError(f"mixed layer/head contracts are not supported: {sorted(contracts)}")
    layers, heads = next(iter(contracts))
    observed = []
    for store in stores:
        row = store.response_rows(0, 0)
        observed.append(float((row.sum(dim=1) > 0).float().mean()))
    return {
        "schema": "prompt-anchored-topology-flow-cache-validation-v1",
        "root": str(Path(input_dir).resolve()),
        "discovered_samples": len(paths),
        "validated_samples": len(stores),
        "layers": layers,
        "heads": heads,
        "minimum_tokens": min(store.token_count for store in stores),
        "maximum_tokens": max(store.token_count for store in stores),
        "minimum_response_tokens": min(store.response_tokens for store in stores),
        "mean_first_channel_observed_row_fraction": sum(observed) / len(observed),
    }


def extract_paths(
    paths: list[Path],
    *,
    topology: TopologyConfig | None = None,
) -> list[TopologySignature]:
    topology = topology or TopologyConfig()
    signatures: list[TopologySignature] = []
    expected_contract: tuple[int, int] | None = None
    for path in paths:
        store = load_store(path)
        contract = _store_contract(store)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ValueError(
                f"mixed model topology contracts: expected {expected_contract}, got {contract} at {path}"
            )
        signatures.append(extract_signature(store, config=topology))
    return signatures


def extract_directory(
    input_dir: str | Path,
    output_path: str | Path,
    *,
    topology: TopologyConfig | None = None,
    recursive: bool = True,
) -> list[TopologySignature]:
    signatures = extract_paths(discover_samples(input_dir, recursive=recursive), topology=topology)
    _write_jsonl((signature.to_record() for signature in signatures), output_path)
    return signatures


def train_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    topology: TopologyConfig | None = None,
    corruption: CorruptionConfig | None = None,
    ranker: RankerConfig | None = None,
    recursive: bool = True,
    device: str = "cpu",
) -> dict[str, object]:
    topology = topology or TopologyConfig()
    corruption = corruption or CorruptionConfig(mode="all")
    ranker = ranker or RankerConfig()
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output_dir must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    paths = discover_samples(input_dir, recursive=recursive)
    originals: list[TopologySignature] = []
    paired_originals: list[TopologySignature] = []
    counterfactuals: list[TopologySignature] = []
    group_keys: list[str] = []
    modes = (
        ("incidence", "collapse", "composite")
        if corruption.mode == "all"
        else (corruption.mode,)
    )
    expected_contract: tuple[int, int] | None = None
    for path in paths:
        store = load_store(path)
        contract = _store_contract(store)
        if expected_contract is None:
            expected_contract = contract
        elif contract != expected_contract:
            raise ValueError(f"mixed layer/head contracts at {path}: {contract} != {expected_contract}")
        original = extract_signature(store, config=topology)
        originals.append(original)
        group = original.source_id if original.source_id != "unknown" else original.sample_id
        for mode in modes:
            paired_originals.append(original)
            group_keys.append(group)
            counterfactuals.append(
                extract_signature(
                    CorruptedAttentionStore(store, replace(corruption, mode=mode)),
                    config=topology,
                )
            )
    checkpoint, history = fit_ranker(
        paired_originals,
        counterfactuals,
        group_keys=group_keys,
        config=ranker,
        topology_config=topology.to_dict(),
        device=device,
    )
    save_checkpoint(checkpoint, output / "topology_flow_ranker.pt")
    _write_jsonl((signature.to_record() for signature in originals), output / "train.original.jsonl")
    _write_jsonl(
        (signature.to_record() for signature in counterfactuals),
        output / "train.counterfactual.jsonl",
    )
    (output / "history.json").write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    run = {
        "schema": "prompt-anchored-topology-flow-run-v2",
        "samples": len(paths),
        "source_groups": len(set(group_keys)),
        "training_pairs": len(counterfactuals),
        "corruption_modes": list(modes),
        "topology_config": topology.to_dict(),
        "corruption_config": corruption.to_dict(),
        "ranker_config": ranker.__dict__,
        "last_history": history[-1],
    }
    (output / "run.json").write_text(
        json.dumps(run, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return run


def score_directory(
    input_dir: str | Path,
    checkpoint_path: str | Path,
    output_path: str | Path,
    *,
    topology: TopologyConfig | None = None,
    recursive: bool = True,
) -> list[dict[str, object]]:
    checkpoint = load_checkpoint(checkpoint_path)
    checkpoint_topology = TopologyConfig(**checkpoint.topology_config)
    if topology is not None and topology.to_dict() != checkpoint_topology.to_dict():
        raise ValueError(
            "score topology configuration must exactly match the training checkpoint: "
            f"requested={topology.to_dict()} checkpoint={checkpoint_topology.to_dict()}"
        )
    topology = checkpoint_topology
    signatures = extract_paths(
        discover_samples(input_dir, recursive=recursive), topology=topology
    )
    if any(signature.feature_names != checkpoint.feature_names for signature in signatures):
        raise ValueError("checkpoint and extracted feature contracts do not match")
    scores = score_signatures(checkpoint, signatures)
    records = []
    for signature, score in zip(signatures, scores.tolist()):
        record = signature.to_record(include_trajectory=False)
        record["topology_anomaly_score"] = float(score)
        records.append(record)
    _write_jsonl(records, output_path)
    return records
