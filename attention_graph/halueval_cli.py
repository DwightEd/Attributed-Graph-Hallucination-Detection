"""Label-free RelationAwareMaskGAE runner for paired HaluEval artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .data import PreparedGraphRecord, load_graph
from .graph import GraphBuildConfig
from .halueval import (
    discover_legacy_halueval_records,
    evaluate_halueval_predictions,
    load_halueval_response_labels,
    prepare_legacy_halueval_graphs,
    split_halueval_pairs,
)
from .model import RelationAwareMaskGAE
from .train import TrainingConfig, score_graphs, train_relation_mae


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    if str(value).strip().casefold() in {"none", "0", "uncapped", "unbounded"}:
        return None
    return _positive_int(value)


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def _rate(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _nonnegative(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Label-free paired HaluEval attention-graph runner")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="prepare, train, score responses, then evaluate")
    run.add_argument("--extraction-dir", type=Path, required=True)
    run.add_argument("--source-run", type=Path, help="parent legacy extraction run, for provenance only")
    run.add_argument("--examples", type=Path, required=True)
    run.add_argument("--evaluation-labels", type=Path, required=True)
    run.add_argument("--output-dir", "--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--selection", choices=("threshold", "global_topk", "typed_topk"), default="threshold")
    run.add_argument("--threshold", type=float)
    run.add_argument("--top-k", type=_positive_int, default=8)
    run.add_argument("--max-edges-per-target", type=_optional_positive_int, default=64)
    run.add_argument("--query-block", type=_positive_int, default=32)
    run.add_argument("--epochs", type=_positive_int, default=30)
    run.add_argument("--patience", type=_positive_int, default=6)
    run.add_argument("--learning-rate", type=float, default=1e-3)
    run.add_argument("--weight-decay", type=float, default=1e-5)
    run.add_argument("--edge-mask-rate", type=_rate, default=0.35)
    run.add_argument("--node-mask-rate", type=_rate, default=0.30)
    run.add_argument("--channel-drop-rate", type=_rate, default=0.10)
    run.add_argument("--support-weight", type=_nonnegative, default=1.0)
    run.add_argument("--attention-weight", type=_nonnegative, default=1.0)
    run.add_argument("--distribution-weight", type=_nonnegative, default=1.0)
    run.add_argument("--node-weight", type=_nonnegative, default=0.25)
    run.add_argument("--embedding-dim", type=_positive_int, default=128)
    run.add_argument("--message-passing-steps", type=int, default=2)
    run.add_argument("--dropout", type=_rate, default=0.10)
    run.add_argument("--max-support-edges", type=_positive_int, default=8_192)
    run.add_argument("--max-weight-traces", type=_positive_int, default=65_536)
    run.add_argument("--max-distribution-groups", type=_positive_int, default=512)
    run.add_argument("--decoder-chunk-size", type=_positive_int, default=16_384)
    run.add_argument("--conversion-chunk-edges", type=_positive_int, default=8_192)
    run.add_argument("--num-score-views", type=_positive_int, default=4)
    run.add_argument("--embedding-only-scoring", action="store_true")
    run.add_argument("--validation-fraction", type=_fraction, default=0.10)
    run.add_argument("--test-fraction", type=_fraction, default=0.20)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--limit-pairs", type=_positive_int)
    run.add_argument(
        "--group-by-prompt",
        action="store_true",
        help="prevent repeated prepared knowledge/question prompts from crossing splits",
    )
    run.add_argument("--require-complete-cache", action="store_true")
    run.add_argument("--skip-evaluation", action="store_true")
    run.add_argument(
        "--no-prepare-resume",
        action="store_true",
        help="rebuild preparation artifacts instead of reusing validated preparation caches",
    )
    run.set_defaults(handler=run_pipeline)
    return parser


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(dict(record), ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _print_prepare_progress(current: int, total: int) -> None:
    """Keep long legacy-cache conversion visibly alive without flooding logs."""

    if current == 1 or current == total or current % 25 == 0:
        print(
            json.dumps(
                {"event": "prepare_progress", "current": current, "total": total},
                sort_keys=True,
            ),
            flush=True,
        )


def _fresh_output(path: Path, *, resume: bool) -> None:
    """Permit only the adapter's independently reusable preparation cache."""

    if not path.exists() or not any(path.iterdir()):
        path.mkdir(parents=True, exist_ok=True)
        return
    final_artifacts = (
        "run.json", "splits.json", "test.response_predictions.jsonl",
        "response_mixture.json", "evaluation.json",
    )
    if any((path / name).exists() for name in final_artifacts):
        raise FileExistsError(f"run output already contains final artifacts: {path}")
    if not resume or not _only_reusable_prepared_artifacts(path):
        raise FileExistsError(f"run output is not safely resumable: {path}")


def _only_reusable_prepared_artifacts(path: Path) -> bool:
    """Recognize the two atomically-produced cache trees owned by this runner."""

    for child in path.rglob("*"):
        relative = child.relative_to(path).parts
        if child.is_dir() and relative in {
            ("prepared",), ("prepared", "adapted_cache"),
            ("prepared", "adapted_cache", "train"), ("prepared", "adapted_cache", "test"),
            ("prepared", "graphs"), ("prepared", "graphs", "train"),
            ("prepared", "graphs", "test"),
        }:
            continue
        if child.is_file() and (
            relative == ("prepared", "graphs", "index.json")
            or (
                len(relative) == 4
                and relative[0] == "prepared"
                and relative[1] == "adapted_cache"
                and relative[2] in {"train", "test"}
                and relative[3].startswith("attention_")
                and relative[3].endswith(".pt")
            )
            or (
                len(relative) == 4
                and relative[0] == "prepared"
                and relative[1] == "graphs"
                and relative[2] in {"train", "test"}
                and relative[3].startswith("attention_")
                and relative[3].endswith(".graph.pt")
            )
        ):
            continue
        return False
    return True


def _read_examples(path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Read label-free response identities and stable source-prompt group ids."""

    pairs: dict[str, str] = {}
    prompt_groups: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"cannot read HaluEval examples: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid examples JSONL row {line_number}") from error
        if not isinstance(row, Mapping):
            raise ValueError("HaluEval examples must be JSON objects")
        if any("label" in str(key).casefold() for key in row):
            raise ValueError("HaluEval graph runner examples must be label-free")
        response_id, pair_id = str(row.get("response_id", row.get("example_id", ""))).strip(), str(row.get("pair_id", "")).strip()
        if not response_id or not pair_id or response_id in pairs:
            raise ValueError("examples require unique response_id and non-empty pair_id")
        source = row.get("knowledge", row.get("passage", row.get("prompt", "")))
        question = row.get("question", "")
        if not str(source).strip() and not str(question).strip():
            raise ValueError("examples require prepared knowledge/passage or prompt content")
        payload = json.dumps(
            [str(source), str(question)], ensure_ascii=False, separators=(",", ":")
        )
        pairs[response_id] = pair_id
        prompt_groups[response_id] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    if not pairs:
        raise ValueError("HaluEval examples are empty")
    return pairs, prompt_groups


def _record_value(record: Mapping[str, object] | object, name: str) -> object:
    return record[name] if isinstance(record, Mapping) else getattr(record, name)


def _record_optional(record: Mapping[str, object] | object, name: str, default: object = None) -> object:
    if isinstance(record, Mapping):
        return record.get(name, default)
    return getattr(record, name, default)


def _pair_id(record: Mapping[str, object] | object) -> str:
    source_id = _record_optional(record, "source_id")
    return str(source_id if source_id is not None else _record_value(record, "pair_id"))


def _legacy_protocol_summary(
    records: Sequence[Mapping[str, object] | object],
    *,
    selection: str,
    threshold: float | None,
) -> dict[str, object]:
    floors = sorted(
        {
            float(value)
            for record in records
            for value in (_record_optional(record, "legacy_tau"),)
            if value is not None
        }
    )
    return {
        "event": "input_protocol",
        "mode": "legacy_tau_censored",
        "selection": selection,
        "requested_threshold": threshold,
        "effective_threshold": (
            ("cache_floor" if threshold is None else threshold)
            if selection == "threshold"
            else None
        ),
        "legacy_attention_floor_values": floors,
        "supports_floor_0_01": bool(floors) and max(floors) <= 0.01,
    }


def _limit_complete_pairs(records: Sequence[Mapping[str, object] | object], limit: int | None) -> list[Mapping[str, object] | object]:
    grouped: dict[str, list[Mapping[str, object] | object]] = defaultdict(list)
    for record in records:
        grouped[str(_record_value(record, "pair_id"))].append(record)
    if any(len(candidates) != 2 for candidates in grouped.values()):
        raise ValueError("HaluEval selection requires exactly two complete candidates per pair")
    complete_pair_ids = [
        pair_id for pair_id, candidates in grouped.items()
        if all(_record_optional(candidate, "trace_path") is not None for candidate in candidates)
    ]
    if limit is None and len(complete_pair_ids) != len(grouped):
        raise ValueError("legacy graph preparation requires a complete graph and trace for every response")
    pair_ids = sorted(complete_pair_ids)
    if limit is not None:
        pair_ids = pair_ids[:limit]
    if not pair_ids:
        raise ValueError("no complete HaluEval graph-and-trace pairs are available")
    return [record for pair_id in pair_ids for record in grouped[pair_id]]


class _PreparedDataset(Sequence[object]):
    def __init__(self, records: Sequence[PreparedGraphRecord], *, mmap: bool = True) -> None:
        self.records, self.mmap = tuple(records), mmap
        self._validated: set[Path] = set()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int | slice) -> object | list[object]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        path = self.records[index].graph_path.resolve()
        graph = load_graph(path, device="cpu", mmap=self.mmap, validate=path not in self._validated)
        self._validated.add(path)
        return graph


def _model_dimensions(*datasets: _PreparedDataset) -> tuple[int, int]:
    first = next((dataset[0] for dataset in datasets if len(dataset)), None)
    if first is None:
        raise ValueError("no prepared HaluEval graphs are available")
    layers, heads = int(getattr(first, "num_layers")), int(getattr(first, "num_heads"))
    for dataset in datasets:
        for index in range(len(dataset)):
            graph = dataset[index]
            if int(getattr(graph, "num_layers")) != layers or int(getattr(graph, "num_heads")) != heads:
                raise ValueError("prepared HaluEval graphs disagree on layer/head dimensions")
    return layers, heads


def _record_manifest(record: PreparedGraphRecord) -> dict[str, object]:
    return {
        "response_id": str(_record_value(record, "response_id")), "pair_id": _pair_id(record),
        "dataset_split": _record_optional(record, "dataset_split"),
        "graph_path": str(_record_value(record, "graph_path")),
        "cache_path": str(_record_optional(record, "cache_path", "")),
    }


def _git_revision() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1], check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    value = completed.stdout.strip()
    return value if len(value) == 40 else None


def _git_dirty() -> bool | None:
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return bool(completed.stdout.strip())


def _file_sha256(path: Path) -> str | None:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return digest.hexdigest()


def _manifest_provenance(extraction_dir: Path) -> dict[str, object]:
    path = extraction_dir.expanduser().resolve() / "extraction_manifest.json"
    sha256 = _file_sha256(path)
    state: str | None = None
    if sha256 is not None:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"cannot read extraction manifest for provenance: {path}") from error
        if not isinstance(value, Mapping):
            raise ValueError("extraction manifest must be an object")
        raw_state = value.get("state")
        state = str(raw_state).strip().casefold() if raw_state is not None else None
    return {"path": str(path), "sha256": sha256, "state": state}


def _run_provenance(
    args: argparse.Namespace,
    *,
    records: Sequence[Mapping[str, object] | object],
) -> dict[str, object]:
    """Collect small, label-blind provenance fields for reproducibility."""

    manifest = _manifest_provenance(args.extraction_dir)
    taus = sorted(
        {
            float(value)
            for record in records
            for value in (_record_optional(record, "legacy_tau"),)
            if value is not None
        }
    )
    models = sorted(
        {
            str(value)
            for record in records
            for value in (_record_optional(record, "extractor_model_id"),)
            if value is not None and str(value).strip()
        }
    )
    fingerprints = sorted(
        {
            str(value)
            for record in records
            for value in (_record_optional(record, "extraction_fingerprint"),)
            if value is not None and str(value).strip()
        }
    )
    code_root = Path(__file__).resolve().parent
    code_digest = hashlib.sha256()
    for name in ("halueval.py", "halueval_cli.py"):
        code_digest.update(name.encode("utf-8"))
        code_digest.update((code_root / name).read_bytes())
    return {
        "source_run": str(args.source_run.expanduser().resolve()) if args.source_run else None,
        "extraction_dir": str(args.extraction_dir.expanduser().resolve()),
        "examples_path": str(args.examples.expanduser().resolve()),
        "examples_sha256": _file_sha256(args.examples.expanduser()),
        # This is intentionally a path only: labels are opened after scores freeze.
        "evaluation_labels_path": str(args.evaluation_labels.expanduser().resolve()),
        "manifest": manifest,
        "legacy_tau_values": taus,
        "extractor_model_ids": models,
        "extraction_fingerprints": {
            "count": len(fingerprints),
            "summary_sha256": hashlib.sha256("\x1f".join(fingerprints).encode("utf-8")).hexdigest(),
        },
        "code": {"git_revision": _git_revision(), "git_dirty": _git_dirty(), "sha256": code_digest.hexdigest()},
    }


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Run label-free preparation/training/scoring before opening labels."""

    if args.message_passing_steps < 0 or args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("invalid model or optimizer configuration")
    if args.validation_fraction <= 0.0 or args.test_fraction <= 0.0:
        raise ValueError("validation and test fractions must both be greater than zero for training")
    output = args.output_dir.expanduser().resolve()
    _fresh_output(output, resume=not args.no_prepare_resume)
    example_pairs, prompt_groups = _read_examples(args.examples.expanduser())
    discovered = discover_legacy_halueval_records(args.extraction_dir)
    discovered_ids = {str(_record_value(record, "response_id")) for record in discovered}
    if not discovered_ids.issubset(example_pairs):
        raise ValueError("legacy manifest contains response IDs absent from examples")
    if any(
        example_pairs[str(_record_value(record, "response_id"))]
        != str(_record_value(record, "pair_id"))
        for record in discovered
    ):
        raise ValueError("legacy manifest pair_id disagrees with examples")
    full_coverage = discovered_ids == set(example_pairs)
    complete = all(str(_record_value(record, "artifact_status")) == "full" for record in discovered)
    if args.require_complete_cache and (not full_coverage or not complete):
        raise ValueError("complete manifest/example coverage with graph and trace artifacts is required")
    selected = _limit_complete_pairs(discovered, args.limit_pairs)
    print(
        json.dumps(
            _legacy_protocol_summary(
                selected,
                selection=args.selection,
                threshold=args.threshold,
            ),
            sort_keys=True,
        ),
        flush=True,
    )
    if args.group_by_prompt:
        selected = [
            {
                **(dict(record) if isinstance(record, Mapping) else vars(record)),
                "group_id": prompt_groups[str(_record_value(record, "response_id"))],
            }
            for record in selected
        ]
    partitions_any = split_halueval_pairs(
        selected,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        group_by_prompt=args.group_by_prompt,
    )
    if any(not partitions_any[name] for name in ("train", "validation", "test")):
        raise ValueError("pair split produced an empty train, validation, or test partition")
    assignment = {
        str(_record_value(record, "response_id")): "test" if name == "test" else "train"
        for name, records in partitions_any.items() for record in records
    }
    prepared = prepare_legacy_halueval_graphs(
        selected, output_dir=output / "prepared", config=GraphBuildConfig(
            selection=args.selection, threshold=args.threshold, top_k=args.top_k,
            max_edges_per_target=args.max_edges_per_target, query_block=args.query_block,
        ), dataset_split_by_response=assignment, conversion_device=args.device,
        build_device=args.device,
        conversion_chunk_edges=args.conversion_chunk_edges,
        resume=not args.no_prepare_resume,
        progress_callback=_print_prepare_progress,
    )
    prepared_by_response = {record.response_id: record for record in prepared}
    partitions = {
        name: [prepared_by_response[str(_record_value(record, "response_id"))] for record in records]
        for name, records in partitions_any.items()
    }
    if any(not partitions[name] for name in ("train", "validation", "test")):
        raise ValueError("pair split produced an empty train, validation, or test partition")
    _write_json(output / "splits.json", {
        "schema": "halueval-paired-response-splits-v1", "validation_fraction": args.validation_fraction,
        "test_fraction": args.test_fraction, "seed": args.seed,
        "mode": "prompt_group" if args.group_by_prompt else "compatibility_pair_hash",
        "group_by_prompt": args.group_by_prompt,
        "counts": {name: len(records) for name, records in partitions.items()},
        "pair_ids": {name: sorted({_pair_id(record) for record in records}) for name, records in partitions.items()},
        "partitions": {name: [_record_manifest(record) for record in records] for name, records in partitions.items()},
    })
    train_graphs, validation_graphs, test_graphs = (_PreparedDataset(partitions[name]) for name in ("train", "validation", "test"))
    layers, heads = _model_dimensions(train_graphs, validation_graphs, test_graphs)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = RelationAwareMaskGAE(num_layers=layers, num_heads=heads, embedding_dim=args.embedding_dim, message_passing_steps=args.message_passing_steps, dropout=args.dropout).to(args.device)
    training_config = TrainingConfig(
        epochs=args.epochs, patience=args.patience, learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        edge_mask_rate=args.edge_mask_rate, node_mask_rate=args.node_mask_rate, channel_drop_rate=args.channel_drop_rate,
        support_weight=args.support_weight, attention_weight=args.attention_weight,
        distribution_weight=args.distribution_weight, node_weight=args.node_weight,
        max_support_edges=args.max_support_edges, max_weight_traces=args.max_weight_traces,
        max_distribution_groups=args.max_distribution_groups, decoder_chunk_size=args.decoder_chunk_size, seed=args.seed,
    )
    training = train_relation_mae(model, train_graphs=train_graphs, validation_graphs=validation_graphs, config=training_config, output_dir=output / "training")
    _write_json(output / "training" / "history.json", training.history)
    predictions, mixture = score_graphs(
        model, fit_graphs=train_graphs, score_graphs=test_graphs, num_views=args.num_score_views,
        include_reconstruction=not args.embedding_only_scoring, max_support_edges=args.max_support_edges,
        max_weight_traces=args.max_weight_traces, max_distribution_groups=args.max_distribution_groups,
        decoder_chunk_size=args.decoder_chunk_size, seed=args.seed,
    )
    response_path = output / "test.response_predictions.jsonl"
    _write_jsonl(response_path, predictions)
    _write_json(output / "response_mixture.json", mixture.to_dict())
    evaluation_path: Path | None = None
    if not args.skip_evaluation:
        # The immutable response score stream is written before this sole label read.
        labels = load_halueval_response_labels(args.evaluation_labels)
        test_ids = {record.response_id for record in partitions["test"]}
        if not test_ids.issubset(labels):
            raise ValueError("evaluation labels do not cover the held-out response subset")
        test_labels = {response_id: labels[response_id] for response_id in test_ids}
        evaluation = evaluate_halueval_predictions(
            predictions, test_labels,
            {str(_record_value(record, "response_id")): _pair_id(record) for record in partitions["test"]},
            response_length_by_id={
                response_id: prepared_by_response[response_id].num_response_nodes
                for response_id in test_ids
            },
            seed=args.seed,
        )
        prompt_length_evaluation = evaluate_halueval_predictions(
            predictions,
            test_labels,
            {str(_record_value(record, "response_id")): _pair_id(record) for record in partitions["test"]},
            response_length_by_id={
                response_id: prepared_by_response[response_id].num_nodes
                - prepared_by_response[response_id].num_response_nodes
                for response_id in test_ids
            },
            seed=args.seed,
        )
        # Response tokens are the formal length-only baseline.  Prompt-token
        # length is persisted only as a separately named diagnostic.
        if "length_only" in prompt_length_evaluation:
            evaluation["prompt_length_only_diagnostic"] = prompt_length_evaluation["length_only"]
        evaluation_path = output / "evaluation.json"
        _write_json(evaluation_path, evaluation)
    scope = "legacy_cache_complete" if complete and full_coverage and args.limit_pairs is None else "legacy_cache_partial_pilot"
    result: dict[str, object] = {
        "schema": "halueval-attention-graph-run-v1", "status": "complete", "experiment_scope": scope,
        "labels_read_during": "never" if args.skip_evaluation else "evaluation_only",
        "output_dir": str(output), "prepared_graph_dir": str(output / "prepared" / "graphs"),
        "partition_counts": {name: len(records) for name, records in partitions.items()},
        "splits": str(output / "splits.json"), "training_history": str(output / "training" / "history.json"),
        "checkpoint": str(training.checkpoint_path), "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss, "response_predictions": str(response_path),
        "response_mixture": str(output / "response_mixture.json"), "evaluation": str(evaluation_path) if evaluation_path else None,
        "core_metrics": {
            key: evaluation[key]
            for key in ("auroc", "average_precision", "paired_accuracy", "positive_fraction")
            if not args.skip_evaluation and key in evaluation
        },
        "configuration": {
            "graph": asdict(GraphBuildConfig(selection=args.selection, threshold=args.threshold, top_k=args.top_k, max_edges_per_target=args.max_edges_per_target, query_block=args.query_block)),
            "model": {"embedding_dim": args.embedding_dim, "message_passing_steps": args.message_passing_steps, "dropout": args.dropout},
            "training": asdict(training_config),
            "scoring": {"num_score_views": args.num_score_views, "embedding_only": args.embedding_only_scoring},
            "preparation": {"conversion_chunk_edges": args.conversion_chunk_edges, "resume": not args.no_prepare_resume},
            "split": {"mode": "prompt_group" if args.group_by_prompt else "compatibility_pair_hash", "group_by_prompt": args.group_by_prompt},
            "limit_pairs": args.limit_pairs,
            "require_complete_cache": args.require_complete_cache,
        },
        "provenance": _run_provenance(args, records=selected),
    }
    _write_json(output / "run.json", result)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    metrics = result.get("core_metrics", {})
    evaluation = result.get("evaluation")
    length = "n/a"
    if evaluation:
        try:
            length = str(json.loads(Path(str(evaluation)).read_text(encoding="utf-8")).get("length_only", {}).get("auroc", "n/a"))
        except (OSError, json.JSONDecodeError):
            length = "n/a"
    print(
        "AUROC={auroc} AUPRC={auprc} paired_accuracy={paired} "
        "response_length_AUROC={length} output_dir={output}".format(
            auroc=metrics.get("auroc", "n/a"), auprc=metrics.get("average_precision", "n/a"),
            paired=metrics.get("paired_accuracy", "n/a"), length=length,
            output=result["output_dir"],
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
