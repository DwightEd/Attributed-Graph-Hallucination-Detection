"""One-command frontend for the formal RAGTruth attention-graph experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .ablation import (
    collapse_relations,
    mean_attention_heads,
    relation_preserving_source_shuffle,
)
from .data import (
    PreparedGraphRecord,
    audit_attention_cache,
    discover_attention_cache,
    load_graph,
    official_partitions,
    prepare_graphs,
)
from .evaluate import (
    evaluate_predictions,
    evaluate_sentence_predictions,
    load_evaluation_labels,
)
from .graph import GraphBuildConfig
from .model import RelationAwareMaskGAE
from .ragtruth import prepare_ragtruth_sentence_scores
from .train import (
    TrainingConfig,
    score_graphs,
    score_tokens,
    train_relation_mae,
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    normalized = str(value).strip().casefold()
    if normalized in {"none", "uncapped", "unbounded", "0"}:
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
    parser = argparse.ArgumentParser(
        description="Attention-only relation-aware RAGTruth graph experiments"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run",
        help=(
            "prepare graphs, make official source splits, train label-blind, "
            "fit mixtures, score held-out graphs/tokens, and evaluate last"
        ),
    )
    run.add_argument("--cache-root", "--cache", type=Path, required=True)
    run.add_argument("--output-dir", "--output", type=Path, required=True)
    run.add_argument(
        "--graph-dir",
        type=Path,
        help="optional shared prepared-graph directory for controlled ablations",
    )
    run.add_argument("--device", default="cuda")
    run.add_argument(
        "--selection",
        choices=("threshold", "global_topk", "typed_topk"),
        default="threshold",
    )
    run.add_argument("--threshold", type=float)
    run.add_argument("--top-k", type=_positive_int, default=8)
    run.add_argument(
        "--max-edges-per-target",
        type=_optional_positive_int,
        default=64,
        help=(
            "threshold-mode safety cap; use 'none' or 0 for uncapped threshold "
            "support after checking graph-size diagnostics"
        ),
    )
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
    run.add_argument(
        "--embedding-only-scoring",
        action="store_true",
        help="exclude reconstruction energies from both response/token mixtures",
    )
    run.add_argument("--embedding-dim", type=_positive_int, default=128)
    run.add_argument("--message-passing-steps", type=int, default=2)
    run.add_argument(
        "--graph-transform",
        choices=("none", "source_shuffle", "collapse_relations", "mean_heads"),
        default="none",
        help="label-free graph-dependence/channel ablation applied after loading",
    )
    run.add_argument("--dropout", type=_rate, default=0.10)
    run.add_argument("--max-support-edges", type=_positive_int, default=8_192)
    run.add_argument("--max-weight-traces", type=_positive_int, default=65_536)
    run.add_argument(
        "--max-distribution-groups", type=_positive_int, default=512
    )
    run.add_argument("--decoder-chunk-size", type=_positive_int, default=16_384)
    run.add_argument("--validation-fraction", type=_fraction, default=0.20)
    run.add_argument("--num-score-views", type=_positive_int, default=4)
    run.add_argument("--token-mask-stride", type=_positive_int, default=8)
    run.add_argument("--token-edge-mask-rate", type=_rate, default=0.50)
    run.add_argument("--max-fit-tokens", type=_positive_int, default=100_000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument(
        "--require-complete-cache",
        action="store_true",
        help=(
            "fail unless both official split manifests certify their exact "
            "attention-file inventories; otherwise partial-cache pilots remain allowed"
        ),
    )
    run.add_argument(
        "--limit",
        type=_positive_int,
        help=(
            "SMOKE TEST ONLY: select at most N cache files independently from "
            "each official split before graph preparation; never use for a "
            "reported full result"
        ),
    )
    run.add_argument(
        "--sentence-output",
        type=Path,
        help=(
            "optional sentence JSONL path; defaults to "
            "<output>/test.sentence_predictions.jsonl when --responses, "
            "--sources, and --tokenizer are all supplied"
        ),
    )
    run.add_argument("--responses", type=Path, help="RAGTruth response.jsonl")
    run.add_argument("--sources", type=Path, help="RAGTruth source_info.jsonl")
    run.add_argument(
        "--tokenizer",
        help="exact extraction-model tokenizer path or Transformers identifier",
    )
    run.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="SMOKE TEST ONLY: freeze label-free scores without reading labels",
    )
    run.add_argument("--no-resume", action="store_true")
    run.set_defaults(handler=run_pipeline)
    return parser


class PreparedGraphDataset(Sequence[object]):
    """Mmap-backed graph sequence that owns no full in-memory graph list."""

    def __init__(
        self,
        records: Sequence[PreparedGraphRecord],
        *,
        mmap: bool = True,
        graph_transform: str = "none",
        seed: int = 42,
    ) -> None:
        self._records = tuple(records)
        self._mmap = bool(mmap)
        self._validated_paths: set[Path] = set()
        self._graph_transform = str(graph_transform)
        self._seed = int(seed)
        if self._graph_transform not in {
            "none",
            "source_shuffle",
            "collapse_relations",
            "mean_heads",
        }:
            raise ValueError(f"unsupported graph transform: {self._graph_transform}")

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int | slice) -> object | list[object]:
        if isinstance(index, slice):
            return [self[position] for position in range(*index.indices(len(self)))]
        record = self._records[index]
        path = record.graph_path.resolve()
        validate = path not in self._validated_paths
        graph = load_graph(
            path,
            device="cpu",
            mmap=self._mmap,
            validate=validate,
        )
        self._validated_paths.add(path)
        if self._graph_transform == "source_shuffle":
            identity = f"{graph.source_id}\0{graph.sample_id}\0{self._seed}"
            transform_seed = int.from_bytes(
                hashlib.sha256(identity.encode("utf-8")).digest()[:8],
                byteorder="little",
                signed=False,
            )
            graph = relation_preserving_source_shuffle(
                graph,
                generator=torch.Generator().manual_seed(transform_seed),
            )
        elif self._graph_transform == "collapse_relations":
            graph = collapse_relations(graph)
        elif self._graph_transform == "mean_heads":
            graph = mean_attention_heads(graph)
        return graph


def _link_input(source: Path, destination: Path) -> None:
    if destination.exists():
        try:
            if os.path.samefile(source, destination):
                return
        except OSError:
            pass
        raise FileExistsError(f"smoke-cache destination conflicts: {destination}")
    try:
        os.link(source, destination)
    except OSError as hardlink_error:
        try:
            os.symlink(source, destination)
        except OSError as symlink_error:
            raise RuntimeError(
                "could not create a zero-copy smoke-cache view with either "
                f"hard links or symbolic links: {destination}"
            ) from symlink_error
        if not destination.is_file():
            raise RuntimeError(
                f"created smoke-cache link is not readable: {destination}"
            ) from hardlink_error


def create_smoke_cache(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    limit: int,
) -> Path:
    """Create a zero-copy cache view containing at most ``limit`` files/split."""

    if limit < 1:
        raise ValueError("smoke limit must be positive")
    source_root = Path(cache_root).expanduser().resolve()
    selected: dict[str, list[Path]] = {}
    for split in ("train", "test"):
        records = discover_attention_cache(source_root, splits=(split,))
        selected[split] = [record.path for record in records[:limit]]
        if not selected[split]:
            raise ValueError(f"smoke selection found no {split} cache files")
    identity = "\n".join(
        f"{split}:{path.resolve()}" for split in ("train", "test") for path in selected[split]
    )
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:12]
    smoke_root = (
        Path(output_dir).expanduser().resolve()
        / "_smoke_cache"
        / f"limit_{limit}_{digest}"
    )
    for split, paths in selected.items():
        destination = smoke_root / split
        destination.mkdir(parents=True, exist_ok=True)
        for source in paths:
            _link_input(source, destination / source.name)
    return smoke_root


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _write_jsonl(path: Path, records: Sequence[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n"
            )
    os.replace(temporary, path)


def _require_partitions(
    partitions: dict[str, list[PreparedGraphRecord]],
    *,
    smoke_limit: int | None,
) -> None:
    empty = [name for name in ("train", "validation", "test") if not partitions[name]]
    if not empty:
        return
    suffix = (
        " Increase --limit so the smoke sample contains multiple train source IDs."
        if smoke_limit is not None
        else ""
    )
    raise ValueError(
        "official source split produced empty partition(s): "
        + ", ".join(empty)
        + "."
        + suffix
    )


def _model_dimensions(*datasets: PreparedGraphDataset) -> tuple[int, int]:
    if not datasets or not any(len(dataset) for dataset in datasets):
        raise ValueError("no graphs are available to initialize the model")
    first = next(dataset[0] for dataset in datasets if len(dataset))
    layers = int(getattr(first, "num_layers"))
    heads = int(getattr(first, "num_heads"))
    mismatched: list[str] = []
    for partition_index, dataset in enumerate(datasets):
        for graph_index in range(len(dataset)):
            graph = dataset[graph_index]
            if (
                int(getattr(graph, "num_layers")) != layers
                or int(getattr(graph, "num_heads")) != heads
            ):
                mismatched.append(f"{partition_index}:{graph_index}")
    if mismatched:
        raise ValueError(
            "prepared graphs disagree on layer/head dimensions at "
            "partition:index positions: "
            + ", ".join(mismatched[:10])
        )
    return layers, heads


def _split_record(record: PreparedGraphRecord) -> dict[str, object]:
    return {
        "source_id": record.source_id,
        "sample_id": record.sample_id,
        "response_id": record.response_id,
        "dataset_split": record.dataset_split,
        "cache_path": str(record.cache_path),
        "graph_path": str(record.graph_path),
        "num_nodes": record.num_nodes,
        "num_response_nodes": record.num_response_nodes,
        "num_edges": record.num_edges,
        "num_rp_edges": record.num_rp_edges,
        "num_rr_edges": record.num_rr_edges,
        "num_traces": record.num_traces,
    }


def _split_manifest(
    partitions: dict[str, list[PreparedGraphRecord]],
    *,
    validation_fraction: float,
    seed: int,
) -> dict[str, object]:
    return {
        "schema": "ragtruth-attention-graph-official-splits-v1",
        "policy": "official test held out; official train grouped by source_id",
        "validation_fraction": validation_fraction,
        "seed": seed,
        "counts": {name: len(partitions[name]) for name in partitions},
        "partitions": {
            name: [_split_record(record) for record in partitions[name]]
            for name in ("train", "validation", "test")
        },
    }


def _load_tokenizer(reference: str) -> object:
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers is required when sentence scoring is enabled"
        ) from error
    tokenizer = AutoTokenizer.from_pretrained(reference, use_fast=True)
    if not bool(getattr(tokenizer, "is_fast", False)):
        raise ValueError("sentence scoring requires a fast tokenizer with offsets")
    return tokenizer


def _require_fresh_output(path: Path) -> None:
    """Refuse to mix checkpoints, scores, or labels from different runs."""

    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"run output is not empty: {path}; choose a new empty output directory"
        )
    path.mkdir(parents=True, exist_ok=True)


def _code_revision() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    revision = completed.stdout.strip()
    return revision if len(revision) == 40 else None


def _experiment_scope(
    cache_inventory: dict[str, dict[str, object]],
    *,
    smoke_limit: int | None,
) -> str:
    if smoke_limit is not None:
        return "smoke_test"
    if all(bool(cache_inventory[split]["complete"]) for split in ("train", "test")):
        return "official_complete_cache"
    return "partial_cache_pilot"


def run_pipeline(args: argparse.Namespace) -> dict[str, object]:
    """Execute preparation through held-out post-hoc evaluation in order."""

    if args.message_passing_steps < 0:
        raise ValueError("message-passing-steps cannot be negative")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    sentence_inputs = (args.responses, args.sources, args.tokenizer)
    if any(value is not None for value in sentence_inputs) and not all(
        value is not None for value in sentence_inputs
    ):
        raise ValueError(
            "sentence scoring requires --responses, --sources, and --tokenizer "
            "as a complete set"
        )
    sentence_enabled = all(value is not None for value in sentence_inputs)
    if args.sentence_output is not None and not sentence_enabled:
        raise ValueError(
            "--sentence-output requires --responses, --sources, and --tokenizer"
        )
    if args.sentence_output is not None and args.sentence_output.expanduser().exists():
        raise FileExistsError(
            "sentence output already exists; choose a new path to avoid mixing runs: "
            f"{args.sentence_output.expanduser().resolve()}"
        )
    if args.skip_evaluation and args.limit is None:
        raise ValueError("--skip-evaluation is restricted to --limit smoke runs")

    output = args.output_dir.expanduser().resolve()
    _require_fresh_output(output)
    cache_root = args.cache_root.expanduser().resolve()
    cache_inventory = audit_attention_cache(cache_root, splits=("train", "test"))
    incomplete_splits = [
        split
        for split in ("train", "test")
        if not bool(cache_inventory[split]["complete"])
    ]
    if args.require_complete_cache and incomplete_splits:
        raise RuntimeError(
            "complete attention cache required, but manifest/inventory is partial for: "
            + ", ".join(incomplete_splits)
        )
    scope = _experiment_scope(cache_inventory, smoke_limit=args.limit)
    print(
        json.dumps(
            {
                "event": "attention_cache_audit",
                "experiment_scope": scope,
                "cache_inventory": cache_inventory,
                "warning": (
                    "available_cache_is_a_partial_pilot_not_a_complete_official_run"
                    if scope == "partial_cache_pilot"
                    else None
                ),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    preparation_cache = cache_root
    if args.limit is not None:
        preparation_cache = create_smoke_cache(
            cache_root, output, limit=args.limit
        )
        print(
            json.dumps(
                {
                    "event": "smoke_limit",
                    "warning": "limit_is_smoke_only_and_applied_per_official_split",
                    "limit_per_split": args.limit,
                    "cache_view": str(preparation_cache),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    graph_config = GraphBuildConfig(
        selection=args.selection,
        threshold=args.threshold,
        top_k=args.top_k,
        max_edges_per_target=args.max_edges_per_target,
        query_block=args.query_block,
    )
    graph_output = (
        args.graph_dir.expanduser().resolve()
        if args.graph_dir is not None
        else output / "graphs"
    )
    records = prepare_graphs(
        cache_root=preparation_cache,
        output_dir=graph_output,
        config=graph_config,
        splits=("train", "test"),
        build_device=args.device,
        resume=not args.no_resume,
    )
    partitions = official_partitions(
        records,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    _require_partitions(partitions, smoke_limit=args.limit)
    _write_json(
        output / "splits.json",
        _split_manifest(
            partitions,
            validation_fraction=args.validation_fraction,
            seed=args.seed,
        ),
    )

    dataset_arguments = {
        "graph_transform": args.graph_transform,
        "seed": args.seed,
    }
    train_graphs = PreparedGraphDataset(partitions["train"], **dataset_arguments)
    validation_graphs = PreparedGraphDataset(
        partitions["validation"], **dataset_arguments
    )
    test_graphs = PreparedGraphDataset(partitions["test"], **dataset_arguments)
    num_layers, num_heads = _model_dimensions(
        train_graphs, validation_graphs, test_graphs
    )
    # Model initialization must be identical across same-seed ablation processes.
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = RelationAwareMaskGAE(
        num_layers=num_layers,
        num_heads=num_heads,
        embedding_dim=args.embedding_dim,
        message_passing_steps=args.message_passing_steps,
        dropout=args.dropout,
    ).to(args.device)
    training_config = TrainingConfig(
        epochs=args.epochs,
        patience=args.patience,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        edge_mask_rate=args.edge_mask_rate,
        node_mask_rate=args.node_mask_rate,
        channel_drop_rate=args.channel_drop_rate,
        support_weight=args.support_weight,
        attention_weight=args.attention_weight,
        distribution_weight=args.distribution_weight,
        node_weight=args.node_weight,
        max_support_edges=args.max_support_edges,
        max_weight_traces=args.max_weight_traces,
        max_distribution_groups=args.max_distribution_groups,
        decoder_chunk_size=args.decoder_chunk_size,
        seed=args.seed,
    )
    training = train_relation_mae(
        model,
        train_graphs=train_graphs,
        validation_graphs=validation_graphs,
        config=training_config,
        output_dir=output / "training",
    )
    _write_json(output / "training" / "history.json", training.history)

    response_records, response_mixture = score_graphs(
        model,
        fit_graphs=train_graphs,
        score_graphs=test_graphs,
        num_views=args.num_score_views,
        include_reconstruction=not args.embedding_only_scoring,
        max_support_edges=args.max_support_edges,
        max_weight_traces=args.max_weight_traces,
        max_distribution_groups=args.max_distribution_groups,
        decoder_chunk_size=args.decoder_chunk_size,
        seed=args.seed,
    )
    response_path = output / "test.response_predictions.jsonl"
    _write_jsonl(response_path, response_records)
    _write_json(output / "response_mixture.json", response_mixture.to_dict())

    token_records, token_mixture = score_tokens(
        model,
        fit_graphs=train_graphs,
        score_graphs=test_graphs,
        mask_stride=args.token_mask_stride,
        edge_mask_rate=args.token_edge_mask_rate,
        max_fit_tokens=args.max_fit_tokens,
        include_reconstruction=not args.embedding_only_scoring,
        max_support_edges=args.max_support_edges,
        max_weight_traces=args.max_weight_traces,
        max_distribution_groups=args.max_distribution_groups,
        decoder_chunk_size=args.decoder_chunk_size,
        seed=args.seed,
    )
    token_path = output / "test.token_predictions.jsonl"
    _write_jsonl(token_path, token_records)
    _write_json(output / "token_mixture.json", token_mixture.to_dict())

    sentence_path: Path | None = None
    sentence_records: list[dict[str, object]] | None = None
    if sentence_enabled:
        sentence_path = (
            args.sentence_output.expanduser().resolve()
            if args.sentence_output is not None
            else output / "test.sentence_predictions.jsonl"
        )
        tokenizer = _load_tokenizer(str(args.tokenizer))
        sentence_records = prepare_ragtruth_sentence_scores(
            token_records,
            [record.cache_path for record in partitions["test"]],
            response_path=args.responses,
            source_path=args.sources,
            tokenizer=tokenizer,
            output_path=sentence_path,
        )

    test_attention_paths = [record.cache_path for record in partitions["test"]]
    evaluation_path: Path | None = None
    if not args.skip_evaluation:
        # This is the sole pipeline stage that requests y_token. Predictions,
        # mixtures, and optional label-free sentence scores are frozen first.
        labels = load_evaluation_labels(test_attention_paths)
        evaluation = evaluate_predictions(response_records, token_records, labels)
        if sentence_records is not None:
            evaluation = {
                **evaluation,
                "sentence": evaluate_sentence_predictions(sentence_records, labels),
            }
        evaluation_path = output / "evaluation.json"
        _write_json(evaluation_path, evaluation)

    result: dict[str, object] = {
        "schema": "ragtruth-attention-graph-run-v2",
        "status": "complete",
        "experiment_scope": scope,
        "cache_inventory": cache_inventory,
        "require_complete_cache": bool(args.require_complete_cache),
        "cache_root": str(cache_root),
        "preparation_cache_root": str(preparation_cache),
        "output_dir": str(output),
        "device": args.device,
        "smoke_limit_per_split": args.limit,
        "selection": args.selection,
        "graph_transform": args.graph_transform,
        "prepared_graph_dir": str(graph_output),
        "graph_build_limits": {
            "threshold": args.threshold,
            "top_k": args.top_k,
            "query_block": args.query_block,
            "max_edges_per_target": args.max_edges_per_target,
        },
        "training_limits": {
            "max_support_edges": args.max_support_edges,
            "max_weight_traces": args.max_weight_traces,
            "max_distribution_groups": args.max_distribution_groups,
            "decoder_chunk_size": args.decoder_chunk_size,
        },
        "loss_weights": {
            "support": args.support_weight,
            "attention": args.attention_weight,
            "distribution": args.distribution_weight,
            "node": args.node_weight,
        },
        "configuration": {
            "graph": asdict(graph_config),
            "training": asdict(training_config),
            "model": {
                "embedding_dim": args.embedding_dim,
                "message_passing_steps": args.message_passing_steps,
                "dropout": args.dropout,
                "num_layers": num_layers,
                "num_heads": num_heads,
            },
            "partition": {
                "validation_fraction": args.validation_fraction,
                "seed": args.seed,
            },
            "scoring": {
                "num_score_views": args.num_score_views,
                "token_mask_stride": args.token_mask_stride,
                "token_edge_mask_rate": args.token_edge_mask_rate,
                "max_fit_tokens": args.max_fit_tokens,
                "embedding_only": bool(args.embedding_only_scoring),
            },
            "sentence": {
                "enabled": sentence_enabled,
                "responses": str(args.responses) if args.responses else None,
                "sources": str(args.sources) if args.sources else None,
                "tokenizer": str(args.tokenizer) if args.tokenizer else None,
            },
        },
        "provenance": {
            "git_revision": _code_revision(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "scoring_features": (
            "embedding_only"
            if args.embedding_only_scoring
            else "embedding_plus_reconstruction"
        ),
        "partition_counts": {
            name: len(partitions[name]) for name in ("train", "validation", "test")
        },
        "splits": str(output / "splits.json"),
        "training_history": str(output / "training" / "history.json"),
        "best_epoch": training.best_epoch,
        "best_validation_loss": training.best_validation_loss,
        "checkpoint": str(training.checkpoint_path),
        "response_predictions": str(response_path),
        "response_mixture": str(output / "response_mixture.json"),
        "token_predictions": str(token_path),
        "token_mixture": str(output / "token_mixture.json"),
        "sentence_predictions": str(sentence_path) if sentence_path else None,
        "evaluation": str(evaluation_path) if evaluation_path else None,
        "labels_read_during": (
            "never" if args.skip_evaluation else "evaluation_only"
        ),
    }
    _write_json(output / "run.json", result)
    print(json.dumps({"event": "run_complete", **result}, sort_keys=True), flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.handler(args)
    return 0


__all__ = [
    "PreparedGraphDataset",
    "build_parser",
    "create_smoke_cache",
    "main",
    "run_pipeline",
]


if __name__ == "__main__":
    raise SystemExit(main())
