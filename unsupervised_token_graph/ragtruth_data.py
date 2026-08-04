"""Label-free RAGTruth graph storage, batching, and data-partition primitives."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .ragtruth_graph import build_compact_topk_graph, load_attention_sample

_FORBIDDEN_KEYS = {
    "gold", "is_correct", "is_hallucinated", "label", "labels", "target",
    "token_labels", "y", "y_token",
}


def validate_label_free(value: object, path: str = "graph") -> None:
    """Recursively reject supervision before a graph reaches the optimizer."""

    if isinstance(value, Mapping):
        for key, nested in value.items():
            name = str(key).casefold()
            if name in _FORBIDDEN_KEYS or "label" in name:
                raise ValueError(f"label/evaluation field {path}.{key} is forbidden")
            validate_label_free(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            validate_label_free(nested, f"{path}[{index}]")


def validate_compact_graph(graph: object) -> None:
    """Validate a persisted graph before it enters GPU residency or training."""

    if not isinstance(graph, Mapping):
        raise TypeError("compact graph must be a mapping")
    validate_label_free(graph)
    if graph.get("schema_version") != "ragtruth_typed_topk_v3":
        raise ValueError("compact graph schema must be ragtruth_typed_topk_v3")
    required = {
        "source_id", "sample_id", "response_idx", "x", "node_context",
        "edge_index", "edge_attr", "edge_type", "response_mask",
        "neighbor_mean_target", "neighbor_log_variance_target",
        "route_stats_target", "graph_config",
    }
    missing = required.difference(graph)
    if missing:
        raise ValueError(f"compact graph is missing fields: {sorted(missing)}")
    if not str(graph["source_id"]).strip() or not str(graph["sample_id"]).strip():
        raise ValueError("compact graph source_id and sample_id must be non-empty")
    config = graph["graph_config"]
    if not isinstance(config, Mapping):
        raise TypeError("compact graph graph_config must be a mapping")
    if not str(config.get("cache_format", "")).strip():
        raise ValueError("compact graph graph_config.cache_format is required")
    if config.get("cache_format") == "formal_sparse_csr" and not isinstance(
        config.get("raw_identity"), Mapping
    ):
        raise ValueError("formal compact graph graph_config.raw_identity is required")
    x = torch.as_tensor(graph["x"])
    context = torch.as_tensor(graph["node_context"])
    edge_index = torch.as_tensor(graph["edge_index"]).long()
    edge_attr = torch.as_tensor(graph["edge_attr"])
    edge_type = torch.as_tensor(graph["edge_type"]).long()
    response = torch.as_tensor(graph["response_mask"]).bool()
    if x.ndim != 2 or context.shape != (len(x), 3):
        raise ValueError("invalid compact node/context shape")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edges]")
    edge_count = edge_index.shape[1]
    if edge_attr.shape != (edge_count, 8) or edge_type.shape != (edge_count,):
        raise ValueError("invalid compact edge feature/type shape")
    if response.shape != (len(x),) or not bool(response.any()):
        raise ValueError("response_mask must select response nodes")
    if edge_count and (
        bool((edge_index < 0).any())
        or bool((edge_index >= len(x)).any())
        or not bool((edge_index[0] < edge_index[1]).all())
        or not bool(response[edge_index[1]].all())
        or bool(((edge_type < 0) | (edge_type > 1)).any())
    ):
        raise ValueError("compact edges must be causal typed response in-edges")
    expected_neighborhood = (len(x), 2, x.shape[1])
    if torch.as_tensor(graph["neighbor_mean_target"]).shape != expected_neighborhood:
        raise ValueError("invalid neighbor_mean_target shape")
    if torch.as_tensor(graph["neighbor_log_variance_target"]).shape != expected_neighborhood:
        raise ValueError("invalid neighbor_log_variance_target shape")
    if torch.as_tensor(graph["route_stats_target"]).shape != (len(x), 2, 4):
        raise ValueError("invalid route_stats_target shape")


def _tensor_device(graph: Mapping[str, object]) -> torch.device:
    value = graph.get("x")
    if not isinstance(value, torch.Tensor):
        raise TypeError("graph.x must be a tensor")
    return value.device


def collate_graphs(graphs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    """Concatenate compact graphs without padding and offset their COO edges."""

    if not graphs:
        raise ValueError("at least one graph is required")
    for graph in graphs:
        validate_label_free(graph)
    device = _tensor_device(graphs[0])
    if any(_tensor_device(graph) != device for graph in graphs[1:]):
        raise ValueError("all graphs in a batch must reside on one device")
    node_counts = [int(graph["x"].shape[0]) for graph in graphs]
    offsets = torch.tensor([0, *np.cumsum(node_counts).tolist()], dtype=torch.long, device=device)
    edges = []
    for graph, offset in zip(graphs, offsets[:-1]):
        edge_index = torch.as_tensor(graph["edge_index"], device=device).long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape (2, edges)")
        edges.append(edge_index + offset)
    batch: dict[str, object] = {
        "x": torch.cat([graph["x"] for graph in graphs], dim=0),
        "edge_index": torch.cat(edges, dim=1),
        "edge_attr": torch.cat([graph["edge_attr"] for graph in graphs], dim=0),
        "edge_type": torch.cat([graph["edge_type"] for graph in graphs], dim=0).long(),
        "response_mask": torch.cat([graph["response_mask"] for graph in graphs], dim=0).bool(),
        "graph_ptr": offsets,
        "source_id": [str(graph["source_id"]) for graph in graphs],
        "sample_id": [str(graph["sample_id"]) for graph in graphs],
    }
    original_indices = ["original_idx" in graph for graph in graphs]
    if any(original_indices) and not all(original_indices):
        raise ValueError("original_idx must be present in every graph or absent from every graph")
    if all(original_indices):
        batch["original_idx"] = [int(graph["original_idx"]) for graph in graphs]
    for name in (
        "node_context", "token_ids", "neighbor_mean_target",
        "neighbor_log_variance_target", "route_stats_target",
    ):
        present = [name in graph for graph in graphs]
        if any(present) and not all(present):
            raise ValueError(f"optional graph field {name!r} must be present in every graph")
        if all(present):
            batch[name] = torch.cat([graph[name] for graph in graphs], dim=0)
    return batch


def _stable_key(source_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\x1f{source_id}".encode()).hexdigest()


def split_paths_by_source(
    records: Sequence[Mapping[str, object]], *, train_fraction: float,
    validation_fraction: float, seed: int,
) -> dict[str, list[Path]]:
    """Make a deterministic group split with no ``source_id`` leakage."""

    if not 0.0 < train_fraction < 1.0 or not 0.0 < validation_fraction < 1.0:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must be below one")
    grouped: dict[str, list[Path]] = {}
    for record in records:
        grouped.setdefault(str(record["source_id"]), []).append(Path(record["path"]))
    if len(grouped) < 3:
        raise ValueError("at least three source groups are required")
    sources = sorted(grouped, key=lambda value: _stable_key(value, seed))
    train_count = max(1, round(len(sources) * train_fraction))
    validation_count = max(1, round(len(sources) * validation_fraction))
    while train_count + validation_count >= len(sources):
        if train_count >= validation_count and train_count > 1:
            train_count -= 1
        elif validation_count > 1:
            validation_count -= 1
        else:
            raise ValueError("source groups cannot form three non-empty partitions")
    assignments = {
        "train": sources[:train_count],
        "validation": sources[train_count:train_count + validation_count],
        "test": sources[train_count + validation_count:],
    }
    return {name: sorted(path for source in source_ids for path in grouped[source])
            for name, source_ids in assignments.items()}


def split_paths_by_official_split(
    records: Sequence[Mapping[str, object]], *, validation_fraction: float, seed: int,
) -> dict[str, list[Path]]:
    """Hold out official RAGTruth test and split only official train by source.

    ``dataset_split`` is label-free corpus metadata.  No hallucination label is
    accepted or inspected here, and all responses sharing one source remain in
    the same train/validation partition.
    """

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between zero and one")
    grouped: dict[str, dict[str, list[Path]]] = {"train": {}, "test": {}}
    for record in records:
        split = str(record.get("dataset_split", "")).strip().casefold()
        if split not in grouped:
            raise ValueError(
                "official split requires dataset_split=train/test on every compact graph"
            )
        source_id = str(record["source_id"])
        grouped[split].setdefault(source_id, []).append(Path(record["path"]))
    train_sources, test_sources = set(grouped["train"]), set(grouped["test"])
    if not test_sources:
        raise ValueError(
            "official test cache is absent; compact the cache root containing train/ and test/"
        )
    overlap = train_sources.intersection(test_sources)
    if overlap:
        raise ValueError(
            f"source leakage across official train/test splits: {sorted(overlap)[:8]}"
        )
    if len(train_sources) < 2:
        raise ValueError("at least two official-train source groups are required")
    ordered = sorted(train_sources, key=lambda value: _stable_key(value, seed))
    validation_count = min(
        len(ordered) - 1, max(1, round(len(ordered) * validation_fraction))
    )
    validation_sources = set(ordered[:validation_count])
    training_sources = set(ordered[validation_count:])
    return {
        "train": sorted(
            path for source in training_sources for path in grouped["train"][source]
        ),
        "validation": sorted(
            path for source in validation_sources for path in grouped["train"][source]
        ),
        "test": sorted(
            path for source in test_sources for path in grouped["test"][source]
        ),
    }


def make_answer_mask(
    response_mask: torch.Tensor, graph_ptr: torch.Tensor, *, mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample response-only masks and guarantee at least one per graph."""

    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    response = torch.as_tensor(response_mask, dtype=torch.bool)
    pointers = torch.as_tensor(graph_ptr, dtype=torch.long, device=response.device)
    if (pointers.ndim != 1 or len(pointers) < 2 or int(pointers[0]) != 0
            or int(pointers[-1]) != len(response) or bool((pointers[1:] <= pointers[:-1]).any())):
        raise ValueError("graph_ptr must increase from zero to the node count")
    random_values = torch.rand(len(response), device=response.device, generator=generator)
    masked = response & (random_values < mask_ratio)
    graph_count = len(pointers) - 1
    graph_ids = torch.repeat_interleave(
        torch.arange(graph_count, device=response.device), pointers.diff()
    )
    response_counts = torch.zeros(graph_count, dtype=torch.long, device=response.device)
    response_counts.index_add_(0, graph_ids, response.long())
    if bool((response_counts == 0).any()):
        raise ValueError("every graph must contain at least one response token")
    masked_counts = torch.zeros_like(response_counts)
    masked_counts.index_add_(0, graph_ids, masked.long())
    missing = masked_counts == 0
    candidates = random_values.masked_fill(~response, torch.inf)
    best = torch.full((graph_count,), torch.inf, device=response.device,
                      dtype=random_values.dtype)
    best.scatter_reduce_(0, graph_ids, candidates, reduce="amin", include_self=True)
    masked |= response & missing[graph_ids] & (candidates == best[graph_ids])
    return masked


def _tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _move_tensors(value: object, device: torch.device) -> object:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_tensors(item, device) for item in value)
    return value


def _compact_for_disk(graph: Mapping[str, object], *, storage_dtype: torch.dtype) -> dict[str, object]:
    validate_label_free(graph)
    float_fields = {
        "x", "node_context", "edge_attr", "neighbor_mean_target",
        "neighbor_log_variance_target", "route_stats_target",
    }
    integer_dtypes = {"edge_index": torch.int32, "edge_type": torch.int8, "token_ids": torch.int32}
    compact: dict[str, object] = {}
    for key, value in graph.items():
        if not isinstance(value, torch.Tensor):
            compact[key] = value
        elif key in float_fields:
            compact[key] = value.detach().to(device="cpu", dtype=storage_dtype)
        elif key in integer_dtypes:
            compact[key] = value.detach().to(device="cpu", dtype=integer_dtypes[key])
        else:
            compact[key] = value.detach().cpu()
    validate_compact_graph(compact)
    return compact


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_jsonl(path: Path, records: Sequence[Mapping[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_torch_save(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def discover_attention_paths(attention_dir: str | Path, *, limit: int | None = None) -> list[Path]:
    """Discover a flat cache or an official root containing train/ and test/."""

    root = Path(attention_dir).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"attention_dir must be an existing directory: {root}")
    official_dirs = [root / name for name in ("train", "test") if (root / name).is_dir()]
    roots = official_dirs or [root]
    files = sorted(
        path
        for cache_root in roots
        for path in cache_root.iterdir()
        if path.is_file() and path.suffix == ".pt"
    )
    formal = [path for path in files if path.name.startswith("attention_")]
    legacy = [path for path in files if path.name.startswith("sample_")]
    known = set(formal)
    known.update(legacy)
    unknown = [path.name for path in files if path not in known]
    if unknown:
        raise ValueError(
            "attention_dir contains non-sample .pt files: "
            + ", ".join(unknown[:8])
        )
    if formal and legacy:
        raise ValueError("attention_dir mixes formal attention_*.pt and legacy sample_*.pt caches")
    paths = formal or legacy
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        paths = paths[:limit]
    if not paths:
        raise ValueError(
            f"no attention_*.pt or sample_*.pt cache files found under {root}"
        )
    return paths


def _json_identity(value: Mapping[str, object]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _raw_cache_identity(sample: Mapping[str, object], stat: object) -> dict[str, object]:
    cache_format = str(sample["cache_format"])
    attention = torch.as_tensor(
        sample["attention_diagonal"]
        if cache_format == "formal_sparse_csr"
        else sample["attention"]
    )
    if cache_format == "formal_sparse_csr":
        layers = int(sample["num_attention_layers"])
        heads = int(sample["num_attention_heads"])
        token_count = int(torch.as_tensor(sample["token_ids"]).numel())
        fingerprint: str | None = str(sample["attention_cache_fingerprint"])
        cache_dtype = str(sample["cache_dtype"])
        attention_floor: float | None = float(sample["attention_floor"])
        input_policy: str | None = str(sample["input_policy"])
        was_truncated: bool | None = bool(sample["was_truncated"])
    else:
        layers, heads, token_count = map(int, attention.shape[:3])
        fingerprint = input_policy = None
        cache_dtype = str(attention.dtype).removeprefix("torch.")
        attention_floor = was_truncated = None
    return {
        "source_id": str(sample["source_id"]),
        "sample_id": str(sample["sample_id"]),
        "dataset_split": sample.get("dataset_split"),
        "cache_format": cache_format,
        "attention_cache_fingerprint": fingerprint,
        "cache_dtype": cache_dtype,
        "input_policy": input_policy,
        "was_truncated": was_truncated,
        "response_idx": int(sample["response_idx"]),
        "token_count": token_count,
        "layers": layers,
        "heads": heads,
        "attention_floor": attention_floor,
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _bind_graph_raw_identity(
    graph: dict[str, object], raw_identity: Mapping[str, object]
) -> None:
    config = dict(graph["graph_config"])
    config["raw_identity"] = dict(raw_identity)
    graph["graph_config"] = config


def _verify_graph_raw_identity(
    graph: Mapping[str, object], expected: Mapping[str, object], path: Path
) -> None:
    config = graph.get("graph_config")
    actual = config.get("raw_identity") if isinstance(config, Mapping) else None
    if not isinstance(actual, Mapping):
        raise RuntimeError(f"stale graph has no raw identity: {path}")  # noqa: TRY004
    divergent = [key for key, value in expected.items() if actual.get(key) != value]
    if divergent:
        raise RuntimeError(
            f"stale graph raw identity diverges for {path}: {', '.join(divergent)}"
        )
    if (
        int(graph["response_idx"]) != int(expected["response_idx"])
        or int(torch.as_tensor(graph["x"]).shape[0]) != int(expected["token_count"])
        or str(graph["source_id"]) != str(expected["source_id"])
        or str(graph["sample_id"]) != str(expected["sample_id"])
    ):
        raise RuntimeError(f"stale graph tensor identity diverges from raw cache: {path}")


_SEMANTIC_CONFIG_KEYS = (
    "cache_format", "attention_cache_schema", "selection_scope",
    "feature_semantics", "attention_floor", "top_k", "top_k_prefix",
    "top_k_history", "use_hidden", "hidden_projection_dim",
)


def _graph_semantics(graph: Mapping[str, object]) -> dict[str, object]:
    config = graph["graph_config"]
    if not isinstance(config, Mapping):
        raise TypeError("graph_config must be a mapping")
    return {
        "graph_schema": str(graph["schema_version"]),
        "node_dim": int(torch.as_tensor(graph["x"]).shape[1]),
        "edge_dim": int(torch.as_tensor(graph["edge_attr"]).shape[1]),
        "context_dim": int(torch.as_tensor(graph["node_context"]).shape[1]),
        "route_dim": int(torch.as_tensor(graph["route_stats_target"]).shape[-1]),
        "storage_dtype": str(torch.as_tensor(graph["x"]).dtype).removeprefix("torch."),
        "edge_types": 2,
        "construction": {key: config.get(key) for key in _SEMANTIC_CONFIG_KEYS},
    }


def _semantic_signature(semantics: Mapping[str, object]) -> str:
    return hashlib.sha256(_json_identity(semantics).encode("utf-8")).hexdigest()


def load_graph_semantic_signature(graph_dir: str | Path) -> str:
    path = Path(graph_dir) / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"compact manifest not found: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    signature = str(summary.get("graph_semantic_signature", "")).strip()
    if not signature:
        raise RuntimeError("compact manifest is missing graph_semantic_signature")
    return signature


def _preflight_attention_samples(paths: Sequence[Path]) -> dict[Path, dict[str, object]]:
    """Validate formal sparse or legacy dense caches before GPU allocation."""

    metadata: dict[Path, dict[str, object]] = {}
    failures: list[str] = []
    identities: set[tuple[str, str]] = set()
    for path in paths:
        try:
            before = path.stat()
            sample = load_attention_sample(path, device="cpu", mmap=True, include_labels=False)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise RuntimeError("raw cache changed during preflight")
            sample_id = str(sample["sample_id"])
            source_id = str(sample["source_id"])
            cache_format = str(sample["cache_format"])
            expected_format = (
                "formal_sparse_csr"
                if path.name.startswith("attention_")
                else "legacy_dense"
            )
            if cache_format != expected_format:
                raise ValueError(
                    f"filename/cache format mismatch: {path.name} declares {cache_format!r}"
                )
            attention_name = (
                "attention_diagonal"
                if cache_format == "formal_sparse_csr"
                else "attention"
            )
            attention = torch.as_tensor(sample[attention_name])
            if not sample_id or not source_id or not cache_format or not attention.is_floating_point():
                raise ValueError("sample_id/source_id/cache_format must be non-empty and attention floating")
            identity = (source_id, sample_id)
            if identity in identities:
                raise ValueError(f"duplicate (source_id, sample_id): {identity}")
            identities.add(identity)
            raw_identity = _raw_cache_identity(sample, after)
            metadata[path] = {**raw_identity, "raw_identity": raw_identity}
            del sample
        except Exception as error:  # noqa: BLE001 - aggregate all corrupt-cache failures
            failures.append(f"{path.name}: {type(error).__name__}: {error}")
    if failures:
        raise ValueError(
            f"raw attention preflight failed for {len(failures)}/{len(paths)} files; "
            + " | ".join(failures[:8])
        )
    return metadata


def compact_attention_cache(
    attention_dir: str | Path, output_dir: str | Path, *, device: str,
    prefix_top_k: int = 8, history_top_k: int = 8, query_block: int = 64,
    layer_chunk: int = 2, storage_dtype: str = "float16", limit: int | None = None,
    resume: bool = False,
) -> dict[str, object]:
    """Compact formal sparse-CSR or legacy dense caches one GPU sample at a time."""

    requested = torch.device(device)
    if requested.type != "cuda":
        raise ValueError("cache compaction requires an explicit CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    dtype_by_name = {"float16": torch.float16, "float32": torch.float32}
    if storage_dtype not in dtype_by_name:
        raise ValueError("storage_dtype must be float16 or float32")
    source_root, destination = Path(attention_dir).resolve(), Path(output_dir).resolve()
    if source_root == destination or source_root in destination.parents or destination in source_root.parents:
        raise ValueError("attention_dir and output_dir must not be equal or nested")
    paths = discover_attention_paths(source_root)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        paths = paths[:limit]
    metadata = _preflight_attention_samples(paths)
    graph_dir = destination / "graphs"
    graph_dir.mkdir(parents=True, exist_ok=True)
    identity = json.dumps({"builder_schema": "ragtruth_typed_topk_v3",
                           "prefix_top_k": prefix_top_k, "history_top_k": history_top_k,
                           "query_block": query_block, "layer_chunk": layer_chunk,
                           "storage_dtype": storage_dtype, "uses_hidden": False},
                          sort_keys=True, separators=(",", ":"))
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError("tqdm is required; install requirements-unsupervised-token-graph.txt") from error
    records: list[dict[str, object]] = []
    feature_dims: set[tuple[int, int]] = set()
    semantic_payloads: dict[str, dict[str, object]] = {}
    identities: set[tuple[str, str]] = set()
    for path in tqdm(paths, desc="compact attention -> token graphs", unit="sample"):
        relative = path.relative_to(source_root).as_posix()
        raw = metadata[path]
        current = path.stat()
        if (current.st_size, current.st_mtime_ns) != (
            raw["source_size"], raw["source_mtime_ns"]
        ):
            raise RuntimeError(f"raw cache changed after preflight: {path}")
        raw_json = _json_identity(raw["raw_identity"])
        digest = hashlib.sha256(
            f"{relative}\x1f{identity}\x1f{raw_json}".encode()
        ).hexdigest()[:16]
        graph_path = graph_dir / f"{path.stem}_{digest}.pt"
        if graph_path.exists() and not resume:
            raise FileExistsError(f"compact graph already exists: {graph_path}; pass --resume to reuse it")
        if graph_path.exists():
            graph = torch.load(graph_path, map_location="cpu", weights_only=True, mmap=True)
            validate_compact_graph(graph)
        else:
            sample = load_attention_sample(path, device="cpu", mmap=True, include_labels=False)
            loaded_stat = path.stat()
            actual_identity = _raw_cache_identity(sample, loaded_stat)
            if _json_identity(actual_identity) != raw_json:
                raise RuntimeError(f"raw cache changed between preflight and graph build: {path}")
            graph_on_gpu = build_compact_topk_graph(
                sample, top_k=max(prefix_top_k, history_top_k),
                top_k_prefix=prefix_top_k, top_k_history=history_top_k,
                query_block=query_block, layer_chunk=layer_chunk, device=requested,
                use_hidden=False,
            )
            _bind_graph_raw_identity(graph_on_gpu, actual_identity)
            graph = _compact_for_disk(graph_on_gpu, storage_dtype=dtype_by_name[storage_dtype])
            atomic_torch_save(graph_path, graph)
            del graph_on_gpu, sample
        _verify_graph_raw_identity(graph, raw["raw_identity"], path)
        semantics = _graph_semantics(graph)
        semantic_signature = _semantic_signature(semantics)
        semantic_payloads[semantic_signature] = semantics
        node_dim, edge_dim = int(graph["x"].shape[1]), int(graph["edge_attr"].shape[1])
        feature_dims.add((node_dim, edge_dim))
        sample_identity = (str(graph["source_id"]), str(graph["sample_id"]))
        if sample_identity in identities:
            raise ValueError(f"duplicate attention sample identity: {sample_identity}")
        identities.add(sample_identity)
        records.append({"path": graph_path.relative_to(destination).as_posix(),
                        "source_file": relative, "source_id": str(graph["source_id"]),
                        "sample_id": str(graph["sample_id"]),
                        "dataset_split": raw["dataset_split"],
                        "cache_format": str(graph["graph_config"]["cache_format"]),
                        "attention_floor": raw["attention_floor"],
                        "attention_cache_fingerprint": raw["attention_cache_fingerprint"],
                        "cache_dtype": raw["cache_dtype"],
                        "input_policy": raw["input_policy"],
                        "was_truncated": raw["was_truncated"],
                        "response_idx": int(graph["response_idx"]),
                        "token_count": raw["token_count"],
                        "layers": raw["layers"], "heads": raw["heads"],
                        "raw_identity_sha256": hashlib.sha256(raw_json.encode("utf-8")).hexdigest(),
                        "graph_semantic_signature": semantic_signature,
                        "node_count": int(graph["x"].shape[0]),
                        "response_count": int(graph["response_mask"].sum()),
                        "edge_count": int(graph["edge_index"].shape[1]),
                        "bytes": _tensor_bytes(graph)})
        del graph
    if len(feature_dims) != 1:
        raise ValueError(f"inconsistent compact graph dimensions: {sorted(feature_dims)}")
    if len(semantic_payloads) != 1:
        raise ValueError("compact corpus mixes incompatible graph feature semantics")
    node_dim, edge_dim = feature_dims.pop()
    graph_semantic_signature, graph_semantics = next(iter(semantic_payloads.items()))
    atomic_jsonl(destination / "manifest.jsonl", records)
    summary: dict[str, object] = {
        "schema_version": "ragtruth_typed_topk_corpus_v3", "state": "complete",
        "samples": len(records), "source_groups": len({record["source_id"] for record in records}),
        "dataset_split_counts": {
            name: sum(record["dataset_split"] == name for record in records)
            for name in ("train", "test")
        },
        "nodes": sum(int(record["node_count"]) for record in records),
        "edges": sum(int(record["edge_count"]) for record in records),
        "compact_bytes": sum(int(record["bytes"]) for record in records),
        "node_dim": node_dim, "edge_dim": edge_dim,
        "graph_semantic_signature": graph_semantic_signature,
        "graph_semantics": graph_semantics,
        "config": {"prefix_top_k": prefix_top_k, "history_top_k": history_top_k,
                   "query_block": query_block, "layer_chunk": layer_chunk,
                   "storage_dtype": storage_dtype, "raw_loader": "cpu_mmap",
                   "graph_builder": str(requested), "uses_hidden": False,
                   "discovery": "root_attention_or_sample_pt_preflight"},
    }
    atomic_json(destination / "manifest.json", summary)
    return summary


def load_compact_manifest(graph_dir: str | Path) -> list[dict[str, object]]:
    root = Path(graph_dir)
    summary_path, records_path = root / "manifest.json", root / "manifest.jsonl"
    if not summary_path.is_file() or not records_path.is_file():
        raise FileNotFoundError(f"compact manifest not found under {root}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("state") != "complete":
        raise RuntimeError(f"compact graph corpus is incomplete: {summary_path}")
    if summary.get("schema_version") != "ragtruth_typed_topk_corpus_v3":
        raise RuntimeError(f"incompatible compact corpus schema: {summary_path}")
    semantic_signature = str(summary.get("graph_semantic_signature", "")).strip()
    if not semantic_signature:
        raise RuntimeError("compact manifest is missing graph_semantic_signature")
    root_resolved, records = root.resolve(), []
    for line in records_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = dict(json.loads(line))
        if not all(str(record.get(name, "")).strip() for name in ("source_id", "sample_id", "cache_format")):
            raise RuntimeError("compact manifest record is missing source_id/sample_id/cache_format")
        required_metadata = {
            "attention_floor", "attention_cache_fingerprint", "cache_dtype",
            "input_policy", "was_truncated", "response_idx", "token_count",
            "layers", "heads", "raw_identity_sha256", "graph_semantic_signature",
        }
        if not required_metadata.issubset(record):
            raise RuntimeError("compact manifest record is missing raw identity metadata")
        if record["graph_semantic_signature"] != semantic_signature:
            raise RuntimeError("compact manifest mixes graph semantic signatures")
        path = (root / str(record["path"])).resolve()
        if root_resolved not in path.parents or not path.is_file():
            raise RuntimeError(f"unsafe or missing compact graph path: {path}")
        record["path"] = path
        records.append(record)
    expected = int(summary.get("samples", -1))
    if len(records) != expected:
        raise RuntimeError("compact manifest sample count does not match manifest.jsonl")
    if not records:
        raise ValueError("compact graph corpus is empty")
    return records


class CompactGraphStore:
    """Explicit GPU-resident or streaming store; never silently falls back."""

    def __init__(self, records: Sequence[Mapping[str, object]], *, device: str,
                 residency: str = "cuda", max_resident_gib: float = 0.0,
                 show_progress: bool = True) -> None:
        if residency not in {"cuda", "stream"}:
            raise ValueError("residency must be cuda or stream")
        self.device, self.residency = torch.device(device), residency
        self._resident: dict[Path, Mapping[str, object]] = {}
        if self.device.type != "cuda":
            raise ValueError("typed graph training/scoring requires an explicit CUDA device")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if max_resident_gib < 0:
            raise ValueError("max_resident_gib cannot be negative")
        if residency == "stream":
            return
        required = sum(int(record["bytes"]) for record in records)
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        automatic_budget = int(free_bytes * 0.45)
        budget = min(int(max_resident_gib * 1024**3) if max_resident_gib > 0 else automatic_budget,
                     automatic_budget)
        if required > budget:
            raise RuntimeError("compact graph corpus exceeds the CUDA-resident budget: "
                               f"required={required / 1024**3:.2f} GiB, budget={budget / 1024**3:.2f} GiB; "
                               "use --residency stream or a smaller corpus explicitly")
        iterable: Any = records
        if show_progress:
            try:
                from tqdm.auto import tqdm
            except ImportError as error:
                raise RuntimeError("tqdm is required; install requirements-unsupervised-token-graph.txt") from error
            iterable = tqdm(records, desc="load compact graphs -> GPU", unit="graph")
        for record in iterable:
            path = Path(record["path"])
            graph = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            validate_compact_graph(graph)
            self._resident[path] = _move_tensors(graph, self.device)
            del graph

    def get(self, path: str | Path) -> Mapping[str, object]:
        graph_path = Path(path)
        if self.residency == "cuda":
            if graph_path not in self._resident:
                raise KeyError(f"graph is not resident in this store: {graph_path}")
            return self._resident[graph_path]
        graph = torch.load(graph_path, map_location="cpu", weights_only=True, mmap=True)
        validate_compact_graph(graph)
        return _move_tensors(graph, self.device)

    def close(self) -> None:
        self._resident.clear()


def budget_batches(paths: Sequence[Path], record_by_path: Mapping[Path, Mapping[str, object]], *,
                   max_nodes: int, max_edges: int) -> list[list[Path]]:
    if max_nodes < 1 or max_edges < 1:
        raise ValueError("max_nodes and max_edges must be positive")
    batches, current = [], []
    nodes = edges = 0
    for path in paths:
        record = record_by_path[path]
        graph_nodes, graph_edges = int(record["node_count"]), int(record["edge_count"])
        if graph_nodes > max_nodes or graph_edges > max_edges:
            raise RuntimeError(f"one graph exceeds the batch budget: {path} nodes={graph_nodes}, edges={graph_edges}")
        if current and (nodes + graph_nodes > max_nodes or edges + graph_edges > max_edges):
            batches.append(current)
            current, nodes, edges = [], 0, 0
        current.append(path)
        nodes += graph_nodes
        edges += graph_edges
    if current:
        batches.append(current)
    return batches


def autocast_context(device: torch.device, amp: str):
    from contextlib import nullcontext
    if amp == "none":
        return nullcontext()
    if device.type != "cuda":
        raise ValueError("mixed precision is only supported on CUDA")
    if amp == "bfloat16":
        if not torch.cuda.is_bf16_supported():
            raise RuntimeError("this CUDA device does not support bfloat16")
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    if amp == "float16":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    raise ValueError("amp must be none, bfloat16, or float16")


def new_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator
