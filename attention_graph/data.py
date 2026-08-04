"""Formal RAGTruth attention-cache I/O and prepared-graph storage.

The training path in this module is deliberately label blind.  Raw labels can
only be requested from :func:`load_attention_record` for post-hoc evaluation;
they are never copied into a prepared graph or its index.
"""

from __future__ import annotations

import json
import math
import os
import random
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch

from .graph import (
    FORMAL_CACHE_SCHEMA,
    AttentionGraph,
    GraphBuildConfig,
    build_attention_graph,
)


GRAPH_ARTIFACT_SCHEMA = "attention-graph-sparse-channels-v1"

_FORMAL_REQUIRED = frozenset(
    {
        "attention_cache_schema",
        "attention_cache_fingerprint",
        "cache_dtype",
        "input_policy",
        "was_truncated",
        "source_id",
        "response_id",
        "response_idx",
        "token_ids",
        "num_attention_layers",
        "num_attention_heads",
        "attention_diagonal",
        "response_row_ptr",
        "response_column_indices",
        "response_values",
        "attention_floor",
    }
)
_CACHE_DTYPES = {
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
    "float32": torch.float32,
    "fp32": torch.float32,
}
_GRAPH_TENSORS = frozenset(
    {
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
    }
)
_GRAPH_SCALARS = frozenset(
    {
        "source_id",
        "response_id",
        "sample_id",
        "response_idx",
        "num_layers",
        "num_heads",
        "attention_floor",
        "build_config",
    }
)
_RAW_IDENTITY_FIELDS = frozenset(
    {
        "source_id",
        "response_id",
        "dataset_split",
        "attention_cache_schema",
        "attention_cache_fingerprint",
        "cache_dtype",
        "response_idx",
        "token_count",
        "num_layers",
        "num_heads",
        "attention_floor",
        "source_size",
        "source_mtime_ns",
    }
)


@dataclass(frozen=True)
class AttentionCacheRecord:
    """One raw formal-cache file discovered under an official split."""

    path: Path
    dataset_split: str


@dataclass(frozen=True)
class PreparedGraphRecord:
    """Label-free index entry for one prepared graph."""

    source_id: str
    response_id: str
    sample_id: str
    dataset_split: str
    cache_path: Path
    graph_path: Path
    state: str
    num_nodes: int
    num_response_nodes: int
    num_edges: int
    num_rp_edges: int
    num_rr_edges: int
    num_traces: int


def _normalise_splits(splits: Sequence[str]) -> tuple[str, ...]:
    normalised = tuple(str(value).strip().casefold() for value in splits)
    if not normalised:
        raise ValueError("splits must contain train and/or test")
    if len(set(normalised)) != len(normalised):
        raise ValueError("splits must not contain duplicates")
    unsupported = sorted(set(normalised).difference({"train", "test"}))
    if unsupported:
        raise ValueError(f"unsupported official RAGTruth splits: {unsupported}")
    return normalised


def discover_attention_cache(
    cache_root: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
) -> list[AttentionCacheRecord]:
    """Discover formal ``attention_*.pt`` files without consulting manifests.

    An ``in_progress`` or absent manifest is valid: each atomically written
    sample is independently usable.  Callers explicitly choose which official
    splits are required, so a partial train-only pilot does not require test.
    """

    root = Path(cache_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"attention cache root does not exist: {root}")
    requested = _normalise_splits(splits)
    records: list[AttentionCacheRecord] = []
    official_layout = any((root / name).is_dir() for name in ("train", "test"))
    for split in requested:
        if official_layout:
            split_root = root / split
            if not split_root.is_dir():
                raise FileNotFoundError(
                    f"official RAGTruth {split} attention cache is absent: {split_root}"
                )
        else:
            if len(requested) != 1:
                raise ValueError(
                    "a flat attention cache can only be assigned to one explicit split"
                )
            split_root = root
        paths = sorted(
            path
            for path in split_root.iterdir()
            if path.is_file()
            and path.name.startswith("attention_")
            and path.suffix == ".pt"
        )
        records.extend(AttentionCacheRecord(path=path, dataset_split=split) for path in paths)
    if not records:
        joined = ", ".join(requested)
        raise ValueError(f"no formal attention_*.pt files found for split(s): {joined}")
    return records


def _manifest_nonnegative_count(
    manifest: Mapping[str, object],
    field: str,
    *,
    path: Path,
) -> int | None:
    if field not in manifest:
        return None
    value = manifest[field]
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"cache manifest {field} must be a non-negative integer: {path}"
        )
    return value


def _manifest_file_names(
    manifest: Mapping[str, object],
    field: str,
    *,
    path: Path,
) -> list[str] | None:
    if field not in manifest:
        return None
    raw = manifest[field]
    if not isinstance(raw, list) or any(not isinstance(value, str) for value in raw):
        raise ValueError(f"cache manifest {field} must be a string list: {path}")
    names = [str(value) for value in raw]
    if len(names) != len(set(names)):
        raise ValueError(f"cache manifest {field} contains duplicate files: {path}")
    invalid = [
        name
        for name in names
        if Path(name).name != name
        or not name.startswith("attention_")
        or not name.endswith(".pt")
    ]
    if invalid:
        raise ValueError(
            f"cache manifest {field} contains invalid attention file names: {path}"
        )
    return names


def _audit_attention_split(root: Path, split: str) -> dict[str, object]:
    split_root = root / split
    directory_present = split_root.is_dir()
    observed_names = (
        sorted(
            path.name
            for path in split_root.iterdir()
            if path.is_file()
            and path.name.startswith("attention_")
            and path.suffix == ".pt"
        )
        if directory_present
        else []
    )
    manifest_path = split_root / "manifest.json"
    manifest_present = manifest_path.is_file()
    if not manifest_present:
        return {
            "split": split,
            "directory": str(split_root),
            "directory_present": directory_present,
            "manifest_path": str(manifest_path),
            "manifest_present": False,
            "manifest_state": None,
            "observed_file_count": len(observed_names),
            "declared_matched_samples": None,
            "declared_cache_files": None,
            "declared_file_count": None,
            "declared_inventory_field": None,
            "missing_declared_file_count": None,
            "unexpected_observed_file_count": None,
            "inventory_exact": False,
            "complete": False,
        }
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid cache manifest JSON: {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError(f"cache manifest must contain a JSON object: {manifest_path}")
    state_value = manifest.get("state")
    if not isinstance(state_value, str) or not state_value.strip():
        raise ValueError(f"cache manifest state must be a non-empty string: {manifest_path}")
    state = state_value.strip().casefold()
    matched_samples = _manifest_nonnegative_count(
        manifest, "matched_samples", path=manifest_path
    )
    cache_files = _manifest_nonnegative_count(
        manifest, "cache_files", path=manifest_path
    )
    completed_names = _manifest_file_names(
        manifest, "cache_file_names", path=manifest_path
    )
    expected_names = _manifest_file_names(
        manifest, "expected_files", path=manifest_path
    )
    if completed_names is not None and expected_names is not None:
        if set(completed_names) != set(expected_names):
            raise ValueError(
                f"cache manifest expected/completed inventories conflict: {manifest_path}"
            )
    if completed_names is not None:
        declared_names = completed_names
        inventory_field: str | None = "cache_file_names"
    elif expected_names is not None:
        declared_names = expected_names
        inventory_field = "expected_files"
    else:
        declared_names = None
        inventory_field = None

    observed_set = set(observed_names)
    declared_set = set(declared_names or ())
    names_exact = declared_names is not None and observed_set == declared_set
    counts_exact = (
        matched_samples is not None
        and cache_files is not None
        and matched_samples == cache_files == len(observed_names)
    )
    inventory_exact = bool(names_exact and counts_exact)
    return {
        "split": split,
        "directory": str(split_root),
        "directory_present": directory_present,
        "manifest_path": str(manifest_path),
        "manifest_present": True,
        "manifest_state": state,
        "observed_file_count": len(observed_names),
        "declared_matched_samples": matched_samples,
        "declared_cache_files": cache_files,
        "declared_file_count": (
            len(declared_names) if declared_names is not None else None
        ),
        "declared_inventory_field": inventory_field,
        "missing_declared_file_count": (
            len(declared_set.difference(observed_set))
            if declared_names is not None
            else None
        ),
        "unexpected_observed_file_count": (
            len(observed_set.difference(declared_set))
            if declared_names is not None
            else None
        ),
        "inventory_exact": inventory_exact,
        "complete": state == "complete" and inventory_exact,
    }


def audit_attention_cache(
    cache_root: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
) -> dict[str, dict[str, object]]:
    """Summarize cache completeness without blocking partial-cache discovery.

    A split is complete only when a valid ``state=complete`` manifest declares
    matching sample/cache counts and an exact attention-file inventory.
    Missing or in-progress manifests are reported as partial, while malformed
    manifest JSON or schema-critical fields fail closed.
    """

    root = Path(cache_root).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"attention cache root does not exist: {root}")
    requested = _normalise_splits(splits)
    return {
        split: _audit_attention_split(root, split) for split in requested
    }


def _scalar_int(value: object, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())


def _strict_bool(value: object, name: str) -> bool:
    if isinstance(value, bool):
        return value
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1 or tensor.dtype != torch.bool:
        raise ValueError(f"{name} must be a boolean scalar")
    return bool(tensor.item())


def _cache_dtype(value: object) -> tuple[str, torch.dtype]:
    name = str(value).strip().casefold().removeprefix("torch.")
    if name not in _CACHE_DTYPES:
        raise ValueError("cache_dtype must be float16, bfloat16, or float32")
    dtype = _CACHE_DTYPES[name]
    return str(dtype).removeprefix("torch."), dtype


def _safe_torch_load(path: Path, *, mmap: bool) -> Mapping[str, object]:
    try:
        loaded = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
            mmap=bool(mmap),
        )
    except TypeError as error:
        if mmap:
            raise RuntimeError(
                "memory-mapped cache loading requires a PyTorch build with "
                "torch.load(..., mmap=...); upgrade PyTorch or explicitly set "
                "mmap=False"
            ) from error
        raise RuntimeError(
            "safe cache loading requires a PyTorch build with "
            "torch.load(..., weights_only=True); upgrade PyTorch"
        ) from error
    if not isinstance(loaded, Mapping):
        raise TypeError(f"attention cache must contain a mapping: {path}")
    return loaded


def _validate_formal_record(loaded: Mapping[str, object], path: Path) -> None:
    missing = sorted(_FORMAL_REQUIRED.difference(loaded))
    if missing:
        raise ValueError(f"formal attention cache is missing fields {missing}: {path}")
    if str(loaded["attention_cache_schema"]) != FORMAL_CACHE_SCHEMA:
        raise ValueError(
            f"unsupported attention_cache_schema {loaded['attention_cache_schema']!r}: {path}"
        )
    if not str(loaded["attention_cache_fingerprint"]).strip():
        raise ValueError("attention_cache_fingerprint must be non-empty")
    if not str(loaded["input_policy"]).strip():
        raise ValueError("input_policy must be non-empty")
    if _strict_bool(loaded["was_truncated"], "was_truncated"):
        raise ValueError("truncated attention caches cannot be compared as full graphs")
    _, storage_dtype = _cache_dtype(loaded["cache_dtype"])
    required_dtypes = {
        "token_ids": torch.int64,
        "response_row_ptr": torch.int64,
        "response_column_indices": torch.int32,
        "attention_diagonal": storage_dtype,
        "response_values": storage_dtype,
    }
    for name, expected in required_dtypes.items():
        actual = torch.as_tensor(loaded[name]).dtype
        if actual != expected:
            raise ValueError(f"{name} dtype must be {expected}; got {actual}")

    layers = _scalar_int(loaded["num_attention_layers"], "num_attention_layers")
    heads = _scalar_int(loaded["num_attention_heads"], "num_attention_heads")
    diagonal = torch.as_tensor(loaded["attention_diagonal"])
    if layers < 1 or heads < 1 or diagonal.ndim != 3:
        raise ValueError("attention_diagonal must have shape [layers, heads, tokens]")
    if tuple(diagonal.shape[:2]) != (layers, heads):
        raise ValueError("attention_diagonal layer/head dimensions are inconsistent")
    token_count = int(diagonal.shape[2])
    response_idx = _scalar_int(loaded["response_idx"], "response_idx")
    if not 0 < response_idx < token_count:
        raise ValueError("response_idx must split non-empty prompt and response")
    if torch.as_tensor(loaded["token_ids"]).numel() != token_count:
        raise ValueError("token_ids must contain one id per attention node")
    row_ptr = torch.as_tensor(loaded["response_row_ptr"]).flatten()
    columns = torch.as_tensor(loaded["response_column_indices"]).flatten()
    values = torch.as_tensor(loaded["response_values"]).flatten()
    expected_rows = layers * heads * (token_count - response_idx)
    valid_ptr = (
        row_ptr.numel() == expected_rows + 1
        and int(row_ptr[0]) == 0
        and int(row_ptr[-1]) == values.numel()
        and not bool((row_ptr[1:] < row_ptr[:-1]).any())
    )
    if not valid_ptr or columns.numel() != values.numel():
        raise ValueError("formal response attention CSR arrays are inconsistent")
    floor = float(loaded["attention_floor"])
    if not math.isfinite(floor) or not 0.0 < floor <= 1.0:
        raise ValueError("attention_floor must be finite in (0, 1]")


def _to_device(
    value: object,
    *,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach()
    return tensor.to(
        device=device,
        dtype=tensor.dtype if dtype is None else dtype,
        non_blocking=device.type == "cuda",
    )


def load_attention_record(
    path: str | Path,
    *,
    device: str | torch.device,
    mmap: bool = True,
    include_labels: bool = False,
) -> dict[str, object]:
    """Load the formal sparse cache through an explicit field whitelist."""

    cache_path = Path(path).expanduser().resolve()
    loaded = _safe_torch_load(cache_path, mmap=mmap)
    _validate_formal_record(loaded, cache_path)
    requested = torch.device(device)
    cache_dtype_name, _ = _cache_dtype(loaded["cache_dtype"])
    split = str(loaded.get("split", cache_path.parent.name)).strip().casefold()
    if split not in {"train", "test"}:
        raise ValueError("formal RAGTruth split must be train or test")
    response_id = str(loaded["response_id"]).strip()
    source_id = str(loaded["source_id"]).strip()
    if not response_id or not source_id:
        raise ValueError("formal source_id and response_id must be non-empty")

    # This is a whitelist: unrelated raw metadata and newly added target fields
    # cannot silently enter graph construction.
    record: dict[str, object] = {
        "cache_format": "formal_sparse_csr",
        "attention_cache_schema": FORMAL_CACHE_SCHEMA,
        "attention_cache_fingerprint": str(loaded["attention_cache_fingerprint"]),
        "cache_dtype": cache_dtype_name,
        "input_policy": str(loaded["input_policy"]),
        "was_truncated": _strict_bool(loaded["was_truncated"], "was_truncated"),
        "source_id": source_id,
        "response_id": response_id,
        "sample_id": response_id,
        "dataset_split": split,
        "response_idx": _scalar_int(loaded["response_idx"], "response_idx"),
        "num_attention_layers": _scalar_int(
            loaded["num_attention_layers"], "num_attention_layers"
        ),
        "num_attention_heads": _scalar_int(
            loaded["num_attention_heads"], "num_attention_heads"
        ),
        "attention_floor": float(loaded["attention_floor"]),
        "token_ids": _to_device(loaded["token_ids"], device=requested, dtype=torch.long),
        "attention_diagonal": _to_device(
            loaded["attention_diagonal"], device=requested
        ),
        "response_row_ptr": _to_device(
            loaded["response_row_ptr"], device=requested, dtype=torch.long
        ),
        "response_column_indices": _to_device(
            loaded["response_column_indices"], device=requested
        ),
        "response_values": _to_device(loaded["response_values"], device=requested),
    }
    if include_labels and "y_token" in loaded:
        labels = _to_device(loaded["y_token"], device=requested, dtype=torch.float32).flatten()
        if labels.numel() != torch.as_tensor(record["token_ids"]).numel():
            raise ValueError("y_token must contain one evaluation label per token")
        record["y_token"] = labels
    return record


def _config_mapping(config: GraphBuildConfig) -> dict[str, object]:
    return asdict(config)


def _raw_identity(
    sample: Mapping[str, object],
    cache_record: AttentionCacheRecord,
) -> dict[str, object]:
    stat = cache_record.path.stat()
    return {
        "source_id": str(sample["source_id"]),
        "response_id": str(sample["response_id"]),
        "dataset_split": cache_record.dataset_split,
        "attention_cache_schema": str(sample["attention_cache_schema"]),
        "attention_cache_fingerprint": str(sample["attention_cache_fingerprint"]),
        "cache_dtype": str(sample["cache_dtype"]),
        "response_idx": int(sample["response_idx"]),
        "token_count": int(torch.as_tensor(sample["token_ids"]).numel()),
        "num_layers": int(sample["num_attention_layers"]),
        "num_heads": int(sample["num_attention_heads"]),
        "attention_floor": float(sample["attention_floor"]),
        "source_size": int(stat.st_size),
        "source_mtime_ns": int(stat.st_mtime_ns),
    }


def _graph_mapping(graph: AttentionGraph) -> dict[str, object]:
    mapping: dict[str, object] = {}
    for field in fields(graph):
        value = getattr(graph, field.name)
        if isinstance(value, torch.Tensor):
            mapping[field.name] = value.detach().cpu()
        elif isinstance(value, GraphBuildConfig):
            mapping[field.name] = _config_mapping(value)
        else:
            mapping[field.name] = value
    return mapping


def _artifact(
    graph: AttentionGraph,
    *,
    raw_identity: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema": GRAPH_ARTIFACT_SCHEMA,
        "build_config": _config_mapping(graph.build_config),
        "raw_identity": dict(raw_identity),
        "graph": _graph_mapping(graph),
    }


def _atomic_torch_save(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        torch.save(dict(value), temporary)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _validate_artifact_mapping(
    artifact: Mapping[str, object],
    *,
    expected_config: Mapping[str, object] | None = None,
    expected_raw_identity: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    if set(artifact) != {"schema", "build_config", "raw_identity", "graph"}:
        raise ValueError("prepared graph artifact has unexpected top-level fields")
    if artifact["schema"] != GRAPH_ARTIFACT_SCHEMA:
        raise ValueError("unsupported prepared graph schema")
    config = artifact["build_config"]
    raw_identity = artifact["raw_identity"]
    graph = artifact["graph"]
    if not isinstance(config, Mapping) or not isinstance(raw_identity, Mapping):
        raise TypeError("prepared graph config and raw identity must be mappings")
    if not isinstance(graph, Mapping):
        raise TypeError("prepared graph payload must be a mapping")
    expected_fields = _GRAPH_SCALARS | _GRAPH_TENSORS
    if set(graph) != expected_fields:
        raise ValueError("prepared graph payload fields do not match the graph schema")
    if set(raw_identity) != _RAW_IDENTITY_FIELDS:
        raise ValueError("prepared graph raw-cache identity fields are incomplete")
    if dict(graph["build_config"]) != dict(config):  # type: ignore[arg-type]
        raise ValueError("prepared graph contains conflicting build configurations")
    graph_identity = {
        "source_id": str(graph["source_id"]),
        "response_id": str(graph["response_id"]),
        "response_idx": int(graph["response_idx"]),
        "token_count": int(torch.as_tensor(graph["token_ids"]).numel()),
        "num_layers": int(graph["num_layers"]),
        "num_heads": int(graph["num_heads"]),
        "attention_floor": float(graph["attention_floor"]),
    }
    for name, value in graph_identity.items():
        if raw_identity[name] != value:
            raise ValueError(
                f"prepared graph raw identity conflicts with graph field {name}"
            )
    if raw_identity["attention_cache_schema"] != FORMAL_CACHE_SCHEMA:
        raise ValueError("prepared graph does not identify the formal cache schema")
    if raw_identity["dataset_split"] not in {"train", "test"}:
        raise ValueError("prepared graph raw identity has an invalid split")
    if int(raw_identity["source_size"]) < 1 or int(raw_identity["source_mtime_ns"]) < 0:
        raise ValueError("prepared graph raw identity has invalid file metadata")
    if expected_config is not None and dict(config) != dict(expected_config):
        raise ValueError("prepared graph build configuration does not match the request")
    if expected_raw_identity is not None and dict(raw_identity) != dict(expected_raw_identity):
        raise ValueError("prepared graph raw-cache identity is stale")
    forbidden = [key for key in graph if "label" in key.casefold() or key.casefold() == "y_token"]
    if forbidden:
        raise ValueError("prepared graph must not persist evaluation labels")
    return graph


def _load_artifact(path: Path, *, mmap: bool = True) -> Mapping[str, object]:
    loaded = _safe_torch_load(path, mmap=mmap)
    return loaded


def _graph_from_mapping(
    graph: Mapping[str, object],
    *,
    device: str | torch.device,
    validate: bool = True,
) -> AttentionGraph:
    requested = torch.device(device)
    config_value = graph["build_config"]
    if not isinstance(config_value, Mapping):
        raise TypeError("graph build_config must be a mapping")
    config = GraphBuildConfig(**dict(config_value))
    config.validate(attention_floor=float(graph["attention_floor"]))
    values: dict[str, Any] = {}
    for name in _GRAPH_SCALARS:
        if name == "build_config":
            values[name] = config
        else:
            values[name] = graph[name]
    for name in _GRAPH_TENSORS:
        value = graph[name]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"prepared graph {name} must be a tensor")
        values[name] = value.to(device=requested, non_blocking=requested.type == "cuda")
    result = AttentionGraph(**values)
    if validate:
        _validate_loaded_graph(result)
    return result


def _validate_loaded_graph(graph: AttentionGraph) -> None:
    n, e, c = graph.num_nodes, graph.num_edges, graph.num_channels
    if n < 2 or not 0 < graph.response_idx < n:
        raise ValueError("prepared graph has an invalid prompt/response boundary")
    if graph.node_attr.shape != (n, c):
        raise ValueError("prepared graph node_attr shape is inconsistent")
    if graph.node_context.shape != (n, 3):
        raise ValueError("prepared graph node_context shape is inconsistent")
    if graph.response_mask.shape != (n,) or graph.response_mask.dtype != torch.bool:
        raise ValueError("prepared graph response_mask is invalid")
    if graph.edge_index.shape != (2, e):
        raise ValueError("prepared graph edge_index shape is inconsistent")
    if graph.edge_type.shape != (e,) or graph.edge_score.shape != (e,):
        raise ValueError("prepared graph edge attributes do not align")
    trace_count = graph.trace_edge_id.numel()
    if graph.trace_channel.shape != (trace_count,) or graph.trace_value.shape != (trace_count,):
        raise ValueError("prepared graph sparse channel trace does not align")
    if graph.token_ids.shape != (n,):
        raise ValueError("prepared graph token_ids shape is inconsistent")
    if not graph.source_id or not graph.response_id or not graph.sample_id:
        raise ValueError("prepared graph identities must be non-empty")
    expected_response = torch.arange(n, device=graph.response_mask.device) >= graph.response_idx
    if not torch.equal(graph.response_mask, expected_response):
        raise ValueError("prepared graph response_mask conflicts with response_idx")
    if not bool(torch.isfinite(graph.node_attr).all()) or bool(
        ((graph.node_attr < 0) | (graph.node_attr > 1)).any()
    ):
        raise ValueError("prepared graph node attention must be finite in [0, 1]")
    if not bool(torch.isfinite(graph.edge_score).all()) or bool(
        ((graph.edge_score < 0) | (graph.edge_score > 1)).any()
    ):
        raise ValueError("prepared graph edge scores must be finite in [0, 1]")
    if not bool(torch.isfinite(graph.trace_value).all()) or bool(
        ((graph.trace_value < 0) | (graph.trace_value > 1)).any()
    ):
        raise ValueError("prepared graph trace attention must be finite in [0, 1]")
    if e and (bool((graph.edge_index < 0).any()) or bool((graph.edge_index >= n).any())):
        raise ValueError("prepared graph edge_index is out of range")
    if e and bool((graph.edge_index[0] >= graph.edge_index[1]).any()):
        raise ValueError("prepared graph edges must follow causal ordering")
    if e and bool((graph.edge_index[1] < graph.response_idx).any()):
        raise ValueError("prepared graph edges must target response nodes")
    if e and bool(((graph.edge_type < 0) | (graph.edge_type > 1)).any()):
        raise ValueError("prepared graph edge types must be RP or RR")
    if e:
        expected_type = (graph.edge_index[0] >= graph.response_idx).to(graph.edge_type.dtype)
        if not torch.equal(graph.edge_type, expected_type):
            raise ValueError("prepared graph RP/RR edge type is inconsistent")
        pair_key = graph.edge_index[1] * n + graph.edge_index[0]
        if torch.unique(pair_key).numel() != e:
            raise ValueError("prepared graph contains duplicate token-pair edges")
    if trace_count and (
        bool((graph.trace_edge_id < 0).any())
        or bool((graph.trace_edge_id >= e).any())
        or bool((graph.trace_channel < 0).any())
        or bool((graph.trace_channel >= c).any())
    ):
        raise ValueError("prepared graph sparse trace index is out of range")
    if trace_count > 1:
        trace_key = graph.trace_edge_id * c + graph.trace_channel
        if not bool((trace_key[1:] > trace_key[:-1]).all()):
            raise ValueError(
                "prepared graph traces must be unique and sorted by edge/channel"
            )


def load_graph(
    path: str | Path,
    *,
    device: str | torch.device,
    mmap: bool = True,
    validate: bool = True,
) -> AttentionGraph:
    """Load and strictly validate one pure-mapping graph artifact."""

    artifact = _load_artifact(Path(path).expanduser().resolve(), mmap=mmap)
    graph = _validate_artifact_mapping(artifact)
    return _graph_from_mapping(graph, device=device, validate=validate)


def _record_from_graph(
    graph: AttentionGraph,
    *,
    cache: AttentionCacheRecord,
    graph_path: Path,
    state: str,
) -> PreparedGraphRecord:
    return PreparedGraphRecord(
        source_id=graph.source_id,
        response_id=graph.response_id,
        sample_id=graph.sample_id,
        dataset_split=cache.dataset_split,
        cache_path=cache.path,
        graph_path=graph_path,
        state=state,
        num_nodes=graph.num_nodes,
        num_response_nodes=int(graph.response_mask.sum()),
        num_edges=graph.num_edges,
        num_rp_edges=int((graph.edge_type == 0).sum()),
        num_rr_edges=int((graph.edge_type == 1).sum()),
        num_traces=int(graph.trace_value.numel()),
    )


def _index_mapping(record: PreparedGraphRecord) -> dict[str, object]:
    return {
        "source_id": record.source_id,
        "response_id": record.response_id,
        "sample_id": record.sample_id,
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


def prepare_graphs(
    *,
    cache_root: str | Path,
    output_dir: str | Path,
    config: GraphBuildConfig,
    splits: Sequence[str] = ("train", "test"),
    build_device: str | torch.device = "cpu",
    resume: bool = True,
    mmap: bool = True,
) -> list[PreparedGraphRecord]:
    """Prepare label-free graph files, reusing independently valid artifacts."""

    caches = discover_attention_cache(cache_root, splits=splits)
    destination = Path(output_dir).expanduser().resolve()
    records: list[PreparedGraphRecord] = []
    for cache in caches:
        # Labels are intentionally not requested and therefore cannot reach the
        # graph builder or persisted artifact.
        sample = load_attention_record(
            cache.path,
            device=build_device,
            mmap=mmap,
            include_labels=False,
        )
        if sample["dataset_split"] != cache.dataset_split:
            raise ValueError(
                f"cache split metadata disagrees with its directory: {cache.path}"
            )
        identity = _raw_identity(sample, cache)
        graph_path = destination / cache.dataset_split / f"{cache.path.stem}.graph.pt"
        if resume and graph_path.is_file():
            try:
                artifact = _load_artifact(graph_path, mmap=mmap)
                graph_mapping = _validate_artifact_mapping(
                    artifact,
                    expected_config=_config_mapping(config),
                    expected_raw_identity=identity,
                )
                graph = _graph_from_mapping(graph_mapping, device="cpu")
            except (OSError, RuntimeError, TypeError, ValueError, KeyError):
                graph = build_attention_graph(sample, config, device=build_device)
                _atomic_torch_save(graph_path, _artifact(graph, raw_identity=identity))
                state = "rebuilt"
            else:
                state = "reused"
        else:
            graph = build_attention_graph(sample, config, device=build_device)
            _atomic_torch_save(graph_path, _artifact(graph, raw_identity=identity))
            state = "prepared"
        records.append(
            _record_from_graph(
                graph,
                cache=cache,
                graph_path=graph_path,
                state=state,
            )
        )

    _atomic_json(destination / "index.json", [_index_mapping(record) for record in records])
    return records


def official_partitions(
    records: Sequence[PreparedGraphRecord],
    *,
    validation_fraction: float = 0.2,
    seed: int = 42,
) -> dict[str, list[PreparedGraphRecord]]:
    """Hold official test out and split official train by source identity."""

    if not math.isfinite(validation_fraction) or not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be finite in [0, 1)")
    official_train = [record for record in records if record.dataset_split == "train"]
    official_test = [record for record in records if record.dataset_split == "test"]
    unknown = sorted(
        {record.dataset_split for record in records}.difference({"train", "test"})
    )
    if unknown:
        raise ValueError(f"unexpected dataset splits in prepared records: {unknown}")
    train_sources = sorted({record.source_id for record in official_train})
    shuffled = list(train_sources)
    random.Random(int(seed)).shuffle(shuffled)
    if validation_fraction == 0.0 or len(shuffled) < 2:
        validation_sources: set[str] = set()
    else:
        count = round(len(shuffled) * validation_fraction)
        count = min(max(int(count), 1), len(shuffled) - 1)
        validation_sources = set(shuffled[:count])
    train = [record for record in official_train if record.source_id not in validation_sources]
    validation = [
        record for record in official_train if record.source_id in validation_sources
    ]
    overlap = {record.source_id for record in train} & {
        record.source_id for record in validation
    }
    if overlap:
        raise RuntimeError("source-group split leaked between train and validation")
    test_overlap = {record.source_id for record in official_train} & {
        record.source_id for record in official_test
    }
    if test_overlap:
        raise ValueError("official train and test records overlap by source_id")
    return {"train": train, "validation": validation, "test": official_test}


__all__ = [
    "AttentionCacheRecord",
    "PreparedGraphRecord",
    "audit_attention_cache",
    "discover_attention_cache",
    "load_attention_record",
    "load_graph",
    "official_partitions",
    "prepare_graphs",
]
