"""Adapt formal RAGTruth attention caches to the upstream graph schema.

The graph definition is the one in upstream ``processed_graphs_attribute.py``:
one node per token, layer/head self-attention on nodes, causal edges ending at
response tokens, and a per-layer/head attention vector on every edge.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from attention_graph.data import (
    audit_attention_cache,
    discover_attention_cache,
    load_attention_record,
)

FORMAL_CACHE_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"
GRAPH_SCHEMA = "original-ragtruth-attributed-graph-v2"
DATASET_SCHEMA = "original-ragtruth-attributed-graphs-v2"
GRAPH_METADATA_SCHEMA = "original-ragtruth-attributed-graph-metadata-v1"
UPSTREAM_COMMIT = "13e907693aa954bf070e809d8afecdf26b3b88d8"

# JSON-safe field definitions are written once to the dataset manifest.  Each
# graph carries the concrete layer/head dimensions and encodings in metadata,
# so a copied graph remains interpretable without reopening its attention cache.
GRAPH_FIELD_SCHEMA: dict[str, dict[str, object]] = {
    "response_idx": {
        "type": "int",
        "meaning": "first response node; nodes before it are prompt tokens",
        "model_input": False,
        "training_mask": True,
    },
    "token_ids": {
        "shape": "[N]",
        "dtype": "int64",
        "meaning": "token id at each global prompt-then-response node index",
        "model_input": False,
    },
    "node_role": {
        "shape": "[N]",
        "dtype": "int8",
        "encoding": {"0": "prompt", "1": "response"},
        "model_input": False,
    },
    "x": {
        "shape": "[N, C]",
        "dtype": "float32",
        "value": "attention[layer, head, token, token]",
        "channel": "C=L*H; channel=layer*H+head",
        "model_input": True,
    },
    "y_token": {
        "shape": "[N]",
        "dtype": "int64",
        "coordinate": "global prompt-then-response token index",
        "encoding": {"0": "non-hallucinated", "1": "hallucinated"},
        "constraint": "prompt entries are always 0",
        "model_input": False,
        "training_target": True,
    },
    "edge_index": {
        "shape": "[2, E]",
        "dtype": "int64",
        "rows": {"0": "source/key token", "1": "target/query token"},
        "constraints": "source < target and target is a response token",
        "model_input": True,
    },
    "edge_attr": {
        "shape": "[E, C]",
        "dtype": "float32",
        "value": "attention[layer, head, target, source] if >tau, else 0",
        "channel": "same layer-major/head-minor order as x",
        "model_input": True,
    },
    "edge_mark": {
        "shape": "[E, 2]",
        "dtype": "float32",
        "encoding": {"RP": [1.0, 0.0], "RR": [0.0, 1.0]},
        "model_input": True,
    },
    "response_label": {
        "type": "int",
        "encoding": {"0": "correct", "1": "contains hallucinated token"},
        "definition": "any(y_token[response_idx:])",
        "model_input": False,
    },
}


def _graph_metadata(
    *,
    layers: int,
    heads: int,
    input_policy: str,
    cache_dtype: str,
) -> dict[str, object]:
    return {
        "schema": GRAPH_METADATA_SCHEMA,
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "num_attention_channels": layers * heads,
        "channel_order": "layer_major_head_minor",
        "channel_index_formula": "channel = layer * num_attention_heads + head",
        "edge_direction": "source_key_to_target_query",
        "edge_selection": (
            "edge iff any attention[layer, head, target, source] > tau; "
            "channels <= tau are stored as zero"
        ),
        "relation_encoding": {"RP": [1.0, 0.0], "RR": [0.0, 1.0]},
        "node_role_encoding": {"prompt": 0, "response": 1},
        "label_coordinate": "global_prompt_then_response_token_index",
        "label_encoding": {"non_hallucinated": 0, "hallucinated": 1},
        "label_source": "RAGTruth y_token from the formal attention cache",
        "input_policy": input_policy,
        "source_cache_dtype": cache_dtype,
    }


def _scalar_int(value: object, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())


def _identity(sample: Mapping[str, object], name: str) -> str:
    value = str(sample.get(name, "")).strip()
    if not value:
        raise ValueError(f"{name} must be non-empty")
    return value


def build_original_graph(
    sample: Mapping[str, object],
    *,
    tau: float = 0.05,
) -> dict[str, object]:
    """Build exactly the graph retained by the upstream dense implementation.

    The sparse cache is lossless for this operation whenever ``tau`` is not
    below its storage floor. Values at or below ``tau`` are set to zero, just
    as in the upstream implementation.
    """

    if str(sample.get("attention_cache_schema", "")) != FORMAL_CACHE_SCHEMA:
        raise ValueError("only the formal sparse RAGTruth attention cache is supported")
    try:
        threshold = float(tau)
        attention_floor = float(sample["attention_floor"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("tau and attention floor must be finite scalars") from error
    if not math.isfinite(threshold) or not 0 < threshold <= 1:
        raise ValueError("tau must be a finite scalar in (0, 1]")
    if not math.isfinite(attention_floor) or not 0 < attention_floor <= 1:
        raise ValueError("attention floor must be finite in (0, 1]")
    if threshold < attention_floor:
        raise ValueError(
            f"tau={threshold} is below attention cache floor={attention_floor}"
        )

    layers = _scalar_int(sample["num_attention_layers"], "num_attention_layers")
    heads = _scalar_int(sample["num_attention_heads"], "num_attention_heads")
    diagonal = torch.as_tensor(sample["attention_diagonal"])
    if diagonal.ndim != 3 or tuple(diagonal.shape[:2]) != (layers, heads):
        raise ValueError("attention_diagonal must have shape [layers, heads, tokens]")
    device = diagonal.device
    diagonal = diagonal.to(device=device, dtype=torch.float32)
    token_count = int(diagonal.shape[2])
    response_idx = _scalar_int(sample["response_idx"], "response_idx")
    if not 0 < response_idx < token_count:
        raise ValueError("response_idx must split non-empty prompt and response")

    token_ids = torch.as_tensor(sample["token_ids"], device=device).flatten().long()
    raw_labels = torch.as_tensor(sample["y_token"], device=device).flatten()
    if token_ids.numel() != token_count or raw_labels.numel() != token_count:
        raise ValueError("token_ids and y_token must contain one value per node")
    if not bool(torch.isfinite(raw_labels).all()) or bool(
        (~((raw_labels == 0) | (raw_labels == 1))).any()
    ):
        raise ValueError("y_token must be binary")
    labels = raw_labels.long()
    if bool(labels[:response_idx].any()):
        raise ValueError("prompt tokens cannot carry hallucination labels")

    row_ptr = torch.as_tensor(sample["response_row_ptr"], device=device).flatten().long()
    columns = torch.as_tensor(
        sample["response_column_indices"], device=device
    ).flatten().long()
    values = torch.as_tensor(sample["response_values"], device=device).flatten().float()
    response_tokens = token_count - response_idx
    channels = layers * heads
    row_count = channels * response_tokens
    valid_csr = (
        row_ptr.numel() == row_count + 1
        and int(row_ptr[0]) == 0
        and int(row_ptr[-1]) == values.numel()
        and columns.numel() == values.numel()
        and not bool((row_ptr[1:] < row_ptr[:-1]).any())
    )
    if not valid_csr:
        raise ValueError("formal response attention CSR arrays are inconsistent")
    if not bool(torch.isfinite(diagonal).all()) or not bool(torch.isfinite(values).all()):
        raise ValueError("attention values must be finite")
    if bool(((diagonal < 0) | (diagonal > 1)).any()) or bool(
        ((values < 0) | (values > 1)).any()
    ):
        raise ValueError("attention values must lie in [0, 1]")

    lengths = row_ptr[1:] - row_ptr[:-1]
    entry_rows = torch.repeat_interleave(
        torch.arange(row_count, device=device), lengths
    )
    entry_channels = entry_rows // response_tokens
    entry_targets = response_idx + entry_rows.remainder(response_tokens)
    if bool((columns < 0).any()) or bool((columns >= entry_targets).any()):
        raise ValueError("response attention columns violate causal ordering")

    keep = values > threshold
    kept_sources = columns[keep]
    kept_targets = entry_targets[keep]
    kept_channels = entry_channels[keep]
    kept_values = values[keep]
    if kept_values.numel():
        pair_keys, inverse = torch.unique(
            kept_targets * token_count + kept_sources,
            sorted=True,
            return_inverse=True,
        )
        edge_sources = pair_keys.remainder(token_count)
        edge_targets = pair_keys // token_count
        edge_attr = torch.zeros(
            (pair_keys.numel(), channels), device=device, dtype=torch.float32
        )
        flat_positions = inverse * channels + kept_channels
        if torch.unique(flat_positions).numel() != flat_positions.numel():
            raise ValueError("CSR cache contains duplicate channel/source entries")
        edge_attr.view(-1)[flat_positions] = kept_values
        edge_index = torch.stack([edge_sources, edge_targets]).long()
        relation = (edge_sources >= response_idx).long()
        edge_mark = torch.nn.functional.one_hot(relation, num_classes=2).float()
    else:
        edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
        edge_attr = torch.zeros((0, channels), device=device, dtype=torch.float32)
        edge_mark = torch.zeros((0, 2), device=device, dtype=torch.float32)

    node_attr = diagonal.permute(2, 0, 1).reshape(token_count, channels)
    node_role = (
        torch.arange(token_count, device=device) >= response_idx
    ).to(dtype=torch.int8)
    source_id = _identity(sample, "source_id")
    response_id = _identity(sample, "response_id")
    split = str(sample.get("split", sample.get("dataset_split", ""))).strip().casefold()
    if split not in {"train", "test"}:
        raise ValueError("split must be train or test")

    return {
        "schema": GRAPH_SCHEMA,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_id": source_id,
        "response_id": response_id,
        "sample_id": response_id,
        "split": split,
        "response_idx": response_idx,
        "token_ids": token_ids,
        "node_role": node_role,
        "x": node_attr,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mark": edge_mark,
        "y_token": labels,
        "response_label": int(labels[response_idx:].any().item()),
        "tau": threshold,
        "attention_floor": attention_floor,
        "metadata": _graph_metadata(
            layers=layers,
            heads=heads,
            input_policy=str(sample.get("input_policy", "")),
            cache_dtype=str(sample.get("cache_dtype", "")),
        ),
    }


def _atomic_torch_save(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _cpu_graph(graph: Mapping[str, object]) -> dict[str, object]:
    return {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in graph.items()
    }


def _load_saved_graph(path: Path) -> Mapping[str, object]:
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError(f"saved graph must contain a mapping: {path}")
    return loaded


def validate_original_graph(graph: Mapping[str, object]) -> None:
    """Validate one self-describing graph produced by this adapter."""

    required = {
        "schema",
        "source_id",
        "response_id",
        "sample_id",
        "split",
        "response_idx",
        "token_ids",
        "node_role",
        "x",
        "edge_index",
        "edge_attr",
        "edge_mark",
        "y_token",
        "response_label",
        "tau",
        "attention_floor",
        "metadata",
    }
    missing = sorted(required.difference(graph))
    if missing:
        raise ValueError(f"original graph is missing fields: {missing}")
    if graph["schema"] != GRAPH_SCHEMA:
        raise ValueError(f"unsupported original graph schema: {graph['schema']!r}")

    token_ids = torch.as_tensor(graph["token_ids"])
    node_role = torch.as_tensor(graph["node_role"])
    node_attr = torch.as_tensor(graph["x"])
    labels = torch.as_tensor(graph["y_token"])
    edge_index = torch.as_tensor(graph["edge_index"])
    edge_attr = torch.as_tensor(graph["edge_attr"])
    edge_mark = torch.as_tensor(graph["edge_mark"])
    response_idx = int(graph["response_idx"])
    node_count = int(token_ids.numel())
    edge_count = int(edge_index.shape[1]) if edge_index.ndim == 2 else -1
    channel_count = int(node_attr.shape[1]) if node_attr.ndim == 2 else -1

    valid_shapes = (
        token_ids.ndim == 1
        and node_role.shape == (node_count,)
        and node_attr.shape == (node_count, channel_count)
        and labels.shape == (node_count,)
        and edge_index.shape == (2, edge_count)
        and edge_attr.shape == (edge_count, channel_count)
        and edge_mark.shape == (edge_count, 2)
        and 0 < response_idx < node_count
    )
    if not valid_shapes:
        raise ValueError("original graph tensor shapes are inconsistent")
    expected_dtypes = {
        "token_ids": (token_ids.dtype, torch.int64),
        "node_role": (node_role.dtype, torch.int8),
        "x": (node_attr.dtype, torch.float32),
        "edge_index": (edge_index.dtype, torch.int64),
        "edge_attr": (edge_attr.dtype, torch.float32),
        "edge_mark": (edge_mark.dtype, torch.float32),
        "y_token": (labels.dtype, torch.int64),
    }
    wrong_dtypes = [
        f"{name}={actual} (expected {expected})"
        for name, (actual, expected) in expected_dtypes.items()
        if actual != expected
    ]
    if wrong_dtypes:
        raise ValueError("original graph tensor dtype mismatch: " + ", ".join(wrong_dtypes))
    identities = {
        name: str(graph[name]).strip()
        for name in ("source_id", "response_id", "sample_id")
    }
    if not all(identities.values()) or identities["sample_id"] != identities["response_id"]:
        raise ValueError("graph identities must be non-empty and sample_id=response_id")
    if str(graph["split"]).strip().casefold() not in {"train", "test"}:
        raise ValueError("graph split must be train or test")
    tau = float(graph["tau"])
    attention_floor = float(graph["attention_floor"])
    if not (
        math.isfinite(tau)
        and math.isfinite(attention_floor)
        and 0 < attention_floor <= tau <= 1
    ):
        raise ValueError("graph requires 0 < attention_floor <= tau <= 1")
    expected_role = (torch.arange(node_count) >= response_idx).to(node_role.dtype)
    if not torch.equal(node_role.cpu(), expected_role):
        raise ValueError("node_role does not match response_idx")
    if bool((~((labels == 0) | (labels == 1))).any()) or bool(
        labels[:response_idx].any()
    ):
        raise ValueError("y_token must be binary and prompt labels must be zero")
    if int(graph["response_label"]) != int(labels[response_idx:].any().item()):
        raise ValueError("response_label does not match y_token")
    if not bool(torch.isfinite(node_attr).all()) or not bool(
        torch.isfinite(edge_attr).all()
    ):
        raise ValueError("attention graph tensors must be finite")
    if bool(((node_attr < 0) | (node_attr > 1)).any()) or bool(
        ((edge_attr < 0) | (edge_attr > 1)).any()
    ):
        raise ValueError("attention graph tensors must lie in [0, 1]")

    if edge_count:
        source, target = edge_index
        valid_edges = (
            not bool((source < 0).any())
            and not bool((target >= node_count).any())
            and bool((source < target).all())
            and bool((target >= response_idx).all())
        )
        if not valid_edges:
            raise ValueError("edge_index violates causal RP/RR graph constraints")
        pair_keys = source * node_count + target
        if torch.unique(pair_keys).numel() != edge_count:
            raise ValueError("edge_index contains duplicate token-pair edges")
        expected_mark = torch.nn.functional.one_hot(
            (source >= response_idx).long(), num_classes=2
        ).to(dtype=edge_mark.dtype)
        if not torch.equal(edge_mark.cpu(), expected_mark.cpu()):
            raise ValueError("edge_mark does not match RP/RR source roles")
        supported = (edge_attr > tau).any(dim=1)
        censored = (edge_attr > 0) & (edge_attr <= tau)
        if not bool(supported.all()) or bool(censored.any()):
            raise ValueError("each edge_attr needs >tau support and zeroed <=tau channels")

    metadata = graph["metadata"]
    if not isinstance(metadata, Mapping) or metadata.get("schema") != (
        GRAPH_METADATA_SCHEMA
    ):
        raise ValueError("original graph metadata is absent or unsupported")
    layers = int(metadata.get("num_attention_layers", 0))
    heads = int(metadata.get("num_attention_heads", 0))
    channels = int(metadata.get("num_attention_channels", 0))
    if layers < 1 or heads < 1 or channels != layers * heads or channels != channel_count:
        raise ValueError("graph layer/head metadata does not match feature channels")
    input_policy = str(metadata.get("input_policy", "")).strip()
    cache_dtype = str(metadata.get("source_cache_dtype", "")).strip()
    expected_metadata = _graph_metadata(
        layers=layers,
        heads=heads,
        input_policy=input_policy,
        cache_dtype=cache_dtype,
    )
    if not input_policy or not cache_dtype or dict(metadata) != expected_metadata:
        raise ValueError("graph metadata does not match the declared field semantics")


def load_original_graph(path: str | Path) -> dict[str, object]:
    """Safely load and validate one reusable ``.graph.pt`` artifact."""

    graph_path = Path(path).expanduser().resolve()
    graph = dict(_load_saved_graph(graph_path))
    validate_original_graph(graph)
    return graph


def inspect_original_graph(
    graph_or_path: Mapping[str, object] | str | Path,
    *,
    max_nodes: int = 5,
    max_edges: int = 5,
) -> dict[str, object]:
    """Return a JSON-safe description and bounded preview of one graph."""

    if max_nodes < 0 or max_edges < 0:
        raise ValueError("max_nodes and max_edges cannot be negative")
    if isinstance(graph_or_path, (str, Path)):
        graph = load_original_graph(graph_or_path)
    else:
        graph = dict(graph_or_path)
        validate_original_graph(graph)

    token_ids = torch.as_tensor(graph["token_ids"])
    node_role = torch.as_tensor(graph["node_role"])
    node_attr = torch.as_tensor(graph["x"])
    labels = torch.as_tensor(graph["y_token"])
    edge_index = torch.as_tensor(graph["edge_index"])
    edge_attr = torch.as_tensor(graph["edge_attr"])
    edge_mark = torch.as_tensor(graph["edge_mark"])
    metadata = dict(graph["metadata"])
    response_idx = int(graph["response_idx"])
    node_count = int(token_ids.numel())
    edge_count = int(edge_index.shape[1])
    rp_edges = int((edge_mark[:, 0] == 1).sum().item())
    rr_edges = int((edge_mark[:, 1] == 1).sum().item())

    def tensor_description(value: torch.Tensor) -> dict[str, object]:
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
        }

    preview_nodes = min(max_nodes, node_count)
    preview_token_ids = token_ids[:preview_nodes].detach().cpu()
    preview_node_role = node_role[:preview_nodes].detach().cpu()
    preview_labels = labels[:preview_nodes].detach().cpu()
    preview_node_attr = node_attr[:preview_nodes].detach().cpu()
    node_preview = []
    for node_index in range(preview_nodes):
        node_preview.append(
            {
                "node_index": node_index,
                "token_id": int(preview_token_ids[node_index]),
                "role": "response" if int(preview_node_role[node_index]) else "prompt",
                "y_token": int(preview_labels[node_index]),
                "x": preview_node_attr[node_index].tolist(),
            }
        )
    preview_edges = min(max_edges, edge_count)
    preview_edge_index = edge_index[:, :preview_edges].detach().cpu()
    preview_edge_attr = edge_attr[:preview_edges].detach().cpu()
    preview_edge_mark = edge_mark[:preview_edges].detach().cpu()
    edge_preview = []
    for edge_id in range(preview_edges):
        source = int(preview_edge_index[0, edge_id])
        target = int(preview_edge_index[1, edge_id])
        edge_preview.append(
            {
                "edge_id": edge_id,
                "source": source,
                "target": target,
                "relation": "RP" if int(preview_edge_mark[edge_id, 0]) else "RR",
                "edge_mark": preview_edge_mark[edge_id].tolist(),
                "attention_by_channel": preview_edge_attr[edge_id].tolist(),
            }
        )

    return {
        "schema": "original-ragtruth-graph-inspection-v1",
        "graph_schema": str(graph["schema"]),
        "identity": {
            "source_id": str(graph["source_id"]),
            "response_id": str(graph["response_id"]),
            "sample_id": str(graph["sample_id"]),
            "split": str(graph["split"]),
        },
        "dimensions": {
            "nodes": node_count,
            "prompt_nodes": response_idx,
            "response_nodes": node_count - response_idx,
            "edges": edge_count,
            "rp_edges": rp_edges,
            "rr_edges": rr_edges,
            "layers": int(metadata["num_attention_layers"]),
            "heads": int(metadata["num_attention_heads"]),
            "channels": int(metadata["num_attention_channels"]),
        },
        "labels": {
            "hallucinated_response_tokens": int(labels[response_idx:].sum().item()),
            "response_label": int(graph["response_label"]),
        },
        "construction": {
            "tau": float(graph["tau"]),
            "attention_floor": float(graph["attention_floor"]),
            "upstream_commit": str(graph.get("upstream_commit", "")),
            "attention_cache_fingerprint": str(
                graph.get("attention_cache_fingerprint", "")
            ),
        },
        "tensors": {
            name: tensor_description(torch.as_tensor(graph[name]))
            for name in (
                "token_ids",
                "node_role",
                "x",
                "edge_index",
                "edge_attr",
                "edge_mark",
                "y_token",
            )
        },
        "metadata": metadata,
        "field_schema": GRAPH_FIELD_SCHEMA,
        "node_preview": node_preview,
        "edge_preview": edge_preview,
    }


def _graph_matches_source(
    graph: Mapping[str, object],
    *,
    cache_path: Path,
    cache_size: int,
    cache_mtime_ns: int,
    tau: float,
) -> bool:
    try:
        validate_original_graph(graph)
        return bool(
            float(graph.get("tau", float("nan"))) == float(tau)
            and str(graph.get("source_cache_path", "")) == str(cache_path)
            and int(graph.get("source_cache_size", -1)) == cache_size
            and int(graph.get("source_cache_mtime_ns", -1)) == cache_mtime_ns
        )
    except (KeyError, TypeError, ValueError, IndexError, RuntimeError):
        return False


def _available_splits(cache_root: Path) -> tuple[str, ...]:
    available = tuple(
        split
        for split in ("train", "test")
        if (cache_root / split).is_dir()
        and any((cache_root / split).glob("attention_*.pt"))
    )
    if not available:
        raise ValueError(f"no train/test attention_*.pt files found under {cache_root}")
    return available


def prepare_original_graphs(
    cache_root: str | Path,
    output_dir: str | Path,
    *,
    tau: float = 0.05,
    splits: Sequence[str] | None = None,
    device: str | torch.device = "cpu",
    resume: bool = True,
    limit: int | None = None,
    require_complete_cache: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
) -> list[dict[str, object]]:
    """Persist labeled upstream-format graphs and a reusable dataset index."""

    source_root = Path(cache_root).expanduser().resolve()
    destination = Path(output_dir).expanduser().resolve()
    requested_splits = (
        _available_splits(source_root)
        if splits is None
        else tuple(str(value).strip().casefold() for value in splits)
    )
    if not requested_splits or set(requested_splits).difference({"train", "test"}):
        raise ValueError("splits must contain train and/or test")
    if len(set(requested_splits)) != len(requested_splits):
        raise ValueError("splits must not contain duplicates")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    cache_audit = audit_attention_cache(source_root, splits=requested_splits)
    incomplete = [split for split, row in cache_audit.items() if not row["complete"]]
    if require_complete_cache and incomplete:
        raise RuntimeError(f"attention cache is incomplete for split(s): {incomplete}")

    cache_records = discover_attention_cache(source_root, splits=requested_splits)
    if limit is not None:
        selected = []
        for split in requested_splits:
            selected.extend(
                [row for row in cache_records if row.dataset_split == split][:limit]
            )
        cache_records = selected

    index: list[dict[str, object]] = []
    total = len(cache_records)
    for current, cache_record in enumerate(cache_records, start=1):
        cache_path = cache_record.path.resolve()
        stat = cache_path.stat()
        graph_path = (
            destination
            / "graphs"
            / cache_record.dataset_split
            / f"{cache_path.stem}.graph.pt"
        )
        graph_existed = graph_path.is_file()
        state = "prepared"
        graph: Mapping[str, object] | None = None
        if resume and graph_existed:
            try:
                candidate = _load_saved_graph(graph_path)
            except (OSError, RuntimeError, TypeError, ValueError):
                candidate = {}
            if _graph_matches_source(
                candidate,
                cache_path=cache_path,
                cache_size=stat.st_size,
                cache_mtime_ns=stat.st_mtime_ns,
                tau=tau,
            ):
                graph = candidate
                state = "reused"

        if graph is None:
            sample = load_attention_record(
                cache_path,
                device=device,
                mmap=True,
                include_labels=True,
            )
            if "y_token" not in sample:
                raise ValueError(f"RAGTruth token labels are absent: {cache_path}")
            built = build_original_graph(sample, tau=tau)
            built.update(
                {
                    "source_cache_path": str(cache_path),
                    "source_cache_size": int(stat.st_size),
                    "source_cache_mtime_ns": int(stat.st_mtime_ns),
                    "attention_cache_fingerprint": str(
                        sample["attention_cache_fingerprint"]
                    ),
                }
            )
            graph = _cpu_graph(built)
            _atomic_torch_save(graph_path, graph)
            state = "rebuilt" if graph_existed else "prepared"

        labels = torch.as_tensor(graph["y_token"]).flatten()
        response_idx = int(graph["response_idx"])
        edge_mark = torch.as_tensor(graph["edge_mark"])
        relative_graph_path = graph_path.relative_to(destination).as_posix()
        index.append(
            {
                "source_id": str(graph["source_id"]),
                "response_id": str(graph["response_id"]),
                "sample_id": str(graph["response_id"]),
                "split": cache_record.dataset_split,
                "cache_path": str(cache_path),
                "graph_path": relative_graph_path,
                "state": state,
                "num_nodes": int(labels.numel()),
                "num_response_nodes": int(labels.numel() - response_idx),
                "num_edges": int(torch.as_tensor(graph["edge_index"]).shape[1]),
                "num_rp_edges": int((edge_mark[:, 0] == 1).sum().item()),
                "num_rr_edges": int((edge_mark[:, 1] == 1).sum().item()),
                "hallucinated_token_count": int(labels[response_idx:].sum().item()),
                "response_label": int(labels[response_idx:].any().item()),
            }
        )
        if progress_callback is not None:
            progress_callback(current, total)

    index_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in index
    )
    _atomic_text(destination / "index.jsonl", index_text)
    split_counts = {
        split: sum(row["split"] == split for row in index) for split in requested_splits
    }
    manifest = {
        "schema": DATASET_SCHEMA,
        "graph_schema": GRAPH_SCHEMA,
        "upstream_repository": (
            "https://github.com/liuzhishun/Attributed-Graph-Hallucination-Detection"
        ),
        "upstream_commit": UPSTREAM_COMMIT,
        "cache_root": str(source_root),
        "tau": float(tau),
        "contains_token_labels": True,
        "graph_fields": GRAPH_FIELD_SCHEMA,
        "graph_count": len(index),
        "split_counts": split_counts,
        "cache_audit": cache_audit,
        "experiment_scope": (
            "smoke_limit"
            if limit is not None
            else ("official_complete_cache" if not incomplete else "partial_cache")
        ),
        "limit_per_split": limit,
        "index": "index.jsonl",
    }
    _atomic_text(
        destination / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )
    return index
