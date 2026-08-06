"""Strict, label-blind adapters for persisted attention caches.

The formal RAGTruth attention cache may contain ``y_token`` for downstream
supervised baselines. PATF never copies or reads that field: recognized cache
schemas are loaded through an explicit attention/identity whitelist. This keeps
training label blind without requiring a multi-terabyte cache rewrite.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

FORMAL_SPARSE_CSR_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"
_EXACT_SUPERVISION_FIELDS = {
    "y",
    "label",
    "labels",
    "target_label",
    "token_labels",
    "y_token",
    "hallucination_label",
    "hallucination_labels",
}


def _scalar_int(value: object, name: str) -> int:
    tensor = torch.as_tensor(value)
    if tensor.numel() != 1:
        raise ValueError(f"{name} must be scalar")
    return int(tensor.item())


def assert_label_blind(sample: Mapping[str, object]) -> None:
    """Reject supervision fields in already-derived feature records.

    Raw dense/formal attention caches are handled by :func:`store_from_sample`
    through a whitelist and may physically contain labels. This helper remains
    useful for records that are expected to be intrinsically label free.
    """

    forbidden = []
    for key in sample:
        lowered = str(key).casefold()
        if lowered in _EXACT_SUPERVISION_FIELDS or (
            "halluc" in lowered and ("label" in lowered or "target" in lowered)
        ):
            forbidden.append(str(key))
    if forbidden:
        raise ValueError(
            "topology-flow extraction is label blind; remove supervision fields: "
            + ", ".join(sorted(forbidden))
        )


class AttentionStore(ABC):
    """Read-only response-attention rows for one sample."""

    sample_id: str
    source_id: str
    original_idx: int | None
    layers: int
    heads: int
    token_count: int
    response_idx: int

    @property
    def response_tokens(self) -> int:
        return self.token_count - self.response_idx

    @abstractmethod
    def response_rows(self, layer: int, head: int) -> torch.Tensor:
        """Return ``[response_tokens, token_count]`` rows on CPU as float32."""


@dataclass
class DenseAttentionStore(AttentionStore):
    attention: torch.Tensor
    response_idx: int
    sample_id: str = "unknown"
    source_id: str = "unknown"
    original_idx: int | None = None

    def __post_init__(self) -> None:
        values = torch.as_tensor(self.attention).detach()
        if values.ndim != 4 or values.shape[-1] != values.shape[-2]:
            raise ValueError("attention must have shape [layers, heads, tokens, tokens]")
        if not bool(torch.isfinite(values).all()) or bool((values < 0).any()):
            raise ValueError("attention values must be finite and non-negative")
        self.attention = values
        self.layers, self.heads, self.token_count, _ = values.shape
        if not 0 < int(self.response_idx) < self.token_count:
            raise ValueError("response_idx must split a non-empty prompt and response")
        self.response_idx = int(self.response_idx)

    def response_rows(self, layer: int, head: int) -> torch.Tensor:
        if not 0 <= layer < self.layers or not 0 <= head < self.heads:
            raise IndexError("layer/head out of range")
        return self.attention[layer, head, self.response_idx :, :].to(
            device="cpu", dtype=torch.float32
        )


@dataclass
class SparseCSRAttentionStore(AttentionStore):
    row_ptr: torch.Tensor
    columns: torch.Tensor
    values: torch.Tensor
    layers: int
    heads: int
    token_count: int
    response_idx: int
    sample_id: str = "unknown"
    source_id: str = "unknown"
    original_idx: int | None = None

    def __post_init__(self) -> None:
        self.layers = int(self.layers)
        self.heads = int(self.heads)
        self.token_count = int(self.token_count)
        self.response_idx = int(self.response_idx)
        if self.layers < 1 or self.heads < 1:
            raise ValueError("sparse cache requires at least one layer and head")
        if not 0 < self.response_idx < self.token_count:
            raise ValueError("response_idx must split a non-empty prompt and response")
        self.row_ptr = torch.as_tensor(self.row_ptr, dtype=torch.long).flatten().cpu()
        self.columns = torch.as_tensor(self.columns, dtype=torch.long).flatten().cpu()
        self.values = torch.as_tensor(self.values, dtype=torch.float32).flatten().cpu()
        expected_rows = self.layers * self.heads * self.response_tokens
        if self.row_ptr.numel() != expected_rows + 1:
            raise ValueError("sparse row_ptr length does not match cache dimensions")
        if int(self.row_ptr[0]) != 0 or int(self.row_ptr[-1]) != self.values.numel():
            raise ValueError("sparse row_ptr endpoints are invalid")
        if bool((self.row_ptr[1:] < self.row_ptr[:-1]).any()):
            raise ValueError("sparse row_ptr must be monotone")
        if self.columns.numel() != self.values.numel():
            raise ValueError("sparse columns and values must align")
        if self.columns.numel() and (
            bool((self.columns < 0).any())
            or bool((self.columns >= self.token_count).any())
        ):
            raise ValueError("sparse source indices are out of range")
        if not bool(torch.isfinite(self.values).all()) or bool((self.values < 0).any()):
            raise ValueError("sparse attention values must be finite and non-negative")

    def response_rows(self, layer: int, head: int) -> torch.Tensor:
        if not 0 <= layer < self.layers or not 0 <= head < self.heads:
            raise IndexError("layer/head out of range")
        rows = torch.zeros((self.response_tokens, self.token_count), dtype=torch.float32)
        channel = layer * self.heads + head
        base = channel * self.response_tokens
        for local_query in range(self.response_tokens):
            row_id = base + local_query
            start = int(self.row_ptr[row_id])
            end = int(self.row_ptr[row_id + 1])
            if start == end:
                continue
            source = self.columns[start:end]
            target = self.response_idx + local_query
            if bool((source >= target).any()):
                raise ValueError("sparse cache contains non-causal source indices")
            rows[local_query, source] = self.values[start:end]
        return rows


def _metadata(sample: Mapping[str, object]) -> tuple[str, str, int | None]:
    source_id = str(sample.get("source_id", "unknown"))
    original = sample.get("original_idx")
    original_idx = None if original is None else _scalar_int(original, "original_idx")
    fallback = original_idx if original_idx is not None else source_id
    sample_id = str(sample.get("response_id", sample.get("sample_id", fallback)))
    return sample_id, source_id, original_idx


def _dense_store(sample: Mapping[str, object]) -> DenseAttentionStore:
    sample_id, source_id, original_idx = _metadata(sample)
    if "response_idx" not in sample:
        raise ValueError("dense attention cache is missing response_idx")
    return DenseAttentionStore(
        attention=torch.as_tensor(sample["attention"]),
        response_idx=_scalar_int(sample["response_idx"], "response_idx"),
        sample_id=sample_id,
        source_id=source_id,
        original_idx=original_idx,
    )


def _formal_sparse_store(sample: Mapping[str, object]) -> SparseCSRAttentionStore:
    """Build a store from a label-containing raw cache via a safe whitelist."""

    required = {
        "attention_diagonal",
        "response_row_ptr",
        "response_column_indices",
        "response_values",
        "response_idx",
    }
    missing = sorted(required.difference(sample))
    if missing:
        raise ValueError(f"formal sparse cache is missing fields: {missing}")
    diagonal = torch.as_tensor(sample["attention_diagonal"])
    if diagonal.ndim != 3:
        raise ValueError("formal attention_diagonal must have shape [L,H,N]")
    layers = _scalar_int(
        sample.get("num_attention_layers", diagonal.shape[0]),
        "num_attention_layers",
    )
    heads = _scalar_int(
        sample.get("num_attention_heads", diagonal.shape[1]),
        "num_attention_heads",
    )
    if tuple(diagonal.shape[:2]) != (layers, heads):
        raise ValueError("formal layer/head metadata disagrees with attention_diagonal")
    sample_id, source_id, original_idx = _metadata(sample)
    return SparseCSRAttentionStore(
        row_ptr=torch.as_tensor(sample["response_row_ptr"]),
        columns=torch.as_tensor(sample["response_column_indices"]),
        values=torch.as_tensor(sample["response_values"]),
        layers=layers,
        heads=heads,
        token_count=int(diagonal.shape[-1]),
        response_idx=_scalar_int(sample["response_idx"], "response_idx"),
        sample_id=sample_id,
        source_id=source_id,
        original_idx=original_idx,
    )


def store_from_sample(
    sample: Mapping[str, object], *, require_label_blind: bool = True
) -> AttentionStore:
    """Build the smallest store adapter for a persisted sample mapping.

    ``require_label_blind`` means labels must not enter the returned store. For
    recognized raw attention caches this is guaranteed by explicit field
    whitelisting, so a physically present ``y_token`` is safely ignored.
    """

    schema = str(sample.get("attention_cache_schema", ""))
    if schema == FORMAL_SPARSE_CSR_SCHEMA:
        return _formal_sparse_store(sample)
    if "attention" in sample:
        if require_label_blind:
            assert_label_blind(sample)
        return _dense_store(sample)
    if require_label_blind:
        assert_label_blind(sample)
    raise ValueError("sample contains neither dense attention nor the formal sparse CSR cache")


def load_store(
    path: str | Path,
    *,
    require_label_blind: bool = True,
    mmap: bool = True,
) -> AttentionStore:
    """Load a cache safely; mmap avoids eagerly materialising large tensor files."""

    try:
        sample = torch.load(
            Path(path), map_location="cpu", weights_only=True, mmap=bool(mmap)
        )
    except TypeError as error:
        if mmap:
            raise RuntimeError(
                "this PyTorch build does not support torch.load(..., mmap=True); "
                "upgrade PyTorch or call load_store(..., mmap=False)"
            ) from error
        raise
    if not isinstance(sample, Mapping):
        raise ValueError(f"attention sample must be a mapping: {path}")
    return store_from_sample(sample, require_label_blind=require_label_blind)
