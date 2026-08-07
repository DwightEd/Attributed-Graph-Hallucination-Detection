"""Shared sparse access to the canonical formal attention cache."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from attention_graph.data import (
    AttentionCacheRecord,
    discover_attention_cache as _discover_attention_cache,
    load_attention_record as _load_attention_record,
)


@dataclass(frozen=True)
class SparseLayer:
    head: torch.Tensor
    query: torch.Tensor
    source: torch.Tensor
    weight: torch.Tensor
    diagonal: torch.Tensor


@dataclass(frozen=True)
class AttentionSample:
    path: Path
    dataset_split: str
    sample_id: str
    source_id: str
    original_idx: int | None
    response_idx: int
    token_count: int
    num_layers: int
    num_heads: int
    row_ptr: torch.Tensor
    columns: torch.Tensor
    values: torch.Tensor
    diagonal: torch.Tensor

    @property
    def response_tokens(self) -> int:
        return self.token_count - self.response_idx

    @property
    def num_channels(self) -> int:
        return self.num_layers * self.num_heads

    def layer(self, layer: int) -> SparseLayer:
        rows_per_layer = self.num_heads * self.response_tokens
        first_row = layer * rows_per_layer
        ptr = self.row_ptr[first_row : first_row + rows_per_layer + 1]
        lengths = ptr[1:] - ptr[:-1]
        row = torch.repeat_interleave(torch.arange(rows_per_layer), lengths)
        start, end = int(ptr[0]), int(ptr[-1])
        source = self.columns[start:end].long()
        weight = self.values[start:end].float()
        head = torch.div(row, self.response_tokens, rounding_mode="floor")
        query = row.remainder(self.response_tokens)
        target = self.response_idx + query
        keep = (source < target) & (weight > 0)
        return SparseLayer(
            head=head[keep],
            query=query[keep],
            source=source[keep],
            weight=weight[keep],
            diagonal=self.diagonal[layer, :, self.response_idx :].float().clamp_min(0),
        )


def discover_attention_cache(
    cache_root: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
) -> list[AttentionCacheRecord]:
    return _discover_attention_cache(cache_root, splits=splits)


def discover_split(cache_root: str | Path, split: str) -> list[Path]:
    return [record.path for record in discover_attention_cache(cache_root, splits=(split,))]


def load_attention(path: str | Path) -> AttentionSample:
    """Load only the label-blind fields used by PATF."""
    path = Path(path).expanduser().resolve()
    record = _load_attention_record(path, device="cpu", mmap=True, include_labels=False)
    diagonal = torch.as_tensor(record["attention_diagonal"])
    original = record.get("original_idx")
    return AttentionSample(
        path=path,
        dataset_split=str(record["dataset_split"]),
        sample_id=str(record["sample_id"]),
        source_id=str(record["source_id"]),
        original_idx=None if original is None else int(torch.as_tensor(original).item()),
        response_idx=int(record["response_idx"]),
        token_count=int(diagonal.shape[-1]),
        num_layers=int(record["num_attention_layers"]),
        num_heads=int(record["num_attention_heads"]),
        row_ptr=torch.as_tensor(record["response_row_ptr"]).long(),
        columns=torch.as_tensor(record["response_column_indices"]),
        values=torch.as_tensor(record["response_values"]),
        diagonal=diagonal,
    )
