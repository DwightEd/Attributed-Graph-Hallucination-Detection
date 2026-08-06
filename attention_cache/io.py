"""Method-independent sparse-row view over the canonical attention cache loader."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Sequence

import torch

from attention_graph.data import (
    AttentionCacheRecord,
    discover_attention_cache as _discover_attention_cache,
    load_attention_record as _load_attention_record,
)


@dataclass(frozen=True)
class SparseRow:
    source: torch.Tensor
    weight: torch.Tensor
    retained_mass: float


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
    labels: torch.Tensor | None = None

    @property
    def response_tokens(self) -> int:
        return self.token_count - self.response_idx

    @property
    def num_channels(self) -> int:
        return self.num_layers * self.num_heads

    def rows(self, channel: int) -> Iterator[SparseRow]:
        base = channel * self.response_tokens
        for local_query in range(self.response_tokens):
            row = base + local_query
            start = int(self.row_ptr[row])
            end = int(self.row_ptr[row + 1])
            target = self.response_idx + local_query
            source = self.columns[start:end].long()
            weight = self.values[start:end].float()
            keep = (source < target) & (weight > 0)
            source = source[keep]
            weight = weight[keep]
            yield SparseRow(source, weight, float(weight.sum()))


def discover_attention_cache(
    cache_root: str | Path,
    *,
    splits: Sequence[str] = ("train", "test"),
) -> list[AttentionCacheRecord]:
    return _discover_attention_cache(cache_root, splits=splits)


def discover_split(cache_root: str | Path, split: str) -> list[Path]:
    return [
        record.path
        for record in discover_attention_cache(cache_root, splits=(split,))
    ]


def load_attention(path: str | Path) -> AttentionSample:
    """Load a label-blind sparse row view for PATF."""
    path = Path(path).expanduser().resolve()
    record = _load_attention_record(
        path,
        device="cpu",
        mmap=True,
        include_labels=False,
    )
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
    )
