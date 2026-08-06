from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

import torch

CACHE_SCHEMA = "ragtruth-all-layers-all-heads-sparse-response-csr-v1"


@dataclass(frozen=True)
class SparseRow:
    source: torch.Tensor
    weight: torch.Tensor
    retained_mass: float


@dataclass(frozen=True)
class AttentionSample:
    path: Path
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
            source = self.columns[start:end].long()
            weight = self.values[start:end].float()
            target = self.response_idx + local_query
            keep = (source < target) & (weight > 0)
            source = source[keep]
            weight = weight[keep]
            yield SparseRow(source, weight, float(weight.sum()))


def discover_split(attention_root: str | Path, split: str) -> list[Path]:
    directory = Path(attention_root) / split
    paths = sorted(directory.glob("attention_*.pt"))
    if not paths:
        raise FileNotFoundError(f"no attention_*.pt files in {directory}")
    return paths


def _scalar_int(value: object) -> int:
    return int(torch.as_tensor(value).item())


def load_attention(path: str | Path) -> AttentionSample:
    path = Path(path)
    raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(raw, Mapping) or raw.get("attention_cache_schema") != CACHE_SCHEMA:
        raise ValueError(f"unsupported attention cache: {path}")

    diagonal = torch.as_tensor(raw["attention_diagonal"])
    layers = _scalar_int(raw["num_attention_layers"])
    heads = _scalar_int(raw["num_attention_heads"])
    response_idx = _scalar_int(raw["response_idx"])
    token_count = int(diagonal.shape[-1])
    row_ptr = torch.as_tensor(raw["response_row_ptr"], dtype=torch.long).flatten()
    columns = torch.as_tensor(raw["response_column_indices"], dtype=torch.long).flatten()
    values = torch.as_tensor(raw["response_values"], dtype=torch.float32).flatten()

    expected_rows = layers * heads * (token_count - response_idx)
    if row_ptr.numel() != expected_rows + 1 or int(row_ptr[-1]) != values.numel():
        raise ValueError(f"invalid CSR dimensions: {path}")

    sample_id = str(raw.get("response_id", raw.get("sample_id", path.stem)))
    original = raw.get("original_idx")
    original_idx = None if original is None else _scalar_int(original)
    return AttentionSample(
        path=path,
        sample_id=sample_id,
        source_id=str(raw.get("source_id", "unknown")),
        original_idx=original_idx,
        response_idx=response_idx,
        token_count=token_count,
        num_layers=layers,
        num_heads=heads,
        row_ptr=row_ptr,
        columns=columns,
        values=values,
    )
