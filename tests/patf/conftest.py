from __future__ import annotations

from pathlib import Path

import torch


def write_cache(
    path: Path,
    *,
    sample_id: str,
    source_id: str,
    variant: float = 0.0,
    layers: int = 2,
    heads: int = 2,
) -> None:
    response_idx = 2
    token_count = 5
    rows = [
        {0: 0.6, 1: 0.4},
        {0: 0.3, 1: 0.2, 2: 0.5 + variant},
        {0: 0.2, 1: 0.1, 2: 0.3, 3: 0.4 + variant},
    ]
    row_ptr = [0]
    columns: list[int] = []
    values: list[float] = []
    for _channel in range(layers * heads):
        for row in rows:
            total = sum(row.values())
            for source, weight in row.items():
                columns.append(source)
                values.append(weight / total)
            row_ptr.append(len(values))

    torch.save(
        {
            "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
            "attention_cache_fingerprint": f"fingerprint-{sample_id}",
            "cache_dtype": "float16",
            "input_policy": "prompt_response",
            "was_truncated": False,
            "source_id": source_id,
            "response_id": sample_id,
            "split": path.parent.name if path.parent.name in {"train", "test"} else "train",
            "response_idx": response_idx,
            "num_attention_layers": layers,
            "num_attention_heads": heads,
            "attention_floor": 0.01,
            "token_ids": torch.arange(token_count),
            "attention_diagonal": torch.zeros((layers, heads, token_count), dtype=torch.float16),
            "response_row_ptr": torch.tensor(row_ptr, dtype=torch.int64),
            "response_column_indices": torch.tensor(columns, dtype=torch.int32),
            "response_values": torch.tensor(values, dtype=torch.float16),
            "y_token": torch.tensor([0, 0, 0, 1, 0]),
        },
        path,
    )
