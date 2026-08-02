"""Pure-PyTorch CHARM-style masked token-graph autoencoder."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def make_answer_block_mask(
    segment_ids: torch.Tensor,
    *,
    mask_ratio: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Mask one contiguous block of answer tokens and no input tokens."""

    if not 0.0 < mask_ratio <= 1.0:
        raise ValueError("mask_ratio must be in (0, 1]")
    segments = torch.as_tensor(segment_ids, dtype=torch.long)
    answer_indices = torch.nonzero(segments == 3, as_tuple=False).flatten()
    if not len(answer_indices):
        raise ValueError("the graph has no answer tokens")
    masked_count = max(1, round(len(answer_indices) * mask_ratio))
    masked_count = min(masked_count, len(answer_indices))
    possible_starts = len(answer_indices) - masked_count + 1
    start = int(
        torch.randint(possible_starts, (1,), generator=generator).item()
    )
    mask = torch.zeros_like(segments, dtype=torch.bool)
    mask[answer_indices[start : start + masked_count]] = True
    return mask


def masked_reconstruction_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    masked_nodes: torch.Tensor,
    *,
    trim_fraction: float = 0.0,
) -> torch.Tensor:
    """MSE over masked nodes, optionally trimming the largest node residuals."""

    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have identical shapes")
    mask = torch.as_tensor(masked_nodes, dtype=torch.bool, device=predictions.device)
    if mask.ndim != 1 or len(mask) != len(predictions) or not bool(mask.any()):
        raise ValueError("masked_nodes must select at least one node")
    if not 0.0 <= trim_fraction < 1.0:
        raise ValueError("trim_fraction must be in [0, 1)")
    per_node = (predictions[mask] - targets[mask]).square().mean(dim=-1)
    if trim_fraction:
        keep = max(1, int(len(per_node) * (1.0 - trim_fraction)))
        per_node = torch.topk(per_node, keep, largest=False).values
    return per_node.mean()


def _make_mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


class CharmMeanMessagePassing(nn.Module):
    """Original CHARM message/update equations with target in-degree mean."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_mark_dim: int,
        *,
        dropout: float,
    ):
        super().__init__()
        self.message_mlp = _make_mlp(
            node_dim + edge_dim + edge_mark_dim,
            node_dim,
            node_dim,
            dropout,
        )
        self.update_mlp = _make_mlp(
            node_dim * 2,
            node_dim,
            node_dim,
            dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_mark: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        message_input = torch.cat((x[source], edge_attr, edge_mark), dim=-1)
        messages = self.message_mlp(message_input)
        aggregate = x.new_zeros((len(x), x.shape[1]))
        aggregate.index_add_(0, target, messages)
        target_degree = x.new_zeros(len(x))
        target_degree.index_add_(0, target, x.new_ones(len(target)))
        neighbour_mean = aggregate / target_degree.clamp_min(1.0).unsqueeze(1)
        update = self.update_mlp(torch.cat((x, neighbour_mean), dim=-1))
        return x + update


class TokenGraphMaskedAutoencoder(nn.Module):
    """Mask answer-node attributes and reconstruct them through token edges."""

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        edge_mark_dim: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        self.mask_token = nn.Parameter(torch.zeros(node_dim))
        self.input_projection = nn.Linear(node_dim, hidden_dim)
        self.message_layers = nn.ModuleList(
            CharmMeanMessagePassing(
                hidden_dim,
                edge_dim,
                edge_mark_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )
        self.decoder = _make_mlp(hidden_dim, hidden_dim, node_dim, dropout)

    def forward(
        self,
        x,
        edge_index,
        edge_attr,
        edge_mark,
        masked_nodes,
    ):
        mask = torch.as_tensor(masked_nodes, dtype=torch.bool, device=x.device)
        masked_x = torch.where(mask.unsqueeze(1), self.mask_token, x)
        hidden = F.relu(self.input_projection(masked_x))
        for layer in self.message_layers:
            hidden = F.relu(layer(hidden, edge_index, edge_attr, edge_mark))
        return self.decoder(hidden)
