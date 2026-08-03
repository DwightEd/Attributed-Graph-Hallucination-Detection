"""GPU-vectorised typed-neighbourhood masked autoencoder.

The module is deliberately label-blind: it receives only compact token-graph
features and reconstructs masked response-token attributes plus their two
incoming attention neighbourhoods (prefix and response history).
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
import torch.nn.functional as F


def _field(graph: Mapping[str, object] | object, name: str):
    """Read a field from the compact graph mapping or a graph dataclass."""

    return graph[name] if isinstance(graph, Mapping) else getattr(graph, name)


def _mlp(input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, output_dim),
    )


def _typed_mean(
    values: torch.Tensor,
    target: torch.Tensor,
    edge_type: torch.Tensor,
    *,
    node_count: int,
    num_edge_types: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Aggregate edge values into ``[target token, relation, feature]``.

    A flattened ``target * relation`` index makes this a single ``index_add``
    operation, avoiding Python loops over tokens, edges, or relation types.
    """

    group = target * num_edge_types + edge_type
    groups = node_count * num_edge_types
    total = values.new_zeros((groups, values.shape[-1]))
    total.index_add_(0, group, values)
    count = values.new_zeros(groups)
    count.index_add_(0, group, values.new_ones(len(group)))
    return (
        (total / count.clamp_min(1).unsqueeze(1)).reshape(
            node_count, num_edge_types, values.shape[-1]
        ),
        count.reshape(node_count, num_edge_types),
    )


class _TypedMessageLayer(nn.Module):
    """One edge-conditioned message-passing layer with relation separation."""

    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_edge_types: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.num_edge_types = num_edge_types
        self.type_embedding = nn.Embedding(num_edge_types, hidden_dim)
        self.message = _mlp(hidden_dim * 2 + edge_dim, hidden_dim, hidden_dim, dropout)
        self.update = _mlp(hidden_dim * (num_edge_types + 1), hidden_dim, hidden_dim, dropout)

    def forward(
        self,
        hidden: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_type: torch.Tensor,
        source_available: torch.Tensor,
    ) -> torch.Tensor:
        source, target = edge_index
        keep = source_available[source]
        source = source[keep]
        target = target[keep]
        edge_type = edge_type[keep]
        attributes = edge_attr[keep]
        if not len(source):
            typed = hidden.new_zeros((len(hidden), self.num_edge_types, hidden.shape[-1]))
        else:
            message_input = torch.cat(
                (hidden[source], attributes, self.type_embedding(edge_type)), dim=-1
            )
            typed, _ = _typed_mean(
                self.message(message_input),
                target,
                edge_type,
                node_count=len(hidden),
                num_edge_types=self.num_edge_types,
            )
        update = self.update(torch.cat((hidden, typed.flatten(1)), dim=-1))
        return hidden + update


class TypedNeighborhoodAutoencoder(nn.Module):
    """Reconstruct masked response tokens and typed incoming neighbourhoods.

    Edge type ``0`` denotes prefix/source-to-response transport and type ``1``
    denotes response-history transport.  The implementation supports more
    relations for ablations while retaining the same vectorised aggregation.
    """

    def __init__(
        self,
        *,
        node_dim: int,
        edge_dim: int,
        num_edge_types: int,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        route_dim: int = 4,
        context_dim: int = 0,
    ) -> None:
        super().__init__()
        if min(node_dim, edge_dim, num_edge_types, hidden_dim, num_layers, route_dim) < 1:
            raise ValueError("all model dimensions and num_layers must be positive")
        if context_dim < 0:
            raise ValueError("context_dim cannot be negative")
        self.node_dim = int(node_dim)
        self.context_dim = int(context_dim)
        self.num_edge_types = int(num_edge_types)
        self.route_dim = int(route_dim)
        self.mask_token = nn.Parameter(torch.zeros(node_dim))
        self.input_projection = nn.Linear(node_dim + context_dim, hidden_dim)
        self.layers = nn.ModuleList(
            _TypedMessageLayer(hidden_dim, edge_dim, num_edge_types, dropout)
            for _ in range(num_layers)
        )
        self.node_decoder = _mlp(hidden_dim, hidden_dim, node_dim, dropout)
        self.neighbour_decoder = _mlp(
            hidden_dim,
            hidden_dim,
            num_edge_types * node_dim * 2,
            dropout,
        )
        self.route_decoder = _mlp(
            hidden_dim,
            hidden_dim,
            num_edge_types * route_dim,
            dropout,
        )

    def forward(
        self,
        graph: Mapping[str, object] | object,
        masked_nodes: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        x = torch.as_tensor(_field(graph, "x"), dtype=torch.float32)
        edge_index = torch.as_tensor(_field(graph, "edge_index"), device=x.device, dtype=torch.long)
        edge_attr = torch.as_tensor(_field(graph, "edge_attr"), device=x.device, dtype=x.dtype)
        edge_type = torch.as_tensor(_field(graph, "edge_type"), device=x.device, dtype=torch.long)
        mask = torch.as_tensor(masked_nodes, device=x.device, dtype=torch.bool)
        if x.ndim != 2 or x.shape[1] != self.node_dim:
            raise ValueError("graph x must have shape [nodes, node_dim]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, edges]")
        if edge_attr.ndim != 2 or len(edge_attr) != edge_index.shape[1]:
            raise ValueError("edge_attr must have one row per edge")
        if edge_type.shape != (edge_index.shape[1],):
            raise ValueError("edge_type must have one value per edge")
        if mask.shape != (len(x),):
            raise ValueError("masked_nodes must align with graph nodes")
        # A masked token cannot contribute its original attributes to a later
        # target through an outgoing/history edge.  Its incoming context remains
        # observable, which is exactly the reconstruction signal.
        masked_x = torch.where(mask[:, None], self.mask_token, x)
        if self.context_dim:
            context = torch.as_tensor(
                _field(graph, "node_context"), device=x.device, dtype=x.dtype
            )
            if context.shape != (len(x), self.context_dim):
                raise ValueError("node_context must have shape [nodes, context_dim]")
            masked_x = torch.cat((masked_x, context), dim=1)
        hidden = F.gelu(self.input_projection(masked_x))
        source_available = ~mask
        for layer in self.layers:
            hidden = F.gelu(layer(hidden, edge_index, edge_attr, edge_type, source_available))

        neighbourhood = self.neighbour_decoder(hidden).reshape(
            len(x), self.num_edge_types, 2, self.node_dim
        )
        return {
            "node_reconstruction": self.node_decoder(hidden),
            "neighborhood_mean": neighbourhood[:, :, 0],
            "neighborhood_log_variance": neighbourhood[:, :, 1],
            "route_stats": self.route_decoder(hidden).reshape(
                len(x), self.num_edge_types, self.route_dim
            ),
        }


def _masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    selected: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target tensors must have identical shapes")
    if prediction.shape[0] != len(selected):
        raise ValueError("prediction must align with the selected-node mask")
    if not bool(selected.any()):
        raise ValueError("at least one response token must be masked")
    per_node = F.smooth_l1_loss(prediction, target, reduction="none").flatten(1).mean(1)
    return per_node[selected].mean()


def typed_reconstruction_loss(
    outputs: Mapping[str, torch.Tensor],
    graph: Mapping[str, object] | object,
    masked_nodes: torch.Tensor,
    *,
    node_weight: float = 1.0,
    neighborhood_weight: float = 1.0,
    route_weight: float = 0.25,
) -> torch.Tensor:
    """Robust reconstruction loss over *masked response tokens only*."""

    prediction = outputs["node_reconstruction"]
    device = prediction.device
    response = torch.as_tensor(_field(graph, "response_mask"), device=device, dtype=torch.bool)
    masked = torch.as_tensor(masked_nodes, device=device, dtype=torch.bool)
    selected = response & masked
    node_target = torch.as_tensor(_field(graph, "x"), device=device, dtype=prediction.dtype)
    mean_target = torch.as_tensor(
        _field(graph, "neighbor_mean_target"), device=device, dtype=prediction.dtype
    )
    log_variance_target = torch.as_tensor(
        _field(graph, "neighbor_log_variance_target"), device=device, dtype=prediction.dtype
    )
    route_target = torch.as_tensor(
        _field(graph, "route_stats_target"), device=device, dtype=prediction.dtype
    )
    node_loss = _masked_smooth_l1(prediction, node_target, selected)
    mean_loss = _masked_smooth_l1(outputs["neighborhood_mean"], mean_target, selected)
    variance_loss = _masked_smooth_l1(
        outputs["neighborhood_log_variance"], log_variance_target, selected
    )
    route_loss = _masked_smooth_l1(outputs["route_stats"], route_target, selected)
    return node_weight * node_loss + neighborhood_weight * (mean_loss + variance_loss) + route_weight * route_loss


@torch.no_grad()
def score_masked_tokens(
    model: TypedNeighborhoodAutoencoder,
    graph: Mapping[str, object] | object,
    masked_nodes: torch.Tensor,
    *,
    neighborhood_weight: float = 0.25,
    route_weight: float = 0.10,
) -> dict[str, object]:
    """Return one label-free reconstruction score for each masked response token."""

    was_training = model.training
    model.eval()
    outputs = model(graph, masked_nodes)
    prediction = outputs["node_reconstruction"]
    device = prediction.device
    response = torch.as_tensor(_field(graph, "response_mask"), device=device, dtype=torch.bool)
    mask = torch.as_tensor(masked_nodes, device=device, dtype=torch.bool)
    selected = response & mask
    if not bool(selected.any()):
        raise ValueError("masked_nodes must select at least one response token")
    node_target = torch.as_tensor(_field(graph, "x"), device=device, dtype=prediction.dtype)
    mean_target = torch.as_tensor(_field(graph, "neighbor_mean_target"), device=device, dtype=prediction.dtype)
    log_variance_target = torch.as_tensor(_field(graph, "neighbor_log_variance_target"), device=device, dtype=prediction.dtype)
    route_target = torch.as_tensor(_field(graph, "route_stats_target"), device=device, dtype=prediction.dtype)
    node_score = F.smooth_l1_loss(
        prediction, node_target, reduction="none"
    ).flatten(1).mean(1)
    neighborhood_score = F.smooth_l1_loss(
        outputs["neighborhood_mean"], mean_target, reduction="none"
    ).flatten(1).mean(1) + F.smooth_l1_loss(
        outputs["neighborhood_log_variance"],
        log_variance_target,
        reduction="none",
    ).flatten(1).mean(1)
    route_score = F.smooth_l1_loss(
        outputs["route_stats"], route_target, reduction="none"
    ).flatten(1).mean(1)
    score = (
        node_score
        + neighborhood_weight * neighborhood_score
        + route_weight * route_score
    )
    if was_training:
        model.train()
    return {
        "original_idx": _field(graph, "original_idx"),
        "source_id": _field(graph, "source_id"),
        "token_idx": torch.nonzero(selected, as_tuple=False).flatten(),
        "scores": score[selected],
        "node_scores": node_score[selected],
        "neighborhood_scores": neighborhood_score[selected],
        "route_scores": route_score[selected],
    }
