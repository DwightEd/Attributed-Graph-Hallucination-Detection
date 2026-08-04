"""Construct original-style causal token graphs without evaluation labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _tensor_on(value, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(device=device, dtype=dtype)


def build_token_graph(
    input_ids,
    attention,
    segment_ids,
    *,
    hidden_states=None,
    token_log_probs=None,
    next_token_entropy=None,
    token_stat_valid=None,
    edge_presence=None,
    tau: float = 0.05,
    include_prefix_edges: bool = True,
    include_logit_node_features: bool = True,
    pure_attention: bool = False,
) -> dict[str, object]:
    """Build a token-node graph using causal attention-threshold edges.

    ``edge_index`` follows the original CHARM convention: key/source token
    ``j`` points to query/target token ``i``. Unlike the old preprocessing,
    prefix edges can be retained to expose passage-to-question paths.
    """

    values = torch.as_tensor(attention).detach()
    if not values.is_floating_point():
        values = values.to(torch.float32)
    device = values.device
    token_ids = _tensor_on(input_ids, device=device, dtype=torch.long).flatten()
    segments = _tensor_on(segment_ids, device=device, dtype=torch.long).flatten()
    token_count = len(token_ids)
    if len(segments) != token_count:
        raise ValueError("segment_ids must align with input_ids")
    if values.ndim != 4 or values.shape[-2:] != (token_count, token_count):
        raise ValueError("attention must have shape (layers, heads, N, N)")
    if torch.any((segments < 0) | (segments > 3)):
        raise ValueError("segment_ids must be in the range zero through three")

    diagonal = values.diagonal(dim1=-2, dim2=-1).to(torch.float32)
    attention_diagonal = diagonal.permute(2, 0, 1).reshape(token_count, -1)
    node_views: dict[str, torch.Tensor] = {
        "attention_diagonal": attention_diagonal,
    }
    if not pure_attention:
        node_views["segment_one_hot"] = F.one_hot(
            segments, num_classes=4
        ).to(torch.float32)
        denominator = max(token_count - 1, 1)
        node_views["position"] = (
            torch.arange(token_count, dtype=torch.float32, device=device).unsqueeze(1)
            / denominator
        )

    if not pure_attention and hidden_states is not None:
        hidden = _tensor_on(
            hidden_states, device=device, dtype=torch.float32
        )
        if hidden.ndim == 3:
            hidden = hidden[-1]
        if hidden.shape[0] != token_count:
            raise ValueError("hidden_states must align with input_ids")
        node_views["hidden"] = hidden
    effective_logit_features = bool(include_logit_node_features and not pure_attention)
    if effective_logit_features and token_log_probs is not None:
        log_probs = _tensor_on(
            token_log_probs, device=device, dtype=torch.float32
        ).reshape(token_count, 1)
        node_views["token_log_prob"] = torch.nan_to_num(log_probs)
        valid = (
            _tensor_on(
                token_stat_valid, device=device, dtype=torch.float32
            ).reshape(token_count, 1)
            if token_stat_valid is not None
            else torch.isfinite(log_probs).float()
        )
        node_views["token_log_prob_valid"] = valid
    if effective_logit_features and next_token_entropy is not None:
        entropy = _tensor_on(
            next_token_entropy, device=device, dtype=torch.float32
        ).reshape(token_count, 1)
        node_views["next_token_entropy"] = torch.nan_to_num(entropy)
        valid = (
            _tensor_on(
                token_stat_valid, device=device, dtype=torch.float32
            ).reshape(token_count, 1)
            if token_stat_valid is not None
            else torch.isfinite(entropy).float()
        )
        node_views["next_token_entropy_valid"] = valid
    x = torch.cat(list(node_views.values()), dim=1)
    x_view_slices = {}
    view_start = 0
    for name, view in node_views.items():
        view_end = view_start + view.shape[1]
        x_view_slices[name] = (view_start, view_end)
        view_start = view_end

    # Match the previous float32 threshold semantics without promoting the
    # multi-GiB dense attention tensor itself.
    if edge_presence is None:
        edge_presence = values.amax(dim=(0, 1)).to(torch.float32) > float(tau)
    else:
        edge_presence = _tensor_on(
            edge_presence, device=device, dtype=torch.bool
        )
        if edge_presence.shape != (token_count, token_count):
            raise ValueError("edge_presence must have shape (N, N)")
        edge_presence = edge_presence.clone()
    causal = torch.tril(
        torch.ones(
            (token_count, token_count), dtype=torch.bool, device=device
        ),
        diagonal=-1,
    )
    edge_presence &= causal
    if not include_prefix_edges:
        edge_presence &= (segments == 3).unsqueeze(1)
    target, source = torch.nonzero(edge_presence, as_tuple=True)
    edge_index = torch.stack((source, target), dim=0)
    # Index in edge-major order so the advanced-index result is already
    # contiguous as (edges, layers, heads); this avoids another E*L*H copy.
    edge_feature_dim = int(values.shape[0] * values.shape[1])
    edge_attr = values.permute(2, 3, 0, 1)[target, source]
    edge_attr = edge_attr.reshape(len(source), edge_feature_dim).to(torch.float32)
    edge_attr.masked_fill_(edge_attr <= float(tau), 0.0)
    if pure_attention:
        edge_mark = torch.empty((len(source), 0), dtype=torch.float32, device=device)
    else:
        source_mark = F.one_hot(segments[source], num_classes=4)
        target_mark = F.one_hot(segments[target], num_classes=4)
        edge_mark = torch.cat((source_mark, target_mark), dim=1).float()

    return {
        "schema_version": "token_graph_v2",
        "token_ids": token_ids,
        "segment_ids": segments,
        "answer_mask": segments == 3,
        "x": x,
        "x_view_slices": x_view_slices,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_mark": edge_mark,
        "graph_config": {
            "tau": float(tau),
            "include_prefix_edges": bool(include_prefix_edges),
            "include_logit_node_features": effective_logit_features,
            "pure_attention": bool(pure_attention),
        },
    }
