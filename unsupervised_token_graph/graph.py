"""Construct original-style causal token graphs without evaluation labels."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _float_tensor(value) -> torch.Tensor:
    return torch.as_tensor(value).detach().to(dtype=torch.float32, device="cpu")


def build_token_graph(
    input_ids,
    attention,
    segment_ids,
    *,
    hidden_states=None,
    token_log_probs=None,
    next_token_entropy=None,
    token_stat_valid=None,
    tau: float = 0.05,
    include_prefix_edges: bool = True,
) -> dict[str, object]:
    """Build a token-node graph using causal attention-threshold edges.

    ``edge_index`` follows the original CHARM convention: key/source token
    ``j`` points to query/target token ``i``. Unlike the old preprocessing,
    prefix edges can be retained to expose passage-to-question paths.
    """

    token_ids = torch.as_tensor(input_ids, dtype=torch.long).flatten().cpu()
    segments = torch.as_tensor(segment_ids, dtype=torch.long).flatten().cpu()
    values = _float_tensor(attention)
    token_count = len(token_ids)
    if len(segments) != token_count:
        raise ValueError("segment_ids must align with input_ids")
    if values.ndim != 4 or values.shape[-2:] != (token_count, token_count):
        raise ValueError("attention must have shape (layers, heads, N, N)")
    if torch.any((segments < 0) | (segments > 3)):
        raise ValueError("segment_ids must be in the range zero through three")

    diagonal = values.diagonal(dim1=-2, dim2=-1)
    attention_diagonal = diagonal.permute(2, 0, 1).reshape(token_count, -1)
    segment_one_hot = F.one_hot(segments, num_classes=4).to(torch.float32)
    denominator = max(token_count - 1, 1)
    position = torch.arange(token_count, dtype=torch.float32).unsqueeze(1)
    position = position / denominator
    node_views: dict[str, torch.Tensor] = {
        "attention_diagonal": attention_diagonal,
        "segment_one_hot": segment_one_hot,
        "position": position,
    }

    if hidden_states is not None:
        hidden = _float_tensor(hidden_states)
        if hidden.ndim == 3:
            hidden = hidden[-1]
        if hidden.shape[0] != token_count:
            raise ValueError("hidden_states must align with input_ids")
        node_views["hidden"] = hidden
    if token_log_probs is not None:
        log_probs = _float_tensor(token_log_probs).reshape(token_count, 1)
        node_views["token_log_prob"] = torch.nan_to_num(log_probs)
        valid = (
            _float_tensor(token_stat_valid).reshape(token_count, 1)
            if token_stat_valid is not None
            else torch.isfinite(log_probs).float()
        )
        node_views["token_log_prob_valid"] = valid
    if next_token_entropy is not None:
        entropy = _float_tensor(next_token_entropy).reshape(token_count, 1)
        node_views["next_token_entropy"] = torch.nan_to_num(entropy)
        valid = (
            _float_tensor(token_stat_valid).reshape(token_count, 1)
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

    edge_presence = values.amax(dim=(0, 1)) > float(tau)
    causal = torch.tril(
        torch.ones((token_count, token_count), dtype=torch.bool), diagonal=-1
    )
    edge_presence &= causal
    if not include_prefix_edges:
        edge_presence &= (segments == 3).unsqueeze(1)
    target, source = torch.nonzero(edge_presence, as_tuple=True)
    edge_index = torch.stack((source, target), dim=0)
    edge_values = values[:, :, target, source]
    edge_values = torch.where(
        edge_values > float(tau), edge_values, torch.zeros_like(edge_values)
    )
    edge_attr = edge_values.permute(2, 0, 1).reshape(len(source), -1)
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
        },
    }
