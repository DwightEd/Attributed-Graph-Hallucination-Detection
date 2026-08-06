from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch
from .config import TopologyConfig
from .contracts import AttentionStore
from .topology_core import _channel_features, _head_center, _head_iqr
BASE_FEATURE_NAMES = ('direct_prompt_mass', 'grounded_response_relay', 'unsupported_response_feedback', 'unknown_response_feedback', 'discounted_prompt_ancestry', 'expected_prompt_hops', 'retained_causal_mass', 'observed_row_fraction', 'mass_cover_support_size', 'edge_sparsity', 'response_locality', 'weight_concentration', 'prompt_rooted_reachability', 'prompt_source_coverage', 'source_hub_concentration')

def trajectory_feature_names() -> tuple[str, ...]:
    return tuple((f'{stat}::{name}' for stat in ('head_center', 'head_iqr') for name in BASE_FEATURE_NAMES))

@dataclass(frozen=True)
class TopologySignature:
    sample_id: str
    source_id: str
    original_idx: int | None
    response_idx: int
    token_count: int
    trajectory: torch.Tensor
    feature_names: tuple[str, ...]
    phenomena: dict[str, float]
    config: dict[str, object]

    def to_record(self, *, include_trajectory: bool=True) -> dict[str, Any]:
        record: dict[str, Any] = {'sample_id': self.sample_id, 'source_id': self.source_id, 'original_idx': self.original_idx, 'response_idx': self.response_idx, 'token_count': self.token_count, 'response_token_count': self.token_count - self.response_idx, 'feature_names': list(self.feature_names), 'phenomena': dict(self.phenomena), 'config': dict(self.config)}
        if include_trajectory:
            record['trajectory'] = self.trajectory.tolist()
        record['summary'] = summarize_trajectory(self.trajectory, self.feature_names)
        return record

def extract_signature(store: AttentionStore, *, config: TopologyConfig | None=None) -> TopologySignature:
    config = config or TopologyConfig()
    per_layer = []
    for layer in range(store.layers):
        head_features = [_channel_features(store.response_rows(layer, head), response_idx=store.response_idx, config=config) for head in range(store.heads)]
        stacked = torch.stack(head_features)
        per_layer.append(torch.cat((_head_center(stacked, config.head_reducer), _head_iqr(stacked))))
    trajectory = torch.stack(per_layer)
    names = trajectory_feature_names()
    index = {name: offset for offset, name in enumerate(names)}

    def mean(name: str) -> float:
        return float(trajectory[:, index[f'head_center::{name}']].mean())
    phenomena = {'prompt_connection_weakness': 1.0 - mean('direct_prompt_mass'), 'discounted_grounding_loss': 1.0 - mean('discounted_prompt_ancestry'), 'response_self_dependence': mean('grounded_response_relay') + mean('unsupported_response_feedback') + mean('unknown_response_feedback'), 'unsupported_response_feedback': mean('unsupported_response_feedback'), 'unknown_response_feedback': mean('unknown_response_feedback'), 'edge_sparsity': mean('edge_sparsity'), 'response_locality': mean('response_locality'), 'edge_concentration': 0.5 * (mean('weight_concentration') + mean('source_hub_concentration')), 'observed_row_fraction': mean('observed_row_fraction')}
    return TopologySignature(sample_id=store.sample_id, source_id=store.source_id, original_idx=store.original_idx, response_idx=store.response_idx, token_count=store.token_count, trajectory=trajectory, feature_names=names, phenomena=phenomena, config=config.to_dict())

def summarize_trajectory(trajectory: torch.Tensor, feature_names: tuple[str, ...]) -> dict[str, float]:
    values = torch.as_tensor(trajectory, dtype=torch.float32)
    layers = values.shape[0]
    early_count = max(1, layers // 4)
    late_start = max(0, layers - early_count)
    output: dict[str, float] = {}
    for index, name in enumerate(feature_names):
        column = values[:, index]
        output[f'mean::{name}'] = float(column.mean())
        output[f'late::{name}'] = float(column[late_start:].mean())
        output[f'drift::{name}'] = float(column[late_start:].mean() - column[:early_count].mean())
        output[f'max::{name}'] = float(column.max())
    return output
