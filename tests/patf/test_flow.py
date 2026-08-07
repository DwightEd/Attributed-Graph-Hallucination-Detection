from pathlib import Path

import torch

from attention_cache import AttentionSample
from patf.config import CorruptionConfig, FlowConfig
from patf.flow import FEATURE_NAMES, extract_flow


def _sample() -> AttentionSample:
    rows = [
        [(0, 0.5), (1, 0.5)],
        [(2, 1.0)],
        [(0, 0.5), (1, 0.5)],
        [(2, 1.0)],
    ]
    row_ptr = [0]
    columns, values = [], []
    for row in rows:
        for source, weight in row:
            columns.append(source)
            values.append(weight)
        row_ptr.append(len(values))
    return AttentionSample(
        path=Path("sample.pt"),
        dataset_split="train",
        sample_id="sample",
        source_id="source",
        original_idx=None,
        response_idx=2,
        token_count=4,
        num_layers=2,
        num_heads=1,
        row_ptr=torch.tensor(row_ptr),
        columns=torch.tensor(columns),
        values=torch.tensor(values),
        diagonal=torch.zeros((2, 1, 4)),
    )


def test_flow_is_cross_layer_not_same_layer_recursion() -> None:
    trajectory = extract_flow(
        _sample(),
        flow=FlowConfig(residual_weight=0.0),
        corruption=CorruptionConfig(),
    )["original"]
    support = FEATURE_NAMES.index("head_center::prompt_support")
    grounded = FEATURE_NAMES.index("head_center::grounded_response_relay")
    assert torch.isclose(trajectory[0, support], torch.tensor(0.5))
    assert torch.isclose(trajectory[0, grounded], torch.tensor(0.0))
    assert trajectory[1, grounded] > trajectory[0, grounded]
    assert trajectory[1, support] > trajectory[0, support]


def test_grounding_erosion_moves_in_expected_direction() -> None:
    trajectories = extract_flow(
        _sample(),
        flow=FlowConfig(residual_weight=0.0),
        corruption=CorruptionConfig(),
        counterfactual=True,
    )
    prompt = FEATURE_NAMES.index("head_center::direct_prompt_mass")
    concentration = FEATURE_NAMES.index("head_center::attention_concentration")
    assert trajectories["eroded"][:, prompt].mean() <= trajectories["original"][:, prompt].mean()
    assert trajectories["eroded"][:, concentration].mean() >= trajectories["original"][:, concentration].mean()
    assert trajectories["original"].shape[1] == len(FEATURE_NAMES)
