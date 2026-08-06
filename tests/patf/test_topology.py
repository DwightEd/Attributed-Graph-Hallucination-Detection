from pathlib import Path

import torch

from attention_cache import load_attention
from patf.config import CorruptionConfig, TopologyConfig
from patf.topology import FEATURE_NAMES, extract_trajectories

from .conftest import write_cache


def test_topology_and_counterfactual_shapes(tmp_path: Path) -> None:
    path = tmp_path / "attention_0.pt"
    write_cache(path, sample_id="0", source_id="s0")
    trajectories = extract_trajectories(
        load_attention(path),
        topology=TopologyConfig(),
        corruption=CorruptionConfig(),
        modes=("incidence", "collapse", "composite"),
    )
    assert set(trajectories) == {
        "original", "incidence", "collapse", "composite"
    }
    assert trajectories["original"].shape == (2, len(FEATURE_NAMES))
    prompt = FEATURE_NAMES.index("head_center::direct_prompt_mass")
    sparse = FEATURE_NAMES.index("head_center::edge_sparsity")
    assert trajectories["incidence"][:, prompt].mean() < trajectories["original"][:, prompt].mean()
    assert trajectories["collapse"][:, sparse].mean() > trajectories["original"][:, sparse].mean()


def test_source_incidence_changes_prompt_ancestry(tmp_path: Path) -> None:
    def save(path: Path, final_source: int) -> None:
        rows = [{0: 1.0}, {2: 1.0}, {3: 1.0}, {final_source: 1.0}]
        row_ptr = [0]
        columns: list[int] = []
        values: list[float] = []
        for row in rows:
            columns.extend(row)
            values.extend(row.values())
            row_ptr.append(len(values))
        torch.save(
            {
                "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
                "attention_cache_fingerprint": f"fingerprint-{path.stem}",
                "cache_dtype": "float32",
                "input_policy": "prompt_response",
                "was_truncated": False,
                "attention_floor": 0.01,
                "token_ids": torch.arange(6),
                "source_id": "s",
                "split": "train",
                "response_id": path.stem,
                "response_idx": 2,
                "num_attention_layers": 1,
                "num_attention_heads": 1,
                "attention_diagonal": torch.zeros((1, 1, 6)),
                "response_row_ptr": torch.tensor(row_ptr),
                "response_column_indices": torch.tensor(columns, dtype=torch.int32),
                "response_values": torch.tensor(values),
            },
            path,
        )

    strong_path = tmp_path / "strong.pt"
    weak_path = tmp_path / "weak.pt"
    save(strong_path, 2)
    save(weak_path, 4)
    kwargs = {
        "topology": TopologyConfig(),
        "corruption": CorruptionConfig(),
    }
    strong = extract_trajectories(load_attention(strong_path), **kwargs)["original"]
    weak = extract_trajectories(load_attention(weak_path), **kwargs)["original"]
    ancestry = FEATURE_NAMES.index("head_center::discounted_prompt_ancestry")
    unsupported = FEATURE_NAMES.index("head_center::unsupported_response_feedback")
    assert strong[:, ancestry].mean() > weak[:, ancestry].mean()
    assert strong[:, unsupported].mean() < weak[:, unsupported].mean()
