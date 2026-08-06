import json
from pathlib import Path
import tempfile
import unittest

import torch

from topology_flow.model import RankerConfig
from topology_flow.config import TopologyConfig
from topology_flow.pipeline import score_directory, train_directory, validate_directory


def _sample(scale: float, original_idx: int):
    response_idx = 3
    token_count = 6
    attention = torch.zeros((2, 1, token_count, token_count), dtype=torch.float32)
    for layer in range(2):
        attention[layer, 0, 3, :3] = torch.tensor([0.34, 0.33, 0.33])
        attention[layer, 0, 4, :4] = torch.tensor([0.25, 0.25, 0.25, 0.25])
        attention[layer, 0, 5, :5] = torch.tensor([0.20, 0.20, 0.20, 0.20, 0.20])
        attention[layer, 0, 4, 3] *= scale
        attention[layer, 0, 5, 4] *= scale
    return {
        "source_id": f"source-{original_idx}",
        "original_idx": original_idx,
        "response_idx": response_idx,
        "attention": attention,
    }


class TopologyFlowPipelineTests(unittest.TestCase):
    def test_train_and_score_vertical_slice(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_dir = root / "train"
            test_dir = root / "test"
            train_dir.mkdir()
            test_dir.mkdir()
            for index in range(8):
                torch.save(_sample(1.0 + index * 0.02, index), train_dir / f"sample_{index}.pt")
            torch.save(_sample(1.1, 100), test_dir / "sample_100.pt")
            output_dir = root / "training"

            run = train_directory(
                train_dir,
                output_dir,
                ranker=RankerConfig(
                    epochs=4, hidden_dim=12, batch_size=4, learning_rate=5e-3
                ),
            )
            records = score_directory(
                test_dir,
                output_dir / "topology_flow_ranker.pt",
                root / "scores.jsonl",
            )
            persisted = json.loads((root / "scores.jsonl").read_text().strip())

        self.assertEqual(run["samples"], 8)
        self.assertEqual(run["training_pairs"], 24)
        self.assertEqual(len(records), 1)
        self.assertIn("topology_anomaly_score", records[0])
        self.assertEqual(persisted["original_idx"], 100)

    def test_validation_and_checkpoint_config_guard(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            train_dir = root / "train"
            train_dir.mkdir()
            for index in range(4):
                torch.save(_sample(1.0, index), train_dir / f"sample_{index}.pt")
            report = validate_directory(train_dir, limit=2)
            output_dir = root / "training"
            train_directory(
                train_dir,
                output_dir,
                ranker=RankerConfig(epochs=2, hidden_dim=8, batch_size=2),
            )
            with self.assertRaisesRegex(ValueError, "exactly match"):
                score_directory(
                    train_dir,
                    output_dir / "topology_flow_ranker.pt",
                    root / "scores.jsonl",
                    topology=TopologyConfig(mass_cover=0.7),
                )
        self.assertEqual(report["validated_samples"], 2)


if __name__ == "__main__":
    unittest.main()
