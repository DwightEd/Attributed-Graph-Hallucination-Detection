from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from original.train import run_training


def _graph(response_id: str, label: int) -> dict[str, object]:
    return {
        "schema": "original-ragtruth-attributed-graph-v1",
        "source_id": f"source-{response_id}",
        "response_id": response_id,
        "sample_id": response_id,
        "split": "train",
        "response_idx": 2,
        "token_ids": torch.tensor([10, 11, 12]),
        "x": torch.tensor(
            [[0.10, 0.20], [0.20, 0.10], [0.80, 0.70] if label else [0.15, 0.10]]
        ),
        "edge_index": torch.tensor([[0, 1], [2, 2]], dtype=torch.long),
        "edge_attr": torch.tensor([[0.20, 0.10], [0.10, 0.20]]),
        "edge_mark": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        "y_token": torch.tensor([0, 0, label], dtype=torch.long),
        "response_label": label,
    }


class OriginalTrainerTests(unittest.TestCase):
    def test_one_epoch_smoke_saves_checkpoint_history_and_token_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_root = root / "dataset"
            output = root / "training"
            for split in ("train", "test"):
                (graph_root / "graphs" / split).mkdir(parents=True)

            index_records = []
            for sample_index in range(8):
                graph = _graph(f"train-{sample_index}", sample_index % 2)
                relative_path = f"graphs/train/graph-{sample_index}.pt"
                torch.save(
                    graph,
                    graph_root / relative_path,
                )
                index_records.append({"split": "train", "graph_path": relative_path})
            for sample_index in range(4):
                graph = _graph(f"test-{sample_index}", sample_index % 2)
                graph["split"] = "test"
                relative_path = f"graphs/test/graph-{sample_index}.pt"
                torch.save(
                    graph,
                    graph_root / relative_path,
                )
                index_records.append({"split": "test", "graph_path": relative_path})
            (graph_root / "index.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in index_records),
                encoding="utf-8",
            )
            (graph_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "original-ragtruth-attributed-graphs-v1",
                        "experiment_scope": "unit_test_fixture",
                        "tau": 0.05,
                    }
                ),
                encoding="utf-8",
            )
            # Training must follow index.jsonl and ignore stale/unindexed files.
            torch.save({"not": "a graph"}, graph_root / "graphs" / "train" / "stale.pt")

            summary = run_training(
                graph_root=graph_root,
                output_dir=output,
                seeds=(0,),
                validation_fraction=0.25,
                epochs=1,
                patience=1,
                batch_size=2,
                hidden_dim=8,
                gnn_layers=1,
                device="cpu",
                allow_partial_cache=True,
            )

            saved = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            history = json.loads(
                (output / "seed_0" / "history.json").read_text(encoding="utf-8")
            )

            self.assertEqual(summary, saved)
            self.assertEqual(summary["schema"], "original-charm-supervised-token-v1")
            self.assertEqual(summary["seeds"], [0])
            self.assertEqual(summary["test_samples"], 4)
            self.assertEqual(summary["graph_experiment_scope"], "unit_test_fixture")
            self.assertTrue((output / "seed_0" / "checkpoint.pt").is_file())
            self.assertEqual(len(history), 1)
            self.assertIn("test_auroc", summary["per_seed"][0])
            self.assertIn("test_auprc", summary["per_seed"][0])

    def test_training_rejects_non_official_graph_scope_by_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_root = root / "dataset"
            graph_root.mkdir()
            (graph_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema": "original-ragtruth-attributed-graphs-v1",
                        "experiment_scope": "partial_cache",
                        "tau": 0.05,
                    }
                ),
                encoding="utf-8",
            )
            (graph_root / "index.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "partial_cache|official_complete"):
                run_training(
                    graph_root=graph_root,
                    output_dir=root / "training",
                    seeds=(0,),
                    device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
