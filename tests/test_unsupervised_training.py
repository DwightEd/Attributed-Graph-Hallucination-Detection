import inspect
import json
import tempfile
import unittest
from pathlib import Path

import torch

from unsupervised_token_graph.model import TokenGraphMaskedAutoencoder
from unsupervised_token_graph.evaluate_scores import evaluate_score_file
from unsupervised_token_graph.train import (
    collate_token_graphs,
    reconstruction_step,
    run_training,
    score_graph,
    validate_label_free_graph,
)


def _graph(example_id, node_offset=0):
    return {
        "example_id": example_id,
        "pair_id": example_id,
        "x": torch.tensor(
            [
                [1.0 + node_offset, 0.0, 0.0],
                [0.0, 1.0 + node_offset, 0.0],
                [0.0, 0.0, 1.0 + node_offset],
                [0.5, 0.5, 0.5],
            ]
        ),
        "segment_ids": torch.tensor([1, 2, 3, 3]),
        "edge_index": torch.tensor([[0, 1, 2], [2, 2, 3]]),
        "edge_attr": torch.ones((3, 1)),
        "edge_mark": torch.zeros((3, 8)),
    }


class LabelIsolationTests(unittest.TestCase):
    def test_nested_evaluation_field_is_rejected_before_training(self):
        graph = _graph("sample")
        graph["metadata"] = {"label": 1}

        with self.assertRaisesRegex(ValueError, "evaluation field"):
            validate_label_free_graph(graph)

    def test_training_entry_points_have_no_labels_or_y_parameter(self):
        for callable_object in (reconstruction_step, score_graph):
            parameters = inspect.signature(callable_object).parameters
            self.assertNotIn("labels", parameters)
            self.assertNotIn("y", parameters)


class PureTorchBatchTests(unittest.TestCase):
    def test_collation_offsets_edges_and_preserves_graph_boundaries(self):
        batch = collate_token_graphs([_graph("first"), _graph("second", 1)])

        self.assertEqual(tuple(batch["x"].shape), (8, 3))
        self.assertEqual(batch["graph_ptr"].tolist(), [0, 4, 8])
        self.assertEqual(
            batch["edge_index"][:, 3:].tolist(),
            [[4, 5, 6], [6, 6, 7]],
        )
        self.assertEqual(batch["example_ids"], ["first", "second"])

    def test_one_reconstruction_step_and_graph_score_are_finite(self):
        graph = _graph("sample")
        batch = collate_token_graphs([graph])
        model = TokenGraphMaskedAutoencoder(
            node_dim=3,
            edge_dim=1,
            edge_mark_dim=8,
            hidden_dim=6,
            num_layers=1,
            dropout=0.0,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        loss = reconstruction_step(
            model,
            batch,
            optimizer,
            mask_ratio=0.5,
            trim_fraction=0.0,
            generator=torch.Generator().manual_seed(5),
        )
        score = score_graph(model, graph, block_size=1)

        self.assertTrue(torch.isfinite(torch.tensor(loss)))
        self.assertGreaterEqual(score["anomaly_score"], 0.0)
        self.assertEqual(len(score["answer_node_scores"]), 2)

    def test_training_pipeline_uses_pair_split_and_writes_unlabeled_scores(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_dir = root / "graphs"
            output_dir = root / "training"
            graph_dir.mkdir()
            for pair_index in range(5):
                for candidate_index in range(2):
                    graph = _graph(f"pair-{pair_index}-candidate-{candidate_index}")
                    graph["pair_id"] = f"pair-{pair_index}"
                    torch.save(graph, graph_dir / f"graph-{pair_index}-{candidate_index}.pt")
            stale_graph = _graph("stale-candidate")
            stale_graph["pair_id"] = "stale-pair"
            torch.save(stale_graph, graph_dir / "stale.pt")
            (root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "graph_files": [
                            f"graph-{pair_index}-{candidate_index}.pt"
                            for pair_index in range(5)
                            for candidate_index in range(2)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = run_training(
                graph_dir,
                output_dir,
                device="cpu",
                hidden_dim=6,
                num_layers=1,
                dropout=0.0,
                learning_rate=1e-3,
                epochs=1,
                batch_size=2,
                mask_ratio=0.5,
                trim_fraction=0.0,
                validation_fraction=0.2,
                patience=1,
                seed=3,
                score_block_size=1,
            )

            self.assertEqual(summary["state"], "complete")
            self.assertEqual(summary["train_graphs"], 6)
            self.assertEqual(summary["validation_graphs"], 2)
            self.assertEqual(summary["test_graphs"], 2)
            self.assertTrue((output_dir / "best_model.pt").exists())
            self.assertTrue((output_dir / "unsupervised_scores.jsonl").exists())
            score_rows = [
                json.loads(line)
                for line in (output_dir / "unsupervised_scores.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            pair_splits = {}
            for row in score_rows:
                pair_splits.setdefault(row["pair_id"], set()).add(row["split"])
            self.assertTrue(all(len(splits) == 1 for splits in pair_splits.values()))
            self.assertEqual(
                {next(iter(splits)) for splits in pair_splits.values()},
                {"train", "validation", "test"},
            )

    def test_final_evaluation_filters_out_training_scores(self):
        scores = [
            {"example_id": "train-correct", "pair_id": "train", "split": "train", "anomaly_score": 9.0},
            {"example_id": "train-error", "pair_id": "train", "split": "train", "anomaly_score": 0.0},
            {"example_id": "heldout-correct", "pair_id": "heldout", "split": "validation", "anomaly_score": 0.1},
            {"example_id": "heldout-error", "pair_id": "heldout", "split": "validation", "anomaly_score": 1.0},
        ]
        labels = [
            {"example_id": "train-correct", "label": 0},
            {"example_id": "train-error", "label": 1},
            {"example_id": "heldout-correct", "label": 0},
            {"example_id": "heldout-error", "label": 1},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            scores_path = root / "scores.jsonl"
            labels_path = root / "labels.jsonl"
            output_path = root / "evaluation.json"
            scores_path.write_text(
                "\n".join(json.dumps(record) for record in scores) + "\n",
                encoding="utf-8",
            )
            labels_path.write_text(
                "\n".join(json.dumps(record) for record in labels) + "\n",
                encoding="utf-8",
            )

            report = evaluate_score_file(
                scores_path,
                labels_path,
                output_path,
                split="validation",
            )

        self.assertEqual(report["samples"], 2)
        self.assertEqual(report["split"], "validation")
        self.assertEqual(report["score_separation"]["anomaly_score"]["auc"], 1.0)


if __name__ == "__main__":
    unittest.main()
