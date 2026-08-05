from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from grounding_flow.graph_index import (
    build_labeled_graph_index,
    export_completed_run_graph_index,
)


class GroundingFlowGraphIndexTests(unittest.TestCase):
    def test_build_labeled_index_keeps_only_identity_split_label_and_path(self) -> None:
        rows = [
            {
                "response_id": "validation-b",
                "pair_id": "pair-b",
                "split": "validation",
                "graph_path": "/graphs/b.graph.pt",
            },
            {
                "response_id": "train-a",
                "pair_id": "pair-a",
                "split": "train",
                "graph_path": "/graphs/a.graph.pt",
            },
            {
                "response_id": "test-c",
                "pair_id": "pair-c",
                "split": "test",
                "graph_path": "/graphs/c.graph.pt",
            },
        ]
        labels = {
            "train-a": 0,
            "validation-b": 1,
            "test-c": 1,
            "unselected-sidecar-row": 0,
        }

        index = build_labeled_graph_index(rows, labels)

        self.assertEqual(
            index,
            [
                {
                    "response_id": "train-a",
                    "pair_id": "pair-a",
                    "split": "train",
                    "label": 0,
                    "graph_path": "/graphs/a.graph.pt",
                },
                {
                    "response_id": "validation-b",
                    "pair_id": "pair-b",
                    "split": "validation",
                    "label": 1,
                    "graph_path": "/graphs/b.graph.pt",
                },
                {
                    "response_id": "test-c",
                    "pair_id": "pair-c",
                    "split": "test",
                    "label": 1,
                    "graph_path": "/graphs/c.graph.pt",
                },
            ],
        )
        self.assertTrue(
            all(
                set(row) == {"response_id", "pair_id", "split", "label", "graph_path"}
                for row in index
            )
        )

    def test_export_completed_run_separates_artifact_and_labeled_indexes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            graph_dir = run_dir / "prepared" / "graphs"
            graph_dir.mkdir(parents=True)
            split_rows: dict[str, list[dict[str, object]]] = {
                "train": [],
                "validation": [],
                "test": [],
            }
            technical_rows: list[dict[str, object]] = []
            label_rows: list[dict[str, object]] = []
            for offset, split in enumerate(("train", "validation", "test")):
                response_id = f"response-{split}"
                cache_split = "test" if split == "test" else "train"
                graph_path = graph_dir / cache_split / f"{response_id}.graph.pt"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                graph_path.write_bytes(b"graph fixture")
                split_rows[split].append(
                    {
                        "response_id": response_id,
                        "pair_id": f"pair-{split}",
                        "partition": split,
                        "graph_path": str(graph_path),
                        "legacy_graph_path": f"/legacy/{response_id}.pt",
                        "response_tokens": 4,
                    }
                )
                technical_rows.append(
                    {
                        "response_id": response_id,
                        "dataset_split": cache_split,
                        "graph_path": str(graph_path),
                        "cache_path": f"/cache/{response_id}.pt",
                        "num_nodes": 10,
                        "num_response_nodes": 4,
                        "num_edges": 12,
                        "num_rp_edges": 8,
                        "num_rr_edges": 4,
                        "num_traces": 20,
                    }
                )
                label_rows.append({"example_id": response_id, "label": offset % 2})

            (graph_dir / "index.json").write_text(
                json.dumps(technical_rows), encoding="utf-8"
            )
            (run_dir / "splits.json").write_text(
                json.dumps({"partitions": split_rows}), encoding="utf-8"
            )
            (run_dir / "score_freeze.json").write_text(
                json.dumps({"state": "scores_frozen_before_label_read"}),
                encoding="utf-8",
            )
            labels_path = Path(temporary) / "evaluation_labels.jsonl"
            labels_path.write_text(
                "\n".join(json.dumps(row) for row in label_rows) + "\n",
                encoding="utf-8",
            )
            (run_dir / "run.json").write_text(
                json.dumps(
                    {"provenance": {"evaluation_label_sidecar": str(labels_path)}}
                ),
                encoding="utf-8",
            )

            result = export_completed_run_graph_index(run_dir)
            labeled = json.loads((graph_dir / "index.json").read_text())
            artifacts = json.loads((graph_dir / "artifact_index.json").read_text())

        self.assertEqual(result["samples"], 3)
        self.assertEqual(
            result["split_counts"], {"train": 1, "validation": 1, "test": 1}
        )
        self.assertEqual(artifacts, technical_rows)
        self.assertTrue(
            all(
                set(row) == {"response_id", "pair_id", "split", "label", "graph_path"}
                for row in labeled
            )
        )
        self.assertEqual(
            {row["response_id"]: row["label"] for row in labeled},
            {
                "response-train": 0,
                "response-validation": 1,
                "response-test": 0,
            },
        )

    def test_export_rejects_run_without_frozen_scores(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "scores.*frozen"),
        ):
            export_completed_run_graph_index(Path(temporary))


if __name__ == "__main__":
    unittest.main()
