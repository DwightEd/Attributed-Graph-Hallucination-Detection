from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from grounding_flow.graph_index import (
    build_labeled_graph_index,
    export_completed_run_graph_index,
)
from unsupervised_token_graph.data import load_halueval_qa


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

    def test_build_labeled_index_rejects_missing_label_and_duplicate_graph_path(
        self,
    ) -> None:
        row = {
            "response_id": "response-a",
            "pair_id": "pair-a",
            "split": "train",
            "graph_path": "/graphs/shared.graph.pt",
        }
        with self.assertRaisesRegex(ValueError, "label is absent"):
            build_labeled_graph_index([row], {})
        with self.assertRaisesRegex(ValueError, "duplicate graph_path"):
            build_labeled_graph_index(
                [
                    row,
                    {
                        **row,
                        "response_id": "response-b",
                        "pair_id": "pair-b",
                    },
                ],
                {"response-a": 0, "response-b": 1},
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
            graph_digests_before = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in graph_dir.rglob("*.graph.pt")
            }

            result = export_completed_run_graph_index(run_dir)
            repeated = export_completed_run_graph_index(run_dir)
            labeled = json.loads((graph_dir / "index.json").read_text())
            artifacts = json.loads((graph_dir / "artifact_index.json").read_text())
            graph_digests_after = {
                path: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in graph_dir.rglob("*.graph.pt")
            }

        self.assertEqual(result["samples"], 3)
        self.assertEqual(
            result["split_counts"], {"train": 1, "validation": 1, "test": 1}
        )
        self.assertEqual(artifacts, technical_rows)
        self.assertEqual(repeated["samples"], 3)
        self.assertEqual(graph_digests_after, graph_digests_before)
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

    def test_export_validation_failure_keeps_legacy_technical_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            graph_dir = run_dir / "prepared" / "graphs"
            graph_path = graph_dir / "train" / "response-a.graph.pt"
            graph_path.parent.mkdir(parents=True)
            graph_path.write_bytes(b"graph")
            technical = [
                {
                    "response_id": "response-a",
                    "dataset_split": "train",
                    "graph_path": str(graph_path),
                    "num_nodes": 1,
                }
            ]
            (graph_dir / "index.json").write_text(
                json.dumps(technical), encoding="utf-8"
            )
            (run_dir / "splits.json").write_text(
                json.dumps(
                    {
                        "partitions": {
                            "train": [
                                {
                                    "response_id": "response-a",
                                    "pair_id": "pair-a",
                                    "partition": "train",
                                    "graph_path": str(graph_path),
                                }
                            ],
                            "validation": [],
                            "test": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "score_freeze.json").write_text(
                json.dumps({"state": "scores_frozen_before_label_read"}),
                encoding="utf-8",
            )
            labels_path = Path(temporary) / "labels.jsonl"
            labels_path.write_text(
                json.dumps({"example_id": "different-response", "label": 0}) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "label is absent"):
                export_completed_run_graph_index(run_dir, labels_path=labels_path)

            self.assertEqual(
                json.loads((graph_dir / "index.json").read_text()), technical
            )
            self.assertFalse((graph_dir / "artifact_index.json").exists())

    def test_export_rejects_run_without_frozen_scores(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary,
            self.assertRaisesRegex(ValueError, "scores.*frozen"),
        ):
            export_completed_run_graph_index(Path(temporary))

    def test_export_rebuilds_missing_sidecar_from_halueval_qa(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            qa_path = root / "qa_data.json"
            qa_path.write_text(
                json.dumps(
                    [
                        {
                            "knowledge": "Evidence",
                            "question": "Question?",
                            "right_answer": "Supported",
                            "hallucinated_answer": "Unsupported",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            examples, expected_labels = load_halueval_qa(qa_path)
            run_dir = root / "run"
            graph_dir = run_dir / "prepared" / "graphs"
            technical_rows = []
            split_rows = []
            for example in examples:
                graph_path = graph_dir / "train" / f"{example.example_id}.graph.pt"
                graph_path.parent.mkdir(parents=True, exist_ok=True)
                graph_path.write_bytes(b"graph")
                technical_rows.append(
                    {
                        "response_id": example.example_id,
                        "source_id": example.pair_id,
                        "dataset_split": "train",
                        "graph_path": str(graph_path),
                        "num_nodes": 1,
                    }
                )
                split_rows.append(
                    {
                        "response_id": example.example_id,
                        "pair_id": example.pair_id,
                        "partition": "train",
                        "graph_path": str(graph_path),
                    }
                )
            (graph_dir / "artifact_index.json").write_text(
                json.dumps(technical_rows), encoding="utf-8"
            )
            (run_dir / "splits.json").write_text(
                json.dumps(
                    {
                        "partitions": {
                            "train": split_rows,
                            "validation": [],
                            "test": [],
                        }
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "score_freeze.json").write_text(
                json.dumps({"state": "scores_frozen_before_label_read"}),
                encoding="utf-8",
            )

            result = export_completed_run_graph_index(
                run_dir, halueval_qa_path=qa_path
            )
            index = json.loads((graph_dir / "index.json").read_text())

        self.assertEqual(result["label_source"], str(qa_path.resolve()))
        self.assertEqual(
            {str(row["response_id"]): int(row["label"]) for row in index},
            expected_labels,
        )


if __name__ == "__main__":
    unittest.main()
