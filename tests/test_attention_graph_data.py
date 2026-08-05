from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from attention_graph.data import (
    audit_attention_cache,
    discover_attention_cache,
    load_attention_record,
    load_graph,
    official_partitions,
    prepare_graphs,
)
from attention_graph.graph import GraphBuildConfig
from tests.test_attention_graph import _formal_sample


def _write_sample(path: Path, *, source_id: str, response_id: str, split: str) -> None:
    sample = _formal_sample()
    sample.pop("cache_format", None)
    sample["source_id"] = source_id
    sample["response_id"] = response_id
    sample["split"] = split
    sample["y_token"] = torch.tensor([0, 0, 0, 1, 0, 0], dtype=torch.float32)
    torch.save(sample, path)


class AttentionGraphDataTests(unittest.TestCase):
    def test_cache_audit_accepts_exact_complete_manifest_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            train.mkdir()
            _write_sample(
                train / "attention_0001.pt",
                source_id="s1",
                response_id="r1",
                split="train",
            )
            (train / "manifest.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "matched_samples": 1,
                        "cache_files": 1,
                        "cache_file_names": ["attention_0001.pt"],
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_attention_cache(root, splits=("train",))

        train_summary = summary["train"]
        self.assertTrue(train_summary["manifest_present"])
        self.assertTrue(train_summary["inventory_exact"])
        self.assertTrue(train_summary["complete"])

    def test_cache_audit_reports_in_progress_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            train.mkdir()
            _write_sample(
                train / "attention_0001.pt",
                source_id="s1",
                response_id="r1",
                split="train",
            )
            (train / "manifest.json").write_text(
                json.dumps(
                    {
                        "state": "in_progress",
                        "matched_samples": 2,
                        "saved": 1,
                        "reused": 0,
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_attention_cache(root, splits=("train",))
            discovered = discover_attention_cache(root, splits=("train",))

        self.assertEqual(summary["train"]["manifest_state"], "in_progress")
        self.assertEqual(summary["train"]["observed_file_count"], 1)
        self.assertFalse(summary["train"]["complete"])
        self.assertEqual(len(discovered), 1)

    def test_cache_audit_reports_missing_manifest_as_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "test").mkdir()
            _write_sample(
                root / "test" / "attention_0001.pt",
                source_id="s1",
                response_id="r1",
                split="test",
            )

            summary = audit_attention_cache(root, splits=("test",))

        test_summary = summary["test"]
        self.assertFalse(test_summary["manifest_present"])
        self.assertIsNone(test_summary["manifest_state"])
        self.assertFalse(test_summary["inventory_exact"])
        self.assertFalse(test_summary["complete"])

    def test_cache_audit_marks_declared_inventory_mismatch_incomplete(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            train.mkdir()
            _write_sample(
                train / "attention_0001.pt",
                source_id="s1",
                response_id="r1",
                split="train",
            )
            (train / "manifest.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "matched_samples": 2,
                        "cache_files": 2,
                        "cache_file_names": [
                            "attention_0001.pt",
                            "attention_0002.pt",
                        ],
                    }
                ),
                encoding="utf-8",
            )

            summary = audit_attention_cache(root, splits=("train",))

        train_summary = summary["train"]
        self.assertEqual(train_summary["declared_matched_samples"], 2)
        self.assertEqual(train_summary["declared_cache_files"], 2)
        self.assertFalse(train_summary["inventory_exact"])
        self.assertFalse(train_summary["complete"])

    def test_cache_audit_fails_closed_on_damaged_manifest_json(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            train.mkdir()
            (train / "manifest.json").write_text("{not-json", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "invalid cache manifest JSON"):
                audit_attention_cache(root, splits=("train",))

    def test_partial_cache_discovery_does_not_require_complete_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "train"
            train.mkdir()
            _write_sample(
                train / "attention_0001.pt",
                source_id="s1",
                response_id="r1",
                split="train",
            )
            (train / "manifest.json").write_text(
                json.dumps({"state": "in_progress"}), encoding="utf-8"
            )

            records = discover_attention_cache(root, splits=("train",))

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].dataset_split, "train")
        self.assertEqual(records[0].path.name, "attention_0001.pt")

    def test_loader_hides_labels_unless_evaluation_explicitly_requests_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "attention_0001.pt"
            _write_sample(path, source_id="s1", response_id="r1", split="train")

            training = load_attention_record(path, device="cpu", include_labels=False)
            evaluation = load_attention_record(path, device="cpu", include_labels=True)

        self.assertNotIn("y_token", training)
        self.assertIn("y_token", evaluation)
        self.assertEqual(training["dataset_split"], "train")

    def test_prepare_is_resumable_and_round_trips_sparse_channel_graph(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            output = Path(directory) / "graphs"
            (root / "train").mkdir(parents=True)
            _write_sample(
                root / "train" / "attention_0001.pt",
                source_id="s1",
                response_id="r1",
                split="train",
            )
            config = GraphBuildConfig(selection="threshold", threshold=0.05)
            first_progress: list[tuple[int, int]] = []
            second_progress: list[tuple[int, int]] = []

            first = prepare_graphs(
                cache_root=root,
                output_dir=output,
                config=config,
                splits=("train",),
                build_device="cpu",
                resume=True,
                progress_callback=lambda current, total: first_progress.append(
                    (current, total)
                ),
            )
            second = prepare_graphs(
                cache_root=root,
                output_dir=output,
                config=config,
                splits=("train",),
                build_device="cpu",
                resume=True,
                progress_callback=lambda current, total: second_progress.append(
                    (current, total)
                ),
            )
            graph = load_graph(first[0].graph_path, device="cpu")
            artifact_index = json.loads(
                (output / "artifact_index.json").read_text(encoding="utf-8")
            )
            legacy_index_exists = (output / "index.json").exists()

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(second[0].state, "reused")
        self.assertEqual(graph.num_channels, 4)
        self.assertEqual(graph.trace_edge_id.shape, graph.trace_channel.shape)
        self.assertEqual(graph.trace_channel.shape, graph.trace_value.shape)
        self.assertEqual(first_progress, [(1, 1)])
        self.assertEqual(second_progress, [(1, 1)])
        self.assertEqual(artifact_index[0]["response_id"], "r1")
        self.assertFalse(legacy_index_exists)

    def test_official_partition_holds_test_out_and_groups_train_by_source(self):
        records = []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache"
            output = Path(directory) / "graphs"
            for split in ("train", "test"):
                (root / split).mkdir(parents=True)
            identities = [
                ("train", "a", "r1"),
                ("train", "a", "r2"),
                ("train", "b", "r3"),
                ("train", "c", "r4"),
                ("test", "z", "r5"),
            ]
            for index, (split, source, response) in enumerate(identities):
                _write_sample(
                    root / split / f"attention_{index:04d}.pt",
                    source_id=source,
                    response_id=response,
                    split=split,
                )
            records = prepare_graphs(
                cache_root=root,
                output_dir=output,
                config=GraphBuildConfig(selection="global_topk", top_k=2),
                splits=("train", "test"),
                build_device="cpu",
            )

            partitions = official_partitions(
                records, validation_fraction=0.34, seed=42
            )

        train_sources = {record.source_id for record in partitions["train"]}
        validation_sources = {
            record.source_id for record in partitions["validation"]
        }
        test_sources = {record.source_id for record in partitions["test"]}
        self.assertFalse(train_sources & validation_sources)
        self.assertEqual(test_sources, {"z"})
        self.assertEqual(
            len(partitions["train"]) + len(partitions["validation"]), 4
        )


if __name__ == "__main__":
    unittest.main()
