from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from attention_graph.dataset import load_indexed_graph, read_index


class AttentionGraphDatasetTests(unittest.TestCase):
    def test_read_index_filters_split_and_resolves_relative_graph_paths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [
                {
                    "response_id": "train-a",
                    "pair_id": "pair-a",
                    "split": "train",
                    "label": 0,
                    "graph_path": "train/a.graph.pt",
                },
                {
                    "response_id": "test-b",
                    "pair_id": "pair-b",
                    "split": "test",
                    "label": 1,
                    "graph_path": "test/b.graph.pt",
                },
            ]
            index_path = root / "index.json"
            index_path.write_text(json.dumps(rows), encoding="utf-8")

            records = read_index(index_path, split="test")

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["response_id"], "test-b")
        self.assertEqual(records[0]["label"], 1)
        self.assertEqual(records[0]["graph_path"], (root / "test/b.graph.pt").resolve())

    def test_read_index_requires_exactly_the_five_public_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "index.json"
            index_path.write_text(
                json.dumps(
                    [
                        {
                            "response_id": "a",
                            "pair_id": "pair-a",
                            "split": "train",
                            "label": 0,
                            "graph_path": "a.graph.pt",
                            "num_nodes": 10,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "exactly.*graph_path"):
                read_index(index_path)

    def test_read_index_rejects_non_list_json_as_wrong_type(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            index_path = Path(temporary) / "index.json"
            index_path.write_text(json.dumps({"not": "a list"}), encoding="utf-8")

            with self.assertRaisesRegex(TypeError, "JSON list"):
                read_index(index_path)

    def test_load_indexed_graph_delegates_to_strict_graph_loader(self) -> None:
        record = {
            "response_id": "a",
            "pair_id": "pair-a",
            "split": "train",
            "label": 0,
            "graph_path": Path("a.graph.pt"),
        }
        loaded = object()

        with patch("attention_graph.dataset.load_graph", return_value=loaded) as loader:
            result = load_indexed_graph(record, device="cuda", mmap=False)

        self.assertIs(result, loaded)
        loader.assert_called_once_with(Path("a.graph.pt"), device="cuda", mmap=False)


if __name__ == "__main__":
    unittest.main()
