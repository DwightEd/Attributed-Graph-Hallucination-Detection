"""Contracts for preserving the upstream attributed-graph construction.

These tests intentionally spell out the behavior of
``processed_graphs_attribute.py`` instead of depending on that import-unsafe
script.  The new ``original`` package must adapt the formal sparse RAGTruth
cache without changing the upstream graph definition.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from original.ragtruth_graph import build_original_graph, prepare_original_graphs

ROOT = Path(__file__).resolve().parents[1]


def _dense_fixture() -> tuple[torch.Tensor, int]:
    """Return [layer, head, query, key] attention with known RP/RR edges."""

    layers, heads, token_count, response_idx = 2, 2, 5, 2
    attention = torch.zeros(layers, heads, token_count, token_count)
    flat = attention.reshape(layers * heads, token_count, token_count)

    # Distinct self-attention values make the expected node channel order
    # observable: layer-major, then head-major, exactly as in the upstream file.
    for channel in range(layers * heads):
        for token_idx in range(token_count):
            flat[channel, token_idx, token_idx] = 0.10 + channel * 0.10 + token_idx * 0.01

    # A strong prompt->prompt attention is deliberately present in the dense
    # reference.  The upstream graph builder must discard it.
    flat[:, 1, 0] = torch.tensor([0.90, 0.80, 0.70, 0.60])

    # target=2: one RP edge; source=1 is entirely <= tau and must disappear.
    flat[:, 2, 0] = torch.tensor([0.060, 0.040, 0.010, 0.051])
    flat[:, 2, 1] = torch.tensor([0.020, 0.050, 0.049, 0.010])

    # target=3: two RP edges and one RR edge.
    flat[:, 3, 0] = torch.tensor([0.070, 0.010, 0.010, 0.010])
    flat[:, 3, 1] = torch.tensor([0.010, 0.080, 0.010, 0.010])
    flat[:, 3, 2] = torch.tensor([0.090, 0.010, 0.010, 0.010])

    # target=4: one RP edge and one RR edge.  The other causal pairs stay
    # below/equal to tau so the test also observes natural (unfilled) degree.
    flat[:, 4, 0] = torch.tensor([0.010, 0.020, 0.030, 0.040])
    flat[:, 4, 1] = torch.tensor([0.055, 0.010, 0.010, 0.010])
    flat[:, 4, 2] = torch.tensor([0.040, 0.030, 0.020, 0.010])
    flat[:, 4, 3] = torch.tensor([0.010, 0.200, 0.060, 0.010])
    return attention, response_idx


def _formal_sparse_fixture(*, split: str = "train") -> dict[str, object]:
    """Encode the response rows of ``_dense_fixture`` as the formal CSR cache."""

    attention, response_idx = _dense_fixture()
    layers, heads, token_count, _ = attention.shape
    channels = layers * heads
    flat = attention.reshape(channels, token_count, token_count)
    attention_floor = 0.01

    row_ptr = [0]
    columns: list[int] = []
    values: list[float] = []
    for channel in range(channels):
        for target in range(response_idx, token_count):
            for source in range(target):
                value = float(flat[channel, target, source])
                if value >= attention_floor:
                    columns.append(source)
                    values.append(value)
            row_ptr.append(len(columns))

    return {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "attention_cache_fingerprint": "fixture-original-v1",
        "cache_dtype": "float32",
        "input_policy": "full_context_no_truncation",
        "was_truncated": False,
        "source_id": "source-17",
        "response_id": "response-23",
        "split": split,
        "response_idx": response_idx,
        "token_ids": torch.tensor([101, 102, 201, 202, 203], dtype=torch.long),
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "attention_diagonal": torch.diagonal(
            attention, dim1=-2, dim2=-1
        ).contiguous(),
        "response_row_ptr": torch.tensor(row_ptr, dtype=torch.long),
        "response_column_indices": torch.tensor(columns, dtype=torch.int32),
        "response_values": torch.tensor(values, dtype=torch.float32),
        "attention_floor": attention_floor,
        "y_token": torch.tensor([0, 0, 0, 1, 1], dtype=torch.long),
    }


def _upstream_dense_reference(
    attention: torch.Tensor,
    *,
    response_idx: int,
    token_ids: torch.Tensor,
    y_token: torch.Tensor,
    tau: float,
) -> dict[str, torch.Tensor | int]:
    """Literal small reference for upstream processed_graphs_attribute.py."""

    layers, heads, token_count, token_count_2 = attention.shape
    if token_count != token_count_2:
        raise AssertionError("fixture attention must be square")
    attention_flat = attention.float().reshape(layers * heads, token_count, token_count)
    node_attr = torch.stack(
        [
            attention_flat[channel, torch.arange(token_count), torch.arange(token_count)]
            for channel in range(layers * heads)
        ],
        dim=1,
    )

    sources: list[int] = []
    targets: list[int] = []
    edge_attrs: list[torch.Tensor] = []
    edge_marks: list[torch.Tensor] = []
    for target in range(token_count):
        for source in range(target):
            if source < response_idx and target < response_idx:
                continue
            channel_attention = attention_flat[:, target, source].clone()
            channel_attention[channel_attention <= tau] = 0.0
            if bool((channel_attention > 0).any()):
                sources.append(source)
                targets.append(target)
                edge_attrs.append(channel_attention)
                edge_marks.append(
                    torch.tensor(
                        [1.0, 0.0]
                        if source < response_idx <= target
                        else [0.0, 1.0]
                    )
                )

    return {
        "response_idx": response_idx,
        "token_ids": token_ids.clone(),
        "x": node_attr,
        "edge_index": torch.tensor([sources, targets], dtype=torch.long),
        "edge_attr": torch.stack(edge_attrs).float(),
        "edge_mark": torch.stack(edge_marks).float(),
        "y_token": y_token.clone().long(),
    }


class OriginalRagTruthGraphTests(unittest.TestCase):
    def test_sparse_adapter_exactly_matches_upstream_dense_tau_logic(self):
        sample = _formal_sparse_fixture()
        dense, response_idx = _dense_fixture()
        expected = _upstream_dense_reference(
            dense,
            response_idx=response_idx,
            token_ids=sample["token_ids"],
            y_token=sample["y_token"],
            tau=0.05,
        )

        actual = build_original_graph(sample, tau=0.05)

        for field in ("x", "edge_index", "edge_attr", "edge_mark", "y_token"):
            torch.testing.assert_close(actual[field], expected[field], rtol=0, atol=0)
        self.assertEqual(actual["response_idx"], expected["response_idx"])
        torch.testing.assert_close(actual["token_ids"], expected["token_ids"])

    def test_graph_has_no_prompt_prompt_edges_and_marks_rp_rr_relations(self):
        graph = build_original_graph(_formal_sparse_fixture(), tau=0.05)
        edge_index = graph["edge_index"]
        edge_mark = graph["edge_mark"]

        self.assertEqual(
            edge_index.tolist(),
            [[0, 0, 1, 2, 1, 3], [2, 3, 3, 3, 4, 4]],
        )
        self.assertTrue(bool((edge_index[1] >= graph["response_idx"]).all()))
        for edge_id, source in enumerate(edge_index[0].tolist()):
            expected = [1.0, 0.0] if source < graph["response_idx"] else [0.0, 1.0]
            self.assertEqual(edge_mark[edge_id].tolist(), expected)

    def test_graph_preserves_token_labels_and_dataset_identity(self):
        graph = build_original_graph(_formal_sparse_fixture(split="test"), tau=0.05)

        self.assertEqual(graph["source_id"], "source-17")
        self.assertEqual(graph["response_id"], "response-23")
        self.assertEqual(graph["split"], "test")
        self.assertEqual(graph["y_token"].dtype, torch.long)
        self.assertEqual(graph["y_token"].tolist(), [0, 0, 0, 1, 1])
        self.assertEqual(graph["token_ids"].tolist(), [101, 102, 201, 202, 203])

    def test_graph_is_self_describing_without_reopening_the_attention_cache(self):
        graph = build_original_graph(_formal_sparse_fixture(), tau=0.05)

        self.assertEqual(graph["node_role"].dtype, torch.int8)
        self.assertEqual(graph["node_role"].tolist(), [0, 0, 1, 1, 1])
        self.assertEqual(
            graph["metadata"],
            {
                "schema": "original-ragtruth-attributed-graph-metadata-v1",
                "num_attention_layers": 2,
                "num_attention_heads": 2,
                "num_attention_channels": 4,
                "channel_order": "layer_major_head_minor",
                "channel_index_formula": (
                    "channel = layer * num_attention_heads + head"
                ),
                "edge_direction": "source_key_to_target_query",
                "edge_selection": (
                    "edge iff any attention[layer, head, target, source] > tau; "
                    "channels <= tau are stored as zero"
                ),
                "relation_encoding": {
                    "RP": [1.0, 0.0],
                    "RR": [0.0, 1.0],
                },
                "node_role_encoding": {"prompt": 0, "response": 1},
                "label_coordinate": "global_prompt_then_response_token_index",
                "label_encoding": {"non_hallucinated": 0, "hallucinated": 1},
                "label_source": "RAGTruth y_token from the formal attention cache",
                "input_policy": "full_context_no_truncation",
                "source_cache_dtype": "float32",
            },
        )

    def test_tau_below_sparse_cache_floor_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "tau.*attention.*floor|floor.*tau"):
            build_original_graph(_formal_sparse_fixture(), tau=0.005)

    def test_prepare_persists_labeled_graph_manifest_and_reuses_valid_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache_root = root / "cache"
            output_root = root / "prepared"
            (cache_root / "train").mkdir(parents=True)
            cache_path = cache_root / "train" / "attention_fixture.pt"
            torch.save(_formal_sparse_fixture(), cache_path)

            prepare_original_graphs(
                cache_root=cache_root,
                output_dir=output_root,
                tau=0.05,
                resume=True,
            )
            graph_path = (
                output_root / "graphs" / "train" / "attention_fixture.graph.pt"
            )
            manifest_path = output_root / "manifest.json"
            index_path = output_root / "index.jsonl"
            first_mtime = graph_path.stat().st_mtime_ns

            prepare_original_graphs(
                cache_root=cache_root,
                output_dir=output_root,
                tau=0.05,
                resume=True,
            )

            second_mtime = graph_path.stat().st_mtime_ns
            graph = torch.load(graph_path, map_location="cpu", weights_only=True)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            index = [
                json.loads(line)
                for line in index_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(second_mtime, first_mtime)
        self.assertEqual(manifest["schema"], "original-ragtruth-attributed-graphs-v1")
        self.assertEqual(manifest["tau"], 0.05)
        self.assertEqual(manifest["graph_count"], 1)
        self.assertEqual(manifest["graph_fields"]["x"]["shape"], "[N, C]")
        self.assertEqual(
            manifest["graph_fields"]["edge_index"]["rows"],
            {"0": "source/key token", "1": "target/query token"},
        )
        self.assertEqual(
            manifest["graph_fields"]["y_token"]["encoding"],
            {"0": "non-hallucinated", "1": "hallucinated"},
        )
        self.assertEqual(len(index), 1)
        self.assertEqual(index[0]["source_id"], "source-17")
        self.assertEqual(index[0]["response_id"], "response-23")
        self.assertEqual(index[0]["split"], "train")
        self.assertEqual(
            index[0]["graph_path"],
            "graphs/train/attention_fixture.graph.pt",
        )
        self.assertIn("y_token", graph)
        self.assertEqual(graph["y_token"].tolist(), [0, 0, 0, 1, 1])


class OriginalRunnerContractTests(unittest.TestCase):
    def test_runner_defaults_to_fresh_cache_and_a_stable_graph_directory(self):
        runner = ROOT / "original" / "run_ragtruth_original.sh"
        text = runner.read_text(encoding="utf-8")

        self.assertIn(
            "fresh_attention_c8847872bedf_20260731T074520Z_p876",
            text,
        )
        graph_defaults = [
            line.strip()
            for line in text.splitlines()
            if line.strip().startswith(("GRAPH_DIR=", "GRAPH_ROOT="))
            and ":-" in line
        ]
        self.assertEqual(len(graph_defaults), 1)
        self.assertIn("ragtruth_original_attribute_graphs", graph_defaults[0])
        self.assertIn('ALLOW_PARTIAL_CACHE="${ALLOW_PARTIAL_CACHE:-0}"', text)
        self.assertIn("--require-complete-cache", text)
        for unstable_fragment in ("date ", "TIMESTAMP", "RUN_ID", "RANDOM", "mktemp"):
            self.assertNotIn(unstable_fragment, graph_defaults[0])

    def test_attention_resume_reuses_the_corresponding_hypergraph_directory(self):
        helper = (ROOT / "original" / "prepare_attention_split.sh").read_text(
            encoding="utf-8"
        )

        self.assertIn("fresh_hypergraphs_", helper)
        self.assertIn("RAGTruth/hypergraphs", helper)
        self.assertNotIn("attention_preparation", helper)


if __name__ == "__main__":
    unittest.main()
