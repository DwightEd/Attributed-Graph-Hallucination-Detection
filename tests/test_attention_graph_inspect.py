from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path

import torch

from attention_graph.data import GRAPH_ARTIFACT_SCHEMA
from attention_graph.graph import AttentionGraph, GraphBuildConfig
from attention_graph.inspect import inspect_run, main, resolve_graph, summarize_graph


def _toy_graph(*, response_id: str = "response-a") -> AttentionGraph:
    return AttentionGraph(
        source_id="source-a",
        response_id=response_id,
        sample_id=response_id,
        response_idx=3,
        num_layers=2,
        num_heads=2,
        attention_floor=0.01,
        node_attr=torch.tensor(
            [
                [0.10, 0.20, 0.30, 0.40],
                [0.20, 0.30, 0.40, 0.50],
                [0.30, 0.40, 0.50, 0.60],
                [0.40, 0.50, 0.60, 0.70],
                [0.50, 0.60, 0.70, 0.80],
            ],
            dtype=torch.float32,
        ),
        node_context=torch.tensor(
            [
                [0.00, 1.0, 0.0],
                [0.25, 1.0, 0.0],
                [0.50, 1.0, 0.0],
                [0.75, 0.0, 1.0],
                [1.00, 0.0, 1.0],
            ],
            dtype=torch.float32,
        ),
        response_mask=torch.tensor([False, False, False, True, True]),
        edge_index=torch.tensor([[0, 2, 3, 1], [3, 3, 4, 4]], dtype=torch.long),
        edge_type=torch.tensor([0, 0, 1, 0], dtype=torch.long),
        edge_score=torch.tensor([0.20, 0.10, 0.30, 0.05]),
        trace_edge_id=torch.tensor([0, 0, 1, 2, 2, 3], dtype=torch.long),
        trace_channel=torch.tensor([0, 2, 1, 0, 3, 2], dtype=torch.long),
        trace_value=torch.tensor([0.40, 0.40, 0.40, 0.60, 0.60, 0.20]),
        token_ids=torch.tensor([10, 11, 12, 13, 14], dtype=torch.long),
        build_config=GraphBuildConfig(
            selection="threshold", threshold=None, max_edges_per_target=None
        ),
    )


def _write_graph(path: Path, graph: AttentionGraph) -> None:
    graph_mapping: dict[str, object] = {}
    for field in fields(graph):
        value = getattr(graph, field.name)
        if isinstance(value, torch.Tensor):
            graph_mapping[field.name] = value.cpu()
        elif isinstance(value, GraphBuildConfig):
            graph_mapping[field.name] = {
                "selection": value.selection,
                "threshold": value.threshold,
                "top_k": value.top_k,
                "max_edges_per_target": value.max_edges_per_target,
                "query_block": value.query_block,
            }
        else:
            graph_mapping[field.name] = value
    config = dict(graph_mapping["build_config"])
    raw_identity = {
        "source_id": graph.source_id,
        "response_id": graph.response_id,
        "dataset_split": "train",
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "attention_cache_fingerprint": "fixture-fingerprint",
        "cache_dtype": "float32",
        "response_idx": graph.response_idx,
        "token_count": graph.num_nodes,
        "num_layers": graph.num_layers,
        "num_heads": graph.num_heads,
        "attention_floor": graph.attention_floor,
        "source_size": 1,
        "source_mtime_ns": 0,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema": GRAPH_ARTIFACT_SCHEMA,
            "build_config": config,
            "raw_identity": raw_identity,
            "graph": graph_mapping,
        },
        path,
    )


class AttentionGraphInspectTests(unittest.TestCase):
    def test_summarize_graph_reports_structure_and_ranked_edges(self) -> None:
        report = summarize_graph(_toy_graph(), top_edges=3, max_targets=8)

        self.assertEqual(report["schema"], "attention-graph-structure-v1")
        self.assertEqual(
            report["dimensions"],
            {
                "nodes": 5,
                "prompt_nodes": 3,
                "response_nodes": 2,
                "edges": 4,
                "rp_edges": 3,
                "rr_edges": 1,
                "layers": 2,
                "heads": 2,
                "channels": 4,
                "traces": 6,
            },
        )
        self.assertEqual(report["tensor_schema"]["edge_index"]["shape"], [2, 4])
        self.assertEqual(report["relation_counts"], {"RP": 3, "RR": 1})
        self.assertAlmostEqual(
            report["attention_statistics"]["self_attention_diagonal"]["median"],
            0.45,
        )
        self.assertTrue(report["censorship"]["cache_censored"])
        self.assertEqual(report["censorship"]["attention_floor"], 0.01)
        self.assertEqual(report["top_edges"][0]["relation"], "RR")
        self.assertEqual(report["top_edges"][0]["source"], 3)
        self.assertEqual(report["top_edges"][0]["target"], 4)
        self.assertEqual(report["top_edges"][0]["causal_lag"], 1)
        self.assertEqual(report["top_edges"][0]["observed_channels"], 2)
        self.assertAlmostEqual(report["top_edges"][0]["edge_score"], 0.30)
        self.assertAlmostEqual(report["top_edges"][0]["retained_attention_sum"], 1.20)
        self.assertEqual(report["per_target"][0]["target"], 3)
        self.assertEqual(report["per_target"][0]["rp_edges"], 2)
        self.assertAlmostEqual(
            report["per_target"][0]["rp_retained_attention_lower_bound"], 0.30
        )

    def test_inspect_run_resolves_copied_graph_from_relative_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            old_graph_path = (
                Path(temporary)
                / "original-run"
                / "prepared"
                / "graphs"
                / "train"
                / "attention_a.graph.pt"
            )
            _write_graph(old_graph_path, _toy_graph(response_id="stale-original"))
            graph_path = (
                run_dir / "prepared" / "graphs" / "train" / "attention_a.graph.pt"
            )
            _write_graph(graph_path, _toy_graph())
            index_path = run_dir / "prepared" / "graphs" / "index.json"
            index_path.write_text(
                json.dumps(
                    [
                        {
                            "source_id": "source-a",
                            "response_id": "response-a",
                            "sample_id": "response-a",
                            "dataset_split": "train",
                            "cache_path": "/old/machine/attention_a.pt",
                            "graph_path": str(old_graph_path),
                            "num_nodes": 5,
                            "num_response_nodes": 2,
                            "num_edges": 4,
                            "num_rp_edges": 3,
                            "num_rr_edges": 1,
                            "num_traces": 6,
                        }
                    ]
                ),
                encoding="utf-8",
            )

            resolved = resolve_graph(run_dir, response_id="response-a")
            report = inspect_run(run_dir, response_id="response-a", top_edges=2)

        self.assertEqual(resolved, graph_path.resolve())
        self.assertEqual(report["inventory"]["indexed_graphs"], 1)
        self.assertEqual(report["inventory"]["split_counts"], {"train": 1})
        self.assertEqual(
            report["selected_graph"]["identity"]["response_id"], "response-a"
        )
        self.assertTrue(report["reusable"]["selected_graph_label_free"])
        self.assertEqual(
            report["reusable"]["loader"], "attention_graph.data.load_graph"
        )

    def test_resolve_graph_rejects_duplicate_response_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            index_path = run_dir / "prepared" / "graphs" / "index.json"
            index_path.parent.mkdir(parents=True)
            duplicate = {
                "response_id": "duplicate",
                "dataset_split": "train",
                "graph_path": "/tmp/one.graph.pt",
            }
            index_path.write_text(json.dumps([duplicate, duplicate]), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate response_id"):
                resolve_graph(run_dir, response_id="duplicate")

    def test_inspector_prefers_artifact_index_over_minimal_labeled_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary) / "run"
            graph_dir = run_dir / "prepared" / "graphs"
            graph_path = graph_dir / "train" / "attention_a.graph.pt"
            _write_graph(graph_path, _toy_graph())
            artifact_index = graph_dir / "artifact_index.json"
            artifact_index.write_text(
                json.dumps(
                    [
                        {
                            "response_id": "response-a",
                            "dataset_split": "train",
                            "graph_path": str(graph_path),
                            "num_nodes": 5,
                            "num_response_nodes": 2,
                            "num_edges": 4,
                            "num_rp_edges": 3,
                            "num_rr_edges": 1,
                            "num_traces": 6,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (graph_dir / "index.json").write_text(
                json.dumps(
                    [
                        {
                            "response_id": "response-a",
                            "pair_id": "source-a",
                            "split": "validation",
                            "label": 1,
                            "graph_path": str(graph_path),
                        }
                    ]
                ),
                encoding="utf-8",
            )

            report = inspect_run(run_dir)

        self.assertEqual(report["graph_index"], str(artifact_index.resolve()))
        self.assertEqual(report["inventory"]["split_counts"], {"train": 1})

    def test_resolve_graph_rejects_index_path_outside_prepared_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "run"
            outside_graph = root / "outside" / "attention_a.graph.pt"
            _write_graph(outside_graph, _toy_graph())
            index_path = run_dir / "prepared" / "graphs" / "index.json"
            index_path.parent.mkdir(parents=True)
            index_path.write_text(
                json.dumps(
                    [
                        {
                            "response_id": "response-a",
                            "dataset_split": "train",
                            "graph_path": str(outside_graph),
                        }
                    ]
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "outside prepared graph directory"):
                resolve_graph(run_dir, response_id="response-a")

    def test_cli_prints_machine_readable_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            graph_path = Path(temporary) / "attention_a.graph.pt"
            _write_graph(graph_path, _toy_graph())
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = main(
                    [
                        "--graph",
                        str(graph_path),
                        "--top-edges",
                        "1",
                        "--max-targets",
                        "1",
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 0)
        self.assertEqual(report["identity"]["response_id"], "response-a")
        self.assertEqual(len(report["top_edges"]), 1)
        self.assertEqual(len(report["per_target"]), 1)


if __name__ == "__main__":
    unittest.main()
