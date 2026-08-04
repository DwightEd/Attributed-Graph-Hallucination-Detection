"""RED contracts for the label-free legacy HaluEval attention-graph adapter."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from attention_graph.graph import GraphBuildConfig, RP, RR
from attention_graph.halueval import (
    discover_legacy_halueval_records,
    evaluate_halueval_predictions,
    legacy_graph_to_attention_graph,
    legacy_graph_to_formal_attention_cache,
    load_halueval_response_labels,
    prepare_legacy_halueval_graphs,
    split_halueval_pairs,
)
from attention_graph.data import PreparedGraphRecord, load_graph


def _legacy_graph(*, example_id: str = "candidate-a", pair_id: str = "pair-a") -> dict[str, object]:
    """A persisted token_graph_v2 whose dense trace attention was discarded.

    The first four node columns and all four edge columns are the ordered
    ``(layer, head)`` attention channels.  Remaining node columns deliberately
    model old hidden/log-probability/entropy features, which the new adapter
    must not use.
    """

    diagonal = torch.tensor(
        [
            [0.01, 0.02, 0.03, 0.04],
            [0.11, 0.12, 0.13, 0.14],
            [0.21, 0.22, 0.23, 0.24],
            [0.31, 0.32, 0.33, 0.34],
            [0.41, 0.42, 0.43, 0.44],
        ],
        dtype=torch.float32,
    )
    nuisance = torch.tensor(
        [
            [7.0, -1.0, 0.1, 1.0],
            [8.0, -2.0, 0.2, 1.0],
            [9.0, -3.0, 0.3, 1.0],
            [10.0, -4.0, 0.4, 1.0],
            [11.0, -5.0, 0.5, 1.0],
        ],
        dtype=torch.float32,
    )
    return {
        "schema_version": "token_graph_v2",
        "example_id": example_id,
        "pair_id": pair_id,
        "token_ids": torch.tensor([101, 102, 103, 104, 105]),
        "segment_ids": torch.tensor([1, 2, 2, 3, 3]),
        "answer_mask": torch.tensor([False, False, False, True, True]),
        "x": torch.cat((diagonal, nuisance), dim=1),
        "x_view_slices": {
            "attention_diagonal": (0, 4),
            "hidden": (4, 5),
            "token_log_prob": (5, 6),
            "next_token_entropy": (6, 7),
            "token_stat_valid": (7, 8),
        },
        # source -> target, in the old token graph's same ordered channels.
        "edge_index": torch.tensor([[0, 2, 1, 3], [3, 3, 4, 4]]),
        "edge_attr": torch.tensor(
            [
                [0.51, 0.52, 0.53, 0.54],
                [0.61, 0.62, 0.63, 0.64],
                [0.71, 0.72, 0.73, 0.74],
                [0.81, 0.82, 0.83, 0.84],
            ],
            dtype=torch.float32,
        ),
        "graph_config": {"tau": 0.05, "include_prefix_edges": True},
        "extraction_fingerprint": f"fingerprint-{example_id}",
    }


def _trace_metadata(*, example_id: str = "candidate-a", pair_id: str = "pair-a") -> dict[str, object]:
    return {
        "example_id": example_id,
        "pair_id": pair_id,
        "dataset": "halueval_qa",
        "input_ids": torch.tensor([101, 102, 103, 104, 105]),
        "segment_ids": torch.tensor([1, 2, 2, 3, 3]),
        "attention_shape": [2, 2, 5, 5],
        "attention_storage": "discarded_after_postprocessing",
        "edge_threshold": 0.05,
        "extraction_fingerprint": f"fingerprint-{example_id}",
    }


def _field(value: object, name: str):
    return value[name] if isinstance(value, dict) else getattr(value, name)


class LegacyTokenGraphConversionTests(unittest.TestCase):
    def test_conversion_builds_source_sorted_csr_in_small_cpu_chunks(self):
        legacy = _legacy_graph()
        # Exercise multiple chunks while preserving a non-source-sorted legacy
        # edge order and a fully censored channel.
        permutation = torch.tensor([2, 0, 3, 1])
        legacy["edge_index"] = legacy["edge_index"][:, permutation]
        legacy["edge_attr"] = legacy["edge_attr"][permutation]
        legacy["edge_attr"][:, 1] = 0.0

        formal = legacy_graph_to_formal_attention_cache(
            legacy,
            _trace_metadata(),
            conversion_device="cpu",
            conversion_chunk_edges=1,
            storage_dtype=torch.float32,
        )

        # 4 channels x 2 response queries; channel 1 is entirely censored.
        self.assertEqual(formal["response_row_ptr"].tolist(), [0, 2, 4, 4, 4, 6, 8, 10, 12])
        self.assertEqual(
            formal["response_column_indices"].tolist(),
            [0, 2, 1, 3, 0, 2, 1, 3, 0, 2, 1, 3],
        )
        self.assertTrue(bool((formal["response_values"] > 0).all()))
        self.assertTrue(
            all(
                formal[name].device.type == "cpu"
                for name in (
                    "attention_diagonal",
                    "response_row_ptr",
                    "response_column_indices",
                    "response_values",
                    "token_ids",
                )
            )
        )

    def test_conversion_keeps_chunk_row_bookkeeping_in_tensors(self):
        """Chunking must not synchronize each active CSR row into Python."""

        legacy = _legacy_graph()
        with mock.patch.object(torch.Tensor, "tolist", side_effect=AssertionError):
            formal = legacy_graph_to_formal_attention_cache(
                legacy,
                _trace_metadata(),
                conversion_device="cpu",
                conversion_chunk_edges=1,
            )

        self.assertEqual(formal["response_row_ptr"].numel(), 9)

    def test_conversion_rejects_duplicate_token_pair_with_disjoint_channels(self):
        legacy = _legacy_graph()
        legacy["edge_attr"][0] = torch.tensor([0.51, 0.0, 0.0, 0.0])
        legacy["edge_index"] = torch.cat((legacy["edge_index"], torch.tensor([[0], [3]])), dim=1)
        legacy["edge_attr"] = torch.cat(
            (legacy["edge_attr"], torch.tensor([[0.0, 0.52, 0.0, 0.0]])), dim=0
        )

        with self.assertRaisesRegex(ValueError, "duplicate"):
            legacy_graph_to_formal_attention_cache(
                legacy,
                _trace_metadata(),
                conversion_device="cpu",
                conversion_chunk_edges=1,
            )

    def test_discarded_dense_trace_converts_ordered_diagonal_and_rp_rr_traces(self):
        legacy = _legacy_graph()
        trace = _trace_metadata()

        graph = legacy_graph_to_attention_graph(
            legacy,
            trace,
            GraphBuildConfig(selection="threshold", threshold=0.05),
            device="cpu",
        )

        expected_diagonal = legacy["x"][:, :4]
        torch.testing.assert_close(_field(graph, "node_attr"), expected_diagonal)
        self.assertEqual(_field(graph, "num_layers"), 2)
        self.assertEqual(_field(graph, "num_heads"), 2)
        self.assertEqual(_field(graph, "response_idx"), 3)
        self.assertEqual(_field(graph, "response_mask").tolist(), [False, False, False, True, True])

        edge_index = _field(graph, "edge_index")
        self.assertEqual(
            {tuple(edge) for edge in edge_index.t().tolist()},
            {(0, 3), (2, 3), (1, 4), (3, 4)},
        )
        self.assertTrue(bool(_field(graph, "response_mask")[edge_index[1]].all()))
        self.assertTrue(bool((edge_index[0] < edge_index[1]).all()))
        edge_type = _field(graph, "edge_type")
        self.assertTrue(bool((edge_type[:2] == RP).all()))
        self.assertTrue(bool((edge_type[2:] == torch.tensor([RP, RR])).all()))

        # Each retained old edge becomes four explicit (edge, layer/head) traces.
        self.assertEqual(_field(graph, "trace_edge_id").numel(), 16)
        self.assertEqual(_field(graph, "trace_channel").tolist(), [0, 1, 2, 3] * 4)
        torch.testing.assert_close(
            _field(graph, "trace_value"), legacy["edge_attr"].reshape(-1)
        )

    def test_conversion_is_invariant_to_old_hidden_logprob_and_entropy_features(self):
        legacy = _legacy_graph()
        changed = _legacy_graph()
        changed["x"][:, 4:] = torch.tensor(
            [[-99.0, 999.0, -9.0, 0.0]] * 5, dtype=torch.float32
        )
        config = GraphBuildConfig(selection="threshold", threshold=0.05)

        original = legacy_graph_to_attention_graph(legacy, _trace_metadata(), config)
        converted = legacy_graph_to_attention_graph(changed, _trace_metadata(), config)

        for name in (
            "node_attr",
            "edge_index",
            "edge_type",
            "edge_score",
            "trace_edge_id",
            "trace_channel",
            "trace_value",
        ):
            torch.testing.assert_close(_field(original, name), _field(converted, name))

    def test_conversion_cannot_lower_the_legacy_tau(self):
        with self.assertRaisesRegex(ValueError, "tau|threshold|floor"):
            legacy_graph_to_attention_graph(
                _legacy_graph(),
                _trace_metadata(),
                GraphBuildConfig(selection="threshold", threshold=0.049),
            )

    def test_conversion_rejects_non_suffix_answers_and_mismatched_identity(self):
        non_suffix = _legacy_graph()
        non_suffix["segment_ids"] = torch.tensor([1, 2, 3, 2, 3])
        non_suffix["answer_mask"] = non_suffix["segment_ids"] == 3
        bad_trace = _trace_metadata()
        bad_trace["segment_ids"] = non_suffix["segment_ids"]
        with self.assertRaisesRegex(ValueError, "answer.*suffix|contiguous"):
            legacy_graph_to_attention_graph(non_suffix, bad_trace, GraphBuildConfig())

        mismatched_fingerprint = _trace_metadata()
        mismatched_fingerprint["extraction_fingerprint"] = "other-run"
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            legacy_graph_to_attention_graph(
                _legacy_graph(), mismatched_fingerprint, GraphBuildConfig()
            )

        mismatched_identity = _trace_metadata(example_id="candidate-b")
        with self.assertRaisesRegex(ValueError, "example_id|identity"):
            legacy_graph_to_attention_graph(
                _legacy_graph(), mismatched_identity, GraphBuildConfig()
            )

    def test_conversion_uses_pair_source_and_treats_zero_channels_as_censored(self):
        legacy = _legacy_graph()
        legacy["edge_attr"][0, 0] = 0.0

        graph = legacy_graph_to_attention_graph(legacy, _trace_metadata(), GraphBuildConfig())

        self.assertEqual(_field(graph, "source_id"), legacy["pair_id"])
        self.assertEqual(_field(graph, "trace_edge_id").numel(), 15)
        self.assertTrue(bool((_field(graph, "trace_value") > 0).all()))

    def test_conversion_rejects_retained_channels_at_or_below_legacy_tau(self):
        legacy = _legacy_graph()
        legacy["edge_attr"][0, 0] = legacy["graph_config"]["tau"]

        with self.assertRaisesRegex(ValueError, "tau|floor|censored"):
            legacy_graph_to_attention_graph(legacy, _trace_metadata(), GraphBuildConfig())

    def test_conversion_ignores_causal_prefix_edges_without_tracing_them(self):
        legacy = _legacy_graph()
        legacy["edge_index"] = torch.cat(
            (legacy["edge_index"], torch.tensor([[0], [1]], dtype=torch.long)), dim=1
        )
        legacy["edge_attr"] = torch.cat(
            (legacy["edge_attr"], torch.tensor([[0.91, 0.92, 0.93, 0.94]])), dim=0
        )

        graph = legacy_graph_to_attention_graph(legacy, _trace_metadata(), GraphBuildConfig())

        self.assertEqual(_field(graph, "trace_edge_id").numel(), 16)
        self.assertNotIn((0, 1), {tuple(edge) for edge in _field(graph, "edge_index").t().tolist()})

    def test_conversion_preserves_channel_coo_under_edge_reordering_and_censoring(self):
        legacy = _legacy_graph()
        permutation = torch.tensor([3, 1, 0, 2])
        legacy["edge_index"] = legacy["edge_index"][:, permutation]
        legacy["edge_attr"] = legacy["edge_attr"][permutation]
        legacy["edge_attr"][1, 2] = 0.0

        graph = legacy_graph_to_attention_graph(legacy, _trace_metadata(), GraphBuildConfig())

        expected = {
            (int(legacy["edge_index"][0, edge]), int(legacy["edge_index"][1, edge]), channel): float(legacy["edge_attr"][edge, channel])
            for edge in range(legacy["edge_index"].shape[1])
            for channel in range(legacy["edge_attr"].shape[1])
            if legacy["edge_attr"][edge, channel] != 0
        }
        actual = {
            (
                int(graph.edge_index[0, graph.trace_edge_id[trace]]),
                int(graph.edge_index[1, graph.trace_edge_id[trace]]),
                int(graph.trace_channel[trace]),
            ): float(graph.trace_value[trace])
            for trace in range(graph.trace_value.numel())
        }
        self.assertEqual(set(actual), set(expected))
        for key, value in expected.items():
            self.assertAlmostEqual(actual[key], value)
        self.assertEqual(graph.trace_value.numel(), len(expected))

    def test_conversion_handles_chaotic_edges_and_an_entirely_censored_channel(self):
        legacy = _legacy_graph()
        legacy["edge_index"] = legacy["edge_index"][:, torch.tensor([2, 0, 3, 1])]
        legacy["edge_attr"] = legacy["edge_attr"][torch.tensor([2, 0, 3, 1])]
        legacy["edge_attr"][:, 1] = 0.0

        graph = legacy_graph_to_attention_graph(legacy, _trace_metadata(), GraphBuildConfig())

        actual = {
            (
                int(graph.edge_index[0, graph.trace_edge_id[index]]),
                int(graph.edge_index[1, graph.trace_edge_id[index]]),
                int(graph.trace_channel[index]),
            ): float(graph.trace_value[index])
            for index in range(graph.trace_value.numel())
        }
        expected = {
            (int(legacy["edge_index"][0, edge]), int(legacy["edge_index"][1, edge]), channel): float(legacy["edge_attr"][edge, channel])
            for edge in range(legacy["edge_index"].shape[1])
            for channel in range(legacy["edge_attr"].shape[1])
            if legacy["edge_attr"][edge, channel] != 0
        }
        self.assertEqual(actual, expected)
        self.assertNotIn(1, graph.trace_channel.tolist())


class ManifestAndPairSplitTests(unittest.TestCase):
    def _write_artifact_pair(self, root: Path, name: str, *, pair_id: str, trace: bool = True) -> None:
        graph_dir, trace_dir = root / "graphs", root / "traces"
        graph_dir.mkdir(exist_ok=True)
        trace_dir.mkdir(exist_ok=True)
        torch.save(_legacy_graph(example_id=name, pair_id=pair_id), graph_dir / f"{name}.pt")
        if trace:
            torch.save(_trace_metadata(example_id=name, pair_id=pair_id), trace_dir / f"{name}.pt")

    def test_manifest_discovery_rejects_orphans_and_marks_full_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_artifact_pair(root, "a", pair_id="pair-1")
            self._write_artifact_pair(root, "b", pair_id="pair-2", trace=False)
            self._write_artifact_pair(root, "orphan", pair_id="pair-x")
            (root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "graph_dir": str(root / "graphs"),
                        "trace_dir": str(root / "traces"),
                        "graph_files": ["a.pt", "b.pt"],
                        "example_ids": ["a", "b"],
                    }
                ),
                encoding="utf-8",
            )

            records = discover_legacy_halueval_records(root)

        by_id = {_field(record, "response_id"): record for record in records}
        self.assertEqual(set(by_id), {"a", "b"})
        self.assertEqual(_field(by_id["a"], "artifact_status"), "full")
        self.assertEqual(_field(by_id["b"], "artifact_status"), "partial")
        self.assertEqual(_field(by_id["a"], "pair_id"), "pair-1")

    def test_manifest_load_uses_safe_memory_mapped_weights_only_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_dir = root / "graphs"
            graph_dir.mkdir()
            graph_path = graph_dir / "a.pt"
            graph_path.touch()
            (root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "graph_dir": str(graph_dir),
                        "graph_files": ["graphs/a.pt"],
                        "example_ids": ["candidate-a"],
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch("attention_graph.halueval.torch.load", return_value=_legacy_graph()) as load:
                discover_legacy_halueval_records(root)

        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.args, (graph_path,))
        self.assertEqual(
            load.call_args.kwargs,
            {"map_location": "cpu", "weights_only": True, "mmap": True},
        )

    def test_manifest_prefers_root_artifact_dirs_and_validates_listed_identities(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_artifact_pair(root, "a", pair_id="pair-a")
            (root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "state": "running",
                        "graph_dir": "incorrect-relative-dir",
                        "trace_dir": "also-incorrect",
                        "graph_files": ["a.pt"],
                        "example_ids": ["a"],
                    }
                ),
                encoding="utf-8",
            )
            records = discover_legacy_halueval_records(root)
            self.assertEqual(records[0]["artifact_status"], "partial")
            self.assertEqual(records[0]["manifest_state"], "running")

            for graph_files, example_ids in (
                (["a.pt", "a.pt"], ["a", "a"]),
                (["a.pt"], []),
                (["a.pt"], ["not-a"]),
            ):
                (root / "extraction_manifest.json").write_text(
                    json.dumps({"graph_files": graph_files, "example_ids": example_ids}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(ValueError, "duplicate|length|identity|example"):
                    discover_legacy_halueval_records(root)

    def test_pair_split_is_seeded_pair_disjoint_and_rejects_a_candidate_limit(self):
        records = []
        for pair_number in range(10):
            for candidate in range(2):
                records.append(
                    {
                        "response_id": f"r-{pair_number}-{candidate}",
                        "pair_id": f"pair-{pair_number}",
                    }
                )

        first = split_halueval_pairs(
            records, validation_fraction=0.2, test_fraction=0.2, seed=7
        )
        second = split_halueval_pairs(
            records, validation_fraction=0.2, test_fraction=0.2, seed=7
        )

        self.assertEqual(
            {name: [_field(item, "response_id") for item in values] for name, values in first.items()},
            {name: [_field(item, "response_id") for item in values] for name, values in second.items()},
        )
        pair_sets = {
            name: {_field(item, "pair_id") for item in values}
            for name, values in first.items()
        }
        self.assertFalse(pair_sets["train"] & pair_sets["validation"])
        self.assertFalse(pair_sets["train"] & pair_sets["test"])
        self.assertFalse(pair_sets["validation"] & pair_sets["test"])
        self.assertEqual({len(values) for values in pair_sets.values()}, {2, 6})
        for values in first.values():
            self.assertEqual(len(values) % 2, 0)

        # A candidate-count limit that leaves one half of a HaluEval pair is unsafe.
        with self.assertRaisesRegex(ValueError, "pair|two|complete"):
            split_halueval_pairs(records[:-1], seed=7)

    def test_pair_split_matches_the_legacy_stable_hash_and_nonempty_partition_counts(self):
        import hashlib

        records = [
            {"response_id": f"r-{pair}-{candidate}", "pair_id": f"pair-{pair}"}
            for pair in range(3)
            for candidate in range(2)
        ]

        partitions = split_halueval_pairs(
            records, validation_fraction=0.1, test_fraction=0.2, seed=17
        )
        ordered = sorted(
            {record["pair_id"] for record in records},
            key=lambda pair_id: hashlib.sha256(f"17\x1f{pair_id}".encode("utf-8")).hexdigest(),
        )

        self.assertEqual(
            {_field(record, "pair_id") for record in partitions["test"]}, {ordered[0]}
        )
        self.assertEqual(
            {_field(record, "pair_id") for record in partitions["validation"]}, {ordered[1]}
        )
        self.assertEqual(len(partitions["train"]), 2)

    def test_pair_split_rejects_fewer_than_three_pairs_before_allocating_partitions(self):
        records = [
            {"response_id": f"r-{pair}-{candidate}", "pair_id": f"pair-{pair}"}
            for pair in range(2)
            for candidate in range(2)
        ]

        with self.assertRaisesRegex(ValueError, "at least three|three pairs"):
            split_halueval_pairs(records, validation_fraction=0.1, test_fraction=0.2)

    def test_zero_split_fraction_creates_a_genuinely_empty_partition(self):
        records = [
            {"response_id": f"r-{pair}-{candidate}", "pair_id": f"pair-{pair}"}
            for pair in range(4)
            for candidate in range(2)
        ]

        partitions = split_halueval_pairs(
            records, validation_fraction=0.0, test_fraction=0.25, seed=3
        )

        self.assertEqual(partitions["validation"], [])
        self.assertEqual(len(partitions["test"]), 2)
        self.assertEqual(len(partitions["train"]), 6)

    def test_prompt_group_split_keeps_duplicate_prompt_pairs_in_one_partition(self):
        records = [
            {
                "response_id": f"r-{pair}-{candidate}",
                "pair_id": f"pair-{pair}",
                # pair-0 and pair-1 deliberately repeat the exact source prompt.
                "group_id": "same-prompt" if pair < 2 else f"prompt-{pair}",
            }
            for pair in range(5)
            for candidate in range(2)
        ]

        partitions = split_halueval_pairs(
            records,
            validation_fraction=0.2,
            test_fraction=0.2,
            seed=5,
            group_by_prompt=True,
        )

        allocation = {
            _field(record, "pair_id"): split
            for split, values in partitions.items()
            for record in values
        }
        self.assertEqual(allocation["pair-0"], allocation["pair-1"])


class LegacyPreparationIntegrationTests(unittest.TestCase):
    def test_preparation_resume_validates_lightweight_identity_before_expensive_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_dir, trace_dir = root / "graphs", root / "traces"
            graph_dir.mkdir()
            trace_dir.mkdir()
            graph_path, trace_path = graph_dir / "response.pt", trace_dir / "response.pt"
            torch.save(_legacy_graph(example_id="response", pair_id="pair"), graph_path)
            torch.save(_trace_metadata(example_id="response", pair_id="pair"), trace_path)
            record = {"response_id": "response", "pair_id": "pair", "graph_path": graph_path, "trace_path": trace_path}
            output = root / "prepared"
            config = GraphBuildConfig(selection="threshold", threshold=0.05)
            prepare_legacy_halueval_graphs([record], output_dir=output, config=config)

            with mock.patch(
                "attention_graph.halueval.legacy_graph_to_formal_attention_cache",
                side_effect=AssertionError("reused cache must skip expensive conversion"),
            ) as convert:
                prepare_legacy_halueval_graphs([record], output_dir=output, config=config)

            convert.assert_not_called()

    def test_preparation_cache_name_is_stable_across_source_mtime_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_dir, trace_dir = root / "graphs", root / "traces"
            graph_dir.mkdir()
            trace_dir.mkdir()
            graph_path, trace_path = graph_dir / "response.pt", trace_dir / "response.pt"
            torch.save(_legacy_graph(example_id="response", pair_id="pair"), graph_path)
            torch.save(_trace_metadata(example_id="response", pair_id="pair"), trace_path)
            record = {"response_id": "response", "pair_id": "pair", "graph_path": graph_path, "trace_path": trace_path}
            output = root / "prepared"
            config = GraphBuildConfig(selection="threshold", threshold=0.05)

            prepare_legacy_halueval_graphs([record], output_dir=output, config=config)
            first = sorted((output / "adapted_cache" / "train").glob("attention_*.pt"))
            os.utime(graph_path, ns=(graph_path.stat().st_atime_ns, graph_path.stat().st_mtime_ns + 1_000_000))
            prepare_legacy_halueval_graphs([record], output_dir=output, config=config)
            second = sorted((output / "adapted_cache" / "train").glob("attention_*.pt"))

            self.assertEqual(first, second)
            self.assertEqual(len(second), 1)

    def test_preparation_rebuilds_same_identity_cache_when_extraction_fingerprint_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph_dir, trace_dir = root / "graphs", root / "traces"
            graph_dir.mkdir()
            trace_dir.mkdir()
            graph_path, trace_path = graph_dir / "response.pt", trace_dir / "response.pt"
            torch.save(_legacy_graph(example_id="response", pair_id="pair"), graph_path)
            torch.save(_trace_metadata(example_id="response", pair_id="pair"), trace_path)
            record = {"response_id": "response", "pair_id": "pair", "graph_path": graph_path, "trace_path": trace_path}
            output = root / "prepared"
            config = GraphBuildConfig(selection="threshold", threshold=0.05)

            prepare_legacy_halueval_graphs([record], output_dir=output, config=config)
            cache_path = next((output / "adapted_cache" / "train").glob("attention_*.pt"))
            changed_graph, changed_trace = _legacy_graph(example_id="response", pair_id="pair"), _trace_metadata(example_id="response", pair_id="pair")
            changed_graph["extraction_fingerprint"] = "new-fingerprint"
            changed_trace["extraction_fingerprint"] = "new-fingerprint"
            torch.save(changed_graph, graph_path)
            torch.save(changed_trace, trace_path)
            prepare_legacy_halueval_graphs([record], output_dir=output, config=config)

            caches = sorted((output / "adapted_cache" / "train").glob("attention_*.pt"))
            self.assertEqual(caches, [cache_path])
            self.assertEqual(
                torch.load(cache_path, map_location="cpu", weights_only=True)["attention_cache_fingerprint"],
                "new-fingerprint",
            )

    def test_prepare_reuses_standard_label_free_mmap_artifacts_for_pair_partitions(self):
        """The legacy adapter must feed the normal prepared-graph contract."""

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy_root = root / "legacy"
            graph_dir, trace_dir = legacy_root / "graphs", legacy_root / "traces"
            graph_dir.mkdir(parents=True)
            trace_dir.mkdir()
            graph_files, example_ids = [], []
            for pair in range(3):
                for candidate in range(2):
                    response_id = f"response-{pair}-{candidate}"
                    filename = f"{response_id}.pt"
                    torch.save(
                        _legacy_graph(example_id=response_id, pair_id=f"pair-{pair}"),
                        graph_dir / filename,
                    )
                    torch.save(
                        _trace_metadata(example_id=response_id, pair_id=f"pair-{pair}"),
                        trace_dir / filename,
                    )
                    graph_files.append(filename)
                    example_ids.append(response_id)
            (legacy_root / "extraction_manifest.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "graph_files": graph_files,
                        "example_ids": example_ids,
                    }
                ),
                encoding="utf-8",
            )
            discovered = discover_legacy_halueval_records(legacy_root)
            partitions = split_halueval_pairs(
                discovered, validation_fraction=0.0, test_fraction=1.0 / 3.0, seed=5
            )
            assignment = {
                record["response_id"]: "test" if record in partitions["test"] else "train"
                for record in discovered
            }
            output = root / "prepared"
            config = GraphBuildConfig(selection="threshold", threshold=0.05)

            first = prepare_legacy_halueval_graphs(
                discovered,
                output_dir=output,
                config=config,
                dataset_split_by_response=assignment,
            )
            second = prepare_legacy_halueval_graphs(
                discovered,
                output_dir=output,
                config=config,
                dataset_split_by_response=assignment,
            )

            self.assertTrue(all(isinstance(record, PreparedGraphRecord) for record in first))
            self.assertTrue(all(record.state == "reused" for record in second))
            self.assertEqual({record.response_id for record in first}, set(example_ids))
            for record in first:
                graph = load_graph(record.graph_path, device="cpu", mmap=True)
                self.assertEqual(graph.response_id, record.response_id)
                self.assertEqual(graph.source_id, record.source_id)
                self.assertTrue(bool((graph.edge_index[1] >= graph.response_idx).all()))
                self.assertEqual(
                    graph.edge_type.tolist(),
                    (graph.edge_index[0] >= graph.response_idx).long().tolist(),
                )
                self.assertFalse(any("label" in key.casefold() for key in torch.load(
                    record.graph_path, map_location="cpu", weights_only=True, mmap=True
                )["graph"]))

            cache = torch.load(
                first[0].cache_path, map_location="cpu", weights_only=True, mmap=True
            )
            self.assertEqual(cache["attention_diagonal"].dtype, torch.float16)
            self.assertEqual(cache["response_values"].dtype, torch.float16)
            self.assertEqual(cache["response_column_indices"].dtype, torch.int32)
            self.assertEqual(cache["response_row_ptr"].dtype, torch.int64)


class HaluEvalResponseEvaluationTests(unittest.TestCase):
    def test_explicit_sidecar_and_exact_response_join_produce_rank_and_paired_metrics(self):
        labels_rows = [
            {"response_id": "r0", "label": 0},
            {"response_id": "r1", "label": 1},
            {"response_id": "r2", "label": 0},
            {"response_id": "r3", "label": 1},
            {"response_id": "r4", "label": 0},
            {"response_id": "r5", "label": 1},
        ]
        predictions = [
            {"response_id": "r0", "score": 0.10},
            {"response_id": "r1", "score": 0.90},
            {"response_id": "r2", "score": 0.20},
            {"response_id": "r3", "score": 0.80},
            {"response_id": "r4", "score": 0.70},
            {"response_id": "r5", "score": 0.60},
        ]
        pair_by_response = {
            "r0": "p0", "r1": "p0", "r2": "p1", "r3": "p1", "r4": "p2", "r5": "p2"
        }
        lengths = {response_id: index + 1 for index, response_id in enumerate(pair_by_response)}
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "halueval_response_labels.jsonl"
            sidecar.write_text(
                "".join(json.dumps(row) + "\n" for row in labels_rows), encoding="utf-8"
            )
            labels = load_halueval_response_labels(sidecar)

        self.assertEqual(labels, {row["response_id"]: row["label"] for row in labels_rows})
        metrics = evaluate_halueval_predictions(
            predictions,
            labels,
            pair_by_response,
            response_length_by_id=lengths,
            bootstrap_samples=200,
            seed=9,
        )
        repeated = evaluate_halueval_predictions(
            predictions, labels, pair_by_response, response_length_by_id=lengths,
            bootstrap_samples=200, seed=9,
        )

        self.assertAlmostEqual(metrics["auroc"], 8.0 / 9.0)
        self.assertAlmostEqual(metrics["average_precision"], 11.0 / 12.0)
        self.assertAlmostEqual(metrics["paired_accuracy"], 2.0 / 3.0)
        self.assertEqual(metrics["paired_bootstrap_ci"], repeated["paired_bootstrap_ci"])
        ci = metrics["paired_bootstrap_ci"]
        self.assertEqual(ci["samples"], 200)
        outcomes = np.asarray([1.0, 1.0, 0.0])
        generator = np.random.default_rng(9)
        bootstrap = np.asarray(
            [generator.choice(outcomes, size=len(outcomes), replace=True).mean() for _ in range(200)]
        )
        expected_low, expected_high = np.quantile(bootstrap, (0.025, 0.975))
        self.assertEqual(ci["low"], expected_low)
        self.assertEqual(ci["high"], expected_high)
        self.assertTrue(0.0 <= ci["low"] <= ci["high"] <= 1.0)

        with self.assertRaisesRegex(ValueError, "join|missing|extra|response_id"):
            evaluate_halueval_predictions(
                predictions[:-1], labels, pair_by_response, bootstrap_samples=10
            )

    def test_paired_score_tie_receives_half_credit(self):
        metrics = evaluate_halueval_predictions(
            [{"response_id": "clean", "score": 0.5}, {"response_id": "hallucinated", "score": 0.5}],
            {"clean": 0, "hallucinated": 1},
            {"clean": "pair", "hallucinated": "pair"},
            bootstrap_samples=5,
        )

        self.assertEqual(metrics["paired_accuracy"], 0.5)

    def test_bootstrap_ci_is_the_unmodified_empirical_quantile(self):
        predictions = [
            {"response_id": "clean-a", "score": 0.1},
            {"response_id": "hallucinated-a", "score": 0.9},
            {"response_id": "clean-b", "score": 0.9},
            {"response_id": "hallucinated-b", "score": 0.1},
        ]
        labels = {"clean-a": 0, "hallucinated-a": 1, "clean-b": 0, "hallucinated-b": 1}
        pairs = {
            "clean-a": "a", "hallucinated-a": "a",
            "clean-b": "b", "hallucinated-b": "b",
        }
        metrics = evaluate_halueval_predictions(
            predictions, labels, pairs, bootstrap_samples=1, seed=0
        )
        generator = np.random.default_rng(0)
        expected = np.quantile(
            np.asarray([generator.choice(np.asarray([1.0, 0.0]), size=2, replace=True).mean()]),
            (0.025, 0.975),
        )
        ci = metrics["paired_bootstrap_ci"]
        self.assertEqual((ci["low"], ci["high"]), tuple(expected.tolist()))

    def test_prediction_field_and_pair_id_normalization_are_unambiguous(self):
        with self.assertRaisesRegex(ValueError, "score|probability|disagree"):
            evaluate_halueval_predictions(
                [
                    {"response_id": "clean", "score": 0.2, "hallucination_probability": 0.8},
                    {"response_id": "hallucinated", "score": 0.8},
                ],
                {"clean": 0, "hallucinated": 1},
                {"clean": "pair", "hallucinated": "pair"},
            )

        predictions = [{"response_id": "1", "score": 0.2}, {"response_id": "other", "score": 0.8}]
        labels = {"1": 0, "other": 1}
        with self.assertRaisesRegex(ValueError, "collision|unique"):
            evaluate_halueval_predictions(
                predictions, labels, {1: "pair", "1": "pair", "other": "pair"}
            )
        with self.assertRaisesRegex(ValueError, "pair"):
            evaluate_halueval_predictions(
                predictions, labels, {"1": "", "other": ""}
            )


if __name__ == "__main__":
    unittest.main()
