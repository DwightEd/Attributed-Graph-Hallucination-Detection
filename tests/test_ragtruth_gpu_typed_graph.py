"""Contracts for sparse upstream RAGTruth attention payloads and typed graphs."""

from __future__ import annotations

import inspect
import json
import tempfile
import unittest
from collections.abc import Mapping
from pathlib import Path
from unittest.mock import patch

import torch

from unsupervised_token_graph.ragtruth_data import compact_attention_cache
from unsupervised_token_graph.ragtruth_graph import (
    build_compact_topk_graph,
    load_attention_sample,
)
from unsupervised_token_graph.typed_model import (
    TypedNeighborhoodAutoencoder,
    score_masked_tokens,
    typed_reconstruction_loss,
)

FORBIDDEN_TRAINING_FIELDS = {"y_token", "label", "labels", "target", "y"}


def _field(value, name):
    return value[name] if isinstance(value, Mapping) else getattr(value, name)


def _keys(value):
    return set(value) if isinstance(value, Mapping) else set(vars(value))


def _test_device() -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda", torch.cuda.current_device())


def _contains_label_key(value) -> bool:
    return any("label" in str(name).casefold() for name in _keys(value))


def _legacy_attention() -> torch.Tensor:
    attention = torch.zeros((2, 2, 5, 5), dtype=torch.float32)
    attention[:, :, 3, 0] = 0.90
    attention[:, :, 3, 2] = 0.80
    attention[:, :, 3, 1] = 0.10
    attention[:, :, 4, 1] = 0.95
    attention[:, :, 4, 3] = 0.85
    attention[:, :, 4, 2] = 0.15
    return attention


def _write_legacy_cache(path: Path, *, include_hidden: bool = True) -> None:
    sample = {
        "source_id": "legacy-source-42",
        "original_idx": 19,
        "response_idx": 3,
        "token_ids": torch.tensor([10, 11, 12, 13, 14]),
        "attention": _legacy_attention(),
        "hallucination_labels": torch.tensor([0, 0, 0, 1, 0]),
    }
    if include_hidden:
        sample["hidden"] = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    torch.save(sample, path)


class LegacyAttentionCacheCompatibilityTests(unittest.TestCase):
    """Keep the pre-v3 dense cache contract covered during migration."""

    def test_legacy_loader_keeps_original_idx_and_hides_legacy_labels(self):
        device = _test_device()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy_sample.pt"
            _write_legacy_cache(path)
            sample = load_attention_sample(path, device=device, mmap=True)

        self.assertEqual(sample["source_id"], "legacy-source-42")
        self.assertEqual(sample["original_idx"], 19)
        self.assertFalse(_contains_label_key(sample))
        self.assertTrue(FORBIDDEN_TRAINING_FIELDS.isdisjoint(_keys(sample)))
        for name in ("token_ids", "attention", "hidden"):
            self.assertEqual(sample[name].device, device)

    def test_attention_only_legacy_cache_remains_buildable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy_attention_only.pt"
            _write_legacy_cache(path, include_hidden=False)
            sample = load_attention_sample(path, device="cpu", mmap=True)
            graph = build_compact_topk_graph(sample, top_k=2, device="cpu")

        self.assertNotIn("hidden", sample)
        self.assertEqual(_field(graph, "x").shape[0], 5)

    def test_legacy_dense_attention_still_builds_causal_typed_edges(self):
        device = _test_device()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy_sample.pt"
            _write_legacy_cache(path)
            graph = build_compact_topk_graph(
                load_attention_sample(path, device=device, mmap=True),
                top_k=2,
                device=device,
            )

        edge_index = _field(graph, "edge_index")
        self.assertEqual(_field(graph, "original_idx"), 19)
        self.assertEqual(
            {tuple(edge) for edge in edge_index.t().cpu().tolist()},
            {(0, 3), (2, 3), (1, 4), (3, 4)},
        )
        self.assertTrue(bool((edge_index[0] < edge_index[1]).all()))
        self.assertTrue(bool(_field(graph, "response_mask")[edge_index[1]].all()))

    def test_legacy_graph_still_trains_the_typed_autoencoder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "legacy_sample.pt"
            _write_legacy_cache(path)
            graph = build_compact_topk_graph(
                load_attention_sample(path, device="cpu", mmap=True),
                top_k=2,
                device="cpu",
            )
        model = TypedNeighborhoodAutoencoder(
            node_dim=_field(graph, "x").shape[1],
            edge_dim=_field(graph, "edge_attr").shape[1],
            num_edge_types=int(_field(graph, "edge_type").max()) + 1,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            context_dim=_field(graph, "node_context").shape[1],
        )
        outputs = model(graph, _field(graph, "response_mask"))
        loss = typed_reconstruction_loss(
            outputs, graph, _field(graph, "response_mask")
        )
        self.assertTrue(torch.isfinite(loss))


def _upstream_sparse_payload() -> dict[str, object]:
    """Exact v1-style sparse response CSR payload with shared row topology.

    CSR rows are ordered ``(layer, head, response-token)``.  Upstream retains
    strictly causal keys only, so no row is allowed to contain its target.
    """

    layers, heads, token_count, response_idx = 2, 2, 5, 3
    rows = ([0, 2], [1, 3]) * (layers * heads)
    weights = ([0.50, 0.40], [0.55, 0.35]) * (layers * heads)
    row_columns = [column for row in rows for column in row]
    row_values = [value for row in weights for value in row]
    row_ptr = torch.arange(0, len(row_columns) + 1, 2, dtype=torch.long)
    return {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "source_id": "source-42",
        "response_id": "response-19",
        "token_ids": torch.tensor([10, 11, 12, 13, 14]),
        "y_token": torch.tensor([0, 0, 0, 1, 0], dtype=torch.float32),
        "response_idx": response_idx,
        "num_attention_layers": layers,
        "num_attention_heads": heads,
        "attention_diagonal": torch.full(
            (layers, heads, token_count), 0.05, dtype=torch.float16
        ),
        "response_row_ptr": row_ptr,
        "response_column_indices": torch.tensor(row_columns, dtype=torch.int32),
        "response_values": torch.tensor(row_values, dtype=torch.float16),
        "attention_floor": 0.01,
        "cache_dtype": str(torch.float16),
        "attention_cache_fingerprint": "fresh_attention_c8847872bedf",
        "split": "train",
        "task_type": "QA",
        "generator_model": "upstream-model",
        "quality": "high",
        "input_policy": "full_context_no_truncation",
        "was_truncated": False,
    }


def _write_sparse_payload(path: Path, *, malformed_csr: bool = False) -> None:
    payload = _upstream_sparse_payload()
    if malformed_csr:
        payload["response_row_ptr"] = torch.tensor(
            [0, 2, 1, 4, 6, 8, 10, 12, 14], dtype=torch.long
        )
    torch.save(payload, path)


class SparseAttentionLoaderContractTests(unittest.TestCase):
    def test_loader_normalizes_response_id_to_sample_id_and_hides_y_token_by_default(self):
        device = _test_device()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_19.pt"
            _write_sparse_payload(path)
            sample = load_attention_sample(
                path, device=device, mmap=True, include_labels=False
            )

        self.assertEqual(sample["source_id"], "source-42")
        self.assertEqual(sample["sample_id"], "response-19")
        self.assertNotIn("original_idx", sample)
        self.assertTrue(FORBIDDEN_TRAINING_FIELDS.isdisjoint(_keys(sample)))
        for name in (
            "token_ids",
            "attention_diagonal",
            "response_row_ptr",
            "response_column_indices",
            "response_values",
        ):
            self.assertEqual(sample[name].device, device)

    def test_labels_are_opt_in_for_final_evaluation_and_rejected_by_graph_builder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_19.pt"
            _write_sparse_payload(path)
            evaluation_sample = load_attention_sample(
                path, device="cpu", mmap=True, include_labels=True
            )

        self.assertEqual(evaluation_sample["sample_id"], "response-19")
        self.assertEqual(evaluation_sample["y_token"].tolist(), [0, 0, 0, 1, 0])
        with self.assertRaisesRegex(ValueError, "label|y_token"):
            build_compact_topk_graph(evaluation_sample, top_k=2, device="cpu")

    def test_malformed_csr_fails_before_graph_construction(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_corrupt.pt"
            _write_sparse_payload(path, malformed_csr=True)

            with self.assertRaisesRegex(ValueError, "response_row_ptr|CSR|monotonic"):
                load_attention_sample(path, device="cpu", mmap=True)

    def test_formal_storage_metadata_is_validated_before_any_index_cast(self):
        corruptions = {
            "token ids": ("token_ids", torch.arange(5, dtype=torch.int32)),
            "row pointers": ("response_row_ptr", torch.arange(9, dtype=torch.int32) * 2),
            "columns": ("response_column_indices", torch.arange(16, dtype=torch.float32)),
            "diagonal dtype": (
                "attention_diagonal",
                torch.full((2, 2, 5), 0.05, dtype=torch.float32),
            ),
            "empty values dtype": ("response_values", torch.empty(0, dtype=torch.float64)),
            "zero floor": ("attention_floor", 0.0),
            "truncated input": ("was_truncated", True),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for name, (field, value) in corruptions.items():
                with self.subTest(name=name):
                    payload = _upstream_sparse_payload()
                    payload[field] = value
                    if field == "response_values":
                        payload["response_column_indices"] = torch.empty(0, dtype=torch.int32)
                        payload["response_row_ptr"] = torch.zeros(9, dtype=torch.int64)
                    path = root / f"{name.replace(' ', '_')}.pt"
                    torch.save(payload, path)
                    with self.assertRaisesRegex(
                        (ValueError, TypeError),
                        "dtype|int64|int32|cache_dtype|floor|truncated",
                    ):
                        load_attention_sample(path, device="cpu", mmap=True)

    def test_formal_loader_preserves_provenance_needed_for_resume(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "attention_19.pt"
            _write_sparse_payload(path)
            sample = load_attention_sample(path, device="cpu", mmap=True)

        self.assertEqual(
            sample["attention_cache_fingerprint"], "fresh_attention_c8847872bedf"
        )
        self.assertEqual(sample["cache_dtype"], "float16")
        self.assertEqual(sample["input_policy"], "full_context_no_truncation")
        self.assertIs(sample["was_truncated"], False)

    def test_graph_loader_has_no_direct_supervision_parameter(self):
        parameters = inspect.signature(load_attention_sample).parameters
        self.assertIn("include_labels", parameters)
        self.assertNotIn("labels", parameters)
        self.assertNotIn("y_token", parameters)

    def test_bf16_source_to_fp16_mass_roundoff_is_accepted_but_real_inflation_is_not(self):
        def payload_with_row_mass(second_value: float) -> dict[str, object]:
            payload = _upstream_sparse_payload()
            payload["attention_diagonal"] = torch.full(
                (2, 2, 5), 0.20, dtype=torch.float16
            )
            payload["response_values"] = torch.tensor(
                [0.50, second_value] * 8, dtype=torch.float16
            )
            return payload

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            explainable = root / "bf16_to_fp16_rounding.pt"
            inflated = root / "inflated.pt"
            torch.save(payload_with_row_mass(0.304), explainable)
            torch.save(payload_with_row_mass(0.350), inflated)

            accepted = build_compact_topk_graph(
                load_attention_sample(explainable, device="cpu", mmap=True),
                top_k=2,
                device="cpu",
            )
            self.assertEqual(accepted["edge_attr"].shape[1], 8)
            with self.assertRaisesRegex(ValueError, "mass exceeds one"):
                build_compact_topk_graph(
                    load_attention_sample(inflated, device="cpu", mmap=True),
                    top_k=2,
                    device="cpu",
                )


class SparseTopKGraphContractTests(unittest.TestCase):
    def _graph(self, *, device: torch.device | None = None):
        device = torch.device("cpu") if device is None else device
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_19.pt"
            _write_sparse_payload(path)
            sample = load_attention_sample(
                path, device=device, mmap=True, include_labels=False
            )
            return build_compact_topk_graph(sample, top_k=2, device=device)

    def test_sparse_csr_builds_causal_typed_gpu_resident_topk_graph(self):
        device = _test_device()
        graph = self._graph(device=device)

        required = {
            "schema_version",
            "sample_id",
            "source_id",
            "x",
            "edge_index",
            "edge_attr",
            "edge_type",
            "response_mask",
        }
        self.assertTrue(required.issubset(_keys(graph)))
        self.assertEqual(_field(graph, "schema_version"), "ragtruth_typed_topk_v3")
        self.assertEqual(_field(graph, "source_id"), "source-42")
        self.assertEqual(_field(graph, "sample_id"), "response-19")
        self.assertNotIn("original_idx", _keys(graph))
        self.assertTrue(FORBIDDEN_TRAINING_FIELDS.isdisjoint(_keys(graph)))

        edge_index = _field(graph, "edge_index")
        edge_attr = _field(graph, "edge_attr")
        edge_type = _field(graph, "edge_type")
        response_mask = _field(graph, "response_mask")
        for value in (
            _field(graph, "x"),
            edge_index,
            edge_attr,
            edge_type,
            response_mask,
        ):
            self.assertEqual(value.device, device)
        self.assertEqual(response_mask.cpu().tolist(), [False, False, False, True, True])
        self.assertEqual(edge_index.shape, (2, 4))
        self.assertEqual(edge_attr.shape[0], 4)
        self.assertEqual(edge_type.shape, (4,))
        self.assertEqual(
            {tuple(edge) for edge in edge_index.t().cpu().tolist()},
            {(0, 3), (2, 3), (1, 4), (3, 4)},
        )
        self.assertTrue(bool((edge_index[0] < edge_index[1]).all()))
        self.assertTrue(bool(response_mask[edge_index[1]].all()))
        source_type = edge_type[edge_index[1] == 3]
        history_type = edge_type[(edge_index[0] == 3) & (edge_index[1] == 4)]
        self.assertTrue(bool((source_type == source_type[0]).all()))
        self.assertNotEqual(int(source_type[0]), int(history_type[0]))

    def test_formal_edge_statistics_preserve_layer_channel_mapping(self):
        payload = _upstream_sparse_payload()
        payload.update(
            {
                "num_attention_layers": 4,
                "num_attention_heads": 1,
                "attention_diagonal": torch.full((4, 1, 5), 0.05, dtype=torch.float16),
                "response_row_ptr": torch.arange(9, dtype=torch.int64),
                "response_column_indices": torch.tensor(
                    [0, 3, 0, 3, 0, 3, 0, 3], dtype=torch.int32
                ),
                "response_values": torch.tensor(
                    [0.8, 0.8, 0.6, 0.6, 0.4, 0.4, 0.2, 0.2],
                    dtype=torch.float16,
                ),
            }
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "sample_19.pt"
            torch.save(payload, path)
            graph = build_compact_topk_graph(
                load_attention_sample(path, device="cpu", mmap=True),
                top_k=1,
                device="cpu",
            )

        edges = graph["edge_index"].t().tolist()
        attr = graph["edge_attr"][edges.index([0, 3])]
        torch.testing.assert_close(
            attr,
            torch.tensor([0.5, 0.8, 0.05**0.5, 0.7, 0.3, 1.0, 1.0, 1.0]),
            rtol=2e-3,
            atol=2e-3,
        )

    def test_formal_empty_csr_rows_build_empty_or_partial_graphs(self):
        for name, row_ptr, columns, values, expected_edges in (
            ("empty", torch.zeros(9, dtype=torch.int64), torch.empty(0, dtype=torch.int32),
             torch.empty(0, dtype=torch.float16), 0),
            ("partial", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1]),
             torch.tensor([1], dtype=torch.int32), torch.tensor([0.5], dtype=torch.float16), 1),
        ):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary_directory:
                payload = _upstream_sparse_payload()
                payload["response_row_ptr"] = row_ptr
                payload["response_column_indices"] = columns
                payload["response_values"] = values
                path = Path(temporary_directory) / "sample_19.pt"
                torch.save(payload, path)
                graph = build_compact_topk_graph(
                    load_attention_sample(path, device="cpu", mmap=True),
                    top_k=2,
                    device="cpu",
                )
            self.assertEqual(graph["edge_index"].shape, (2, expected_edges))


class FormalCompactionResumeContractTests(unittest.TestCase):
    @staticmethod
    def _cpu_builder(sample, **kwargs):
        kwargs["device"] = "cpu"
        return build_compact_topk_graph(sample, **kwargs)

    def test_formal_csr_compacts_to_auditable_manifest_and_rejects_stale_graph(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_dir, output_dir = root / "raw", root / "compact"
            raw_dir.mkdir()
            _write_sparse_payload(raw_dir / "attention_19.pt")
            with (
                patch(
                    "unsupervised_token_graph.ragtruth_data.torch.cuda.is_available",
                    return_value=True,
                ),
                patch(
                    "unsupervised_token_graph.ragtruth_data.build_compact_topk_graph",
                    side_effect=self._cpu_builder,
                ),
            ):
                summary = compact_attention_cache(
                    raw_dir, output_dir, device="cuda:0", resume=False
                )

            record = json.loads(
                (output_dir / "manifest.jsonl").read_text(encoding="utf-8").strip()
            )
            self.assertEqual(summary["samples"], 1)
            self.assertEqual(record["attention_cache_fingerprint"], "fresh_attention_c8847872bedf")
            self.assertEqual(record["token_count"], 5)
            self.assertEqual(record["layers"], 2)
            self.assertEqual(record["heads"], 2)
            graph_path = output_dir / record["path"]
            graph = torch.load(graph_path, map_location="cpu", weights_only=True)
            self.assertEqual(
                graph["graph_config"]["raw_identity"]["attention_cache_fingerprint"],
                record["attention_cache_fingerprint"],
            )

            graph["graph_config"]["raw_identity"]["attention_floor"] = 0.5
            torch.save(graph, graph_path)
            with (
                patch(
                    "unsupervised_token_graph.ragtruth_data.torch.cuda.is_available",
                    return_value=True,
                ),
                self.assertRaisesRegex(RuntimeError, "raw identity|attention_floor|stale"),
            ):
                compact_attention_cache(
                    raw_dir, output_dir, device="cuda:0", resume=True
                )

    def test_changed_formal_fingerprint_cannot_reuse_previous_graph_filename(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_raw, replacement_raw, output_dir = (
                root / "raw_a",
                root / "raw_b",
                root / "compact",
            )
            first_raw.mkdir()
            replacement_raw.mkdir()
            _write_sparse_payload(first_raw / "attention_19.pt")
            payload = _upstream_sparse_payload()
            payload["attention_cache_fingerprint"] = "replacement-fingerprint"
            torch.save(payload, replacement_raw / "attention_19.pt")
            builder = patch(
                "unsupervised_token_graph.ragtruth_data.build_compact_topk_graph",
                side_effect=self._cpu_builder,
            )
            with patch(
                "unsupervised_token_graph.ragtruth_data.torch.cuda.is_available",
                return_value=True,
            ), builder as mocked_builder:
                compact_attention_cache(first_raw, output_dir, device="cuda:0")
                compact_attention_cache(
                    replacement_raw, output_dir, device="cuda:0", resume=True
                )

            self.assertEqual(mocked_builder.call_count, 2)
            records = [
                json.loads(line)
                for line in (output_dir / "manifest.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
                if line.strip()
            ]
            self.assertEqual(records[0]["attention_cache_fingerprint"], "replacement-fingerprint")


class TypedNeighborhoodAutoencoderContractTests(unittest.TestCase):
    def _graph(self):
        return SparseTopKGraphContractTests()._graph()

    def _model(self, graph):
        return TypedNeighborhoodAutoencoder(
            node_dim=_field(graph, "x").shape[1],
            edge_dim=_field(graph, "edge_attr").shape[1],
            num_edge_types=int(_field(graph, "edge_type").max()) + 1,
            hidden_dim=8,
            num_layers=1,
            dropout=0.0,
            context_dim=_field(graph, "node_context").shape[1],
        )

    def test_typed_reconstruction_uses_response_mask_and_has_gradients(self):
        graph = self._graph()
        model = self._model(graph)
        mask = _field(graph, "response_mask")
        outputs = model(graph, mask)
        self.assertTrue(
            {
                "node_reconstruction",
                "neighborhood_mean",
                "neighborhood_log_variance",
                "route_stats",
            }.issubset(outputs)
        )
        self.assertEqual(outputs["node_reconstruction"].shape, _field(graph, "x").shape)
        self.assertEqual(outputs["neighborhood_mean"].shape[:2], (len(_field(graph, "x")), 2))
        self.assertEqual(
            outputs["neighborhood_log_variance"].shape,
            outputs["neighborhood_mean"].shape,
        )
        self.assertEqual(outputs["route_stats"].shape[:2], (len(_field(graph, "x")), 2))
        loss = typed_reconstruction_loss(outputs, graph, mask)
        self.assertEqual(loss.ndim, 0)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_token_scores_are_aligned_to_sample_id_without_labels(self):
        graph = self._graph()
        model = self._model(graph)
        parameters = inspect.signature(score_masked_tokens).parameters
        self.assertNotIn("labels", parameters)
        self.assertNotIn("y_token", parameters)
        scores = score_masked_tokens(model, graph, _field(graph, "response_mask"))

        self.assertTrue({"source_id", "sample_id", "token_idx", "scores"}.issubset(scores))
        self.assertEqual(scores["source_id"], "source-42")
        self.assertEqual(scores["sample_id"], "response-19")
        self.assertEqual(scores["token_idx"].tolist(), [3, 4])
        self.assertEqual(scores["scores"].shape, (2,))
        self.assertTrue(torch.isfinite(scores["scores"]).all())
        self.assertTrue(FORBIDDEN_TRAINING_FIELDS.isdisjoint(_keys(scores)))


if __name__ == "__main__":
    unittest.main()
