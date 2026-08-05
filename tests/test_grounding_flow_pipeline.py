from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from grounding_flow.evaluation import (
    add_length_residual_scores,
    evaluate_frozen_halueval_predictions,
    freeze_prediction_files,
)
from grounding_flow.experiment import TrajectoryRecord
from grounding_flow.pipeline import (
    GroundingFlowRunConfig,
    _protocol_payload,
    _test_pair_coverage_gate,
    _validate_output,
    canonical_protocol_id,
)


class GroundingFlowPipelineTests(unittest.TestCase):
    def test_trajectory_protocol_tracks_null_seed_but_not_runtime_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = root / "extraction"
            extraction.mkdir()
            (extraction / "extraction_manifest.json").write_text("{}", encoding="utf-8")
            examples = root / "examples.jsonl"
            examples.write_text("{}\n", encoding="utf-8")
            common = dict(
                extraction_dir=extraction,
                examples_path=examples,
                output_dir=root / "output",
                skip_evaluation=True,
            )
            metadata = {"input_protocol": {"legacy_tau_values": [0.01]}}

            payload = _protocol_payload(
                GroundingFlowRunConfig(**common, seed=3, device="cpu"),
                preparation_metadata=metadata,
            )
            first = canonical_protocol_id(payload)
            changed_seed = canonical_protocol_id(
                _protocol_payload(
                    GroundingFlowRunConfig(**common, seed=5, device="cpu"),
                    preparation_metadata=metadata,
                )
            )
            changed_device = canonical_protocol_id(
                _protocol_payload(
                    GroundingFlowRunConfig(**common, seed=3, device="cuda"),
                    preparation_metadata=metadata,
                )
            )

        self.assertNotEqual(first, changed_seed)
        self.assertEqual(first, changed_device)
        self.assertEqual(
            set(payload["upstream_graph_implementation"]),
            {"graph.py", "data.py", "halueval.py"},
        )

    def test_resume_accepts_regenerable_tail_but_rejects_unknown_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "trajectory_cache").mkdir()
            (output / "run_state.json").write_text(
                '{"state":"in_progress","protocol_id":"test"}\n',
                encoding="utf-8",
            )
            for name in (
                "detector.json",
                "test.response_predictions.jsonl",
                "score_freeze.json",
                "evaluation.json",
            ):
                (output / name).write_text("{}\n", encoding="utf-8")

            _validate_output(output, resume=True)
            (output / "run_state.json").unlink()
            with self.assertRaisesRegex(FileExistsError, "run_state"):
                _validate_output(output, resume=True)
            (output / "run_state.json").write_text(
                '{"state":"in_progress","protocol_id":"test"}\n',
                encoding="utf-8",
            )
            (output / "foreign-result.txt").write_text(
                "do not overwrite", encoding="utf-8"
            )
            with self.assertRaisesRegex(FileExistsError, "foreign-result"):
                _validate_output(output, resume=True)

    def test_test_pair_coverage_gate_counts_only_complete_identifiable_pairs(self):
        records = [
            SimpleNamespace(response_id=f"p{pair}-{candidate}", pair_id=f"p{pair}")
            for pair in range(5)
            for candidate in range(2)
        ]
        scored = [
            {"response_id": record.response_id}
            for record in records
            if record.pair_id != "p4"
        ]

        passed = _test_pair_coverage_gate(records, scored, minimum=0.8)
        exclusions = [
            {
                "response_id": record.response_id,
                "exclusion_reason": "unswappable",
                "null_calibration_status": "unswappable",
                "response_tokens": 3,
            }
            for record in records
            if record.pair_id == "p4"
        ]
        failed = _test_pair_coverage_gate(
            records,
            scored,
            minimum=0.9,
            excluded_rows=exclusions,
            fail_on_low_coverage=False,
        )
        strict = _test_pair_coverage_gate(
            records,
            scored,
            minimum=0.9,
            excluded_rows=exclusions,
            fail_on_low_coverage=True,
        )

        self.assertEqual(passed["test_pair_coverage"], 0.8)
        self.assertTrue(passed["passed"])
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["action"], "evaluate_identifiable_subset")
        self.assertFalse(failed["coverage_target_met"])
        self.assertEqual(
            failed["exclusion_summary"]["response_reason_counts"],
            {"unswappable": 2},
        )
        self.assertEqual(strict["action"], "fail_before_label_read")

        with self.assertRaisesRegex(ValueError, "whole HaluEval pairs"):
            _test_pair_coverage_gate(
                records,
                [*scored, {"response_id": "p4-0"}],
                minimum=0.9,
            )
        malformed_exclusions = {
            "missing": exclusions[:-1],
            "duplicate": [*exclusions, exclusions[0]],
            "extra": [
                *exclusions,
                {
                    **exclusions[0],
                    "response_id": "not-in-test",
                },
            ],
        }
        for case, rows in malformed_exclusions.items():
            with self.subTest(case=case), self.assertRaisesRegex(
                ValueError, "exactly identify"
            ):
                _test_pair_coverage_gate(
                    records,
                    scored,
                    minimum=0.9,
                    excluded_rows=rows,
                )

    def test_trajectory_cache_is_weights_only_safe_and_checks_source_identity(self):
        record = TrajectoryRecord(
            response_id="candidate-a",
            pair_id="pair-a",
            partition="train",
            response_token_indices=np.arange(4, 7),
            model_surface=np.zeros((3, 2, 2, 3)),
            mechanism_anchor=np.zeros(3),
            raw_token_summary={"ancestry": np.ones(3), "debt": np.zeros(3)},
            null_swap_fraction=0.2,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trajectory.pt"
            torch.save(
                record.to_artifact(protocol_id="protocol-a", source_identity="source-a"),
                path,
            )
            loaded = torch.load(path, map_location="cpu", weights_only=True)

        restored = TrajectoryRecord.from_artifact(
            loaded,
            expected_protocol_id="protocol-a",
            expected_source_identity="source-a",
        )
        np.testing.assert_allclose(restored.model_surface, record.model_surface)
        with self.assertRaisesRegex(ValueError, "source"):
            TrajectoryRecord.from_artifact(
                loaded,
                expected_protocol_id="protocol-a",
                expected_source_identity="different-source",
            )

    def test_length_residual_is_fit_on_train_scores_without_labels(self):
        train = [
            {"response_id": "a", "score": 1.0, "response_tokens": 1},
            {"response_id": "b", "score": 2.0, "response_tokens": 3},
            {"response_id": "c", "score": 3.0, "response_tokens": 8},
        ]
        test = [
            {"response_id": "d", "score": 2.5, "response_tokens": 5},
        ]

        transformed, coefficients = add_length_residual_scores(train, test)

        self.assertEqual(len(transformed), 1)
        self.assertIn("length_residual_score", transformed[0])
        self.assertEqual(set(coefficients), {"intercept", "log1p_length"})
        self.assertNotIn("label", json.dumps(coefficients))

    def test_labels_are_opened_only_after_prediction_freeze(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            predictions = output / "test.response_predictions.jsonl"
            predictions.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"response_id": "negative", "score": 0.1},
                        {"response_id": "positive", "score": 0.9},
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            labels_path = output / "evaluation_labels.jsonl"
            labels_path.write_text("unused by patched loader\n", encoding="utf-8")
            freeze = freeze_prediction_files(output, [predictions])

            def guarded_label_read(path):
                self.assertTrue((output / "score_freeze.json").is_file())
                self.assertEqual(path, labels_path)
                self.assertEqual(freeze["files"][predictions.name]["records"], 2)
                return {"negative": 0, "positive": 1}

            with mock.patch(
                "grounding_flow.evaluation.load_halueval_response_labels",
                side_effect=guarded_label_read,
            ):
                result = evaluate_frozen_halueval_predictions(
                    output_dir=output,
                    prediction_path=predictions,
                    labels_path=labels_path,
                    pair_by_response={"negative": "pair", "positive": "pair"},
                    response_length_by_id={"negative": 2, "positive": 3},
                    graph_index_rows=[
                        {
                            "response_id": "negative",
                            "pair_id": "pair",
                            "split": "test",
                            "graph_path": "/graphs/negative.graph.pt",
                        },
                        {
                            "response_id": "positive",
                            "pair_id": "pair",
                            "split": "test",
                            "graph_path": "/graphs/positive.graph.pt",
                        },
                    ],
                    graph_index_path=output / "prepared" / "graphs" / "index.json",
                    seed=7,
                    bootstrap_samples=20,
                )

            labeled_index = json.loads(
                (output / "prepared" / "graphs" / "index.json").read_text()
            )

        self.assertAlmostEqual(result["auroc"], 1.0)
        self.assertAlmostEqual(result["paired_accuracy"], 1.0)
        self.assertEqual(
            {row["response_id"]: row["label"] for row in labeled_index},
            {"negative": 0, "positive": 1},
        )

    def test_prediction_freeze_rejects_nested_label_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            predictions = output / "test.response_predictions.jsonl"
            predictions.write_text(
                json.dumps(
                    {
                        "response_id": "candidate",
                        "score": 0.5,
                        "diagnostics": {"evaluation_label": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "label-blind"):
                freeze_prediction_files(output, [predictions])


if __name__ == "__main__":
    unittest.main()
