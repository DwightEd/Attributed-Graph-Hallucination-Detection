from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from grounding_flow.pipeline import GroundingFlowRunConfig, run_grounding_flow


SEGMENTS = torch.tensor([1, 1, 2, 2, 3, 3, 3, 3])


def _rows(detached: bool) -> list[list[tuple[int, float]]]:
    if detached:
        return [
            [(0, 0.10), (1, 0.10), (2, 0.10), (3, 0.10)],
            [(0, 0.05), (1, 0.05), (2, 0.10), (3, 0.10), (4, 0.50)],
            [(0, 0.05), (1, 0.05), (2, 0.05), (3, 0.05), (4, 0.30), (5, 0.30)],
            [(0, 0.05), (1, 0.05), (2, 0.05), (3, 0.05), (4, 0.25), (5, 0.25), (6, 0.20)],
        ]
    return [
        [(0, 0.30), (1, 0.30), (2, 0.10), (3, 0.10)],
        [(0, 0.10), (1, 0.10), (2, 0.30), (3, 0.30), (4, 0.10)],
        [(0, 0.20), (1, 0.20), (2, 0.20), (3, 0.20), (4, 0.10), (5, 0.05)],
        [(0, 0.10), (1, 0.10), (2, 0.10), (3, 0.10), (4, 0.15), (5, 0.15), (6, 0.10)],
    ]


def _legacy(
    response_id: str,
    pair_id: str,
    *,
    detached: bool,
    unswappable: bool = False,
) -> tuple[dict, dict]:
    layers, heads, tokens = 2, 1, len(SEGMENTS)
    rows = (
        [[(0, 0.20), (1, 0.20)] for _ in range(tokens - 4)]
        if unswappable
        else _rows(detached)
    )
    layer_rows = [rows, rows]
    pairs = sorted(
        {
            (source, query)
            for rows in layer_rows
            for query, row in enumerate(rows, start=4)
            for source, _ in row
        },
        key=lambda item: (item[1], item[0]),
    )
    edge_attr = torch.zeros((len(pairs), layers * heads), dtype=torch.float32)
    lookup = {pair: index for index, pair in enumerate(pairs)}
    diagonal = torch.zeros((tokens, layers * heads), dtype=torch.float32)
    for layer, rows in enumerate(layer_rows):
        for query, row in enumerate(rows, start=4):
            for source, weight in row:
                edge_attr[lookup[(source, query)], layer] = weight
            diagonal[query, layer] = 1.0 - sum(weight for _, weight in row)
    fingerprint = f"fingerprint-{response_id}"
    legacy = {
        "schema_version": "token_graph_v2",
        "example_id": response_id,
        "pair_id": pair_id,
        "token_ids": torch.arange(100, 100 + tokens),
        "segment_ids": SEGMENTS.clone(),
        "answer_mask": SEGMENTS == 3,
        "x": diagonal,
        "x_view_slices": {"attention_diagonal": (0, layers * heads)},
        "edge_index": torch.tensor(pairs, dtype=torch.long).T.contiguous(),
        "edge_attr": edge_attr,
        "graph_config": {"tau": 0.01, "include_prefix_edges": True},
        "extraction_fingerprint": fingerprint,
    }
    trace = {
        "example_id": response_id,
        "pair_id": pair_id,
        "dataset": "halueval_qa",
        "input_ids": legacy["token_ids"].clone(),
        "segment_ids": SEGMENTS.clone(),
        "attention_shape": [layers, heads, tokens, tokens],
        "edge_threshold": 0.01,
        "extraction_fingerprint": fingerprint,
    }
    return legacy, trace


def _write_fixture(
    root: Path,
    *,
    pair_count: int,
    unswappable_responses: frozenset[str] = frozenset(),
) -> tuple[Path, Path, Path]:
    extraction = root / "extraction"
    (extraction / "graphs").mkdir(parents=True)
    (extraction / "traces").mkdir()
    names: list[str] = []
    response_ids: list[str] = []
    examples: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for pair_index in range(pair_count):
        pair_id = f"pair-{pair_index}"
        for candidate_index in range(2):
            response_id = f"candidate-{pair_index}-{candidate_index}"
            name = response_id + ".pt"
            legacy, trace = _legacy(
                response_id,
                pair_id,
                detached=bool(candidate_index),
                unswappable=response_id in unswappable_responses,
            )
            torch.save(legacy, extraction / "graphs" / name)
            torch.save(trace, extraction / "traces" / name)
            names.append(name)
            response_ids.append(response_id)
            examples.append(
                {
                    "example_id": response_id,
                    "pair_id": pair_id,
                    "passage": f"evidence {pair_index}",
                    "question": f"question {pair_index}",
                    "answer": f"candidate {candidate_index}",
                }
            )
            labels.append({"example_id": response_id, "label": candidate_index})
    (extraction / "extraction_manifest.json").write_text(
        json.dumps(
            {
                "state": "complete",
                "graph_files": names,
                "example_ids": response_ids,
            }
        ),
        encoding="utf-8",
    )
    examples_path = root / "examples.jsonl"
    labels_path = root / "labels.jsonl"
    examples_path.write_text(
        "\n".join(json.dumps(row) for row in examples) + "\n",
        encoding="utf-8",
    )
    labels_path.write_text(
        "\n".join(json.dumps(row) for row in labels) + "\n",
        encoding="utf-8",
    )
    return extraction, examples_path, labels_path


class GroundingFlowEndToEndTests(unittest.TestCase):
    def test_cpu_run_trains_freezes_scores_then_evaluates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction, examples_path, labels_path = _write_fixture(
                root, pair_count=6
            )
            output = root / "output"

            result = run_grounding_flow(
                GroundingFlowRunConfig(
                    extraction_dir=extraction,
                    examples_path=examples_path,
                    evaluation_labels_path=labels_path,
                    output_dir=output,
                    device="cpu",
                    validation_fraction=0.2,
                    test_fraction=0.2,
                    group_by_prompt=False,
                    num_nulls=4,
                    pca_components=3,
                    pca_fit_tokens=100,
                    hmm_iterations=20,
                    bootstrap_samples=20,
                    seed=17,
                )
            )

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["labels_read_during"], "evaluation_only")
            self.assertTrue((output / "detector.json").is_file())
            self.assertTrue((output / "score_freeze.json").is_file())
            self.assertTrue((output / "evaluation.json").is_file())
            self.assertIn("auroc", result["core_metrics"])
            self.assertGreater(result["wall_time_seconds"]["total"], 0.0)
            self.assertEqual(result["training"]["fit_weighting"], "response_balanced")
            self.assertTrue(result["identifiable_coverage_gate"]["passed"])
            frozen = json.loads((output / "score_freeze.json").read_text())
            self.assertEqual(frozen["state"], "scores_frozen_before_label_read")

    def test_low_coverage_is_scoped_and_strict_mode_stops_before_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction, examples_path, labels_path = _write_fixture(
                root,
                pair_count=10,
                # With split_seed=17 and pair splitting, pair-1 and pair-4 are
                # test pairs.  Making one candidate in pair-4 unswappable
                # yields one scored pair out of two without affecting fit.
                unswappable_responses=frozenset({"candidate-4-0"}),
            )

            common = {
                "extraction_dir": extraction,
                "examples_path": examples_path,
                "evaluation_labels_path": labels_path,
                "device": "cpu",
                "validation_fraction": 0.2,
                "test_fraction": 0.2,
                "split_seed": 17,
                "group_by_prompt": False,
                "num_nulls": 4,
                "pca_components": 3,
                "pca_fit_tokens": 100,
                "hmm_iterations": 5,
                "bootstrap_samples": 5,
                "seed": 17,
            }
            report_output = root / "report-low-coverage"

            from grounding_flow import evaluation as flow_evaluation

            original_loader = flow_evaluation.load_halueval_response_labels

            def guarded_loader(path):
                self.assertTrue((report_output / "score_freeze.json").is_file())
                return original_loader(path)

            with mock.patch(
                "grounding_flow.evaluation.load_halueval_response_labels",
                side_effect=guarded_loader,
            ) as label_loader:
                result = run_grounding_flow(
                    GroundingFlowRunConfig(
                        output_dir=report_output,
                        **common,
                    )
                )

            self.assertEqual(label_loader.call_count, 1)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(
                result["experiment_scope"], "low_identifiability_subset_pilot"
            )
            self.assertEqual(result["identifiable_coverage_gate"]["test_pair_coverage"], 0.5)
            self.assertEqual(
                result["identifiable_coverage_gate"]["action"],
                "evaluate_identifiable_subset",
            )
            self.assertFalse(
                result["identifiable_coverage_gate"]["coverage_target_met"]
            )
            self.assertTrue((report_output / "evaluation.json").is_file())
            test_predictions = [
                json.loads(line)
                for line in (report_output / "test.response_predictions.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
                if line.strip()
            ]
            self.assertEqual(len(test_predictions), 2)
            self.assertEqual(
                {row["pair_id"] for row in test_predictions}, {"pair-1"}
            )

            strict_output = root / "strict-low-coverage"
            with mock.patch(
                "grounding_flow.evaluation.load_halueval_response_labels"
            ) as strict_label_loader:
                with self.assertRaisesRegex(RuntimeError, "coverage"):
                    run_grounding_flow(
                        GroundingFlowRunConfig(
                            output_dir=strict_output,
                            fail_on_low_coverage=True,
                            **common,
                        )
                    )
            strict_label_loader.assert_not_called()
            self.assertFalse((strict_output / "score_freeze.json").exists())
            self.assertFalse((strict_output / "evaluation.json").exists())


if __name__ == "__main__":
    unittest.main()
