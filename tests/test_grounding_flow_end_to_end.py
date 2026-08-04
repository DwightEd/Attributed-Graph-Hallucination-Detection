from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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


def _legacy(response_id: str, pair_id: str, *, detached: bool) -> tuple[dict, dict]:
    layers, heads, tokens = 2, 1, len(SEGMENTS)
    layer_rows = [_rows(detached), _rows(detached)]
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


class GroundingFlowEndToEndTests(unittest.TestCase):
    def test_cpu_run_trains_freezes_scores_then_evaluates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = root / "extraction"
            (extraction / "graphs").mkdir(parents=True)
            (extraction / "traces").mkdir()
            names: list[str] = []
            response_ids: list[str] = []
            examples: list[dict[str, object]] = []
            labels: list[dict[str, object]] = []
            for pair_index in range(6):
                pair_id = f"pair-{pair_index}"
                for candidate_index in range(2):
                    response_id = f"candidate-{pair_index}-{candidate_index}"
                    name = response_id + ".pt"
                    legacy, trace = _legacy(
                        response_id, pair_id, detached=bool(candidate_index)
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
                    labels.append(
                        {"example_id": response_id, "label": candidate_index}
                    )
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


if __name__ == "__main__":
    unittest.main()
