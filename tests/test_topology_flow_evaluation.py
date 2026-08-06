import json
from pathlib import Path
import tempfile
import unittest

from topology_flow.evaluation import evaluate_ragtruth, join_ragtruth_scores


class TopologyFlowEvaluationTests(unittest.TestCase):
    def test_labels_are_joined_only_in_evaluation(self):
        scores = [
            {"original_idx": 0, "source_id": "s0", "topology_anomaly_score": 0.1},
            {"original_idx": 1, "source_id": "s1", "topology_anomaly_score": 0.9},
        ]
        responses = [
            {"source_id": "s0", "model": "m", "split": "test", "labels": []},
            {"source_id": "s1", "model": "m", "split": "test", "labels": [{"start": 0}]},
        ]
        sources = [
            {"source_id": "s0", "task_type": "QA"},
            {"source_id": "s1", "task_type": "QA"},
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            response_path = root / "response.jsonl"
            source_path = root  / "source.jsonl"
            score_path = root / "scores.jsonl"
            output_path = root / "evaluation.json"
            response_path.write_text("\n".join(json.dumps(row) for row in responses) + "\n")
            source_path.write_text("\n".join(json.dumps(row) for row in sources) + "\n")
            score_path.write_text("\n".join(json.dumps(row) for row in scores) + "\n")
            report = evaluate_ragtruth(
                score_path,
                response_path=response_path,
                source_path=source_path,
                output_path=output_path,
            )

        self.assertEqual(report["overall"]["auroc"], 1.0)
        self.assertEqual(report["overall"]["average_precision"], 1.0)
        self.assertTrue(output_path.exists() is False)  # temporary directory is gone

    def test_score_records_with_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            responses = root / "responses.jsonl"
            sources = root / "sources.jsonl"
            responses.write_text(json.dumps({"source_id": "s", "labels": []}) + "\n")
            sources.write_text(json.dumps({"source_id": "s", "task_type": "QA"}) + "\n")
            with self.assertRaisesRegex(ValueError, "label-free"):
                join_ragtruth_scores(
                    [
                        {
                            "original_idx": 0,
                            "source_id": "s",
                            "topology_anomaly_score": 0.1,
                            "label": 0,
                        }
                    ],
                    response_path=responses,
                    source_path=sources,
                )


if __name__ == "__main__":
    unittest.main()
