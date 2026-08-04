from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from grounding_flow.data import (
    FlowDataset,
    prepare_halueval_flow_records,
    read_label_free_examples,
)
from tests.test_attention_graph_halueval import _legacy_graph, _trace_metadata


class GroundingFlowDataTests(unittest.TestCase):
    def test_label_fields_are_rejected_even_when_nested(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "examples.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "example_id": "candidate",
                        "pair_id": "pair",
                        "passage": "knowledge",
                        "question": "question",
                        "metadata": {"gold_label": 1},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "label-blind"):
                read_label_free_examples(path)

    def test_legacy_adapter_keeps_segments_structural_and_splits_whole_pairs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            extraction = root / "extraction"
            graph_dir = extraction / "graphs"
            trace_dir = extraction / "traces"
            graph_dir.mkdir(parents=True)
            trace_dir.mkdir(parents=True)
            names: list[str] = []
            response_ids: list[str] = []
            examples: list[dict[str, object]] = []
            for pair_index in range(3):
                pair_id = f"pair-{pair_index}"
                for candidate_index in range(2):
                    response_id = f"candidate-{pair_index}-{candidate_index}"
                    name = f"{response_id}.pt"
                    torch.save(
                        _legacy_graph(example_id=response_id, pair_id=pair_id),
                        graph_dir / name,
                    )
                    torch.save(
                        _trace_metadata(example_id=response_id, pair_id=pair_id),
                        trace_dir / name,
                    )
                    names.append(name)
                    response_ids.append(response_id)
                    examples.append(
                        {
                            "example_id": response_id,
                            "pair_id": pair_id,
                            "passage": f"knowledge {pair_index}",
                            "question": f"question {pair_index}",
                            "answer": f"answer {candidate_index}",
                        }
                    )
            # The extraction run may intentionally contain a 2,000-candidate
            # cohort while its label-free examples sidecar retains the full
            # 20,000-candidate HaluEval inventory.  Extra examples are valid
            # as long as every manifest response is covered.
            examples.extend(
                [
                    {
                        "example_id": "not-extracted-0",
                        "pair_id": "not-extracted-pair",
                        "passage": "unused knowledge",
                        "question": "unused question",
                        "answer": "unused candidate 0",
                    },
                    {
                        "example_id": "not-extracted-1",
                        "pair_id": "not-extracted-pair",
                        "passage": "unused knowledge",
                        "question": "unused question",
                        "answer": "unused candidate 1",
                    },
                ]
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
            examples_path.write_text(
                "\n".join(json.dumps(row) for row in examples) + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "expected candidate count"):
                prepare_halueval_flow_records(
                    extraction_dir=extraction,
                    examples_path=examples_path,
                    output_dir=root / "rejected",
                    expected_candidates=8,
                    require_complete_cache=True,
                    conversion_device="cpu",
                )

            partitions, metadata = prepare_halueval_flow_records(
                extraction_dir=extraction,
                examples_path=examples_path,
                output_dir=root / "output",
                validation_fraction=0.2,
                test_fraction=0.2,
                seed=11,
                group_by_prompt=False,
                expected_candidates=6,
                require_complete_cache=True,
                conversion_device="cpu",
            )

            self.assertEqual(
                {name: len(records) for name, records in partitions.items()},
                {"train": 2, "validation": 2, "test": 2},
            )
            pair_sets = [
                {record.pair_id for record in partitions[name]}
                for name in ("train", "validation", "test")
            ]
            self.assertFalse(pair_sets[0] & pair_sets[1])
            self.assertFalse(pair_sets[0] & pair_sets[2])
            self.assertFalse(pair_sets[1] & pair_sets[2])
            self.assertEqual(
                metadata["input_protocol"]["segment_role"],
                "structural_source_type_only",
            )
            self.assertTrue(metadata["inventory"]["examples_cover_manifest"])
            self.assertFalse(
                metadata["inventory"]["manifest_covers_all_examples"]
            )
            graph, segments, record = FlowDataset(partitions["train"])[0]
            self.assertTrue(torch.equal(graph.token_ids.cpu(), record.token_ids))
            self.assertEqual(int(torch.nonzero(segments == 3)[0]), graph.response_idx)

            record.token_ids[0] = -1
            with self.assertRaisesRegex(ValueError, "token identity"):
                FlowDataset([record])[0]

            missing_example_path = root / "examples_missing_manifest_candidate.jsonl"
            missing_example_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in examples
                    if row["example_id"] != response_ids[0]
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "absent from examples"):
                prepare_halueval_flow_records(
                    extraction_dir=extraction,
                    examples_path=missing_example_path,
                    output_dir=root / "missing-example-rejected",
                    expected_candidates=6,
                    require_complete_cache=True,
                    conversion_device="cpu",
                )


if __name__ == "__main__":
    unittest.main()
