from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from attention_graph.ragtruth import (
    build_ragtruth_sentence_records,
    reconstruct_response_offsets,
    write_ragtruth_sentence_records,
)


class _CharacterTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        if tokenize or not add_generation_prompt:
            raise AssertionError("unexpected chat-template options")
        return f"<system>{messages[0]['content']}</system><user>{messages[1]['content']}</user><assistant>"

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        if add_special_tokens or not return_offsets_mapping:
            raise AssertionError("unexpected tokenization options")
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


class RAGTruthSentenceAdapterTests(unittest.TestCase):
    def test_reconstructs_offsets_with_exact_extraction_chat_template(self):
        tokenizer = _CharacterTokenizer()
        prompt = "Question?"
        response = "Answer."
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        combined = rendered + response

        offsets = reconstruct_response_offsets(
            tokenizer,
            prompt=prompt,
            response=response,
            expected_token_ids=[ord(character) for character in combined],
            expected_response_idx=len(rendered),
        )

        self.assertEqual(offsets[len(rendered)], (0, 1))
        self.assertEqual(offsets[len(combined) - 1], (len(response) - 1, len(response)))
        with self.assertRaisesRegex(RuntimeError, "token_ids mismatch"):
            reconstruct_response_offsets(
                tokenizer,
                prompt=prompt,
                response=response,
                expected_token_ids=[0] * len(combined),
                expected_response_idx=len(rendered),
            )
        with self.assertRaisesRegex(RuntimeError, "response_idx mismatch"):
            reconstruct_response_offsets(
                tokenizer,
                prompt=prompt,
                response=response,
                expected_token_ids=[ord(character) for character in combined],
                expected_response_idx=len(rendered) + 1,
            )

    def test_builds_mean_pooled_label_free_sentence_records(self):
        tokenizer = _CharacterTokenizer()
        prompt = "Question"
        response = "First. Second!"
        rendered = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        expected_ids = [ord(character) for character in rendered + response]
        response_idx = len(rendered)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            response_path = root / "response.jsonl"
            source_path = root / "source_info.jsonl"
            attention_path = root / "attention_0001.pt"
            output_path = root / "sentence_scores.jsonl"
            _write_jsonl(
                response_path,
                [
                    {
                        "id": "r1",
                        "source_id": "s1",
                        "response": response,
                        "split": "test",
                    }
                ],
            )
            _write_jsonl(source_path, [{"source_id": "s1", "prompt": prompt}])
            sample = {
                "source_id": "s1",
                "sample_id": "r1",
                "response_id": "r1",
                "dataset_split": "test",
                "response_idx": response_idx,
                "token_ids": torch.tensor(expected_ids, dtype=torch.long),
            }
            token_records = [
                {
                    "source_id": "s1",
                    "sample_id": "r1",
                    "token_idx": response_idx + index,
                    "score": 0.2 if index <= response.index(".") else 0.8,
                }
                for index in range(len(response))
            ]

            with patch(
                "attention_graph.ragtruth.load_attention_record",
                return_value=sample,
            ) as loader:
                records = build_ragtruth_sentence_records(
                    token_records,
                    [attention_path],
                    response_path=response_path,
                    source_path=source_path,
                    tokenizer=tokenizer,
                )
            written = write_ragtruth_sentence_records(records, output_path)
            persisted = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertFalse(loader.call_args.kwargs["include_labels"])
        self.assertEqual(len(records), 2)
        self.assertEqual([record["pooling"] for record in records], ["mean", "mean"])
        self.assertAlmostEqual(records[0]["score"], 0.2)
        self.assertAlmostEqual(records[1]["score"], 0.8)
        self.assertTrue(all("label" not in key for record in records for key in record))
        self.assertEqual(persisted, records)
        self.assertEqual(written.name, "sentence_scores.jsonl")


if __name__ == "__main__":
    unittest.main()
