"""Gate 0 contracts for label-blind RAGTruth E/Q/R layout replay.

The production adapter does not exist yet.  These tests intentionally freeze
the smallest public API needed to turn each RAGTruth task prompt into a token
layout without consulting hallucination labels.
"""

from __future__ import annotations

import ast
import unittest
from collections.abc import Sequence

from counterfactual_grounding.data.graph import Segment
from counterfactual_grounding.data.ragtruth import build_ragtruth_layout


class CharacterTokenizer:
    """A deterministic offset-aware tokenizer with one token per character."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        if add_special_tokens:
            raise AssertionError("layout replay must not insert unregistered tokens")
        encoded: dict[str, list[int] | list[tuple[int, int]]] = {
            "input_ids": [ord(character) for character in text]
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = [
                (position, position + 1) for position in range(len(text))
            ]
        return encoded


def _as_list(values: object) -> list[int]:
    if hasattr(values, "tolist"):
        converted = values.tolist()  # type: ignore[union-attr]
        return [int(value) for value in converted]
    if isinstance(values, Sequence):
        return [int(value) for value in values]
    raise TypeError(f"expected a one-dimensional token sequence, got {type(values)!r}")


def _assert_evidence_and_query_partition(
    test: unittest.TestCase,
    layout: object,
    prompt: str,
) -> None:
    segment_ids = _as_list(layout.segment_ids)  # type: ignore[attr-defined]
    evidence_positions = set(
        _as_list(layout.evidence_token_positions)  # type: ignore[attr-defined]
    )

    test.assertEqual(layout.response_idx, len(prompt))  # type: ignore[attr-defined]
    test.assertEqual(
        evidence_positions,
        {
            position
            for chunk in layout.evidence_chunks  # type: ignore[attr-defined]
            for position in range(chunk.char_start, chunk.char_end)
        },
    )
    for position in range(len(prompt)):
        expected = Segment.EVIDENCE if position in evidence_positions else Segment.QUERY
        test.assertEqual(
            Segment(segment_ids[position]),
            expected,
            msg=f"wrong segment at prompt character {position}: {prompt[position]!r}",
        )


class RagTruthLayoutGate0Tests(unittest.TestCase):
    def test_summary_source_info_is_the_exact_evidence_span(self):
        source_info = "Ada wrote the first published algorithm."
        prompt = f"Summarize the following note in ten words:\n{source_info}\n\noutput:"
        record = {
            "source_id": "summary-1",
            "task_type": "Summary",
            "source_info": source_info,
            "prompt": prompt,
        }

        layout = build_ragtruth_layout(
            record,
            response_text="Ada authored an early algorithm.",
            tokenizer=CharacterTokenizer(),
        )

        self.assertEqual(len(layout.evidence_chunks), 1)
        chunk = layout.evidence_chunks[0]
        self.assertEqual(chunk.chunk_id, 0)
        self.assertEqual(chunk.text, source_info)
        self.assertEqual(
            layout.rendered_text[chunk.char_start : chunk.char_end], source_info
        )
        _assert_evidence_and_query_partition(self, layout, prompt)

    def test_qa_question_is_query_and_each_passage_is_a_separate_evidence_chunk(self):
        question = "Which city is the capital of France?"
        passages = (
            "passage 1:Paris is the capital of France.\n\n"
            "passage 2:Berlin is the capital of Germany.\n\n"
            "passage 3:Madrid is the capital of Spain.\n\n"
        )
        expected_chunks = [
            "passage 1:Paris is the capital of France.",
            "passage 2:Berlin is the capital of Germany.",
            "passage 3:Madrid is the capital of Spain.",
        ]
        prompt = (
            "Briefly answer the following question:\n"
            f"{question}\n"
            "Use only the following passages:\n"
            f"{passages}"
            "If they are insufficient, say so.\n"
            "output:"
        )
        record = {
            "source_id": "qa-1",
            "task_type": "QA",
            "source_info": {"question": question, "passages": passages},
            "prompt": prompt,
        }

        layout = build_ragtruth_layout(
            record,
            response_text="Paris.",
            tokenizer=CharacterTokenizer(),
        )

        self.assertEqual(
            [chunk.text for chunk in layout.evidence_chunks], expected_chunks
        )
        for expected_id, chunk in enumerate(layout.evidence_chunks):
            self.assertEqual(chunk.chunk_id, expected_id)
            self.assertEqual(
                layout.rendered_text[chunk.char_start : chunk.char_end], chunk.text
            )

        segment_ids = _as_list(layout.segment_ids)
        question_start = prompt.index(question)
        self.assertTrue(
            all(
                Segment(segment_ids[position]) is Segment.QUERY
                for position in range(question_start, question_start + len(question))
            )
        )
        _assert_evidence_and_query_partition(self, layout, prompt)

    def test_qa_question_may_be_quoted_again_inside_an_evidence_passage(self):
        question = "What is the launch code?"
        passages = (
            f"passage 1:The document asks: {question}\n\n"
            "passage 2:The launch code is 314.\n\n"
            "passage 3:No other code is listed.\n\n"
        )
        prompt = f"Briefly answer:\n{question}\nUse these passages:\n{passages}output:"
        layout = build_ragtruth_layout(
            {
                "source_id": "qa-duplicate-question",
                "task_type": "QA",
                "source_info": {"question": question, "passages": passages},
                "prompt": prompt,
            },
            response_text="314.",
            tokenizer=CharacterTokenizer(),
        )

        self.assertEqual(len(layout.evidence_chunks), 3)
        question_start = prompt.index(question)
        segment_ids = _as_list(layout.segment_ids)
        self.assertTrue(
            all(
                Segment(segment_ids[position]) is Segment.QUERY
                for position in range(question_start, question_start + len(question))
            )
        )

    def test_data2txt_python_literal_is_structurally_equal_to_source_info_and_evidence(
        self,
    ):
        source_info = {
            "name": "Cafe Example",
            "open": True,
            "parking": None,
            "reviews": [{"stars": 5, "text": "Fresh bread."}],
        }
        structured_literal = repr(source_info)
        for output_marker in ("Overview:", "output:"):
            with self.subTest(output_marker=output_marker):
                prompt = (
                    "Instruction:\n"
                    "Write an objective overview using only the record.\n"
                    "Structured data:\n"
                    f"{structured_literal}\n"
                    f"{output_marker}"
                )
                record = {
                    "source_id": "data2txt-1",
                    "task_type": "Data2txt",
                    "source_info": source_info,
                    "prompt": prompt,
                }

                layout = build_ragtruth_layout(
                    record,
                    response_text="Cafe Example serves fresh bread.",
                    tokenizer=CharacterTokenizer(),
                )

                self.assertEqual(len(layout.evidence_chunks), 1)
                chunk = layout.evidence_chunks[0]
                self.assertEqual(chunk.text, structured_literal)
                self.assertEqual(ast.literal_eval(chunk.text), source_info)
                self.assertEqual(
                    layout.rendered_text[chunk.char_start : chunk.char_end],
                    structured_literal,
                )
                _assert_evidence_and_query_partition(self, layout, prompt)

    def test_token_replay_preserves_complete_prompt_and_marks_response_suffix(self):
        source_info = "Grounded fact."
        prompt = f"Summarize:\n{source_info}\noutput:"
        response = "A grounded response."
        record = {
            "source_id": "summary-replay",
            "task_type": "Summary",
            "source_info": source_info,
            "prompt": prompt,
        }

        layout = build_ragtruth_layout(
            record,
            response_text=response,
            tokenizer=CharacterTokenizer(),
        )

        self.assertEqual(layout.rendered_text, prompt + response)
        self.assertEqual(
            _as_list(layout.input_ids), [ord(char) for char in prompt + response]
        )
        self.assertEqual(layout.response_idx, len(prompt))
        self.assertEqual(
            [Segment(value) for value in _as_list(layout.segment_ids)[len(prompt) :]],
            [Segment.RESPONSE] * len(response),
        )

        expected_evidence_positions = list(
            range(
                prompt.index(source_info), prompt.index(source_info) + len(source_info)
            )
        )
        self.assertEqual(
            _as_list(layout.evidence_token_positions), expected_evidence_positions
        )
        self.assertEqual(
            _as_list(layout.evidence_chunks[0].token_positions),
            expected_evidence_positions,
        )

    def test_data2txt_rejects_a_structured_literal_that_disagrees_with_source_info(
        self,
    ):
        record = {
            "source_id": "data2txt-mismatch",
            "task_type": "Data2txt",
            "source_info": {"name": "Expected Cafe", "open": True},
            "prompt": (
                "Instruction:\nStructured data:\n"
                "{'name': 'Different Cafe', 'open': True}\nOverview:"
            ),
        }

        with self.assertRaisesRegex(ValueError, "source_info|structur|equivalent"):
            build_ragtruth_layout(
                record,
                response_text="An overview.",
                tokenizer=CharacterTokenizer(),
            )

    def test_source_records_with_hallucination_labels_are_rejected(self):
        record = {
            "source_id": "summary-labeled",
            "task_type": "Summary",
            "source_info": "Evidence.",
            "prompt": "Summarize:\nEvidence.\noutput:",
            "hallucination_labels": [{"start": 0, "end": 3}],
        }

        with self.assertRaisesRegex(ValueError, "label|label-blind|forbidden"):
            build_ragtruth_layout(
                record,
                response_text="Answer.",
                tokenizer=CharacterTokenizer(),
            )


if __name__ == "__main__":
    unittest.main()
