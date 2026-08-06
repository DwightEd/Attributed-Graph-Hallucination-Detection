"""Gate 0 contracts for minimal, equal-token evidence counterfactuals."""

from __future__ import annotations

import unittest
from collections.abc import Sequence

import torch

from counterfactual_grounding.data.ragtruth import build_ragtruth_layout
from counterfactual_grounding.teacher.counterfactuals import (
    CounterfactualGenerationError,
    generate_equal_token_counterfactual,
    validate_counterfactual_pair,
)


class CharacterTokenizer:
    """Deterministic tokenizer with one token and one offset per character."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        if add_special_tokens:
            raise AssertionError("counterfactual replay must not add tokens")
        encoded: dict[str, list[int] | list[tuple[int, int]]] = {
            "input_ids": [ord(character) for character in text]
        }
        if return_offsets_mapping:
            encoded["offset_mapping"] = [
                (position, position + 1) for position in range(len(text))
            ]
        return encoded


class FirstNumberLengthChangingTokenizer(CharacterTokenizer):
    """Makes every edit to ``A1`` add a token, while edits to ``B2`` remain valid."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        encoded = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        if "A1" not in text:
            input_ids = encoded["input_ids"]
            assert isinstance(input_ids, list)
            input_ids.insert(0, 999_001)
            if return_offsets_mapping:
                offsets = encoded["offset_mapping"]
                assert isinstance(offsets, list)
                offsets.insert(0, (0, 0))
        return encoded


class EveryNumericEditLengthChangingTokenizer(CharacterTokenizer):
    """Accepts the factual string but changes token count for every numeric edit."""

    def __init__(self, factual_text: str) -> None:
        self.factual_text = factual_text

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, list[int] | list[tuple[int, int]]]:
        encoded = super().__call__(
            text,
            add_special_tokens=add_special_tokens,
            return_offsets_mapping=return_offsets_mapping,
        )
        if text != self.factual_text:
            input_ids = encoded["input_ids"]
            assert isinstance(input_ids, list)
            input_ids.append(999_002)
            if return_offsets_mapping:
                offsets = encoded["offset_mapping"]
                assert isinstance(offsets, list)
                offsets.append((len(text), len(text)))
        return encoded


def _summary_layout(
    evidence: str,
    response: str,
    tokenizer: object,
) -> object:
    prompt = f"Summarize only this evidence:\n{evidence}\noutput:"
    return build_ragtruth_layout(
        {
            "source_id": "summary-counterfactual",
            "task_type": "Summary",
            "source_info": evidence,
            "prompt": prompt,
        },
        response_text=response,
        tokenizer=tokenizer,
    )


def _as_long_tensor(values: object) -> torch.Tensor:
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().to(torch.long).flatten()
    if isinstance(values, Sequence):
        return torch.tensor(list(values), dtype=torch.long)
    raise TypeError(f"expected token sequence, got {type(values)!r}")


class EqualTokenCounterfactualGenerationGate0Tests(unittest.TestCase):
    def test_generates_one_character_numeric_edit_inside_evidence_only(self):
        tokenizer = CharacterTokenizer()
        response = "The reported score was eight."
        layout = _summary_layout(
            evidence="The report records score 8 out of 10.",
            response=response,
            tokenizer=tokenizer,
        )

        result = generate_equal_token_counterfactual(layout, tokenizer)

        self.assertEqual(result.factual_text, layout.rendered_text)
        self.assertEqual(len(result.counterfactual_text), len(result.factual_text))
        self.assertEqual(
            _as_long_tensor(result.factual_input_ids).numel(),
            _as_long_tensor(result.counterfactual_input_ids).numel(),
        )
        self.assertEqual(result.audit.response_idx, layout.response_idx)
        self.assertEqual(
            result.counterfactual_text[layout.response_idx :],
            result.factual_text[layout.response_idx :],
        )

        char_start, char_end = result.changed_char_span
        self.assertEqual(char_end - char_start, 1)
        self.assertTrue(result.original_text.isdigit())
        self.assertTrue(result.replacement_text.isdigit())
        self.assertNotEqual(result.original_text, result.replacement_text)
        self.assertEqual(
            result.factual_text[:char_start]
            + result.replacement_text
            + result.factual_text[char_end:],
            result.counterfactual_text,
        )
        evidence_chunk = layout.evidence_chunks[0]
        self.assertGreaterEqual(char_start, evidence_chunk.char_start)
        self.assertLess(char_start, evidence_chunk.char_end)

        independently_validated = validate_counterfactual_pair(
            {
                "input_ids": result.factual_input_ids,
                "response_idx": layout.response_idx,
                "evidence_token_positions": layout.evidence_token_positions,
            },
            {
                "input_ids": result.counterfactual_input_ids,
                "response_idx": layout.response_idx,
                "evidence_token_positions": layout.evidence_token_positions,
            },
        )
        self.assertTrue(
            set(_as_long_tensor(result.changed_positions).tolist())
            <= set(_as_long_tensor(layout.evidence_token_positions).tolist())
        )
        self.assertEqual(
            _as_long_tensor(result.changed_positions).tolist(),
            independently_validated.changed_positions.tolist(),
        )
        self.assertEqual(
            _as_long_tensor(result.changed_positions).tolist(),
            result.audit.changed_positions.tolist(),
        )

    def test_skips_a_numeric_edit_that_changes_token_count_and_tries_later_evidence(
        self,
    ):
        tokenizer = FirstNumberLengthChangingTokenizer()
        layout = _summary_layout(
            evidence="Measurements A1 and B2 were recorded.",
            response="Both measurements were recorded.",
            tokenizer=tokenizer,
        )
        expected_second_digit = layout.rendered_text.index("B2") + 1

        result = generate_equal_token_counterfactual(layout, tokenizer)

        self.assertEqual(
            result.changed_char_span, (expected_second_digit, expected_second_digit + 1)
        )
        self.assertEqual(result.original_text, "2")
        self.assertEqual(
            _as_long_tensor(result.factual_input_ids).numel(),
            _as_long_tensor(result.counterfactual_input_ids).numel(),
        )
        self.assertEqual(result.audit.response_idx, layout.response_idx)

    def test_rejects_evidence_without_a_numeric_perturbation_target(self):
        tokenizer = CharacterTokenizer()
        layout = _summary_layout(
            evidence="The report contains words but no numerals.",
            response="The report contains no measured value.",
            tokenizer=tokenizer,
        )

        with self.assertRaisesRegex(
            CounterfactualGenerationError,
            "digit|numeric|number|candidate",
        ):
            generate_equal_token_counterfactual(layout, tokenizer)

    def test_rejects_when_all_numeric_edits_change_token_length(self):
        evidence = "Measurements A1 and B2 were recorded."
        response = "Both measurements were recorded."
        factual_text = f"Summarize only this evidence:\n{evidence}\noutput:{response}"
        tokenizer = EveryNumericEditLengthChangingTokenizer(factual_text)
        layout = _summary_layout(
            evidence=evidence,
            response=response,
            tokenizer=tokenizer,
        )

        with self.assertRaisesRegex(
            CounterfactualGenerationError,
            "equal.token|length|candidate",
        ):
            generate_equal_token_counterfactual(layout, tokenizer)


if __name__ == "__main__":
    unittest.main()
