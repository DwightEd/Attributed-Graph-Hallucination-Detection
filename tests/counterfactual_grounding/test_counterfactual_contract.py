"""Gate 0 token-alignment contracts for CEPT counterfactual evidence."""

from __future__ import annotations

import unittest

import torch

from counterfactual_grounding.teacher.counterfactuals import (
    validate_counterfactual_pair,
)


def _factual_record() -> dict[str, object]:
    return {
        "source_id": "source-1",
        "response_id": "response-1",
        "input_ids": torch.tensor(
            [101, 102, 201, 202, 301, 302], dtype=torch.long
        ),
        "response_idx": 4,
        "evidence_token_positions": torch.tensor([2, 3], dtype=torch.long),
    }


class CounterfactualTokenContractGate0Tests(unittest.TestCase):
    def test_equal_token_length_evidence_replacement_preserves_response_predictors(self):
        factual = _factual_record()
        counterfactual = {
            **factual,
            "input_ids": torch.tensor(
                [101, 102, 211, 212, 301, 302], dtype=torch.long
            ),
        }

        audit = validate_counterfactual_pair(factual, counterfactual)

        self.assertEqual(audit.token_count, 6)
        self.assertEqual(audit.changed_positions.tolist(), [2, 3])
        self.assertEqual(audit.predictor_positions.tolist(), [3, 4])
        self.assertEqual(audit.response_token_ids.tolist(), [301, 302])

    def test_rejects_replacement_with_different_full_token_length(self):
        factual = _factual_record()
        counterfactual = {
            **factual,
            "input_ids": torch.tensor(
                [101, 102, 211, 212, 213, 301, 302], dtype=torch.long
            ),
            "response_idx": 5,
            "evidence_token_positions": torch.tensor([2, 3, 4], dtype=torch.long),
        }

        with self.assertRaisesRegex(ValueError, "equal token length|token.*length"):
            validate_counterfactual_pair(factual, counterfactual)

    def test_equal_total_length_is_insufficient_if_response_boundary_moves(self):
        factual = _factual_record()
        counterfactual = {
            **factual,
            "input_ids": torch.tensor(
                [101, 102, 211, 301, 302, 303], dtype=torch.long
            ),
            "response_idx": 3,
            "evidence_token_positions": torch.tensor([2], dtype=torch.long),
        }

        with self.assertRaisesRegex(
            ValueError, "response.*boundary|predictor.*position"
        ):
            validate_counterfactual_pair(factual, counterfactual)

    def test_rejects_changes_outside_the_registered_evidence_positions(self):
        factual = _factual_record()
        counterfactual = {
            **factual,
            # Position 1 is query, so this is not a minimal evidence-only edit.
            "input_ids": torch.tensor(
                [101, 999, 211, 212, 301, 302], dtype=torch.long
            ),
        }

        with self.assertRaisesRegex(ValueError, "evidence|changed position"):
            validate_counterfactual_pair(factual, counterfactual)

    def test_rejects_hallucination_labels_in_teacher_records(self):
        factual = _factual_record()
        factual["hallucination_labels"] = torch.tensor([0, 1])
        counterfactual = {
            **_factual_record(),
            "input_ids": torch.tensor(
                [101, 102, 211, 212, 301, 302], dtype=torch.long
            ),
        }

        with self.assertRaisesRegex(ValueError, "label|label-blind|forbidden"):
            validate_counterfactual_pair(factual, counterfactual)


if __name__ == "__main__":
    unittest.main()
