from __future__ import annotations

import unittest
from dataclasses import replace

import numpy as np

from grounding_flow.experiment import (
    DetectorFitConfig,
    TrajectoryRecord,
    fit_detector,
)


def _record(identifier: str, location: float, *, seed: int) -> TrajectoryRecord:
    generator = np.random.default_rng(seed)
    surface = generator.normal(location, 0.05, size=(5, 2, 2, 3))
    anchor = np.full(5, location, dtype=np.float64)
    return TrajectoryRecord(
        response_id=identifier,
        pair_id=f"pair-{identifier}",
        partition="train",
        response_token_indices=np.arange(10, 15),
        model_surface=surface,
        mechanism_anchor=anchor,
        raw_token_summary={
            "ancestry": np.full(5, 0.9 if location < 0 else 0.1),
            "debt": np.full(5, 0.1 if location < 0 else 0.9),
            "unknown": np.zeros(5),
        },
        null_swap_fraction=0.25,
        null_effective_fraction=1.0,
        null_stable_model_fraction=1.0,
        null_samples=8,
    )


class GroundingFlowExperimentTests(unittest.TestCase):
    def test_detector_fits_without_labels_and_scores_tokens_and_responses(self):
        train = [
            *[_record(f"grounded-{index}", -2.0, seed=index) for index in range(4)],
            *[_record(f"detached-{index}", 2.0, seed=100 + index) for index in range(4)],
        ]
        config = DetectorFitConfig(
            pca_components=4,
            pca_fit_tokens=100,
            hmm_iterations=30,
            seed=53,
        )

        detector = fit_detector(train, config=config)
        grounded_response, grounded_tokens = detector.score(
            _record("held-grounded", -1.9, seed=211), partition="test"
        )
        detached_response, detached_tokens = detector.score(
            _record("held-detached", 1.9, seed=223), partition="test"
        )

        self.assertGreater(detached_response["score"], grounded_response["score"])
        self.assertEqual(len(grounded_tokens), 5)
        self.assertEqual(len(detached_tokens), 5)
        self.assertEqual(
            [row["token_idx"] for row in grounded_tokens], list(range(10, 15))
        )
        self.assertTrue(
            all(0.0 <= row["score"] <= 1.0 for row in grounded_tokens + detached_tokens)
        )
        self.assertNotIn("label", detector.to_dict())

    def test_detector_fit_is_reproducible(self):
        records = [
            _record("a", -2.0, seed=1),
            _record("b", -1.8, seed=2),
            _record("c", 1.8, seed=3),
            _record("d", 2.0, seed=4),
        ]
        config = DetectorFitConfig(
            pca_components=3, pca_fit_tokens=100, hmm_iterations=20, seed=59
        )

        first = fit_detector(records, config=config)
        second = fit_detector(records, config=config)

        np.testing.assert_allclose(first.state_model.means, second.state_model.means)
        self.assertEqual(first.to_dict(), second.to_dict())

    def test_unidentifiable_nulls_never_enter_fit_or_scoring(self):
        valid = [
            _record("a", -2.0, seed=1),
            _record("b", -1.8, seed=2),
            _record("c", 1.8, seed=3),
            _record("d", 2.0, seed=4),
        ]
        unidentifiable = replace(
            _record("unswappable", 0.0, seed=5),
            model_surface=np.zeros((5, 2, 2, 3)),
            null_effective_fraction=0.0,
            null_samples=0,
            calibration_status="unswappable",
        )
        config = DetectorFitConfig(
            pca_components=3, pca_fit_tokens=100, hmm_iterations=20, seed=61
        )

        baseline = fit_detector(valid, config=config)
        filtered = fit_detector([*valid, unidentifiable], config=config)

        self.assertEqual(baseline.to_dict(), filtered.to_dict())
        with self.assertRaisesRegex(ValueError, "not null-identifiable"):
            filtered.score(unidentifiable)

    def test_trajectory_record_rejects_label_fields_hidden_in_raw_summary(self):
        with self.assertRaisesRegex(ValueError, "label-blind"):
            TrajectoryRecord(
                response_id="bad",
                pair_id="pair-bad",
                partition="train",
                response_token_indices=np.arange(2),
                model_surface=np.zeros((2, 1, 1, 3)),
                mechanism_anchor=np.zeros(2),
                raw_token_summary={"label": np.zeros(2)},
                null_swap_fraction=0.1,
            )

    def test_trajectory_record_rejects_non_structural_token_identity(self):
        with self.assertRaisesRegex(ValueError, "token indices"):
            TrajectoryRecord(
                response_id="bad-order",
                pair_id="pair-bad-order",
                partition="train",
                response_token_indices=np.asarray([4.0, 3.0]),
                model_surface=np.zeros((2, 1, 1, 3)),
                mechanism_anchor=np.zeros(2),
                raw_token_summary={"ancestry": np.zeros(2)},
                null_swap_fraction=0.1,
            )
        with self.assertRaisesRegex(ValueError, "three mechanistic channels"):
            TrajectoryRecord(
                response_id="bad-surface",
                pair_id="pair-bad-surface",
                partition="train",
                response_token_indices=np.asarray([3, 4]),
                model_surface=np.zeros((2, 1, 1, 2)),
                mechanism_anchor=np.zeros(2),
                raw_token_summary={"ancestry": np.zeros(2)},
                null_swap_fraction=0.1,
            )


if __name__ == "__main__":
    unittest.main()
