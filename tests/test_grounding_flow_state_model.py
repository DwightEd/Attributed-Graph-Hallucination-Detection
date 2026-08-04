from __future__ import annotations

import inspect
import unittest

import numpy as np

from grounding_flow.state_model import (
    TrajectoryProjector,
    fit_trajectory_projector,
    fit_two_state_hmm,
)


class GroundingStateModelTests(unittest.TestCase):
    def test_label_free_projector_keeps_layer_head_channels_until_fitted(self):
        generator = np.random.default_rng(29)
        surfaces = [
            generator.normal(size=(5, 2, 3, 3)),
            generator.normal(size=(4, 2, 3, 3)),
        ]

        projector = fit_trajectory_projector(
            surfaces, max_components=4, max_fit_tokens=20, seed=29
        )
        projected = [projector.transform(surface) for surface in surfaces]
        restored = TrajectoryProjector.from_dict(projector.to_dict())

        self.assertEqual(projector.input_shape, (2, 3, 3))
        self.assertEqual(projected[0].shape, (5, 4))
        self.assertEqual(projected[1].shape, (4, 4))
        np.testing.assert_allclose(
            projected[0], restored.transform(surfaces[0]), atol=1e-10
        )

    def test_projector_gives_each_response_equal_sampling_mass(self):
        long_response = np.full((40, 1, 1, 1), -1.0)
        short_response = np.full((4, 1, 1, 1), 1.0)

        projector = fit_trajectory_projector(
            [long_response, short_response],
            max_components=1,
            max_fit_tokens=100,
            seed=7,
        )

        self.assertAlmostEqual(float(projector.feature_mean[0]), 0.0, places=8)

    def test_fit_interface_is_label_blind_and_does_not_assume_detached_is_rare(self):
        parameters = inspect.signature(fit_two_state_hmm).parameters
        self.assertFalse(
            {"label", "labels", "y", "direction_score"}.intersection(parameters)
        )
        generator = np.random.default_rng(31)
        grounded = [generator.normal(-2.0, 0.08, size=(4, 3)) for _ in range(20)]
        detached = [generator.normal(2.0, 0.08, size=(4, 3)) for _ in range(80)]
        sequences = grounded + detached
        anchors = [np.full(4, -1.0) for _ in grounded] + [
            np.full(4, 1.0) for _ in detached
        ]

        model = fit_two_state_hmm(
            sequences,
            anchors,
            seed=31,
            max_iterations=40,
            tolerance=1e-6,
        )
        grounded_scores = [model.score(sequence)["mean"] for sequence in grounded]
        detached_scores = [model.score(sequence)["mean"] for sequence in detached]

        self.assertGreater(np.mean(detached_scores), 0.95)
        self.assertLess(np.mean(grounded_scores), 0.05)
        self.assertGreater(model.state_occupancy[model.detached_state], 0.5)
        self.assertEqual(model.fit_weighting, "response_balanced")
        posterior = model.posterior(detached[0])
        self.assertEqual(posterior.shape, (4, 2))
        np.testing.assert_allclose(posterior.sum(axis=1), 1.0, atol=1e-8)
        self.assertTrue(np.isfinite(posterior).all())

    def test_fit_and_scoring_are_seed_reproducible(self):
        generator = np.random.default_rng(37)
        sequences = [
            generator.normal(location, 0.1, size=(5, 2))
            for location in (-2.0, -1.8, 1.8, 2.0)
        ]
        anchors = [np.full(5, -1.0), np.full(5, -0.8), np.full(5, 0.8), np.full(5, 1.0)]

        first = fit_two_state_hmm(sequences, anchors, seed=41, max_iterations=30)
        second = fit_two_state_hmm(sequences, anchors, seed=41, max_iterations=30)

        np.testing.assert_allclose(first.initial_probability, second.initial_probability)
        np.testing.assert_allclose(first.transition, second.transition)
        np.testing.assert_allclose(first.means, second.means)
        np.testing.assert_allclose(first.variances, second.variances)
        self.assertEqual(first.detached_state, second.detached_state)

    def test_response_weighted_em_objective_is_monotone_for_unequal_lengths(self):
        generator = np.random.default_rng(47)
        sequences = [
            generator.normal(-1.5, 0.25, size=(2, 2)),
            generator.normal(-1.2, 0.25, size=(19, 2)),
            generator.normal(1.2, 0.25, size=(5, 2)),
            generator.normal(1.5, 0.25, size=(41, 2)),
        ]
        anchors = [
            np.full(len(sequence), -1.0 if index < 2 else 1.0)
            for index, sequence in enumerate(sequences)
        ]

        model = fit_two_state_hmm(
            sequences,
            anchors,
            seed=47,
            max_iterations=30,
            tolerance=0.0,
        )

        differences = np.diff(model.log_likelihood_history)
        self.assertTrue(np.all(differences >= -1e-8), differences)

    def test_invalid_training_data_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "two states|observations|sequences"):
            fit_two_state_hmm([np.zeros((1, 2))], [np.zeros(1)])
        with self.assertRaisesRegex(ValueError, "finite"):
            fit_two_state_hmm(
                [np.zeros((2, 2)), np.full((2, 2), np.nan)],
                [np.zeros(2), np.ones(2)],
            )
        with self.assertRaisesRegex(ValueError, "anchor"):
            fit_two_state_hmm(
                [np.zeros((2, 2)), np.ones((2, 2))],
                [np.zeros(1), np.ones(2)],
            )
        with self.assertRaisesRegex(ValueError, "variation|distinct"):
            fit_trajectory_projector(
                [np.zeros((4, 2, 2, 3))], max_components=3, max_fit_tokens=10
            )
        with self.assertRaisesRegex(ValueError, "distinct|states"):
            fit_two_state_hmm(
                [np.zeros((2, 2)), np.zeros((2, 2))],
                [np.zeros(2), np.zeros(2)],
            )
        with self.assertRaisesRegex(ValueError, "orientation"):
            fit_two_state_hmm(
                [np.full((3, 2), -1.0), np.full((3, 2), 1.0)],
                [np.zeros(3), np.zeros(3)],
                seed=3,
            )

    def test_serialized_models_fail_closed_on_invalid_probabilities(self):
        generator = np.random.default_rng(43)
        projector = fit_trajectory_projector(
            [generator.normal(size=(6, 2, 2, 3))],
            max_components=3,
            max_fit_tokens=10,
        )
        invalid_projector = projector.to_dict()
        invalid_projector["feature_scale"][0] = 0.0
        with self.assertRaisesRegex(ValueError, "scale"):
            TrajectoryProjector.from_dict(invalid_projector)

        model = fit_two_state_hmm(
            [generator.normal(-1.0, 0.1, size=(4, 2)), generator.normal(1.0, 0.1, size=(4, 2))],
            [np.full(4, -1.0), np.full(4, 1.0)],
            seed=5,
        )
        invalid_model = model.to_dict()
        invalid_model["transition"][0] = [2.0, 1.0]
        with self.assertRaisesRegex(ValueError, "transition"):
            type(model).from_dict(invalid_model)


if __name__ == "__main__":
    unittest.main()
