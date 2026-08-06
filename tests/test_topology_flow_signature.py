import unittest

import torch

from topology_flow import (
    CorruptedAttentionStore,
    CorruptionConfig,
    DenseAttentionStore,
    RelationPreservingSourceShuffleStore,
)
from topology_flow.signature import extract_signature


def _attention_from_rows(rows_by_layer):
    layers = len(rows_by_layer)
    token_count = 1 + max(max(rows) for rows in rows_by_layer)
    attention = torch.zeros((layers, 1, token_count, token_count), dtype=torch.float32)
    for layer, rows in enumerate(rows_by_layer):
        for target, sources in rows.items():
            for source, weight in sources.items():
                attention[layer, 0, target, source] = weight
    return attention


class TopologyFlowSignatureTests(unittest.TestCase):
    def test_signature_exposes_all_four_hallucination_patterns(self):
        response_idx = 3
        grounded_rows = {
            3: {0: 0.34, 1: 0.33, 2: 0.33},
            4: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            5: {0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20},
        }
        hallucinated_rows = {
            3: {0: 1.0},
            4: {3: 0.92, 0: 0.08},
            5: {4: 0.95, 0: 0.05},
        }
        grounded = DenseAttentionStore(
            _attention_from_rows([grounded_rows, grounded_rows]), response_idx
        )
        hallucinated = DenseAttentionStore(
            _attention_from_rows([hallucinated_rows, hallucinated_rows]), response_idx
        )

        clean_signature = extract_signature(grounded)
        error_signature = extract_signature(hallucinated)
        clean = clean_signature.phenomena
        error = error_signature.phenomena

        self.assertGreater(error["prompt_connection_weakness"], clean["prompt_connection_weakness"])
        self.assertGreater(error["response_self_dependence"], clean["response_self_dependence"])
        self.assertGreater(error["edge_sparsity"], clean["edge_sparsity"])
        self.assertGreater(error["response_locality"], clean["response_locality"])
        self.assertGreater(error["edge_concentration"], clean["edge_concentration"])
        self.assertGreater(error["discounted_grounding_loss"], clean["discounted_grounding_loss"])

    def test_source_incidence_changes_path_features_with_same_final_rr_mass(self):
        response_idx = 2
        common = {
            2: {0: 1.0},
            3: {2: 1.0},
            4: {3: 1.0},
        }
        strong_source = dict(common)
        weak_source = dict(common)
        strong_source[5] = {2: 1.0}
        weak_source[5] = {4: 1.0}

        strong = extract_signature(
            DenseAttentionStore(_attention_from_rows([strong_source]), response_idx)
        )
        weak = extract_signature(
            DenseAttentionStore(_attention_from_rows([weak_source]), response_idx)
        )
        names = strong.feature_names
        unsupported_index = names.index("head_center::unsupported_response_feedback")
        grounding_index = names.index("head_center::discounted_prompt_ancestry")

        self.assertGreater(
            float(weak.trajectory[0, unsupported_index]),
            float(strong.trajectory[0, unsupported_index]),
        )
        self.assertLess(
            float(weak.trajectory[0, grounding_index]),
            float(strong.trajectory[0, grounding_index]),
        )

    def test_composite_corruption_moves_signature_in_hypothesized_direction(self):
        response_idx = 3
        rows = {
            3: {0: 0.34, 1: 0.33, 2: 0.33},
            4: {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25},
            5: {0: 0.20, 1: 0.20, 2: 0.20, 3: 0.20, 4: 0.20},
        }
        base = DenseAttentionStore(_attention_from_rows([rows, rows]), response_idx)
        original = extract_signature(base)
        corrupted = extract_signature(
            CorruptedAttentionStore(
                base,
                CorruptionConfig(
                    prompt_transfer=0.6,
                    support_keep_fraction=0.4,
                    concentration_power=2.0,
                    local_window=2,
                ),
            )
        )

        self.assertGreater(
            corrupted.phenomena["prompt_connection_weakness"],
            original.phenomena["prompt_connection_weakness"],
        )
        self.assertGreater(
            corrupted.phenomena["edge_sparsity"], original.phenomena["edge_sparsity"]
        )
        self.assertGreater(
            corrupted.phenomena["edge_concentration"],
            original.phenomena["edge_concentration"],
        )

    def test_relation_preserving_shuffle_isolates_source_path_topology(self):
        response_idx = 2
        rows = {
            2: {0: 0.9, 1: 0.1},
            3: {0: 0.3, 2: 0.7},
            4: {2: 0.2, 3: 0.8},
            5: {2: 0.9, 4: 0.1},
        }
        base = DenseAttentionStore(_attention_from_rows([rows]), response_idx)
        original = extract_signature(base)
        shuffled = extract_signature(RelationPreservingSourceShuffleStore(base, seed=3))
        names = original.feature_names

        for feature in (
            "direct_prompt_mass",
            "edge_sparsity",
            "weight_concentration",
        ):
            index = names.index(f"head_center::{feature}")
            self.assertAlmostEqual(
                float(original.trajectory[0, index]),
                float(shuffled.trajectory[0, index]),
                places=6,
            )
        grounding = names.index("head_center::discounted_prompt_ancestry")
        self.assertNotAlmostEqual(
            float(original.trajectory[0, grounding]),
            float(shuffled.trajectory[0, grounding]),
            places=5,
        )


if __name__ == "__main__":
    unittest.main()
