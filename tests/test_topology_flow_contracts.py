import unittest

import torch

from topology_flow.contracts import FORMAL_SPARSE_CSR_SCHEMA, store_from_sample
from topology_flow.signature import extract_signature


class TopologyFlowContractTests(unittest.TestCase):
    def test_dense_and_formal_sparse_views_produce_same_signature(self):
        response_idx = 2
        token_count = 5
        dense = torch.zeros((1, 1, token_count, token_count), dtype=torch.float32)
        dense[0, 0, 2, :2] = torch.tensor([0.6, 0.4])
        dense[0, 0, 3, :3] = torch.tensor([0.3, 0.2, 0.5])
        dense[0, 0, 4, :4] = torch.tensor([0.2, 0.1, 0.3, 0.4])
        dense_store = store_from_sample(
            {
                "source_id": "s",
                "original_idx": 0,
                "response_idx": response_idx,
                "attention": dense,
            }
        )

        columns = []
        values = []
        row_ptr = [0]
        for target in range(response_idx, token_count):
            row = dense[0, 0, target, :target]
            active = torch.nonzero(row > 0, as_tuple=False).flatten()
            columns.extend(active.tolist())
            values.extend(row[active].tolist())
            row_ptr.append(len(values))
        sparse_store = store_from_sample(
            {
                "attention_cache_schema": FORMAL_SPARSE_CSR_SCHEMA,
                "source_id": "s",
                "response_id": "r",
                "original_idx": 0,
                "response_idx": response_idx,
                "num_attention_layers": 1,
                "num_attention_heads": 1,
                "attention_diagonal": torch.zeros((1, 1, token_count)),
                "response_row_ptr": torch.tensor(row_ptr, dtype=torch.int64),
                "response_column_indices": torch.tensor(columns, dtype=torch.int32),
                "response_values": torch.tensor(values, dtype=torch.float16),
            }
        )

        dense_signature = extract_signature(dense_store)
        sparse_signature = extract_signature(sparse_store)
        self.assertTrue(
            torch.allclose(
                dense_signature.trajectory,
                sparse_signature.trajectory,
                atol=5e-4,
            )
        )

    def test_label_fields_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "label blind"):
            store_from_sample(
                {
                    "source_id": "s",
                    "original_idx": 0,
                    "response_idx": 1,
                    "attention": torch.ones((1, 1, 2, 2)),
                    "label": 1,
                }
            )

    def test_sparse_cache_with_unobserved_row_is_explicit_not_fatal(self):
        response_idx = 2
        token_count = 5
        store = store_from_sample(
            {
                "attention_cache_schema": FORMAL_SPARSE_CSR_SCHEMA,
                "source_id": "s",
                "response_id": "r",
                "original_idx": 0,
                "response_idx": response_idx,
                "num_attention_layers": 1,
                "num_attention_heads": 1,
                "attention_diagonal": torch.zeros((1, 1, token_count)),
                "response_row_ptr": torch.tensor([0, 2, 2, 4], dtype=torch.int64),
                "response_column_indices": torch.tensor([0, 1, 2, 3], dtype=torch.int32),
                "response_values": torch.tensor([0.6, 0.4, 0.5, 0.5]),
            }
        )
        signature = extract_signature(store)
        index = signature.feature_names.index("head_center::observed_row_fraction")
        self.assertAlmostEqual(float(signature.trajectory[0, index]), 2.0 / 3.0, places=6)


if __name__ == "__main__":
    unittest.main()
