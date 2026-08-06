import torch

from original.ragtruth_graph import build_original_graph


def test_original_graph_uses_threshold_union_not_max_pooling() -> None:
    sample = {
        "attention_cache_schema": "ragtruth-all-layers-all-heads-sparse-response-csr-v1",
        "source_id": "source",
        "response_id": "response",
        "split": "train",
        "response_idx": 2,
        "num_attention_layers": 1,
        "num_attention_heads": 3,
        "attention_floor": 0.01,
        "cache_dtype": "float16",
        "input_policy": "prompt_response",
        "token_ids": torch.tensor([10, 11, 12]),
        "y_token": torch.tensor([0, 0, 1]),
        "attention_diagonal": torch.zeros((1, 3, 3)),
        "response_row_ptr": torch.tensor([0, 1, 2, 3]),
        "response_column_indices": torch.tensor([0, 0, 0]),
        "response_values": torch.tensor([0.02, 0.07, 0.10]),
    }
    graph = build_original_graph(sample, tau=0.05)
    assert graph["edge_index"].tolist() == [[0], [2]]
    assert torch.allclose(
        graph["edge_attr"], torch.tensor([[0.0, 0.07, 0.10]])
    )
