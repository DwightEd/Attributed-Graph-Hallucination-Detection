import unittest

import numpy as np
import torch

from unsupervised_token_graph.features import summarize_attention_trace
from unsupervised_token_graph.graph import build_token_graph


class DeviceAwareAttentionSummaryTests(unittest.TestCase):
    def test_tensor_summary_stays_on_its_device_and_matches_numpy_reference(self):
        generator = torch.Generator().manual_seed(17)
        attention = torch.rand((2, 3, 6, 6), generator=generator)
        spans = {"passage": (0, 2), "question": (2, 4), "answer": (4, 6)}

        numpy_summary = summarize_attention_trace(attention.numpy(), spans)
        tensor_summary = summarize_attention_trace(attention, spans)

        self.assertEqual(set(tensor_summary), set(numpy_summary))
        for name, tensor_value in tensor_summary.items():
            self.assertIsInstance(tensor_value, torch.Tensor, name)
            self.assertEqual(tensor_value.device, attention.device, name)
            torch.testing.assert_close(
                tensor_value.detach().cpu().to(torch.float64),
                torch.as_tensor(np.asarray(numpy_summary[name]), dtype=torch.float64),
                rtol=1e-5,
                atol=1e-6,
                msg=name,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_summary_stays_on_cuda_and_matches_cpu_reference(self):
        generator = torch.Generator().manual_seed(29)
        attention_cpu = torch.rand((2, 3, 6, 6), generator=generator)
        spans = {"passage": (0, 2), "question": (2, 4), "answer": (4, 6)}

        cpu_summary = summarize_attention_trace(attention_cpu, spans)
        cuda_summary = summarize_attention_trace(attention_cpu.to("cuda:0"), spans)

        for name, cuda_value in cuda_summary.items():
            self.assertTrue(cuda_value.is_cuda, name)
            torch.testing.assert_close(
                cuda_value.detach().cpu(),
                cpu_summary[name],
                rtol=1e-5,
                atol=1e-6,
                msg=name,
            )


class DeviceAwareGraphConstructionTests(unittest.TestCase):
    def test_graph_supports_a_sample_with_no_edges_above_threshold(self):
        graph = build_token_graph(
            torch.tensor([10, 20, 30]),
            torch.zeros((2, 3, 3, 3), dtype=torch.float16),
            torch.tensor([1, 2, 3]),
        )

        self.assertEqual(tuple(graph["edge_index"].shape), (2, 0))
        self.assertEqual(tuple(graph["edge_attr"].shape), (0, 6))
        self.assertEqual(tuple(graph["edge_mark"].shape), (0, 8))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_attention_builds_an_equivalent_graph_without_cpu_fallback(self):
        attention_cpu = torch.zeros((2, 2, 6, 6), dtype=torch.float16)
        attention_cpu[0, 0, 2, 0] = 0.20
        attention_cpu[0, 1, 4, 2] = 0.30
        attention_cpu[1, 0, 5, 1] = 0.40
        input_ids = torch.tensor([10, 11, 20, 21, 30, 31])
        segment_ids = torch.tensor([1, 1, 2, 2, 3, 3])

        graph_cpu = build_token_graph(input_ids, attention_cpu, segment_ids)
        graph_cuda = build_token_graph(
            input_ids,
            attention_cpu.to("cuda:0"),
            segment_ids,
        )

        for name in (
            "token_ids",
            "segment_ids",
            "answer_mask",
            "x",
            "edge_index",
            "edge_attr",
            "edge_mark",
        ):
            self.assertEqual(graph_cuda[name].device.type, "cuda", name)
            torch.testing.assert_close(
                graph_cuda[name].detach().cpu(),
                graph_cpu[name],
                msg=name,
            )


if __name__ == "__main__":
    unittest.main()
