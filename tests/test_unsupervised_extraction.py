import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from unsupervised_token_graph.data import compose_example
from unsupervised_token_graph.extract import (
    _artifact_path,
    _estimate_attention_storage_bytes,
    _fingerprint,
    _resolve_postprocess_device,
    build_parser,
    compute_teacher_forced_statistics,
    extract_example_trace,
    extract_prepared_dataset,
    summarize_trace_record,
    tokenize_example_once,
)
from unsupervised_token_graph.identity import model_source_signature


class _FakeTokenizer:
    def __init__(self, example):
        self.example = example
        self.call_count = 0
        self.last_options = None

    def __call__(self, text, **options):
        self.call_count += 1
        self.last_options = options
        offsets = [(0, 0)] + [
            self.example.segment_char_spans[name]
            for name in ("passage", "question", "answer")
        ]
        return {
            "input_ids": torch.tensor([[1, 11, 22, 33]]),
            "attention_mask": torch.ones((1, 4), dtype=torch.long),
            "offset_mapping": torch.tensor([offsets]),
            "special_tokens_mask": torch.tensor([[1, 0, 0, 0]]),
        }


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, attention_mask, **options):
        token_count = input_ids.shape[1]
        logits = torch.zeros((1, token_count, 40), device=input_ids.device)
        attention = torch.zeros((1, 1, token_count, token_count), device=input_ids.device)
        attention[:, :, 3, 1:4] = torch.tensor([0.6, 0.2, 0.2], device=input_ids.device)
        hidden = torch.arange(token_count * 3, device=input_ids.device).reshape(1, token_count, 3).float()
        return SimpleNamespace(
            logits=logits,
            attentions=(attention,),
            hidden_states=(hidden * 0.0, hidden),
        )


class SingleTokenizationTests(unittest.TestCase):
    def test_extraction_cli_can_request_strict_pure_attention_graphs(self):
        args = build_parser().parse_args(
            [
                "--examples",
                "examples.jsonl",
                "--model",
                "model-a",
                "--output-dir",
                "output",
                "--pure-attention",
            ]
        )

        self.assertTrue(args.pure_attention)

    def test_extraction_cli_can_exclude_logit_node_features(self):
        args = build_parser().parse_args(
            [
                "--examples",
                "examples.jsonl",
                "--model",
                "model-a",
                "--output-dir",
                "output",
                "--exclude-logit-node-features",
            ]
        )

        self.assertTrue(args.exclude_logit_node_features)

    def test_final_composed_text_is_tokenized_once_and_segments_are_aligned(self):
        example = compose_example(
            "Passage evidence.",
            "Question?",
            "Answer.",
            example_id="sample",
        )
        tokenizer = _FakeTokenizer(example)

        encoded = tokenize_example_once(tokenizer, example, max_tokens=8)

        self.assertEqual(tokenizer.call_count, 1)
        self.assertFalse(tokenizer.last_options["truncation"])
        self.assertEqual(encoded["segment_ids"].tolist(), [0, 1, 2, 3])
        self.assertEqual(encoded["answer_mask"].tolist(), [False, False, False, True])

    def test_full_context_limit_raises_instead_of_silently_truncating(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        tokenizer = _FakeTokenizer(example)

        with self.assertRaisesRegex(ValueError, "exceeds max_tokens"):
            tokenize_example_once(tokenizer, example, max_tokens=3)

    def test_boolq_exact_generation_tokens_are_replayed_without_retokenizing(self):
        class MustNotRetokenize:
            call_count = 0

            def __call__(self, *args, **kwargs):
                self.call_count += 1
                raise AssertionError("exact generation replay must bypass tokenization")

        tokenizer = MustNotRetokenize()
        example = compose_example(
            "P",
            "Q",
            "Yes",
            example_id="boolq-sample",
            dataset="boolq",
            metadata={
                "replay_input_ids": [1, 11, 22, 33],
                "replay_attention_mask": [1, 1, 1, 1],
                "replay_offset_mapping": [[0, 0], [9, 10], [23, 24], [42, 42]],
                "replay_special_tokens_mask": [1, 0, 0, 0],
                "replay_segment_ids": [0, 1, 2, 3],
            },
        )

        encoded = tokenize_example_once(tokenizer, example, max_tokens=8)

        self.assertEqual(tokenizer.call_count, 0)
        self.assertEqual(encoded["input_ids"].tolist(), [1, 11, 22, 33])
        self.assertEqual(encoded["segment_ids"].tolist(), [0, 1, 2, 3])
        self.assertEqual(encoded["answer_mask"].tolist(), [False, False, False, True])

    def test_cache_fingerprint_changes_when_graph_edge_policy_changes(self):
        example = compose_example("P", "Q", "A", example_id="sample")

        with_prefix_edges = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
        )
        without_prefix_edges = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            False,
            False,
        )

        self.assertNotEqual(with_prefix_edges, without_prefix_edges)

        lower_limit = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            max_tokens=128,
        )
        higher_limit = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            max_tokens=256,
        )
        self.assertNotEqual(lower_limit, higher_limit)

    def test_cache_fingerprint_changes_when_logit_node_policy_changes(self):
        example = compose_example("P", "Q", "A", example_id="sample")

        with_logits = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            include_logit_node_features=True,
        )
        without_logits = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            include_logit_node_features=False,
        )

        self.assertNotEqual(with_logits, without_logits)

    def test_cache_fingerprint_changes_for_strict_pure_attention(self):
        example = compose_example("P", "Q", "A", example_id="sample")

        no_logits = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            include_logit_node_features=False,
            pure_attention=False,
        )
        pure_attention = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            include_logit_node_features=False,
            pure_attention=True,
        )

        self.assertNotEqual(no_logits, pure_attention)

    def test_legacy_no_logits_fingerprint_remains_resume_compatible(self):
        example = compose_example("P", "Q", "A", example_id="sample")

        fingerprint = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            include_logit_node_features=False,
            pure_attention=False,
        )

        self.assertEqual(
            fingerprint,
            "9f801de9ff8cfb738be488b8bae7aeacaea3f23ccb07f8398fee7bf3d4a8c35e",
        )

    def test_cache_fingerprint_changes_when_extraction_dtype_changes(self):
        example = compose_example("P", "Q", "A", example_id="sample")

        float16 = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            extraction_dtype="float16",
        )
        bfloat16 = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            extraction_dtype="bfloat16",
        )

        self.assertNotEqual(float16, bfloat16)

    def test_cache_fingerprint_changes_when_dense_attention_policy_changes(self):
        example = compose_example("P", "Q", "A", example_id="sample")

        retained = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            retain_dense_attention=True,
        )
        discarded = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            retain_dense_attention=False,
        )

        self.assertNotEqual(retained, discarded)

        cpu_postprocess = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            postprocess_device="cpu",
        )
        model_postprocess = _fingerprint(
            example,
            "model-a",
            (1,),
            0.05,
            True,
            False,
            postprocess_device="model",
        )
        self.assertNotEqual(cpu_postprocess, model_postprocess)

    def test_auto_postprocess_uses_model_device_only_with_memory_headroom(self):
        cuda = torch.device("cuda:0")

        enough = _resolve_postprocess_device(
            "auto",
            cuda,
            attention_bytes=100,
            cuda_free_bytes=411,
            reserve_bytes=10,
        )
        insufficient = _resolve_postprocess_device(
            "auto",
            cuda,
            attention_bytes=100,
            cuda_free_bytes=409,
            reserve_bytes=10,
        )

        self.assertEqual(enough, cuda)
        self.assertEqual(insufficient, torch.device("cpu"))

    def test_cache_fingerprint_includes_exact_generation_replay(self):
        first = compose_example(
            "P",
            "Q",
            "Yes",
            example_id="sample",
            metadata={"replay_input_ids": [1, 2, 3], "replay_segment_ids": [1, 2, 3]},
        )
        second = compose_example(
            "P",
            "Q",
            "Yes",
            example_id="sample",
            metadata={"replay_input_ids": [1, 2, 4], "replay_segment_ids": [1, 2, 3]},
        )

        first_fingerprint = _fingerprint(first, "model-a", (1,), 0.05, True, False)
        second_fingerprint = _fingerprint(second, "model-a", (1,), 0.05, True, False)

        self.assertNotEqual(first_fingerprint, second_fingerprint)

    def test_external_example_ids_cannot_escape_the_artifact_directory(self):
        artifact_directory = Path("safe-artifacts")
        artifact = _artifact_path(artifact_directory, "../../outside")

        self.assertEqual(artifact.parent, artifact_directory)
        self.assertNotIn("..", artifact.name)
        self.assertEqual(artifact.suffix, ".pt")

    def test_local_model_signature_changes_when_checkpoint_files_change(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            model_directory = Path(temporary_directory) / "model"
            model_directory.mkdir()
            checkpoint = model_directory / "model.safetensors"
            checkpoint.write_bytes(b"first")
            first = model_source_signature(model_directory)
            checkpoint.write_bytes(b"second-version")
            second = model_source_signature(model_directory)

        self.assertNotEqual(first, second)

    def test_attention_storage_estimate_exposes_4096_token_cost(self):
        estimated = _estimate_attention_storage_bytes(32, 32, 4096)

        self.assertEqual(estimated, 64 * 1024**3)


class TeacherForcedStatisticsTests(unittest.TestCase):
    def test_log_probability_and_entropy_are_aligned_to_the_predicted_token(self):
        logits = torch.tensor(
            [
                [
                    [2.0, 0.0, -1.0],
                    [0.0, 2.0, -1.0],
                    [-1.0, 0.0, 2.0],
                    [0.0, 0.0, 0.0],
                ]
            ]
        )
        input_ids = torch.tensor([[0, 0, 1, 2]])

        token_log_prob, entropy, valid = compute_teacher_forced_statistics(
            logits, input_ids
        )

        self.assertEqual(token_log_prob.shape, (4,))
        self.assertEqual(entropy.shape, (4,))
        self.assertEqual(valid.tolist(), [False, True, True, True])
        expected = torch.log_softmax(logits[0, :-1], dim=-1)
        self.assertAlmostEqual(float(token_log_prob[1]), float(expected[0, 0]))
        self.assertAlmostEqual(float(token_log_prob[2]), float(expected[1, 1]))
        self.assertAlmostEqual(float(token_log_prob[3]), float(expected[2, 2]))

    def test_complete_trace_contains_model_observables_but_no_evaluation_fields(self):
        example = compose_example("P", "Q", "A", example_id="sample", pair_id="pair")

        trace = extract_example_trace(
            _FakeModel(),
            _FakeTokenizer(example),
            example,
            selected_hidden_layers=(1,),
            max_tokens=8,
        )

        self.assertEqual(tuple(trace["attention"].shape), (1, 1, 4, 4))
        self.assertEqual(tuple(trace["hidden_states"].shape), (1, 4, 3))
        self.assertEqual(trace["segment_ids"].tolist(), [0, 1, 2, 3])
        self.assertTrue(set(trace).isdisjoint({"label", "labels", "y", "y_token"}))


class TraceSummaryTests(unittest.TestCase):
    def test_trace_summary_exports_scalar_evidence_and_self_reliance_features(self):
        attention = torch.zeros((1, 1, 4, 4))
        attention[0, 0, 3] = torch.tensor([0.0, 0.6, 0.2, 0.2])
        trace = {
            "example_id": "sample",
            "pair_id": "pair",
            "dataset": "halueval_qa",
            "attention": attention,
            "segment_ids": torch.tensor([0, 1, 2, 3]),
            "token_log_prob": torch.tensor([0.0, -1.0, -1.0, -0.1]),
            "token_stat_valid": torch.tensor([False, True, True, True]),
            "edge_threshold": 0.03,
        }
        attention[0, 0, 3, 0] = 0.04

        record = summarize_trace_record(trace)

        self.assertEqual(record["example_id"], "sample")
        self.assertAlmostEqual(record["answer_to_passage_mass"], 0.6)
        self.assertAlmostEqual(record["answer_to_question_mass"], 0.2)
        self.assertAlmostEqual(record["answer_to_passage_ratio"], 0.75)
        self.assertAlmostEqual(record["answer_to_question_ratio"], 0.25)
        self.assertAlmostEqual(record["answer_self_reliance"], 0.0)
        self.assertAlmostEqual(record["mean_answer_log_prob"], -0.1)
        self.assertGreater(record["answer_edge_density"], 0.0)


class ExtractionCacheIntegrityTests(unittest.TestCase):
    def test_extraction_writes_a_strict_pure_attention_graph_and_manifest(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            manifest = extract_prepared_dataset(
                _FakeModel(),
                _FakeTokenizer(example),
                [example],
                output_dir=output,
                model_id="fake-model",
                selected_hidden_layers=(1,),
                max_tokens=8,
                tau=0.05,
                include_prefix_edges=True,
                include_hidden_nodes=False,
                include_logit_node_features=False,
                pure_attention=True,
            )
            graph = torch.load(
                next((output / "graphs").glob("*.pt")),
                map_location="cpu",
                weights_only=True,
            )

        self.assertTrue(manifest["pure_attention"])
        self.assertEqual(list(graph["x_view_slices"]), ["attention_diagonal"])
        self.assertEqual(tuple(graph["edge_mark"].shape), (2, 0))
        self.assertTrue(graph["graph_config"]["pure_attention"])

    def test_pure_attention_rejects_a_legacy_no_logits_cache(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            options = {
                "output_dir": Path(temporary_directory) / "extraction",
                "model_id": "fake-model",
                "selected_hidden_layers": (1,),
                "max_tokens": 8,
                "tau": 0.05,
                "include_prefix_edges": True,
                "include_hidden_nodes": False,
                "include_logit_node_features": False,
            }
            extract_prepared_dataset(
                _FakeModel(),
                _FakeTokenizer(example),
                [example],
                pure_attention=False,
                **options,
            )

            with self.assertRaisesRegex(RuntimeError, "Stale extraction cache"):
                extract_prepared_dataset(
                    _FakeModel(),
                    _FakeTokenizer(example),
                    [example],
                    pure_attention=True,
                    **options,
                )

    def test_extraction_excludes_logit_views_and_records_policy(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            manifest = extract_prepared_dataset(
                _FakeModel(),
                _FakeTokenizer(example),
                [example],
                output_dir=output,
                model_id="fake-model",
                selected_hidden_layers=(1,),
                max_tokens=8,
                tau=0.05,
                include_prefix_edges=True,
                include_hidden_nodes=False,
                include_logit_node_features=False,
            )
            graph = torch.load(
                next((output / "graphs").glob("*.pt")),
                map_location="cpu",
                weights_only=True,
            )

        self.assertFalse(manifest["include_logit_node_features"])
        self.assertEqual(
            list(graph["x_view_slices"]),
            ["attention_diagonal", "segment_one_hot", "position"],
        )
        self.assertFalse(graph["graph_config"]["include_logit_node_features"])

    def test_changed_logit_node_policy_rejects_existing_cache_end_to_end(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            options = {
                "output_dir": output,
                "model_id": "fake-model",
                "selected_hidden_layers": (1,),
                "max_tokens": 8,
                "tau": 0.05,
                "include_prefix_edges": True,
                "include_hidden_nodes": False,
            }
            extract_prepared_dataset(
                _FakeModel(),
                _FakeTokenizer(example),
                [example],
                include_logit_node_features=True,
                **options,
            )

            with self.assertRaisesRegex(RuntimeError, "Stale extraction cache"):
                extract_prepared_dataset(
                    _FakeModel(),
                    _FakeTokenizer(example),
                    [example],
                    include_logit_node_features=False,
                    **options,
                )

    def test_no_logit_extraction_can_resume_its_own_cache(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            options = {
                "output_dir": output,
                "model_id": "fake-model",
                "selected_hidden_layers": (1,),
                "max_tokens": 8,
                "tau": 0.05,
                "include_prefix_edges": True,
                "include_hidden_nodes": False,
                "include_logit_node_features": False,
            }
            extract_prepared_dataset(
                _FakeModel(), _FakeTokenizer(example), [example], **options
            )
            resumed = extract_prepared_dataset(
                _FakeModel(), _FakeTokenizer(example), [example], **options
            )

        self.assertEqual(resumed["extracted_examples"], 0)
        self.assertEqual(resumed["reused_examples"], 1)

    def test_first_pass_and_resumed_pass_write_identical_scalar_features(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            options = {
                "output_dir": output,
                "model_id": "fake-model",
                "selected_hidden_layers": (1,),
                "max_tokens": 8,
                "tau": 0.05,
                "include_prefix_edges": True,
                "include_hidden_nodes": False,
                "extraction_dtype": "float16",
                "retain_dense_attention": False,
            }
            extract_prepared_dataset(
                _FakeModel(), _FakeTokenizer(example), [example], **options
            )
            first_features = (output / "features.jsonl").read_bytes()

            extract_prepared_dataset(
                _FakeModel(), _FakeTokenizer(example), [example], **options
            )
            resumed_features = (output / "features.jsonl").read_bytes()

        self.assertEqual(first_features, resumed_features)

    def test_changed_extraction_dtype_rejects_existing_cache_end_to_end(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            options = {
                "output_dir": output,
                "model_id": "fake-model",
                "selected_hidden_layers": (1,),
                "max_tokens": 8,
                "tau": 0.05,
                "include_prefix_edges": True,
                "include_hidden_nodes": False,
                "retain_dense_attention": False,
            }
            extract_prepared_dataset(
                _FakeModel(),
                _FakeTokenizer(example),
                [example],
                extraction_dtype="float16",
                **options,
            )

            with self.assertRaisesRegex(RuntimeError, "Stale extraction cache"):
                extract_prepared_dataset(
                    _FakeModel(),
                    _FakeTokenizer(example),
                    [example],
                    extraction_dtype="bfloat16",
                    **options,
                )

    def test_discarded_dense_attention_has_cached_features_and_portable_tensors(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            manifest = extract_prepared_dataset(
                _FakeModel(),
                _FakeTokenizer(example),
                [example],
                output_dir=output,
                model_id="fake-model",
                selected_hidden_layers=(1,),
                max_tokens=8,
                tau=0.05,
                include_prefix_edges=True,
                include_hidden_nodes=False,
                extraction_dtype="float16",
                retain_dense_attention=False,
            )
            trace = torch.load(
                next((output / "traces").glob("*.pt")),
                weights_only=True,
            )
            graph = torch.load(
                next((output / "graphs").glob("*.pt")),
                weights_only=True,
            )

        self.assertNotIn("attention", trace)
        self.assertEqual(trace["attention_shape"], [1, 1, 4, 4])
        self.assertEqual(trace["attention_storage"], "discarded_after_postprocessing")
        self.assertIn("feature_record", trace)
        self.assertEqual(trace["hidden_states"].dtype, torch.float16)
        self.assertEqual(manifest["postprocess_device_counts"], {"cpu": 1})
        self.assertEqual(manifest["extracted_examples"], 1)
        self.assertEqual(manifest["reused_examples"], 0)

        def tensor_leaves(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from tensor_leaves(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from tensor_leaves(child)

        self.assertTrue(all(value.device.type == "cpu" for value in tensor_leaves(trace)))
        self.assertTrue(all(value.device.type == "cpu" for value in tensor_leaves(graph)))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_gpu_postprocessing_still_persists_cpu_portable_artifacts(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            manifest = extract_prepared_dataset(
                _FakeModel().to("cuda:0"),
                _FakeTokenizer(example),
                [example],
                output_dir=output,
                model_id="fake-model",
                selected_hidden_layers=(1,),
                max_tokens=8,
                tau=0.05,
                include_prefix_edges=True,
                include_hidden_nodes=False,
                extraction_dtype="float16",
                postprocess_device="model",
                retain_dense_attention=False,
            )
            trace = torch.load(
                next((output / "traces").glob("*.pt")),
                weights_only=True,
            )
            graph = torch.load(
                next((output / "graphs").glob("*.pt")),
                weights_only=True,
            )

        self.assertEqual(trace["postprocess_device"], "cuda:0")
        self.assertEqual(manifest["postprocess_device_counts"], {"cuda:0": 1})

        def tensor_leaves(value):
            if isinstance(value, torch.Tensor):
                yield value
            elif isinstance(value, dict):
                for child in value.values():
                    yield from tensor_leaves(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    yield from tensor_leaves(child)

        for artifact in (trace, graph):
            self.assertTrue(
                all(value.device.type == "cpu" for value in tensor_leaves(artifact))
            )

    def test_cached_trace_and_graph_fingerprints_must_both_match(self):
        example = compose_example("P", "Q", "A", example_id="sample")
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "extraction"
            options = {
                "output_dir": output,
                "model_id": "fake-model",
                "selected_hidden_layers": (1,),
                "max_tokens": 8,
                "tau": 0.05,
                "include_prefix_edges": True,
                "include_hidden_nodes": False,
            }
            extract_prepared_dataset(
                _FakeModel(), _FakeTokenizer(example), [example], **options
            )
            graph_path = next((output / "graphs").glob("*.pt"))
            graph = torch.load(graph_path, map_location="cpu", weights_only=False)
            graph["extraction_fingerprint"] = "corrupted"
            torch.save(graph, graph_path)

            with self.assertRaisesRegex(RuntimeError, "graph cache"):
                extract_prepared_dataset(
                    _FakeModel(), _FakeTokenizer(example), [example], **options
                )


if __name__ == "__main__":
    unittest.main()
