from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class AttentionGraphCliContractTests(unittest.TestCase):
    def test_run_parser_exposes_experiment_and_smoke_controls(self):
        from attention_graph.cli import build_parser

        args = build_parser().parse_args(
            [
                "run",
                "--cache-root",
                "/cache",
                "--output-dir",
                "/output",
                "--device",
                "cuda:1",
                "--selection",
                "typed_topk",
                "--threshold",
                "0.02",
                "--top-k",
                "4",
                "--max-edges-per-target",
                "none",
                "--require-complete-cache",
                "--graph-transform",
                "source_shuffle",
                "--epochs",
                "3",
                "--patience",
                "2",
                "--limit",
                "7",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.cache_root, Path("/cache"))
        self.assertEqual(args.output_dir, Path("/output"))
        self.assertEqual(args.device, "cuda:1")
        self.assertEqual(args.selection, "typed_topk")
        self.assertEqual(args.threshold, 0.02)
        self.assertEqual(args.top_k, 4)
        self.assertIsNone(args.max_edges_per_target)
        self.assertTrue(args.require_complete_cache)
        self.assertEqual(args.graph_transform, "source_shuffle")
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.patience, 2)
        self.assertEqual(args.limit, 7)

    def test_run_parser_defaults_bound_formal_experiment_cost(self):
        from attention_graph.cli import build_parser

        args = build_parser().parse_args(
            ["run", "--cache-root", "/cache", "--output-dir", "/output"]
        )

        self.assertEqual(args.epochs, 30)
        self.assertEqual(args.patience, 6)
        self.assertEqual(args.num_score_views, 4)
        self.assertEqual(args.query_block, 32)
        self.assertEqual(args.max_edges_per_target, 64)
        self.assertEqual(args.max_support_edges, 8_192)
        self.assertEqual(args.max_weight_traces, 65_536)
        self.assertEqual(args.max_distribution_groups, 512)
        self.assertEqual(args.decoder_chunk_size, 16_384)
        self.assertEqual(args.graph_transform, "none")
        self.assertEqual(args.support_weight, 1.0)
        self.assertEqual(args.attention_weight, 1.0)
        self.assertEqual(args.distribution_weight, 1.0)
        self.assertEqual(args.node_weight, 0.25)
        self.assertFalse(args.embedding_only_scoring)

    def test_prepared_graph_dataset_loads_each_graph_lazily_with_mmap(self):
        from attention_graph.cli import PreparedGraphDataset

        records = [
            SimpleNamespace(graph_path=Path("first.graph.pt")),
            SimpleNamespace(graph_path=Path("second.graph.pt")),
        ]
        first_graph = SimpleNamespace(name="first")

        with mock.patch(
            "attention_graph.cli.load_graph", return_value=first_graph
        ) as loader:
            dataset = PreparedGraphDataset(records)
            self.assertEqual(len(dataset), 2)
            loader.assert_not_called()
            self.assertIs(dataset[0], first_graph)
            loader.assert_called_once_with(
                records[0].graph_path.resolve(),
                device="cpu",
                mmap=True,
                validate=True,
            )
            self.assertIs(dataset[0], first_graph)
            self.assertFalse(loader.call_args.kwargs["validate"])

    def test_smoke_limit_creates_exactly_n_inputs_per_official_split(self):
        from attention_graph.cli import create_smoke_cache

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "cache"
            for split in ("train", "test"):
                split_dir = cache / split
                split_dir.mkdir(parents=True)
                for index in range(3):
                    (split_dir / f"attention_{index:04d}.pt").write_bytes(
                        f"{split}-{index}".encode()
                    )

            smoke = create_smoke_cache(cache, root / "output", limit=2)

            self.assertEqual(
                sorted(path.name for path in (smoke / "train").glob("attention_*.pt")),
                ["attention_0000.pt", "attention_0001.pt"],
            )
            self.assertEqual(
                sorted(path.name for path in (smoke / "test").glob("attention_*.pt")),
                ["attention_0000.pt", "attention_0001.pt"],
            )

    def test_run_refuses_a_nonempty_output_directory(self):
        from attention_graph.cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / "evaluation.json").write_text("{}", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "run",
                    "--cache-root",
                    str(Path(directory) / "cache"),
                    "--output-dir",
                    str(output),
                ]
            )

            with self.assertRaisesRegex(FileExistsError, "new empty output"):
                run_pipeline(args)

    def test_complete_cache_requirement_fails_closed_but_partial_pilot_is_allowed(self):
        from attention_graph.cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = build_parser().parse_args(
                [
                    "run",
                    "--cache-root",
                    str(root / "cache"),
                    "--output-dir",
                    str(root / "output"),
                    "--require-complete-cache",
                ]
            )
            inventory = {
                "train": {"complete": False, "observed_files": 3},
                "test": {"complete": True, "observed_files": 2},
            }
            with mock.patch(
                "attention_graph.cli.audit_attention_cache", return_value=inventory
            ):
                with self.assertRaisesRegex(RuntimeError, "train"):
                    run_pipeline(args)

    def test_run_pipeline_freezes_response_and_token_scores_before_evaluation(self):
        from attention_graph.cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            args = build_parser().parse_args(
                [
                    "run",
                    "--cache-root",
                    str(root / "cache"),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--patience",
                    "1",
                    "--responses",
                    str(root / "response.jsonl"),
                    "--sources",
                    str(root / "source_info.jsonl"),
                    "--tokenizer",
                    str(root / "tokenizer"),
                ]
            )
            train_record = SimpleNamespace(
                graph_path=root / "train.graph.pt",
                cache_path=root / "train.pt",
                source_id="source-train",
                sample_id="sample-train",
                response_id="sample-train",
                dataset_split="train",
                num_nodes=6,
                num_response_nodes=3,
                num_edges=5,
                num_rp_edges=3,
                num_rr_edges=2,
                num_traces=12,
            )
            validation_record = SimpleNamespace(
                graph_path=root / "validation.graph.pt",
                cache_path=root / "validation.pt",
                source_id="source-validation",
                sample_id="sample-validation",
                response_id="sample-validation",
                dataset_split="train",
                num_nodes=6,
                num_response_nodes=3,
                num_edges=5,
                num_rp_edges=3,
                num_rr_edges=2,
                num_traces=12,
            )
            test_record = SimpleNamespace(
                graph_path=root / "test.graph.pt",
                cache_path=root / "test.pt",
                source_id="source-test",
                sample_id="sample-test",
                response_id="sample-test",
                dataset_split="test",
                num_nodes=6,
                num_response_nodes=3,
                num_edges=5,
                num_rp_edges=3,
                num_rr_edges=2,
                num_traces=12,
            )
            graph = SimpleNamespace(num_layers=2, num_heads=3)
            model = mock.Mock()
            model.to.return_value = model
            response_records = [
                {
                    "source_id": "source-test",
                    "sample_id": "sample-test",
                    "response_id": "sample-test",
                    "hallucination_probability": 0.75,
                }
            ]
            token_records = [
                {
                    "source_id": "source-test",
                    "sample_id": "sample-test",
                    "response_id": "sample-test",
                    "token_idx": 4,
                    "score": 0.80,
                }
            ]
            response_mixture = mock.Mock()
            response_mixture.to_dict.return_value = {"schema": "response-mixture"}
            token_mixture = mock.Mock()
            token_mixture.to_dict.return_value = {"schema": "token-mixture"}
            metrics = {
                "artifact_type": "attention_graph_posthoc_evaluation_v1",
                "labels_read_during": "evaluation_only",
            }
            sentence_metrics = {"pooling": "mean", "metrics": {"auroc": 0.7}}
            cache_inventory = {
                "train": {"complete": False, "observed_files": 2},
                "test": {"complete": False, "observed_files": 1},
            }
            labels = object()
            sentence_records = [{"sentence_idx": 0, "score": 0.8}]
            initialization_events: list[str] = []

            with (
                mock.patch(
                    "attention_graph.cli.audit_attention_cache",
                    return_value=cache_inventory,
                ),
                mock.patch(
                    "attention_graph.cli.prepare_graphs",
                    return_value=[train_record, validation_record, test_record],
                ),
                mock.patch(
                    "attention_graph.cli.official_partitions",
                    return_value={
                        "train": [train_record],
                        "validation": [validation_record],
                        "test": [test_record],
                    },
                ),
                mock.patch("attention_graph.cli.load_graph", return_value=graph),
                mock.patch(
                    "attention_graph.cli.torch.manual_seed",
                    side_effect=lambda seed: initialization_events.append(f"seed:{seed}"),
                ),
                mock.patch(
                    "attention_graph.cli.RelationAwareMaskGAE",
                    side_effect=lambda **_kwargs: (
                        initialization_events.append("model") or model
                    ),
                ),
                mock.patch(
                    "attention_graph.cli.train_relation_mae",
                    return_value=SimpleNamespace(
                        history=[{"epoch": 1, "train_total": 0.2}],
                        best_epoch=1,
                        best_validation_loss=0.1,
                        checkpoint_path=output / "training" / "encoder.pt",
                    ),
                ),
                mock.patch(
                    "attention_graph.cli.score_graphs",
                    return_value=(response_records, response_mixture),
                ) as score_responses,
                mock.patch(
                    "attention_graph.cli.score_tokens",
                    return_value=(token_records, token_mixture),
                ) as score_token_nodes,
                mock.patch(
                    "attention_graph.cli.load_evaluation_labels", return_value=labels
                ) as load_labels,
                mock.patch(
                    "attention_graph.cli.evaluate_predictions", return_value=metrics
                ) as evaluate,
                mock.patch(
                    "attention_graph.cli.evaluate_sentence_predictions",
                    return_value=sentence_metrics,
                ) as evaluate_sentences,
                mock.patch(
                    "attention_graph.cli._load_tokenizer",
                    return_value=object(),
                ),
                mock.patch(
                    "attention_graph.cli.prepare_ragtruth_sentence_scores",
                    return_value=sentence_records,
                ) as prepare_sentences,
            ):
                result = run_pipeline(args)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(initialization_events, ["seed:42", "model"])
            for scorer in (score_responses, score_token_nodes):
                self.assertEqual(scorer.call_args.kwargs["max_support_edges"], 8_192)
                self.assertEqual(scorer.call_args.kwargs["max_weight_traces"], 65_536)
                self.assertEqual(
                    scorer.call_args.kwargs["max_distribution_groups"], 512
                )
                self.assertEqual(scorer.call_args.kwargs["decoder_chunk_size"], 16_384)
            self.assertEqual(
                json.loads((output / "evaluation.json").read_text()),
                {**metrics, "sentence": sentence_metrics},
            )
            self.assertEqual(
                json.loads((output / "response_mixture.json").read_text())["schema"],
                "response-mixture",
            )
            response_line = json.loads(
                (output / "test.response_predictions.jsonl").read_text().strip()
            )
            token_line = json.loads(
                (output / "test.token_predictions.jsonl").read_text().strip()
            )
            self.assertEqual(response_line["sample_id"], "sample-test")
            self.assertEqual(token_line["token_idx"], 4)
            self.assertEqual(evaluate.call_args.args[0], response_records)
            self.assertEqual(evaluate.call_args.args[1], token_records)
            self.assertIs(evaluate.call_args.args[2], labels)
            load_labels.assert_called_once_with([test_record.cache_path])
            evaluate_sentences.assert_called_once_with(sentence_records, labels)
            self.assertEqual(
                json.loads((output / "training" / "history.json").read_text()),
                [{"epoch": 1, "train_total": 0.2}],
            )
            splits = json.loads((output / "splits.json").read_text())
            self.assertEqual(splits["counts"], {"test": 1, "train": 1, "validation": 1})
            self.assertEqual(
                prepare_sentences.call_args.kwargs["output_path"],
                output / "test.sentence_predictions.jsonl",
            )
            self.assertEqual(
                result["sentence_predictions"],
                str(output / "test.sentence_predictions.jsonl"),
            )
            self.assertEqual(result["experiment_scope"], "partial_cache_pilot")

    def test_skip_evaluation_is_restricted_to_explicit_smoke_runs(self):
        from attention_graph.cli import build_parser, run_pipeline

        args = build_parser().parse_args(
            [
                "run",
                "--cache-root",
                "/cache",
                "--output-dir",
                "/output",
                "--skip-evaluation",
            ]
        )

        with self.assertRaisesRegex(ValueError, "smoke"):
            run_pipeline(args)

    def test_smoke_run_can_freeze_scores_without_reading_evaluation_labels(self):
        from attention_graph.cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            args = build_parser().parse_args(
                [
                    "run",
                    "--cache-root",
                    str(root / "cache"),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--limit",
                    "3",
                    "--skip-evaluation",
                ]
            )
            records = [
                SimpleNamespace(
                    graph_path=root / f"{name}.graph.pt",
                    cache_path=root / f"{name}.pt",
                    source_id=f"source-{name}",
                    sample_id=f"sample-{name}",
                    response_id=f"sample-{name}",
                    dataset_split="test" if name == "test" else "train",
                    num_nodes=6,
                    num_response_nodes=3,
                    num_edges=5,
                    num_rp_edges=3,
                    num_rr_edges=2,
                    num_traces=12,
                )
                for name in ("train", "validation", "test")
            ]
            graph = SimpleNamespace(num_layers=2, num_heads=2)
            model = mock.Mock()
            model.to.return_value = model
            mixture = mock.Mock()
            mixture.to_dict.return_value = {"schema": "mixture"}

            with (
                mock.patch(
                    "attention_graph.cli.audit_attention_cache",
                    return_value={
                        "train": {"complete": False, "observed_files": 3},
                        "test": {"complete": False, "observed_files": 3},
                    },
                ),
                mock.patch(
                    "attention_graph.cli.create_smoke_cache",
                    return_value=root / "smoke-cache",
                ),
                mock.patch("attention_graph.cli.prepare_graphs", return_value=records),
                mock.patch(
                    "attention_graph.cli.official_partitions",
                    return_value={
                        "train": [records[0]],
                        "validation": [records[1]],
                        "test": [records[2]],
                    },
                ),
                mock.patch("attention_graph.cli.load_graph", return_value=graph),
                mock.patch(
                    "attention_graph.cli.RelationAwareMaskGAE", return_value=model
                ),
                mock.patch(
                    "attention_graph.cli.train_relation_mae",
                    return_value=SimpleNamespace(
                        history=[],
                        best_epoch=1,
                        best_validation_loss=0.1,
                        checkpoint_path=output / "training" / "encoder.pt",
                    ),
                ),
                mock.patch(
                    "attention_graph.cli.score_graphs",
                    return_value=([{"sample_id": "sample-test"}], mixture),
                ),
                mock.patch(
                    "attention_graph.cli.score_tokens",
                    return_value=([{"sample_id": "sample-test"}], mixture),
                ),
                mock.patch("attention_graph.cli.load_evaluation_labels") as evaluate,
            ):
                result = run_pipeline(args)

            evaluate.assert_not_called()
            self.assertEqual(result["labels_read_during"], "never")
            self.assertIsNone(result["evaluation"])
            self.assertFalse((output / "evaluation.json").exists())

    def test_sentence_arguments_must_be_supplied_as_a_complete_set(self):
        from attention_graph.cli import build_parser, run_pipeline

        args = build_parser().parse_args(
            [
                "run",
                "--cache-root",
                "/cache",
                "--output-dir",
                "/output",
                "--responses",
                "/response.jsonl",
            ]
        )

        with self.assertRaisesRegex(ValueError, "responses.*sources.*tokenizer"):
            run_pipeline(args)

    def test_root_entrypoints_are_one_command_and_test_only_recovery(self):
        root = Path(__file__).resolve().parents[1]
        main_source = (root / "main.py").read_text()
        shell_source = (root / "run_ragtruth_attention_graph.sh").read_text()

        self.assertIn("attention_graph.cli", main_source)
        self.assertIn("scripts/data/prepare_ragtruth_test_attention.sh", shell_source)
        self.assertIn('"${ATTENTION_CACHE_ROOT}/test"', shell_source)
        self.assertNotIn("run_ragtruth_extract_validate.sh", shell_source)
        self.assertIn("main.py\" run", shell_source)
        self.assertIn('--max-edges-per-target "${MAX_EDGES_PER_TARGET}"', shell_source)
        self.assertIn('--responses "${RESPONSES}"', shell_source)
        self.assertIn('--tokenizer "${TOKENIZER}"', shell_source)


if __name__ == "__main__":
    unittest.main()
