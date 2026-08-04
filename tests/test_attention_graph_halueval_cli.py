"""RED contracts for the label-free HaluEval attention-graph runner.

The runner intentionally has a separate CLI from ``attention_graph.cli``:
HaluEval's two response candidates must be split as an inseparable pair and
its labels may be opened only after held-out response scores are immutable.
"""

from __future__ import annotations

import json
import argparse
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _records(root: Path, *, status: str = "full") -> list[SimpleNamespace]:
    """Four complete pairs are enough for deterministic train/val/test splits."""

    return [
        SimpleNamespace(
            response_id=f"response-{pair}-{candidate}",
            pair_id=f"pair-{pair}",
            graph_path=root / "graphs" / f"response-{pair}-{candidate}.pt",
            trace_path=(root / "traces" / f"response-{pair}-{candidate}.pt")
            if status == "full"
            else None,
            artifact_status=status,
            response_length=8 + candidate,
            num_response_nodes=12 + candidate,
            num_nodes=30 + candidate,
        )
        for pair in range(4)
        for candidate in range(2)
    ]


def _write_examples(path: Path, records: list[SimpleNamespace]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "response_id": record.response_id,
                    "pair_id": record.pair_id,
                    "prompt": "label-free prompt",
                    "response": "label-free response",
                }
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _write_labels(path: Path, records: list[SimpleNamespace]) -> None:
    path.write_text(
        "".join(
            json.dumps(
                {
                    "response_id": record.response_id,
                    "label": int(record.response_id.endswith("-1")),
                }
            )
            + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def _fake_graph_loader(records: list[SimpleNamespace]):
    by_path = {record.graph_path.resolve(): record for record in records}

    def load_graph(path: str | Path, **_kwargs):
        record = by_path[Path(path).resolve()]
        return SimpleNamespace(
            source_id=record.pair_id,
            response_id=record.response_id,
            num_layers=2,
            num_heads=2,
        )

    return load_graph


class HaluEvalAttentionGraphCliParserTests(unittest.TestCase):
    def test_protocol_summary_exposes_the_effective_legacy_attention_floor(self):
        from attention_graph.halueval_cli import _legacy_protocol_summary

        summary = _legacy_protocol_summary(
            [SimpleNamespace(legacy_tau=0.05), SimpleNamespace(legacy_tau=0.05)],
            selection="threshold",
            threshold=None,
        )

        self.assertEqual(summary["event"], "input_protocol")
        self.assertEqual(summary["mode"], "legacy_tau_censored")
        self.assertEqual(summary["legacy_attention_floor_values"], [0.05])
        self.assertEqual(summary["effective_threshold"], "cache_floor")
        self.assertFalse(summary["supports_floor_0_01"])

    def test_protocol_summary_marks_threshold_inapplicable_for_top_k_selection(self):
        from attention_graph.halueval_cli import _legacy_protocol_summary

        summary = _legacy_protocol_summary(
            [SimpleNamespace(legacy_tau=0.05)],
            selection="typed_topk",
            threshold=0.10,
        )

        self.assertIsNone(summary["effective_threshold"])

    def test_pipeline_progress_is_staged_throttled_and_reports_start_and_finish(self):
        from attention_graph.halueval_cli import _print_pipeline_progress

        with mock.patch("builtins.print") as printed:
            for current in (1, 2, 25, 49, 50):
                _print_pipeline_progress("legacy_discovery", current, 50)

        payloads = [json.loads(call.args[0]) for call in printed.call_args_list]
        self.assertEqual(
            payloads,
            [
                {
                    "event": "progress",
                    "stage": "legacy_discovery",
                    "current": 1,
                    "total": 50,
                    "percent": 2.0,
                },
                {
                    "event": "progress",
                    "stage": "legacy_discovery",
                    "current": 25,
                    "total": 50,
                    "percent": 50.0,
                },
                {
                    "event": "progress",
                    "stage": "legacy_discovery",
                    "current": 50,
                    "total": 50,
                    "percent": 100.0,
                },
            ],
        )

    def test_run_parser_exposes_label_free_input_and_experiment_controls(self):
        from attention_graph.halueval_cli import build_parser

        args = build_parser().parse_args(
            [
                "run",
                "--extraction-dir",
                "/extraction",
                "--examples",
                "/examples.jsonl",
                "--evaluation-labels",
                "/labels.jsonl",
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
                "--epochs",
                "3",
                "--patience",
                "2",
                "--validation-fraction",
                "0.25",
                "--test-fraction",
                "0.25",
                "--seed",
                "9",
                "--limit-pairs",
                "2",
                "--group-by-prompt",
                "--require-complete-cache",
                "--skip-evaluation",
            ]
        )

        self.assertEqual(args.command, "run")
        self.assertEqual(args.extraction_dir, Path("/extraction"))
        self.assertEqual(args.examples, Path("/examples.jsonl"))
        self.assertEqual(args.evaluation_labels, Path("/labels.jsonl"))
        self.assertEqual(args.output_dir, Path("/output"))
        self.assertEqual(args.device, "cuda:1")
        self.assertEqual(args.selection, "typed_topk")
        self.assertEqual(args.threshold, 0.02)
        self.assertEqual(args.top_k, 4)
        self.assertIsNone(args.max_edges_per_target)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.patience, 2)
        self.assertEqual(args.validation_fraction, 0.25)
        self.assertEqual(args.test_fraction, 0.25)
        self.assertEqual(args.seed, 9)
        self.assertEqual(args.limit_pairs, 2)
        self.assertTrue(args.group_by_prompt)
        self.assertTrue(args.require_complete_cache)
        self.assertTrue(args.skip_evaluation)

    def test_run_parser_defaults_to_a_twenty_percent_held_out_pair_partition(self):
        from attention_graph.halueval_cli import build_parser

        args = build_parser().parse_args(
            [
                "run", "--extraction-dir", "/extraction", "--examples", "/examples.jsonl",
                "--evaluation-labels", "/labels.jsonl", "--output-dir", "/output",
            ]
        )

        self.assertEqual(args.test_fraction, 0.20)

    def test_run_parser_names_preparation_only_resume_and_conversion_chunk_size(self):
        from attention_graph.halueval_cli import build_parser

        args = build_parser().parse_args(
            [
                "run", "--extraction-dir", "/extraction", "--examples", "/examples.jsonl",
                "--evaluation-labels", "/labels.jsonl", "--output-dir", "/output",
                "--no-prepare-resume", "--conversion-chunk-edges", "1234",
            ]
        )

        self.assertTrue(args.no_prepare_resume)
        self.assertEqual(args.conversion_chunk_edges, 1234)
        parser = build_parser()
        run_parser = next(
            action.choices["run"]
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        help_text = run_parser.format_help()
        self.assertIn("--no-prepare-resume", help_text)
        self.assertNotIn("--no-resume", help_text)

    def test_main_prints_average_precision_as_auprc(self):
        from attention_graph import halueval_cli

        with mock.patch.object(
            halueval_cli, "run_pipeline",
            return_value={
                "output_dir": "/output",
                "evaluation": None,
                "core_metrics": {
                    "auroc": 0.7,
                    "average_precision": 0.6,
                    "paired_accuracy": 0.5,
                    "positive_fraction": 0.5,
                },
            },
        ), mock.patch("builtins.print") as printed:
            halueval_cli.main(
                ["run", "--extraction-dir", "/extraction", "--examples", "/examples.jsonl",
                 "--evaluation-labels", "/labels.jsonl", "--output-dir", "/output"]
            )

        self.assertIn("AUPRC=0.6", printed.call_args.args[0])
        self.assertNotIn("AUPRC=n/a", printed.call_args.args[0])


class HaluEvalAttentionGraphCliInputTests(unittest.TestCase):
    def test_run_provenance_hashes_label_free_examples_and_uses_selected_records_only(self):
        from attention_graph.halueval_cli import _run_provenance

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            examples.write_text('{"response_id":"selected","pair_id":"pair","prompt":"label-free"}\n', encoding="utf-8")
            labels.write_text('{"response_id":"selected","label":1}\n', encoding="utf-8")
            import hashlib
            expected_examples_sha256 = hashlib.sha256(examples.read_bytes()).hexdigest()
            selected = [SimpleNamespace(legacy_tau=0.1, extractor_model_id="selected-model", extraction_fingerprint="selected-fp")]
            provenance = _run_provenance(
                SimpleNamespace(extraction_dir=root / "extraction", source_run=None, examples=examples, evaluation_labels=labels),
                records=selected,
            )

        self.assertEqual(provenance["examples_sha256"], expected_examples_sha256)
        self.assertEqual(provenance["legacy_tau_values"], [0.1])
        self.assertEqual(provenance["extractor_model_ids"], ["selected-model"])
        self.assertEqual(provenance["extraction_fingerprints"]["count"], 1)
        self.assertNotIn("evaluation_labels_sha256", provenance)

    def test_examples_derive_a_stable_prompt_group_without_reading_response_length(self):
        from attention_graph.halueval_cli import _read_examples

        with tempfile.TemporaryDirectory() as directory:
            examples = Path(directory) / "examples.jsonl"
            examples.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (
                        {"response_id": "answer", "pair_id": "one", "prompt": "same", "answer": "one two three"},
                        {"response_id": "response", "pair_id": "two", "prompt": "same", "response": "one two"},
                    )
                ) + "\n",
                encoding="utf-8",
            )

            pairs, groups = _read_examples(examples)

        self.assertEqual(pairs, {"answer": "one", "response": "two"})
        self.assertEqual(groups["answer"], groups["response"])

    def test_pipeline_rejects_manifest_pair_identity_that_disagrees_with_examples(self):
        from attention_graph.halueval_cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _records(root)
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            _write_examples(examples, records)
            rows = [json.loads(line) for line in examples.read_text(encoding="utf-8").splitlines()]
            rows[0]["pair_id"] = "wrong-pair"
            examples.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            _write_labels(labels, records)
            args = build_parser().parse_args(
                ["run", "--extraction-dir", str(root / "extraction"), "--examples", str(examples),
                 "--evaluation-labels", str(labels), "--output-dir", str(root / "output")]
            )

            with mock.patch("attention_graph.halueval_cli.discover_legacy_halueval_records", return_value=records):
                with self.assertRaisesRegex(ValueError, "pair_id|pair identity|pair"):
                    run_pipeline(args)

    def test_pipeline_rejects_missing_trace_before_preparation(self):
        from attention_graph.halueval_cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _records(root)
            records[0].trace_path = None
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            _write_examples(examples, records)
            _write_labels(labels, records)
            args = build_parser().parse_args(
                ["run", "--extraction-dir", str(root / "extraction"), "--examples", str(examples),
                 "--evaluation-labels", str(labels), "--output-dir", str(root / "output")]
            )

            with mock.patch("attention_graph.halueval_cli.discover_legacy_halueval_records", return_value=records), mock.patch(
                "attention_graph.halueval_cli.prepare_legacy_halueval_graphs"
            ) as prepare:
                with self.assertRaisesRegex(ValueError, "graph and trace|complete graph and trace"):
                    run_pipeline(args)

            prepare.assert_not_called()

    def test_pipeline_rejects_zero_held_out_fraction_before_preparation(self):
        from attention_graph.halueval_cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _records(root)
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            _write_examples(examples, records)
            _write_labels(labels, records)
            args = build_parser().parse_args(
                ["run", "--extraction-dir", str(root / "extraction"), "--examples", str(examples),
                 "--evaluation-labels", str(labels), "--output-dir", str(root / "output"),
                 "--validation-fraction", "0"]
            )
            with mock.patch("attention_graph.halueval_cli.discover_legacy_halueval_records", return_value=records), mock.patch(
                "attention_graph.halueval_cli.prepare_legacy_halueval_graphs"
            ) as prepare:
                with self.assertRaisesRegex(ValueError, "validation|test|fraction"):
                    run_pipeline(args)

            prepare.assert_not_called()

    def test_pipeline_checks_empty_partitions_before_preparation(self):
        from attention_graph.halueval_cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _records(root)
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            _write_examples(examples, records)
            _write_labels(labels, records)
            args = build_parser().parse_args(
                ["run", "--extraction-dir", str(root / "extraction"), "--examples", str(examples),
                 "--evaluation-labels", str(labels), "--output-dir", str(root / "output")]
            )
            with mock.patch("attention_graph.halueval_cli.discover_legacy_halueval_records", return_value=records), mock.patch(
                "attention_graph.halueval_cli.split_halueval_pairs",
                return_value={"train": records[:4], "validation": [], "test": records[4:]},
            ), mock.patch("attention_graph.halueval_cli.prepare_legacy_halueval_graphs") as prepare:
                with self.assertRaisesRegex(ValueError, "empty train, validation, or test"):
                    run_pipeline(args)

            prepare.assert_not_called()

    def test_limited_pilot_skips_incomplete_pairs_when_complete_pairs_are_available(self):
        from attention_graph.halueval_cli import _limit_complete_pairs

        records = _records(Path("/legacy"))
        for record in records[:2]:
            record.trace_path = None

        selected = _limit_complete_pairs(records, limit=3)

        self.assertEqual(
            {record.pair_id for record in selected}, {"pair-1", "pair-2", "pair-3"}
        )
        self.assertTrue(all(record.trace_path is not None for record in selected))

    def test_fresh_output_allows_only_reusable_prepared_artifacts_and_never_final_results(self):
        from attention_graph.halueval_cli import _fresh_output

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            reusable = output / "prepared" / "graphs" / "train" / ("attention_" + "a" * 64 + ".graph.pt")
            reusable.parent.mkdir(parents=True)
            reusable.write_bytes(b"prepared")
            _fresh_output(output, resume=True)

            (output / "run.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "final|output"):
                _fresh_output(output, resume=True)


class HaluEvalAttentionGraphPipelineTests(unittest.TestCase):
    def test_pipeline_keeps_pairs_disjoint_and_reads_labels_only_after_predictions_are_written(self):
        from attention_graph.halueval_cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            extraction, output = root / "extraction", root / "output"
            extraction.mkdir()
            records = _records(extraction)
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            _write_examples(examples, records)
            _write_labels(labels, records)
            args = build_parser().parse_args(
                [
                    "run",
                    "--extraction-dir",
                    str(extraction),
                    "--examples",
                    str(examples),
                    "--evaluation-labels",
                    str(labels),
                    "--output-dir",
                    str(output),
                    "--device",
                    "cpu",
                    "--epochs",
                    "1",
                    "--patience",
                    "1",
                    "--validation-fraction",
                    "0.25",
                    "--test-fraction",
                    "0.25",
                    "--seed",
                    "11",
                ]
            )
            model = mock.Mock()
            model.to.return_value = model
            mixture = mock.Mock()
            mixture.to_dict.return_value = {"schema": "label-free-mixture"}
            events: list[str] = []

            def score_graphs(*_args, **kwargs):
                # The fit and held-out graph collections must be pair-disjoint.
                fit_pairs = {graph.source_id for graph in kwargs["fit_graphs"]}
                test_pairs = {graph.source_id for graph in kwargs["score_graphs"]}
                self.assertFalse(fit_pairs & test_pairs)
                events.append("score")
                return (
                    [
                        {
                            "response_id": graph.response_id,
                            "pair_id": graph.source_id,
                            "score": 0.8,
                        }
                        for graph in kwargs["score_graphs"]
                    ],
                    mixture,
                )

            def load_labels(path: Path):
                # The evaluation sidecar is deliberately inaccessible until scores
                # have been persisted.  This catches label access during discovery,
                # conversion, split, training, and scoring.
                prediction_path = output / "test.response_predictions.jsonl"
                self.assertTrue(prediction_path.is_file())
                self.assertTrue(prediction_path.read_text(encoding="utf-8").strip())
                events.append("labels")
                self.assertEqual(path, labels)
                return {
                    record.response_id: int(record.response_id.endswith("-1"))
                    for record in records
                }

            with (
                mock.patch(
                    "attention_graph.halueval_cli.discover_legacy_halueval_records",
                    return_value=records,
                ),
                mock.patch(
                    "attention_graph.halueval_cli.prepare_legacy_halueval_graphs",
                    side_effect=lambda records, **_kwargs: records,
                ) as prepare,
                mock.patch(
                    "attention_graph.halueval_cli.load_graph",
                    side_effect=_fake_graph_loader(records),
                ),
                mock.patch(
                    "attention_graph.halueval_cli.RelationAwareMaskGAE",
                    return_value=model,
                ),
                mock.patch(
                    "attention_graph.halueval_cli.train_relation_mae",
                    return_value=SimpleNamespace(
                        history=[{"epoch": 1, "train_total": 0.1}],
                        best_epoch=1,
                        best_validation_loss=0.1,
                        checkpoint_path=output / "training" / "encoder.pt",
                    ),
                ) as train,
                mock.patch(
                    "attention_graph.halueval_cli.score_graphs",
                    side_effect=score_graphs,
                ),
                mock.patch(
                    "attention_graph.halueval_cli.load_halueval_response_labels",
                    side_effect=load_labels,
                ),
                mock.patch(
                    "attention_graph.halueval_cli.evaluate_halueval_predictions",
                    return_value={"auroc": 1.0, "average_precision": 1.0, "paired_accuracy": 1.0, "positive_fraction": 0.5},
                ) as evaluate,
            ):
                result = run_pipeline(args)

            self.assertEqual(events, ["score", "labels"])
            self.assertEqual(train.call_count, 1)
            self.assertEqual(prepare.call_count, 1)
            self.assertEqual(prepare.call_args.kwargs["conversion_device"], args.device)
            self.assertEqual(prepare.call_args.kwargs["build_device"], args.device)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["labels_read_during"], "evaluation_only")
            self.assertEqual(result["experiment_scope"], "legacy_cache_complete")
            self.assertTrue((output / "run.json").is_file())
            self.assertTrue((output / "splits.json").is_file())
            self.assertTrue((output / "test.response_predictions.jsonl").is_file())
            self.assertTrue((output / "evaluation.json").is_file())
            run_manifest = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(
                run_manifest["core_metrics"],
                {"auroc": 1.0, "average_precision": 1.0, "paired_accuracy": 1.0, "positive_fraction": 0.5},
            )
            self.assertEqual(run_manifest["configuration"]["preparation"]["conversion_chunk_edges"], 8192)

            split_manifest = json.loads((output / "splits.json").read_text(encoding="utf-8"))
            self.assertEqual(split_manifest["mode"], "compatibility_pair_hash")
            pair_sets = {
                name: set(split_manifest["pair_ids"][name])
                for name in ("train", "validation", "test")
            }
            self.assertFalse(pair_sets["train"] & pair_sets["validation"])
            self.assertFalse(pair_sets["train"] & pair_sets["test"])
            self.assertFalse(pair_sets["validation"] & pair_sets["test"])
            frozen_predictions = [
                json.loads(line)
                for line in (output / "test.response_predictions.jsonl").read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            # HaluEval is response-level only: evaluation receives the one frozen
            # response score stream, not token labels or token predictions.
            self.assertEqual(evaluate.call_args.args[0], frozen_predictions)
            self.assertEqual(
                set(evaluate.call_args.args[1]),
                {record["response_id"] for record in frozen_predictions},
            )
            self.assertEqual(
                evaluate.call_args_list[0].kwargs["response_length_by_id"],
                {
                    record.response_id: record.num_response_nodes
                    for record in records
                    if record.response_id in {
                        item["response_id"] for item in frozen_predictions
                    }
                },
            )

    def test_partial_legacy_extraction_is_marked_as_a_pilot_and_full_run_requires_manifest_ids_to_cover_examples(self):
        from attention_graph.halueval_cli import build_parser, run_pipeline

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            partial_records = _records(root, status="partial")
            # A non-complete extraction manifest is a provenance downgrade, not
            # a data-loss condition, when both conversion inputs are present.
            for record in partial_records:
                record.trace_path = root / "traces" / f"{record.response_id}.pt"
            _write_examples(examples, partial_records)
            _write_labels(labels, partial_records)
            args = build_parser().parse_args(
                [
                    "run",
                    "--extraction-dir",
                    str(root / "extraction"),
                    "--examples",
                    str(examples),
                    "--evaluation-labels",
                    str(labels),
                    "--output-dir",
                    str(root / "output"),
                    "--limit-pairs",
                    "3",
                    "--validation-fraction",
                    "0.3333333333333333",
                    "--test-fraction",
                    "0.3333333333333333",
                    "--skip-evaluation",
                ]
            )

            # A limited run is explicitly a pilot even when every selected
            # record has the graph-and-trace inputs needed for conversion.
            model = mock.Mock()
            model.to.return_value = model
            mixture = mock.Mock()
            mixture.to_dict.return_value = {"schema": "label-free-mixture"}
            with (
                mock.patch(
                    "attention_graph.halueval_cli.discover_legacy_halueval_records",
                    return_value=partial_records,
                ),
                mock.patch(
                    "attention_graph.halueval_cli.prepare_legacy_halueval_graphs",
                    side_effect=lambda records, **_kwargs: records,
                ),
                mock.patch(
                    "attention_graph.halueval_cli.load_graph",
                    side_effect=_fake_graph_loader(partial_records),
                ),
                mock.patch(
                    "attention_graph.halueval_cli.RelationAwareMaskGAE",
                    return_value=model,
                ),
                mock.patch(
                    "attention_graph.halueval_cli.train_relation_mae",
                    return_value=SimpleNamespace(
                        history=[],
                        best_epoch=1,
                        best_validation_loss=0.1,
                        checkpoint_path=root / "output" / "training" / "encoder.pt",
                    ),
                ),
                mock.patch(
                    "attention_graph.halueval_cli.score_graphs",
                    side_effect=lambda *_args, **kwargs: (
                        [
                            {
                                "response_id": graph.response_id,
                                "pair_id": graph.source_id,
                                "score": 0.2,
                            }
                            for graph in kwargs["score_graphs"]
                        ],
                        mixture,
                    ),
                ),
                mock.patch(
                    "attention_graph.halueval_cli.load_halueval_response_labels"
                ) as load_labels,
            ):
                result = run_pipeline(args)

            self.assertEqual(result["experiment_scope"], "legacy_cache_partial_pilot")
            self.assertEqual(result["labels_read_during"], "never")
            load_labels.assert_not_called()
            self.assertTrue((root / "output" / "run.json").is_file())

        # The full/official label is guarded by exact manifest/example coverage:
        # a manifest that omits one label-free candidate can never be reported full.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = _records(root)
            examples, labels = root / "examples.jsonl", root / "labels.jsonl"
            _write_examples(examples, records)
            _write_labels(labels, records)
            args = build_parser().parse_args(
                [
                    "run",
                    "--extraction-dir",
                    str(root / "extraction"),
                    "--examples",
                    str(examples),
                    "--evaluation-labels",
                    str(labels),
                    "--output-dir",
                    str(root / "output"),
                    "--require-complete-cache",
                    "--skip-evaluation",
                ]
            )
            with mock.patch(
                "attention_graph.halueval_cli.discover_legacy_halueval_records",
                return_value=records[:-1],
            ):
                with self.assertRaisesRegex(ValueError, "manifest|example|coverage|complete"):
                    run_pipeline(args)


if __name__ == "__main__":
    unittest.main()
