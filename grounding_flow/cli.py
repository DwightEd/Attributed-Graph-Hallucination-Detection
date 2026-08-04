"""Command-line entrypoint for the grounding-flow experiment."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from .pipeline import GroundingFlowRunConfig, run_grounding_flow


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _optional_positive_int(value: str) -> int | None:
    if value.strip().casefold() in {"none", "all", "0", "unlimited"}:
        return None
    return _positive_int(value)


def _fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in (0, 1)")
    return parsed


def _coverage_fraction(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0.0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _integer_tuple(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("value must be comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("value must contain at least one integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Attention-only evidence-flow conditional-null hallucination study"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser(
        "run", help="prepare graphs, fit label-free states, freeze scores, then evaluate"
    )
    run.add_argument("--extraction-dir", type=Path, required=True)
    run.add_argument("--examples", type=Path, required=True)
    run.add_argument("--evaluation-labels", type=Path)
    run.add_argument("--output-dir", "--output", type=Path, required=True)
    run.add_argument("--device", default="cuda")
    run.add_argument("--conversion-chunk-edges", type=_positive_int, default=8_192)
    run.add_argument("--query-block", type=_positive_int, default=32)
    run.add_argument("--validation-fraction", type=_fraction, default=0.10)
    run.add_argument("--test-fraction", type=_fraction, default=0.20)
    run.add_argument("--split-seed", type=int, default=42)
    run.add_argument("--limit-pairs", type=_optional_positive_int)
    run.add_argument(
        "--expected-candidates", type=_optional_positive_int, default=2_000
    )
    run.add_argument(
        "--min-test-pair-coverage", type=_coverage_fraction, default=0.90
    )
    run.add_argument(
        "--fail-on-low-coverage",
        action="store_true",
        help="stop before labels instead of reporting a scoped identifiable-subset pilot",
    )
    group = run.add_mutually_exclusive_group()
    group.add_argument(
        "--group-by-prompt", dest="group_by_prompt", action="store_true"
    )
    group.add_argument(
        "--no-group-by-prompt", dest="group_by_prompt", action="store_false"
    )
    run.set_defaults(group_by_prompt=True)
    cache = run.add_mutually_exclusive_group()
    cache.add_argument(
        "--require-complete-cache",
        dest="require_complete_cache",
        action="store_true",
    )
    cache.add_argument(
        "--allow-partial-cache",
        dest="require_complete_cache",
        action="store_false",
    )
    run.set_defaults(require_complete_cache=True)
    run.add_argument("--evidence-segments", type=_integer_tuple, default=(1,))
    run.add_argument("--num-nulls", type=_positive_int, default=32)
    run.add_argument("--null-swaps-per-edge", type=_positive_int, default=4)
    run.add_argument(
        "--lag-boundaries",
        type=_integer_tuple,
        default=(4, 8, 16, 32, 64, 128),
    )
    run.add_argument("--null-max-attempt-factor", type=_positive_int, default=12)
    run.add_argument("--null-std-floor", type=_positive_float, default=1e-6)
    run.add_argument("--pca-components", type=_positive_int, default=32)
    run.add_argument("--pca-fit-tokens", type=_positive_int, default=20_000)
    run.add_argument("--hmm-iterations", type=_positive_int, default=50)
    run.add_argument("--hmm-tolerance", type=_nonnegative_float, default=1e-4)
    run.add_argument("--hmm-variance-floor", type=_positive_float, default=1e-4)
    run.add_argument("--bootstrap-samples", type=_positive_int, default=1_000)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--skip-evaluation", action="store_true")
    run.add_argument("--no-resume", dest="resume", action="store_false")
    run.set_defaults(resume=True, handler=_run)
    return parser


def config_from_args(args: argparse.Namespace) -> GroundingFlowRunConfig:
    return GroundingFlowRunConfig(
        extraction_dir=args.extraction_dir,
        examples_path=args.examples,
        evaluation_labels_path=args.evaluation_labels,
        output_dir=args.output_dir,
        device=args.device,
        conversion_chunk_edges=args.conversion_chunk_edges,
        query_block=args.query_block,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        split_seed=args.split_seed,
        limit_pairs=args.limit_pairs,
        expected_candidates=args.expected_candidates,
        min_test_pair_coverage=args.min_test_pair_coverage,
        fail_on_low_coverage=args.fail_on_low_coverage,
        group_by_prompt=args.group_by_prompt,
        require_complete_cache=args.require_complete_cache,
        evidence_segment_ids=tuple(args.evidence_segments),
        num_nulls=args.num_nulls,
        null_swaps_per_edge=args.null_swaps_per_edge,
        null_lag_boundaries=tuple(args.lag_boundaries),
        null_max_attempt_factor=args.null_max_attempt_factor,
        null_std_floor=args.null_std_floor,
        pca_components=args.pca_components,
        pca_fit_tokens=args.pca_fit_tokens,
        hmm_iterations=args.hmm_iterations,
        hmm_tolerance=args.hmm_tolerance,
        hmm_variance_floor=args.hmm_variance_floor,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
        resume=args.resume,
        skip_evaluation=args.skip_evaluation,
    )


def _progress(stage: str, current: int, total: int) -> None:
    if total < 1:
        return
    if current == 1 or current == total or current % 25 == 0:
        print(
            json.dumps(
                {
                    "event": "progress",
                    "stage": stage,
                    "current": current,
                    "total": total,
                    "percent": round(100.0 * current / total, 1),
                },
                sort_keys=True,
            ),
            flush=True,
        )


def _metric(name: str, step: int, value: float) -> None:
    print(
        json.dumps(
            {"event": "training_metric", "name": name, "step": step, "value": value},
            sort_keys=True,
        ),
        flush=True,
    )


def _run(args: argparse.Namespace) -> dict[str, object]:
    config = config_from_args(args)
    print(
        json.dumps(
            {
                "event": "grounding_flow_start",
                "method": "conditional_null_evidence_transport_hmm",
                "device": config.device,
                "num_nulls": config.num_nulls,
                "pca_components": config.pca_components,
                "hmm_iterations": config.hmm_iterations,
                "labels": "evaluation_only" if not config.skip_evaluation else "never",
                "output_dir": str(config.output_dir),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return run_grounding_flow(
        config, progress_callback=_progress, metric_callback=_metric
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    metrics = result.get("core_metrics", {})
    coverage = result.get("identifiable_coverage_gate", {})
    if isinstance(coverage, dict) and not coverage.get("coverage_target_met", True):
        print(
            json.dumps(
                {
                    "event": "low_identifiability_coverage",
                    "coverage": coverage.get("test_pair_coverage"),
                    "minimum": coverage.get("minimum_test_pair_coverage"),
                    "population": coverage.get("evaluation_population"),
                    "warning": coverage.get("warning"),
                },
                sort_keys=True,
            ),
            flush=True,
        )
    length_auroc: object = "n/a"
    evaluation_path = result.get("evaluation")
    if evaluation_path:
        value = json.loads(Path(str(evaluation_path)).read_text(encoding="utf-8"))
        length_auroc = (
            value.get("primary", {}).get("length_only", {}).get("auroc", "n/a")
        )
    print(
        "AUROC={auroc} AUPRC={auprc} paired_accuracy={paired} "
        "response_length_AUROC={length} scope={scope} scored={scored} total={total} "
        "output_dir={output}".format(
            auroc=metrics.get("auroc", "n/a"),
            auprc=metrics.get("average_precision", "n/a"),
            paired=metrics.get("paired_accuracy", "n/a"),
            length=length_auroc,
            scope=result["experiment_scope"],
            scored=json.dumps(result["scored_partition_counts"], sort_keys=True),
            total=json.dumps(result["partition_counts"], sort_keys=True),
            output=result["output_dir"],
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "config_from_args", "main"]
