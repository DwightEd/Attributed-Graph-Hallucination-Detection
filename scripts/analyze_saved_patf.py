from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from patf.model import RobustScaler, TrajectoryRanker


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def auroc(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=scores.__getitem__)
    rank_sum = 0.0
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        rank_sum += average_rank * sum(labels[index] for index in order[start:end])
        start = end
    return (
        rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)


def average_precision(labels: list[int], scores: list[float]) -> float:
    positives = sum(labels)
    if positives == 0:
        return float("nan")
    order = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    true_positive = 0
    precision_sum = 0.0
    for rank, index in enumerate(order, 1):
        if labels[index]:
            true_positive += 1
            precision_sum += true_positive / rank
    return precision_sum / positives


def median(values: list[float]) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    middle = len(values) // 2
    if len(values) % 2:
        return float(values[middle])
    return float((values[middle - 1] + values[middle]) / 2.0)


def metric_row(labels: list[int], scores: list[float]) -> dict[str, object]:
    auc = auroc(labels, scores)
    label0 = [score for label, score in zip(labels, scores) if label == 0]
    label1 = [score for label, score in zip(labels, scores) if label == 1]
    median0 = median(label0)
    median1 = median(label1)
    if math.isnan(auc):
        separability = float("nan")
        direction = "undefined"
        oriented_ap = float("nan")
    else:
        higher = auc >= 0.5
        separability = max(auc, 1.0 - auc)
        direction = (
            "higher_in_hallucination"
            if higher
            else "lower_in_hallucination"
        )
        oriented_scores = scores if higher else [-score for score in scores]
        oriented_ap = average_precision(labels, oriented_scores)
    return {
        "samples": len(labels),
        "positive_samples": sum(labels),
        "positive_fraction": sum(labels) / len(labels) if labels else float("nan"),
        "median_label_0": median0,
        "median_label_1": median1,
        "median_delta_1_minus_0": median1 - median0,
        "auc": auc,
        "separability": separability,
        "direction": direction,
        "average_precision": average_precision(labels, scores),
        "oriented_average_precision": oriented_ap,
    }


def resolve_response(record, responses, response_by_id):
    sample_id = str(record.get("sample_id", ""))
    response = response_by_id.get(sample_id)
    if response is None and record.get("original_idx") is not None:
        index = int(record["original_idx"])
        if 0 <= index < len(responses):
            response = responses[index]
    if response is None:
        raise KeyError(f"RAGTruth response not found for sample_id={sample_id}")
    return response


def load_saved_features(feature_paths: list[Path], ragtruth_root: Path):
    responses = read_jsonl(ragtruth_root / "response.jsonl")
    response_by_id = {str(row["id"]): row for row in responses}
    sources = {
        str(row["source_id"]): row
        for row in read_jsonl(ragtruth_root / "source_info.jsonl")
    }

    originals = []
    variant_values = defaultdict(list)
    metadata = []
    feature_names = None
    schema = None
    layer_count = None

    for path in feature_paths:
        record = torch.load(path, map_location="cpu", weights_only=True)
        current_schema = str(record.get("schema", "unknown"))
        names = tuple(str(name) for name in record["feature_names"])
        trajectories = {
            str(name): torch.as_tensor(value).float()
            for name, value in record["trajectories"].items()
        }
        if "original" not in trajectories:
            raise ValueError(f"missing original trajectory: {path}")
        if schema is None:
            schema = current_schema
            feature_names = names
            layer_count = int(trajectories["original"].shape[0])
        elif (
            current_schema != schema
            or names != feature_names
            or int(trajectories["original"].shape[0]) != layer_count
        ):
            raise ValueError(f"incompatible feature contract: {path}")

        response = resolve_response(record, responses, response_by_id)
        source_id = str(
            record.get("source_id", response.get("source_id", "unknown"))
        )
        source = sources.get(source_id, {})
        originals.append(trajectories["original"])
        for name, value in trajectories.items():
            variant_values[name].append(value)
        metadata.append(
            {
                "feature_file": str(path),
                "sample_id": str(
                    record.get("sample_id", response.get("id", ""))
                ),
                "source_id": source_id,
                "task": str(source.get("task_type", "unknown")),
                "model": str(response.get("model", "unknown")),
                "token_count": int(record.get("token_count", 0)),
                "response_idx": int(record.get("response_idx", 0)),
                "label": int(bool(response.get("labels", []))),
            }
        )

    variants = {
        name: torch.stack(values)
        for name, values in variant_values.items()
        if len(values) == len(originals)
    }
    return (
        torch.stack(originals),
        variants,
        feature_names or (),
        metadata,
        schema or "unknown",
    )


def slope(values: torch.Tensor) -> torch.Tensor:
    layers = values.shape[1]
    if layers <= 1:
        return torch.zeros(values.shape[0], values.shape[2])
    x = torch.linspace(-1.0, 1.0, layers)
    return (
        values * x.view(1, layers, 1)
    ).sum(dim=1) / x.square().sum().clamp_min(1e-12)


def feature_diagnostics(original, feature_names, labels, output_dir):
    layer_rows = []
    for layer in range(original.shape[1]):
        for feature, name in enumerate(feature_names):
            layer_rows.append(
                {
                    "layer": layer,
                    "feature": name,
                    **metric_row(labels, original[:, layer, feature].tolist()),
                }
            )
    write_csv(output_dir / "layer_feature_metrics.csv", layer_rows)

    reductions = {
        "first": original[:, 0],
        "last": original[:, -1],
        "mean": original.mean(dim=1),
        "delta_last_minus_first": original[:, -1] - original[:, 0],
        "slope": slope(original),
    }
    aggregate_rows = []
    for reduction, values in reductions.items():
        for feature, name in enumerate(feature_names):
            aggregate_rows.append(
                {
                    "reduction": reduction,
                    "feature": name,
                    **metric_row(labels, values[:, feature].tolist()),
                }
            )
    write_csv(output_dir / "aggregate_feature_metrics.csv", aggregate_rows)

    def rank_key(row):
        value = float(row["separability"])
        return -value if not math.isnan(value) else math.inf

    ranked_layers = sorted(layer_rows, key=rank_key)
    ranked_aggregates = sorted(aggregate_rows, key=rank_key)
    write_csv(output_dir / "best_layer_features.csv", ranked_layers[:100])
    write_csv(
        output_dir / "best_aggregate_features.csv",
        ranked_aggregates[:100],
    )
    return {
        "best_layer_features": ranked_layers[:10],
        "best_aggregate_features": ranked_aggregates[:10],
    }


def counterfactual_diagnostics(variants, feature_names, output_dir):
    original = variants.get("original")
    if original is None:
        return {"variants": []}
    rows = []
    summary = {}
    for variant, values in sorted(variants.items()):
        if variant == "original":
            continue
        delta = values - original
        l2 = delta.flatten(1).norm(dim=1)
        summary[variant] = {
            "samples": int(delta.shape[0]),
            "mean_l2_shift": float(l2.mean()),
            "median_l2_shift": float(l2.median()),
        }
        for feature, name in enumerate(feature_names):
            feature_delta = delta[:, :, feature].mean(dim=1)
            rows.append(
                {
                    "variant": variant,
                    "feature": name,
                    "mean_delta_variant_minus_original": float(
                        feature_delta.mean()
                    ),
                    "median_delta_variant_minus_original": float(
                        feature_delta.median()
                    ),
                    "fraction_positive_delta": float(
                        (feature_delta > 0).float().mean()
                    ),
                }
            )
    write_csv(output_dir / "counterfactual_shifts.csv", rows)
    return {"variants": list(summary), "summary": summary}


def split_groups(metadata, holdout_fraction, seed):
    groups = sorted(set(str(row["source_id"]) for row in metadata))
    if len(groups) < 2:
        raise ValueError(
            "ranker diagnostic requires at least two source_id groups"
        )
    rng = random.Random(seed)
    rng.shuffle(groups)
    holdout_count = min(
        max(1, round(len(groups) * holdout_fraction)),
        len(groups) - 1,
    )
    holdout_groups = set(groups[:holdout_count])
    fit = [
        index
        for index, row in enumerate(metadata)
        if str(row["source_id"]) not in holdout_groups
    ]
    holdout = [
        index
        for index, row in enumerate(metadata)
        if str(row["source_id"]) in holdout_groups
    ]
    return fit, holdout


def choose_counterfactual(variants, requested):
    available = [name for name in variants if name != "original"]
    if requested != "auto":
        if requested not in variants:
            raise ValueError(
                f"counterfactual '{requested}' not found; available={available}"
            )
        return requested
    for preferred in ("composite", "eroded", "incidence", "collapse"):
        if preferred in variants:
            return preferred
    if not available:
        raise ValueError(
            "no counterfactual trajectory stored in these feature files"
        )
    return sorted(available)[0]


def ranker_diagnostic(
    variants,
    feature_names,
    metadata,
    output_dir,
    args,
):
    counterfactual_name = choose_counterfactual(
        variants,
        args.counterfactual,
    )
    original = variants["original"]
    counterfactual = variants[counterfactual_name]
    fit, holdout = split_groups(
        metadata,
        args.holdout_fraction,
        args.seed,
    )

    scaler = RobustScaler.fit([original[index] for index in fit])
    clean = torch.stack(
        [scaler.transform(value) for value in original]
    ).to(args.device)
    corrupt = torch.stack(
        [scaler.transform(value) for value in counterfactual]
    ).to(args.device)

    torch.manual_seed(args.seed)
    model = TrajectoryRanker(
        len(feature_names),
        args.hidden_dim,
    ).to(args.device)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )
    generator = torch.Generator().manual_seed(args.seed)
    fit_index = torch.tensor(fit, dtype=torch.long)
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        order = fit_index[
            torch.randperm(len(fit_index), generator=generator)
        ].to(args.device)
        losses = []
        pair_acc = []
        for start in range(0, len(order), args.batch_size):
            index = order[start : start + args.batch_size]
            clean_score = model(clean[index])
            corrupt_score = model(corrupt[index])
            ranking = torch.relu(
                args.margin - (corrupt_score - clean_score)
            ).mean()
            loss = ranking + (
                args.origin_regularization * clean_score.square().mean()
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
            losses.append(float(loss.detach()))
            pair_acc.append(
                float((corrupt_score > clean_score).float().mean())
            )
        history.append(
            {
                "epoch": epoch,
                "loss": sum(losses) / len(losses),
                "pair_accuracy": sum(pair_acc) / len(pair_acc),
            }
        )

    model.eval()
    holdout_tensor = torch.tensor(
        holdout,
        dtype=torch.long,
        device=args.device,
    )
    with torch.no_grad():
        natural_score = model(clean[holdout_tensor]).cpu().tolist()
        holdout_pair_accuracy = float(
            (
                model(corrupt[holdout_tensor])
                > model(clean[holdout_tensor])
            ).float().mean()
        )

    labels = [int(metadata[index]["label"]) for index in holdout]
    overall = metric_row(
        labels,
        [float(score) for score in natural_score],
    )
    score_rows = []
    task_scores = defaultdict(list)
    for index, score in zip(holdout, natural_score):
        score_rows.append(
            {
                **metadata[index],
                "diagnostic_anomaly_score": float(score),
            }
        )
        task_scores[str(metadata[index]["task"])].append(
            (int(metadata[index]["label"]), float(score))
        )
    write_csv(output_dir / "ranker_holdout_scores.csv", score_rows)

    by_task = {
        task: metric_row(
            [label for label, _ in values],
            [score for _, score in values],
        )
        for task, values in sorted(task_scores.items())
    }
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": len(feature_names),
            "hidden_dim": args.hidden_dim,
            "feature_names": feature_names,
            "scaler_median": scaler.median,
            "scaler_scale": scaler.scale,
            "counterfactual": counterfactual_name,
            "fit_indices": fit,
            "holdout_indices": holdout,
        },
        output_dir / "diagnostic_ranker.pt",
    )
    (output_dir / "ranker_history.json").write_text(
        json.dumps(history, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "scope": "partial_train_holdout_diagnostic_not_formal_test",
        "counterfactual": counterfactual_name,
        "fit_samples": len(fit),
        "holdout_samples": len(holdout),
        "holdout_pair_accuracy": holdout_pair_accuracy,
        "overall": overall,
        "by_task": by_task,
    }
    (output_dir / "ranker_diagnostic.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved PATF feature files without recomputing attention."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ragtruth-root", type=Path, required=True)
    parser.add_argument(
        "--feature-split",
        choices=("train", "test"),
        default="train",
    )
    parser.add_argument("--analysis-dir", type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fit-ranker", action="store_true")
    parser.add_argument("--counterfactual", default="auto")
    parser.add_argument("--holdout-fraction", type=float, default=0.20)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--origin-regularization", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    feature_dir = args.output_dir / "features" / args.feature_split
    feature_paths = sorted(feature_dir.glob("*.features.pt"))
    if args.limit is not None:
        feature_paths = feature_paths[: args.limit]
    if not feature_paths:
        raise FileNotFoundError(
            f"no *.features.pt files in {feature_dir}"
        )

    analysis_dir = (
        args.analysis_dir
        or args.output_dir / "diagnostics" / args.feature_split
    )
    analysis_dir.mkdir(parents=True, exist_ok=True)
    original, variants, feature_names, metadata, schema = (
        load_saved_features(feature_paths, args.ragtruth_root)
    )
    labels = [int(row["label"]) for row in metadata]
    write_csv(analysis_dir / "sample_inventory.csv", metadata)

    report = {
        "scope": "saved_partial_feature_diagnostic",
        "feature_schema": schema,
        "feature_split": args.feature_split,
        "samples": len(metadata),
        "positive_samples": sum(labels),
        "positive_fraction": sum(labels) / len(labels),
        "layers": int(original.shape[1]),
        "features_per_layer": int(original.shape[2]),
        "trajectory_variants": sorted(variants),
        "token_count_median": median(
            [float(row["token_count"]) for row in metadata]
        ),
        "response_length_median": median(
            [
                float(
                    max(
                        0,
                        int(row["token_count"])
                        - int(row["response_idx"]),
                    )
                )
                for row in metadata
            ]
        ),
        "warning": (
            "This is a partial completed-feature subset. Parallel extraction "
            "can over-represent shorter/faster examples, so these metrics are "
            "diagnostic rather than a formal RAGTruth test result."
        ),
        "feature_diagnostics": feature_diagnostics(
            original,
            feature_names,
            labels,
            analysis_dir,
        ),
        "counterfactual_diagnostics": counterfactual_diagnostics(
            variants,
            feature_names,
            analysis_dir,
        ),
    }
    if args.fit_ranker:
        report["ranker_diagnostic"] = ranker_diagnostic(
            variants,
            feature_names,
            metadata,
            analysis_dir,
            args,
        )

    (analysis_dir / "summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "analysis_dir": str(analysis_dir),
                "feature_schema": schema,
                "samples": len(metadata),
                "positive_samples": sum(labels),
                "trajectory_variants": sorted(variants),
                "ranker_diagnostic": report.get("ranker_diagnostic"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
