from __future__ import annotations

import json
from pathlib import Path

from sklearn.metrics import average_precision_score, roc_auc_score

from .data import discover_graphs


def _read_scores(path: str | Path) -> dict[str, float]:
    result: dict[str, float] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            result[str(row["sample_id"])] = float(row["cola_anomaly_score"])
    return result


def evaluate(
    graph_root: str | Path,
    response_scores: str | Path,
    output_path: str | Path,
) -> dict[str, float | int]:
    """Read labels only after frozen test scores have been written."""
    scores = _read_scores(response_scores)
    labels: list[int] = []
    values: list[float] = []
    for path in discover_graphs(graph_root, "test"):
        import torch

        raw = torch.load(path, map_location="cpu", weights_only=True)
        sample_id = str(raw.get("sample_id", raw.get("response_id", path.stem)))
        if sample_id not in scores:
            raise KeyError(f"missing CoLA score for {sample_id}")
        labels.append(int(raw["response_label"]))
        values.append(scores[sample_id])

    report = {
        "samples": len(labels),
        "positive_samples": int(sum(labels)),
        "positive_fraction": float(sum(labels) / len(labels)) if labels else 0.0,
        "auroc": float(roc_auc_score(labels, values)),
        "average_precision": float(average_precision_score(labels, values)),
    }
    Path(output_path).write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report
