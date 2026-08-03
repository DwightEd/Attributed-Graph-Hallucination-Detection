"""Label-gated metrics and cached annotation joins for frozen token scores."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path

import numpy as np
import torch

from .ragtruth_data import atomic_json, discover_attention_paths, load_compact_manifest
from .ragtruth_graph import load_attention_sample


def evaluate_token_score_records(
    score_records: Sequence[Mapping[str, object]],
    token_labels: Mapping[tuple[str, int, int], int],
) -> dict[str, float | int]:
    """Join frozen scores to external labels by exact token identity."""

    if not score_records:
        raise ValueError("score records are empty")
    score_by_key: dict[tuple[str, int, int], float] = {}
    for record in score_records:
        key = (str(record["source_id"]), int(record["original_idx"]), int(record["token_idx"]))
        if key in score_by_key:
            raise ValueError(f"duplicate score token identity: {key}")
        score_by_key[key] = float(record["score"])
    label_keys, score_keys = set(token_labels), set(score_by_key)
    if label_keys != score_keys:
        raise ValueError(f"score/label token alignment failed: missing={len(score_keys - label_keys)}, "
                         f"extra={len(label_keys - score_keys)}")
    ordered = sorted(score_keys)
    labels = np.asarray([int(token_labels[key]) for key in ordered], dtype=np.int8)
    scores = np.asarray([score_by_key[key] for key in ordered], dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("both negative and positive token labels are required")
    if not np.isfinite(scores).all():
        raise ValueError("token scores must be finite")
    from sklearn.metrics import average_precision_score, roc_auc_score
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, scores))
    return {"token_count": int(len(labels)), "positive_count": int(labels.sum()),
            "prevalence": prevalence, "auroc": float(roc_auc_score(labels, scores)),
            "auprc": auprc, "auprc_lift": float(auprc / prevalence)}


def read_score_records(path: str | Path) -> list[dict[str, object]]:
    records = [dict(json.loads(line)) for line in Path(path).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    if not records:
        raise ValueError(f"score file is empty: {path}")
    return records


def load_cached_token_labels(
    attention_dir: str | Path, score_records: Sequence[Mapping[str, object]], *,
    label_shift: int = 0, attention_paths: Sequence[Path] | None = None,
) -> tuple[dict[tuple[str, int, int], int], dict[str, int]]:
    """Read legacy annotations only at final evaluation time."""

    if not -8 <= label_shift <= 8:
        raise ValueError("label_shift must be between -8 and 8")
    expected: dict[tuple[str, int], set[int]] = {}
    for record in score_records:
        expected.setdefault((str(record["source_id"]), int(record["original_idx"])), set()).add(int(record["token_idx"]))
    paths = list(attention_paths) if attention_paths is not None else discover_attention_paths(attention_dir)
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError("tqdm is required; install requirements-unsupervised-token-graph.txt") from error
    labels: dict[tuple[str, int, int], int] = {}
    seen: set[tuple[str, int]] = set()
    positive_count = 0
    for path in tqdm(paths, desc="join held-out token labels", unit="sample"):
        sample = load_attention_sample(path, device="cpu", mmap=True, include_labels=True)
        identity = (str(sample["source_id"]), int(sample["original_idx"]))
        if identity not in expected:
            del sample
            continue
        if identity in seen:
            raise ValueError(f"duplicate attention sample identity: {identity}")
        seen.add(identity)
        if "hallucination_labels" not in sample:
            raise ValueError(f"cached token labels are absent: {path}")
        raw = torch.as_tensor(sample["hallucination_labels"]).flatten().long()
        token_count, response_idx = int(torch.as_tensor(sample["attention"]).shape[-1]), int(sample["response_idx"])
        if len(raw) != token_count or bool(((raw != 0) & (raw != 1)).any()):
            raise ValueError(f"invalid cached token labels: {path}")
        effective_positive = torch.nonzero(raw, as_tuple=False).flatten() + label_shift
        if bool((effective_positive < response_idx).any()):
            raise ValueError(f"positive labels fall before response_idx in {path}; verify the span/token alignment "
                             "and pass an explicit --label-shift only if justified")
        for token_idx in expected[identity]:
            source_idx = token_idx - label_shift
            if not response_idx <= token_idx < token_count or not 0 <= source_idx < len(raw):
                raise ValueError(f"score token index cannot align to cached labels: sample={identity}, token={token_idx}")
            value = int(raw[source_idx])
            labels[(identity[0], identity[1], token_idx)] = value
            positive_count += value
        del sample
    missing = set(expected).difference(seen)
    if missing:
        raise ValueError(f"attention cache is missing {len(missing)} scored samples")
    return labels, {"matched_samples": len(seen), "matched_tokens": len(labels),
                    "positive_tokens": positive_count, "label_shift": label_shift}


def evaluate_score_file(
    score_path: str | Path, attention_dir: str | Path, output_path: str | Path, *,
    label_shift: int = 0, graph_dir: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate frozen scores; the sole label-reading experiment stage."""

    scores = read_score_records(score_path)
    selected_attention_paths = None
    if graph_dir is not None:
        identities = {(str(record["source_id"]), int(record["original_idx"])) for record in scores}
        attention_root = Path(attention_dir).resolve()
        source_by_identity = {(str(record["source_id"]), int(record["original_idx"])): str(record["source_file"])
                              for record in load_compact_manifest(graph_dir)}
        if not identities.issubset(source_by_identity):
            raise ValueError("compact manifest is missing scored sample identities")
        selected_attention_paths = []
        for identity in sorted(identities):
            path = (attention_root / source_by_identity[identity]).resolve()
            if attention_root not in path.parents or not path.is_file():
                raise RuntimeError(f"unsafe or missing source attention file: {path}")
            selected_attention_paths.append(path)
    token_labels, alignment = load_cached_token_labels(attention_dir, scores, label_shift=label_shift,
                                                        attention_paths=selected_attention_paths)
    overall = evaluate_token_score_records(scores, token_labels)
    components = {}
    for name in ("node_residual", "neighborhood_residual", "route_residual"):
        component_records = [{"source_id": record["source_id"], "original_idx": record["original_idx"],
                              "token_idx": record["token_idx"], "score": record[name]} for record in scores]
        components[name] = evaluate_token_score_records(component_records, token_labels)
    grouped: dict[tuple[str, int], dict[str, float | int]] = {}
    for record in scores:
        key = (str(record["source_id"]), int(record["original_idx"]))
        token_key = (key[0], key[1], int(record["token_idx"]))
        current = grouped.setdefault(key, {"score": -math.inf, "label": 0})
        current["score"] = max(float(current["score"]), float(record["score"]))
        current["label"] = max(int(current["label"]), int(token_labels[token_key]))
    sample_labels = np.asarray([int(grouped[key]["label"]) for key in sorted(grouped)], dtype=np.int8)
    sample_scores = np.asarray([float(grouped[key]["score"]) for key in sorted(grouped)], dtype=np.float64)
    if set(sample_labels.tolist()) != {0, 1}:
        raise ValueError("both clean and hallucinated held-out samples are required")
    from sklearn.metrics import average_precision_score, roc_auc_score
    prevalence = float(sample_labels.mean())
    auprc = float(average_precision_score(sample_labels, sample_scores))
    report: dict[str, object] = {
        "token": overall,
        "sample_max": {"sample_count": len(grouped), "positive_count": int(sample_labels.sum()),
                       "prevalence": prevalence, "auroc": float(roc_auc_score(sample_labels, sample_scores)),
                       "auprc": auprc, "auprc_lift": auprc / prevalence},
        "components": components, "alignment": alignment, "score_file": str(score_path),
    }
    atomic_json(Path(output_path), report)
    return report
