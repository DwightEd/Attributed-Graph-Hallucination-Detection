"""Label-gated metrics and cached annotation joins for frozen token scores."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from .ragtruth_data import atomic_json, discover_attention_paths, load_compact_manifest
from .ragtruth_graph import load_attention_sample


def _binary_label(value: object, context: str) -> int:
    try:
        tensor = torch.as_tensor(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{context} must be a finite binary label in {{0, 1}}") from error
    if tensor.numel() != 1 or tensor.is_complex():
        raise ValueError(f"{context} must be a finite binary label in {{0, 1}}")
    if tensor.is_floating_point() and not bool(torch.isfinite(tensor).item()):
        raise ValueError(f"{context} must be a finite binary label in {{0, 1}}")
    if not bool(((tensor == 0) | (tensor == 1)).item()):
        raise ValueError(f"{context} must be a finite binary label in {{0, 1}}")
    return int(tensor.item())


def _binary_label_tensor(value: object, context: str) -> torch.Tensor:
    raw = torch.as_tensor(value).flatten()
    if raw.is_complex() or (
        raw.is_floating_point() and not bool(torch.isfinite(raw).all())
    ) or bool((~((raw == 0) | (raw == 1))).any()):
        raise ValueError(f"{context} must contain only finite binary labels in {{0, 1}}")
    return raw.long()


def evaluate_token_score_records(
    score_records: Sequence[Mapping[str, object]],
    token_labels: Mapping[tuple[str, str, int], int],
) -> dict[str, float | int]:
    """Join frozen scores to external labels by exact token identity."""

    if not score_records:
        raise ValueError("score records are empty")
    score_by_key: dict[tuple[str, str, int], float] = {}
    for record in score_records:
        key = (str(record["source_id"]), str(record["sample_id"]), int(record["token_idx"]))
        if key in score_by_key:
            raise ValueError(f"duplicate score token identity: {key}")
        score_by_key[key] = float(record["score"])
    label_keys, score_keys = set(token_labels), set(score_by_key)
    if label_keys != score_keys:
        raise ValueError(f"score/label token alignment failed: missing={len(score_keys - label_keys)}, "
                         f"extra={len(label_keys - score_keys)}")
    ordered = sorted(score_keys)
    labels = np.asarray(
        [_binary_label(token_labels[key], f"token label {key}") for key in ordered],
        dtype=np.int8,
    )
    scores = np.asarray([score_by_key[key] for key in ordered], dtype=np.float64)
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("both negative and positive token labels are required")
    if not np.isfinite(scores).all():
        raise ValueError("token scores must be finite")
    from sklearn.metrics import average_precision_score, roc_auc_score
    prevalence = float(labels.mean())
    auprc = float(average_precision_score(labels, scores))
    return {"token_count": len(labels), "positive_count": int(labels.sum()),
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
    expected_provenance: Mapping[tuple[str, str], Mapping[str, object]] | None = None,
) -> tuple[dict[tuple[str, str, int], int], dict[str, int]]:
    """Read formal or legacy token annotations only after scores have frozen."""

    if not -8 <= label_shift <= 8:
        raise ValueError("label_shift must be between -8 and 8")
    expected: dict[tuple[str, str], set[int]] = {}
    for record in score_records:
        expected.setdefault((str(record["source_id"]), str(record["sample_id"])), set()).add(int(record["token_idx"]))
    paths = list(attention_paths) if attention_paths is not None else discover_attention_paths(attention_dir)
    try:
        from tqdm.auto import tqdm
    except ImportError as error:
        raise RuntimeError("tqdm is required; install requirements-unsupervised-token-graph.txt") from error
    labels: dict[tuple[str, str, int], int] = {}
    seen: set[tuple[str, str]] = set()
    positive_count = 0
    for path in tqdm(paths, desc="join held-out token labels", unit="sample"):
        sample = load_attention_sample(path, device="cpu", mmap=True, include_labels=True)
        identity = (str(sample["source_id"]), str(sample["sample_id"]))
        if identity not in expected:
            del sample
            continue
        if identity in seen:
            raise ValueError(f"duplicate attention sample identity: {identity}")
        seen.add(identity)
        if expected_provenance is not None:
            expected_raw = expected_provenance.get(identity)
            if expected_raw is None:
                raise ValueError(f"compact manifest is missing raw provenance for {identity}")
            cache_format = str(sample["cache_format"])
            if cache_format == "formal_sparse_csr":
                layers = int(sample["num_attention_layers"])
                heads = int(sample["num_attention_heads"])
                token_count = int(torch.as_tensor(sample["token_ids"]).numel())
                cache_dtype = sample["cache_dtype"]
                fingerprint = sample["attention_cache_fingerprint"]
                input_policy = sample["input_policy"]
                was_truncated = sample["was_truncated"]
                attention_floor = sample["attention_floor"]
            else:
                attention = torch.as_tensor(sample["attention"])
                layers, heads = map(int, attention.shape[:2])
                token_count = (
                    int(torch.as_tensor(sample["token_ids"]).numel())
                    if "token_ids" in sample
                    else int(attention.shape[-1])
                )
                cache_dtype = str(attention.dtype).removeprefix("torch.")
                fingerprint = input_policy = was_truncated = attention_floor = None
            observed = {
                "cache_format": cache_format,
                "attention_cache_fingerprint": fingerprint,
                "cache_dtype": cache_dtype,
                "input_policy": input_policy,
                "was_truncated": was_truncated,
                "attention_floor": attention_floor,
                "response_idx": int(sample["response_idx"]),
                "token_count": token_count,
                "layers": layers,
                "heads": heads,
            }
            divergent = [
                key for key, value in expected_raw.items() if observed.get(key) != value
            ]
            if divergent:
                raise RuntimeError(
                    f"raw cache provenance diverges for {path}: {', '.join(divergent)}"
                )
        cache_format = str(sample.get("cache_format", ""))
        formal = cache_format == "formal_sparse_csr"
        if formal and label_shift:
            raise ValueError("label_shift is unsupported for formal global y_token annotations")
        if "y_token" in sample:
            raw = _binary_label_tensor(sample["y_token"], f"cached labels in {path}")
        elif "hallucination_labels" in sample:
            raw = _binary_label_tensor(
                sample["hallucination_labels"], f"cached labels in {path}"
            )
        else:
            raise ValueError(f"cached token labels are absent: {path}")
        if formal or "token_ids" in sample:
            token_count = int(torch.as_tensor(sample["token_ids"]).numel())
        else:
            token_count = int(torch.as_tensor(sample["attention"]).shape[-1])
        response_idx = int(sample["response_idx"])
        if len(raw) != token_count:
            raise ValueError(f"invalid cached token labels: {path}")
        effective_positive = torch.nonzero(raw, as_tuple=False).flatten()
        if not formal:
            effective_positive = effective_positive + label_shift
        if bool(
            (effective_positive.lt(response_idx) | effective_positive.ge(token_count)).any()
        ):
            raise ValueError(
                f"shifted positive labels fall outside the response token range in {path}; "
                "verify the span/token alignment and pass an explicit --label-shift only if justified"
            )
        for token_idx in expected[identity]:
            source_idx = token_idx if formal else token_idx - label_shift
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
    sentence_score_path: str | Path | None = None,
) -> dict[str, object]:
    """Evaluate frozen scores; the sole label-reading experiment stage."""

    scores = read_score_records(score_path)
    selected_attention_paths = None
    if graph_dir is not None:
        identities = {(str(record["source_id"]), str(record["sample_id"])) for record in scores}
        attention_root = Path(attention_dir).resolve()
        manifest_records = load_compact_manifest(graph_dir)
        source_by_identity = {
            (str(record["source_id"]), str(record["sample_id"])): record
            for record in manifest_records
        }
        if len(source_by_identity) != len(manifest_records):
            raise RuntimeError("compact manifest contains duplicate sample identities")
        if not identities.issubset(source_by_identity):
            raise ValueError("compact manifest is missing scored sample identities")
        selected_attention_paths = []
        for identity in sorted(identities):
            path = (attention_root / str(source_by_identity[identity]["source_file"])).resolve()
            if attention_root not in path.parents or not path.is_file():
                raise RuntimeError(f"unsafe or missing source attention file: {path}")
            selected_attention_paths.append(path)
    expected_provenance = None
    if graph_dir is not None:
        provenance_keys = (
            "cache_format", "attention_cache_fingerprint", "cache_dtype",
            "input_policy", "was_truncated", "attention_floor", "response_idx",
            "token_count", "layers", "heads",
        )
        expected_provenance = {
            identity: {key: source_by_identity[identity][key] for key in provenance_keys}
            for identity in identities
        }
    token_labels, alignment = load_cached_token_labels(
        attention_dir,
        scores,
        label_shift=label_shift,
        attention_paths=selected_attention_paths,
        expected_provenance=expected_provenance,
    )
    overall = evaluate_token_score_records(scores, token_labels)
    components = {}
    for name in ("node_residual", "neighborhood_residual", "route_residual"):
        component_records = [{"source_id": record["source_id"], "sample_id": record["sample_id"],
                              "token_idx": record["token_idx"], "score": record[name]} for record in scores]
        components[name] = evaluate_token_score_records(component_records, token_labels)
    grouped: dict[tuple[str, str], dict[str, float | int]] = {}
    for record in scores:
        key = (str(record["source_id"]), str(record["sample_id"]))
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
    if sentence_score_path is not None:
        from .ragtruth_sentences import evaluate_sentence_score_records

        sentence_records = read_score_records(sentence_score_path)
        report["sentence"] = evaluate_sentence_score_records(
            sentence_records, token_labels
        )
        report["sentence_score_file"] = str(sentence_score_path)
    atomic_json(Path(output_path), report)
    return report
