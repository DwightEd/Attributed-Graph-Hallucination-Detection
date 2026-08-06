from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from counterfactual_grounding import graph_build
from counterfactual_grounding.data.store import StoredPredictionGraph

OBSERVER_RUNTIME_SPEC = {
    "transformers_version": "4.52.3",
    "torch_version": "2.6.0+cu124",
    "dtype": "torch.float16",
    "attn_implementation": "eager",
}


def _write_cache_manifest(
    path: Path,
    cache_paths: list[Path],
    *,
    model_files_sha256: dict[str, str] | None = None,
) -> Path:
    spec: dict[str, object] = dict(OBSERVER_RUNTIME_SPEC)
    if model_files_sha256 is not None:
        spec["model_files_sha256"] = model_files_sha256
    path.write_text(
        json.dumps(
            {
                "attention_cache_spec": spec,
                "cache_files_sha256": {
                    item.name: graph_build.file_sha256(item) for item in cache_paths
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class _Example:
    def __init__(self, response_id: str, source_id: str) -> None:
        self.response_id = response_id
        self.source_id = source_id
        self.response = "answer"
        self.task_type = "Summary"
        self.generator_model = "observer"

    def source_record(self) -> dict[str, object]:
        return {}


def _install_graph_build_fakes(
    monkeypatch: pytest.MonkeyPatch,
    *,
    cache_paths: list[Path],
    record_split: str,
    inventory_complete: bool,
    manifest_path: Path | None = None,
) -> None:
    import transformers

    if manifest_path is None and cache_paths:
        manifest_path = _write_cache_manifest(
            cache_paths[0].parent / "manifest.json", cache_paths
        )
    monkeypatch.setattr(
        transformers,
        "AutoTokenizer",
        SimpleNamespace(
            from_pretrained=lambda *args, **kwargs: SimpleNamespace(is_fast=True)
        ),
    )
    monkeypatch.setattr(
        graph_build,
        "discover_attention_cache",
        lambda *args, **kwargs: [
            SimpleNamespace(path=path, dataset_split="train") for path in cache_paths
        ],
    )
    monkeypatch.setattr(
        graph_build,
        "audit_attention_cache",
        lambda *args, **kwargs: {
            "train": {
                "complete": inventory_complete,
                "manifest_state": "complete" if inventory_complete else "in_progress",
                "observed_file_count": len(cache_paths),
                "manifest_path": str(manifest_path) if manifest_path else None,
            }
        },
        raising=False,
    )
    records = {
        path: {
            "source_id": f"source-{index}",
            "response_id": f"response-{index}",
            "dataset_split": record_split,
        }
        for index, path in enumerate(cache_paths)
    }
    monkeypatch.setattr(
        graph_build,
        "load_attention_record",
        lambda path, **kwargs: records[Path(path)],
    )
    monkeypatch.setattr(
        graph_build,
        "load_ragtruth_examples",
        lambda *args, **kwargs: [
            _Example(record["response_id"], record["source_id"])
            for record in records.values()
        ],
    )
    monkeypatch.setattr(graph_build, "build_ragtruth_layout", lambda *args: object())
    monkeypatch.setattr(
        graph_build,
        "generate_equal_token_counterfactuals",
        lambda *args, **kwargs: (
            SimpleNamespace(changed_positions=torch.tensor([0])),
        ),
    )
    monkeypatch.setattr(
        graph_build,
        "adapt_legacy_cache_to_layout",
        lambda record, layout, **kwargs: dict(record),
    )
    monkeypatch.setattr(
        graph_build, "build_prediction_event_graph", lambda adapted: adapted
    )

    def _save(graph: dict[str, object], output_dir: Path) -> StoredPredictionGraph:
        return StoredPredictionGraph(
            path=output_dir / f"{graph['response_id']}.graph.pt",
            sha256="c" * 64,
            source_id=str(graph["source_id"]),
            response_id=str(graph["response_id"]),
            dataset_split=str(graph["dataset_split"]),
            available_events=2,
            total_events=3,
        )

    monkeypatch.setattr(graph_build, "save_prediction_event_graph", _save)


def _metadata_files(tmp_path: Path) -> tuple[Path, Path, Path]:
    responses = tmp_path / "response.jsonl"
    sources = tmp_path / "source_info.jsonl"
    tokenizer = tmp_path / "tokenizer"
    responses.write_text("{}\n", encoding="utf-8")
    sources.write_text("{}\n", encoding="utf-8")
    tokenizer.mkdir()
    return responses, sources, tokenizer


def test_limit_and_partial_cache_are_not_reported_as_complete_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_paths = [tmp_path / "attention_0.pt", tmp_path / "attention_1.pt"]
    for path in cache_paths:
        path.write_bytes(path.name.encode("utf-8"))
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=cache_paths,
        record_split="train",
        inventory_complete=False,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    manifest = graph_build.build_legacy_prediction_graph_store(
        cache_root=tmp_path,
        responses=responses,
        sources=sources,
        tokenizer_source=tokenizer,
        output_dir=tmp_path / "out",
        split="train",
        limit=1,
    )

    assert manifest["build_state"] == "complete"
    assert manifest["state"] != "complete"
    assert manifest["cache_inventory_complete"] is False
    assert manifest["graph_inventory_complete"] is False
    assert manifest["selection_scope"] == "limited_prefix"
    assert manifest["requested_limit"] == 1
    assert manifest["discovered_cache_files"] == 2
    assert manifest["selected_cache_files"] == 1
    assert manifest["cache_model_identity_status"] == "unverified_legacy_manifest"


def test_full_build_reports_complete_only_for_audited_complete_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "attention_0.pt"
    cache_path.write_bytes(b"cache")
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[cache_path],
        record_split="train",
        inventory_complete=True,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    manifest = graph_build.build_legacy_prediction_graph_store(
        cache_root=tmp_path,
        responses=responses,
        sources=sources,
        tokenizer_source=tokenizer,
        output_dir=tmp_path / "out",
        split="train",
        limit=None,
    )

    assert manifest["state"] == "complete"
    assert manifest["build_state"] == "complete"
    assert manifest["cache_inventory_complete"] is True
    assert manifest["graph_inventory_complete"] is True
    assert manifest["selection_scope"] == "full_cache_inventory"
    assert manifest["requested_limit"] is None
    assert manifest["cache_model_identity_status"] == "unverified_legacy_manifest"


def test_graph_build_rejects_cache_record_split_mismatching_its_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_path = tmp_path / "attention_0.pt"
    cache_path.write_bytes(b"cache")
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[cache_path],
        record_split="test",
        inventory_complete=True,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    with pytest.raises(ValueError, match="split"):
        graph_build.build_legacy_prediction_graph_store(
            cache_root=tmp_path,
            responses=responses,
            sources=sources,
            tokenizer_source=tokenizer,
            output_dir=tmp_path / "out",
            split="train",
            limit=None,
        )


def test_graph_build_rejects_empty_cache_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest_path = _write_cache_manifest(tmp_path / "manifest.json", [])
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[],
        record_split="train",
        inventory_complete=False,
        manifest_path=manifest_path,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    with pytest.raises(FileNotFoundError, match="no formal attention-cache"):
        graph_build.build_legacy_prediction_graph_store(
            cache_root=tmp_path,
            responses=responses,
            sources=sources,
            tokenizer_source=tokenizer,
            output_dir=tmp_path / "out",
            split="train",
            limit=None,
        )


def test_graph_build_recomputes_supplied_model_source_signature(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[],
        record_split="train",
        inventory_complete=False,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    with pytest.raises(RuntimeError, match="actual model/tokenizer source bytes"):
        graph_build.build_legacy_prediction_graph_store(
            cache_root=tmp_path,
            responses=responses,
            sources=sources,
            tokenizer_source=tokenizer,
            output_dir=tmp_path / "out",
            split="train",
            limit=None,
            model_source_signature="sha256:" + "0" * 64,
        )


def test_graph_build_verifies_legacy_extractor_model_file_hash_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "attention_0.pt"
    cache_path.write_bytes(b"cache")
    responses, sources, tokenizer = _metadata_files(tmp_path)
    (tokenizer / "config.json").write_bytes(b"model-config")
    manifest_path = tmp_path / "manifest.json"
    _write_cache_manifest(
        manifest_path,
        [cache_path],
        model_files_sha256={
            "config.json": graph_build.file_sha256(tokenizer / "config.json")
        },
    )
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[cache_path],
        record_split="train",
        inventory_complete=True,
        manifest_path=manifest_path,
    )

    manifest = graph_build.build_legacy_prediction_graph_store(
        cache_root=tmp_path,
        responses=responses,
        sources=sources,
        tokenizer_source=tokenizer,
        output_dir=tmp_path / "out",
        split="train",
        limit=None,
    )

    assert manifest["cache_model_identity_status"] == (
        "verified_legacy_content_manifest"
    )
    assert manifest["cache_model_identity_evidence"] == (
        "legacy_model_files_sha256_exact_inventory"
    )


def test_graph_build_rejects_changed_legacy_extractor_model_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "attention_0.pt"
    cache_path.write_bytes(b"cache")
    responses, sources, tokenizer = _metadata_files(tmp_path)
    (tokenizer / "config.json").write_bytes(b"current-model-config")
    manifest_path = tmp_path / "manifest.json"
    _write_cache_manifest(
        manifest_path,
        [cache_path],
        model_files_sha256={"config.json": "0" * 64},
    )
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[cache_path],
        record_split="train",
        inventory_complete=True,
        manifest_path=manifest_path,
    )

    with pytest.raises(RuntimeError, match="model-file bytes differ"):
        graph_build.build_legacy_prediction_graph_store(
            cache_root=tmp_path,
            responses=responses,
            sources=sources,
            tokenizer_source=tokenizer,
            output_dir=tmp_path / "out",
            split="train",
            limit=None,
        )


def test_graph_build_records_observer_runtime_and_verified_cache_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "attention_0.pt"
    cache_path.write_bytes(b"cache")
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[cache_path],
        record_split="train",
        inventory_complete=True,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    manifest = graph_build.build_legacy_prediction_graph_store(
        cache_root=tmp_path,
        responses=responses,
        sources=sources,
        tokenizer_source=tokenizer,
        output_dir=tmp_path / "out",
        split="train",
        limit=None,
    )
    row = json.loads((tmp_path / "out" / "train.index.jsonl").read_text())

    assert manifest["observer_runtime"]["dtype"] == "float16"
    assert manifest["observer_runtime_signature"].startswith("sha256:")
    assert row["observer_runtime"] == manifest["observer_runtime"]
    assert row["observer_runtime_signature"] == manifest[
        "observer_runtime_signature"
    ]
    assert row["cache_content_identity_status"] == (
        "verified_extractor_cache_files_sha256"
    )
    assert row["extractor_declared_cache_sha256"] == graph_build.file_sha256(
        cache_path
    )


def test_graph_build_rejects_cache_bytes_changed_after_extractor_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "attention_0.pt"
    cache_path.write_bytes(b"original")
    manifest_path = _write_cache_manifest(tmp_path / "manifest.json", [cache_path])
    cache_path.write_bytes(b"tampered")
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[cache_path],
        record_split="train",
        inventory_complete=True,
        manifest_path=manifest_path,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    with pytest.raises(RuntimeError, match="cache-file bytes differ"):
        graph_build.build_legacy_prediction_graph_store(
            cache_root=tmp_path,
            responses=responses,
            sources=sources,
            tokenizer_source=tokenizer,
            output_dir=tmp_path / "out",
            split="train",
            limit=None,
        )


def test_graph_build_rejects_cache_manifest_without_observer_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache_path = tmp_path / "attention_0.pt"
    cache_path.write_bytes(b"cache")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "attention_cache_spec": {},
                "cache_files_sha256": {
                    cache_path.name: graph_build.file_sha256(cache_path)
                },
            }
        ),
        encoding="utf-8",
    )
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[cache_path],
        record_split="train",
        inventory_complete=True,
        manifest_path=manifest_path,
    )
    responses, sources, tokenizer = _metadata_files(tmp_path)

    with pytest.raises(ValueError, match="transformers_version"):
        graph_build.build_legacy_prediction_graph_store(
            cache_root=tmp_path,
            responses=responses,
            sources=sources,
            tokenizer_source=tokenizer,
            output_dir=tmp_path / "out",
            split="train",
            limit=None,
        )
