from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from counterfactual_grounding import graph_build
from counterfactual_grounding.data.store import StoredPredictionGraph


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
) -> None:
    import transformers

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
    _install_graph_build_fakes(
        monkeypatch,
        cache_paths=[],
        record_split="train",
        inventory_complete=False,
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
