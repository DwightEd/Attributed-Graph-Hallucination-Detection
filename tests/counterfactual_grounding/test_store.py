from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch

from counterfactual_grounding.data.graph import Segment, build_prediction_event_graph
from counterfactual_grounding.data.ragtruth import RagTruthTokenLayout
from counterfactual_grounding.data.store import (
    adapt_legacy_cache_to_layout,
    save_prediction_event_graph,
)


def _layout() -> RagTruthTokenLayout:
    return RagTruthTokenLayout(
        rendered_text="abcdefg",
        input_ids=torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        segment_ids=torch.tensor(
            [
                Segment.QUERY,
                Segment.QUERY,
                Segment.EVIDENCE,
                Segment.EVIDENCE,
                Segment.RESPONSE,
                Segment.RESPONSE,
                Segment.RESPONSE,
            ],
            dtype=torch.int8,
        ),
        response_idx=4,
        evidence_token_positions=torch.tensor([2, 3]),
        evidence_chunks=(),
    )


def _raw() -> dict[str, object]:
    return {
        "source_id": "s1",
        "response_id": "r1",
        "dataset_split": "train",
        "attention_cache_fingerprint": "fp1",
        "response_idx": 4,
        "num_attention_layers": 1,
        "num_attention_heads": 1,
        "attention_floor": 0.01,
        "token_ids": torch.tensor([1, 2, 3, 4, 5, 6, 7]),
        "attention_diagonal": torch.tensor([[[0.1, 0.1, 0.1, 0.1, 0.3, 0.3, 0.3]]]),
        "response_row_ptr": torch.tensor([0, 1, 2, 3]),
        "response_column_indices": torch.tensor([0, 1, 2]),
        "response_values": torch.tensor([0.2, 0.2, 0.2]),
        "y_token": torch.ones(7),
        "labels": [{"start": 0, "end": 1}],
    }


def test_legacy_adapter_whitelists_cache_fields_and_never_copies_labels():
    adapted = adapt_legacy_cache_to_layout(_raw(), _layout())

    assert "y_token" not in adapted
    assert "labels" not in adapted
    torch.testing.assert_close(adapted["segment_ids"], _layout().segment_ids)


def test_canonical_graph_is_saved_once_with_no_label_fields(tmp_path: Path):
    adapted = adapt_legacy_cache_to_layout(_raw(), _layout(), cache_sha256="a" * 64)
    graph = build_prediction_event_graph(adapted)

    first = save_prediction_event_graph(graph, tmp_path)
    second = save_prediction_event_graph(graph, tmp_path)

    assert first.path == second.path
    assert first.sha256 == second.sha256
    payload = torch.load(first.path, map_location="cpu", weights_only=True)
    assert payload["schema"] == "cept-prediction-event-graph-v2"
    assert all("label" not in key.casefold() for key in payload)


def test_canonical_path_changes_when_layout_changes_under_the_same_cache_sha(
    tmp_path: Path,
):
    factual_layout = _layout()
    changed_segments = factual_layout.segment_ids.clone()
    changed_segments[1] = int(Segment.EVIDENCE)
    revised_layout = replace(factual_layout, segment_ids=changed_segments)
    cache_sha256 = "b" * 64

    factual_graph = build_prediction_event_graph(
        adapt_legacy_cache_to_layout(_raw(), factual_layout, cache_sha256=cache_sha256)
    )
    revised_graph = build_prediction_event_graph(
        adapt_legacy_cache_to_layout(_raw(), revised_layout, cache_sha256=cache_sha256)
    )

    factual = save_prediction_event_graph(factual_graph, tmp_path)
    revised = save_prediction_event_graph(revised_graph, tmp_path)

    assert factual.path != revised.path
    assert factual.path.is_file()
    assert revised.path.is_file()
    factual_payload = torch.load(factual.path, map_location="cpu", weights_only=True)
    revised_payload = torch.load(revised.path, map_location="cpu", weights_only=True)
    assert not torch.equal(
        factual_payload["segment_ids"], revised_payload["segment_ids"]
    )
