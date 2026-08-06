from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from counterfactual_grounding.artifacts import canonical_hash, file_sha256
from counterfactual_grounding.observer_runtime import observer_runtime_identity
from counterfactual_grounding.transport.data import TeacherTargets
from counterfactual_grounding.transport.experiment import (
    REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT,
    TransportTrainConfig,
    _available_unknown_mass_mean,
    _detection_claim_report,
    _informative_block_pairs,
    _ungated_control_from,
    evaluate_frozen_transport,
    train_transport,
)
from counterfactual_grounding.transport.model import CausalProvenanceTransport

OBSERVER_RUNTIME, OBSERVER_RUNTIME_SIGNATURE = observer_runtime_identity(
    transformers_version="4.52.3",
    torch_version="2.6.0+cu124",
    dtype="float16",
    attn_implementation="eager",
)


def _graph(source_id: str, response_id: str) -> dict[str, object]:
    return {
        "schema": "cept-prediction-event-graph-v2",
        "source_id": source_id,
        "response_id": response_id,
        "dataset_split": "train",
        "num_layers": 1,
        "num_heads": 1,
        "token_ids": torch.tensor([10, 11, 12, 13]),
        "segment_ids": torch.tensor([0, 1, 2, 2], dtype=torch.int8),
        "target_token_ids": torch.tensor([12, 13]),
        "target_token_positions": torch.tensor([2, 3]),
        "predictor_positions": torch.tensor([1, 2]),
        "row_available": torch.tensor([False, True]),
        "edge_index": torch.tensor([[0], [1]]),
        "edge_relation": torch.tensor([0], dtype=torch.int8),
        "trace_edge_id": torch.tensor([0]),
        "trace_channel": torch.tensor([0]),
        "trace_value": torch.tensor([1.0]),
        "unknown_mass": torch.tensor([[1.0], [0.0]]),
        "attention_cache_sha256": "c" * 64,
    }


def _teacher(source_id: str, response_id: str) -> dict[str, object]:
    events = []
    for position, token_id, effect in ((2, 12, 0.0), (3, 13, 0.5)):
        events.append(
            {
                "target_position": position,
                "predictor_position": position - 1,
                "target_token_id": token_id,
                "total_effect": effect,
                "direct_effect": effect,
                "history_mediated_effect": 0.0,
                "alternate_history_effect": 0.0,
                "interaction": 0.0,
                "decomposition_residual": 0.0,
                "direct_seed_rescue": effect,
                "joint_seed_history_rescue": effect,
                "representation_residual": 0.0,
                "seed_history_interaction": 0.0,
                "self_patch_error": 0.0,
                "block_rescue": {},
            }
        )
    return {
        "schema": "cept-causal-transport-teacher-v2",
        "source_id": source_id,
        "response_id": response_id,
        "official_split": "train",
        "model_signature": "sha256:model",
        "model_source_signature": "sha256:source",
        "tokenizer_signature": "sha256:tokenizer",
        "observer_runtime": OBSERVER_RUNTIME,
        "observer_runtime_signature": OBSERVER_RUNTIME_SIGNATURE,
        "factual_input_ids_sha256": canonical_hash([10, 11, 12, 13]),
        "response_idx": 2,
        "changed_evidence_positions": [0],
        "block_definitions": [],
        "events": events,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    index_rows = []
    teachers = []
    for number in range(4):
        source_id = f"source-{number}"
        response_id = f"response-{number}"
        path = tmp_path / f"{response_id}.graph.pt"
        torch.save(_graph(source_id, response_id), path)
        index_rows.append(
            {
                "schema": "cept-canonical-graph-index-v2",
                "source_id": source_id,
                "response_id": response_id,
                "official_split": "train",
                "task_type": "QA",
                "generator_model": "llama-2-7b-chat",
                "graph_path": str(path),
                "graph_sha256": file_sha256(path),
                "model_source_signature": "sha256:source",
                "cache_model_source_signature": "sha256:source",
                "cache_model_identity_status": "verified_extractor_manifest",
                "cache_path": str(tmp_path / f"{response_id}.cache.pt"),
                "cache_sha256": "c" * 64,
                "extractor_declared_cache_sha256": "c" * 64,
                "cache_content_identity_status": (
                    "verified_extractor_cache_files_sha256"
                ),
                "cache_content_identity_evidence": (
                    "cache_files_sha256_exact_bytes"
                ),
                "cache_content_inventory_signature": "sha256:" + "d" * 64,
                "observer_runtime": OBSERVER_RUNTIME,
                "observer_runtime_signature": OBSERVER_RUNTIME_SIGNATURE,
                "deployment_seed_protocol": (
                    "numeric_digit_surface_preserving_v1_first_candidate"
                ),
                "deployment_seed_positions": [0],
            }
        )
        teachers.append(_teacher(source_id, response_id))
    index = tmp_path / "train.index.jsonl"
    teacher = tmp_path / "transport_teacher.jsonl"
    _write_jsonl(index, index_rows)
    _write_jsonl(teacher, teachers)
    return index, teacher


def test_transport_training_runs_on_cpu_and_writes_label_free_fidelity(tmp_path: Path):
    index, teacher = _fixture(tmp_path)

    result = train_transport(
        TransportTrainConfig(
            graph_index=index,
            teacher=teacher,
            output_dir=tmp_path / "run",
            device="cpu",
            epochs=2,
            learning_rate=0.01,
            validation_fraction=0.5,
            allow_unidentifiable_pilot=True,
            minimum_mechanism_sources=2,
            minimum_positive_sources=2,
            minimum_positive_events=2,
            seed=7,
            variants=("true",),
        )
    )

    assert result["state"] == "complete"
    assert result["labels_read_during_training_or_scoring"] is False
    assert result["split"]["pairwise_source_overlap"] == 0
    assert result["split"]["mechanism_holdout_sources"] == 2
    assert result["estimand_scope"]["training_cache"] == {
        "cache_inventory_samples": 4,
        "generator_models": {"llama-2-7b-chat": 4},
        "official_split": "train",
        "source_count": 4,
        "task_types": {"QA": 4},
    }
    assert result["estimand_scope"]["eligible_population"] == (
        "verified_attention_cache_inventory_intersect_"
        "numeric_digit_surface_preserving_identifiable_subset"
    )
    assert (tmp_path / "run" / "true.pt").is_file()
    assert (tmp_path / "run" / "teacher_fidelity.json").is_file()


def test_transport_rejects_unidentifiable_teacher_population_before_training(
    tmp_path: Path,
):
    index, teacher = _fixture(tmp_path)
    output = tmp_path / "run"

    with pytest.raises(
        RuntimeError,
        match="teacher population is not identifiable.*allow-unidentifiable-pilot",
    ):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=output,
                device="cpu",
                epochs=1,
                validation_fraction=0.5,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )

    assert not output.exists()


def test_graph_hash_or_identity_mismatch_fails_before_creating_run(tmp_path: Path):
    index, teacher = _fixture(tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[0]["graph_sha256"] = "0" * 64
    _write_jsonl(index, rows)
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="hash"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=output,
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )

    assert not output.exists()


def test_transport_rejects_existing_output_before_graph_hash_preflight(tmp_path: Path):
    index, teacher = _fixture(tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[0]["graph_sha256"] = "0" * 64
    _write_jsonl(index, rows)
    output = tmp_path / "run"
    output.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=output,
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )


def test_training_rejects_teacher_deployment_seed_mismatch(tmp_path: Path):
    index, teacher = _fixture(tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[0]["deployment_seed_positions"] = [1]
    _write_jsonl(index, rows)
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="seed operator disagree"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=output,
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )

    assert not output.exists()


def test_training_rejects_attention_from_transformers_446_for_teacher_452(
    tmp_path: Path,
):
    index, teacher = _fixture(tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    stale_runtime, stale_signature = observer_runtime_identity(
        transformers_version="4.46.3",
        torch_version="2.6.0+cu124",
        dtype="float16",
        attn_implementation="eager",
    )
    for row in rows:
        row["observer_runtime"] = stale_runtime
        row["observer_runtime_signature"] = stale_signature
    _write_jsonl(index, rows)
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="observer runtime mismatch"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=output,
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )

    assert not output.exists()


def test_training_rejects_legacy_graph_index_without_observer_runtime(
    tmp_path: Path,
):
    index, teacher = _fixture(tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[0].pop("observer_runtime")
    rows[0].pop("observer_runtime_signature")
    _write_jsonl(index, rows)

    with pytest.raises(ValueError, match="observer runtime identity"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=tmp_path / "run",
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )


def test_training_rejects_unverified_nonfirst_cache_model_identity(
    tmp_path: Path,
):
    index, teacher = _fixture(tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    rows[1]["cache_model_identity_status"] = "unverified_legacy_manifest"
    rows[1]["cache_model_source_signature"] = None
    _write_jsonl(index, rows)

    with pytest.raises(RuntimeError, match="unverified cache-model identity"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=tmp_path / "run",
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )


def test_scoring_rejects_observer_runtime_different_from_teacher_before_training(
    tmp_path: Path,
):
    index, teacher = _fixture(tmp_path)
    score_index = tmp_path / "test.index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    stale_runtime, stale_signature = observer_runtime_identity(
        transformers_version="4.46.3",
        torch_version="2.6.0+cu124",
        dtype="float16",
        attn_implementation="eager",
    )
    for row in rows:
        row["official_split"] = "test"
        row["observer_runtime"] = stale_runtime
        row["observer_runtime_signature"] = stale_signature
    _write_jsonl(score_index, rows)

    with pytest.raises(RuntimeError, match="score observer runtime mismatch"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                score_index=score_index,
                output_dir=tmp_path / "run",
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )


def test_training_and_scoring_require_official_train_test_boundaries(tmp_path: Path):
    index, teacher = _fixture(tmp_path)
    output = tmp_path / "run"

    with pytest.raises(ValueError, match="frozen scoring requires only official test"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                score_index=index,
                output_dir=output,
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )

    assert not output.exists()

    rows = [json.loads(line) for line in index.read_text().splitlines()]
    for row in rows:
        row["official_split"] = "test"
    _write_jsonl(index, rows)
    with pytest.raises(ValueError, match="training requires only official train"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                output_dir=output,
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )

    assert not output.exists()


def test_scoring_rejects_a_different_observer_model_before_training(tmp_path: Path):
    index, teacher = _fixture(tmp_path)
    score_index = tmp_path / "test.index.jsonl"
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    for row in rows:
        row["official_split"] = "test"
        row["model_source_signature"] = "sha256:other-source"
        row["cache_model_source_signature"] = "sha256:other-source"
    _write_jsonl(score_index, rows)
    output = tmp_path / "run"

    with pytest.raises(RuntimeError, match="different model sources"):
        train_transport(
            TransportTrainConfig(
                graph_index=index,
                teacher=teacher,
                score_index=score_index,
                output_dir=output,
                device="cpu",
                epochs=1,
                minimum_mechanism_sources=2,
                minimum_positive_sources=2,
                minimum_positive_events=2,
                variants=("true",),
            )
        )

    assert not output.exists()


def test_evaluation_cannot_open_labels_when_label_free_gate2_failed(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "cept-layer-provenance-transport-run-v2",
                "state": "complete",
                "gate2": {"claim_supported": False},
                "scoring": {"true": {}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="labels remain closed"):
        evaluate_frozen_transport(
            predictions_dir=run,
            test_graph_index=tmp_path / "never-opened.index.jsonl",
            output=run / "evaluation.json",
        )

    assert not (run / "evaluation.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("schema", "stale-schema"),
        ("state", "in_progress"),
        ("labels_read_during_training_or_scoring", True),
    ),
)
def test_evaluation_rejects_stale_incomplete_or_label_tainted_manifest(
    tmp_path: Path, field: str, value: object
):
    run = tmp_path / "run"
    run.mkdir()
    manifest = {
        "schema": "cept-layer-provenance-transport-run-v2",
        "state": "complete",
        "labels_read_during_training_or_scoring": False,
        "gate2": {"claim_supported": True},
        "scoring": {"true": {}},
    }
    manifest[field] = value
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="labels remain closed"):
        evaluate_frozen_transport(
            predictions_dir=run,
            test_graph_index=tmp_path / "never-opened.index.jsonl",
            output=run / "evaluation.json",
        )

    assert not (run / "evaluation.json").exists()


def test_corrupt_frozen_predictions_are_rejected_before_labels_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import attention_graph.evaluate as attention_evaluate

    index, _ = _fixture(tmp_path)
    rows = [json.loads(line) for line in index.read_text().splitlines()]
    for position, row in enumerate(rows):
        row["official_split"] = "test"
        cache = tmp_path / f"test-{position}.cache.pt"
        cache.write_bytes(b"frozen-cache")
        row["cache_path"] = str(cache)
        row["cache_sha256"] = file_sha256(cache)
    test_index = tmp_path / "test.index.jsonl"
    _write_jsonl(test_index, rows)

    run = tmp_path / "run"
    run.mkdir()
    token_predictions = run / "true.token_predictions.jsonl"
    response_predictions = run / "true.response_predictions.jsonl"
    token_predictions.write_text("{}\n", encoding="utf-8")
    response_predictions.write_text("{}\n", encoding="utf-8")
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "cept-layer-provenance-transport-run-v2",
                "state": "complete",
                "labels_read_during_training_or_scoring": False,
                "gate2": {"claim_supported": True},
                "scoring": {
                    "true": {
                        "score_index_sha256": file_sha256(test_index),
                        "token_predictions": str(token_predictions),
                        "response_predictions": str(response_predictions),
                        "variant": "true",
                        "tokens": 1,
                        "responses": 1,
                        "prediction_sha256": "corrupted",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    labels_opened = False

    def forbidden_label_read(paths):
        del paths
        nonlocal labels_opened
        labels_opened = True
        return object()

    monkeypatch.setattr(
        attention_evaluate, "load_evaluation_labels", forbidden_label_read
    )

    with pytest.raises(RuntimeError, match="prediction hash mismatch"):
        evaluate_frozen_transport(
            predictions_dir=run,
            test_graph_index=test_index,
            output=run / "evaluation.json",
        )

    assert labels_opened is False


def test_ungated_control_changes_only_relation_gates():
    trained = CausalProvenanceTransport(num_layers=2, num_heads=2)
    with torch.no_grad():
        trained.raw_relation_channel_gates.fill_(3.0)
        trained.raw_residual_persistence.copy_(torch.tensor([2.0, -1.0]))

    control = _ungated_control_from(trained)

    assert torch.allclose(control.gates(), torch.ones_like(control.gates()))
    assert torch.equal(
        control.raw_residual_persistence, trained.raw_residual_persistence
    )


def test_block_fidelity_uses_only_within_event_pair_ordering():
    targets = TeacherTargets(
        support=torch.zeros(2),
        history=torch.zeros(2),
        block=torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
        event_weight=torch.ones(2),
        positive_mask=torch.ones(2, dtype=torch.bool),
        null_mask=torch.zeros(2, dtype=torch.bool),
        contradictory_mask=torch.zeros(2, dtype=torch.bool),
        block_mask=torch.ones((2, 2), dtype=torch.bool),
        block_ids=("left", "right"),
    )
    reversed_prediction = torch.tensor([[0.9, 0.1], [0.8, 0.2]])

    outcomes, informative_events = _informative_block_pairs(
        targets,
        torch.ones(2, dtype=torch.bool),
        prediction=reversed_prediction,
    )

    assert informative_events == 2
    assert outcomes == [0.0, 0.0]


def _synthetic_claim_inputs(
    *, absolute_ci: float = 0.1, population_pass: bool = True
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    paired: dict[str, dict[str, object]] = {}
    for endpoint, controls in REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT.items():
        for control in controls:
            paired.setdefault(control, {})[endpoint] = {
                "auroc_delta": {"ci95_low": 0.1},
                "average_precision_delta": {"ci95_low": 0.1},
            }
    absolute = {
        endpoint: {
            "auroc_minus_0_5": {"ci95_low": absolute_ci},
            "average_precision_minus_prevalence": {"ci95_low": absolute_ci},
        }
        for endpoint in ("token", "response")
    }
    population = {
        endpoint: {"pass": population_pass}
        for endpoint in ("token", "response")
    }
    return paired, absolute, population


def test_detection_claim_rejects_below_random_score_even_if_controls_are_worse():
    claim = _detection_claim_report(*_synthetic_claim_inputs(absolute_ci=-0.1))

    assert claim["structure_superiority_supported"] == {
        "token": True,
        "response": True,
    }
    assert claim["absolute_detection_supported"] == {
        "token": False,
        "response": False,
    }
    assert claim["claim_supported"] is False


def test_detection_claim_rejects_too_small_source_or_class_population():
    claim = _detection_claim_report(*_synthetic_claim_inputs(population_pass=False))

    assert claim["claim_supported"] is False


def test_detection_claim_requires_and_accepts_relative_and_absolute_evidence():
    claim = _detection_claim_report(*_synthetic_claim_inputs())

    assert claim["token_supported"] is True
    assert claim["response_supported"] is True
    assert claim["claim_supported"] is True


def test_detection_claim_rejects_response_length_nuisance_failure_only_for_response():
    paired, absolute, population = _synthetic_claim_inputs()
    paired["response_length"]["response"]["auroc_delta"]["ci95_low"] = -0.01

    claim = _detection_claim_report(paired, absolute, population)

    assert claim["token_supported"] is True
    assert claim["response_supported"] is False
    assert claim["claim_supported"] is False


def test_detection_claim_reports_endpoint_specific_missing_control():
    paired, absolute, population = _synthetic_claim_inputs()
    del paired["response_length"]["response"]

    claim = _detection_claim_report(paired, absolute, population)

    assert claim["missing_controls"]["response"] == ["response_length"]
    assert claim["missing_controls"]["token"] == []
    assert claim["claim_supported"] is False


def test_detection_claim_requires_context_seed_and_coverage_nuisances():
    assert {
        "response_evidence_token_count",
        "negative_response_evidence_token_count",
        "response_query_token_count",
        "negative_response_query_token_count",
        "response_seed_count",
        "negative_response_seed_count",
        "response_unknown_mass",
        "negative_response_unknown_mass",
    }.issubset(REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT["response"])
    assert {
        "token_evidence_token_count",
        "negative_token_evidence_token_count",
        "token_query_token_count",
        "negative_token_query_token_count",
        "token_seed_count",
        "negative_token_seed_count",
        "token_response_unknown_mass",
        "negative_token_response_unknown_mass",
    }.issubset(REQUIRED_DETECTION_CONTROLS_BY_ENDPOINT["token"])


def test_response_unknown_mass_control_ignores_unavailable_prediction_rows():
    available = torch.tensor([False, True, True])

    first = _available_unknown_mass_mean(
        torch.tensor([1.0, 0.2, 0.4]), available
    )
    changed_unavailable = _available_unknown_mass_mean(
        torch.tensor([0.0, 0.2, 0.4]), available
    )

    assert first == pytest.approx(0.3)
    assert changed_unavailable == pytest.approx(first)
