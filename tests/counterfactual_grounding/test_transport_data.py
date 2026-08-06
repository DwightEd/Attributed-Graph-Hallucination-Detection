from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from counterfactual_grounding.teacher.pilot import (
    Gate1Pair,
    Gate1Record,
    TokenMediationEffect,
)
from counterfactual_grounding.transport.data import (
    EffectCalibration,
    align_teacher_to_graph,
    build_teacher_targets,
    fit_effect_calibration,
    make_transport_teacher_record,
    parse_transport_teacher,
)

OBSERVER_RUNTIME = {
    "schema": "cept-observer-runtime-v1",
    "transformers_version": "4.52.3",
    "torch_version": "2.6.0+cu124",
    "dtype": "float16",
    "attn_implementation": "eager",
}
OBSERVER_RUNTIME_SIGNATURE = (
    "sha256:1e64b8793e952051f8f6e5ca931a17b24149e73858d758c9b251f60d51b1a9cb"
)


def _effect(position: int, token_id: int) -> TokenMediationEffect:
    return TokenMediationEffect(
        target_position=position,
        predictor_position=position - 1,
        target_token_id=token_id,
        y11=-0.2,
        y00=-0.7,
        y10=-0.4,
        y01=-0.6,
        total=0.5,
        direct=0.3,
        mediated=0.2,
        alternate=0.1,
        interaction=0.1,
        contract_residual=0.0,
        direct_seed_rescue=0.25,
        joint_seed_history_rescue=0.35,
        representation_residual=0.15,
        seed_history_interaction=0.0,
        self_patch_error=1e-6,
        block_rescue={"history-0": 0.15},
    )


def _teacher_inputs() -> tuple[Gate1Record, Gate1Pair]:
    record = Gate1Record(
        sample_id="response-1",
        source_id="source-1",
        task_type="QA",
        generator_model="model",
        response_idx=3,
        token_effects=(_effect(3, 13), _effect(4, 14)),
    )
    pair = Gate1Pair(
        sample_id="response-1",
        source_id="source-1",
        task_type="QA",
        generator_model="model",
        factual={
            "input_ids": torch.tensor([10, 11, 12, 13, 14]),
            "response_idx": 3,
            "evidence_token_positions": torch.tensor([0, 1]),
        },
        counterfactual={
            "input_ids": torch.tensor([10, 99, 12, 13, 14]),
            "response_idx": 3,
            "evidence_token_positions": torch.tensor([0, 1]),
        },
        rescue_blocks={"history-0": torch.tensor([3])},
    )
    return record, pair


def _graph() -> dict[str, object]:
    return {
        "schema": "cept-prediction-event-graph-v2",
        "source_id": "source-1",
        "response_id": "response-1",
        "dataset_split": "train",
        "token_ids": torch.tensor([10, 11, 12, 13, 14]),
        "target_token_ids": torch.tensor([13, 14]),
        "target_token_positions": torch.tensor([3, 4]),
        "predictor_positions": torch.tensor([2, 3]),
        "row_available": torch.tensor([False, True]),
    }


def test_structured_teacher_keeps_intervention_and_history_block_positions():
    record, pair = _teacher_inputs()

    payload = make_transport_teacher_record(
        record,
        pair,
        changed_evidence_positions=(1,),
        official_split="train",
        model_signature="sha256:model",
        model_source_signature="sha256:source",
        tokenizer_signature="sha256:tokenizer",
        observer_runtime=OBSERVER_RUNTIME,
        observer_runtime_signature=OBSERVER_RUNTIME_SIGNATURE,
    )
    teacher = parse_transport_teacher(payload)

    assert teacher.changed_evidence_positions == (1,)
    assert teacher.block_positions == {"history-0": (3,)}
    assert teacher.observer_runtime == OBSERVER_RUNTIME
    assert teacher.observer_runtime_signature == OBSERVER_RUNTIME_SIGNATURE
    assert teacher.events[1].history_mediated_effect == pytest.approx(0.2)
    assert teacher.events[1].direct_seed_rescue == pytest.approx(0.25)
    assert teacher.events[1].joint_seed_history_rescue == pytest.approx(0.35)
    assert "label" not in str(payload).casefold()
    align_teacher_to_graph(teacher, _graph())


def test_teacher_graph_join_fails_closed_on_prediction_event_misalignment():
    record, pair = _teacher_inputs()
    teacher = parse_transport_teacher(
        make_transport_teacher_record(
            record,
            pair,
            changed_evidence_positions=(1,),
            official_split="train",
            model_signature="sha256:model",
            model_source_signature="sha256:source",
            tokenizer_signature="sha256:tokenizer",
            observer_runtime=OBSERVER_RUNTIME,
            observer_runtime_signature=OBSERVER_RUNTIME_SIGNATURE,
        )
    )
    bad_event = replace(teacher.events[1], predictor_position=99)
    bad_teacher = replace(teacher, events=(teacher.events[0], bad_event))

    with pytest.raises(ValueError, match="predictor"):
        align_teacher_to_graph(bad_teacher, _graph())


def test_teacher_parser_rejects_any_hallucination_label_leakage():
    record, pair = _teacher_inputs()
    payload = make_transport_teacher_record(
        record,
        pair,
        changed_evidence_positions=(1,),
        official_split="train",
        model_signature="sha256:model",
        model_source_signature="sha256:source",
        tokenizer_signature="sha256:tokenizer",
        observer_runtime=OBSERVER_RUNTIME,
        observer_runtime_signature=OBSERVER_RUNTIME_SIGNATURE,
    )
    payload["hallucination_label"] = 1

    with pytest.raises(ValueError, match="label"):
        parse_transport_teacher(payload)


def test_teacher_parser_rejects_legacy_row_without_observer_runtime():
    record, pair = _teacher_inputs()
    payload = make_transport_teacher_record(
        record,
        pair,
        changed_evidence_positions=(1,),
        official_split="train",
        model_signature="sha256:model",
        model_source_signature="sha256:source",
        tokenizer_signature="sha256:tokenizer",
        observer_runtime=OBSERVER_RUNTIME,
        observer_runtime_signature=OBSERVER_RUNTIME_SIGNATURE,
    )
    payload.pop("observer_runtime")
    payload.pop("observer_runtime_signature")

    with pytest.raises((TypeError, ValueError), match="observer_runtime"):
        parse_transport_teacher(payload)


def test_teacher_parser_recomputes_intervention_algebra_instead_of_trusting_residuals():
    record, pair = _teacher_inputs()
    payload = make_transport_teacher_record(
        record,
        pair,
        changed_evidence_positions=(1,),
        official_split="train",
        model_signature="sha256:model",
        model_source_signature="sha256:source",
        tokenizer_signature="sha256:tokenizer",
        observer_runtime=OBSERVER_RUNTIME,
        observer_runtime_signature=OBSERVER_RUNTIME_SIGNATURE,
    )
    payload["events"][0]["representation_residual"] = 999.0

    with pytest.raises(ValueError, match="algebraic identity"):
        parse_transport_teacher(payload)


def test_direction_reversing_total_effect_is_not_mislabeled_as_a_null_target():
    record, pair = _teacher_inputs()
    teacher = parse_transport_teacher(
        make_transport_teacher_record(
            record,
            pair,
            changed_evidence_positions=(1,),
            official_split="train",
            model_signature="sha256:model",
            model_source_signature="sha256:source",
            tokenizer_signature="sha256:tokenizer",
            observer_runtime=OBSERVER_RUNTIME,
            observer_runtime_signature=OBSERVER_RUNTIME_SIGNATURE,
        )
    )
    negative = replace(teacher.events[0], total_effect=-0.5)
    teacher = replace(teacher, events=(negative, teacher.events[1]))

    targets = build_teacher_targets(
        teacher,
        calibration=EffectCalibration(
            null_floor=0.01,
            support_scale=1.0,
            history_scale=1.0,
            block_scale=1.0,
        ),
    )

    assert targets.contradictory_mask.tolist() == [True, False]
    assert targets.null_mask.tolist() == [False, False]
    assert targets.event_weight[0].item() == 0.0


def test_effect_calibration_ignores_events_without_an_attention_row():
    record, pair = _teacher_inputs()
    teacher = parse_transport_teacher(
        make_transport_teacher_record(
            record,
            pair,
            changed_evidence_positions=(1,),
            official_split="train",
            model_signature="sha256:model",
            model_source_signature="sha256:source",
            tokenizer_signature="sha256:tokenizer",
            observer_runtime=OBSERVER_RUNTIME,
            observer_runtime_signature=OBSERVER_RUNTIME_SIGNATURE,
        )
    )
    unavailable_outlier = replace(
        teacher.events[0],
        total_effect=1_000_000.0,
        direct_seed_rescue=900_000.0,
        joint_seed_history_rescue=950_000.0,
        alternate_history_effect=800_000.0,
        representation_residual=0.0,
        seed_history_interaction=0.0,
        self_patch_error=100_000.0,
        block_rescue={"history-0": 700_000.0},
    )
    teacher = replace(teacher, events=(unavailable_outlier, teacher.events[1]))

    calibration = fit_effect_calibration(
        [teacher],
        availability_by_identity={("source-1", "response-1"): (False, True)},
        minimum_null_floor=0.01,
    )

    assert calibration.null_floor == pytest.approx(0.01)
    assert calibration.support_scale == pytest.approx(0.25)
    assert calibration.history_scale == pytest.approx(0.1)
    assert calibration.block_scale == pytest.approx(0.15)
