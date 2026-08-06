from __future__ import annotations

import pytest
import torch

from counterfactual_grounding.transport.data import (
    EffectCalibration,
    TeacherTargets,
    TransportEventTarget,
    TransportTeacher,
    build_teacher_targets,
)
from counterfactual_grounding.transport.model import TransportOutput, transport_loss


def _teacher() -> TransportTeacher:
    return TransportTeacher(
        response_id="response-1",
        source_id="source-1",
        official_split="train",
        model_signature="sha256:model",
        model_source_signature="sha256:source",
        tokenizer_signature="sha256:tokenizer",
        observer_runtime={
            "schema": "cept-observer-runtime-v1",
            "transformers_version": "4.52.3",
            "torch_version": "2.6.0+cu124",
            "dtype": "float16",
            "attn_implementation": "eager",
        },
        observer_runtime_signature=(
            "sha256:1e64b8793e952051f8f6e5ca931a17b24149e73858d758c9b251f60d51b1a9cb"
        ),
        factual_input_ids_sha256="sha256:input",
        response_idx=3,
        changed_evidence_positions=(1,),
        block_positions={"past": (3,), "future": (5,)},
        events=(
            TransportEventTarget(
                target_position=4,
                predictor_position=3,
                target_token_id=14,
                total_effect=0.5,
                direct_effect=0.3,
                history_mediated_effect=0.2,
                alternate_history_effect=0.1,
                interaction=0.1,
                decomposition_residual=0.0,
                direct_seed_rescue=0.3,
                joint_seed_history_rescue=0.5,
                representation_residual=0.0,
                seed_history_interaction=0.0,
                self_patch_error=0.0,
                block_rescue={"past": 0.2, "future": 0.0},
            ),
        ),
    )


def test_targets_use_effect_magnitudes_and_mask_future_history_blocks():
    targets = build_teacher_targets(
        _teacher(),
        calibration=EffectCalibration(
            null_floor=0.1,
            support_scale=0.5,
            history_scale=0.5,
            block_scale=0.2,
        ),
    )

    assert targets.support.tolist() == pytest.approx([0.5])
    assert targets.history.tolist() == pytest.approx([0.0])
    assert targets.block_mask[:, 0].tolist() == [True, False]
    assert targets.block[:, 0].tolist() == pytest.approx([1.0, 0.0])


def test_loss_is_normalized_within_each_graph_and_masks_unavailable_rows():
    targets = build_teacher_targets(
        _teacher(),
        calibration=EffectCalibration(
            null_floor=0.1,
            support_scale=0.5,
            history_scale=0.5,
            block_scale=0.2,
        ),
    )
    output = TransportOutput(
        direct=torch.tensor([0.0]),
        history_lower=torch.tensor([0.0]),
        history_upper=torch.tensor([0.0]),
        support_lower=torch.tensor([0.0]),
        support_upper=torch.tensor([0.0]),
        block=torch.zeros((2, 1)),
        row_available=torch.tensor([False]),
        unsupported_history_lower=torch.tensor([0.0]),
        unsupported_history_upper=torch.tensor([1.0]),
    )

    loss = transport_loss(output, targets)

    assert loss.item() == pytest.approx(0.0)


def test_zero_or_negative_total_effect_is_not_mislabeled_as_support():
    teacher = _teacher()
    cancelled = teacher.events[0]
    cancelled = type(cancelled)(
        **{
            **cancelled.__dict__,
            "total_effect": 0.0,
            "direct_seed_rescue": 1.0,
            "history_mediated_effect": -1.0,
        }
    )
    teacher = type(teacher)(**{**teacher.__dict__, "events": (cancelled,)})

    targets = build_teacher_targets(
        teacher,
        calibration=EffectCalibration(
            null_floor=0.1,
            support_scale=0.5,
            history_scale=0.5,
            block_scale=0.2,
        ),
    )

    assert targets.support.item() == pytest.approx(0.0)
    assert targets.history.item() == pytest.approx(0.0)
    assert targets.event_weight.item() > 0.0
    assert not targets.positive_mask.item()
    assert targets.null_mask.item()
    assert not targets.block_mask.any()


def test_event_loss_balances_reliable_positive_and_null_populations():
    count = 102
    support_target = torch.zeros(count)
    support_target[0] = 1.0
    support_target[-1] = 1.0
    positive_mask = torch.zeros(count, dtype=torch.bool)
    positive_mask[0] = True
    null_mask = torch.zeros(count, dtype=torch.bool)
    null_mask[1:-1] = True
    targets = TeacherTargets(
        support=support_target,
        history=torch.zeros(count),
        block=torch.zeros((0, count)),
        event_weight=torch.ones(count),
        positive_mask=positive_mask,
        null_mask=null_mask,
        contradictory_mask=torch.zeros(count, dtype=torch.bool),
        block_mask=torch.zeros((0, count), dtype=torch.bool),
        block_ids=(),
    )
    output = TransportOutput(
        direct=torch.zeros(count),
        history_lower=torch.zeros(count),
        history_upper=torch.zeros(count),
        support_lower=torch.zeros(count),
        support_upper=torch.zeros(count),
        block=torch.zeros((0, count)),
        row_available=torch.ones(count, dtype=torch.bool),
        unsupported_history_lower=torch.zeros(count),
        unsupported_history_upper=torch.ones(count),
    )

    loss = transport_loss(output, targets)

    # The one reliable positive contributes SmoothL1(0, 1)=0.5; the 100 nulls
    # contribute zero. Equal population weighting gives (0.5 + 0.0) / 2.
    # The final neither-positive-nor-null event is intentionally excluded.
    assert loss.item() == pytest.approx(0.25)
