"""Gate 1 contract tests for the label-blind mediation pilot orchestrator.

These tests use a deterministic backend fake.  They freeze orchestration and
artifact semantics without loading a tokenizer or a language model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest
import torch

from counterfactual_grounding.teacher.mediation import KVStore, MediationRun
from counterfactual_grounding.teacher.pilot import (
    Gate1Pair,
    Gate1RuntimeIdentity,
    run_gate1_pilot,
)

FACTUAL_IDS = torch.tensor([10, 11, 20, 21, 22], dtype=torch.long)
COUNTERFACTUAL_IDS = torch.tensor([12, 11, 20, 21, 22], dtype=torch.long)


class FakeMediationBackend:
    """Return auditable outcomes for natural, cross-patch, and block runs."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    @staticmethod
    def _store(tag: int, positions: torch.Tensor) -> KVStore:
        values = torch.full(
            (1, 1, positions.numel(), 1), float(tag), dtype=torch.float32
        )
        return KVStore(
            positions=positions.detach().cpu().clone(),
            keys={0: values.clone()},
            values={0: values.clone()},
        )

    def run(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_positions: torch.Tensor,
        capture_positions: torch.Tensor | None = None,
        sender: KVStore | None = None,
        patch_positions: torch.Tensor | None = None,
    ) -> MediationRun:
        del attention_mask
        receiver = (
            "factual" if torch.equal(input_ids[0], FACTUAL_IDS) else "counterfactual"
        )
        targets = target_positions.detach().cpu().clone()

        if capture_positions is not None:
            assert sender is None and patch_positions is None
            capture = capture_positions.detach().cpu().clone()
            tag = 1 if receiver == "factual" else 0
            condition = "Y11" if receiver == "factual" else "Y00"
            scores = (
                torch.tensor([[1.0, 3.0, 5.0]])
                if receiver == "factual"
                else torch.tensor([[0.5, 1.0, 2.0]])
            )
            self.calls.append(
                {
                    "condition": condition,
                    "receiver": receiver,
                    "sender": None,
                    "targets": targets,
                    "capture": capture,
                    "patch": None,
                }
            )
            return MediationRun(
                target_log_probs=scores,
                kv=self._store(tag, capture),
            )

        assert sender is not None and patch_positions is not None
        patch = patch_positions.detach().cpu().clone()
        sender_tag = int(sender.keys[0][0, 0, 0, 0].item())
        sender_name = "factual" if sender_tag == 1 else "counterfactual"
        full_history = torch.equal(patch, torch.tensor([2, 3]))
        joint_seed_history = torch.equal(patch, torch.tensor([0, 2, 3]))
        if receiver == "factual" and sender_name == "counterfactual" and full_history:
            condition = "Y10"
            scores = torch.tensor([[1.0, 2.0, 4.0]])
        elif receiver == "counterfactual" and sender_name == "factual" and full_history:
            condition = "Y01"
            scores = torch.tensor([[0.5, 1.4, 3.5]])
        elif (
            receiver == "counterfactual"
            and sender_name == "factual"
            and joint_seed_history
        ):
            condition = "joint"
            scores = torch.tensor([[0.8, 2.0, 4.0]])
        elif receiver == "counterfactual" and sender_name == "factual":
            condition = "block"
            scores = torch.tensor([[0.5, 1.2, 2.8]])
        else:  # pragma: no cover - a useful diagnostic for a broken orchestrator
            raise AssertionError(
                f"unexpected mediation run: receiver={receiver}, sender={sender_name}, "
                f"patch={patch.tolist()}"
            )
        self.calls.append(
            {
                "condition": condition,
                "receiver": receiver,
                "sender": sender_name,
                "targets": targets,
                "capture": None,
                "patch": patch,
            }
        )
        return MediationRun(target_log_probs=scores, kv=None)


class SelfPatchFakeMediationBackend(FakeMediationBackend):
    def run(self, **kwargs: Any) -> MediationRun:
        sender = kwargs.get("sender")
        input_ids = kwargs["input_ids"]
        patch_positions = kwargs.get("patch_positions")
        if sender is not None and patch_positions is not None:
            receiver_is_factual = torch.equal(input_ids[0], FACTUAL_IDS)
            sender_is_factual = int(sender.keys[0][0, 0, 0, 0].item()) == 1
            if receiver_is_factual == sender_is_factual:
                condition = (
                    "self-factual" if receiver_is_factual else "self-counterfactual"
                )
                scores = (
                    torch.tensor([[1.0, 3.0, 5.0]])
                    if receiver_is_factual
                    else torch.tensor([[0.5, 1.0, 2.0]])
                )
                self.calls.append({"condition": condition})
                return MediationRun(target_log_probs=scores, kv=None)
        return super().run(**kwargs)


def _record(input_ids: torch.Tensor) -> dict[str, object]:
    return {
        "input_ids": input_ids.clone(),
        "response_idx": 2,
        "evidence_token_positions": torch.tensor([0], dtype=torch.long),
    }


def _valid_pair(*, with_block: bool) -> Gate1Pair:
    return Gate1Pair(
        sample_id="valid-1",
        factual=_record(FACTUAL_IDS),
        counterfactual=_record(COUNTERFACTUAL_IDS),
        rescue_blocks=(
            {"evidence-chunk-0": torch.tensor([2], dtype=torch.long)}
            if with_block
            else {}
        ),
    )


def _runtime() -> Gate1RuntimeIdentity:
    return Gate1RuntimeIdentity(
        model_signature="observer-llama@sha256:abc123",
        tokenizer_signature="observer-tokenizer@sha256:def456",
        transformers_version="4.52.3",
        torch_version="2.6.0+cu124",
        backend_id="cept_mediation_eager",
        patch_site="post_rope_kv_pre_repeat_kv",
    )


def test_pilot_runs_four_conditions_cross_patches_history_and_emits_token_effects():
    backend = FakeMediationBackend()
    bad_counterfactual = _record(COUNTERFACTUAL_IDS)
    bad_counterfactual["input_ids"] = torch.tensor(
        [10, 99, 20, 21, 22], dtype=torch.long
    )
    invalid_pair = Gate1Pair(
        sample_id="invalid-query-edit",
        factual=_record(FACTUAL_IDS),
        counterfactual=bad_counterfactual,
    )

    result = run_gate1_pilot(
        [_valid_pair(with_block=True), invalid_pair],
        backend=backend,
        runtime=_runtime(),
    )

    assert [call["condition"] for call in backend.calls] == [
        "Y11",
        "Y00",
        "Y10",
        "Y01",
        "block",
        "joint",
        "block",
    ]
    y10 = next(call for call in backend.calls if call["condition"] == "Y10")
    y01 = next(call for call in backend.calls if call["condition"] == "Y01")
    assert (y10["receiver"], y10["sender"]) == ("factual", "counterfactual")
    assert (y01["receiver"], y01["sender"]) == ("counterfactual", "factual")
    torch.testing.assert_close(y10["patch"], torch.tensor([2, 3]))
    torch.testing.assert_close(y01["patch"], torch.tensor([2, 3]))
    direct_seed = next(
        call
        for call in backend.calls
        if call["condition"] == "block" and call["patch"].tolist() == [0]
    )
    block = next(
        call
        for call in backend.calls
        if call["condition"] == "block" and call["patch"].tolist() == [2]
    )
    torch.testing.assert_close(direct_seed["patch"], torch.tensor([0]))
    joint = next(call for call in backend.calls if call["condition"] == "joint")
    torch.testing.assert_close(joint["patch"], torch.tensor([0, 2, 3]))
    assert (block["receiver"], block["sender"]) == ("counterfactual", "factual")
    torch.testing.assert_close(block["patch"], torch.tensor([2]))

    assert len(result.records) == 1
    token_rows = result.records[0].token_effects
    assert [row.target_position for row in token_rows] == [2, 3, 4]
    assert [row.predictor_position for row in token_rows] == [1, 2, 3]
    assert [row.target_token_id for row in token_rows] == [20, 21, 22]
    assert [row.y11 for row in token_rows] == pytest.approx([1.0, 3.0, 5.0])
    assert [row.y00 for row in token_rows] == pytest.approx([0.5, 1.0, 2.0])
    assert [row.y10 for row in token_rows] == pytest.approx([1.0, 2.0, 4.0])
    assert [row.y01 for row in token_rows] == pytest.approx([0.5, 1.4, 3.5])
    assert [row.total for row in token_rows] == pytest.approx([0.5, 2.0, 3.0])
    assert [row.direct for row in token_rows] == pytest.approx([0.5, 1.0, 2.0])
    assert [row.mediated for row in token_rows] == pytest.approx([0.0, 1.0, 1.0])
    assert [row.alternate for row in token_rows] == pytest.approx([0.0, 0.4, 1.5])
    assert [row.interaction for row in token_rows] == pytest.approx([0.0, 0.6, -0.5])
    assert [row.contract_residual for row in token_rows] == pytest.approx(
        [0.0, 0.0, 0.0], abs=1e-7
    )
    assert [row.direct_seed_rescue for row in token_rows] == pytest.approx(
        [0.0, 0.2, 0.8]
    )
    assert [row.joint_seed_history_rescue for row in token_rows] == pytest.approx(
        [0.3, 1.0, 2.0]
    )
    assert [row.representation_residual for row in token_rows] == pytest.approx(
        [0.2, 1.0, 1.0]
    )
    assert [row.seed_history_interaction for row in token_rows] == pytest.approx(
        [0.3, 0.4, -0.3]
    )
    assert [
        row.block_rescue["evidence-chunk-0"] for row in token_rows
    ] == pytest.approx([0.0, 0.2, 0.8])
    # The first response event has no response-history token to mediate it.
    assert token_rows[0].mediated == pytest.approx(0.0, abs=1e-7)

    manifest = result.manifest
    assert manifest.requested_samples == 2
    assert manifest.completed_samples == 1
    assert manifest.coverage == pytest.approx(0.5)
    assert len(manifest.rejections) == 1
    assert manifest.rejections[0].sample_id == "invalid-query-edit"
    assert "evidence" in manifest.rejections[0].reason.casefold()
    assert manifest.model_signature == "observer-llama@sha256:abc123"
    assert manifest.tokenizer_signature == "observer-tokenizer@sha256:def456"
    assert manifest.transformers_version == "4.52.3"
    assert manifest.torch_version == "2.6.0+cu124"
    assert manifest.backend_id == "cept_mediation_eager"
    assert manifest.patch_site == "post_rope_kv_pre_repeat_kv"
    direct_definition = manifest.effect_definitions["direct"].casefold()
    assert "operational" in direct_definition
    assert "non-history" in direct_definition


def test_history_rescue_block_is_optional_but_direct_seed_rescue_is_required():
    backend = FakeMediationBackend()

    result = run_gate1_pilot(
        [_valid_pair(with_block=False)],
        backend=backend,
        runtime=_runtime(),
    )

    assert [call["condition"] for call in backend.calls] == [
        "Y11",
        "Y00",
        "Y10",
        "Y01",
        "block",
        "joint",
    ]
    assert all(not row.block_rescue for row in result.records[0].token_effects)


def test_any_label_field_aborts_before_the_backend_observes_a_sample():
    backend = FakeMediationBackend()
    labeled = replace(
        _valid_pair(with_block=False),
        factual={**_record(FACTUAL_IDS), "hallucination_label": 1},
    )

    with pytest.raises(ValueError, match="label|label-blind|forbidden"):
        run_gate1_pilot(
            [_valid_pair(with_block=False), labeled],
            backend=backend,
            runtime=_runtime(),
        )

    assert backend.calls == []


def test_optional_real_runtime_self_patch_audit_is_recorded():
    backend = SelfPatchFakeMediationBackend()

    result = run_gate1_pilot(
        [_valid_pair(with_block=False)],
        backend=backend,
        runtime=_runtime(),
        audit_self_patch=True,
    )

    assert [call["condition"] for call in backend.calls] == [
        "Y11",
        "Y00",
        "Y10",
        "Y01",
        "self-factual",
        "self-counterfactual",
        "block",
        "joint",
    ]
    assert result.manifest.self_patch_max_abs == pytest.approx(0.0)


def test_condition_progress_reports_each_completed_condition_without_changing_sample_progress():
    backend = SelfPatchFakeMediationBackend()
    completed_conditions: list[tuple[str, str]] = []
    completed_samples: list[tuple[int, int, str]] = []

    run_gate1_pilot(
        [_valid_pair(with_block=False)],
        backend=backend,
        runtime=_runtime(),
        audit_self_patch=True,
        condition_progress=lambda sample_id, condition: completed_conditions.append(
            (sample_id, condition)
        ),
        progress=lambda current, total, sample_id: completed_samples.append(
            (current, total, sample_id)
        ),
    )

    assert completed_conditions == [
        ("valid-1", "Y11"),
        ("valid-1", "Y00"),
        ("valid-1", "Y10"),
        ("valid-1", "Y01"),
        ("valid-1", "self_patch_Y11"),
        ("valid-1", "self_patch_Y00"),
        ("valid-1", "direct_seed_rescue"),
        ("valid-1", "joint_seed_history_rescue"),
    ]
    assert completed_samples == [(1, 1, "valid-1")]
