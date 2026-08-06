"""Label-blind orchestration of the four-condition Gate-1 mediation pilot."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from .counterfactuals import CounterfactualAudit, validate_counterfactual_pair
from .mediation import (
    KVStore,
    MediationEffects,
    MediationRun,
    decompose_mediation_effects,
)


class MediationBackend(Protocol):
    def run(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        target_positions: torch.Tensor,
        capture_positions: torch.Tensor | None = None,
        sender: KVStore | None = None,
        patch_positions: torch.Tensor | None = None,
    ) -> MediationRun: ...


@dataclass(frozen=True)
class Gate1Pair:
    sample_id: str
    factual: Mapping[str, object]
    counterfactual: Mapping[str, object]
    rescue_blocks: Mapping[str, torch.Tensor] = field(default_factory=dict)
    source_id: str = ""
    task_type: str = ""
    generator_model: str = ""


@dataclass(frozen=True)
class Gate1RuntimeIdentity:
    model_signature: str
    tokenizer_signature: str
    transformers_version: str
    torch_version: str
    backend_id: str
    patch_site: str


@dataclass(frozen=True)
class TokenMediationEffect:
    target_position: int
    predictor_position: int
    target_token_id: int
    y11: float
    y00: float
    y10: float
    y01: float
    total: float
    direct: float
    mediated: float
    alternate: float
    interaction: float
    contract_residual: float
    direct_seed_rescue: float
    joint_seed_history_rescue: float
    representation_residual: float
    seed_history_interaction: float
    self_patch_error: float
    block_rescue: dict[str, float]

    @property
    def non_history_kv_effect(self) -> float:
        """Unambiguous artifact name for the legacy-short ``direct`` field."""

        return self.direct


@dataclass(frozen=True)
class Gate1Record:
    sample_id: str
    source_id: str
    task_type: str
    generator_model: str
    response_idx: int
    token_effects: tuple[TokenMediationEffect, ...]


@dataclass(frozen=True)
class Gate1Rejection:
    sample_id: str
    reason: str


@dataclass(frozen=True)
class Gate1Manifest:
    schema: str
    requested_samples: int
    completed_samples: int
    coverage: float
    rejections: tuple[Gate1Rejection, ...]
    model_signature: str
    tokenizer_signature: str
    transformers_version: str
    torch_version: str
    backend_id: str
    patch_site: str
    effect_definitions: dict[str, str]
    counterfactual_protocol: str
    self_patch_max_abs: float | None


@dataclass(frozen=True)
class Gate1PilotResult:
    records: tuple[Gate1Record, ...]
    manifest: Gate1Manifest


def _reject_labels(record: Mapping[str, object]) -> None:
    forbidden = [str(key) for key in record if "label" in str(key).casefold()]
    if forbidden:
        raise ValueError(
            "Gate-1 pilot must be label-blind; forbidden fields: "
            + ", ".join(sorted(forbidden))
        )


def _input_ids(record: Mapping[str, object], *, device: torch.device) -> torch.Tensor:
    ids = torch.as_tensor(record["input_ids"], dtype=torch.long, device=device)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("Gate-1 input_ids must describe one sample")
    return ids


def _backend_device(backend: MediationBackend) -> torch.device:
    model = getattr(backend, "model", None)
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return torch.device("cpu")
    first = next(parameters(), None)
    return torch.device("cpu") if first is None else first.device


def _report_condition(
    callback: Callable[[str, str], None] | None,
    *,
    sample_id: str,
    condition: str,
    run: MediationRun,
) -> None:
    if callback is None:
        return
    if run.target_log_probs.is_cuda:
        torch.cuda.synchronize(run.target_log_probs.device)
    callback(sample_id, condition)


def _scalar_rows(
    *,
    sample_id: str,
    source_id: str,
    task_type: str,
    generator_model: str,
    input_ids: torch.Tensor,
    target_positions: torch.Tensor,
    effects: MediationEffects,
    y11: torch.Tensor,
    y00: torch.Tensor,
    y10: torch.Tensor,
    y01: torch.Tensor,
    direct_seed_rescue: torch.Tensor,
    joint_seed_history_rescue: torch.Tensor,
    representation_residual: torch.Tensor,
    seed_history_interaction: torch.Tensor,
    self_patch_error: torch.Tensor,
    block_values: Mapping[str, torch.Tensor],
) -> Gate1Record:
    response_idx = int(target_positions[0])
    target_list = target_positions.detach().cpu().tolist()
    target_ids = (
        input_ids.detach()
        .cpu()[0]
        .index_select(0, target_positions.detach().cpu())
        .tolist()
    )
    scalar_vectors = {
        "y11": y11.detach().cpu()[0].tolist(),
        "y00": y00.detach().cpu()[0].tolist(),
        "y10": y10.detach().cpu()[0].tolist(),
        "y01": y01.detach().cpu()[0].tolist(),
        "total": effects.total.detach().cpu()[0].tolist(),
        "direct": effects.direct.detach().cpu()[0].tolist(),
        "mediated": effects.mediated.detach().cpu()[0].tolist(),
        "alternate": effects.alternate_mediated.detach().cpu()[0].tolist(),
        "interaction": effects.interaction.detach().cpu()[0].tolist(),
        "residual": effects.contract_residual.detach().cpu()[0].tolist(),
        "direct_seed_rescue": direct_seed_rescue.detach().cpu()[0].tolist(),
        "joint_seed_history_rescue": (
            joint_seed_history_rescue.detach().cpu()[0].tolist()
        ),
        "representation_residual": representation_residual.detach().cpu()[0].tolist(),
        "seed_history_interaction": (
            seed_history_interaction.detach().cpu()[0].tolist()
        ),
        "self_patch_error": self_patch_error.detach().cpu()[0].tolist(),
    }
    blocks = {
        name: values.detach().cpu()[0].tolist() for name, values in block_values.items()
    }
    rows: list[TokenMediationEffect] = []
    for offset, position in enumerate(target_list):
        rows.append(
            TokenMediationEffect(
                target_position=int(position),
                predictor_position=int(position) - 1,
                target_token_id=int(target_ids[offset]),
                y11=float(scalar_vectors["y11"][offset]),
                y00=float(scalar_vectors["y00"][offset]),
                y10=float(scalar_vectors["y10"][offset]),
                y01=float(scalar_vectors["y01"][offset]),
                total=float(scalar_vectors["total"][offset]),
                direct=float(scalar_vectors["direct"][offset]),
                mediated=float(scalar_vectors["mediated"][offset]),
                alternate=float(scalar_vectors["alternate"][offset]),
                interaction=float(scalar_vectors["interaction"][offset]),
                contract_residual=float(scalar_vectors["residual"][offset]),
                direct_seed_rescue=float(scalar_vectors["direct_seed_rescue"][offset]),
                joint_seed_history_rescue=float(
                    scalar_vectors["joint_seed_history_rescue"][offset]
                ),
                representation_residual=float(
                    scalar_vectors["representation_residual"][offset]
                ),
                seed_history_interaction=float(
                    scalar_vectors["seed_history_interaction"][offset]
                ),
                self_patch_error=float(scalar_vectors["self_patch_error"][offset]),
                block_rescue={
                    name: float(values[offset]) for name, values in blocks.items()
                },
            )
        )
    return Gate1Record(
        sample_id=sample_id,
        source_id=source_id,
        task_type=task_type,
        generator_model=generator_model,
        response_idx=response_idx,
        token_effects=tuple(rows),
    )


def run_gate1_pilot(
    pairs: Sequence[Gate1Pair],
    *,
    backend: MediationBackend,
    runtime: Gate1RuntimeIdentity,
    audit_self_patch: bool = False,
    progress: Callable[[int, int, str], None] | None = None,
    condition_progress: Callable[[str, str], None] | None = None,
) -> Gate1PilotResult:
    """Run natural/cross-patched conditions without reading any labels."""

    if not pairs:
        raise ValueError("Gate-1 pilot requires at least one pair")
    # Fail the entire batch before GPU work if a label leaked anywhere.  This
    # is stricter than treating a labeled row as an ordinary unavailable pair.
    for pair in pairs:
        if not pair.sample_id:
            raise ValueError("Gate-1 sample_id must be non-empty")
        _reject_labels(pair.factual)
        _reject_labels(pair.counterfactual)

    validated: list[tuple[Gate1Pair, CounterfactualAudit]] = []
    rejections: list[Gate1Rejection] = []
    for pair in pairs:
        try:
            audit = validate_counterfactual_pair(pair.factual, pair.counterfactual)
            if audit.response_token_ids.numel() < 2:
                raise ValueError(
                    "response has no non-empty history mediator; at least two tokens required"
                )
            history = torch.arange(
                audit.response_idx,
                audit.token_count - 1,
                dtype=torch.long,
            )
            for name, raw_positions in pair.rescue_blocks.items():
                if not name:
                    raise ValueError("rescue block names must be non-empty")
                positions = torch.as_tensor(raw_positions, dtype=torch.long).flatten()
                if positions.numel() == 0 or not bool(
                    torch.isin(positions, history).all()
                ):
                    raise ValueError(
                        f"rescue block {name!r} lies outside response history"
                    )
            validated.append((pair, audit))
        except (KeyError, TypeError, ValueError) as error:
            rejections.append(
                Gate1Rejection(sample_id=pair.sample_id, reason=str(error))
            )

    records: list[Gate1Record] = []
    self_patch_max_abs = 0.0 if audit_self_patch else None
    for pair, audit in validated:
        device = _backend_device(backend)
        factual_ids = _input_ids(pair.factual, device=device)
        counterfactual_ids = _input_ids(pair.counterfactual, device=device)
        attention_mask = torch.ones_like(factual_ids)
        target_positions = torch.arange(
            audit.response_idx, audit.token_count, dtype=torch.long
        )
        history_positions = torch.arange(
            audit.response_idx, audit.token_count - 1, dtype=torch.long
        )
        capture_positions = torch.unique(
            torch.cat([audit.changed_positions, history_positions]), sorted=True
        )

        y11 = backend.run(
            input_ids=factual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            capture_positions=capture_positions,
        )
        _report_condition(
            condition_progress,
            sample_id=pair.sample_id,
            condition="Y11",
            run=y11,
        )
        y00 = backend.run(
            input_ids=counterfactual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            capture_positions=capture_positions,
        )
        _report_condition(
            condition_progress,
            sample_id=pair.sample_id,
            condition="Y00",
            run=y00,
        )
        if y11.kv is None or y00.kv is None:
            raise RuntimeError("natural Gate-1 runs did not return complete K/V stores")
        y10 = backend.run(
            input_ids=factual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            sender=y00.kv,
            patch_positions=history_positions,
        )
        _report_condition(
            condition_progress,
            sample_id=pair.sample_id,
            condition="Y10",
            run=y10,
        )
        y01 = backend.run(
            input_ids=counterfactual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            sender=y11.kv,
            patch_positions=history_positions,
        )
        _report_condition(
            condition_progress,
            sample_id=pair.sample_id,
            condition="Y01",
            run=y01,
        )
        self_patch_error = torch.zeros_like(y11.target_log_probs)
        if audit_self_patch:
            factual_self = backend.run(
                input_ids=factual_ids,
                attention_mask=attention_mask,
                target_positions=target_positions,
                sender=y11.kv,
                patch_positions=history_positions,
            )
            _report_condition(
                condition_progress,
                sample_id=pair.sample_id,
                condition="self_patch_Y11",
                run=factual_self,
            )
            counterfactual_self = backend.run(
                input_ids=counterfactual_ids,
                attention_mask=attention_mask,
                target_positions=target_positions,
                sender=y00.kv,
                patch_positions=history_positions,
            )
            _report_condition(
                condition_progress,
                sample_id=pair.sample_id,
                condition="self_patch_Y00",
                run=counterfactual_self,
            )
            current_max = max(
                float(
                    (factual_self.target_log_probs - y11.target_log_probs).abs().max()
                ),
                float(
                    (counterfactual_self.target_log_probs - y00.target_log_probs)
                    .abs()
                    .max()
                ),
            )
            assert self_patch_max_abs is not None
            self_patch_max_abs = max(self_patch_max_abs, current_max)
            self_patch_error = torch.maximum(
                (factual_self.target_log_probs - y11.target_log_probs).abs(),
                (counterfactual_self.target_log_probs - y00.target_log_probs).abs(),
            )
            if current_max > 1e-5:
                raise RuntimeError(
                    "self-patch changed target log-probabilities beyond tolerance"
                )
        effects = decompose_mediation_effects(
            y11=y11.target_log_probs,
            y00=y00.target_log_probs,
            y10=y10.target_log_probs,
            y01=y01.target_log_probs,
        )
        if float(effects.contract_residual.abs().max()) > 1e-6:
            raise RuntimeError("TE = direct + mediated numerical contract failed")
        if (
            abs(float(effects.mediated[0, 0])) > 1e-6
            or abs(float(effects.alternate_mediated[0, 0])) > 1e-6
        ):
            raise RuntimeError(
                "first response token acquired an impossible history-mediated effect"
            )

        direct_seed = backend.run(
            input_ids=counterfactual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            sender=y11.kv,
            patch_positions=audit.changed_positions,
        )
        _report_condition(
            condition_progress,
            sample_id=pair.sample_id,
            condition="direct_seed_rescue",
            run=direct_seed,
        )
        direct_seed_rescue = direct_seed.target_log_probs - y00.target_log_probs

        joint_seed_history = backend.run(
            input_ids=counterfactual_ids,
            attention_mask=attention_mask,
            target_positions=target_positions,
            sender=y11.kv,
            patch_positions=capture_positions,
        )
        _report_condition(
            condition_progress,
            sample_id=pair.sample_id,
            condition="joint_seed_history_rescue",
            run=joint_seed_history,
        )
        joint_seed_history_rescue = (
            joint_seed_history.target_log_probs - y00.target_log_probs
        )
        representation_residual = effects.total - joint_seed_history_rescue
        seed_history_interaction = (
            joint_seed_history_rescue - direct_seed_rescue - effects.alternate_mediated
        )

        block_values: dict[str, torch.Tensor] = {}
        for name, raw_positions in pair.rescue_blocks.items():
            positions = torch.as_tensor(raw_positions, dtype=torch.long).flatten()
            rescued = backend.run(
                input_ids=counterfactual_ids,
                attention_mask=attention_mask,
                target_positions=target_positions,
                sender=y11.kv,
                patch_positions=positions,
            )
            _report_condition(
                condition_progress,
                sample_id=pair.sample_id,
                condition=f"rescue:{name}",
                run=rescued,
            )
            block_values[name] = rescued.target_log_probs - y00.target_log_probs
        records.append(
            _scalar_rows(
                sample_id=pair.sample_id,
                source_id=pair.source_id,
                task_type=pair.task_type,
                generator_model=pair.generator_model,
                input_ids=factual_ids,
                target_positions=target_positions,
                effects=effects,
                y11=y11.target_log_probs,
                y00=y00.target_log_probs,
                y10=y10.target_log_probs,
                y01=y01.target_log_probs,
                direct_seed_rescue=direct_seed_rescue,
                joint_seed_history_rescue=joint_seed_history_rescue,
                representation_residual=representation_residual,
                seed_history_interaction=seed_history_interaction,
                self_patch_error=self_patch_error,
                block_values=block_values,
            )
        )
        if progress is not None:
            progress(len(records), len(validated), pair.sample_id)

    requested = len(pairs)
    completed = len(records)
    manifest = Gate1Manifest(
        schema="cept-gate1-mediation-manifest-v1",
        requested_samples=requested,
        completed_samples=completed,
        coverage=completed / requested,
        rejections=tuple(rejections),
        model_signature=runtime.model_signature,
        tokenizer_signature=runtime.tokenizer_signature,
        transformers_version=runtime.transformers_version,
        torch_version=runtime.torch_version,
        backend_id=runtime.backend_id,
        patch_site=runtime.patch_site,
        effect_definitions={
            "total": "Y11 - Y00: operational total source intervention effect",
            "direct": (
                "Y10 - Y00: operational non-history-K/V effect; not a natural "
                "direct causal effect"
            ),
            "mediated": "Y11 - Y10: response-history K/V-mediated effect",
            "alternate": "Y01 - Y00: reverse-direction mediation check",
            "interaction": "mediated - alternate: direction sensitivity check",
            "direct_seed_rescue": (
                "Y(E0, patch changed evidence K/V from E1) - Y00: total recursive "
                "support initiated by the changed-evidence seed"
            ),
            "joint_seed_history_rescue": (
                "Y(E0, patch changed-evidence and response-history K/V from E1) "
                "- Y00: graph-variable representable joint rescue"
            ),
            "representation_residual": ("total effect - joint seed/history rescue"),
            "seed_history_interaction": (
                "joint rescue - direct-seed rescue - alternate history rescue"
            ),
        },
        counterfactual_protocol="numeric_digit_surface_preserving_v1",
        self_patch_max_abs=self_patch_max_abs,
    )
    return Gate1PilotResult(records=tuple(records), manifest=manifest)


__all__ = [
    "Gate1Manifest",
    "Gate1Pair",
    "Gate1PilotResult",
    "Gate1Record",
    "Gate1RuntimeIdentity",
    "TokenMediationEffect",
    "run_gate1_pilot",
]
