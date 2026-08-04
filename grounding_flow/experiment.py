"""Label-free fitting and scoring over null-calibrated flow trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import torch

from .method import CalibratedFlow
from .state_model import (
    TrajectoryProjector,
    TwoStateGaussianHMM,
    fit_trajectory_projector,
    fit_two_state_hmm,
)


@dataclass(frozen=True)
class TrajectoryRecord:
    response_id: str
    pair_id: str
    partition: str
    response_token_indices: np.ndarray
    model_surface: np.ndarray
    mechanism_anchor: np.ndarray
    raw_token_summary: Mapping[str, np.ndarray]
    null_swap_fraction: float
    null_effective_fraction: float = 0.0
    null_stable_model_fraction: float = 0.0
    null_samples: int = 0
    duplicate_null_draws: int = 0
    calibration_status: str = "complete"

    def __post_init__(self) -> None:
        if not self.response_id.strip() or not self.pair_id.strip():
            raise ValueError("trajectory records require response and pair identity")
        if self.partition not in {"train", "validation", "test"}:
            raise ValueError("trajectory partition must be train, validation, or test")
        if any(
            "label" in str(name).casefold()
            or str(name).casefold() in {"y", "y_token", "direction_score"}
            for name in self.raw_token_summary
        ):
            raise ValueError("trajectory fitting is label-blind")
        token_indices = np.asarray(self.response_token_indices)
        surface = np.asarray(self.model_surface)
        anchor = np.asarray(self.mechanism_anchor)
        if token_indices.ndim != 1 or surface.ndim != 4 or anchor.ndim != 1:
            raise ValueError("trajectory record tensors have invalid ranks")
        if not np.issubdtype(token_indices.dtype, np.integer) or (
            len(token_indices) > 1 and not np.all(token_indices[1:] > token_indices[:-1])
        ):
            raise ValueError("response token indices must be strictly increasing integers")
        if surface.shape[-1] != 3:
            raise ValueError("model surface must contain exactly three mechanistic channels")
        token_count = len(token_indices)
        if token_count < 1 or len(surface) != token_count or len(anchor) != token_count:
            raise ValueError("trajectory arrays must align on response tokens")
        if not np.isfinite(surface).all() or not np.isfinite(anchor).all():
            raise ValueError("trajectory arrays must be finite")
        for name, value in self.raw_token_summary.items():
            vector = np.asarray(value)
            if vector.shape != (token_count,) or not np.isfinite(vector).all():
                raise ValueError(f"raw token summary {name} must align and be finite")
        if not 0.0 <= float(self.null_swap_fraction) <= 1.0:
            raise ValueError("null swap fraction must be in [0, 1]")
        if not 0.0 <= float(self.null_effective_fraction) <= 1.0:
            raise ValueError("effective null fraction must be in [0, 1]")
        if not 0.0 <= float(self.null_stable_model_fraction) <= 1.0:
            raise ValueError("stable null-model fraction must be in [0, 1]")
        if self.null_samples < 0:
            raise ValueError("null sample count must be non-negative")
        if self.duplicate_null_draws < 0:
            raise ValueError("duplicate null draw count must be non-negative")
        if self.calibration_status not in {
            "complete",
            "partial",
            "unswappable",
            "insufficient_unique_nulls",
        }:
            raise ValueError("unsupported null calibration status")

    @property
    def is_null_identifiable(self) -> bool:
        """Whether the conditional null provides an estimable observation."""

        return (
            self.calibration_status != "unswappable"
            and self.null_samples >= 2
            and self.null_effective_fraction > 0.0
            and self.null_stable_model_fraction > 0.0
        )

    @property
    def null_exclusion_reason(self) -> str | None:
        if self.calibration_status == "unswappable":
            return "unswappable"
        if self.null_samples < 2:
            return "fewer_than_two_null_samples"
        if self.null_effective_fraction <= 0.0:
            return "no_effective_null_perturbation"
        if self.null_stable_model_fraction <= 0.0:
            return "no_stable_null_variance"
        return None

    def to_artifact(
        self, *, protocol_id: str, source_identity: str
    ) -> dict[str, object]:
        if not protocol_id.strip():
            raise ValueError("trajectory cache requires a protocol identity")
        if not source_identity.strip():
            raise ValueError("trajectory cache requires a source identity")
        return {
            "schema": "grounding-flow-trajectory-v1",
            "protocol_id": str(protocol_id),
            "source_identity": str(source_identity),
            "response_id": self.response_id,
            "pair_id": self.pair_id,
            "partition": self.partition,
            "response_token_indices": torch.as_tensor(
                np.asarray(self.response_token_indices), dtype=torch.long
            ),
            "model_surface": torch.as_tensor(
                np.asarray(self.model_surface), dtype=torch.float32
            ),
            "mechanism_anchor": torch.as_tensor(
                np.asarray(self.mechanism_anchor), dtype=torch.float32
            ),
            "raw_token_summary": {
                name: torch.as_tensor(np.asarray(value), dtype=torch.float32)
                for name, value in self.raw_token_summary.items()
            },
            "null_swap_fraction": float(self.null_swap_fraction),
            "null_effective_fraction": float(self.null_effective_fraction),
            "null_stable_model_fraction": float(
                self.null_stable_model_fraction
            ),
            "null_samples": int(self.null_samples),
            "duplicate_null_draws": int(self.duplicate_null_draws),
            "calibration_status": self.calibration_status,
        }

    @classmethod
    def from_artifact(
        cls,
        value: Mapping[str, object],
        *,
        expected_protocol_id: str,
        expected_source_identity: str,
    ) -> "TrajectoryRecord":
        if value.get("schema") != "grounding-flow-trajectory-v1":
            raise ValueError("unsupported grounding-flow trajectory schema")
        if not expected_protocol_id.strip() or not expected_source_identity.strip():
            raise ValueError("expected trajectory identities must be non-empty")
        if str(value.get("protocol_id", "")) != str(expected_protocol_id):
            raise ValueError("trajectory cache protocol does not match this run")
        if str(value.get("source_identity", "")) != str(expected_source_identity):
            raise ValueError("trajectory cache source does not match the prepared graph")
        summaries = value.get("raw_token_summary")
        if not isinstance(summaries, Mapping):
            raise ValueError("trajectory raw summaries must be a mapping")
        return cls(
            response_id=str(value["response_id"]),
            pair_id=str(value["pair_id"]),
            partition=str(value["partition"]),
            response_token_indices=_as_numpy(value["response_token_indices"]),
            model_surface=_as_numpy(value["model_surface"]),
            mechanism_anchor=_as_numpy(value["mechanism_anchor"]),
            raw_token_summary={
                str(name): _as_numpy(item) for name, item in summaries.items()
            },
            null_swap_fraction=float(value["null_swap_fraction"]),
            null_effective_fraction=float(value.get("null_effective_fraction", 0.0)),
            null_stable_model_fraction=float(
                value.get("null_stable_model_fraction", 0.0)
            ),
            null_samples=int(value.get("null_samples", 0)),
            duplicate_null_draws=int(value.get("duplicate_null_draws", 0)),
            calibration_status=str(value.get("calibration_status", "complete")),
        )


def _as_numpy(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


@dataclass(frozen=True)
class DetectorFitConfig:
    pca_components: int = 32
    pca_fit_tokens: int = 20_000
    hmm_iterations: int = 50
    hmm_tolerance: float = 1e-4
    hmm_variance_floor: float = 1e-4
    seed: int = 42

    def validate(self) -> None:
        if self.pca_components < 1 or self.pca_fit_tokens < 2:
            raise ValueError("PCA limits must be positive")
        if self.hmm_iterations < 1 or self.hmm_tolerance < 0.0:
            raise ValueError("HMM iterations must be positive and tolerance non-negative")
        if self.hmm_variance_floor <= 0.0:
            raise ValueError("HMM variance floor must be positive")


@dataclass(frozen=True)
class GroundingFlowDetector:
    projector: TrajectoryProjector
    state_model: TwoStateGaussianHMM
    fit_config: DetectorFitConfig

    def __post_init__(self) -> None:
        self.fit_config.validate()
        if self.projector.output_dimension != self.state_model.dimension:
            raise ValueError("projector and HMM observation dimensions disagree")

    def score(
        self, record: TrajectoryRecord, *, partition: str | None = None
    ) -> tuple[dict[str, object], list[dict[str, object]]]:
        if not record.is_null_identifiable:
            raise ValueError(
                "trajectory is not null-identifiable: "
                f"{record.null_exclusion_reason}"
            )
        output_partition = record.partition if partition is None else partition
        if output_partition not in {"train", "validation", "test"}:
            raise ValueError("score partition must be train, validation, or test")
        projected = self.projector.transform(record.model_surface)
        scored = self.state_model.score(projected)
        token_probability = np.asarray(scored.pop("token_probability"), dtype=np.float64)
        summaries = {
            name: np.asarray(value, dtype=np.float64)
            for name, value in record.raw_token_summary.items()
        }
        token_rows: list[dict[str, object]] = []
        for offset, (token_idx, probability) in enumerate(
            zip(record.response_token_indices, token_probability)
        ):
            row: dict[str, object] = {
                "response_id": record.response_id,
                "example_id": record.response_id,
                "pair_id": record.pair_id,
                "split": output_partition,
                "token_idx": int(token_idx),
                "response_offset": offset,
                "score": float(probability),
            }
            row.update({name: float(value[offset]) for name, value in summaries.items()})
            token_rows.append(row)
        response: dict[str, object] = {
            "response_id": record.response_id,
            "example_id": record.response_id,
            "pair_id": record.pair_id,
            "split": output_partition,
            "score": float(scored["mean"]),
            "anomaly_score": float(scored["mean"]),
            "mean_detached_probability": float(scored["mean"]),
            "inverse_detached_probability": 1.0 - float(scored["mean"]),
            "max_detached_probability": float(scored["max"]),
            "top_10_percent_mean": float(scored["top_10_percent_mean"]),
            "first_detached_offset": scored["first_detached_offset"],
            "response_tokens": len(token_probability),
            "null_swap_fraction": float(record.null_swap_fraction),
            "null_effective_fraction": float(record.null_effective_fraction),
            "null_stable_model_fraction": float(
                record.null_stable_model_fraction
            ),
            "null_samples": int(record.null_samples),
            "unique_null_samples": int(record.null_samples),
            "duplicate_null_draws": int(record.duplicate_null_draws),
            "null_calibration_status": record.calibration_status,
        }
        response.update(
            {f"mean_{name}": float(value.mean()) for name, value in summaries.items()}
        )
        return response, token_rows

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "grounding-flow-detector-v1",
            "fit_config": asdict(self.fit_config),
            "projector": self.projector.to_dict(),
            "state_model": self.state_model.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "GroundingFlowDetector":
        if value.get("schema") != "grounding-flow-detector-v1":
            raise ValueError("unsupported grounding-flow detector schema")
        return cls(
            projector=TrajectoryProjector.from_dict(value["projector"]),
            state_model=TwoStateGaussianHMM.from_dict(value["state_model"]),
            fit_config=DetectorFitConfig(**value["fit_config"]),
        )


def trajectory_from_calibrated(
    *,
    response_id: str,
    pair_id: str,
    partition: str,
    calibrated: CalibratedFlow,
) -> TrajectoryRecord:
    real = calibrated.real
    ancestry = real.ancestry.mean(dim=(1, 2)).detach().cpu().numpy()
    debt = real.debt.mean(dim=(1, 2)).detach().cpu().numpy()
    unknown = real.component("unknown").mean(dim=(1, 2)).detach().cpu().numpy()
    grounded_relay = (
        real.component("grounded_relay").mean(dim=(1, 2)).detach().cpu().numpy()
    )
    direct_evidence = (
        real.component("direct_evidence").mean(dim=(1, 2)).detach().cpu().numpy()
    )
    return TrajectoryRecord(
        response_id=response_id,
        pair_id=pair_id,
        partition=partition,
        response_token_indices=real.response_token_indices.detach().cpu().numpy(),
        model_surface=calibrated.model_tensor().detach().cpu().numpy(),
        # This scalar only resolves the semantic name of the two fitted states.
        # It is never included in PCA or the HMM likelihood.
        mechanism_anchor=debt - ancestry,
        raw_token_summary={
            "ancestry": ancestry,
            "debt": debt,
            "unknown": unknown,
            "grounded_relay": grounded_relay,
            "direct_evidence": direct_evidence,
        },
        null_swap_fraction=calibrated.accepted_swap_fraction,
        null_effective_fraction=calibrated.effective_null_fraction,
        null_stable_model_fraction=calibrated.stable_model_fraction,
        null_samples=calibrated.null_samples,
        duplicate_null_draws=calibrated.duplicate_null_draws,
        calibration_status=calibrated.calibration_status,
    )


def fit_detector(
    records: Sequence[TrajectoryRecord],
    *,
    config: DetectorFitConfig | None = None,
    progress_callback: Callable[[int, float], None] | None = None,
) -> GroundingFlowDetector:
    fit_config = DetectorFitConfig() if config is None else config
    fit_config.validate()
    if any(record.partition != "train" for record in records):
        raise ValueError("detector fitting accepts only unlabeled train trajectories")
    identifiable = [record for record in records if record.is_null_identifiable]
    if len(identifiable) < 2:
        raise ValueError(
            "detector fitting requires at least two null-identifiable trajectories"
        )
    projector = fit_trajectory_projector(
        [record.model_surface for record in identifiable],
        max_components=fit_config.pca_components,
        max_fit_tokens=fit_config.pca_fit_tokens,
        seed=fit_config.seed,
    )
    projected = [projector.transform(record.model_surface) for record in identifiable]
    state_model = fit_two_state_hmm(
        projected,
        [record.mechanism_anchor for record in identifiable],
        seed=fit_config.seed,
        max_iterations=fit_config.hmm_iterations,
        tolerance=fit_config.hmm_tolerance,
        variance_floor=fit_config.hmm_variance_floor,
        progress_callback=progress_callback,
    )
    return GroundingFlowDetector(
        projector=projector, state_model=state_model, fit_config=fit_config
    )


__all__ = [
    "DetectorFitConfig",
    "GroundingFlowDetector",
    "TrajectoryRecord",
    "fit_detector",
    "trajectory_from_calibrated",
]
