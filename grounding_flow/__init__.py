"""Attention-only evidence-provenance flow for label-free hallucination study."""

from .method import (
    COMPONENT_NAMES,
    CalibratedFlow,
    EvidenceFlow,
    NullModelConfig,
    RewireReport,
    calibrate_against_null,
    compute_evidence_flow,
    rewire_source_identity,
)
from .experiment import (
    DetectorFitConfig,
    GroundingFlowDetector,
    TrajectoryRecord,
    fit_detector,
    trajectory_from_calibrated,
)
from .state_model import (
    TrajectoryProjector,
    TwoStateGaussianHMM,
    fit_trajectory_projector,
    fit_two_state_hmm,
)

__all__ = [
    "COMPONENT_NAMES",
    "CalibratedFlow",
    "EvidenceFlow",
    "NullModelConfig",
    "RewireReport",
    "calibrate_against_null",
    "compute_evidence_flow",
    "rewire_source_identity",
    "DetectorFitConfig",
    "GroundingFlowDetector",
    "TrajectoryRecord",
    "fit_detector",
    "trajectory_from_calibrated",
    "TrajectoryProjector",
    "TwoStateGaussianHMM",
    "fit_trajectory_projector",
    "fit_two_state_hmm",
]
