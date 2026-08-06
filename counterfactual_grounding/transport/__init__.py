"""Causal evidence-provenance transport student.

The package is deliberately small: ``data`` owns the label-free teacher/graph
contract and ``model`` owns the only learned operator and its objective.
"""

from .data import (
    AlignedTransportExample,
    EffectCalibration,
    TeacherTargets,
    TransportEventTarget,
    TransportTeacher,
    align_teacher_to_graph,
    build_teacher_targets,
    fit_effect_calibration,
    make_transport_teacher_record,
    parse_transport_teacher,
)
from .model import (
    CausalProvenanceTransport,
    TransportOutput,
    response_risk,
    transport_loss,
)

__all__ = [
    "AlignedTransportExample",
    "CausalProvenanceTransport",
    "EffectCalibration",
    "TeacherTargets",
    "TransportEventTarget",
    "TransportOutput",
    "TransportTeacher",
    "align_teacher_to_graph",
    "build_teacher_targets",
    "fit_effect_calibration",
    "make_transport_teacher_record",
    "parse_transport_teacher",
    "response_risk",
    "transport_loss",
]
