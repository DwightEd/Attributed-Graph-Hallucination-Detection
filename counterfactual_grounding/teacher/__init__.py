"""Counterfactual and activation-mediation teacher utilities."""

from .counterfactuals import (
    CounterfactualAudit,
    CounterfactualGenerationError,
    EqualTokenCounterfactual,
    generate_equal_token_counterfactual,
    generate_equal_token_counterfactuals,
    validate_counterfactual_pair,
)
from .mediation import (
    KVStore,
    LlamaKVMediationBackend,
    MediationEffects,
    MediationRun,
    UnsupportedMediationBackend,
    decompose_mediation_effects,
    target_token_log_probs,
)
from .pilot import Gate1Pair, Gate1RuntimeIdentity, run_gate1_pilot

__all__ = [
    "CounterfactualAudit",
    "CounterfactualGenerationError",
    "EqualTokenCounterfactual",
    "Gate1Pair",
    "Gate1RuntimeIdentity",
    "KVStore",
    "LlamaKVMediationBackend",
    "MediationEffects",
    "MediationRun",
    "UnsupportedMediationBackend",
    "decompose_mediation_effects",
    "generate_equal_token_counterfactual",
    "generate_equal_token_counterfactuals",
    "run_gate1_pilot",
    "target_token_log_probs",
    "validate_counterfactual_pair",
]
