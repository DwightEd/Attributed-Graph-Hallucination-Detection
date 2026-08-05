"""Exact K/V activation mediation for supported Hugging Face Llama models.

The registered attention function observes keys after RoPE and values at the
same pre-``repeat_kv`` site.  This is intentionally narrower than a generic
hook system: unsupported model/runtime combinations fail closed.
"""

from __future__ import annotations

import importlib.metadata
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import torch

BACKEND_ID = "cept_mediation_eager"
SUPPORTED_TRANSFORMERS = frozenset({"4.52.3"})


class UnsupportedMediationBackend(RuntimeError):
    """The exact, audited intervention site is unavailable."""


@dataclass(frozen=True)
class MediationEffects:
    total: torch.Tensor
    direct: torch.Tensor
    mediated: torch.Tensor
    alternate_mediated: torch.Tensor
    interaction: torch.Tensor
    contract_residual: torch.Tensor


@dataclass(frozen=True)
class KVStore:
    """Layer-indexed post-RoPE K and same-site V for fixed token positions."""

    positions: torch.Tensor
    keys: dict[int, torch.Tensor]
    values: dict[int, torch.Tensor]


@dataclass(frozen=True)
class MediationRun:
    target_log_probs: torch.Tensor
    kv: KVStore | None


@dataclass
class _RunContext:
    sequence_length: int
    capture_positions: torch.Tensor | None
    patch_positions: torch.Tensor | None
    sender: KVStore | None
    keys: dict[int, torch.Tensor] = field(default_factory=dict)
    values: dict[int, torch.Tensor] = field(default_factory=dict)
    visited_layers: set[int] = field(default_factory=set)


_ACTIVE_RUN: ContextVar[_RunContext | None] = ContextVar(
    "cept_active_mediation_run", default=None
)
_REGISTERED = False


def target_token_log_probs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Return teacher-forced log p(token[t] | tokens[:t]) for every target t."""

    if logits.ndim != 3 or input_ids.ndim != 2:
        raise ValueError("logits must be [B,S,V] and input_ids must be [B,S]")
    if logits.shape[:2] != input_ids.shape:
        raise ValueError("logits and input_ids must share batch/sequence axes")
    targets = torch.as_tensor(target_positions, dtype=torch.long, device=logits.device)
    if targets.ndim != 1 or targets.numel() == 0:
        raise ValueError("target_positions must be a non-empty one-dimensional vector")
    sequence_length = input_ids.shape[1]
    if bool(((targets <= 0) | (targets >= sequence_length)).any()):
        raise ValueError("target_positions must have valid t-1 predictors")
    predictors = targets - 1
    selected_logits = logits.index_select(1, predictors)
    target_ids = input_ids.to(logits.device).index_select(1, targets)
    return torch.log_softmax(selected_logits.float(), dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)


def _selected_target_log_probs(
    selected_logits: torch.Tensor,
    input_ids: torch.Tensor,
    target_positions: torch.Tensor,
) -> torch.Tensor:
    """Score targets when Llama computed logits only at t-1 predictors."""

    targets = target_positions.to(input_ids.device)
    if selected_logits.ndim != 3 or selected_logits.shape[:2] != (
        input_ids.shape[0],
        targets.numel(),
    ):
        raise UnsupportedMediationBackend(
            "Llama logits_to_keep did not preserve requested predictor order"
        )
    target_ids = input_ids.index_select(1, targets)
    return torch.log_softmax(selected_logits.float(), dim=-1).gather(
        -1, target_ids.unsqueeze(-1)
    ).squeeze(-1)


def decompose_mediation_effects(
    *,
    y11: torch.Tensor,
    y00: torch.Tensor,
    y10: torch.Tensor,
    y01: torch.Tensor,
) -> MediationEffects:
    """Compute the registered four-run CEPT effect decomposition.

    ``direct`` is an operational non-history-K/V effect, not a claim of a
    natural direct effect.  The naming remains compact in artifacts while the
    distinction is documented explicitly here and in the pilot report.
    """

    if not (y11.shape == y00.shape == y10.shape == y01.shape):
        raise ValueError("all four mediation outcomes must have identical shapes")
    total = y11 - y00
    direct = y10 - y00
    mediated = y11 - y10
    alternate = y01 - y00
    return MediationEffects(
        total=total,
        direct=direct,
        mediated=mediated,
        alternate_mediated=alternate,
        interaction=mediated - alternate,
        contract_residual=total - direct - mediated,
    )


def _source_indices(sender_positions: torch.Tensor, patch_positions: torch.Tensor) -> torch.Tensor:
    if sender_positions.ndim != 1 or patch_positions.ndim != 1:
        raise ValueError("K/V positions must be one-dimensional")
    if torch.unique(sender_positions).numel() != sender_positions.numel():
        raise ValueError("sender K/V positions must be unique")
    lookup = {int(position): index for index, position in enumerate(sender_positions)}
    missing = [int(position) for position in patch_positions if int(position) not in lookup]
    if missing:
        raise ValueError(f"sender K/V does not contain patch positions: {missing}")
    return torch.tensor(
        [lookup[int(position)] for position in patch_positions], dtype=torch.long
    )


def _validate_sender_layer(
    *,
    layer: int,
    sender: KVStore,
    key: torch.Tensor,
    value: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if layer not in sender.keys or layer not in sender.values:
        raise ValueError(f"sender K/V is missing layer {layer}")
    sender_key = sender.keys[layer]
    sender_value = sender.values[layer]
    expected_key = (key.shape[0], key.shape[1], sender.positions.numel(), key.shape[3])
    expected_value = (
        value.shape[0],
        value.shape[1],
        sender.positions.numel(),
        value.shape[3],
    )
    if tuple(sender_key.shape) != expected_key:
        raise ValueError(
            f"sender key K/V shape mismatch at layer {layer}: "
            f"expected {expected_key}, got {tuple(sender_key.shape)}"
        )
    if tuple(sender_value.shape) != expected_value:
        raise ValueError(
            f"sender value K/V shape mismatch at layer {layer}: "
            f"expected {expected_value}, got {tuple(sender_value.shape)}"
        )
    if sender_key.dtype != key.dtype or sender_value.dtype != value.dtype:
        raise ValueError("sender K/V dtype does not match the receiver")
    if sender_key.device != key.device or sender_value.device != value.device:
        raise ValueError("sender K/V device does not match the receiver")
    return sender_key, sender_value


def _cept_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    **kwargs: Any,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    context = _ACTIVE_RUN.get()
    if context is None:
        raise RuntimeError("CEPT attention backend was called outside a mediation run")
    layer = getattr(module, "layer_idx", None)
    if not isinstance(layer, int):
        raise UnsupportedMediationBackend("Llama attention layer_idx is unavailable")
    context.visited_layers.add(layer)
    if key.ndim != 4 or value.ndim != 4 or key.shape[:3] != value.shape[:3]:
        raise UnsupportedMediationBackend("unexpected pre-repeat Llama K/V shape")
    if key.shape[2] != context.sequence_length:
        raise UnsupportedMediationBackend(
            "cached/incremental attention is unsupported; use_cache must be false"
        )

    effective_key = key
    effective_value = value
    if context.sender is not None:
        assert context.patch_positions is not None
        sender_key, sender_value = _validate_sender_layer(
            layer=layer,
            sender=context.sender,
            key=key,
            value=value,
        )
        source_index = _source_indices(
            context.sender.positions, context.patch_positions
        ).to(key.device)
        destination_index = context.patch_positions.to(key.device)
        effective_key = key.clone()
        effective_value = value.clone()
        effective_key.index_copy_(
            2, destination_index, sender_key.index_select(2, source_index)
        )
        effective_value.index_copy_(
            2, destination_index, sender_value.index_select(2, source_index)
        )

    if context.capture_positions is not None:
        positions = context.capture_positions.to(key.device)
        context.keys[layer] = effective_key.index_select(2, positions).detach().clone()
        context.values[layer] = (
            effective_value.index_select(2, positions).detach().clone()
        )

    from transformers.models.llama.modeling_llama import eager_attention_forward

    return eager_attention_forward(
        module,
        query,
        effective_key,
        effective_value,
        attention_mask,
        **kwargs,
    )


def _register_backend() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    from transformers import AttentionInterface

    AttentionInterface.register(BACKEND_ID, _cept_attention_forward)
    # Transformers 5 dispatches causal-mask construction by the same custom
    # backend key.  Transformers 4.52 builds the ordinary 4D mask for every
    # non-flash/non-flex implementation and does not expose this interface.
    try:
        from transformers import AttentionMaskInterface
    except ImportError:
        AttentionMaskInterface = None  # type: ignore[assignment,misc]
    if AttentionMaskInterface is not None:
        eager_mask = AttentionMaskInterface._global_mapping.get("eager")
        if eager_mask is None:
            raise UnsupportedMediationBackend("transformers eager causal mask is unavailable")
        AttentionMaskInterface.register(BACKEND_ID, eager_mask)
    _REGISTERED = True


class LlamaKVMediationBackend:
    """Run capture/patch interventions at Llama's exact attention K/V site."""

    def __init__(self, model: torch.nn.Module):
        version_text = importlib.metadata.version("transformers")
        if version_text not in SUPPORTED_TRANSFORMERS:
            raise UnsupportedMediationBackend(
                "unsupported transformers runtime; the audited CEPT K/V patch "
                "site requires exactly transformers==4.52.3"
            )
        config = getattr(model, "config", None)
        if config is None or getattr(config, "model_type", None) != "llama":
            raise UnsupportedMediationBackend("only Hugging Face Llama is supported")
        layers = getattr(getattr(model, "model", None), "layers", None)
        if layers is None or len(layers) != int(config.num_hidden_layers):
            raise UnsupportedMediationBackend("standard Llama decoder layers are required")
        if getattr(model, "is_quantized", False):
            raise UnsupportedMediationBackend("quantized Llama mediation is unsupported")
        _register_backend()
        self.model = model
        self.num_layers = int(config.num_hidden_layers)

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
        """Execute one teacher-forced natural, capture, or patched condition."""

        if self.model.training:
            raise ValueError("mediation requires model.eval()")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("mediation currently requires batch size 1")
        if attention_mask.shape != input_ids.shape:
            raise ValueError("attention_mask must match input_ids")
        sequence_length = input_ids.shape[1]
        targets = torch.as_tensor(target_positions, dtype=torch.long).flatten().cpu()
        if targets.numel() == 0 or bool(((targets <= 0) | (targets >= sequence_length)).any()):
            raise ValueError("target_positions contain no valid prediction events")

        capture = None
        if capture_positions is not None:
            capture = torch.as_tensor(capture_positions, dtype=torch.long).flatten().cpu()
            if capture.numel() == 0 or bool(
                ((capture < 0) | (capture >= sequence_length)).any()
            ):
                raise ValueError("capture_positions are outside the input")
            if torch.unique(capture).numel() != capture.numel():
                raise ValueError("capture_positions must be unique")

        patch = None
        if sender is None and patch_positions is not None:
            raise ValueError("patch_positions require a sender K/V store")
        if sender is not None:
            if patch_positions is None:
                raise ValueError("sender K/V requires patch_positions")
            patch = torch.as_tensor(patch_positions, dtype=torch.long).flatten().cpu()
            if patch.numel() == 0 or bool(((patch < 0) | (patch >= sequence_length)).any()):
                raise ValueError("patch_positions are outside the receiver input")
            if torch.unique(patch).numel() != patch.numel():
                raise ValueError("patch_positions must be unique")
            _source_indices(sender.positions.cpu(), patch)

        context = _RunContext(
            sequence_length=sequence_length,
            capture_positions=capture,
            patch_positions=patch,
            sender=sender,
        )
        config = self.model.config
        original_implementation = getattr(config, "_attn_implementation", None)
        token = _ACTIVE_RUN.set(context)
        try:
            config._attn_implementation = BACKEND_ID
            with torch.inference_mode():
                output = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                    logits_to_keep=(targets - 1).to(input_ids.device),
                )
        finally:
            config._attn_implementation = original_implementation
            _ACTIVE_RUN.reset(token)
        expected_layers = set(range(self.num_layers))
        if context.visited_layers != expected_layers:
            raise UnsupportedMediationBackend(
                "custom attention did not visit every Llama decoder layer"
            )
        probabilities = _selected_target_log_probs(
            output.logits, input_ids, targets
        )
        store = None
        if capture is not None:
            if set(context.keys) != expected_layers or set(context.values) != expected_layers:
                raise RuntimeError("K/V capture is incomplete")
            store = KVStore(
                positions=capture.clone(),
                keys=context.keys,
                values=context.values,
            )
        return MediationRun(target_log_probs=probabilities, kv=store)


__all__ = [
    "BACKEND_ID",
    "KVStore",
    "LlamaKVMediationBackend",
    "MediationEffects",
    "MediationRun",
    "UnsupportedMediationBackend",
    "decompose_mediation_effects",
    "target_token_log_probs",
]
