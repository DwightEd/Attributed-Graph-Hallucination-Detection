"""Canonical identity for the runtime that produced observer attention."""

from __future__ import annotations

import re
from collections.abc import Mapping

from .artifacts import canonical_hash

OBSERVER_RUNTIME_SCHEMA = "cept-observer-runtime-v1"
_VERSION = re.compile(r"^[0-9]+(?:\.[0-9]+)+(?:[+._-][A-Za-z0-9._-]+)?$")
_DTYPES = {
    "float16": "float16",
    "fp16": "float16",
    "half": "float16",
    "torch.float16": "float16",
    "bfloat16": "bfloat16",
    "bf16": "bfloat16",
    "torch.bfloat16": "bfloat16",
    "float32": "float32",
    "fp32": "float32",
    "float": "float32",
    "torch.float32": "float32",
}
_ATTENTION_IMPLEMENTATIONS = frozenset(
    {"eager", "sdpa", "flash_attention_2", "flash_attention_3", "flex_attention"}
)


def _version(value: object, field: str) -> str:
    normalized = str(value).strip()
    if not _VERSION.fullmatch(normalized):
        raise ValueError(f"observer runtime {field} is missing or invalid")
    return normalized


def normalize_observer_dtype(value: object) -> str:
    normalized = str(value).strip().casefold()
    if normalized not in _DTYPES:
        raise ValueError("observer runtime dtype is missing, ambiguous, or invalid")
    return _DTYPES[normalized]


def _attention_implementation(value: object) -> str:
    normalized = str(value).strip().casefold().replace("-", "_")
    if normalized not in _ATTENTION_IMPLEMENTATIONS:
        raise ValueError(
            "observer runtime attn_implementation is missing or unsupported"
        )
    return normalized


def observer_runtime_identity(
    *,
    transformers_version: object,
    torch_version: object,
    dtype: object,
    attn_implementation: object,
) -> tuple[dict[str, str], str]:
    """Return the normalized observer-runtime mapping and its canonical hash."""

    runtime = {
        "schema": OBSERVER_RUNTIME_SCHEMA,
        "transformers_version": _version(
            transformers_version, "transformers_version"
        ),
        "torch_version": _version(torch_version, "torch_version"),
        "dtype": normalize_observer_dtype(dtype),
        "attn_implementation": _attention_implementation(attn_implementation),
    }
    return runtime, "sha256:" + canonical_hash(runtime)


def parse_observer_runtime_identity(
    value: object, signature: object
) -> tuple[dict[str, str], str]:
    """Validate a serialized runtime instead of trusting its declared hash."""

    if not isinstance(value, Mapping):
        raise TypeError("observer_runtime must be an object")
    if value.get("schema") != OBSERVER_RUNTIME_SCHEMA:
        raise ValueError("observer_runtime schema is missing or unsupported")
    runtime, expected = observer_runtime_identity(
        transformers_version=value.get("transformers_version"),
        torch_version=value.get("torch_version"),
        dtype=value.get("dtype"),
        attn_implementation=value.get("attn_implementation"),
    )
    observed = str(signature).strip()
    if observed != expected:
        raise ValueError("observer_runtime_signature does not bind the runtime mapping")
    if set(value) != set(runtime):
        raise ValueError("observer_runtime contains unregistered identity fields")
    return runtime, expected


def observer_runtime_from_cache_manifest(
    manifest: Mapping[str, object],
) -> tuple[dict[str, str], str]:
    """Extract the audited observer runtime from attention_cache_spec."""

    spec = manifest.get("attention_cache_spec")
    if not isinstance(spec, Mapping):
        raise TypeError("attention cache has no valid attention_cache_spec")
    return observer_runtime_identity(
        transformers_version=spec.get("transformers_version"),
        torch_version=spec.get("torch_version"),
        dtype=spec.get("dtype"),
        attn_implementation=spec.get("attn_implementation"),
    )


__all__ = [
    "OBSERVER_RUNTIME_SCHEMA",
    "normalize_observer_dtype",
    "observer_runtime_from_cache_manifest",
    "observer_runtime_identity",
    "parse_observer_runtime_identity",
]
