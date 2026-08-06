from __future__ import annotations

import pytest

from counterfactual_grounding.observer_runtime import (
    observer_runtime_identity,
    parse_observer_runtime_identity,
)


def test_observer_runtime_normalizes_dtype_and_binds_every_semantic_field():
    runtime, signature = observer_runtime_identity(
        transformers_version="4.52.3",
        torch_version="2.6.0+cu124",
        dtype="torch.float16",
        attn_implementation="EAGER",
    )

    assert runtime == {
        "schema": "cept-observer-runtime-v1",
        "transformers_version": "4.52.3",
        "torch_version": "2.6.0+cu124",
        "dtype": "float16",
        "attn_implementation": "eager",
    }
    parsed, parsed_signature = parse_observer_runtime_identity(runtime, signature)
    assert parsed == runtime
    assert parsed_signature == signature


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("transformers_version", ""),
        ("torch_version", "unknown"),
        ("dtype", "auto"),
        ("attn_implementation", "mystery-kernel"),
    ),
)
def test_observer_runtime_fails_closed_on_missing_or_ambiguous_identity(
    field: str, value: str
):
    values = {
        "transformers_version": "4.52.3",
        "torch_version": "2.6.0+cu124",
        "dtype": "float16",
        "attn_implementation": "eager",
    }
    values[field] = value

    with pytest.raises(ValueError, match=field):
        observer_runtime_identity(**values)


def test_observer_runtime_rejects_a_signature_not_bound_to_the_mapping():
    runtime, _ = observer_runtime_identity(
        transformers_version="4.52.3",
        torch_version="2.6.0+cu124",
        dtype="float16",
        attn_implementation="eager",
    )

    with pytest.raises(ValueError, match="signature"):
        parse_observer_runtime_identity(runtime, "sha256:" + "0" * 64)
