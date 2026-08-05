from __future__ import annotations

from pathlib import Path

from counterfactual_grounding.artifacts import source_inventory_signature


def test_model_inventory_hash_binds_same_size_weight_contents(tmp_path: Path) -> None:
    model = tmp_path / "model"
    model.mkdir()
    weight = model / "model-00001-of-00001.safetensors"
    weight.write_bytes(b"first-weight")
    first = source_inventory_signature(model)

    weight.write_bytes(b"other-value!")
    assert weight.stat().st_size == len(b"first-weight")
    second = source_inventory_signature(model)

    assert first != second
