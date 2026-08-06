from pathlib import Path

from patf.data import load_attention

from .conftest import write_cache


def test_sparse_loader_ignores_labels(tmp_path: Path) -> None:
    path = tmp_path / "attention_0.pt"
    write_cache(path, sample_id="0", source_id="s0")
    sample = load_attention(path)
    assert sample.sample_id == "0"
    assert sample.num_channels == 4
    assert sample.response_tokens == 3
