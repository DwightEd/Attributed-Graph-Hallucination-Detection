"""Compatibility facade for the typed RAGTruth token-graph experiment.

Data and graph-cache primitives live in :mod:`ragtruth_data`, label-free model
training/scoring in :mod:`typed_experiment`, and the label-gated final metrics
in :mod:`ragtruth_evaluation`.  Existing callers can keep this import path.
"""

from .ragtruth_data import (
    CompactGraphStore,
    collate_graphs,
    compact_attention_cache,
    discover_attention_paths,
    load_compact_manifest,
    make_answer_mask,
    split_paths_by_source,
    validate_label_free,
)
from .ragtruth_evaluation import (
    evaluate_score_file,
    evaluate_token_score_records,
    load_cached_token_labels,
    read_score_records,
)
from .typed_experiment import (
    score_graph_strided,
    score_typed_autoencoder,
    train_typed_autoencoder,
)

__all__ = [
    "CompactGraphStore",
    "collate_graphs",
    "compact_attention_cache",
    "discover_attention_paths",
    "evaluate_score_file",
    "evaluate_token_score_records",
    "load_cached_token_labels",
    "load_compact_manifest",
    "make_answer_mask",
    "read_score_records",
    "score_graph_strided",
    "score_typed_autoencoder",
    "split_paths_by_source",
    "train_typed_autoencoder",
    "validate_label_free",
]
