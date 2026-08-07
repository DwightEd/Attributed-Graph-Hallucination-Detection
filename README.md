# Prompt-Anchored Topology Flow

PATF is a label-free response-level hallucination detector built from saved LLM attention. The supervised CHARM baseline and PATF share only the formal attention-cache loader; their graph definitions are separate.

## Structure

```text
attention_cache/io.py   shared sparse-CSR cache access
original/               original threshold-union attributed graph + CHARM
patf/flow.py            cross-layer prompt-rooted attention flow
patf/augment.py         grounding-erosion counterfactual
patf/model.py           trajectory ranker
patf/train.py           label-free ranking training and scoring
patf/evaluate.py        response-level evaluation
configs/patf.json       experiment parameters
scripts/run_patf.sh     PATF runner
scripts/run_charm.sh    supervised CHARM runner
scripts/prepare_attention_split.sh
                        attention-cache preparation
run.py                  Python PATF entry
```

The original graph is unchanged: a token-pair edge exists when any layer/head attention is strictly greater than `tau`; all above-threshold channels are retained in `edge_attr`. PATF does not read those persisted graphs.

PATF propagates prompt-rooted support **across Transformer layers**. For each layer, all heads and response rows are processed with sparse tensor operations; there is no per-head graph object and no same-layer recursive evidence path.

## Run

```bash
CUDA_VISIBLE_DEVICES=0 WORKERS=8 TORCH_THREADS=1 bash scripts/run_patf.sh
```

Useful overrides:

```bash
ATTENTION_ROOT=/path/to/attention_cache \
RAGTRUTH_ROOT=/path/to/RAGTruth/dataset \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda EPOCHS=80 WORKERS=8 TORCH_THREADS=1 \
bash scripts/run_patf.sh
```

Feature extraction is sample-parallel and resumable. Completed `features/*.features.pt` files are reused when both the source file and method configuration match.
