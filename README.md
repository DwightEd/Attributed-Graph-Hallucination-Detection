# Prompt-Anchored Topology Flow

PATF is a label-free response-level hallucination detector built from saved LLM attention. The repository also keeps the supervised CHARM baseline and an upstream-code CoLA unsupervised baseline.

## Structure

```text
attention_cache/io.py   shared sparse-CSR cache access
original/               original threshold-union attributed graph + CHARM
patf/flow.py            cross-layer prompt-rooted attention flow
patf/augment.py         grounding-erosion counterfactual
patf/model.py           trajectory ranker
patf/train.py           label-free ranking training and scoring
patf/evaluate.py        response-level evaluation
baselines/cola/         CoLA data/scenario adapter
baselines/cola/upstream/model.py
                        byte-identical upstream CoLA model
configs/patf.json       PATF experiment parameters
scripts/run_patf.sh     PATF runner
scripts/run_charm.sh    supervised CHARM runner
scripts/run_cola.sh     unsupervised CoLA runner
scripts/prepare_attention_split.sh
                        attention-cache preparation
run.py                  Python PATF entry
```

The original graph is unchanged: a token-pair edge exists when any layer/head attention is strictly greater than `tau`; all above-threshold channels are retained in `edge_attr`. PATF does not read those persisted graphs.

PATF propagates prompt-rooted support **across Transformer layers**. For each layer, all heads and response rows are processed with sparse tensor operations; there is no per-head graph object and no same-layer recursive evidence path.

The CoLA baseline consumes the saved original attributed graphs. It uses only `x` and `edge_index`: `x` is the attention-diagonal node feature and `edge_index` is adapted to the undirected, unweighted adjacency expected by upstream CoLA. `edge_attr` and `edge_mark` are not added to the model. The neural model is copied exactly from `TrustAGI-Lab/CoLA@c2b5273fe6368509dfc558a485764003f5f18ca3`; only the RAGTruth multi-graph data interface and response-level aggregation are adapted.

## Run

PATF:

```bash
CUDA_VISIBLE_DEVICES=0 WORKERS=8 TORCH_THREADS=1 bash scripts/run_patf.sh
```

CoLA baseline:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_cola.sh
```

Useful PATF overrides:

```bash
ATTENTION_ROOT=/path/to/attention_cache \
RAGTRUTH_ROOT=/path/to/RAGTruth/dataset \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda EPOCHS=80 WORKERS=8 TORCH_THREADS=1 \
bash scripts/run_patf.sh
```

PATF feature extraction is sample-parallel and resumable. Completed `features/*.features.pt` files are reused when both the source file and method configuration match.
