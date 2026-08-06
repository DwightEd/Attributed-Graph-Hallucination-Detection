# Prompt-Anchored Topology Flow

PATF is a label-free response-level hallucination detector built from saved
layer/head attention. The repository keeps the supervised attributed-graph
baseline and PATF as separate methods over one shared attention-cache API.

## Structure

```text
attention_cache/io.py   shared sparse-CSR cache access
original/               original threshold-union attributed graph + CHARM
patf/                   prompt-anchored topology method
configs/patf.json       method, training, and runtime parameters
scripts/run_patf.sh     one experiment entry
```

The original graph and PATF do not share graph construction:

- `original/ragtruth_graph.py`: a token-pair edge is created when **any**
  layer/head value is strictly greater than `tau`; `edge_attr` keeps all
  above-threshold channels and zeros the rest. This is threshold union, not
  max-value selection and not top-k.
- `patf/topology.py`: no static attributed graph is saved. Each layer/head is
  analysed separately, then aggregated into a `[layers, 30]` topology
  trajectory.

Only the cache schema, sample identity, and CSR row access are shared through
`attention_cache/io.py`.

## Run PATF

```bash
CUDA_VISIBLE_DEVICES=0 WORKERS=8 bash scripts/run_patf.sh
```

Useful overrides:

```bash
ATTENTION_ROOT=/path/to/attention_cache \
RAGTRUTH_ROOT=/path/to/RAGTruth/dataset \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda EPOCHS=80 WORKERS=8 TORCH_THREADS=1 \
bash scripts/run_patf.sh
```

`WORKERS` controls sample-level multiprocessing. Each worker uses
`TORCH_THREADS=1` by default to avoid CPU oversubscription. For shared storage,
start with 4 or 8 workers; increasing beyond the available I/O bandwidth may
slow the run.

## Outputs

```text
output/
  status.json
  config.json
  run.log
  features/train/*.features.pt
  model/model.pt
  model/history.json
  features/test/*.features.pt
  predictions.jsonl
  evaluation.json
```

Feature files are written atomically per sample and reused when both the source
file and method configuration match. During the first stage `model/` remains
empty because the ranker is trained only after all training trajectories are
available; progress is visible in `features/train/`, `status.json`, and
`run.log`.
