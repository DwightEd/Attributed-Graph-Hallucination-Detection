# CEPT: Counterfactual Evidence-path Transport

This package is the new, staged method.  It does not import the legacy
GraphMAE/GMM/Grounding-Flow trainers.

Current implementation scope:

1. **Gate 0**: label-blind RAGTruth loading; exact Summary/QA/Data2txt E/Q/R
   layout; `predictor_position=t-1`; one-time canonical legacy-cache graph
   adaptation with censored mass kept as `unknown`.
2. **Gate 1**: equal-token evidence-only numeric counterfactuals and exact
   post-RoPE, pre-`repeat_kv` Llama K/V mediation for Y11/Y00/Y10/Y01, plus a
   real-runtime self-patch audit.

There is deliberately no student training and no hallucination-label AUROC in
this stage.  The graph student will only be implemented after the intervention
contracts pass on the remote 8B observer.

Code map:

- `data/dataset.py`, `data/ragtruth.py`: label-firewalled RAGTruth loading and
  exact E/Q/R token layout.
- `data/graph.py`, `data/store.py`, `graph_build.py`: prediction-event graph
  contract and one-time content-addressed cache adaptation.
- `teacher/counterfactuals.py`: equal-token, evidence-only counterfactuals.
- `teacher/mediation.py`, `teacher/pilot.py`: exact Llama K/V intervention and
  the four registered outcomes.
- `experiment.py`: recoverable Gate 0/1 orchestration and auditable artifacts.
- `cli.py`: the only Python command-line entrypoint.

Run the 50-sample pilot from the repository root:

```bash
bash ./run_ragtruth_cept_pilot.sh
```

The runner creates and reuses a dedicated `CEPT_VENV` with
`--system-site-packages`.  This inherits the CUDA-enabled PyTorch installation
from the configured `PYTHON_BIN`, while installing the exact
`transformers==4.52.3` runtime only inside the CEPT venv; it does not modify the
shared research conda environment.  Both paths remain configurable:

```bash
PYTHON_BIN=/path/to/cuda/python \
CEPT_VENV=/path/to/cept-venv \
bash ./run_ragtruth_cept_pilot.sh
```

Set `INSTALL_DEPS=0` only when that CEPT venv already contains the exact pinned
runtime.  The runner prints the active Python executable, environment prefixes,
Torch/Transformers versions and module paths before starting the pilot.

The default 8B observer is loaded unsharded with eager attention.  Use a GPU
with at least 24 GiB as the practical baseline; the runner checks free memory
before model loading and fails early instead of silently spilling or sharding.
Eager attention is quadratic in sequence length, hence the explicit 2048-token
no-truncation ceiling.

Artifacts are a single run directory containing `manifest.json`,
`selection.jsonl`, `counterfactual_audit.jsonl`, `effects.jsonl`,
`gate_report.json`, `gate1_shards/`, and `run.log`.  Each completed sample is
atomically saved under `gate1_shards/`; after an interruption, rerun the exact
same output and inputs with `RESUME=1`.  Resume verifies configuration, dataset,
checkpoint, runtime, pair, and shard hashes before skipping finished samples.
`effects.jsonl` keeps the four raw network outcomes as well as derived effects,
so every decomposition is independently recomputable.

```bash
OUTPUT_DIR=/path/to/interrupted/run RESUME=1 \
bash ./run_ragtruth_cept_pilot.sh
```

The canonical attention graph is a separate, one-time cache adaptation (it is
not rebuilt for each teacher run):

```bash
python -m counterfactual_grounding.cli build-graphs \
  --cache-root /path/to/fresh_attention_cache \
  --responses /path/to/RAGTruth/dataset/response.jsonl \
  --sources /path/to/RAGTruth/dataset/source_info.jsonl \
  --tokenizer /path/to/Meta-Llama-3.1-8B-Instruct \
  --output-dir /path/to/ragtruth_cept_canonical \
  --split train
```

Legacy cache adaptation is explicitly partial: the old cache did not store the
prompt-final predictor row needed by the first response token.  That event is
kept with `row_available=false` and `unknown_mass=1`; it is never silently
dropped or filled from the wrong row.

Optional response-history block rescue is enabled with, for example:

```bash
HISTORY_BLOCK_SIZE=4 MAX_HISTORY_BLOCKS=4 bash ./run_ragtruth_cept_pilot.sh
```

The current counterfactual protocol is intentionally narrow:
`numeric_digit_surface_preserving_v1`.  It does not yet justify a claim about
general entity, date, or fluent counterfactual generation.
