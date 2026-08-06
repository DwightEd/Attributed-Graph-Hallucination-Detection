# CEPT: Counterfactual Evidence-path Transport

This package is the new, staged method.  It does not import the legacy
GraphMAE/GMM/Grounding-Flow trainers.

Implemented scope:

1. **Gate 0**: label-blind RAGTruth loading; exact Summary/QA/Data2txt E/Q/R
   layout; `predictor_position=t-1`; one-time canonical legacy-cache graph
   adaptation with censored mass kept as `unknown`.
2. **Gate 1**: equal-token evidence-only numeric counterfactuals and exact
   post-RoPE, pre-`repeat_kv` Llama K/V mediation for Y11/Y00/Y10/Y01, plus a
   real-runtime self-patch audit.
3. **Gate 2**: a low-capacity attention-routing student.  It learns non-negative
   `EY/QY/RY x layer x head` surrogate gates and one bounded operational
   residual-persistence coefficient per layer.  Seed/history rescue and
   selected history-block rankings are supervised; joint rescue and its
   residual/interaction are reliability diagnostics.  The token-by-layer DAG
   preserves the exact prediction-row (`t-1`) and response K/V indexing of the
   attention subgraph.  It does not claim to reconstruct the transformer's
   residual stream, MLP paths, or causal attention heads.

Hallucination labels are absent from graph construction, intervention teaching,
training, validation, score orientation, and checkpoint selection.  They are
opened only by the separate post-hoc evaluation command after prediction hashes
have been frozen.

The research question is not whether an autoregressive model uses response
history.  CEPT asks whether retained response-history attention is consistent
with an intervention-calibrated evidence-ancestry surrogate.  High RR attention
can be a grounded relay.  The primary score is the model-conditional lower end
of unsupported, gated, layer-averaged RR exposure; it is not physical
information flow or proof of self-reinforcing error propagation.  Generic
`1 - support_upper` is retained only as an auxiliary diagnostic.

Only the deployed student is attention-only.  The label-free teacher uses
post-RoPE K/V interventions and target-token log probabilities.  Consequently,
the method studies source-relative faithfulness/routing risk, not open-world
factual truth from attention alone.

Code map:

- `data/dataset.py`, `data/ragtruth.py`: label-firewalled RAGTruth loading and
  exact E/Q/R token layout.
- `data/graph.py`, `data/store.py`, `graph_build.py`: prediction-event graph
  contract and one-time content-addressed cache adaptation.
- `teacher/counterfactuals.py`: equal-token, evidence-only counterfactuals.
- `teacher/mediation.py`, `teacher/pilot.py`: exact Llama K/V intervention and
  the four registered outcomes.
- `transport/data.py`: structured teacher artifact and strict graph/event join.
- `transport/model.py`: the low-capacity token-by-layer transport operator and
  its single intervention-fidelity objective.
- `transport/experiment.py`: source-group training, structural controls,
  frozen scoring, and physically separate post-hoc evaluation.
- `experiment.py`: recoverable Gate 0/1 orchestration and auditable artifacts.
- `cli.py`: the only Python command-line entrypoint.

For a formal run, use the complete transport runner below.  It first derives
`RUN_ROOT/eligible_train_response_ids.jsonl` from the exact, verified train
attention-cache filenames and then passes that immutable sampling frame to the
teacher.  A standalone Gate 0/1 resume must reuse that generated file; the
pilot deliberately refuses to sample from the full multi-generator dataset:

```bash
ELIGIBLE_RESPONSE_IDS=/path/to/eligible_train_response_ids.jsonl \
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
ELIGIBLE_RESPONSE_IDS=/path/to/eligible_train_response_ids.jsonl \
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
`transport_teacher.jsonl`,
`gate_report.json`, `gate1_shards/`, and `run.log`.  Each completed sample is
atomically saved under `gate1_shards/`; after an interruption, rerun the exact
same output and inputs with `RESUME=1`.  Resume verifies configuration, dataset,
checkpoint, runtime, pair, and shard hashes before skipping finished samples.
`effects.jsonl` keeps the four raw network outcomes as well as derived effects,
so every decomposition is independently recomputable.

```bash
OUTPUT_DIR=/path/to/interrupted/run RESUME=1 \
ELIGIBLE_RESPONSE_IDS=/path/to/eligible_train_response_ids.jsonl \
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

Response-history block rescue is enabled by default because it supplies the
endpoint-level causal target needed by the graph student:

```bash
HISTORY_BLOCK_SIZE=4 MAX_HISTORY_BLOCKS=8 \
ELIGIBLE_RESPONSE_IDS=/path/to/eligible_train_response_ids.jsonl \
bash ./run_ragtruth_cept_pilot.sh
```

An older Gate 0/1 run created with both values equal to zero cannot train the
transport model: it contains TE/DE/ME but no history-endpoint rescue targets.

Run the complete RAGTruth pipeline from the repository root with one command:

```bash
bash ./run_ragtruth_cept_transport.sh
```

The runner requires the attention cache and intervention teacher to share the
exact observer runtime (`transformers==4.52.3`, base Torch version, resolved
compute dtype, and eager attention).  A legacy `4.46.3` cache is rejected
before any expensive teacher call; re-extract both official splits under the
pinned CEPT runtime rather than mixing observer implementations.

The script creates one logical run with `teacher/` and `transport/`, reuses one
content-addressed canonical graph store, trains the five pre-registered
variants (`true`, `rewired`, `mass_only`, `one_hop`, `no_residual`), and freezes predictions
for the official test graphs.  It opens test labels only if the label-free
Gate 2 passes.  Important defaults are inside the shell script; environment
overrides remain optional.

`teacher_fidelity.json` is the first result to inspect.  Source-group paired
bootstrap confidence intervals must show that true incidence beats lag-bucket
rewiring, relation-mass-only, one-hop, no-residual, untrained-gate, and
constant-zero controls on an untouched mechanism holdout.  The holdout must
also pass minimum reliable-positive source/event counts, target-variance,
source-cluster bootstrap component-fidelity, and score-informativeness checks.  Otherwise
`gate2.claim_supported=false`: labels stay closed and frozen predictions remain
available only for label-free diagnosis.

Training now fails before fitting any variant when the train-side teacher
population is not identifiable.  Only a deliberately non-claim CLI pilot may
override this with `--allow-unidentifiable-pilot`; the formal runner never does.

Training and inference use the same deterministic seed operator: the positions
changed by the first equal-token numeric counterfactual candidate.  Graphs with
no valid candidate are explicitly unscorable.  This removes the former
train/test mismatch from sparse numeric seeds to all evidence tokens, but it
also narrows the current result to numeric-intervention-eligible examples.

Training, checkpoint selection, and the final mechanism test are split into
three disjoint `source_id` partitions.  Label-free informativeness is decided
on the train-side mechanism holdout, not on the official test distribution.
Official-test score diagnostics are reported but cannot decide whether labels
are opened.  After labels are opened, `detection_claim.claim_supported` is a
separate, stricter decision: true incidence must beat every registered control
in source-cluster bootstrap intervals for both AUROC and average precision at
token and response levels.  In addition to structural ablations, both
endpoints must beat response length, evidence/query token counts, deployment
seed count, and response-level unknown attention mass.  The token endpoint
must additionally beat normalized token position and token-level unknown
mass.  Every nuisance is registered in both signs so evaluation never chooses
its label orientation post hoc.  It must also satisfy pre-registered minimum
independent-source/class counts and have source-bootstrap lower bounds above
both random AUROC (`0.5`) and the prevalence AUPRC baseline.  Merely beating an
even worse control can never support the detection claim.

The canonical graph index records the attention-cache hash, graph hash, current
model signature, and extractor-recorded model identity.  New manifests may
store the canonical signature directly.  The existing RAGTruth extractor
instead stored the exact filename and SHA-256 of every model-directory file;
an exact inventory match is accepted as verified legacy provenance.  A cache
with neither form of byte-level provenance cannot open labels and must be
re-extracted.

Legacy caches lack the prompt-final predictor row for the first response token.
Unavailable events remain explicit and are excluded from primary metrics;
token- and response-level reports include their evaluation coverage.
They also lack prompt query rows.  Reachable non-seed prompt states therefore
receive an explicit `[0,1]` operational interval after the first layer rather
than being silently treated as zero evidence ancestry.

The current counterfactual protocol is intentionally narrow:
`numeric_digit_surface_preserving_v1`.  It does not yet justify a claim about
general entity, relation, negation, date, or fluent counterfactual generation.
Only the first candidate and a uniformly selected subset of history blocks are
intervened in this pilot.  RAGTruth test labels have also been inspected in
earlier project experiments, so any new AUROC is exploratory rather than a
pristine confirmatory result.
