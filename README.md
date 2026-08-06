# Prompt-Anchored Topology Flow

PATF is a label-free hallucination detector built from saved LLM attention.
It measures how response tokens remain connected to the prompt, constructs
mechanism-aligned counterfactuals, and learns an anomaly score over the
layer-wise topology trajectory.

## Structure

```text
main.py                 command-line entry
configs/patf.json       method and training parameters
scripts/run_ragtruth.sh one-command RAGTruth experiment
patf/
  data.py               sparse attention cache loader
  topology.py           prompt-anchored topology features
  augment.py            counterfactual topology erosion
  features.py           incremental feature cache
  model.py              trajectory ranker
  trainer.py            training and scoring
  evaluation.py         response-level AUROC/AUPRC
  experiment.py         end-to-end experiment
```

The main pipeline is intentionally short:

```python
train_features = prepare_features(train_attention)
checkpoint = train_ranker(train_features)
test_features = prepare_features(test_attention)
predictions = score_features(test_features, checkpoint)
report = evaluate(predictions)
```

## Run

```bash
bash run_patf.sh
```

Default server paths can be overridden without editing code:

```bash
ATTENTION_ROOT=/path/to/attention_cache \
RAGTRUTH_ROOT=/path/to/RAGTruth \
OUTPUT_DIR=/path/to/output \
DEVICE=cuda \
EPOCHS=80 \
bash run_patf.sh
```

## Outputs

```text
output/
  config.json
  features/train/*.features.pt
  features/test/*.features.pt
  model/model.pt
  model/history.json
  predictions.jsonl
  evaluation.json
```

Feature extraction is incremental. Existing feature files are reused when the
method configuration matches, so interrupted runs can resume without repeating
completed samples.

## Data

PATF reads the formal sparse response-attention cache:

```text
ragtruth-all-layers-all-heads-sparse-response-csr-v1
```

Labels stored in the same `.pt` file are ignored. RAGTruth labels are opened
only by `patf/evaluation.py`, after test scores have been written.

Legacy experiments remain in their original directories for reproducibility;
they are not imported by PATF.
