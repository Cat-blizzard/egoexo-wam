# EgoExo WAM

Feature-level Ego-only WAM MVP for validating whether frozen FACT shared action
tokens are predictable from ego-only history.

This repository is intentionally separate from the FACT tokenizer repo:

- FACT tokenizer repo: `Cat-blizzard/egoexo-fact-tokenizer`
- WAM repo: `Cat-blizzard/egoexo-wam`

The WAM model consumes precomputed features and frozen FACT labels. It does not
train or finetune the FACT tokenizer.

## Scope

Implemented:

- episode-level WAM dataset
- synthetic WAM data generator
- Ego-only causal Transformer baseline
- confidence-weighted KL + CE loss
- top-k, horizon, confidence bucket, unigram, Markov, and last-repeat metrics
- train / eval / export scripts
- optional label builder that calls a local FACT repo via `--fact-repo-root`

Not implemented:

- raw video encoder
- Video DiT
- robot adapter
- Action Head
- SONIC
- end-to-end FACT + WAM finetuning

## Setup

```bash
pip install -e .
```

## Synthetic Smoke Run

```bash
python -m wam.train \
  --config wam/configs/wam_base.yaml \
  --synthetic \
  --output-dir runs/wam_synthetic

python -m wam.eval \
  --config wam/configs/wam_base.yaml \
  --checkpoint runs/wam_synthetic/last.pt \
  --output-dir runs/wam_synthetic/eval

python -m wam.export_predictions \
  --config wam/configs/wam_base.yaml \
  --checkpoint runs/wam_synthetic/last.pt \
  --output runs/wam_synthetic/predictions.pt
```

## Build WAM Labels From FACT

The label builder is optional and expects a local checkout of the FACT tokenizer
repo. Pass it explicitly:

```bash
python -m wam.data.build_fact_wam_labels \
  --fact-repo-root ../egoexo-fact-tokenizer \
  --fact-checkpoint ../egoexo-fact-tokenizer/outputs/fact_tokenizer/fact_synthetic.pt \
  --manifest data/egoexo_feature_episodes/train.json \
  --output-root data/fact_wam_labels \
  --split train
```

The manifest points to `.npz` files containing:

```text
ego_features: [T, D]
exo_features: [T, D]
timestamps: optional [T]
```

## Tests

```bash
python -m pytest tests/test_wam_mvp.py -q
```

## Success Criteria

- train and validation loss decrease
- WAM top-5 is higher than random, unigram, and Markov baselines
- horizon metrics degrade sensibly as prediction horizon increases
- high-confidence labels are easier than low-confidence labels
- longer history improves performance in follow-up sweeps
