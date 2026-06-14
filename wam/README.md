# Ego-only WAM MVP

This module validates whether frozen FACT shared action tokens are predictable
from ego-only feature history.

Current scope:

- feature-level input only: `ego_features: [T, D]`
- frozen FACT labels only: no tokenizer finetuning
- no raw video, Video DiT, robot adapter, Action Head, or SONIC

## Data Format

Each episode is a `.pt` dictionary:

```python
{
  "episode_id": str,
  "ego_features": Tensor[T, D],
  "fact_token_ids": Tensor[T, S],
  "fact_soft_probs": Tensor[T, S, K],
  "confidence": Tensor[T, S],
  "timestamps": Optional[list],
}
```

For a current index `t`, `FactWamDataset` returns:

```text
ego_features[t - T_hist + 1 : t + 1] -> model input
fact_token_ids[t : t + H_pred]       -> prediction target
```

The model never receives future ego features.

## Synthetic Smoke Run

```bash
python -m wam.train \
  --config wam/configs/wam_base.yaml \
  --synthetic \
  --output-dir runs/wam_synthetic
```

Evaluate:

```bash
python -m wam.eval \
  --config wam/configs/wam_base.yaml \
  --checkpoint runs/wam_synthetic/last.pt \
  --output-dir runs/wam_synthetic/eval
```

Export predictions:

```bash
python -m wam.export_predictions \
  --config wam/configs/wam_base.yaml \
  --checkpoint runs/wam_synthetic/last.pt \
  --output runs/wam_synthetic/predictions.pt
```

## Build Labels From Frozen FACT

Input manifest records point to `.npz` files with:

```text
ego_features: [T, D]
exo_features: [T, D]
timestamps: optional [T]
```

Run:

```bash
python -m wam.data.build_fact_wam_labels \
  --fact-repo-root ../egoexo-fact-tokenizer \
  --fact-checkpoint outputs/fact_tokenizer/fact_synthetic.pt \
  --manifest data/egoexo_feature_episodes/train.json \
  --output-root data/fact_wam_labels \
  --split train
```

## Success Criteria

- train/val loss decreases
- WAM top-5 is higher than random, unigram, and Markov baselines
- horizon metrics degrade sensibly as horizon increases
- high-confidence labels are easier than low-confidence labels
- longer history improves performance in follow-up sweeps
