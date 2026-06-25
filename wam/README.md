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
  "take_id": Optional[str],
  "ego_features": Tensor[T, D],
  "exo_features": Optional[Tensor[T, D_exo]],
  "fact_token_ids": Tensor[T, S],
  "fact_soft_probs": Tensor[T, S, K],
  "confidence": Tensor[T, S],
  "timestamps": Optional[list],
  "bucket": Optional[str],
  "bucket_labels": Optional[list[str]],
  "sampling_weight": Optional[float | Tensor[T]],
  "phase_labels": Optional[Tensor[T]],
  "phase_label_names": Optional[dict[int, str]],
}
```

For a current index `t`, `FactWamDataset` returns:

```text
ego_features[t - T_hist + 1 : t + 1] -> model input
fact_token_ids[t : t + H_pred]       -> prediction target
```

The model never receives future ego features.

Optional metadata is not model input. It is used for heldout-take validation,
filtered/unfiltered comparisons, phase diagnostics, calibration checks, and
sampling or loss reweighting.

`exo_features` is consumed only by the privileged teacher config
`model.arch: ego_exo_transformer`.

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

Teacher and ablation configs:

```bash
python -m wam.train \
  --config wam/configs/wam_mlp_last_frame.yaml \
  --synthetic \
  --output-dir runs/wam_mlp

python -m wam.train \
  --config wam/configs/wam_ego_exo_teacher.yaml \
  --synthetic \
  --output-dir runs/wam_teacher
```

Export predictions:

```bash
python -m wam.export_predictions \
  --config wam/configs/wam_base.yaml \
  --checkpoint runs/wam_synthetic/last.pt \
  --output runs/wam_synthetic/predictions.pt
```

## Build Labels From Frozen FACT

This module is aligned to `sxh-kk/fact-tokenizer`, whose extractor writes
`ego_tokens.npz` containing:

```text
indices: [N, transition, action_slot]
soft_probs: [N, transition, action_slot, K]
confidence: [N, transition, action_slot]
```

Combine that token file with the original FACT paired-transition NPZ:

```bash
python -m wam.data.build_fact_wam_labels \
  --source-npz D:/fact-tokenizer/data/fact_egoexo/splits/.../train_by_take.npz \
  --tokens-npz D:/fact-tokenizer/outputs/fact_tokenizer/.../extracted/ego_tokens.npz \
  --output-root data/fact_wam_labels \
  --split train
```

The source or feature NPZ must provide `ego_features: [N, D]` for real WAM
training. `exo_features: [N, D_exo]` is optional and used only by the
ego+exo teacher. Use `--ego-feature-npz` / `--exo-feature-npz` if features live
outside the source NPZ.

For smoke-only sandbox checks, `--derive-raw-frame-features` derives small RGB
delta features from raw `ego`/`exo` arrays. These are not DINO features and
should not be used for final evidence.

The builder can also call the local FACT extractor first:

```bash
python -m wam.data.build_fact_wam_labels \
  --extract-tokens \
  --fact-repo-root D:/fact-tokenizer \
  --fact-checkpoint D:/fact-tokenizer/outputs/fact_tokenizer/.../fact_tokenizer.ckpt \
  --source-npz D:/fact-tokenizer/data/fact_egoexo/splits/.../train_by_take.npz \
  --output-root data/fact_wam_labels \
  --split train \
  --skip-recon-metrics
```

Audit exported pseudo labels:

```bash
python -m wam.data.audit_fact_wam_labels \
  --data-root data/fact_wam_labels \
  --split train \
  --output runs/fact_wam_label_audit/train_summary.json \
  --sample-output runs/fact_wam_label_audit/train_phase_samples.csv
```

## Success Criteria

- train/val loss decreases
- WAM top-5 is higher than random, unigram, and Markov baselines
- horizon metrics degrade sensibly as horizon increases
- high-confidence labels are easier than low-confidence labels
- longer history improves performance in follow-up sweeps
- slot-wise and frequency-normalized code metrics do not collapse to frequent codes
- phase diagnostics are non-trivial on approach/reach/carry/place/release labels
- filtered FACT-main beats unfiltered or random same-size data on heldout takes
