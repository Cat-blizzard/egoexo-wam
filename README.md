# EgoExo WAM

Feature-level Ego-only WAM MVP for validating whether frozen FACT shared action
tokens are predictable from ego-only history.

This repository is intentionally separate from the FACT tokenizer repo:

- FACT tokenizer repo: `sxh-kk/fact-tokenizer`
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
- slot-wise, per-code, phase, bucket, calibration, and take-leakage diagnostics
- optional metadata fields for take-split and filtered/unfiltered evaluation
- MLP last-frame ablation via `model.arch: mlp_last_frame`
- Ego+Exo privileged teacher via `model.arch: ego_exo_transformer`
- pseudo-label audit script for confidence and code-usage checks
- train / eval / export scripts
- label builder for `sxh-kk/fact-tokenizer` `source.npz + ego_tokens.npz` exports

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

The label builder is aligned to `sxh-kk/fact-tokenizer`. Its tokenizer export
script writes `ego_tokens.npz` with:

```text
indices: [N, transition, action_slot]
soft_probs: [N, transition, action_slot, K]
confidence: [N, transition, action_slot]
```

The WAM builder combines that token NPZ with the original FACT paired-transition
source NPZ, which provides `take_uid`, `timestamp`, and optional metadata. It
then groups samples by take and writes WAM `.pt` episodes.

```bash
python -m wam.data.build_fact_wam_labels \
  --source-npz D:/fact-tokenizer/data/fact_egoexo/splits/.../train_by_take.npz \
  --tokens-npz D:/fact-tokenizer/outputs/fact_tokenizer/.../extracted/ego_tokens.npz \
  --output-root data/fact_wam_labels \
  --split train
```

The source or feature NPZ should contain feature arrays for WAM:

```text
ego_features: [N, D]
exo_features: optional [N, D_exo]  # required only for the ego+exo teacher
take_uid: [N]
timestamp: [N]
```

If features live in separate files, pass `--ego-feature-npz` and
`--exo-feature-npz`. For smoke-only sandbox checks, `--derive-raw-frame-features`
derives tiny RGB delta features from raw `ego`/`exo` arrays, but those are not
DINO features and should not be used as final evidence.

If tokens have not been exported yet, the builder can call the local FACT
extractor:

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

The generated WAM episode format also accepts optional diagnostics:

```python
{
  "take_id": str,
  "bucket": "A_interaction" | "B_loco" | "C_active_view" | "D_scene",
  "bucket_labels": Optional[list[str]],      # length T after label export
  "sampling_weight": Optional[float | Tensor[T]],
  "phase_labels": Optional[Tensor[T]],       # integer phase ids
  "phase_label_names": Optional[dict[int, str]],
}
```

These fields are never fed to the model. They are used for strict heldout-take
evaluation, filtered/unfiltered comparisons, phase diagnostics, and sampling or
loss reweighting.

## Offline Validation Checklist

Use real frozen FACT tokenizer outputs before treating WAM metrics as evidence.
Synthetic smoke runs only validate code paths.

1. Export episode-level labels into split directories such as
   `data/fact_wam_labels/train` and `data/fact_wam_labels/val`.
2. Split by `take_id`, not random transition windows. Prefer the heldout takes
   already used by the FACT tokenizer.
3. Run the same evaluation for unfiltered, filtered FACT-main, and random
   same-size subsets. Configure bucket filters in `data.filter`.
4. Compare WAM against `uniform`, `unigram`, `last_repeat`, and `markov`
   baselines in `eval_results.json`.
5. Inspect `slot_wise`, `code_recall`, `phase_buckets`, `phase_diagnostic`,
   `calibration`, `bucket_groups`, and `take_leakage_probe`.

For ablations, copy `wam/configs/wam_base.yaml` and change:

```yaml
model:
  arch: mlp_last_frame   # or transformer
loss:
  lambda_kl: 0.0         # hard CE only
  lambda_ce: 1.0
```

For soft-KL-only, set `lambda_kl: 1.0` and `lambda_ce: 0.0`. For the default
confidence-weighted objective, keep both non-zero.

Ready-to-run config templates:

```bash
python -m wam.train --config wam/configs/wam_base.yaml --synthetic --output-dir runs/wam_ego_only
python -m wam.train --config wam/configs/wam_mlp_last_frame.yaml --synthetic --output-dir runs/wam_mlp
python -m wam.train --config wam/configs/wam_ego_exo_teacher.yaml --synthetic --output-dir runs/wam_teacher
```

Audit exported pseudo labels before using them for WAM training:

```bash
python -m wam.data.audit_fact_wam_labels \
  --data-root data/fact_wam_labels \
  --split train \
  --output runs/fact_wam_label_audit/train_summary.json \
  --sample-output runs/fact_wam_label_audit/train_phase_samples.csv
```

The audit reports high-confidence token fraction, code usage, code
concentration, effective code count, slot usage, phase counts, and a CSV for
manual phase-label inspection.

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
- filtered FACT-main is more predictable than unfiltered or random same-size data
- phase diagnostics show non-trivial approach/reach/carry/place/release accuracy
- frequency-normalized and rare-code metrics do not collapse to frequent codes
