# EgoExo WAM 中文说明

本仓库用于实现 **feature-level Ego-only WAM MVP**，目标是验证：

```text
ego history feature -> Ego-only WAM -> future FACT shared action token distribution
```

它是 FACT tokenizer 之后的下一阶段验证代码。当前不处理 raw video、不接 Video DiT、不接机器人、不接 SONIC，也不训练 Action Head。

## 1. 仓库定位

当前项目和 FACT tokenizer 仓库分开维护：

```text
sxh-kk/fact-tokenizer
  负责 FACT tokenizer：
  paired ego/exo feature -> shared action token

egoexo-wam
  负责 Ego-only WAM：
  ego feature history -> future FACT token distribution
```

WAM 只消费 frozen FACT tokenizer 生成的 token label，不反传、不 finetune FACT tokenizer。

## 2. 当前能验证什么

当前代码可以验证：

- ego-only history 是否能预测 future FACT token；
- WAM 是否明显优于 random / unigram / Markov / last-repeat baseline；
- prediction horizon 越远，accuracy 是否合理下降；
- high-confidence token label 是否更容易预测；
- `confidence-weighted KL + CE` 是否能正常训练；
- slot-wise / per-code / rare-code 指标是否塌缩到高频 code；
- phase / bucket / calibration / take-leakage 诊断是否正常；
- ego-only student 与 ego+exo privileged teacher 是否存在离线性能差距；
- WAM 预测结果是否可以导出给后续 Action Head 使用。

这一步的核心问题是：

```text
FACT shared action token 是否具备 ego-predictability？
```

如果 WAM 连 Markov baseline 都打不过，说明 FACT token 对 ego-only 预测不友好，后续 Video DiT / Action Head 都不应该急着做。

## 3. 当前不做什么

第一版明确不做：

- raw video encoder；
- DINO / Omnivore / VideoMAE 在线提特征；
- Video DiT；
- contact auxiliary head；
- subgoal embedding；
- robot ego adapter；
- Action Head；
- SONIC；
- tokenizer + WAM end-to-end 联合训练。

第一版输入只用：

```text
ego_features: [T, D]
```

输出只预测：

```text
future FACT token distribution: [H_pred, S, K]
```

## 4. 数据格式

每个 episode 是一个 `.pt` 文件，格式如下：

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

符号说明：

```text
T: episode 长度
D: ego feature 维度
S: FACT token slot 数
K: shared action codebook size
```

`FactWamDataset` 的切窗方式：

```text
输入:
ego_features[t - T_hist + 1 : t + 1]

预测目标:
fact_token_ids[t : t + H_pred]
fact_soft_probs[t : t + H_pred]
confidence[t : t + H_pred]
```

注意：模型输入只包含当前和历史 ego feature，不包含 future ego feature。

`take_id`、`bucket`、`sampling_weight` 和 `phase_labels` 只用于 split、筛选、loss reweighting 和诊断评估，不会输入模型。

`exo_features` 只在 `model.arch: ego_exo_transformer` 的 privileged teacher 对照实验中使用；默认 ego-only WAM 不读取 exo。

## 5. 安装

在仓库根目录运行：

```bash
pip install -e .
```

开发测试需要：

```bash
pip install -e ".[dev]"
```

## 6. Synthetic Smoke Run

当前还没有 Ego-Exo4D 数据时，可以先跑 synthetic 验证：

```bash
python -m wam.train \
  --config wam/configs/wam_base.yaml \
  --synthetic \
  --output-dir runs/wam_synthetic
```

评估：

```bash
python -m wam.eval \
  --config wam/configs/wam_base.yaml \
  --checkpoint runs/wam_synthetic/last.pt \
  --output-dir runs/wam_synthetic/eval
```

导出预测：

```bash
python -m wam.export_predictions \
  --config wam/configs/wam_base.yaml \
  --checkpoint runs/wam_synthetic/last.pt \
  --output runs/wam_synthetic/predictions.pt
```

## 7. 从 FACT tokenizer 构建 WAM labels

当前 builder 对齐的是 `sxh-kk/fact-tokenizer`。它的 `scripts/extract_fact_tokens.py`
会导出 `ego_tokens.npz`：

```text
indices: [N, transition, action_slot]
soft_probs: [N, transition, action_slot, K]
confidence: [N, transition, action_slot]
```

WAM builder 会把这个 token NPZ 和原始 FACT paired-transition source NPZ 合并，
按 `take_uid` 分组、按 `timestamp` 排序，然后写成 WAM episode `.pt`。

```bash
python -m wam.data.build_fact_wam_labels \
  --source-npz D:/fact-tokenizer/data/fact_egoexo/splits/.../train_by_take.npz \
  --tokens-npz D:/fact-tokenizer/outputs/fact_tokenizer/.../extracted/ego_tokens.npz \
  --output-root data/fact_wam_labels \
  --split train
```

source NPZ 或单独的 feature NPZ 应提供：

```text
ego_features: [N, D]
exo_features: optional [N, D_exo]
take_uid: [N]
timestamp: [N]
```

如果 feature 不在 source NPZ 里，使用 `--ego-feature-npz` / `--exo-feature-npz`。
`exo_features` 只给 `ego_exo_transformer` teacher 用。

如果还没有导出 tokens，可以让 WAM builder 调用本地 FACT extractor：

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

`--derive-raw-frame-features` 只能用于 smoke sandbox：它会从 raw `ego`/`exo`
视频数组派生很小的 RGB delta feature，不是 DINO feature，不能作为最终实验结论。

## 8. 运行测试

```bash
python -m pytest tests/test_wam_mvp.py -q
```

当前 smoke test 覆盖：

- dataset window slicing；
- model forward shape；
- loss finite；
- metrics / baselines；
- synthetic WAM data generation。

## 9. 离线验证流程

正式 WAM 结论必须基于真实 frozen FACT tokenizer 输出；synthetic run 只能证明代码路径可用。

建议最小流程：

1. 用 v6b / filtered-v6x / 过 gate 的 tokenizer 导出 episode-level `ego_features`、`fact_token_ids`、`fact_soft_probs`、`confidence`、`take_id` 和 `timestamp`。
2. 按 `take_id` 做 train / heldout split，不做随机 transition split。
3. 分别跑 unfiltered、filtered FACT-main、random same-size，以及需要的 bucket 子集。
4. 对照 `uniform`、`unigram`、`last_repeat`、`markov` baseline。
5. 检查 `slot_wise`、`code_recall`、`phase_buckets`、`phase_diagnostic`、`calibration`、`bucket_groups` 和 `take_leakage_probe`。

最小 ablation 可以复制 `wam/configs/wam_base.yaml` 后修改：

```yaml
model:
  arch: mlp_last_frame   # 或 transformer
loss:
  lambda_kl: 0.0         # hard CE only
  lambda_ce: 1.0
```

soft-KL-only 使用 `lambda_kl: 1.0`、`lambda_ce: 0.0`。默认配置是 confidence-weighted KL + CE。

可直接使用的最小配置：

```bash
python -m wam.train --config wam/configs/wam_base.yaml --synthetic --output-dir runs/wam_ego_only
python -m wam.train --config wam/configs/wam_mlp_last_frame.yaml --synthetic --output-dir runs/wam_mlp
python -m wam.train --config wam/configs/wam_ego_exo_teacher.yaml --synthetic --output-dir runs/wam_teacher
```

unpaired ego pseudo-tokenization 或真实 FACT label 导出后，先做 label audit：

```bash
python -m wam.data.audit_fact_wam_labels \
  --data-root data/fact_wam_labels \
  --split train \
  --output runs/fact_wam_label_audit/train_summary.json \
  --sample-output runs/fact_wam_label_audit/train_phase_samples.csv
```

audit 会统计 high-confidence token 比例、code usage、code 集中度、effective code count、slot usage、phase counts，并导出人工 phase 抽检 CSV。

## 10. 成功标准

初步验证通过的最低标准：

1. train / validation loss 正常下降；
2. WAM top-5 高于 random / unigram / Markov baseline；
3. horizon 越远，accuracy 曲线合理下降；
4. high-confidence bucket 的 accuracy 高于 low-confidence bucket；
5. 增加 `T_hist` 后性能提升；
6. filtered FACT-main 比 unfiltered / random same-size 更可预测；
7. phase diagnostic 对 approach / reach / carry / place / release 有非平凡预测；
8. code frequency-normalized / rare-code 指标不塌缩到高频 code；
9. `confidence-weighted KL + CE` 优于 hard CE only 或 unweighted KL。

如果这些成立，说明 frozen FACT shared action token 具有基本 ego-predictability，可以进入下一阶段：

```text
Video Dynamics Branch / Video DiT
Action Head
SONIC alignment
robot data validation
```

如果这些不成立，应回到 FACT tokenizer 阶段，检查 token 是否过度编码 view/object/background，或者是否无法从 ego history 因果预测。
