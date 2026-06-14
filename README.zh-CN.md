# EgoExo WAM 中文说明

本仓库用于实现 **feature-level Ego-only WAM MVP**，目标是验证：

```text
ego history feature -> Ego-only WAM -> future FACT shared action token distribution
```

它是 FACT tokenizer 之后的下一阶段验证代码。当前不处理 raw video、不接 Video DiT、不接机器人、不接 SONIC，也不训练 Action Head。

## 1. 仓库定位

当前项目和 FACT tokenizer 仓库分开维护：

```text
egoexo-fact-tokenizer
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
    "ego_features": Tensor[T, D],
    "fact_token_ids": Tensor[T, S],
    "fact_soft_probs": Tensor[T, S, K],
    "confidence": Tensor[T, S],
    "timestamps": Optional[list],
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

如果已经有 frozen FACT checkpoint 和 episode-level ego/exo features，可以用：

```bash
python -m wam.data.build_fact_wam_labels \
  --fact-repo-root ../egoexo-fact-tokenizer \
  --fact-checkpoint ../egoexo-fact-tokenizer/outputs/fact_tokenizer/fact_synthetic.pt \
  --manifest data/egoexo_feature_episodes/train.json \
  --output-root data/fact_wam_labels \
  --split train
```

也可以不用 `--fact-repo-root`，改用环境变量：

```bash
export FACT_REPO_ROOT=../egoexo-fact-tokenizer
```

Windows PowerShell：

```powershell
$env:FACT_REPO_ROOT = "..\egoexo-fact-tokenizer"
```

manifest 中每条记录指向一个 `.npz` 文件：

```json
{
  "episode_id": "take_000001",
  "feature_path": "features/take_000001.npz"
}
```

`.npz` 至少包含：

```text
ego_features: [T, D]
exo_features: [T, D]
timestamps: optional [T]
```

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

## 9. 成功标准

初步验证通过的最低标准：

1. train / validation loss 正常下降；
2. WAM top-5 高于 random / unigram / Markov baseline；
3. horizon 越远，accuracy 曲线合理下降；
4. high-confidence bucket 的 accuracy 高于 low-confidence bucket；
5. 增加 `T_hist` 后性能提升；
6. `confidence-weighted KL + CE` 优于 hard CE only 或 unweighted KL。

如果这些成立，说明 frozen FACT shared action token 具有基本 ego-predictability，可以进入下一阶段：

```text
Video Dynamics Branch / Video DiT
Action Head
SONIC alignment
robot data validation
```

如果这些不成立，应回到 FACT tokenizer 阶段，检查 token 是否过度编码 view/object/background，或者是否无法从 ego history 因果预测。
