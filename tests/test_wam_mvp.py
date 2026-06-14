from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from wam.data.fact_wam_dataset import FactWamDataset
from wam.data.synthetic_wam import write_synthetic_wam_dataset
from wam.losses.wam_losses import confidence_weighted_ce, confidence_weighted_kl, wam_loss
from wam.metrics import baseline_logits, compute_wam_metrics, fit_token_baselines
from wam.models.ego_only_wam import EgoOnlyWAM


def test_fact_wam_dataset_window_slicing(tmp_path) -> None:
    root = tmp_path / "wam"
    train_dir = root / "train"
    train_dir.mkdir(parents=True)
    t = 10
    d = 3
    s = 2
    k = 5
    episode = {
        "episode_id": "ep",
        "ego_features": torch.arange(t * d).view(t, d).float(),
        "fact_token_ids": torch.arange(t * s).view(t, s).long() % k,
        "fact_soft_probs": torch.ones(t, s, k).float() / k,
        "confidence": torch.ones(t, s).float(),
    }
    torch.save(episode, train_dir / "episode.pt")

    dataset = FactWamDataset(root, "train", t_hist=3, h_pred=2)
    sample = dataset[0]
    assert sample["t"].item() == 2
    assert torch.equal(sample["ego_features"], episode["ego_features"][0:3])
    assert torch.equal(sample["target_token_ids"], episode["fact_token_ids"][2:4])
    assert torch.equal(sample["context_token_ids"], episode["fact_token_ids"][1])


def test_ego_only_wam_forward_shape() -> None:
    model = EgoOnlyWAM(
        d_feature=8,
        d_model=32,
        num_layers=2,
        num_heads=4,
        h_pred=3,
        token_slots=2,
        codebook_size=7,
        t_hist=4,
    )
    out = model(torch.randn(5, 4, 8))
    assert out["logits"].shape == (5, 3, 2, 7)
    assert out["probs"].shape == (5, 3, 2, 7)
    assert out["hidden"].shape == (5, 3, 32)


def test_wam_losses_are_finite_with_zero_confidence() -> None:
    logits = torch.randn(2, 3, 2, 7)
    target_ids = torch.randint(0, 7, (2, 3, 2))
    target_probs = torch.nn.functional.one_hot(target_ids, num_classes=7).float()
    confidence = torch.zeros(2, 3, 2)
    kl = confidence_weighted_kl(logits, target_probs, confidence)
    ce = confidence_weighted_ce(logits, target_ids, confidence)
    total, parts = wam_loss(logits, target_probs, target_ids, confidence)
    assert torch.isfinite(kl)
    assert torch.isfinite(ce)
    assert torch.isfinite(total)
    for value in parts.values():
        assert torch.isfinite(value)


def test_metrics_and_baselines_run_on_synthetic_dataset(tmp_path) -> None:
    root = write_synthetic_wam_dataset(
        tmp_path / "synthetic",
        train_episodes=2,
        val_episodes=1,
        length=24,
        d_feature=8,
        token_slots=2,
        codebook_size=7,
        seed=1,
    )
    dataset = FactWamDataset(root, "train", t_hist=4, h_pred=3)
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))
    model = EgoOnlyWAM(
        d_feature=8,
        d_model=32,
        num_layers=1,
        num_heads=4,
        h_pred=3,
        token_slots=2,
        codebook_size=7,
        t_hist=4,
    )
    logits = model(batch["ego_features"])["logits"]
    metrics = compute_wam_metrics(
        logits,
        batch["target_token_ids"],
        batch["target_soft_probs"],
        batch["confidence"],
        topk=(1, 5),
    )
    assert "top1" in metrics
    assert "horizon_1/top1" in metrics

    baselines = fit_token_baselines(dataset, codebook_size=7)
    b_logits = baseline_logits("markov", baselines, batch["context_token_ids"], h_pred=3, codebook_size=7)
    assert b_logits.shape == logits.shape
