from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader

from wam.data.audit_fact_wam_labels import audit_split
from wam.data.build_fact_wam_labels import build_wam_episodes_from_fact_npz
from wam.data.fact_wam_dataset import FactWamDataset
from wam.data.synthetic_wam import write_synthetic_wam_dataset
from wam.losses.wam_losses import confidence_weighted_ce, confidence_weighted_kl, wam_loss
from wam.metrics import (
    baseline_logits,
    calibration_metrics,
    compute_wam_metrics,
    fit_code_phase_map,
    fit_token_baselines,
    per_code_recall_metrics,
    phase_bucket_metrics,
    phase_diagnostic_metrics,
    slot_wise_metrics,
)
from wam.models.ego_only_wam import EgoExoWAM, EgoLastFrameMLP, EgoOnlyWAM


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
        "take_id": "take_ep",
        "bucket_labels": ["A_interaction"] * t,
        "sampling_weight": torch.ones(t),
        "phase_labels": torch.arange(t).long() % 3,
    }
    torch.save(episode, train_dir / "episode.pt")

    dataset = FactWamDataset(root, "train", t_hist=3, h_pred=2)
    sample = dataset[0]
    assert sample["t"].item() == 2
    assert torch.equal(sample["ego_features"], episode["ego_features"][0:3])
    assert torch.equal(sample["target_token_ids"], episode["fact_token_ids"][2:4])
    assert torch.equal(sample["context_token_ids"], episode["fact_token_ids"][1])
    assert sample["take_id"] == "take_ep"
    assert sample["bucket"] == "A_interaction"
    assert torch.equal(sample["target_phase_labels"], episode["phase_labels"][2:4])

    filtered = FactWamDataset(root, "train", t_hist=3, h_pred=2, include_buckets=["A_interaction"])
    assert len(filtered) == len(dataset)


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


def test_last_frame_mlp_forward_shape() -> None:
    model = EgoLastFrameMLP(
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


def test_ego_exo_teacher_forward_shape() -> None:
    model = EgoExoWAM(
        d_feature=8,
        d_exo_feature=6,
        d_model=32,
        num_layers=1,
        num_heads=4,
        h_pred=3,
        token_slots=2,
        codebook_size=7,
        t_hist=4,
    )
    out = model(torch.randn(5, 4, 8), torch.randn(5, 4, 6))
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
    assert "slot_0/top1" in slot_wise_metrics(
        logits,
        batch["target_token_ids"],
        batch["target_soft_probs"],
        batch["confidence"],
        topk=(1,),
    )
    code_metrics = per_code_recall_metrics(logits, batch["target_token_ids"], batch["confidence"])
    assert "frequency_normalized_top1" in code_metrics
    cal = calibration_metrics(logits, batch["target_token_ids"], batch["confidence"], bins=[0.0, 0.5, 1.0])
    assert "ece" in cal
    phase_metrics = phase_bucket_metrics(
        logits,
        batch["target_token_ids"],
        batch["target_soft_probs"],
        batch["confidence"],
        batch["target_phase_labels"],
        topk=(1,),
    )
    assert phase_metrics
    phase_map = fit_code_phase_map(dataset, codebook_size=7)
    phase_diag = phase_diagnostic_metrics(logits, batch["target_phase_labels"], batch["confidence"], phase_map)
    assert "slot_accuracy" in phase_diag

    baselines = fit_token_baselines(dataset, codebook_size=7)
    b_logits = baseline_logits("markov", baselines, batch["context_token_ids"], h_pred=3, codebook_size=7)
    assert b_logits.shape == logits.shape

    summary, samples = audit_split(
        root,
        "train",
        high_confidence_threshold=0.8,
        sample_count=4,
        seed=1,
    )
    assert summary["episode_count"] == 2
    assert "high_confidence_token_fraction" in summary
    assert samples


def test_build_wam_labels_from_sxh_fact_tokenizer_npz(tmp_path) -> None:
    source_npz = tmp_path / "source.npz"
    tokens_npz = tmp_path / "ego_tokens.npz"
    output_root = tmp_path / "wam_labels"

    take_uid = np.asarray(["take_b", "take_a", "take_a", "take_b"])
    timestamp = np.asarray([1.0, 2.0, 1.0, 0.0], dtype=np.float32)
    ego_features = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    exo_features = np.arange(4 * 3, dtype=np.float32).reshape(4, 3)
    np.savez_compressed(
        source_npz,
        take_uid=take_uid,
        timestamp=timestamp,
        sample_id=np.asarray(["b1", "a2", "a1", "b0"]),
        ego_features=ego_features,
        exo_features=exo_features,
        phase_labels=np.asarray([1, 2, 3, 4], dtype=np.int64),
        bucket_labels=np.asarray(["B_loco", "A_interaction", "A_interaction", "B_loco"]),
        sampling_weight=np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float32),
    )

    indices = np.asarray([[[1, 2]], [[3, 4]], [[5, 6]], [[0, 1]]], dtype=np.int64)
    soft_probs = np.zeros((4, 1, 2, 7), dtype=np.float32)
    for sample in range(4):
        for slot in range(2):
            soft_probs[sample, 0, slot, indices[sample, 0, slot]] = 1.0
    confidence = np.ones((4, 1, 2), dtype=np.float32)
    np.savez_compressed(tokens_npz, indices=indices, soft_probs=soft_probs, confidence=confidence)

    report = build_wam_episodes_from_fact_npz(
        source_npz=source_npz,
        tokens_npz=tokens_npz,
        output_root=output_root,
        split="train",
    )
    assert report["episodes_written"] == 2
    dataset = FactWamDataset(output_root, "train", t_hist=2, h_pred=1)
    assert dataset.token_slots == 2
    assert dataset.codebook_size == 7
    take_a = next(episode for episode in dataset.episodes if episode["take_id"] == "take_a")
    assert take_a["sample_ids"] == ["a1", "a2"]
    assert torch.equal(take_a["fact_token_ids"], torch.tensor([[5, 6], [3, 4]]))
