from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F


def make_synthetic_wam_episode(
    episode_id: str,
    length: int,
    d_feature: int,
    token_slots: int,
    codebook_size: int,
    switch_prob: float,
    noise_std: float,
    seed: int,
) -> Dict:
    gen = torch.Generator().manual_seed(seed)
    state_embeddings = torch.randn(codebook_size, d_feature, generator=gen)
    state_embeddings = state_embeddings / state_embeddings.std().clamp_min(1e-6)

    states = torch.empty(length, dtype=torch.long)
    states[0] = torch.randint(0, codebook_size, (1,), generator=gen)
    for idx in range(1, length):
        if torch.rand((), generator=gen).item() < switch_prob:
            states[idx] = torch.randint(0, codebook_size, (1,), generator=gen)
        else:
            states[idx] = states[idx - 1]

    token_ids = torch.stack(
        [(states + slot) % codebook_size for slot in range(token_slots)],
        dim=-1,
    )
    soft_probs = F.one_hot(token_ids, num_classes=codebook_size).float()
    soft_probs = soft_probs * 0.9 + 0.1 / float(codebook_size)

    stable = torch.ones(length)
    stable[1:] = (states[1:] == states[:-1]).float()
    confidence = (0.35 + 0.6 * stable).unsqueeze(-1).repeat(1, token_slots)

    ego_features = state_embeddings[states] + noise_std * torch.randn(
        length, d_feature, generator=gen
    )
    exo_features = state_embeddings[states] + (noise_std * 0.5) * torch.randn(
        length, d_feature, generator=gen
    )
    timestamps = torch.arange(length).float()
    phase_labels = (states % 5).long()
    bucket_names = ("A_interaction", "B_loco", "C_active_view", "D_scene")
    bucket = bucket_names[seed % len(bucket_names)]

    return {
        "episode_id": episode_id,
        "take_id": episode_id,
        "ego_features": ego_features.float(),
        "exo_features": exo_features.float(),
        "fact_token_ids": token_ids.long(),
        "fact_soft_probs": soft_probs.float(),
        "confidence": confidence.float(),
        "timestamps": timestamps,
        "bucket": bucket,
        "sampling_weight": 1.0,
        "phase_labels": phase_labels,
        "phase_label_names": {
            0: "approach",
            1: "reach",
            2: "carry",
            3: "place",
            4: "release",
        },
    }


def write_synthetic_wam_dataset(
    root: str | Path,
    train_episodes: int = 32,
    val_episodes: int = 8,
    length: int = 128,
    d_feature: int = 64,
    token_slots: int = 2,
    codebook_size: int = 64,
    switch_prob: float = 0.12,
    noise_std: float = 0.2,
    seed: int = 42,
) -> Path:
    root = Path(root)
    for split, num_episodes, seed_offset in (
        ("train", train_episodes, 0),
        ("val", val_episodes, 10_000),
    ):
        split_dir = root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for episode_idx in range(num_episodes):
            episode = make_synthetic_wam_episode(
                episode_id=f"synthetic_{split}_{episode_idx:05d}",
                length=length,
                d_feature=d_feature,
                token_slots=token_slots,
                codebook_size=codebook_size,
                switch_prob=switch_prob,
                noise_std=noise_std,
                seed=seed + seed_offset + episode_idx,
            )
            torch.save(episode, split_dir / f"episode_{episode_idx:05d}.pt")
    return root
