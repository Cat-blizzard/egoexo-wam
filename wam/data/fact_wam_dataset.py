from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset


REQUIRED_EPISODE_KEYS = {
    "episode_id",
    "ego_features",
    "fact_token_ids",
    "fact_soft_probs",
    "confidence",
}


class FactWamDataset(Dataset):
    """Windowed dataset for causal Ego-only WAM training.

    Episode files are .pt dictionaries with:
      ego_features: [T, D]
      fact_token_ids: [T, S]
      fact_soft_probs: [T, S, K]
      confidence: [T, S]

    For a current index t, the model input is ego_features[t_hist_window]
    ending at t, and the target is fact tokens from t through t + h_pred - 1.
    The target token at t is a future-transition label produced offline by the
    frozen FACT tokenizer; it is never part of the model input.
    """

    def __init__(
        self,
        root: str | Path,
        split: str,
        t_hist: int,
        h_pred: int,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.t_hist = t_hist
        self.h_pred = h_pred
        self.episode_paths = sorted((self.root / split).glob("*.pt"))
        if not self.episode_paths:
            raise FileNotFoundError(f"No .pt WAM episodes found under {self.root / split}")

        self.episodes: List[Dict[str, Any]] = []
        self.index: List[Tuple[int, int]] = []
        for episode_idx, episode_path in enumerate(self.episode_paths):
            episode = torch.load(episode_path, map_location="cpu", weights_only=False)
            self._validate_episode(episode, episode_path)
            self.episodes.append(episode)
            length = int(episode["ego_features"].shape[0])
            start_t = max(t_hist - 1, 1)
            end_t_exclusive = length - h_pred + 1
            for current_t in range(start_t, end_t_exclusive):
                self.index.append((episode_idx, current_t))
        if not self.index:
            raise ValueError(
                f"No valid WAM windows for split={split}; check t_hist={t_hist}, h_pred={h_pred}."
            )

    @staticmethod
    def _validate_episode(episode: Dict[str, Any], path: Path) -> None:
        missing = REQUIRED_EPISODE_KEYS - set(episode)
        if missing:
            raise KeyError(f"{path} missing keys: {sorted(missing)}")
        ego_features = episode["ego_features"]
        token_ids = episode["fact_token_ids"]
        soft_probs = episode["fact_soft_probs"]
        confidence = episode["confidence"]
        if ego_features.ndim != 2:
            raise ValueError(f"{path}: ego_features must be [T, D]")
        if token_ids.ndim != 2:
            raise ValueError(f"{path}: fact_token_ids must be [T, S]")
        if soft_probs.ndim != 3:
            raise ValueError(f"{path}: fact_soft_probs must be [T, S, K]")
        if confidence.ndim != 2:
            raise ValueError(f"{path}: confidence must be [T, S]")
        if not (
            ego_features.shape[0]
            == token_ids.shape[0]
            == soft_probs.shape[0]
            == confidence.shape[0]
        ):
            raise ValueError(f"{path}: all sequence tensors must share T")
        if token_ids.shape != confidence.shape or token_ids.shape != soft_probs.shape[:2]:
            raise ValueError(f"{path}: token/confidence/soft label slot shapes do not match")

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_idx, current_t = self.index[idx]
        episode = self.episodes[episode_idx]
        hist_start = current_t - self.t_hist + 1
        target_end = current_t + self.h_pred

        return {
            "ego_features": episode["ego_features"][hist_start : current_t + 1].float(),
            "target_token_ids": episode["fact_token_ids"][current_t:target_end].long(),
            "target_soft_probs": episode["fact_soft_probs"][current_t:target_end].float(),
            "confidence": episode["confidence"][current_t:target_end].float(),
            "context_token_ids": episode["fact_token_ids"][current_t - 1].long(),
            "episode_id": episode["episode_id"],
            "t": torch.tensor(current_t, dtype=torch.long),
        }

    @property
    def d_feature(self) -> int:
        return int(self.episodes[0]["ego_features"].shape[-1])

    @property
    def token_slots(self) -> int:
        return int(self.episodes[0]["fact_token_ids"].shape[-1])

    @property
    def codebook_size(self) -> int:
        return int(self.episodes[0]["fact_soft_probs"].shape[-1])


def move_wam_batch_to_device(batch: Dict[str, Any], device: torch.device | str) -> Dict[str, Any]:
    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if isinstance(value, Tensor) else value
    return moved
