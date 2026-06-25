from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

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
        include_buckets: Iterable[str] | None = None,
        exclude_buckets: Iterable[str] | None = None,
        min_confidence: float | None = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.t_hist = t_hist
        self.h_pred = h_pred
        self.include_buckets = set(include_buckets or [])
        self.exclude_buckets = set(exclude_buckets or [])
        self.min_confidence = min_confidence
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
                if self._keep_window(episode, current_t):
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
        exo_features = episode.get("exo_features")
        token_ids = episode["fact_token_ids"]
        soft_probs = episode["fact_soft_probs"]
        confidence = episode["confidence"]
        if ego_features.ndim != 2:
            raise ValueError(f"{path}: ego_features must be [T, D]")
        if exo_features is not None:
            if exo_features.ndim != 2:
                raise ValueError(f"{path}: exo_features must be [T, D_exo]")
            if exo_features.shape[0] != ego_features.shape[0]:
                raise ValueError(f"{path}: exo_features must share T with ego_features")
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
        if "phase_labels" in episode:
            phase_labels = torch.as_tensor(episode["phase_labels"])
            if phase_labels.ndim != 1 or phase_labels.shape[0] != ego_features.shape[0]:
                raise ValueError(f"{path}: phase_labels must be [T]")
        if "sampling_weight" in episode:
            sampling_weight = torch.as_tensor(episode["sampling_weight"])
            if sampling_weight.ndim > 0 and sampling_weight.shape[0] != ego_features.shape[0]:
                raise ValueError(f"{path}: sampling_weight must be scalar or start with T")
        if "bucket_labels" in episode and len(episode["bucket_labels"]) != ego_features.shape[0]:
            raise ValueError(f"{path}: bucket_labels must have length T")

    def _bucket_at(self, episode: Dict[str, Any], current_t: int) -> str:
        if "bucket_labels" in episode:
            return str(episode["bucket_labels"][current_t])
        return str(episode.get("bucket", "unknown"))

    def _sampling_weight_at(self, episode: Dict[str, Any], current_t: int) -> Tensor:
        if "sampling_weight" not in episode:
            return torch.tensor(1.0, dtype=torch.float32)
        sampling_weight = torch.as_tensor(episode["sampling_weight"], dtype=torch.float32)
        if sampling_weight.ndim == 0:
            return sampling_weight
        value = sampling_weight[current_t]
        return value.float().mean() if value.ndim > 0 else value.float()

    def _phase_window(self, episode: Dict[str, Any], current_t: int, target_end: int) -> Tensor:
        if "phase_labels" not in episode:
            return torch.full((self.h_pred,), -1, dtype=torch.long)
        return torch.as_tensor(episode["phase_labels"][current_t:target_end], dtype=torch.long)

    def _timestamp_at(self, episode: Dict[str, Any], current_t: int) -> Tensor:
        if "timestamps" not in episode or episode["timestamps"] is None:
            return torch.tensor(float("nan"), dtype=torch.float32)
        return torch.as_tensor(episode["timestamps"][current_t], dtype=torch.float32)

    def _keep_window(self, episode: Dict[str, Any], current_t: int) -> bool:
        bucket = self._bucket_at(episode, current_t)
        if self.include_buckets and bucket not in self.include_buckets:
            return False
        if bucket in self.exclude_buckets:
            return False
        if self.min_confidence is not None:
            target_end = current_t + self.h_pred
            window_conf = episode["confidence"][current_t:target_end].float().mean().item()
            if window_conf < self.min_confidence:
                return False
        return True

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        episode_idx, current_t = self.index[idx]
        episode = self.episodes[episode_idx]
        hist_start = current_t - self.t_hist + 1
        target_end = current_t + self.h_pred

        sample = {
            "ego_features": episode["ego_features"][hist_start : current_t + 1].float(),
            "target_token_ids": episode["fact_token_ids"][current_t:target_end].long(),
            "target_soft_probs": episode["fact_soft_probs"][current_t:target_end].float(),
            "confidence": episode["confidence"][current_t:target_end].float(),
            "context_token_ids": episode["fact_token_ids"][current_t - 1].long(),
            "sample_weight": self._sampling_weight_at(episode, current_t),
            "target_phase_labels": self._phase_window(episode, current_t, target_end),
            "episode_id": episode["episode_id"],
            "take_id": str(episode.get("take_id", episode["episode_id"])),
            "bucket": self._bucket_at(episode, current_t),
            "t": torch.tensor(current_t, dtype=torch.long),
            "timestamp": self._timestamp_at(episode, current_t),
        }
        if "exo_features" in episode:
            sample["exo_features"] = episode["exo_features"][hist_start : current_t + 1].float()
        return sample

    @property
    def d_feature(self) -> int:
        return int(self.episodes[0]["ego_features"].shape[-1])

    @property
    def d_exo_feature(self) -> int | None:
        if "exo_features" not in self.episodes[0]:
            return None
        return int(self.episodes[0]["exo_features"].shape[-1])

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
