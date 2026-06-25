from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class EgoOnlyWAM(nn.Module):
    """Feature-level causal WAM that predicts future FACT token distributions."""

    def __init__(
        self,
        d_feature: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        h_pred: int,
        token_slots: int,
        codebook_size: int,
        t_hist: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.h_pred = h_pred
        self.token_slots = token_slots
        self.codebook_size = codebook_size
        self.t_hist = t_hist

        self.feature_proj = nn.Linear(d_feature, d_model)
        self.history_pos = nn.Parameter(torch.zeros(1, t_hist, d_model))
        self.horizon_queries = nn.Parameter(torch.randn(1, h_pred, d_model) * 0.02)
        self.horizon_pos = nn.Parameter(torch.zeros(1, h_pred, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, token_slots * codebook_size)

    def _attention_mask(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        length = self.t_hist + self.h_pred
        mask = torch.zeros(length, length, device=device, dtype=dtype)

        history_causal_block = torch.triu(
            torch.ones(self.t_hist, self.t_hist, device=device, dtype=torch.bool),
            diagonal=1,
        )
        mask[: self.t_hist, : self.t_hist] = history_causal_block.to(dtype) * torch.finfo(dtype).min
        mask[: self.t_hist, self.t_hist :] = torch.finfo(dtype).min
        return mask

    def forward(self, ego_features: Tensor, exo_features: Tensor | None = None) -> Dict[str, Tensor]:
        del exo_features
        if ego_features.ndim != 3:
            raise ValueError("ego_features must be [B, T_hist, D]")
        if ego_features.shape[1] != self.t_hist:
            raise ValueError(f"expected T_hist={self.t_hist}, got {ego_features.shape[1]}")

        batch_size = ego_features.shape[0]
        history = self.feature_proj(ego_features) + self.history_pos
        queries = self.horizon_queries.expand(batch_size, -1, -1) + self.horizon_pos
        tokens = torch.cat([history, queries], dim=1)
        hidden_all = self.transformer(tokens, mask=self._attention_mask(tokens.device, tokens.dtype))
        hidden = self.norm(hidden_all[:, -self.h_pred :])
        logits = self.token_head(hidden).view(
            batch_size, self.h_pred, self.token_slots, self.codebook_size
        )
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "hidden": hidden,
        }


class EgoLastFrameMLP(nn.Module):
    """Ablation model that predicts future FACT tokens from only the last ego frame."""

    def __init__(
        self,
        d_feature: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        h_pred: int,
        token_slots: int,
        codebook_size: int,
        t_hist: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        del num_heads
        self.h_pred = h_pred
        self.token_slots = token_slots
        self.codebook_size = codebook_size
        self.t_hist = t_hist

        layers: list[nn.Module] = [
            nn.Linear(d_feature, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        ]
        for _ in range(max(1, num_layers) - 1):
            layers.extend(
                [
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.mlp = nn.Sequential(*layers)
        self.token_head = nn.Linear(d_model, h_pred * token_slots * codebook_size)

    def forward(self, ego_features: Tensor, exo_features: Tensor | None = None) -> Dict[str, Tensor]:
        del exo_features
        if ego_features.ndim != 3:
            raise ValueError("ego_features must be [B, T_hist, D]")
        if ego_features.shape[1] != self.t_hist:
            raise ValueError(f"expected T_hist={self.t_hist}, got {ego_features.shape[1]}")

        hidden_last = self.mlp(ego_features[:, -1])
        logits = self.token_head(hidden_last).view(
            ego_features.shape[0], self.h_pred, self.token_slots, self.codebook_size
        )
        hidden = hidden_last.unsqueeze(1).expand(-1, self.h_pred, -1)
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "hidden": hidden,
        }


class EgoExoWAM(nn.Module):
    """Privileged teacher that predicts future FACT tokens from ego + exo history."""

    def __init__(
        self,
        d_feature: int,
        d_exo_feature: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        h_pred: int,
        token_slots: int,
        codebook_size: int,
        t_hist: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.h_pred = h_pred
        self.token_slots = token_slots
        self.codebook_size = codebook_size
        self.t_hist = t_hist

        self.ego_proj = nn.Linear(d_feature, d_model)
        self.exo_proj = nn.Linear(d_exo_feature, d_model)
        self.history_pos = nn.Parameter(torch.zeros(1, t_hist, d_model))
        self.horizon_queries = nn.Parameter(torch.randn(1, h_pred, d_model) * 0.02)
        self.horizon_pos = nn.Parameter(torch.zeros(1, h_pred, d_model))

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.token_head = nn.Linear(d_model, token_slots * codebook_size)

    def _attention_mask(self, device: torch.device, dtype: torch.dtype) -> Tensor:
        length = self.t_hist + self.h_pred
        mask = torch.zeros(length, length, device=device, dtype=dtype)

        history_causal_block = torch.triu(
            torch.ones(self.t_hist, self.t_hist, device=device, dtype=torch.bool),
            diagonal=1,
        )
        mask[: self.t_hist, : self.t_hist] = history_causal_block.to(dtype) * torch.finfo(dtype).min
        mask[: self.t_hist, self.t_hist :] = torch.finfo(dtype).min
        return mask

    def forward(self, ego_features: Tensor, exo_features: Tensor | None = None) -> Dict[str, Tensor]:
        if exo_features is None:
            raise ValueError("ego_exo_transformer requires exo_features in the batch")
        if ego_features.ndim != 3 or exo_features.ndim != 3:
            raise ValueError("ego_features and exo_features must be [B, T_hist, D]")
        if ego_features.shape[:2] != exo_features.shape[:2]:
            raise ValueError("ego_features and exo_features must share [B, T_hist]")
        if ego_features.shape[1] != self.t_hist:
            raise ValueError(f"expected T_hist={self.t_hist}, got {ego_features.shape[1]}")

        batch_size = ego_features.shape[0]
        history = self.ego_proj(ego_features) + self.exo_proj(exo_features) + self.history_pos
        queries = self.horizon_queries.expand(batch_size, -1, -1) + self.horizon_pos
        tokens = torch.cat([history, queries], dim=1)
        hidden_all = self.transformer(tokens, mask=self._attention_mask(tokens.device, tokens.dtype))
        hidden = self.norm(hidden_all[:, -self.h_pred :])
        logits = self.token_head(hidden).view(
            batch_size, self.h_pred, self.token_slots, self.codebook_size
        )
        return {
            "logits": logits,
            "probs": F.softmax(logits, dim=-1),
            "hidden": hidden,
        }
