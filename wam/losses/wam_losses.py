from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def confidence_weighted_kl(
    pred_logits: Tensor,
    target_probs: Tensor,
    confidence: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    log_pred = F.log_softmax(pred_logits, dim=-1)
    target_probs = target_probs.clamp_min(eps)
    kl = target_probs * (target_probs.log() - log_pred)
    kl = kl.sum(dim=-1)
    weighted = kl * confidence
    return weighted.sum() / (confidence.sum() + eps)


def confidence_weighted_ce(
    pred_logits: Tensor,
    target_token_ids: Tensor,
    confidence: Tensor,
    eps: float = 1e-8,
) -> Tensor:
    codebook_size = pred_logits.shape[-1]
    ce = F.cross_entropy(
        pred_logits.reshape(-1, codebook_size),
        target_token_ids.reshape(-1),
        reduction="none",
    ).view_as(confidence)
    return (ce * confidence).sum() / (confidence.sum() + eps)


def wam_loss(
    pred_logits: Tensor,
    target_probs: Tensor,
    target_token_ids: Tensor,
    confidence: Tensor,
    lambda_kl: float = 1.0,
    lambda_ce: float = 0.2,
) -> Tuple[Tensor, Dict[str, Tensor]]:
    kl = confidence_weighted_kl(pred_logits, target_probs, confidence)
    ce = confidence_weighted_ce(pred_logits, target_token_ids, confidence)
    loss = lambda_kl * kl + lambda_ce * ce
    return loss, {
        "loss": loss.detach(),
        "kl_loss": kl.detach(),
        "ce_loss": ce.detach(),
    }
