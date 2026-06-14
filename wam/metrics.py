from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class TokenBaselines:
    unigram_probs: Tensor
    markov_probs: Tensor


def topk_accuracy(logits: Tensor, target_ids: Tensor, k: int, confidence: Tensor | None = None) -> Tensor:
    k = min(k, logits.shape[-1])
    pred = torch.topk(logits, k=k, dim=-1).indices
    correct = (pred == target_ids.unsqueeze(-1)).any(dim=-1).float()
    if confidence is None:
        return correct.mean()
    return (correct * confidence).sum() / confidence.sum().clamp_min(1e-8)


def kl_to_target(logits: Tensor, target_probs: Tensor, confidence: Tensor | None = None) -> Tensor:
    target_probs = target_probs.clamp_min(1e-8)
    log_pred = F.log_softmax(logits, dim=-1)
    kl = (target_probs * (target_probs.log() - log_pred)).sum(dim=-1)
    if confidence is None:
        return kl.mean()
    return (kl * confidence).sum() / confidence.sum().clamp_min(1e-8)


def ce_to_target(logits: Tensor, target_ids: Tensor, confidence: Tensor | None = None) -> Tensor:
    ce = F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).view_as(target_ids).float()
    if confidence is None:
        return ce.mean()
    return (ce * confidence).sum() / confidence.sum().clamp_min(1e-8)


def compute_wam_metrics(
    logits: Tensor,
    target_ids: Tensor,
    target_probs: Tensor,
    confidence: Tensor,
    topk: Iterable[int] = (1, 5),
) -> Dict[str, float]:
    metrics = {
        "kl": float(kl_to_target(logits, target_probs, confidence).detach().cpu()),
        "ce": float(ce_to_target(logits, target_ids, confidence).detach().cpu()),
    }
    for k in topk:
        metrics[f"top{k}"] = float(topk_accuracy(logits, target_ids, k, confidence).detach().cpu())

    for horizon_idx in range(logits.shape[1]):
        h_logits = logits[:, horizon_idx]
        h_ids = target_ids[:, horizon_idx]
        h_probs = target_probs[:, horizon_idx]
        h_conf = confidence[:, horizon_idx]
        metrics[f"horizon_{horizon_idx + 1}/kl"] = float(
            kl_to_target(h_logits, h_probs, h_conf).detach().cpu()
        )
        for k in topk:
            metrics[f"horizon_{horizon_idx + 1}/top{k}"] = float(
                topk_accuracy(h_logits, h_ids, k, h_conf).detach().cpu()
            )
    return metrics


def confidence_bucket_metrics(
    logits: Tensor,
    target_ids: Tensor,
    target_probs: Tensor,
    confidence: Tensor,
    bins: List[float],
) -> Dict[str, float]:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_ids = target_ids.reshape(-1)
    flat_probs = target_probs.reshape(-1, target_probs.shape[-1])
    flat_conf = confidence.reshape(-1)
    metrics: Dict[str, float] = {}
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (flat_conf >= left) & (flat_conf < right if right < 1.0 else flat_conf <= right)
        key = f"conf_{left:.1f}_{right:.1f}"
        if not mask.any():
            metrics[f"{key}/count"] = 0.0
            metrics[f"{key}/top1"] = float("nan")
            metrics[f"{key}/kl"] = float("nan")
            continue
        masked_logits = flat_logits[mask]
        masked_ids = flat_ids[mask]
        masked_probs = flat_probs[mask]
        metrics[f"{key}/count"] = float(mask.sum().item())
        metrics[f"{key}/top1"] = float(topk_accuracy(masked_logits, masked_ids, 1).detach().cpu())
        metrics[f"{key}/kl"] = float(kl_to_target(masked_logits, masked_probs).detach().cpu())
    return metrics


def fit_token_baselines(dataset, codebook_size: int) -> TokenBaselines:
    slot_count = dataset.token_slots
    unigram = torch.ones(slot_count, codebook_size)
    markov = torch.ones(slot_count, codebook_size, codebook_size)
    for episode in dataset.episodes:
        ids = episode["fact_token_ids"].long()
        for slot in range(slot_count):
            slot_ids = ids[:, slot]
            unigram[slot].index_add_(0, slot_ids, torch.ones_like(slot_ids, dtype=unigram.dtype))
            prev_ids = slot_ids[:-1]
            next_ids = slot_ids[1:]
            for prev_id, next_id in zip(prev_ids.tolist(), next_ids.tolist()):
                markov[slot, prev_id, next_id] += 1.0
    return TokenBaselines(
        unigram_probs=unigram / unigram.sum(dim=-1, keepdim=True),
        markov_probs=markov / markov.sum(dim=-1, keepdim=True),
    )


def baseline_logits(
    baseline_name: str,
    baselines: TokenBaselines,
    context_token_ids: Tensor,
    h_pred: int,
    codebook_size: int,
) -> Tensor:
    device = context_token_ids.device
    batch_size, slot_count = context_token_ids.shape
    eps = 1e-8
    if baseline_name == "uniform":
        probs = torch.full(
            (batch_size, h_pred, slot_count, codebook_size),
            1.0 / codebook_size,
            device=device,
        )
        return probs.clamp_min(eps).log()
    if baseline_name == "unigram":
        probs = baselines.unigram_probs.to(device).unsqueeze(0).unsqueeze(0)
        probs = probs.expand(batch_size, h_pred, slot_count, codebook_size)
        return probs.clamp_min(eps).log()
    if baseline_name == "last_repeat":
        probs = F.one_hot(context_token_ids, num_classes=codebook_size).float()
        probs = probs.unsqueeze(1).expand(batch_size, h_pred, slot_count, codebook_size)
        return probs.clamp_min(eps).log()
    if baseline_name == "markov":
        markov = baselines.markov_probs.to(device)
        prev = context_token_ids
        steps = []
        for _ in range(h_pred):
            slot_probs = []
            next_ids = []
            for slot in range(slot_count):
                probs = markov[slot, prev[:, slot]]
                slot_probs.append(probs)
                next_ids.append(probs.argmax(dim=-1))
            step_probs = torch.stack(slot_probs, dim=1)
            steps.append(step_probs)
            prev = torch.stack(next_ids, dim=1)
        return torch.stack(steps, dim=1).clamp_min(eps).log()
    raise ValueError(f"Unknown baseline: {baseline_name}")
