from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

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
        "nll": float(ce_to_target(logits, target_ids, confidence).detach().cpu()),
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


def slot_wise_metrics(
    logits: Tensor,
    target_ids: Tensor,
    target_probs: Tensor,
    confidence: Tensor,
    topk: Iterable[int] = (1, 5),
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    for slot_idx in range(logits.shape[2]):
        prefix = f"slot_{slot_idx}"
        slot_logits = logits[:, :, slot_idx]
        slot_ids = target_ids[:, :, slot_idx]
        slot_probs = target_probs[:, :, slot_idx]
        slot_conf = confidence[:, :, slot_idx]
        metrics[f"{prefix}/kl"] = float(kl_to_target(slot_logits, slot_probs, slot_conf).detach().cpu())
        metrics[f"{prefix}/nll"] = float(ce_to_target(slot_logits, slot_ids, slot_conf).detach().cpu())
        for k in topk:
            metrics[f"{prefix}/top{k}"] = float(
                topk_accuracy(slot_logits, slot_ids, k, slot_conf).detach().cpu()
            )
    return metrics


def code_frequencies(dataset, codebook_size: int) -> Tensor:
    counts = torch.zeros(codebook_size, dtype=torch.float32)
    for episode in dataset.episodes:
        ids = episode["fact_token_ids"].long().reshape(-1)
        counts.index_add_(0, ids, torch.ones_like(ids, dtype=counts.dtype))
    return counts


def per_code_recall_metrics(
    logits: Tensor,
    target_ids: Tensor,
    confidence: Tensor,
    train_code_frequencies: Tensor | None = None,
    rare_quantile: float = 0.2,
) -> Dict[str, Any]:
    pred_ids = logits.argmax(dim=-1).reshape(-1)
    flat_ids = target_ids.reshape(-1)
    flat_conf = confidence.reshape(-1).float()
    correct = (pred_ids == flat_ids).float()
    codebook_size = logits.shape[-1]

    recalls = torch.full((codebook_size,), float("nan"))
    counts = torch.zeros(codebook_size, dtype=torch.float32)
    per_code: Dict[str, Dict[str, float]] = {}
    for code in range(codebook_size):
        mask = flat_ids == code
        counts[code] = float(mask.sum().item())
        if not mask.any():
            per_code[str(code)] = {"count": 0.0, "top1_recall": float("nan")}
            continue
        weights = flat_conf[mask]
        if weights.sum() <= 0:
            recall = correct[mask].mean()
        else:
            recall = (correct[mask] * weights).sum() / weights.sum().clamp_min(1e-8)
        recalls[code] = recall
        per_code[str(code)] = {
            "count": float(mask.sum().item()),
            "top1_recall": float(recall.detach().cpu()),
        }

    present = counts > 0
    freq_norm = recalls[present].nanmean() if present.any() else torch.tensor(float("nan"))
    freq_source = train_code_frequencies.float() if train_code_frequencies is not None else counts
    rare_candidates = present & (freq_source > 0)
    if rare_candidates.any():
        threshold = torch.quantile(freq_source[rare_candidates], rare_quantile)
        rare_mask = rare_candidates & (freq_source <= threshold)
        rare_recall = recalls[rare_mask].nanmean() if rare_mask.any() else torch.tensor(float("nan"))
        rare_count = float(rare_mask.sum().item())
    else:
        rare_recall = torch.tensor(float("nan"))
        rare_count = 0.0

    return {
        "frequency_normalized_top1": float(freq_norm.detach().cpu()),
        "rare_code_top1": float(rare_recall.detach().cpu()),
        "rare_code_count": rare_count,
        "per_code": per_code,
    }


def calibration_metrics(
    logits: Tensor,
    target_ids: Tensor,
    confidence: Tensor,
    bins: Sequence[float],
) -> Dict[str, Any]:
    probs = F.softmax(logits, dim=-1)
    pred_conf, pred_ids = probs.max(dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    correct = (pred_ids == target_ids).float()
    flat_conf = pred_conf.reshape(-1)
    flat_entropy = entropy.reshape(-1)
    flat_correct = correct.reshape(-1)
    weights = confidence.reshape(-1).float()
    total_weight = weights.sum().clamp_min(1e-8)

    def weighted_mean(values: Tensor, mask: Tensor | None = None) -> Tensor:
        local_values = values if mask is None else values[mask]
        local_weights = weights if mask is None else weights[mask]
        if local_values.numel() == 0 or local_weights.sum() <= 0:
            return torch.tensor(float("nan"), device=values.device)
        return (local_values * local_weights).sum() / local_weights.sum().clamp_min(1e-8)

    ece = torch.tensor(0.0, device=logits.device)
    bucket_rows: Dict[str, Dict[str, float]] = {}
    for left, right in zip(bins[:-1], bins[1:]):
        mask = (flat_conf >= left) & (flat_conf < right if right < 1.0 else flat_conf <= right)
        key = f"pred_conf_{left:.1f}_{right:.1f}"
        if not mask.any():
            bucket_rows[key] = {"count": 0.0, "accuracy": float("nan"), "mean_confidence": float("nan")}
            continue
        bin_weight = weights[mask].sum()
        bin_acc = weighted_mean(flat_correct, mask)
        bin_conf = weighted_mean(flat_conf, mask)
        ece = ece + (bin_weight / total_weight) * (bin_acc - bin_conf).abs()
        bucket_rows[key] = {
            "count": float(mask.sum().item()),
            "accuracy": float(bin_acc.detach().cpu()),
            "mean_confidence": float(bin_conf.detach().cpu()),
        }

    error_mask = flat_correct < 0.5
    correct_mask = flat_correct >= 0.5
    entropy_error = weighted_mean(flat_entropy, error_mask)
    entropy_correct = weighted_mean(flat_entropy, correct_mask)
    return {
        "ece": float(ece.detach().cpu()),
        "mean_entropy": float(weighted_mean(flat_entropy).detach().cpu()),
        "entropy_correct": float(entropy_correct.detach().cpu()),
        "entropy_error": float(entropy_error.detach().cpu()),
        "entropy_error_gap": float((entropy_error - entropy_correct).detach().cpu()),
        "mean_pred_confidence": float(weighted_mean(flat_conf).detach().cpu()),
        "bins": bucket_rows,
    }


def phase_bucket_metrics(
    logits: Tensor,
    target_ids: Tensor,
    target_probs: Tensor,
    confidence: Tensor,
    phase_labels: Tensor,
    phase_label_names: Dict[int, str] | None = None,
    topk: Iterable[int] = (1, 5),
) -> Dict[str, float]:
    if phase_labels.numel() == 0:
        return {}
    metrics: Dict[str, float] = {}
    valid_phases = sorted(int(value) for value in phase_labels.unique().tolist() if int(value) >= 0)
    for phase in valid_phases:
        mask = phase_labels == phase
        if not mask.any():
            continue
        name = phase_label_names.get(phase, str(phase)) if phase_label_names else str(phase)
        phase_logits = logits[mask]
        phase_ids = target_ids[mask]
        phase_probs = target_probs[mask]
        phase_conf = confidence[mask]
        prefix = f"phase_{name}"
        metrics[f"{prefix}/count"] = float(mask.sum().item())
        metrics[f"{prefix}/kl"] = float(kl_to_target(phase_logits, phase_probs, phase_conf).detach().cpu())
        metrics[f"{prefix}/nll"] = float(ce_to_target(phase_logits, phase_ids, phase_conf).detach().cpu())
        for k in topk:
            metrics[f"{prefix}/top{k}"] = float(
                topk_accuracy(phase_logits, phase_ids, k, phase_conf).detach().cpu()
            )
    return metrics


def bucket_group_metrics(
    logits: Tensor,
    target_ids: Tensor,
    target_probs: Tensor,
    confidence: Tensor,
    buckets: Sequence[str],
    topk: Iterable[int] = (1, 5),
) -> Dict[str, Dict[str, float]]:
    results: Dict[str, Dict[str, float]] = {}
    if not buckets:
        return results
    unique_buckets = sorted(set(str(bucket) for bucket in buckets))
    for bucket in unique_buckets:
        mask = torch.tensor([str(value) == bucket for value in buckets], dtype=torch.bool)
        if not mask.any():
            continue
        results[bucket] = compute_wam_metrics(
            logits[mask],
            target_ids[mask],
            target_probs[mask],
            confidence[mask],
            topk=topk,
        )
        results[bucket]["window_count"] = float(mask.sum().item())
    return results


def fit_code_phase_map(dataset, codebook_size: int) -> Tensor | None:
    slot_count = dataset.token_slots
    max_phase = -1
    for episode in dataset.episodes:
        if "phase_labels" in episode:
            phase_labels = torch.as_tensor(episode["phase_labels"]).long()
            valid = phase_labels >= 0
            if valid.any():
                max_phase = max(max_phase, int(phase_labels[valid].max().item()))
    if max_phase < 0:
        return None

    counts = torch.zeros(slot_count, codebook_size, max_phase + 1, dtype=torch.float32)
    for episode in dataset.episodes:
        if "phase_labels" not in episode:
            continue
        ids = episode["fact_token_ids"].long()
        phases = torch.as_tensor(episode["phase_labels"]).long()
        valid = phases >= 0
        for slot in range(slot_count):
            slot_ids = ids[valid, slot]
            slot_phases = phases[valid]
            for code, phase in zip(slot_ids.tolist(), slot_phases.tolist()):
                counts[slot, code, phase] += 1.0

    phase_map = counts.argmax(dim=-1)
    phase_map[counts.sum(dim=-1) == 0] = -1
    return phase_map.long()


def phase_diagnostic_metrics(
    logits: Tensor,
    phase_labels: Tensor,
    confidence: Tensor,
    code_phase_map: Tensor | None,
) -> Dict[str, float]:
    if code_phase_map is None or phase_labels.numel() == 0:
        return {}
    pred_ids = logits.argmax(dim=-1)
    slot_maps = code_phase_map.to(pred_ids.device)
    mapped_slots = []
    for slot in range(pred_ids.shape[-1]):
        mapped_slots.append(slot_maps[slot, pred_ids[:, :, slot]])
    mapped = torch.stack(mapped_slots, dim=-1)
    target = phase_labels.unsqueeze(-1).expand_as(mapped)
    valid = (target >= 0) & (mapped >= 0)
    if not valid.any():
        return {"slot_accuracy": float("nan"), "any_slot_accuracy": float("nan"), "count": 0.0}
    slot_conf = confidence[valid]
    slot_correct = (mapped[valid] == target[valid]).float()
    slot_accuracy = (slot_correct * slot_conf).sum() / slot_conf.sum().clamp_min(1e-8)

    any_valid = valid.any(dim=-1)
    any_correct = ((mapped == target) & valid).any(dim=-1).float()
    any_weights = confidence.mean(dim=-1)
    any_accuracy = (any_correct[any_valid] * any_weights[any_valid]).sum() / any_weights[
        any_valid
    ].sum().clamp_min(1e-8)
    return {
        "slot_accuracy": float(slot_accuracy.detach().cpu()),
        "any_slot_accuracy": float(any_accuracy.detach().cpu()),
        "count": float(valid.sum().item()),
    }


def take_leakage_probe_metrics(logits: Tensor, take_ids: Sequence[str]) -> Dict[str, float]:
    if not take_ids:
        return {}
    unique_takes = sorted(set(str(take_id) for take_id in take_ids))
    if len(unique_takes) < 2:
        return {"take_count": float(len(unique_takes)), "probe_top1": float("nan")}

    hist = F.softmax(logits, dim=-1).mean(dim=(1, 2))
    hist = F.normalize(hist, p=2, dim=-1)
    take_to_index = {take_id: idx for idx, take_id in enumerate(unique_takes)}
    labels = torch.tensor([take_to_index[str(take_id)] for take_id in take_ids], dtype=torch.long)

    prototypes = []
    for take_id in unique_takes:
        mask = labels == take_to_index[take_id]
        proto = hist[mask].mean(dim=0)
        prototypes.append(F.normalize(proto, p=2, dim=0))
    prototype_tensor = torch.stack(prototypes, dim=0)
    pred = (hist @ prototype_tensor.T).argmax(dim=-1)
    probe_top1 = (pred.cpu() == labels).float().mean()

    if prototype_tensor.shape[0] > 1:
        cosine = prototype_tensor @ prototype_tensor.T
        off_diag = ~torch.eye(prototype_tensor.shape[0], dtype=torch.bool)
        mean_pairwise_cosine = cosine[off_diag].mean()
    else:
        mean_pairwise_cosine = torch.tensor(float("nan"))

    return {
        "take_count": float(len(unique_takes)),
        "window_count": float(len(take_ids)),
        "probe_top1": float(probe_top1.detach().cpu()),
        "mean_pairwise_prediction_hist_cosine": float(mean_pairwise_cosine.detach().cpu()),
    }


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
