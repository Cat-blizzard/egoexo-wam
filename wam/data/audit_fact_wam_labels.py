from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List

import torch

from wam.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exported FACT-WAM pseudo labels.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output", type=Path, default=Path("runs/fact_wam_label_audit/summary.json"))
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("runs/fact_wam_label_audit/phase_audit_samples.csv"),
    )
    parser.add_argument("--high-confidence-threshold", type=float, default=0.8)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def concentration(counts: torch.Tensor, topn: int) -> float:
    total = counts.sum()
    if total <= 0:
        return float("nan")
    return float(counts.sort(descending=True).values[:topn].sum().item() / total.item())


def effective_code_count(counts: torch.Tensor) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    probs = counts.float() / total
    entropy = -(probs[probs > 0] * probs[probs > 0].log()).sum()
    return float(entropy.exp().item())


def top_code_rows(counts: torch.Tensor, limit: int = 20) -> List[Dict[str, float]]:
    total = counts.sum().clamp_min(1.0)
    rows: List[Dict[str, float]] = []
    values, indices = counts.sort(descending=True)
    for value, index in zip(values[:limit].tolist(), indices[:limit].tolist()):
        rows.append({"code": float(index), "count": float(value), "fraction": float(value / total.item())})
    return rows


def audit_split(
    data_root: Path,
    split: str,
    high_confidence_threshold: float,
    sample_count: int,
    seed: int,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    paths = sorted((data_root / split).glob("*.pt"))
    if not paths:
        raise FileNotFoundError(f"No .pt WAM episodes found under {data_root / split}")

    codebook_size = None
    token_slots = None
    total_tokens = 0
    high_conf_tokens = 0
    code_counts = None
    high_conf_code_counts = None
    slot_counts = None
    phase_counts: Dict[str, int] = {}
    phase_label_names: Dict[int, str] = {}
    sample_candidates: List[Dict[str, Any]] = []

    for path in paths:
        episode = torch.load(path, map_location="cpu", weights_only=False)
        ids = episode["fact_token_ids"].long()
        confidence = episode["confidence"].float()
        if codebook_size is None:
            codebook_size = int(episode["fact_soft_probs"].shape[-1])
            token_slots = int(ids.shape[-1])
            code_counts = torch.zeros(codebook_size, dtype=torch.float32)
            high_conf_code_counts = torch.zeros(codebook_size, dtype=torch.float32)
            slot_counts = torch.zeros(token_slots, codebook_size, dtype=torch.float32)

        flat_ids = ids.reshape(-1)
        flat_conf = confidence.reshape(-1)
        high_mask = flat_conf >= high_confidence_threshold
        total_tokens += int(flat_ids.numel())
        high_conf_tokens += int(high_mask.sum().item())
        assert code_counts is not None and high_conf_code_counts is not None and slot_counts is not None
        code_counts.index_add_(0, flat_ids, torch.ones_like(flat_ids, dtype=code_counts.dtype))
        if high_mask.any():
            high_conf_code_counts.index_add_(
                0,
                flat_ids[high_mask],
                torch.ones_like(flat_ids[high_mask], dtype=high_conf_code_counts.dtype),
            )
        for slot in range(ids.shape[-1]):
            slot_ids = ids[:, slot]
            slot_counts[slot].index_add_(
                0, slot_ids, torch.ones_like(slot_ids, dtype=slot_counts.dtype)
            )

        raw_phase_names = episode.get("phase_label_names") or {}
        for key, value in raw_phase_names.items():
            phase_label_names[int(key)] = str(value)
        phase_labels = episode.get("phase_labels")
        if phase_labels is not None:
            phase_tensor = torch.as_tensor(phase_labels).long()
            for phase in phase_tensor.tolist():
                if phase < 0:
                    continue
                name = phase_label_names.get(int(phase), str(int(phase)))
                phase_counts[name] = phase_counts.get(name, 0) + 1

        mean_conf = confidence.mean(dim=-1)
        candidate_mask = mean_conf >= high_confidence_threshold
        if not candidate_mask.any():
            candidate_mask = torch.ones_like(mean_conf, dtype=torch.bool)
        candidate_indices = candidate_mask.nonzero(as_tuple=False).flatten().tolist()
        bucket_labels = episode.get("bucket_labels")
        for t in candidate_indices:
            timestamp = episode.get("timestamps", None)
            phase = None
            if phase_labels is not None:
                phase_id = int(torch.as_tensor(phase_labels)[t].item())
                phase = phase_label_names.get(phase_id, str(phase_id))
            sample_candidates.append(
                {
                    "episode_id": episode["episode_id"],
                    "take_id": str(episode.get("take_id", episode["episode_id"])),
                    "t": t,
                    "timestamp": timestamp[t] if timestamp is not None else "",
                    "bucket": bucket_labels[t] if bucket_labels is not None else episode.get("bucket", "unknown"),
                    "phase": phase if phase is not None else "",
                    "token_ids": ids[t].tolist(),
                    "confidence": confidence[t].tolist(),
                    "mean_confidence": float(mean_conf[t].item()),
                    "source_path": str(path),
                }
            )

    assert code_counts is not None and high_conf_code_counts is not None and slot_counts is not None
    high_fraction = high_conf_tokens / max(total_tokens, 1)
    summary: Dict[str, Any] = {
        "split": split,
        "episode_count": len(paths),
        "token_slots": float(token_slots or 0),
        "codebook_size": float(codebook_size or 0),
        "token_count": float(total_tokens),
        "high_confidence_threshold": high_confidence_threshold,
        "high_confidence_token_fraction": high_fraction,
        "unique_code_count": float((code_counts > 0).sum().item()),
        "effective_code_count": effective_code_count(code_counts),
        "top1_code_fraction": concentration(code_counts, 1),
        "top5_code_fraction": concentration(code_counts, 5),
        "top10_code_fraction": concentration(code_counts, 10),
        "high_conf_unique_code_count": float((high_conf_code_counts > 0).sum().item()),
        "high_conf_top5_code_fraction": concentration(high_conf_code_counts, 5),
        "top_codes": top_code_rows(code_counts),
        "high_conf_top_codes": top_code_rows(high_conf_code_counts),
        "slot_unique_code_count": [
            float((slot_counts[slot] > 0).sum().item()) for slot in range(slot_counts.shape[0])
        ],
        "phase_counts": phase_counts,
    }

    rng = random.Random(seed)
    rng.shuffle(sample_candidates)
    return summary, sample_candidates[:sample_count]


def write_sample_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "episode_id",
        "take_id",
        "t",
        "timestamp",
        "bucket",
        "phase",
        "token_ids",
        "confidence",
        "mean_confidence",
        "source_path",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary, sample_rows = audit_split(
        args.data_root,
        args.split,
        high_confidence_threshold=args.high_confidence_threshold,
        sample_count=args.sample_count,
        seed=args.seed,
    )
    write_json(args.output, summary)
    write_sample_csv(args.sample_output, sample_rows)
    print(f"saved audit summary: {args.output}")
    print(f"saved phase audit samples: {args.sample_output}")


if __name__ == "__main__":
    main()
