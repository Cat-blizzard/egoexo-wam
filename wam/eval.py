from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from wam.data.fact_wam_dataset import FactWamDataset, move_wam_batch_to_device
from wam.metrics import (
    baseline_logits,
    compute_wam_metrics,
    confidence_bucket_metrics,
    fit_token_baselines,
)
from wam.models.ego_only_wam import EgoOnlyWAM
from wam.utils import load_yaml_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate feature-level Ego-only WAM.")
    parser.add_argument("--config", type=Path, default=Path("wam/configs/wam_base.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/wam_eval"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model_from_checkpoint(checkpoint: Dict[str, Any]) -> EgoOnlyWAM:
    model = EgoOnlyWAM(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state"])
    return model


@torch.no_grad()
def evaluate_model(
    model: EgoOnlyWAM,
    val_loader: DataLoader,
    train_dataset: FactWamDataset,
    device: torch.device,
    topk: List[int],
    confidence_bins: List[float],
) -> Dict[str, Any]:
    model.eval()
    baselines = fit_token_baselines(train_dataset, train_dataset.codebook_size)

    logits_all = []
    target_ids_all = []
    target_probs_all = []
    confidence_all = []
    context_ids_all = []

    for batch in val_loader:
        batch = move_wam_batch_to_device(batch, device)
        outputs = model(batch["ego_features"])
        logits_all.append(outputs["logits"].detach().cpu())
        target_ids_all.append(batch["target_token_ids"].detach().cpu())
        target_probs_all.append(batch["target_soft_probs"].detach().cpu())
        confidence_all.append(batch["confidence"].detach().cpu())
        context_ids_all.append(batch["context_token_ids"].detach().cpu())

    logits = torch.cat(logits_all, dim=0)
    target_ids = torch.cat(target_ids_all, dim=0)
    target_probs = torch.cat(target_probs_all, dim=0)
    confidence = torch.cat(confidence_all, dim=0)
    context_ids = torch.cat(context_ids_all, dim=0)

    results: Dict[str, Any] = {
        "wam": compute_wam_metrics(logits, target_ids, target_probs, confidence, topk=topk),
        "confidence_buckets": confidence_bucket_metrics(
            logits, target_ids, target_probs, confidence, confidence_bins
        ),
        "baselines": {},
    }
    for name in ("uniform", "unigram", "last_repeat", "markov"):
        b_logits = baseline_logits(
            name,
            baselines,
            context_ids,
            h_pred=target_ids.shape[1],
            codebook_size=target_probs.shape[-1],
        )
        results["baselines"][name] = compute_wam_metrics(
            b_logits, target_ids, target_probs, confidence, topk=topk
        )
    return results


def write_horizon_curve(path: Path, metrics: Dict[str, float]) -> None:
    rows = []
    for key, value in metrics.items():
        if key.startswith("horizon_"):
            horizon, metric = key.split("/")
            rows.append({"horizon": horizon.replace("horizon_", ""), "metric": metric, "value": value})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["horizon", "metric", "value"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    data_root = args.data_root or Path(config["data"]["root"])
    batch_size = args.batch_size or int(config["train"]["batch_size"])
    device = torch.device(args.device)

    train_dataset = FactWamDataset(data_root, "train", config["data"]["t_hist"], config["data"]["h_pred"])
    val_dataset = FactWamDataset(data_root, "val", config["data"]["t_hist"], config["data"]["h_pred"])
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint).to(device)
    results = evaluate_model(
        model,
        val_loader,
        train_dataset,
        device,
        topk=list(config["eval"]["topk"]),
        confidence_bins=list(config["eval"]["confidence_bins"]),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_results.json", results)
    write_horizon_curve(args.output_dir / "horizon_curve.csv", results["wam"])
    print(f"saved eval results: {args.output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
