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
    bucket_group_metrics,
    calibration_metrics,
    code_frequencies,
    compute_wam_metrics,
    confidence_bucket_metrics,
    fit_code_phase_map,
    fit_token_baselines,
    per_code_recall_metrics,
    phase_bucket_metrics,
    phase_diagnostic_metrics,
    slot_wise_metrics,
    take_leakage_probe_metrics,
)
from wam.models.ego_only_wam import EgoExoWAM, EgoLastFrameMLP, EgoOnlyWAM
from wam.utils import dataset_filter_kwargs, load_yaml_config, write_json


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
    model_config = dict(checkpoint["model_config"])
    arch = model_config.pop("arch", "transformer")
    model_cls = {
        "transformer": EgoOnlyWAM,
        "mlp_last_frame": EgoLastFrameMLP,
        "ego_exo_transformer": EgoExoWAM,
    }.get(arch)
    if model_cls is None:
        raise ValueError(f"Unknown checkpoint model arch={arch!r}")
    model = model_cls(**model_config)
    model.load_state_dict(checkpoint["model_state"])
    return model


def phase_label_names_from_dataset(dataset: FactWamDataset) -> Dict[int, str]:
    names: Dict[int, str] = {}
    for episode in dataset.episodes:
        raw_names = episode.get("phase_label_names")
        if not raw_names:
            continue
        for key, value in raw_names.items():
            names[int(key)] = str(value)
    return names


@torch.no_grad()
def evaluate_model(
    model: EgoOnlyWAM,
    val_loader: DataLoader,
    train_dataset: FactWamDataset,
    device: torch.device,
    topk: List[int],
    confidence_bins: List[float],
    calibration_bins: List[float] | None = None,
    rare_code_quantile: float = 0.2,
) -> Dict[str, Any]:
    model.eval()
    baselines = fit_token_baselines(train_dataset, train_dataset.codebook_size)
    train_code_freq = code_frequencies(train_dataset, train_dataset.codebook_size)
    code_phase_map = fit_code_phase_map(train_dataset, train_dataset.codebook_size)
    phase_names = phase_label_names_from_dataset(train_dataset)

    logits_all = []
    target_ids_all = []
    target_probs_all = []
    confidence_all = []
    context_ids_all = []
    phase_all = []
    take_ids_all: List[str] = []
    buckets_all: List[str] = []

    for batch in val_loader:
        batch = move_wam_batch_to_device(batch, device)
        outputs = model(batch["ego_features"], batch.get("exo_features"))
        logits_all.append(outputs["logits"].detach().cpu())
        target_ids_all.append(batch["target_token_ids"].detach().cpu())
        target_probs_all.append(batch["target_soft_probs"].detach().cpu())
        confidence_all.append(batch["confidence"].detach().cpu())
        context_ids_all.append(batch["context_token_ids"].detach().cpu())
        phase_all.append(batch["target_phase_labels"].detach().cpu())
        take_ids_all.extend(str(value) for value in batch["take_id"])
        buckets_all.extend(str(value) for value in batch["bucket"])

    logits = torch.cat(logits_all, dim=0)
    target_ids = torch.cat(target_ids_all, dim=0)
    target_probs = torch.cat(target_probs_all, dim=0)
    confidence = torch.cat(confidence_all, dim=0)
    context_ids = torch.cat(context_ids_all, dim=0)
    phase_labels = torch.cat(phase_all, dim=0)
    calibration_bins = calibration_bins or confidence_bins

    results: Dict[str, Any] = {
        "wam": compute_wam_metrics(logits, target_ids, target_probs, confidence, topk=topk),
        "slot_wise": slot_wise_metrics(logits, target_ids, target_probs, confidence, topk=topk),
        "code_recall": per_code_recall_metrics(
            logits,
            target_ids,
            confidence,
            train_code_frequencies=train_code_freq,
            rare_quantile=rare_code_quantile,
        ),
        "calibration": calibration_metrics(logits, target_ids, confidence, calibration_bins),
        "confidence_buckets": confidence_bucket_metrics(
            logits, target_ids, target_probs, confidence, confidence_bins
        ),
        "phase_buckets": phase_bucket_metrics(
            logits,
            target_ids,
            target_probs,
            confidence,
            phase_labels,
            phase_label_names=phase_names,
            topk=topk,
        ),
        "phase_diagnostic": phase_diagnostic_metrics(logits, phase_labels, confidence, code_phase_map),
        "bucket_groups": bucket_group_metrics(
            logits, target_ids, target_probs, confidence, buckets_all, topk=topk
        ),
        "take_leakage_probe": take_leakage_probe_metrics(logits, take_ids_all),
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

    filter_kwargs = dataset_filter_kwargs(config)
    train_dataset = FactWamDataset(
        data_root,
        "train",
        config["data"]["t_hist"],
        config["data"]["h_pred"],
        **filter_kwargs,
    )
    val_dataset = FactWamDataset(
        data_root,
        "val",
        config["data"]["t_hist"],
        config["data"]["h_pred"],
        **filter_kwargs,
    )
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
        calibration_bins=list(config["eval"].get("calibration_bins", config["eval"]["confidence_bins"])),
        rare_code_quantile=float(config["eval"].get("rare_code_quantile", 0.2)),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "eval_results.json", results)
    write_horizon_curve(args.output_dir / "horizon_curve.csv", results["wam"])
    print(f"saved eval results: {args.output_dir / 'eval_results.json'}")


if __name__ == "__main__":
    main()
