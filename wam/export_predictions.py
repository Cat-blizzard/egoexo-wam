from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader

from wam.data.fact_wam_dataset import FactWamDataset, move_wam_batch_to_device
from wam.eval import build_model_from_checkpoint
from wam.utils import load_yaml_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Ego-only WAM top-k predictions.")
    parser.add_argument("--config", type=Path, default=Path("wam/configs/wam_base.yaml"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=Path, default=Path("runs/wam_base/predictions.pt"))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    data_root = args.data_root or Path(config["data"]["root"])
    batch_size = args.batch_size or int(config["train"]["batch_size"])
    dataset = FactWamDataset(data_root, args.split, config["data"]["t_hist"], config["data"]["h_pred"])
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model_from_checkpoint(checkpoint).to(args.device)
    model.eval()

    predictions: List[Dict[str, Any]] = []
    for batch in loader:
        batch = move_wam_batch_to_device(batch, args.device)
        outputs = model(batch["ego_features"])
        probs = outputs["probs"]
        top = torch.topk(probs, k=min(args.topk, probs.shape[-1]), dim=-1)
        batch_size_actual = probs.shape[0]
        for idx in range(batch_size_actual):
            predictions.append(
                {
                    "episode_id": batch["episode_id"][idx],
                    "t": int(batch["t"][idx].detach().cpu()),
                    "pred_logits": outputs["logits"][idx].detach().cpu(),
                    "pred_probs": probs[idx].detach().cpu(),
                    "topk_token_ids": top.indices[idx].detach().cpu(),
                    "topk_probs": top.values[idx].detach().cpu(),
                    "entropy": (-(probs[idx] * probs[idx].clamp_min(1e-8).log()).sum(dim=-1)).detach().cpu(),
                    "target_token_ids": batch["target_token_ids"][idx].detach().cpu(),
                    "confidence": batch["confidence"][idx].detach().cpu(),
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(predictions, args.output)
    print(f"saved predictions: {args.output}")


if __name__ == "__main__":
    main()
