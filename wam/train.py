from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
from torch.utils.data import DataLoader

from wam.data.fact_wam_dataset import FactWamDataset, move_wam_batch_to_device
from wam.data.synthetic_wam import write_synthetic_wam_dataset
from wam.eval import evaluate_model
from wam.losses.wam_losses import wam_loss
from wam.models.ego_only_wam import EgoOnlyWAM
from wam.utils import load_yaml_config, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train feature-level Ego-only WAM.")
    parser.add_argument("--config", type=Path, default=Path("wam/configs/wam_base.yaml"))
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/wam_base"))
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_model(config: Dict[str, Any], train_dataset: FactWamDataset) -> EgoOnlyWAM:
    return EgoOnlyWAM(
        d_feature=train_dataset.d_feature,
        d_model=config["model"]["d_model"],
        num_layers=config["model"]["num_layers"],
        num_heads=config["model"]["num_heads"],
        h_pred=config["data"]["h_pred"],
        token_slots=train_dataset.token_slots,
        codebook_size=train_dataset.codebook_size,
        t_hist=config["data"]["t_hist"],
        dropout=config["model"]["dropout"],
    )


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    torch.manual_seed(int(config["train"]["seed"]))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = args.data_root or Path(config["data"]["root"])
    if args.synthetic or bool(config.get("synthetic", {}).get("enabled", False)):
        synth = config["synthetic"]
        write_synthetic_wam_dataset(
            data_root,
            train_episodes=synth["train_episodes"],
            val_episodes=synth["val_episodes"],
            length=synth["length"],
            d_feature=config["data"]["d_feature"],
            token_slots=config["data"]["token_slots"],
            codebook_size=config["data"]["codebook_size"],
            switch_prob=synth["switch_prob"],
            noise_std=synth["noise_std"],
            seed=config["train"]["seed"],
        )

    train_dataset = FactWamDataset(data_root, "train", config["data"]["t_hist"], config["data"]["h_pred"])
    val_dataset = FactWamDataset(data_root, "val", config["data"]["t_hist"], config["data"]["h_pred"])
    train_loader = DataLoader(train_dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["train"]["batch_size"], shuffle=False)

    device = torch.device(args.device)
    model = build_model(config, train_dataset).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["train"]["lr"],
        weight_decay=config["train"]["weight_decay"],
    )

    max_steps = args.max_steps or int(config["train"]["max_steps"])
    step = 0
    train_log = []
    model.train()
    while step < max_steps:
        for batch in train_loader:
            batch = move_wam_batch_to_device(batch, device)
            outputs = model(batch["ego_features"])
            loss, loss_parts = wam_loss(
                outputs["logits"],
                batch["target_soft_probs"],
                batch["target_token_ids"],
                batch["confidence"],
                lambda_kl=config["loss"]["lambda_kl"],
                lambda_ce=config["loss"]["lambda_ce"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["train"]["grad_clip"])
            optimizer.step()
            if step % 20 == 0 or step == max_steps - 1:
                row = {key: float(value.detach().cpu()) for key, value in loss_parts.items()}
                row["step"] = step
                train_log.append(row)
                print(
                    f"step={step:04d} loss={row['loss']:.4f} "
                    f"kl={row['kl_loss']:.4f} ce={row['ce_loss']:.4f}"
                )
            step += 1
            if step >= max_steps:
                break

    model_config = {
        "d_feature": train_dataset.d_feature,
        "d_model": config["model"]["d_model"],
        "num_layers": config["model"]["num_layers"],
        "num_heads": config["model"]["num_heads"],
        "h_pred": config["data"]["h_pred"],
        "token_slots": train_dataset.token_slots,
        "codebook_size": train_dataset.codebook_size,
        "t_hist": config["data"]["t_hist"],
        "dropout": config["model"]["dropout"],
    }
    checkpoint = {
        "model_state": model.state_dict(),
        "model_config": model_config,
        "config": config,
        "train_log": train_log,
    }
    torch.save(checkpoint, output_dir / "last.pt")

    eval_results = evaluate_model(
        model,
        val_loader,
        train_dataset,
        device,
        topk=list(config["eval"]["topk"]),
        confidence_bins=list(config["eval"]["confidence_bins"]),
    )
    write_json(output_dir / "metrics.json", {"train": train_log, "eval": eval_results})
    write_json(output_dir / "config.json", config)
    print(f"saved checkpoint: {output_dir / 'last.pt'}")
    print(f"saved metrics: {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
