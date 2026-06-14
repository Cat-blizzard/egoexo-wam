from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build dense episode-level WAM labels from frozen FACT.")
    parser.add_argument("--fact-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--fact-repo-root",
        type=Path,
        default=None,
        help="Path to the FACT tokenizer repo. Defaults to FACT_REPO_ROOT if set.",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/fact_wam_labels"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_fact_model_from_repo(fact_repo_root: Path | None, checkpoint_path: Path, device: torch.device):
    root = fact_repo_root or (Path(os.environ["FACT_REPO_ROOT"]) if "FACT_REPO_ROOT" in os.environ else None)
    if root is None:
        raise ValueError("Pass --fact-repo-root or set FACT_REPO_ROOT to the FACT tokenizer repo path.")
    root = root.resolve()
    if not (root / "latent_action_model").exists():
        raise FileNotFoundError(f"{root} does not look like the FACT tokenizer repo root.")
    sys.path.insert(0, str(root))
    from latent_action_model.scripts.generate_fact_labels import load_fact_model

    return load_fact_model(checkpoint_path, device)


def move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device | str) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def read_manifest(path: Path) -> List[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    data = json.loads(path.read_text())
    if isinstance(data, dict) and "episodes" in data:
        return data["episodes"]
    if isinstance(data, list):
        return data
    raise ValueError("Manifest must be JSONL, a JSON list, or a JSON object with episodes.")


def batched(items: List[Dict[str, torch.Tensor]], batch_size: int) -> Iterable[Dict[str, torch.Tensor]]:
    for start in range(0, len(items), batch_size):
        chunk = items[start : start + batch_size]
        yield {
            key: torch.stack([sample[key] for sample in chunk], dim=0)
            for key in ("ego_context", "ego_future", "exo_context", "exo_future")
        }


def build_episode_samples(ego_features: torch.Tensor, exo_features: torch.Tensor, context_len: int, future_len: int):
    samples = []
    valid_indices = []
    length = ego_features.shape[0]
    for current_t in range(context_len - 1, length - future_len):
        samples.append(
            {
                "ego_context": ego_features[current_t - context_len + 1 : current_t + 1],
                "ego_future": ego_features[current_t + 1 : current_t + future_len + 1],
                "exo_context": exo_features[current_t - context_len + 1 : current_t + 1],
                "exo_future": exo_features[current_t + 1 : current_t + future_len + 1],
            }
        )
        valid_indices.append(current_t)
    return samples, valid_indices


def load_episode_features(record: Dict[str, Any], manifest_path: Path) -> Dict[str, Any]:
    feature_path = Path(record["feature_path"])
    if not feature_path.is_absolute():
        feature_path = manifest_path.parent / feature_path
    with np.load(feature_path) as arrays:
        ego_features = torch.from_numpy(arrays["ego_features"]).float()
        exo_features = torch.from_numpy(arrays["exo_features"]).float()
        timestamps = arrays["timestamps"].tolist() if "timestamps" in arrays else None
    return {
        "episode_id": str(record.get("episode_id", feature_path.stem)),
        "ego_features": ego_features,
        "exo_features": exo_features,
        "timestamps": timestamps,
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    model = load_fact_model_from_repo(args.fact_repo_root, args.fact_checkpoint, device)
    records = read_manifest(args.manifest)
    output_dir = args.output_root / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    for episode_idx, record in enumerate(records):
        episode = load_episode_features(record, args.manifest)
        samples, valid_indices = build_episode_samples(
            episode["ego_features"],
            episode["exo_features"],
            model.context_len,
            model.future_len,
        )
        if not samples:
            continue
        store: Dict[str, List[torch.Tensor]] = {
            "ego_token_id": [],
            "ego_soft_assignment": [],
            "exo_soft_assignment": [],
            "ego_confidence_weight": [],
            "exo_confidence_weight": [],
        }
        for batch in batched(samples, args.batch_size):
            batch = move_batch_to_device(batch, device)
            metadata = model.token_metadata(batch)
            for key in store:
                store[key].append(metadata[key].detach().cpu())

        ego_token_ids = torch.cat(store["ego_token_id"], dim=0)
        ego_soft = torch.cat(store["ego_soft_assignment"], dim=0)
        exo_soft = torch.cat(store["exo_soft_assignment"], dim=0)
        ego_conf = torch.cat(store["ego_confidence_weight"], dim=0)
        exo_conf = torch.cat(store["exo_confidence_weight"], dim=0)

        valid_indices_tensor = torch.tensor(valid_indices, dtype=torch.long)
        timestamps = None
        if episode["timestamps"] is not None:
            timestamps = [episode["timestamps"][idx] for idx in valid_indices]

        wam_episode = {
            "episode_id": episode["episode_id"],
            "ego_features": episode["ego_features"][valid_indices_tensor],
            "fact_token_ids": ego_token_ids.long(),
            "fact_soft_probs": 0.5 * (ego_soft + exo_soft),
            "confidence": 0.5 * (ego_conf + exo_conf),
            "timestamps": timestamps,
            "source_feature_path": record["feature_path"],
        }
        torch.save(wam_episode, output_dir / f"{episode_idx:06d}_{episode['episode_id']}.pt")
    print(f"saved WAM labels to: {output_dir}")


if __name__ == "__main__":
    main()
