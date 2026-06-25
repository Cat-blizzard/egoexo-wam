from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch

from wam.utils import write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build WAM episode labels from sxh-kk/fact-tokenizer token exports."
    )
    parser.add_argument("--source-npz", type=Path, required=True, help="FACT paired-transition NPZ.")
    parser.add_argument(
        "--tokens-npz",
        type=Path,
        default=None,
        help="Path to ego_tokens.npz exported by scripts/extract_fact_tokens.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/fact_wam_labels"),
        help="Output root with split subdirectories.",
    )
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--ego-feature-npz", type=Path, default=None)
    parser.add_argument("--exo-feature-npz", type=Path, default=None)
    parser.add_argument("--ego-feature-key", type=str, default="ego_features")
    parser.add_argument("--exo-feature-key", type=str, default="exo_features")
    parser.add_argument(
        "--derive-raw-frame-features",
        action="store_true",
        help="Sandbox-only fallback: derive small RGB delta features from raw ego/exo arrays.",
    )
    parser.add_argument("--min-episode-length", type=int, default=2)
    parser.add_argument("--fact-repo-root", type=Path, default=Path("D:/fact-tokenizer"))
    parser.add_argument("--fact-checkpoint", type=Path, default=None)
    parser.add_argument("--extract-tokens", action="store_true")
    parser.add_argument("--extract-output-dir", type=Path, default=None)
    parser.add_argument("--source-view-keys", nargs="*", default=None)
    parser.add_argument("--view-names", nargs=2, default=None)
    parser.add_argument("--videos-layout", default="VBTCHW")
    parser.add_argument("--frame-pair", choices=["first-last", "first-next"], default="first-last")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--resize", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip-recon-metrics", action="store_true")
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def require_keys(arrays: Dict[str, np.ndarray], required: Iterable[str], path: Path) -> None:
    missing = [key for key in required if key not in arrays]
    if missing:
        raise KeyError(f"{path} missing required keys: {missing}")


def run_fact_extractor(args: argparse.Namespace) -> Path:
    if args.fact_checkpoint is None:
        raise ValueError("--extract-tokens requires --fact-checkpoint")
    repo_root = args.fact_repo_root.resolve()
    extractor = repo_root / "scripts" / "extract_fact_tokens.py"
    if not extractor.exists():
        raise FileNotFoundError(f"{extractor} not found; expected sxh-kk/fact-tokenizer checkout")

    output_dir = args.extract_output_dir or (args.output_root / "_fact_token_exports" / args.split)
    command = [
        sys.executable,
        str(extractor),
        "--checkpoint",
        str(args.fact_checkpoint),
        "--input-npz",
        str(args.source_npz),
        "--output-dir",
        str(output_dir),
        "--videos-layout",
        args.videos_layout,
        "--frame-pair",
        args.frame_pair,
        "--start-index",
        str(args.start_index),
        "--resize",
        str(args.resize),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
    ]
    if args.source_view_keys:
        command.extend(["--source-view-keys", *args.source_view_keys])
    if args.view_names:
        command.extend(["--view-names", *args.view_names])
    if args.skip_recon_metrics:
        command.append("--skip-recon-metrics")
    subprocess.run(command, cwd=repo_root, check=True)
    return output_dir / "ego_tokens.npz"


def flatten_fact_tokens(tokens: Dict[str, np.ndarray]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    require_keys(tokens, ("indices", "soft_probs", "confidence"), Path("tokens-npz"))
    indices = torch.from_numpy(np.asarray(tokens["indices"])).long()
    soft_probs = torch.from_numpy(np.asarray(tokens["soft_probs"])).float()
    confidence = torch.from_numpy(np.asarray(tokens["confidence"])).float()
    if indices.ndim == 2:
        indices = indices[:, None, :]
    if confidence.ndim == 2:
        confidence = confidence[:, None, :]
    if soft_probs.ndim == 3:
        soft_probs = soft_probs[:, None, :, :]
    if indices.ndim != 3:
        raise ValueError(f"indices must be [N, transition, slot] or [N, slot], got {tuple(indices.shape)}")
    if soft_probs.ndim != 4:
        raise ValueError(f"soft_probs must be [N, transition, slot, K], got {tuple(soft_probs.shape)}")
    if confidence.shape != indices.shape:
        raise ValueError(f"confidence shape {tuple(confidence.shape)} does not match indices {tuple(indices.shape)}")
    if soft_probs.shape[:3] != indices.shape:
        raise ValueError(f"soft_probs shape {tuple(soft_probs.shape)} does not match indices {tuple(indices.shape)}")

    sample_count, transition_count, slot_count = indices.shape
    codebook_size = soft_probs.shape[-1]
    flat_indices = indices.reshape(sample_count, transition_count * slot_count)
    flat_soft = soft_probs.reshape(sample_count, transition_count * slot_count, codebook_size)
    flat_confidence = confidence.reshape(sample_count, transition_count * slot_count)
    return flat_indices, flat_soft, flat_confidence


def _video_to_bthwc(array: np.ndarray) -> np.ndarray:
    if array.ndim != 5:
        raise ValueError(f"Expected raw video array [B,T,...], got shape {array.shape}")
    if array.shape[-1] in (1, 3):
        return array
    if array.shape[2] in (1, 3):
        return np.transpose(array, (0, 1, 3, 4, 2))
    raise ValueError(f"Cannot infer channel axis for raw video shape {array.shape}")


def derive_rgb_delta_features(array: np.ndarray) -> torch.Tensor:
    videos = _video_to_bthwc(array).astype(np.float32)
    if videos.size and videos.max() > 1.5:
        videos = videos / 255.0
    current = videos[:, 0]
    future = videos[:, -1]
    cur_mean = current.mean(axis=(1, 2))
    fut_mean = future.mean(axis=(1, 2))
    delta = fut_mean - cur_mean
    cur_std = current.std(axis=(1, 2))
    features = np.concatenate([cur_mean, fut_mean, delta, cur_std], axis=-1)
    return torch.from_numpy(features).float()


def load_feature_array(
    source: Dict[str, np.ndarray],
    feature_npz: Path | None,
    feature_key: str,
    raw_key: str,
    derive_raw: bool,
) -> torch.Tensor:
    feature_source = load_npz(feature_npz) if feature_npz is not None else source
    if feature_key in feature_source:
        return torch.from_numpy(np.asarray(feature_source[feature_key])).float()
    if raw_key in feature_source and derive_raw:
        return derive_rgb_delta_features(feature_source[raw_key])
    raise KeyError(
        f"Could not find '{feature_key}'. Pass --{raw_key}-feature-npz with that key, "
        f"or use --derive-raw-frame-features for sandbox-only RGB features."
    )


def source_metadata(source: Dict[str, np.ndarray], sample_count: int) -> Dict[str, Any]:
    take_uid = (
        np.asarray(source["take_uid"]).astype(str)
        if "take_uid" in source
        else np.asarray([str(index) for index in range(sample_count)])
    )
    timestamp = (
        np.asarray(source["timestamp"], dtype=np.float32)
        if "timestamp" in source
        else np.arange(sample_count, dtype=np.float32)
    )
    sample_id = (
        np.asarray(source["sample_id"]).astype(str)
        if "sample_id" in source
        else np.asarray([str(index) for index in range(sample_count)])
    )
    if len(take_uid) != sample_count or len(timestamp) != sample_count or len(sample_id) != sample_count:
        raise ValueError("source metadata length must match token sample count")
    return {"take_uid": take_uid, "timestamp": timestamp, "sample_id": sample_id}


def optional_per_sample(source: Dict[str, np.ndarray], key: str, sample_count: int) -> np.ndarray | None:
    if key not in source:
        return None
    value = np.asarray(source[key])
    if value.shape[:1] != (sample_count,):
        raise ValueError(f"{key} must have first dimension {sample_count}, got {value.shape}")
    return value


def first_per_sample(source: Dict[str, np.ndarray], keys: Iterable[str], sample_count: int) -> np.ndarray | None:
    for key in keys:
        if key in source:
            return optional_per_sample(source, key, sample_count)
    return None


def scalar_string(source: Dict[str, np.ndarray], key: str) -> str | None:
    if key not in source:
        return None
    value = np.asarray(source[key])
    if value.shape == ():
        return str(value.item())
    return None


def build_wam_episodes_from_fact_npz(
    source_npz: Path,
    tokens_npz: Path,
    output_root: Path,
    split: str,
    ego_feature_npz: Path | None = None,
    exo_feature_npz: Path | None = None,
    ego_feature_key: str = "ego_features",
    exo_feature_key: str = "exo_features",
    derive_raw_frame_features: bool = False,
    min_episode_length: int = 2,
) -> Dict[str, Any]:
    source = load_npz(source_npz)
    tokens = load_npz(tokens_npz)
    token_ids, soft_probs, confidence = flatten_fact_tokens(tokens)
    sample_count = int(token_ids.shape[0])
    metadata = source_metadata(source, sample_count)

    ego_features = load_feature_array(
        source,
        ego_feature_npz,
        ego_feature_key,
        raw_key="ego",
        derive_raw=derive_raw_frame_features,
    )
    if ego_features.shape[0] != sample_count:
        raise ValueError(f"ego features have {ego_features.shape[0]} rows, expected {sample_count}")

    exo_features = None
    try:
        exo_features = load_feature_array(
            source,
            exo_feature_npz,
            exo_feature_key,
            raw_key="exo",
            derive_raw=derive_raw_frame_features,
        )
        if exo_features.shape[0] != sample_count:
            raise ValueError(f"exo features have {exo_features.shape[0]} rows, expected {sample_count}")
    except KeyError:
        exo_features = None

    phase_labels = first_per_sample(source, ("phase_labels", "phase_label"), sample_count)
    bucket_labels = first_per_sample(source, ("bucket_labels", "bucket_label"), sample_count)
    if bucket_labels is None and "bucket" in source and np.asarray(source["bucket"]).shape[:1] == (sample_count,):
        bucket_labels = optional_per_sample(source, "bucket", sample_count)
    bucket_scalar = scalar_string(source, "bucket")
    sampling_weight = first_per_sample(source, ("sampling_weight", "sampling_weights"), sample_count)
    phase_label_names = None
    if "phase_label_names_json" in source:
        phase_label_names = json.loads(str(np.asarray(source["phase_label_names_json"]).item()))

    output_dir = output_root / split
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_path in output_dir.glob("*.pt"):
        stale_path.unlink()

    take_uid = metadata["take_uid"]
    timestamp = metadata["timestamp"]
    unique_takes = sorted(set(take_uid.tolist()))
    written = 0
    skipped_short = 0
    for take in unique_takes:
        indices = np.nonzero(take_uid == take)[0]
        order = indices[np.argsort(timestamp[indices], kind="stable")]
        if len(order) < min_episode_length:
            skipped_short += 1
            continue
        order_tensor = torch.as_tensor(order, dtype=torch.long)
        episode: Dict[str, Any] = {
            "episode_id": str(take),
            "take_id": str(take),
            "ego_features": ego_features.index_select(0, order_tensor).float(),
            "fact_token_ids": token_ids.index_select(0, order_tensor).long(),
            "fact_soft_probs": soft_probs.index_select(0, order_tensor).float(),
            "confidence": confidence.index_select(0, order_tensor).float(),
            "timestamps": timestamp[order].astype(float).tolist(),
            "sample_ids": metadata["sample_id"][order].astype(str).tolist(),
            "source_npz": str(source_npz),
            "tokens_npz": str(tokens_npz),
            "shape_semantics": "FACT samples grouped by take_uid and sorted by timestamp; FACT transition dims flattened into WAM slots.",
        }
        if exo_features is not None:
            episode["exo_features"] = exo_features.index_select(0, order_tensor).float()
        if phase_labels is not None:
            episode["phase_labels"] = torch.as_tensor(phase_labels[order]).long()
        if phase_label_names is not None:
            episode["phase_label_names"] = phase_label_names
        if bucket_labels is not None:
            episode["bucket_labels"] = [str(value) for value in bucket_labels[order].tolist()]
        elif bucket_scalar is not None:
            episode["bucket"] = bucket_scalar
        if sampling_weight is not None:
            episode["sampling_weight"] = torch.as_tensor(sampling_weight[order]).float()

        safe_take = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(take))
        torch.save(episode, output_dir / f"{written:06d}_{safe_take}.pt")
        written += 1

    report = {
        "source_npz": str(source_npz),
        "tokens_npz": str(tokens_npz),
        "output_dir": str(output_dir),
        "split": split,
        "sample_count": sample_count,
        "take_count": len(unique_takes),
        "episodes_written": written,
        "skipped_short_episodes": skipped_short,
        "token_slots": int(token_ids.shape[-1]),
        "codebook_size": int(soft_probs.shape[-1]),
        "ego_feature_dim": int(ego_features.shape[-1]),
        "exo_feature_dim": int(exo_features.shape[-1]) if exo_features is not None else None,
        "derive_raw_frame_features": bool(derive_raw_frame_features),
    }
    write_json(output_dir / "_build_report.json", report)
    return report


def main() -> None:
    args = parse_args()
    tokens_npz = args.tokens_npz
    if args.extract_tokens:
        tokens_npz = run_fact_extractor(args)
    if tokens_npz is None:
        raise ValueError("Pass --tokens-npz or use --extract-tokens with --fact-checkpoint")
    report = build_wam_episodes_from_fact_npz(
        source_npz=args.source_npz,
        tokens_npz=tokens_npz,
        output_root=args.output_root,
        split=args.split,
        ego_feature_npz=args.ego_feature_npz,
        exo_feature_npz=args.exo_feature_npz,
        ego_feature_key=args.ego_feature_key,
        exo_feature_key=args.exo_feature_key,
        derive_raw_frame_features=args.derive_raw_frame_features,
        min_episode_length=args.min_episode_length,
    )
    print(f"saved WAM labels to: {report['output_dir']}")


if __name__ == "__main__":
    main()
