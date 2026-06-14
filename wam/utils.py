from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def flatten_metrics(metrics: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    return {f"{prefix}{key}": value for key, value in metrics.items()}
