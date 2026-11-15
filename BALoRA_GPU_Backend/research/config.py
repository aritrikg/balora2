"""Configuration checks and stable JSON hashing used by BALoRA scripts."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    result = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected YAML mapping at {path}")
    return result


def validate_balora_config(cfg: dict[str, Any]) -> None:
    required_sections = ("experiment", "data", "sam2", "lora", "bayesian", "loss", "training")
    missing = [name for name in required_sections if name not in cfg]
    if missing:
        raise KeyError(f"Missing configuration sections: {missing}")
    data, lora, bayesian, training = cfg["data"], cfg["lora"], cfg["bayesian"], cfg["training"]
    if int(data.get("image_size", 0)) <= 0:
        raise ValueError("data.image_size must be positive")
    if int(data.get("num_classes", 0)) < 2:
        raise ValueError("data.num_classes must include background and foreground")
    rank_bank = lora.get("rank_bank", [])
    if not rank_bank or rank_bank != sorted(set(rank_bank)) or any(int(r) <= 0 for r in rank_bank):
        raise ValueError("lora.rank_bank must contain unique positive ranks in ascending order")
    if int(bayesian.get("k_train", 0)) < 2 or int(bayesian.get("k_infer", 0)) < 2:
        raise ValueError("bayesian.k_train and k_infer must be >=2 for epistemic uncertainty")
    if int(training.get("batch_size", 0)) < 1 or int(training.get("num_epochs", 0)) < 1:
        raise ValueError("training.batch_size and num_epochs must be positive")
    if int(data.get("image_size", 0)) != int(data.get("sam2_image_size", data.get("image_size", 0))):
        raise ValueError("data.image_size must match data.sam2_image_size unless SAM2 is explicitly rebuilt for another resolution")
    if not cfg["sam2"].get("model_cfg") or not cfg["sam2"].get("checkpoint"):
        raise ValueError("sam2.model_cfg and sam2.checkpoint must be specified")


def canonical_config_hash(cfg: dict[str, Any]) -> str:
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
