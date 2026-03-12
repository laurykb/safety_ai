from pathlib import Path
from typing import Any

import yaml

CONFIGS_DIR = Path(__file__).resolve().parent


def load_config(name: str) -> dict[str, Any]:
    path = CONFIGS_DIR / f"{name}.yaml"
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def get_train_config() -> dict[str, Any]:
    """Paramètres ML (modèles, split). Utilisé par run_pipeline, streamlit."""
    return load_config("train")


def get_seed() -> int:
    """Seed reproductible (train.yaml experiment.seed). Utilisé partout pour cohérence."""
    cfg = get_train_config()
    return int(cfg.get("experiment", {}).get("seed", 42))


def get_preprocessing_config() -> dict[str, Any]:
    return load_config("preprocessing")


def get_research_config() -> dict[str, Any]:
    """Parametres recherche optionnels."""
    return load_config("research")
