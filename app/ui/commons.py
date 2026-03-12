"""Shared helpers for the Streamlit app."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from cyberbullying.config import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    REPORTS_DIR,
    EXPERIMENTS_DIR,
    TRAINED_MODELS_DIR,
)
def fmt_duration(seconds: float) -> str:
    """Affiche la durée en ms si < 1s, sinon en secondes."""
    if seconds is None or (isinstance(seconds, float) and seconds != seconds):
        return "-"
    try:
        s = float(seconds)
        if s < 0.001 and s >= 0:
            return "<1ms"
        return f"{s*1000:.0f}ms" if s < 1 else f"{s:.2f}s"
    except (TypeError, ValueError):
        return "-"


@st.cache_data(show_spinner=False)
def list_data_files() -> list[Path]:
    """Liste les datasets disponibles : fichiers plats dans data/raw/ + research/**/*.csv."""
    flat = [p for p in RAW_DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".csv", ".xlsx"}]
    research_dir = RAW_DATA_DIR / "research"
    nested = list(research_dir.rglob("*.csv")) if research_dir.exists() else []
    return sorted(flat + nested, key=lambda p: str(p))


def data_file_label(p: Path) -> str:
    """Clé d'affichage unique pour un fichier (ex: cb1.csv ou research/cyberbullying_cb1/train.csv)."""
    try:
        return str(p.relative_to(RAW_DATA_DIR)).replace("\\", "/")
    except ValueError:
        return p.name


@st.cache_data(show_spinner=False)
def load_binary_dataset(path: Path, n_samples: int | None = None) -> pd.DataFrame:
    """Charge un dataset brut (CSV ou Excel) sans validation stricte."""
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    elif path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.DataFrame()

    if n_samples and len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)

    return df


@st.cache_data(show_spinner=False)
def load_processed_df(embedding: str) -> pd.DataFrame | None:
    file_path = PROCESSED_DATA_DIR / f"df_{embedding}.csv"
    if not file_path.exists():
        return None
    return pd.read_csv(file_path)


def parse_report_file(path: Path) -> dict[str, Any] | None:
    import numpy as np

    try:
        df = pd.read_csv(path, index_col=0)
    except Exception:
        return None

    name = path.stem.replace("_report", "")
    parts = name.split("_")
    model_name = parts[0] if parts else "Unknown"
    embed_name = "Unknown"
    for embed in ["tfidf", "bow", "word2vec", "glove", "bert", "roberta"]:
        if embed in name.lower():
            embed_name = embed
            break

    acc = df.loc["accuracy", "precision"] if "accuracy" in df.index else np.nan
    class_1_idx = [idx for idx in df.index if "1" in str(idx)]
    if class_1_idx:
        row = df.loc[class_1_idx[0]]
        prec = row.get("precision", np.nan)
        rec = row.get("recall", np.nan)
        f1 = row.get("f1-score", np.nan)
    else:
        prec = rec = f1 = np.nan

    return {
        "model": model_name,
        "embedding": embed_name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "path": str(path),
    }


def get_model_params() -> dict[str, dict[str, Any]]:
    defaults = {
        "logistic_regression": {"C": 1.0, "max_iter": 1000},
        "random_forest": {"n_estimators": 200, "max_depth": None},
        "svm": {"C": 1.0, "kernel": "rbf"},
        "lightgbm": {"n_estimators": 300, "learning_rate": 0.05},
        "mlp": {"hidden_layer_sizes": (128, 64), "max_iter": 300},
    }
    if "model_params" not in st.session_state:
        st.session_state.model_params = defaults
    return st.session_state.model_params


def build_model(model_key: str, params: dict[str, Any], enable_proba: bool = False):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    try:
        from lightgbm import LGBMClassifier
    except Exception:
        LGBMClassifier = None

    if model_key == "logistic_regression":
        return LogisticRegression(C=params["C"], max_iter=params["max_iter"])
    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            random_state=42,
        )
    if model_key == "svm":
        return SVC(C=params["C"], kernel=params["kernel"], probability=enable_proba)
    if model_key == "lightgbm":
        if LGBMClassifier is None:
            raise RuntimeError("LightGBM n'est pas disponible dans l'environnement.")
        return LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            random_state=42,
            verbose=-1,
        )
    if model_key == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=params["hidden_layer_sizes"],
            max_iter=params["max_iter"],
            random_state=42,
        )
    raise ValueError("Modèle inconnu.")


def supports_predict_proba(model) -> bool:
    try:
        _ = model.predict_proba
        return True
    except Exception:
        return False


def get_report_paths() -> list[Path]:
    """List report file paths from REPORTS_DIR and EXPERIMENTS_DIR."""
    paths = list(REPORTS_DIR.glob("*_report.csv"))
    results_dir = EXPERIMENTS_DIR / "results"
    if results_dir.exists():
        paths += list(results_dir.glob("*_report.csv"))
    return paths
