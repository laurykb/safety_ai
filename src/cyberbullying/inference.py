"""
Inference Safety AI — chargement modèle et prédiction.
Usage API / Streamlit: load_predictor() -> predict(text)
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import joblib

# Chargé à la demande pour éviter import lourd
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from cyberbullying.embedder import embed_texts
        _embedder = embed_texts
    return _embedder


def _default_deploy_dir() -> Path | None:
    """Chemin par défaut models/trained/pipeline_deploy (relatif à la racine projet)."""
    for base in [Path.cwd(), Path(__file__).resolve().parents[2]]:
        d = base / "models" / "trained" / "pipeline_deploy"
        if (d / "model.pkl").exists():
            return d
    return None


def load_predictor(
    model_path: str | Path | None = None,
    embedding: str | None = None,
    vectorizer_path: str | Path | None = None,
) -> "Predictor | None":
    """
    Charge un prédicteur depuis les chemins fournis ou depuis les variables d'env.
    Env: MODEL_PATH, EMBEDDING, VECTORIZER_PATH.
    Fallback: models/trained/pipeline_deploy/ (créé par run_pipeline --save-model).
    """
    model_path = model_path or os.environ.get("MODEL_PATH")
    vectorizer_path = vectorizer_path or os.environ.get("VECTORIZER_PATH")

    if not model_path:
        deploy = _default_deploy_dir()
        if deploy:
            model_path = deploy / "model.pkl"
            if (deploy / "vectorizer.pkl").exists():
                vectorizer_path = deploy / "vectorizer.pkl"
            emb_file = deploy / "embedding.txt"
            if emb_file.exists():
                embedding = emb_file.read_text().strip()
    if not embedding:
        embedding = os.environ.get("EMBEDDING", "tfidf")

    if not model_path or not Path(model_path).exists():
        return None

    model = joblib.load(model_path)
    vec = None
    if vectorizer_path and Path(vectorizer_path).exists():
        vec = joblib.load(vectorizer_path)

    return Predictor(model=model, embedding=embedding, vectorizer=vec)


class Predictor:
    """Prédicteur Safety AI."""

    def __init__(self, model: Any, embedding: str, vectorizer: Any = None):
        self.model = model
        self.embedding = embedding
        self.vectorizer = vectorizer

    def predict(self, text: str) -> tuple[int, float]:
        """Retourne (label, proba_classe_1)."""
        embed_fn = _get_embedder()
        # Pour tfidf/bow avec vectorizer: on utilise le vectorizer, pas embed_texts
        if self.vectorizer is not None:
            X = self.vectorizer.transform([text])
        else:
            X = embed_fn([text], self.embedding, max_features=500)

        pred = self.model.predict(X)[0]
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0][1]
        else:
            proba = float(pred)
        return int(pred), float(proba)
