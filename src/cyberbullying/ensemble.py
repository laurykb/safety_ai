"""Module pour les ensemble methods (Voting, Stacking, Blending)."""

from __future__ import annotations

import warnings
from typing import Any

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import classification_report, accuracy_score

warnings.filterwarnings("ignore")


def create_voting_ensemble(
    models: dict[str, Any],
    voting: str = "soft",
) -> VotingClassifier:
    """
    Crée un ensemble par voting.
    
    Args:
        models: Dict {nom: modèle sklearn}
        voting: "hard" ou "soft"
    
    Returns:
        VotingClassifier
    """
    estimators = [(name, model) for name, model in models.items()]
    
    ensemble = VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1,
    )
    
    return ensemble


def create_stacking_ensemble(
    base_models: dict[str, Any],
    meta_model: Any,
    cv: int = 5,
) -> StackingClassifier:
    """
    Crée un ensemble par stacking.
    
    Args:
        base_models: Dict {nom: modèle} pour les modèles de base
        meta_model: Modèle sklearn pour le meta-learner
        cv: Nombre de folds pour la cross-validation
    
    Returns:
        StackingClassifier
    """
    estimators = [(name, model) for name, model in base_models.items()]
    
    ensemble = StackingClassifier(
        estimators=estimators,
        final_estimator=meta_model,
        cv=cv,
        n_jobs=-1,
    )
    
    return ensemble


def train_and_evaluate_ensemble(
    ensemble,
    X_train,
    y_train,
    X_test,
    y_test,
) -> dict[str, Any]:
    """Entraîne et évalue un ensemble."""
    
    # Entraînement
    ensemble.fit(X_train, y_train)
    
    # Prédictions
    y_pred = ensemble.predict(X_test)
    
    # Métriques
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    results = {
        "accuracy": accuracy,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1": report["1"]["f1-score"],
        "report": report,
        "predictions": y_pred,
    }
    
    return results


def compare_models_for_ensemble(
    models: dict[str, Any],
    X,
    y,
    cv: int = 5,
) -> pd.DataFrame:
    """
    Compare plusieurs modèles pour sélectionner les meilleurs pour l'ensemble.
    
    Args:
        models: Dict {nom: modèle sklearn}
        X: Features
        y: Labels
        cv: Cross-validation folds
    
    Returns:
        DataFrame avec les scores de chaque modèle
    """
    results = []
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1)
        
        results.append({
            "model": name,
            "f1_mean": scores.mean(),
            "f1_std": scores.std(),
            "f1_min": scores.min(),
            "f1_max": scores.max(),
        })
    
    df = pd.DataFrame(results).sort_values("f1_mean", ascending=False)
    return df


def create_multi_embedding_ensemble(
    embeddings_data: dict[str, tuple],
    base_models: dict[str, Any],
    meta_model: Any = None,
    ensemble_type: str = "voting",
) -> Any:
    """
    Crée un ensemble combinant plusieurs embeddings.
    
    Args:
        embeddings_data: Dict {embedding_name: (X_train, y_train)}
        base_models: Dict {model_name: model} pour chaque embedding
        meta_model: Meta-learner pour stacking
        ensemble_type: "voting" ou "stacking"
    
    Returns:
        Ensemble model
    """
    if ensemble_type == "voting":
        if meta_model is None:
            from sklearn.linear_model import LogisticRegression
            meta_model = LogisticRegression(max_iter=1000)
        
        return create_voting_ensemble(base_models, voting="soft")
    
    elif ensemble_type == "stacking":
        if meta_model is None:
            from sklearn.linear_model import LogisticRegression
            meta_model = LogisticRegression(max_iter=1000)
        
        return create_stacking_ensemble(base_models, meta_model)
    
    else:
        raise ValueError("ensemble_type doit être 'voting' ou 'stacking'")


def save_ensemble(ensemble, path: Path, name: str = "ensemble_model"):
    """Sauvegarde un modèle d'ensemble."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    
    filepath = path / f"{name}.pkl"
    joblib.dump(ensemble, filepath)
    
    return filepath


def load_ensemble(filepath: Path):
    """Charge un modèle d'ensemble."""
    return joblib.load(filepath)


def get_ensemble_predictions_proba(
    ensemble,
    X,
) -> np.ndarray:
    """Retourne les probabilités prédites par l'ensemble."""
    if hasattr(ensemble, "predict_proba"):
        return ensemble.predict_proba(X)
    else:
        # Fallback: prédictions binaires
        predictions = ensemble.predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[np.arange(len(predictions)), predictions] = 1.0
        return proba


def analyze_ensemble_diversity(
    models: dict[str, Any],
    X,
    y,
) -> pd.DataFrame:
    """
    Analyse la diversité des prédictions entre modèles.
    Plus les modèles sont diversifiés, meilleur sera l'ensemble.
    """
    predictions = {}
    
    for name, model in models.items():
        predictions[name] = model.predict(X)
    
    # Calculer les accords/désaccords
    pred_df = pd.DataFrame(predictions)
    
    # Matrice de corrélation des prédictions
    correlation = pred_df.corr()
    
    # Taux d'accord par paire
    n_samples = len(X)
    agreement_matrix = []
    
    model_names = list(models.keys())
    for i, name1 in enumerate(model_names):
        row = []
        for name2 in model_names:
            agreement = (pred_df[name1] == pred_df[name2]).sum() / n_samples
            row.append(agreement)
        agreement_matrix.append(row)
    
    agreement_df = pd.DataFrame(
        agreement_matrix,
        index=model_names,
        columns=model_names
    )
    
    return agreement_df
