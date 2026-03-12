"""Module pour l'optimisation automatique d'hyperparamètres avec Optuna."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


def optimize_logistic_regression(X, y, n_trials: int = 50, cv: int = 5) -> dict[str, Any]:
    """Optimise les hyperparamètres de Logistic Regression."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna n'est pas installé. Installez avec: pip install optuna")

    from sklearn.linear_model import LogisticRegression

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 0.001, 100.0, log=True),
            "max_iter": trial.suggest_int("max_iter", 100, 2000, step=100),
            "solver": trial.suggest_categorical("solver", ["lbfgs", "liblinear", "saga"]),
        }

        model = LogisticRegression(**params, random_state=42)
        score = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }


def optimize_random_forest(X, y, n_trials: int = 50, cv: int = 5) -> dict[str, Any]:
    """Optimise les hyperparamètres de Random Forest."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna n'est pas installé")

    from sklearn.ensemble import RandomForestClassifier

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 50, step=5),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2"]),
        }

        model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        score = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }


def optimize_svm(X, y, n_trials: int = 50, cv: int = 5) -> dict[str, Any]:
    """Optimise les hyperparamètres de SVM."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna n'est pas installé")

    from sklearn.svm import SVC

    def objective(trial):
        params = {
            "C": trial.suggest_float("C", 0.01, 100.0, log=True),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
            "gamma": trial.suggest_categorical("gamma", ["scale", "auto"]),
        }

        model = SVC(**params, random_state=42)
        score = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }


def optimize_lightgbm(X, y, n_trials: int = 50, cv: int = 5) -> dict[str, Any]:
    """Optimise les hyperparamètres de LightGBM."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna n'est pas installé")

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        raise ImportError("LightGBM n'est pas installé")

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 500, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "num_leaves": trial.suggest_int("num_leaves", 10, 100),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        model = LGBMClassifier(**params, random_state=42, verbose=-1, n_jobs=-1)
        score = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }


def optimize_mlp(X, y, n_trials: int = 30, cv: int = 5) -> dict[str, Any]:
    """Optimise les hyperparamètres de MLP."""
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna n'est pas installé")

    from sklearn.neural_network import MLPClassifier

    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 3)
        hidden_layers = tuple([
            trial.suggest_int(f"n_units_l{i}", 32, 256, step=32)
            for i in range(n_layers)
        ])

        params = {
            "hidden_layer_sizes": hidden_layers,
            "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
            "alpha": trial.suggest_float("alpha", 0.0001, 0.01, log=True),
            "learning_rate_init": trial.suggest_float("learning_rate_init", 0.0001, 0.01, log=True),
            "max_iter": 500,
        }

        model = MLPClassifier(**params, random_state=42)
        score = cross_val_score(model, X, y, cv=cv, scoring="f1", n_jobs=-1).mean()
        return score

    study = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    return {
        "best_params": study.best_params,
        "best_score": study.best_value,
        "study": study,
    }


def auto_optimize_model(
    model_name: str,
    X,
    y,
    n_trials: int = 50,
    cv: int = 5,
) -> dict[str, Any]:
    """
    Optimise automatiquement un modèle.

    Args:
        model_name: nom du modèle (logistic_regression, random_forest, svm, lightgbm, mlp)
        X: Features
        y: Labels
        n_trials: Nombre d'essais Optuna
        cv: Nombre de folds pour la cross-validation

    Returns:
        Dict avec best_params, best_score, study
    """
    optimizers = {
        "logistic_regression": optimize_logistic_regression,
        "random_forest": optimize_random_forest,
        "svm": optimize_svm,
        "lightgbm": optimize_lightgbm,
        "mlp": optimize_mlp,
    }

    if model_name not in optimizers:
        raise ValueError(f"Modèle {model_name} non supporté. Choix: {list(optimizers.keys())}")

    return optimizers[model_name](X, y, n_trials=n_trials, cv=cv)


def get_optimization_history(study) -> pd.DataFrame:
    """Retourne l'historique d'optimisation sous forme de DataFrame."""
    trials = study.trials

    data = []
    for trial in trials:
        data.append({
            "trial": trial.number,
            "value": trial.value,
            "params": str(trial.params),
            "state": trial.state.name,
        })

    return pd.DataFrame(data)
