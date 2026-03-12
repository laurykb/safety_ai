"""
Définitions centralisées des modèles Safety AI.
Source unique pour run_pipeline, run_embedding, streamlit_app.
"""
from __future__ import annotations

from typing import Any

DEFAULT_SEED = 42
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logistic_regression": {"C": 1.0, "max_iter": 1000},
    "random_forest": {"n_estimators": 200, "max_depth": None},
    "svm": {"C": 1.0, "kernel": "rbf"},
    "lightgbm": {"n_estimators": 300, "learning_rate": 0.05},
    "mlp": {"hidden_layer_sizes": (128, 64), "max_iter": 300},
}

MODEL_KEYS = ["logistic_regression", "random_forest", "svm", "lightgbm", "mlp"]


def build_model(model_key: str, params: dict[str, Any] | None = None, enable_proba: bool = False):
    """Construit un classifieur sklearn à partir de la clé et des paramètres."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    p = (params or {}).copy()
    defaults = DEFAULT_PARAMS.get(model_key, {})
    for k, v in defaults.items():
        p.setdefault(k, v)

    if model_key == "logistic_regression":
        return LogisticRegression(C=p.get("C", 1.0), max_iter=p.get("max_iter", 1000), random_state=42)
    if model_key == "random_forest":
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth"),
            random_state=DEFAULT_SEED,
        )
    if model_key == "svm":
        return SVC(
            C=p.get("C", 1.0),
            kernel=p.get("kernel", "rbf"),
            probability=enable_proba,
            random_state=DEFAULT_SEED,
        )
    if model_key == "mlp":
        sizes = p.get("hidden_layer_sizes", (128, 64))
        if isinstance(sizes, list):
            sizes = tuple(sizes)
        return MLPClassifier(
            hidden_layer_sizes=sizes,
            max_iter=p.get("max_iter", 300),
            random_state=DEFAULT_SEED,
        )
    if model_key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except ImportError:
            raise RuntimeError("lightgbm non installé: pip install lightgbm")
        return LGBMClassifier(
            n_estimators=p.get("n_estimators", 300),
            learning_rate=p.get("learning_rate", 0.05),
            verbose=-1,
            random_state=DEFAULT_SEED,
        )
    raise ValueError(f"Modèle inconnu: {model_key}")


def get_models_factory(config: dict[str, Any] | None = None) -> dict[str, callable]:
    """Retourne un dict {model_key: callable} pour instanciation lazy."""
    models_cfg = (config or {}).get("models", {})
    seed = (config or {}).get("experiment", {}).get("seed", DEFAULT_SEED)
    factory = {}

    def _lr():
        p = models_cfg.get("logistic_regression", {})
        from sklearn.linear_model import LogisticRegression
        return LogisticRegression(
            C=p.get("C", 1.0), max_iter=p.get("max_iter", 1000), random_state=seed
        )

    def _rf():
        p = models_cfg.get("random_forest", {})
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth"),
            random_state=seed,
        )

    def _svm():
        p = models_cfg.get("svm", {})
        from sklearn.svm import SVC
        return SVC(C=p.get("C", 1.0), kernel=p.get("kernel", "rbf"), random_state=seed)

    def _mlp():
        p = models_cfg.get("mlp", {})
        from sklearn.neural_network import MLPClassifier
        sizes = p.get("hidden_layer_sizes", [128, 64])
        return MLPClassifier(
            hidden_layer_sizes=tuple(sizes) if isinstance(sizes, list) else sizes,
            max_iter=p.get("max_iter", 300),
            random_state=seed,
        )

    factory["logistic_regression"] = _lr
    factory["random_forest"] = _rf
    factory["svm"] = _svm
    factory["mlp"] = _mlp

    try:
        from lightgbm import LGBMClassifier
        def _lgb():
            p = models_cfg.get("lightgbm", {})
            return LGBMClassifier(
                n_estimators=p.get("n_estimators", 300),
                learning_rate=p.get("learning_rate", 0.05),
                verbose=-1,
                random_state=seed,
            )
        factory["lightgbm"] = _lgb
    except ImportError:
        pass

    return factory
