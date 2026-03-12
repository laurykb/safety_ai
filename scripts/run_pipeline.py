"""
Pipeline end-to-end Safety AI.
Usage:
  python scripts/run_pipeline.py              # full pipeline
  python scripts/run_pipeline.py --stage embed --embedding tfidf  # stage only
  python scripts/run_pipeline.py --stage download                  # datasets recherche
  python scripts/run_pipeline.py --override /path/to/override.yaml # override hyperparams
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import PROJECT_ROOT

import joblib
import yaml
from cyberbullying.config import (
    PROCESSED_DATA_DIR,
    EXPERIMENTS_DIR,
    TRAINED_MODELS_DIR,
    DATA_PATHS,
    RAW_DATA_DIR,
    MLFLOW_DIR,
    get_mlflow_tracking_uri,
)
from cyberbullying.loading import binary_load_data, merge_datasets
from cyberbullying.validation import validate_and_preprocess
from cyberbullying.feature_engineering import apply_feature_engineering
from cyberbullying.embedder import embed_texts
from cyberbullying.models import get_models_factory

try:
    from configs.load_config import get_train_config, get_seed
except ImportError:
    get_train_config = lambda: {}
    get_seed = lambda: 42

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

EMBEDDINGS = ["tfidf", "bow", "word2vec", "glove", "bert", "roberta"]
_override_config: dict = {}


def stage_download():
    """Télécharge les datasets de recherche (HuggingFace)."""
    print("[1/4] Stage: download_research")
    import subprocess
    subprocess.run([sys.executable, str(PROJECT_ROOT / "scripts" / "download_research_datasets.py")], cwd=PROJECT_ROOT, check=True)
    print("  OK: data/raw/research/")


def stage_load(n_samples: int | None, data_paths: list[Path] | None = None) -> pd.DataFrame:
    """Charge et fusionne les datasets bruts."""
    print("[2/4] Stage: load")
    paths = data_paths if data_paths else DATA_PATHS
    dfs = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            continue
        try:
            df = binary_load_data(p, n_samples=n_samples)
            dfs.append(df)
        except Exception as e:
            print(f"  skip {p.name}: {e}")
    if not dfs:
        raise RuntimeError("Aucun dataset trouve. Verifiez data/raw/ ou selectionnez des fichiers.")
    df = merge_datasets(dfs)
    df = validate_and_preprocess(df)
    print(f"  OK: {len(df)} lignes (après validation)")
    return df


def stage_embed(df: pd.DataFrame, embedding: str, max_features: int = 5000) -> pd.DataFrame:
    """Applique feature engineering + embedding."""
    print(f"[3/4] Stage: embed ({embedding})")
    df_clean = apply_feature_engineering(df, column_name="text")
    matrix = embed_texts(df_clean["text"].tolist(), embedding, max_features=max_features)
    prefix = f"text_{embedding}"
    cols = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
    df_emb = pd.concat([df_clean.reset_index(drop=True), pd.DataFrame(matrix, columns=cols)], axis=1)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / f"df_{embedding}.csv"
    df_emb.to_csv(out_path, index=False)
    print(f"  OK: {out_path}")
    return df_emb


def stage_train(
    df_emb: pd.DataFrame,
    embedding: str,
    model_keys: list[str],
    n_samples: int | None = None,
    test_size: float = 0.2,
    random_state: int | None = None,
    save_model: str | None = None,
    max_features: int = 5000,
):
    """Entraîne les modèles, enregistre les rapports et optionnellement le modèle pour déploiement."""
    print("[4/4] Stage: train")
    train_cfg = get_train_config()
    # Apply override from --override file (set by caller)
    override = _override_config
    if override:
        def _deep_merge(base: dict, over: dict) -> dict:
            out = dict(base)
            for k, v in over.items():
                if k in out and isinstance(out[k], dict) and isinstance(v, dict):
                    out[k] = _deep_merge(out[k], v)
                else:
                    out[k] = v
            return out
        train_cfg = _deep_merge(train_cfg, override)
    if random_state is None:
        random_state = get_seed()

    prefix = f"text_{embedding}"
    feats = [c for c in df_emb.columns if c.startswith(prefix)]
    X = df_emb[feats]
    y = df_emb["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    out_dir = EXPERIMENTS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    vectorizer = None
    if embedding in ("tfidf", "bow") and save_model:
        texts = df_emb["text"].tolist() if "text" in df_emb.columns else []
        if texts:
            if embedding == "tfidf":
                from cyberbullying.embedder import tfidf_embedding
                _, vectorizer = tfidf_embedding(texts, max_features, (1, 2))
            else:
                from cyberbullying.embedder import bow_embedding
                _, vectorizer = bow_embedding(texts, max_features, (1, 2))

    MODELS = get_models_factory(train_cfg)
    import mlflow
    MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(get_mlflow_tracking_uri())
    mlflow.set_experiment("safety-ai")

    for key in model_keys:
        if key not in MODELS:
            print(f"  skip {key} (inconnu)")
            continue
        with mlflow.start_run(run_name=f"{key}_{embedding}", log_system_metrics=True):
            model = MODELS[key]()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            report = classification_report(y_test, y_pred, output_dict=True)
            path = out_dir / f"{key}_{embedding}_report.csv"
            pd.DataFrame(report).T.to_csv(path)
            r1 = report.get("1", {})
            f1 = r1.get("f1-score", 0)
            acc = float(accuracy_score(y_test, y_pred))
            prec = r1.get("precision", 0)
            rec = r1.get("recall", 0)
            mlflow.log_params({
                "embedding": embedding, "model": key, "n_samples": n_samples or len(df_emb),
                "test_size": test_size, "seed": random_state, "max_features": max_features,
            })
            mlflow.log_metrics({"f1": f1, "accuracy": acc, "precision": prec, "recall": rec})
            mlflow.log_artifact(str(path))
        print(f"  {key}: F1={f1:.3f} -> {path.name}")

        if save_model and key == save_model:
            deploy_dir = TRAINED_MODELS_DIR / "pipeline_deploy"
            deploy_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, deploy_dir / "model.pkl")
            if vectorizer is not None:
                joblib.dump(vectorizer, deploy_dir / "vectorizer.pkl")
            (deploy_dir / "embedding.txt").write_text(embedding)
            print(f"  -> Modèle déployable: {deploy_dir} (MODEL_PATH, VECTORIZER_PATH, EMBEDDING)")


def main():
    parser = argparse.ArgumentParser(description="Pipeline Safety AI end-to-end")
    parser.add_argument("--stage", choices=["download", "load", "embed", "train", "all"], default="all")
    parser.add_argument("--embedding", default="tfidf", choices=EMBEDDINGS)
    parser.add_argument("--models", nargs="+", default=["logistic_regression", "random_forest"])
    parser.add_argument("--save-model", metavar="MODEL", help="Sauvegarder ce modele pour deploiement API")
    parser.add_argument("-n", "--n_samples", type=int, default=5000)
    parser.add_argument("--max-features", type=int, default=5000, help="Nombre max de features (TF-IDF/BoW)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Proportion test set (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Seed reproductibilite")
    parser.add_argument("--override", type=str, help="Fichier YAML override pour hyperparametres modeles")
    parser.add_argument("--data-files", nargs="*", default=None, help="Fichiers CSV (chemins relatifs a data/raw/)")
    args = parser.parse_args()

    data_paths = None
    if getattr(args, "data_files", None):
        data_paths = [RAW_DATA_DIR / f if not Path(f).is_absolute() else Path(f) for f in args.data_files]

    # Merge override config if provided (used by stage_train)
    global _override_config
    _override_config = {}
    if args.override and Path(args.override).exists():
        with open(args.override, encoding="utf-8") as f:
            _override_config = yaml.safe_load(f) or {}

    if args.stage == "download":
        stage_download()
        return

    if args.stage == "load":
        stage_load(args.n_samples, data_paths=data_paths)
        print("  (stage load: donnees chargees, utilisez --stage embed ou all pour continuer)")
        return

    # Pour embed et train: on a besoin de df avec colonnes d'embedding
    if args.stage in ("embed", "all"):
        df = stage_load(args.n_samples, data_paths=data_paths)
        df = stage_embed(df, args.embedding, max_features=args.max_features)
    else:
        emb_path = PROCESSED_DATA_DIR / f"df_{args.embedding}.csv"
        if not emb_path.exists():
            print(f"Fichier absent. Lancez d'abord: python scripts/run_pipeline.py --stage embed --embedding {args.embedding}")
            sys.exit(1)
        df = pd.read_csv(emb_path)

    if args.stage in ("train", "all"):
        stage_train(
            df, args.embedding, args.models,
            n_samples=args.n_samples,
            test_size=args.test_size,
            random_state=args.seed,
            save_model=args.save_model,
            max_features=args.max_features,
        )


if __name__ == "__main__":
    main()
