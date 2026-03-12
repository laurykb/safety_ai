"""Test load -> features -> embed -> train pour un embedding donné. Usage: python run_embedding.py tfidf"""
import _bootstrap  # noqa: F401

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from cyberbullying.loading import binary_load_data, merge_datasets
from cyberbullying.config import DATA_PATHS, EXPERIMENTS_DIR
from cyberbullying.feature_engineering import apply_feature_engineering
from cyberbullying.embedder import embed_texts
from cyberbullying.models import get_models_factory

try:
    from configs.load_config import get_train_config
except ImportError:
    get_train_config = lambda: {}

EMBEDDINGS = ["tfidf", "bow", "word2vec", "glove", "bert", "roberta"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("embedding", choices=EMBEDDINGS, help="tfidf, bow, word2vec, glove, bert")
    parser.add_argument("-n", "--n_samples", type=int, default=100)
    parser.add_argument("--max-features", type=int, default=5000)
    args = parser.parse_args()
    emb = args.embedding
    n = args.n_samples
    max_features = args.max_features

    print("Chargement...")
    dfs = []
    for path in DATA_PATHS:
        try:
            df = binary_load_data(path, n_samples=n)
            dfs.append(df)
        except Exception as e:
            print(f"  skip {path.name}: {e}")
    if not dfs:
        print("Aucun dataset chargé.")
        return
    df_merged = merge_datasets(dfs)
    df_clean = apply_feature_engineering(df_merged, column_name="text")

    print(f"Embedding {emb}...")
    matrix = embed_texts(df_clean["text"].tolist(), emb, max_features=max_features)
    prefix = f"text_{emb}"
    emb_cols = [f"{prefix}_{i}" for i in range(matrix.shape[1])]
    df_emb = pd.concat([df_clean.reset_index(drop=True), pd.DataFrame(matrix, columns=emb_cols)], axis=1)

    X = df_emb[emb_cols]
    y = df_emb["type"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    out_dir = EXPERIMENTS_DIR / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_cfg = get_train_config()
    MODELS = get_models_factory(train_cfg)
    print("Train...")
    for key in ["logistic_regression", "random_forest", "svm", "mlp", "lightgbm"]:
        if key not in MODELS:
            continue
        model = MODELS[key]()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)
        path = out_dir / f"{key}_{emb}_report.csv"
        pd.DataFrame(report).T.to_csv(path)
        print(f"  {key} -> {path.name}")
