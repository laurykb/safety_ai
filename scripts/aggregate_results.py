"""Agrège les *_report.csv et génère des graphiques. Usage: python aggregate_results.py [--per-embedding]"""
import _bootstrap  # noqa: F401

import argparse
import glob
from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from cyberbullying.config import ANALYSIS_DIR, EXPERIMENTS_DIR

RESULTS_DIR = EXPERIMENTS_DIR / "results"
OUTPUT_DIR = ANALYSIS_DIR / "analysis_output"
EMBEDDINGS = ["tfidf", "bow", "word2vec", "glove", "bert"]

sns.set_theme(style="whitegrid", context="notebook")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all():
    files = glob.glob(str(RESULTS_DIR / "*_report.csv"))
    if not files:
        print("Aucun fichier trouvé.")
        return pd.DataFrame()
    data = []
    for fp in files:
        try:
            name = Path(fp).stem.replace("_report", "")
            parts = name.split("_")
            model = parts[0]
            emb = "unknown"
            for e in EMBEDDINGS:
                if e in name.lower():
                    emb = e
                    break
            df = pd.read_csv(fp, index_col=0)
            acc = df.loc["accuracy", "precision"] if "accuracy" in df.index else 0
            idx1 = [i for i in df.index if "1" in str(i)]
            if idx1:
                r = df.loc[idx1[0]]
                prec, rec, f1 = r["precision"], r["recall"], r["f1-score"]
            else:
                prec = rec = f1 = 0
            data.append({"Modèle": model, "Embedding": emb, "Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1})
        except Exception as e:
            print(f"Erreur {fp}: {e}")
    return pd.DataFrame(data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-embedding", action="store_true", help="Graph par embedding")
    args = parser.parse_args()
    df = load_all()
    if df.empty:
        return
    df.to_csv(OUTPUT_DIR / "recap.csv", index=False)
    if args.per_embedding:
        for emb in EMBEDDINGS:
            sub = df[df["Embedding"] == emb]
            if sub.empty:
                continue
            plt.figure(figsize=(8, 5))
            sns.barplot(data=sub, x="Modèle", y="F1", palette="viridis")
            plt.title(f"F1 - {emb}")
            plt.ylim(0, 1)
            plt.tight_layout()
            plt.savefig(OUTPUT_DIR / f"f1_{emb}.png")
            plt.close()
        print(f"Graphiques dans {OUTPUT_DIR}")
    else:
        plt.figure(figsize=(12, 6))
        sns.barplot(data=df, x="Modèle", y="F1", hue="Embedding", palette="viridis")
        plt.title("F1 par modèle et embedding")
        plt.ylim(0, 1)
        plt.legend(bbox_to_anchor=(1.02, 1))
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "f1_full.png")
        plt.close()
        print(f"Graphique sauvegardé: {OUTPUT_DIR / 'f1_full.png'}")


if __name__ == "__main__":
    main()
