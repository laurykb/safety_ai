"""Vérifie les doublons dans les datasets d'entrée."""
from __future__ import annotations

import sys
from pathlib import Path

import _bootstrap  # noqa: F401
from _bootstrap import PROJECT_ROOT

import pandas as pd

RAW = PROJECT_ROOT / "data" / "raw"
RESEARCH = RAW / "research"


def get_text_col(df: pd.DataFrame) -> str:
    for c in ("text", "response", "tweet_text", "comment_text"):
        if c in df.columns:
            return c
    return df.columns[0]


def check_dups(df: pd.DataFrame, name: str) -> dict:
    tc = get_text_col(df)
    texts = df[tc].astype(str).str.strip()
    n = len(df)
    n_unique = texts.nunique()
    n_dup = n - n_unique
    dup_rows = df[texts.duplicated(keep=False)]
    return {
        "file": name,
        "total": n,
        "unique_texts": n_unique,
        "dup_texts": n_dup,
        "dup_rows": len(dup_rows),
        "pct_dup": 100 * (n - n_unique) / n if n else 0,
    }


def _paths_from_selection(selected: list[str] | None) -> list[Path]:
    """Convertit les labels (ex: research/cb1/train.csv) en chemins absolus."""
    if not selected:
        # Par defaut: tous les fichiers connus
        paths = []
        for f in ["train.csv", "test.csv", "validation.csv"]:
            p = RESEARCH / "cyberbullying_cb1" / f
            if p.exists():
                paths.append(p)
        for f in ["train.csv", "test.csv"]:
            p = RESEARCH / "sarcasm_twitter" / f
            if p.exists():
                paths.append(p)
        for f in ["cb2.csv", "wiki_toxic.csv", "hatexplain.csv"]:
            p = RAW / f
            if p.exists():
                paths.append(p)
        return paths
    return [RAW / f if not Path(f).is_absolute() else Path(f) for f in selected]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verifier doublons dans les datasets")
    parser.add_argument("--data-files", nargs="*", default=None,
                        help="Fichiers CSV a verifier (chemins relatifs a data/raw/). Vide = tous.")
    args = parser.parse_args()
    paths = _paths_from_selection(getattr(args, "data_files", None))

    results = []
    for p in paths:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p)
            name = str(p.relative_to(RAW)).replace("\\", "/") if RAW in p.parents or p.parent == RAW else p.name
            r = check_dups(df, name)
            results.append(r)
        except Exception as e:
            print(f"  Erreur {p}: {e}")

    print("=" * 70)
    print("Analyse des doublons (texte identique)")
    print("=" * 70)
    for r in results:
        pct = r["pct_dup"]
        status = "OK" if pct < 1 else "DOUBLONS"
        print(f"  {r['file']:35} {r['total']:>8} rows  {r['unique_texts']:>8} unique  {pct:>5.1f}% dup  [{status}]")

    # Overlap train/test (CB1)
    train_p = RESEARCH / "cyberbullying_cb1" / "train.csv"
    test_p = RESEARCH / "cyberbullying_cb1" / "test.csv"
    if train_p.exists() and test_p.exists():
        print("\n" + "=" * 70)
        print("Overlap train / test (CB1)")
        print("=" * 70)
        df_train = pd.read_csv(train_p)
        df_test = pd.read_csv(test_p)
        tc = get_text_col(df_train)
        set_train = set(df_train[tc].astype(str).str.strip())
        set_test = set(df_test[tc].astype(str).str.strip())
        overlap = set_train & set_test
        print(f"  Train: {len(set_train)} textes uniques")
        print(f"  Test:  {len(set_test)} textes uniques")
        print(f"  Overlap train/test: {len(overlap)} textes ({100*len(overlap)/len(set_test):.1f}% du test)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
