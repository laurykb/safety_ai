"""Télécharge des datasets HuggingFace, normalise et sauvegarde en CSV dans data/raw/."""
from __future__ import annotations

import sys
from pathlib import Path

# Éviter UnicodeEncodeError sous Windows (cp1252) lors des print()
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import _bootstrap  # noqa: F401
from _bootstrap import PROJECT_ROOT

import pandas as pd

UNIFIED_DIR = PROJECT_ROOT / "data" / "raw" / "unified"
RAW_DIR = PROJECT_ROOT / "data" / "raw"
RESEARCH_CB1_DIR = RAW_DIR / "research" / "cyberbullying_cb1"
RESEARCH_SARCASM_DIR = RAW_DIR / "research" / "sarcasm_twitter"

HF_SOURCES: list[dict] = [
    {
        "id": "cb1",
        "label": "CB1 (Surrey)",
        "dataset_id": "surrey-nlp/Cyberbullying-Detection-CB1",
        "loader": "cb1",
    },
    {
        "id": "cb2",
        "label": "CB2 (Surrey)",
        "dataset_id": "surrey-nlp/Cyberbullying-Detection-CB2",
        "loader": "cb2",
    },
    {
        "id": "sarcasm",
        "label": "Sarcasme Twitter",
        "dataset_id": "shiv213/Automatic-Sarcasm-Detection-Twitter",
        "text_col": "text",
        "label_col": "label",
        "negative_class": 0,
    },
    {
        "id": "wiki_toxic",
        "label": "Wiki Toxic",
        "dataset_id": "OxAISH-AL-LLM/wiki_toxic",
        "text_col": "comment_text",
        "label_col": "label",
        "negative_class": 0,
    },
    {
        "id": "hatexplain",
        "label": "HateXplain",
        "dataset_id": "hatexplain",
        "loader": "hatexplain",
    },
]


def _norm_cb1(subset) -> pd.DataFrame:
    df = subset.to_pandas()
    if "text" in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "type"})
        # CB1: "not_cyberbullying"=0, reste=1 (ou déjà 0/1)
        s = df["type"].astype(str).str.strip().str.lower()
        if s.isin(["0", "1"]).all():
            df["type"] = df["type"].astype(int)
        else:
            df["type"] = (s != "not_cyberbullying").astype(int)
        return df[["text", "type"]]
    for col in df.columns:
        if df[col].dtype == object and df[col].str.len().mean() > 20:
            text_col = col
            break
    else:
        return pd.DataFrame(columns=["text", "type"])
    label_col = [c for c in df.columns if c != text_col and df[c].nunique() <= 10]
    if not label_col:
        return pd.DataFrame(columns=["text", "type"])
    lc = label_col[0]
    df = df.rename(columns={text_col: "text", lc: "type"})
    df["type"] = df["type"].apply(lambda x: 1 if x != 0 else 0)
    return df[["text", "type"]]


def _norm_cb2(subset) -> pd.DataFrame:
    df = subset.to_pandas()
    if "text" in df.columns and "label" in df.columns:
        df = df.rename(columns={"label": "type"})
        s = df["type"].astype(str).str.strip().str.lower()
        if s.isin(["0", "1"]).all():
            df["type"] = df["type"].astype(int)
        else:
            df["type"] = (s != "not_cyberbullying").astype(int)
        return df[["text", "type"]]
    for col in df.columns:
        if df[col].dtype == object and df[col].str.len().mean() > 20:
            text_col = col
            break
    else:
        return pd.DataFrame(columns=["text", "type"])
    label_col = [c for c in df.columns if c != text_col and df[c].nunique() <= 10]
    if not label_col:
        return pd.DataFrame(columns=["text", "type"])
    lc = label_col[0]
    df = df.rename(columns={text_col: "text", lc: "type"})
    df["type"] = df["type"].apply(lambda x: 1 if x != 0 else 0)
    return df[["text", "type"]]


def _norm_hatexplain(subset) -> pd.DataFrame:
    df = subset.to_pandas()
    texts, labels = [], []
    for _, row in df.iterrows():
        post_tokens = row.get("post_tokens", [])
        if isinstance(post_tokens, list):
            text = " ".join(str(t) for t in post_tokens)
        else:
            text = str(post_tokens)
        annots = row.get("annotators", {})
        if isinstance(annots, dict):
            raw_labels = annots.get("label", [])
        elif isinstance(annots, list):
            raw_labels = annots
        else:
            raw_labels = []
        if raw_labels:
            from collections import Counter
            majority = Counter(raw_labels).most_common(1)[0][0]
            label = 0 if majority == 2 else 1
        else:
            label = 0
        texts.append(text)
        labels.append(label)
    return pd.DataFrame({"text": texts, "type": labels})


_NORMALIZERS = {
    "cb1": _norm_cb1,
    "cb2": _norm_cb2,
    "hatexplain": _norm_hatexplain,
}


def download_and_unify(
    selected_ids: list[str] | None = None,
    progress_cb=None,
) -> Path:
    """
    Télécharge les datasets sélectionnés, normalise et sauvegarde.

    Chaque dataset est sauvegardé individuellement en CSV dans data/raw/<id>.csv
    et un fichier unifié est créé dans data/raw/unified/train.csv.
    """
    from datasets import load_dataset

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    UNIFIED_DIR.mkdir(parents=True, exist_ok=True)
    sources = HF_SOURCES if selected_ids is None else [s for s in HF_SOURCES if s["id"] in selected_ids]
    if not sources:
        raise ValueError("Aucun dataset sélectionné")

    all_dfs = []
    for src in sources:
        did = src["dataset_id"]
        msg = f"Téléchargement {src['label']}..."
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)

        try:
            import warnings
            with warnings.catch_warnings(action="ignore"):
                ds = load_dataset(did)
            loader = src.get("loader")

            # Gérer les datasets avec train/test (cb1, cb2)
            if loader in _NORMALIZERS and "train" in ds and "test" in ds:
                df_train = _NORMALIZERS[loader](ds["train"])
                df_test = _NORMALIZERS[loader](ds["test"])
                df_train["text"] = df_train["text"].astype(str)
                df_test["text"] = df_test["text"].astype(str)
                df_train = df_train[df_train["text"].str.len() >= 5]
                df_test = df_test[df_test["text"].str.len() >= 5]
                if src["id"] == "cb1":
                    all_dfs.append(df_train)
                    # CB1: research/ uniquement (pas de doublon cb1.csv)
                    RESEARCH_CB1_DIR.mkdir(parents=True, exist_ok=True)
                    df_train.to_csv(RESEARCH_CB1_DIR / "train.csv", index=False)
                    df_test.to_csv(RESEARCH_CB1_DIR / "test.csv", index=False)
                    msg = f"  cb1: train {len(df_train)}, test {len(df_test)} -> research/cyberbullying_cb1/"
                else:
                    # cb2: fichier plat train+test
                    all_dfs.append(df_train)
                    all_dfs.append(df_test)
                    individual_path = RAW_DIR / f"{src['id']}.csv"
                    pd.concat([df_train, df_test], ignore_index=True).to_csv(individual_path, index=False)
                    msg = f"  {src['id']}: {len(df_train) + len(df_test)} lignes -> {individual_path.name}"
                if progress_cb:
                    progress_cb(msg)
                else:
                    print(msg)
                continue
            elif loader in _NORMALIZERS:
                split = "train" if "train" in ds else list(ds.keys())[0]
                subset = ds[split]
                df = _NORMALIZERS[loader](subset)
            else:
                split = "train" if "train" in ds else list(ds.keys())[0]
                subset = ds[split]
                df = subset.to_pandas()
                tc = src.get("text_col", "text")
                lc = src.get("label_col", "label")
                neg = src.get("negative_class")
                if tc not in df.columns or lc not in df.columns:
                    if progress_cb:
                        progress_cb(f"  Skip {src['id']} (colonnes manquantes)")
                    continue
                df = df.rename(columns={tc: "text", lc: "type"})
                if neg is not None:
                    df["type"] = (df["type"] != neg).astype(int)
                else:
                    df["type"] = df["type"].astype(int)
                df = df[["text", "type"]]

            df["text"] = df["text"].astype(str)
            df = df[df["text"].str.len() >= 5]
            all_dfs.append(df)

            if src["id"] == "sarcasm":
                # Sarcasm: research/ uniquement (pas de doublon sarcasm.csv)
                RESEARCH_SARCASM_DIR.mkdir(parents=True, exist_ok=True)
                df.to_csv(RESEARCH_SARCASM_DIR / "train.csv", index=False)
                if "test" in ds:
                    df_test_s = ds["test"].to_pandas()
                    tc, lc = src.get("text_col", "text"), src.get("label_col", "label")
                    if tc in df_test_s.columns and lc in df_test_s.columns:
                        df_test_s = df_test_s.rename(columns={tc: "text", lc: "type"})
                        if src.get("negative_class") is not None:
                            df_test_s["type"] = (df_test_s["type"] != src["negative_class"]).astype(int)
                        df_test_s = df_test_s[["text", "type"]].dropna()
                        df_test_s["text"] = df_test_s["text"].astype(str)
                        df_test_s = df_test_s[df_test_s["text"].str.len() >= 5]
                        df_test_s.to_csv(RESEARCH_SARCASM_DIR / "test.csv", index=False)
                msg = f"  sarcasm: {len(df)} lignes -> research/sarcasm_twitter/"
            else:
                individual_path = RAW_DIR / f"{src['id']}.csv"
                df.to_csv(individual_path, index=False)
                msg = f"  {src['id']}: {len(df)} lignes -> {individual_path.name}"
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

        except Exception as e:
            msg = f"  Erreur {src['id']}: {e}"
            if progress_cb:
                progress_cb(msg)
            else:
                print(msg)

    if all_dfs:
        unified = pd.concat(all_dfs, ignore_index=True)
        unified_path = UNIFIED_DIR / "train.csv"
        unified.to_csv(unified_path, index=False)
        msg = f"Unifie: {len(unified)} lignes -> {unified_path}"
        if progress_cb:
            progress_cb(msg)
        else:
            print(msg)
        return unified_path

    raise RuntimeError("Aucun dataset téléchargé avec succès")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Telecharger datasets HuggingFace")
    parser.add_argument("--datasets", nargs="*", default=None,
                       help="IDs a telecharger: cb1, cb2, sarcasm, wiki_toxic, hatexplain. Vide = tous.")
    args = parser.parse_args()
    selected = args.datasets if args.datasets else None
    download_and_unify(selected_ids=selected)


if __name__ == "__main__":
    main()
