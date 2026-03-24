"""Chargement et nettoyage des datasets cyberbullying/sarcasme.

Les datasets ont des formats très différents selon leur source (Kaggle, HuggingFace, etc.),
donc la détection automatique des colonnes est nécessaire. On normalise tout vers
un DataFrame avec deux colonnes : 'text' (str) et 'type' (0 ou 1).
"""

from pathlib import Path
import pandas as pd


# Custom exceptions pour avoir des messages d'erreur clairs

class ColumnNotFoundError(ValueError):
    """Colonne texte ou label introuvable dans le dataset."""

class InvalidBinaryLabelError(ValueError):
    """Les labels ne sont pas en 0/1 après conversion."""

class DatasetColumnMismatchError(ValueError):
    """Tentative de merger des DataFrames avec des colonnes différentes."""


def _read_data(path: str | Path) -> pd.DataFrame:
    """Lit CSV ou Excel selon l'extension du fichier."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def binary_load_data(
    path: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    negative_class: str | None = None,
    n_samples: int | None = None,
) -> pd.DataFrame:
    """
    Charge un dataset et retourne un DataFrame binaire normalisé.

    Détecte automatiquement les colonnes texte/label si non spécifiées.
    Supporte les formats CSV et Excel.

    Returns:
        DataFrame avec colonnes ['text', 'type'] où type vaut 0 ou 1.
    """
    df = _read_data(path)
    text_col = _detect_text_column(df, text_col, path)
    label_col, negative_class = _detect_label_column(df, label_col, negative_class, path)

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "type"]
    df = df.dropna()

    if n_samples is not None and len(df) > 0:
        n = min(n_samples, len(df))
        df = df.sample(n=n)

    df["type"] = _convert_to_binary(df["type"], negative_class)
    df["text"] = df["text"].astype(str)
    df = df.reset_index(drop=True)

    if not df["type"].isin([0, 1]).all():
        raise InvalidBinaryLabelError(f"Labels non binaires dans {path}")

    return df


def _detect_text_column(df: pd.DataFrame, text_col: str | None, path: str) -> str:
    """Détecte la colonne texte en testant les noms courants dans les datasets publics."""
    if text_col is not None:
        if text_col not in df.columns:
            raise ColumnNotFoundError(f"Colonne '{text_col}' introuvable dans {path}")
        return text_col

    # On teste dans l'ordre de priorité selon les datasets qu'on utilise
    for col in ("tweet_text", "text", "response", "Text", "Summary", "Title",
                "Main_Narrative", "content", "message"):
        if col in df.columns and df[col].dropna().astype(str).str.len().gt(3).any():
            return col

    # Cas spécial pour certains datasets COVID
    if "Direct_Post_1" in df.columns and df["Direct_Post_1"].notna().any():
        return "Direct_Post_1"

    raise ColumnNotFoundError(f"Aucune colonne texte détectée dans {path}")


def _detect_label_column(
    df: pd.DataFrame,
    label_col: str | None,
    negative_class: str | None,
    path: str
) -> tuple[str, str | None]:
    """Détecte la colonne label et la valeur à mapper à 0 (classe négative)."""
    if label_col is not None:
        if label_col not in df.columns:
            raise ColumnNotFoundError(f"Colonne '{label_col}' introuvable dans {path}")
        return label_col, negative_class

    if "oh_label" in df.columns:
        return "oh_label", None
    if "cyberbullying_type" in df.columns:
        return "cyberbullying_type", "not_cyberbullying"
    if "type" in df.columns:
        uniq = df["type"].dropna().astype(int)
        if set(uniq.unique()) <= {0, 1}:
            return "type", None
    if "label" in df.columns:
        vals = df["label"].astype(str).str.strip().str.lower()
        if vals.isin(["not_cyberbullying", "0"]).any():
            return "label", "not_cyberbullying"
        if vals.isin(["not_sarcasm", "sarcasm"]).any():
            return "label", "not_sarcasm"
        return "label", None

    raise ColumnNotFoundError(f"Aucune colonne label détectée dans {path}")


def _convert_to_binary(series: pd.Series, negative_class: str | None) -> pd.Series:
    if negative_class is not None:
        return series.apply(
            lambda x: 0 if str(x).strip().lower() == str(negative_class).strip().lower() else 1
        )
    return series.astype(int)


def load_sarcasm_data(
    path: str | Path,
    text_col: str | None = None,
    label_col: str | None = None,
    n_samples: int | None = None,
) -> pd.DataFrame:
    """
    Charge un dataset sarcasme (format shiv213 ou similaire).

    Les colonnes attendues sont response/tweet/text pour le texte
    et label (SARCASM / NOT_SARCASM) pour le label.

    Returns:
        DataFrame avec ['text', 'type'] où type=1 si sarcasme.
    """
    df = _read_data(path)

    if text_col is None:
        for col in ("response", "tweet_text", "text", "tweet"):
            if col in df.columns and df[col].dropna().astype(str).str.len().gt(3).any():
                text_col = col
                break
        if text_col is None:
            raise ColumnNotFoundError(f"Aucune colonne texte dans {path}")
    elif text_col not in df.columns:
        raise ColumnNotFoundError(f"Colonne '{text_col}' introuvable dans {path}")

    if label_col is None:
        label_col = "label" if "label" in df.columns else "type"
    if label_col not in df.columns:
        raise ColumnNotFoundError(f"Colonne label '{label_col}' introuvable dans {path}")

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "raw_label"]
    df = df.dropna()

    s = df["raw_label"].astype(str).str.strip().str.upper()
    df["type"] = (s == "SARCASM").astype(int)
    df = df[["text", "type"]]
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len().ge(5)].reset_index(drop=True)

    if n_samples is not None:
        df = df.sample(n=min(n_samples, len(df)), random_state=42)

    return df


def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """Fusionne plusieurs DataFrames. Lève une erreur si les colonnes diffèrent."""
    if not datasets:
        return pd.DataFrame()

    reference_cols = datasets[0].columns
    for i, ds in enumerate(datasets[1:], start=2):
        if not ds.columns.equals(reference_cols):
            raise DatasetColumnMismatchError(
                f"Dataset {i} colonnes {list(ds.columns)} != {list(reference_cols)}"
            )

    return pd.concat(datasets, ignore_index=True)
