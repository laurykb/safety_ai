"""Fonctions de chargement et nettoyage des datasets."""
import pandas as pd

# =============================================================================
# EXCEPTIONS
# =============================================================================

class ColumnNotFoundError(ValueError):
    """Colonne introuvable dans le dataset."""


class InvalidBinaryLabelError(ValueError):
    """Les labels ne sont pas binaires (0/1)."""


class DatasetColumnMismatchError(ValueError):
    """Les colonnes des datasets ne correspondent pas."""


# =============================================================================
# BINARY LOADING
# =============================================================================

def binary_load_data(
    path: str,
    text_col: str | None = None,
    label_col: str | None = None,
    negative_class: str | None = None,
    n_samples: int | None = None,
) -> pd.DataFrame:
    """
    Charge un dataset et retourne un DataFrame binaire (0/1).

    Args:
        path: Chemin vers le fichier CSV.
        text_col: Nom de la colonne texte (auto-détecté si None).
        label_col: Nom de la colonne label (auto-détecté si None).
        negative_class: Valeur mappée à 0, le reste à 1. Si None, suppose binaire.
        n_samples: Nombre d'échantillons à charger. Si None, charge tous les échantillons.

    Returns:
        DataFrame avec colonnes ['text', 'type'] où type vaut 0 ou 1.

    Raises:
        ColumnNotFoundError: Colonne spécifiée introuvable.
        InvalidBinaryLabelError: Labels non binaires après traitement.
    """
    df = pd.read_csv(path)

    text_col = _detect_text_column(df, text_col, path)
    label_col, negative_class = _detect_label_column(df, label_col, negative_class, path)

    df = df[[text_col, label_col]].copy()
    df.columns = ["text", "type"]
    df = df.dropna()
    if n_samples is not None:
        df = df.sample(n=n_samples)

    df["type"] = _convert_to_binary(df["type"], negative_class)
    df["text"] = df["text"].astype(str)
    df = df.reset_index(drop=True)

    if not df["type"].isin([0, 1]).all():
        raise InvalidBinaryLabelError(f"Labels non binaires dans {path}")

    return df


def _detect_text_column(df: pd.DataFrame, text_col: str | None, path: str) -> str:
    """Détecte ou valide la colonne texte."""
    if text_col is not None:
        if text_col not in df.columns:
            raise ColumnNotFoundError(f"Colonne '{text_col}' introuvable dans {path}")
        return text_col

    if "tweet_text" in df.columns:
        return "tweet_text"
    if "Text" in df.columns:
        return "Text"

    raise ColumnNotFoundError(f"Aucune colonne texte détectée dans {path}")


def _detect_label_column(
    df: pd.DataFrame,
    label_col: str | None,
    negative_class: str | None,
    path: str
) -> tuple[str, str | None]:
    """Détecte ou valide la colonne label."""
    if label_col is not None:
        if label_col not in df.columns:
            raise ColumnNotFoundError(f"Colonne '{label_col}' introuvable dans {path}")
        return label_col, negative_class

    if "oh_label" in df.columns:
        return "oh_label", None
    if "cyberbullying_type" in df.columns:
        return "cyberbullying_type", "not_cyberbullying"

    raise ColumnNotFoundError(f"Aucune colonne label détectée dans {path}")


def _convert_to_binary(series: pd.Series, negative_class: str | None) -> pd.Series:
    """Convertit une série en binaire (0/1)."""
    if negative_class is not None:
        return series.apply(lambda x: 0 if x == negative_class else 1)
    return series.astype(int)


# =============================================================================
# MULTICLASS LOADING
# =============================================================================

def multiclass_load_data(
    path: str,
    text_col: str = "tweet_text",
    label_col: str = "cyberbullying_type",
    n_samples: int | None = None
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Charge un dataset multi-classes avec mapping catégorie → code.

    Args:
        path: Chemin vers le fichier CSV.
        text_col: Nom de la colonne texte.
        label_col: Nom de la colonne label.
        n_samples: Nombre d'exemplaires.

    Returns:
        Tuple (DataFrame['text', 'type'], mapping {catégorie: code}).

    Raises:
        ColumnNotFoundError: Colonne spécifiée introuvable.
    """
    df = pd.read_csv(path)

    if text_col not in df.columns:
        raise ColumnNotFoundError(f"Colonne '{text_col}' introuvable dans {path}")
    if label_col not in df.columns:
        raise ColumnNotFoundError(f"Colonne '{label_col}' introuvable dans {path}")

    df = df[[text_col, label_col]].copy()
    df = df.dropna()

    if n_samples is not None:
        df = df.sample(n=n_samples)

    categories = sorted(df[label_col].unique())
    mapping = {cat: i for i, cat in enumerate(categories)}

    df.columns = ["text", "type"]
    df["text"] = df["text"].astype(str)
    df["type"] = df["type"].map(mapping)
    df = df.reset_index(drop=True)

    return df, mapping


# =============================================================================
# MERGE
# =============================================================================

def merge_datasets(datasets: list[pd.DataFrame]) -> pd.DataFrame:
    """
    Fusionne plusieurs datasets avec les mêmes colonnes.

    Args:
        datasets: Liste de DataFrames à fusionner.

    Returns:
        DataFrame concaténé avec index réinitialisé.

    Raises:
        DatasetColumnMismatchError: Colonnes différentes entre datasets.
    """
    if not datasets:
        return pd.DataFrame()

    reference_cols = datasets[0].columns
    for i, ds in enumerate(datasets[1:], start=2):
        if not ds.columns.equals(reference_cols):
            raise DatasetColumnMismatchError(
                f"Dataset {i} colonnes {list(ds.columns)} != {list(reference_cols)}"
            )

    return pd.concat(datasets, ignore_index=True)
