"""Fonctions d'extraction de features textuelles."""
import re
import pandas as pd

# =============================================================================
# TEXT METRICS
# =============================================================================

def count_words(text: str) -> int:
    """Compte le nombre de mots."""
    return len(text.split())


def count_characters(text: str) -> int:
    """Compte le nombre de caractères."""
    return len(text)


def count_mentions(text: str) -> int:
    """Compte les mentions (@user)."""
    return len(re.findall(r"@\S+", text))


def count_hashtags(text: str) -> int:
    """Compte les hashtags (#topic)."""
    return len(re.findall(r"#\S+", text))


def count_urls(text: str) -> int:
    """Compte les URLs (http/https)."""
    return len(re.findall(r"https?://\S+", text))


def count_capitals(text: str) -> int:
    """Compte les lettres majuscules."""
    return len(re.findall(r"[A-Z]", text))


# =============================================================================
# TEXT CLEANING
# =============================================================================

def clean_text(text: str) -> str:
    """
    Nettoie le texte : HTML, URLs, mentions, hashtags, puis lowercase.

    Args:
        text: Texte brut.

    Returns:
        Texte nettoyé en minuscules, uniquement lettres.
    """
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"@\S+", "", text)
    text = re.sub(r"#\S+", "", text)
    text = re.sub(r"[^a-zA-Z]", " ", text).lower()
    text = re.sub(r"\s+", " ", text).strip()

    text = re.sub(r'\bRT\b', '', text)

    
    return text


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def apply_feature_engineering(df: pd.DataFrame, *, column_name: str) -> pd.DataFrame:
    """
    Extrait les features textuelles et nettoie la colonne spécifiée.

    Args:
        column_name: Nom de la colonne texte (keyword-only).

    Returns:
        DataFrame avec features ajoutées et texte nettoyé.
    """
    prefix = column_name

    df[f"{prefix}_words"] = df[column_name].apply(count_words)
    df[f"{prefix}_chars"] = df[column_name].apply(count_characters)
    df[f"{prefix}_mentions"] = df[column_name].apply(count_mentions)
    df[f"{prefix}_hashtags"] = df[column_name].apply(count_hashtags)
    df[f"{prefix}_urls"] = df[column_name].apply(count_urls)
    df[f"{prefix}_capitals"] = df[column_name].apply(count_capitals)
    df[column_name] = df[column_name].apply(clean_text)

    # Keeps only if more than 3 words
    df = df[df[f"{prefix}_words"] > 3]
    
    return df
