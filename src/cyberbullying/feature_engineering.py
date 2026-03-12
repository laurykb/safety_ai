"""Fonctions d'extraction de features textuelles (convention top repos)."""
import re

import pandas as pd

try:
    import emoji
    EMOJI_AVAILABLE = True
except ImportError:
    EMOJI_AVAILABLE = False

try:
    import ftfy
    FTFY_AVAILABLE = True
except ImportError:
    FTFY_AVAILABLE = False

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

def _fix_unicode(text: str) -> str:
    """Corrige encodage Unicode (ftfy)."""
    if FTFY_AVAILABLE:
        return ftfy.fix_text(text)
    return text


def _handle_emoji(text: str, replace_with_desc: bool = True) -> str:
    """Gère les emojis : conversion en texte descriptif ou suppression."""
    if not EMOJI_AVAILABLE:
        # Fallback regex : supprime les caractères emoji (Unicode)
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return emoji_pattern.sub("", text)
    if replace_with_desc:
        return emoji.replace_emoji(text, replace="")
    return emoji.demojize(text, delimiters=(" ", " "))


def clean_text(
    text: str,
    fix_unicode: bool = True,
    handle_emoji: bool = True,
    keep_mentions_hashtags: bool = False,
) -> str:
    """
    Nettoie le texte (convention top repos : HTML, URLs, emojis, etc.).

    Args:
        text: Texte brut.
        fix_unicode: Corriger encodage (ftfy).
        handle_emoji: Gérer emojis (suppression ou conversion).
        keep_mentions_hashtags: Si True, garde @USER et #HASHTAG comme tokens.

    Returns:
        Texte nettoyé en minuscules.
    """
    if fix_unicode and FTFY_AVAILABLE:
        text = ftfy.fix_text(text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"https?://\S+", "", text)
    if keep_mentions_hashtags:
        text = re.sub(r"@\S+", " @USER ", text)
        text = re.sub(r"#\S+", " #HASHTAG ", text)
    else:
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r"#\S+", "", text)
    if handle_emoji:
        text = _handle_emoji(text, replace_with_desc=False)
    text = re.sub(r"\bRT\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).lower().strip()
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
