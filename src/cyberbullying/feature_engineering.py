"""Feature engineering textuel : métriques + nettoyage du texte.

On extrait quelques features simples (nb mots, mentions, hashtags, etc.)
avant l'embedding. Le nettoyage est assez agressif : on vire URLs, mentions,
HTML, emojis. Pour les hashtags on a le choix de les garder comme tokens.
"""

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


# --- Métriques textuelles simples ---

def count_words(text: str) -> int:
    return len(text.split())

def count_characters(text: str) -> int:
    return len(text)

def count_mentions(text: str) -> int:
    return len(re.findall(r"@\S+", text))

def count_hashtags(text: str) -> int:
    return len(re.findall(r"#\S+", text))

def count_urls(text: str) -> int:
    return len(re.findall(r"https?://\S+", text))

def count_capitals(text: str) -> int:
    # Utile pour détecter l'agressivité (TOUT EN MAJUSCULES)
    return len(re.findall(r"[A-Z]", text))


# --- Nettoyage du texte ---

def _handle_emoji(text: str, replace_with_desc: bool = True) -> str:
    if not EMOJI_AVAILABLE:
        # Fallback regex si le package emoji n'est pas dispo
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
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
    Nettoie le texte brut pour l'embedding.

    On supprime HTML, URLs, emojis, mentions et hashtags par défaut.
    Si keep_mentions_hashtags=True, on les remplace par des tokens
    génériques (@USER, #HASHTAG) plutôt que de les supprimer.
    """
    if fix_unicode and FTFY_AVAILABLE:
        text = ftfy.fix_text(text)

    text = re.sub(r"<[^>]+>", "", text)  # HTML
    text = re.sub(r"https?://\S+", "", text)  # URLs

    if keep_mentions_hashtags:
        text = re.sub(r"@\S+", " @USER ", text)
        text = re.sub(r"#\S+", " #HASHTAG ", text)
    else:
        text = re.sub(r"@\S+", "", text)
        text = re.sub(r"#\S+", "", text)

    if handle_emoji:
        text = _handle_emoji(text, replace_with_desc=False)

    text = re.sub(r"\bRT\b", "", text, flags=re.IGNORECASE)  # retweet marker
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).lower().strip()

    return text


# --- Pipeline feature engineering ---

def apply_feature_engineering(df: pd.DataFrame, *, column_name: str) -> pd.DataFrame:
    """
    Calcule les features textuelles et nettoie la colonne spécifiée.

    Les features (nb mots, mentions, etc.) sont ajoutées comme nouvelles colonnes.
    Le texte original est remplacé par sa version nettoyée.
    On filtre les textes trop courts (<=3 mots) qui ne servent à rien.
    """
    prefix = column_name
    df[f"{prefix}_words"] = df[column_name].apply(count_words)
    df[f"{prefix}_chars"] = df[column_name].apply(count_characters)
    df[f"{prefix}_mentions"] = df[column_name].apply(count_mentions)
    df[f"{prefix}_hashtags"] = df[column_name].apply(count_hashtags)
    df[f"{prefix}_urls"] = df[column_name].apply(count_urls)
    df[f"{prefix}_capitals"] = df[column_name].apply(count_capitals)

    df[column_name] = df[column_name].apply(clean_text)

    # on vire les textes trop courts après nettoyage
    df = df[df[f"{prefix}_words"] > 3]

    return df
