# validation
from __future__ import annotations

import re
from typing import Any

import pandas as pd

def _get_preprocessing_config() -> dict[str, Any]:
    try:
        import sys
        from pathlib import Path
        root = Path(__file__).resolve().parents[2]
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from configs.load_config import get_preprocessing_config
        return get_preprocessing_config()
    except ImportError:
        return {}


def validate_and_preprocess(df: pd.DataFrame, config: dict[str, Any] | None = None) -> pd.DataFrame:
    cfg = config or _get_preprocessing_config()
    if not cfg:
        return df

    text_col = _resolve_column(df, cfg.get("columns", {}).get("text", "text"))
    label_col = _resolve_column(df, cfg.get("columns", {}).get("label", "type"), optional=True)

    val = cfg.get("validation", {})
    clean = cfg.get("cleaning", {})

    out = df.copy()
    if text_col and text_col in out.columns:
        out = out.rename(columns={text_col: "text"})
    if label_col and label_col in out.columns and label_col != "type":
        out = out.rename(columns={label_col: "type"})

    if val.get("drop_duplicates", True):
        out = out.drop_duplicates(subset=["text"] if "text" in out.columns else None)
    if val.get("drop_na_text", True) and "text" in out.columns:
        out = out.dropna(subset=["text"])
    if "text" in out.columns:
        out["text"] = out["text"].astype(str)
        min_len = val.get("min_text_length", 0)
        max_len = val.get("max_text_length", 999_999)
        mask = (out["text"].str.len() >= min_len) & (out["text"].str.len() <= max_len)
        out = out[mask]

    if clean and "text" in out.columns:
        out["text"] = out["text"].apply(
            lambda s: _clean_text(s, clean)
        )

    return out.reset_index(drop=True)


def _resolve_column(df: pd.DataFrame, name: str, *, optional: bool = False) -> str | None:
    if name in df.columns:
        return name
    aliases = ["content", "tweet", "message", "text"]
    for a in aliases:
        if a in df.columns:
            return a
    return None if optional else name


def _clean_text(text: str, clean: dict[str, Any]) -> str:
    s = str(text)
    if clean.get("fix_unicode", False):
        try:
            import ftfy
            s = ftfy.fix_text(s)
        except ImportError:
            import unicodedata
            s = unicodedata.normalize("NFKC", s)
    if clean.get("remove_urls", False):
        s = re.sub(r"https?://\S+", "", s, flags=re.IGNORECASE)
    if clean.get("remove_mentions", False):
        s = re.sub(r"@\w+", "", s)
    if clean.get("remove_hashtags", False):
        s = re.sub(r"#\w+", "", s)
    if clean.get("lowercase", False):
        s = s.lower()
    if clean.get("remove_extra_whitespace", True):
        s = re.sub(r"\s+", " ", s).strip()
    return s
