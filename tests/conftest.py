"""pytest fixtures — chemin PYTHONPATH et données de test."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in [ROOT / "src", ROOT / "configs"]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


@pytest.fixture
def sample_text() -> str:
    """Texte exemple pour les tests de preprocessing."""
    return "Hello @user check this #covid link https://example.com RT"


@pytest.fixture
def sample_binary_df():
    """DataFrame exemple (text, type) pour les tests."""
    import pandas as pd
    return pd.DataFrame({
        "text": ["foo bar", "baz qux", "normal text", "bad word"],
        "type": [0, 1, 0, 1],
    })
