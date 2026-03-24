"""Tests du chargement des données."""
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from cyberbullying.loading import binary_load_data, merge_datasets, ColumnNotFoundError


def test_merge_datasets_empty():
    assert merge_datasets([]).empty


def test_merge_datasets(sample_binary_df):
    dfs = [sample_binary_df, sample_binary_df.copy()]
    merged = merge_datasets(dfs)
    assert len(merged) == 2 * len(sample_binary_df)
    assert list(merged.columns) == list(sample_binary_df.columns)


def test_binary_load_data_column_not_found():
    # dataset sans colonne texte reconnaissable -> doit lever ColumnNotFoundError
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        pd.DataFrame({"x": [1], "y": [0]}).to_csv(f.name, index=False)
        with pytest.raises(ColumnNotFoundError):
            binary_load_data(f.name)
    Path(f.name).unlink()
