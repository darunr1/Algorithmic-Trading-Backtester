"""Tests for the data loading and cleaning layer."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
import tempfile

from data.loader import load_csv, clean_ohlcv


@pytest.fixture
def sample_csv(tmp_path):
    """Create a temporary CSV with OHLCV data."""
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "open": np.random.uniform(95, 105, 100),
        "high": np.random.uniform(100, 110, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.linspace(100, 150, 100) + np.random.normal(0, 1, 100),
        "volume": np.random.randint(100000, 1000000, 100),
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_load_csv_basic(sample_csv):
    df = load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(df.index, pd.DatetimeIndex)
    assert "close" in df.columns
    assert len(df) == 100


def test_load_csv_missing_date_column(tmp_path):
    df = pd.DataFrame({"price": [1, 2, 3]})
    path = tmp_path / "no_date.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="date"):
        load_csv(str(path))


def test_load_csv_missing_price_column(tmp_path):
    df = pd.DataFrame({"date": ["2020-01-01"], "open": [100]})
    path = tmp_path / "no_close.csv"
    df.to_csv(path, index=False)
    with pytest.raises(ValueError, match="close"):
        load_csv(str(path))


def test_load_csv_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_csv("nonexistent_file.csv")


def test_clean_ohlcv_forward_fills():
    df = pd.DataFrame({
        "close": [100, np.nan, 102, np.nan, 104],
        "volume": [1000, 800, 1200, 900, 1100],  # volume present so rows aren't dropped
    }, index=pd.date_range("2020-01-01", periods=5))

    cleaned = clean_ohlcv(df)
    assert not cleaned["close"].isna().any()
    assert len(cleaned) == 5
    assert cleaned["close"].iloc[1] == 100  # Forward-filled


def test_clean_ohlcv_drops_all_nan():
    df = pd.DataFrame({
        "close": [100, np.nan, 102],
        "volume": [1000, np.nan, 1200],
    }, index=pd.date_range("2020-01-01", periods=3))

    cleaned = clean_ohlcv(df)
    # Row with all NaN in OHLCV should be dropped
    assert len(cleaned) == 2
