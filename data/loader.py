"""Data ingestion and cleaning utilities for OHLCV market data."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_csv(
    path: str,
    price_column: str = "close",
    date_column: str = "date",
) -> pd.DataFrame:
    """Load OHLCV data from a CSV file and return a cleaned DataFrame.

    Args:
        path: Path to the CSV file.
        price_column: Column to validate exists (case-insensitive).
        date_column: Column containing dates (case-insensitive).

    Returns:
        DataFrame indexed by datetime with lowercase column names.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: If required columns are missing.
    """
    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    if date_column not in df.columns:
        raise ValueError(f"CSV must contain a '{date_column}' column.")
    if price_column.lower() not in df.columns:
        raise ValueError(f"CSV must contain a '{price_column}' column.")

    df[date_column] = pd.to_datetime(df[date_column], utc=True)
    df = df.sort_values(date_column).set_index(date_column)
    return clean_ohlcv(df)


def load_parquet(path: str) -> pd.DataFrame:
    """Load OHLCV data from a Parquet file.

    Args:
        path: Path to the Parquet file.

    Returns:
        Cleaned DataFrame with datetime index.
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    df.columns = [c.strip().lower() for c in df.columns]

    # Ensure index is datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df = df.set_index("date")
        else:
            raise ValueError("Parquet must have a datetime index or a 'date' column.")

    df = df.sort_index()
    return clean_ohlcv(df)


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

_OHLCV_COLUMNS = {"open", "high", "low", "close", "volume"}


def clean_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and validate an OHLCV DataFrame.

    Operations:
    - Forward-fill missing prices (common for weekends / holidays).
    - Drop rows where *all* OHLCV values are NaN.
    - Ensure numeric types on OHLCV columns.
    - Sort by index.

    Args:
        df: Raw DataFrame with OHLCV columns (case-insensitive).

    Returns:
        Cleaned DataFrame.
    """
    df = df.copy()

    # Identify which OHLCV columns are present
    present = [c for c in _OHLCV_COLUMNS if c in df.columns]

    # Convert to numeric
    for col in present:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop rows where every OHLCV column is NaN
    if present:
        df = df.dropna(subset=present, how="all")

    # Forward-fill gaps (missing candles on holidays, etc.)
    df[present] = df[present].ffill()

    # Sort chronologically
    df = df.sort_index()

    return df
