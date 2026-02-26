"""Remote data providers for fetching historical market data."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd


class DataProvider(ABC):
    """Abstract interface for remote data sources."""

    @abstractmethod
    def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch historical OHLCV data.

        Args:
            symbol: Ticker symbol (e.g. ``"AAPL"``, ``"BTC-USD"``).
            start: Start datetime.
            end: End datetime.
            interval: Bar interval — ``"1d"``, ``"1h"``, ``"5m"``, etc.

        Returns:
            DataFrame with columns ``[open, high, low, close, volume]``
            indexed by datetime.
        """


# ---------------------------------------------------------------------------
# Yahoo Finance
# ---------------------------------------------------------------------------

class YahooFinanceProvider(DataProvider):
    """Yahoo Finance data provider (free, no API key required).

    Requires the ``yfinance`` package::

        pip install yfinance
    """

    def __init__(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError:
            raise ImportError(
                "yfinance is required for Yahoo Finance data. "
                "Install it with: pip install yfinance"
            )

    # Simple on-disk cache directory
    _CACHE_DIR = Path(".cache/yahoo")

    def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch from Yahoo Finance with a simple file cache."""
        import yfinance as yf

        # Check cache first
        cache_key = f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_{interval}"
        cache_path = self._CACHE_DIR / f"{cache_key}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        df.columns = [c.strip().lower() for c in df.columns]
        df.index.name = "date"

        # Keep only OHLCV
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        # Cache
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

        return df


# ---------------------------------------------------------------------------
# Alpaca Markets
# ---------------------------------------------------------------------------

class AlpacaProvider(DataProvider):
    """Alpaca Markets data provider.

    Requires the ``alpaca-py`` package::

        pip install alpaca-py
    """

    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: Optional[str] = None,
    ) -> None:
        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError:
            raise ImportError(
                "alpaca-py is required for the Alpaca provider. "
                "Install with: pip install alpaca-py"
            )

        self.api_key = api_key
        self.api_secret = api_secret
        self._client = StockHistoricalDataClient(api_key, api_secret)

    # Simple on-disk cache directory
    _CACHE_DIR = Path(".cache/alpaca")

    def fetch_historical(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch from Alpaca with a simple file cache."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        cache_key = f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}_{interval}"
        cache_path = self._CACHE_DIR / f"{cache_key}.parquet"
        if cache_path.exists():
            return pd.read_parquet(cache_path)

        timeframe_map = {
            "1d": TimeFrame.Day,
            "1h": TimeFrame.Hour,
            "1m": TimeFrame.Minute,
        }
        tf = timeframe_map.get(interval, TimeFrame.Day)

        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=tf,
            start=start,
            end=end,
        )
        bars = self._client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            raise ValueError(f"No data returned for {symbol}")

        # Flatten multi-index if present
        if isinstance(df.index, pd.MultiIndex):
            df = df.droplevel("symbol")

        df.columns = [c.strip().lower() for c in df.columns]
        df.index.name = "date"

        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep]

        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)

        return df
