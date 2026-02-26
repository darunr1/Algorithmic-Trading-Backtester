"""Tests for all strategy implementations."""

import numpy as np
import pandas as pd
import pytest

from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from strategies.momentum_breakout import MomentumBreakout
from strategies.pairs_trading import PairsTrading
from strategies.ml_signal import MLSignal


@pytest.fixture
def sample_ohlcv():
    """OHLCV DataFrame with 200 bars of trending data."""
    np.random.seed(42)
    n = 200
    dates = pd.date_range("2020-01-01", periods=n, freq="D")
    close = np.cumsum(np.random.normal(0.05, 1, n)) + 100
    return pd.DataFrame({
        "open": close - np.random.uniform(0, 1, n),
        "high": close + np.random.uniform(0, 2, n),
        "low": close - np.random.uniform(0, 2, n),
        "close": close,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=dates)


class TestMovingAverageCrossover:
    def test_generates_valid_signals(self, sample_ohlcv):
        strat = MovingAverageCrossover(fast_span=10, slow_span=30)
        signals = strat.generate_signals(sample_ohlcv)

        assert "signal" in signals.columns
        assert "position_size" in signals.columns
        assert len(signals) == len(sample_ohlcv)
        assert set(signals["signal"].dropna().unique()).issubset({-1.0, 0.0, 1.0})

    def test_no_nan_in_output(self, sample_ohlcv):
        strat = MovingAverageCrossover()
        signals = strat.generate_signals(sample_ohlcv)
        assert not signals["signal"].isna().any()

    def test_name(self):
        assert MovingAverageCrossover().name == "ma_crossover"


class TestRSIMeanReversion:
    def test_generates_valid_signals(self, sample_ohlcv):
        strat = RSIMeanReversion(period=14, overbought=70, oversold=30)
        signals = strat.generate_signals(sample_ohlcv)

        assert "signal" in signals.columns
        assert len(signals) == len(sample_ohlcv)

    def test_position_size_bounded(self, sample_ohlcv):
        strat = RSIMeanReversion()
        signals = strat.generate_signals(sample_ohlcv)
        assert signals["position_size"].max() <= 1.0
        assert signals["position_size"].min() >= 0.0


class TestMomentumBreakout:
    def test_generates_valid_signals(self, sample_ohlcv):
        strat = MomentumBreakout(breakout_period=20)
        signals = strat.generate_signals(sample_ohlcv)

        assert "signal" in signals.columns
        assert len(signals) == len(sample_ohlcv)

    def test_signals_are_directional(self, sample_ohlcv):
        strat = MomentumBreakout()
        signals = strat.generate_signals(sample_ohlcv)
        unique = set(signals["signal"].dropna().unique())
        assert unique.issubset({-1.0, 0.0, 1.0})


class TestPairsTrading:
    def test_generates_signals_with_two_columns(self):
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        base = np.cumsum(np.random.normal(0, 1, n)) + 100
        data = pd.DataFrame({
            "close_a": base + np.random.normal(0, 0.5, n),
            "close_b": base * 1.1 + np.random.normal(0, 0.5, n),
        }, index=dates)

        strat = PairsTrading(symbol_a="close_a", symbol_b="close_b")
        signals = strat.generate_signals(data)
        assert "signal" in signals.columns
        assert len(signals) == n


class TestMLSignal:
    def test_generates_valid_signals(self, sample_ohlcv):
        strat = MLSignal(train_window=50, retrain_every=10, n_iterations=50)
        signals = strat.generate_signals(sample_ohlcv)

        assert "signal" in signals.columns
        assert "position_size" in signals.columns
        assert len(signals) == len(sample_ohlcv)

    def test_confidence_bounded(self, sample_ohlcv):
        strat = MLSignal(train_window=50, retrain_every=10, n_iterations=50)
        signals = strat.generate_signals(sample_ohlcv)
        assert signals["position_size"].max() <= 1.0
        assert signals["position_size"].min() >= 0.0
