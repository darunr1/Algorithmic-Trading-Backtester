"""Tests for portfolio manager."""

import numpy as np
import pandas as pd
import pytest

from portfolio.manager import Portfolio, Trade


class TestPortfolio:
    def test_initial_state(self):
        port = Portfolio(initial_cash=50_000)
        assert port.initial_cash == 50_000
        assert port.final_equity == 50_000
        assert port.trade_count == 0

    def test_from_backtest_vectors(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        positions = pd.Series(np.ones(100), index=dates)
        prices = pd.Series(np.linspace(100, 150, 100), index=dates)
        daily_ret = prices.pct_change().fillna(0.0)
        costs = pd.Series(0.0, index=dates)

        port = Portfolio.from_backtest_vectors(
            dates, positions, prices, daily_ret, costs
        )

        assert len(port.equity_curve) == 100
        assert port.final_equity > 0
        assert len(port.leverage_series) == 100

    def test_equity_curve_shape(self):
        dates = pd.date_range("2020-01-01", periods=50, freq="D")
        daily_ret = pd.Series(np.random.normal(0.001, 0.01, 50), index=dates)
        positions = pd.Series(1.0, index=dates)
        prices = pd.Series(100.0, index=dates)
        costs = pd.Series(0.0, index=dates)

        port = Portfolio.from_backtest_vectors(
            dates, positions, prices, daily_ret, costs
        )
        assert isinstance(port.equity_curve, pd.Series)
        assert len(port.equity_curve) == 50


class TestTrade:
    def test_notional(self):
        t = Trade(
            timestamp=pd.Timestamp("2020-01-01"),
            side="buy",
            quantity=10,
            price=100.0,
            cost=1.0,
        )
        assert t.notional == 1000.0
