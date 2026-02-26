"""Tests for performance metrics."""

import numpy as np
import pandas as pd
import pytest

from metrics.performance import compute_metrics, format_report


@pytest.fixture
def daily_returns():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    returns = pd.Series(np.random.normal(0.0005, 0.01, 500), index=dates)
    return returns


@pytest.fixture
def positions():
    dates = pd.date_range("2020-01-01", periods=500, freq="D")
    return pd.Series(1.0, index=dates)


class TestComputeMetrics:
    def test_returns_backtest_result(self, daily_returns, positions):
        result = compute_metrics(daily_returns, positions)
        assert result.total_return is not None
        assert result.sharpe_ratio is not None
        assert result.sortino_ratio is not None
        assert result.max_drawdown <= 0
        assert 0 <= result.win_rate <= 1

    def test_equity_curve_starts_at_capital(self, daily_returns, positions):
        result = compute_metrics(daily_returns, positions, initial_capital=100_000)
        assert abs(result.equity_curve.iloc[0] - 100_000 * (1 + daily_returns.iloc[0])) < 1

    def test_sharpe_positive_for_positive_returns(self, positions):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        returns = pd.Series(0.001, index=dates)  # constant positive return
        result = compute_metrics(returns, positions[:100])
        assert result.sharpe_ratio > 0

    def test_max_drawdown_reasonable(self, daily_returns, positions):
        result = compute_metrics(daily_returns, positions)
        assert -1 <= result.max_drawdown <= 0

    def test_monthly_returns_table(self, daily_returns, positions):
        result = compute_metrics(daily_returns, positions)
        assert isinstance(result.monthly_returns, pd.DataFrame)

    def test_rolling_sharpe_length(self, daily_returns, positions):
        result = compute_metrics(daily_returns, positions)
        assert len(result.rolling_sharpe) == len(daily_returns)


class TestFormatReport:
    def test_report_string(self, daily_returns, positions):
        result = compute_metrics(daily_returns, positions, strategy_name="test_strat")
        report = format_report(result)
        assert "test_strat" in report
        assert "Sharpe" in report
        assert "Sortino" in report
        assert "Max Drawdown" in report
