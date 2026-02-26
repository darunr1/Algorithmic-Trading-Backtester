"""Tests for the execution simulator."""

import numpy as np
import pandas as pd
import pytest

from execution.simulator import (
    ExecutionSimulator,
    FixedSlippage,
    VolumeSlippage,
    SpreadSlippage,
    FeeModel,
)


@pytest.fixture
def sample_data():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    prices = pd.Series(np.linspace(100, 150, 100), index=dates)
    volume = pd.Series(np.random.randint(100000, 1000000, 100), index=dates)
    positions = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 100)), index=dates)
    return prices, volume, positions


class TestFixedSlippage:
    def test_computes_correctly(self, sample_data):
        prices, volume, positions = sample_data
        slip = FixedSlippage(bps=10.0)
        cost = slip.compute(prices, positions.diff().fillna(0), volume)
        assert (cost >= 0).all()
        expected = prices * (10.0 / 10_000)
        pd.testing.assert_series_equal(cost, expected)


class TestVolumeSlippage:
    def test_returns_positive_costs(self, sample_data):
        prices, volume, positions = sample_data
        slip = VolumeSlippage()
        cost = slip.compute(prices, positions.diff().fillna(0), volume)
        assert (cost >= 0).all()


class TestSpreadSlippage:
    def test_returns_positive_costs(self, sample_data):
        prices, volume, positions = sample_data
        slip = SpreadSlippage(half_spread_bps=5.0)
        cost = slip.compute(prices, positions.diff().fillna(0), volume)
        assert (cost >= 0).all()


class TestFeeModel:
    def test_percentage_fee(self):
        prices = pd.Series([100.0, 200.0])
        quantity = pd.Series([10.0, 5.0])
        fees = FeeModel(percentage_bps=10.0)
        result = fees.compute(prices, quantity)
        # 100 * 10 * 10/10000 = 1.0, 200 * 5 * 10/10000 = 1.0
        assert abs(result.iloc[0] - 1.0) < 1e-9
        assert abs(result.iloc[1] - 1.0) < 1e-9

    def test_flat_fee(self):
        prices = pd.Series([100.0])
        quantity = pd.Series([10.0])
        fees = FeeModel(per_trade_flat=5.0, percentage_bps=0.0)
        result = fees.compute(prices, quantity)
        assert abs(result.iloc[0] - 5.0) < 1e-9


class TestExecutionSimulator:
    def test_apply_returns_all_columns(self, sample_data):
        prices, volume, positions = sample_data
        sim = ExecutionSimulator()
        result = sim.apply(positions, prices, volume)

        assert "position" in result.columns
        assert "trade_size" in result.columns
        assert "slippage_cost" in result.columns
        assert "fee_cost" in result.columns
        assert "total_cost" in result.columns
        assert len(result) == len(prices)

    def test_no_trades_no_cost(self):
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = pd.Series(100.0, index=dates)
        positions = pd.Series(1.0, index=dates)  # constant position
        sim = ExecutionSimulator()
        result = sim.apply(positions, prices)
        # After the initial position, costs should be near zero
        assert result["trade_size"].iloc[2:].abs().sum() < 1e-9
