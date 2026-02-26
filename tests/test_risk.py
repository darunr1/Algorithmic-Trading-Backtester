"""Tests for risk management controls."""

import numpy as np
import pandas as pd
import pytest

from risk.controls import RiskManager, RiskConfig


@pytest.fixture
def risk_data():
    dates = pd.date_range("2020-01-01", periods=200, freq="D")
    prices = pd.Series(np.linspace(100, 150, 200), index=dates)
    positions = pd.Series(1.0, index=dates)
    return positions, prices


class TestRiskManager:
    def test_default_config(self, risk_data):
        positions, prices = risk_data
        rm = RiskManager()
        adjusted = rm.apply(positions, prices)
        assert len(adjusted) == len(positions)
        assert not adjusted.isna().any()

    def test_leverage_cap(self, risk_data):
        positions, prices = risk_data
        positions = positions * 5.0  # Way over leverage
        cfg = RiskConfig(max_leverage=2.0)
        rm = RiskManager(cfg)
        adjusted = rm.apply(positions, prices)
        assert adjusted.max() <= 2.0

    def test_drawdown_scaling_reduces_position(self):
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        # Create a price series with a big drawdown
        prices = pd.Series(
            np.concatenate([np.linspace(100, 120, 50), np.linspace(120, 80, 50)]),
            index=dates,
        )
        positions = pd.Series(1.0, index=dates)
        cfg = RiskConfig(drawdown_risk_scaling=True, max_drawdown_pct=0.20)
        rm = RiskManager(cfg)
        adjusted = rm.apply(positions, prices)
        # After the drawdown, positions should be scaled down
        assert adjusted.iloc[-1] < 1.0

    def test_no_scaling_when_disabled(self, risk_data):
        positions, prices = risk_data
        cfg = RiskConfig(drawdown_risk_scaling=False)
        rm = RiskManager(cfg)
        adjusted = rm.apply(positions, prices)
        # Should mostly just apply sizing and leverage cap
        assert len(adjusted) == len(positions)


class TestRiskConfig:
    def test_defaults(self):
        cfg = RiskConfig()
        assert cfg.stop_loss_pct == 0.05
        assert cfg.take_profit_pct == 0.10
        assert cfg.max_drawdown_pct == 0.20
        assert cfg.sizing_method == "fixed_pct"
