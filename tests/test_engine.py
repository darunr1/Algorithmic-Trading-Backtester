"""Integration tests for the backtesting engine."""

import numpy as np
import pandas as pd
import pytest

from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from backtester.engine import Backtester


@pytest.fixture
def market_data():
    np.random.seed(42)
    n = 300
    dates = pd.date_range("2019-01-01", periods=n, freq="D")
    close = np.cumsum(np.random.normal(0.05, 1, n)) + 100
    return pd.DataFrame({
        "open": close - np.random.uniform(0, 1, n),
        "high": close + np.random.uniform(0, 2, n),
        "low": close - np.random.uniform(0, 2, n),
        "close": close,
        "volume": np.random.randint(100000, 1000000, n),
    }, index=dates)


class TestBacktester:
    def test_single_backtest(self, market_data):
        bt = Backtester(strategy=MovingAverageCrossover())
        result = bt.run(market_data)

        assert result.total_return is not None
        assert result.strategy_name == "ma_crossover"
        assert len(result.equity_curve) == len(market_data)

    def test_walk_forward(self, market_data):
        bt = Backtester(strategy=MovingAverageCrossover())
        wf = bt.walk_forward(
            market_data,
            train_window_days=60,
            test_window_days=30,
            step_days=15,
        )
        assert len(wf.period_results) > 0
        assert not wf.summary.empty
        text = wf.get_summary_text()
        assert "Walk-Forward" in text

    def test_parameter_sweep(self, market_data):
        results = Backtester.parameter_sweep(
            MovingAverageCrossover,
            market_data,
            param_grid={"fast_span": [10, 20], "slow_span": [30, 50]},
        )
        assert len(results) == 4  # 2 x 2 combinations
        assert "sharpe_ratio" in results.columns

    def test_compare_strategies(self, market_data):
        strategies = [
            MovingAverageCrossover(fast_span=10, slow_span=30),
            RSIMeanReversion(period=14),
        ]
        results = Backtester.compare_strategies(strategies, market_data)
        assert "ma_crossover" in results
        assert "rsi_mean_reversion" in results
