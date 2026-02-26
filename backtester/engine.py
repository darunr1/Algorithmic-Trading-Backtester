"""Core backtesting engine.

Orchestrates the full pipeline::

    Market Data → Strategy → Risk Manager → Execution Simulator
                → Portfolio → Metrics → BacktestResult

Supports walk‑forward validation, train/test splits, parameter sweeps,
and multi‑strategy comparison.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from strategies.base import Strategy
from execution.simulator import ExecutionSimulator, FixedSlippage, FeeModel
from risk.controls import RiskManager, RiskConfig
from metrics.performance import BacktestResult, compute_metrics, format_report
from portfolio.manager import Portfolio


# ============================================================================
# Backtester
# ============================================================================

class Backtester:
    """Main backtesting engine.

    Example::

        from strategies import MovingAverageCrossover
        bt = Backtester(strategy=MovingAverageCrossover(fast_span=20, slow_span=50))
        result = bt.run(market_data)
        print(format_report(result))
    """

    def __init__(
        self,
        strategy: Strategy,
        execution: Optional[ExecutionSimulator] = None,
        risk_manager: Optional[RiskManager] = None,
        initial_capital: float = 100_000.0,
    ) -> None:
        self.strategy = strategy
        self.execution = execution or ExecutionSimulator(
            slippage_model=FixedSlippage(bps=5.0),
            fee_model=FeeModel(percentage_bps=1.0),
        )
        self.risk_manager = risk_manager or RiskManager()
        self.initial_capital = initial_capital

    # ------------------------------------------------------------------
    # Single backtest
    # ------------------------------------------------------------------

    def run(self, market_data: pd.DataFrame) -> BacktestResult:
        """Run one backtest on the given market data.

        Args:
            market_data: OHLCV DataFrame with DatetimeIndex.

        Returns:
            Fully‑populated :class:`BacktestResult`.
        """
        prices = market_data["close"]
        volume = market_data.get("volume")

        # 1. Strategy signals
        signals = self.strategy.generate_signals(market_data)
        raw_position = (signals["signal"] * signals["position_size"]).fillna(0.0)

        # 2. Risk management
        adjusted_position = self.risk_manager.apply(raw_position, prices)

        # 3. Execution simulation
        exec_result = self.execution.apply(
            adjusted_position, prices, volume
        )
        effective_pos = exec_result["position"]
        total_cost = exec_result["total_cost"]

        # 4. Compute strategy returns
        daily_ret = prices.pct_change().fillna(0.0)
        strategy_returns = effective_pos.shift(1).fillna(0.0) * daily_ret
        strategy_returns -= total_cost / (self.initial_capital * (1 + strategy_returns).cumprod().shift(1).fillna(1.0))

        # 5. Metrics
        return compute_metrics(
            daily_returns=strategy_returns,
            positions=effective_pos,
            strategy_name=self.strategy.name,
            initial_capital=self.initial_capital,
        )

    # ------------------------------------------------------------------
    # Walk‑forward validation
    # ------------------------------------------------------------------

    def walk_forward(
        self,
        market_data: pd.DataFrame,
        train_window_days: int = 252,
        test_window_days: int = 63,
        step_days: int = 21,
    ) -> WalkForwardResult:
        """Run walk‑forward analysis.

        Slides a train/test window through the data and collects
        out‑of‑sample performance for each period.

        Args:
            market_data: Full OHLCV dataset.
            train_window_days: Length of training window.
            test_window_days: Length of test window.
            step_days: Step between windows.

        Returns:
            :class:`WalkForwardResult` with per‑period results + summary.
        """
        results: List[BacktestResult] = []
        periods: List[dict] = []

        dates = market_data.index
        start = dates[0]
        end = dates[-1]

        cursor = start + timedelta(days=train_window_days)

        while cursor + timedelta(days=test_window_days) <= end:
            # Train period (not used for param opt yet, but available)
            train_start = cursor - timedelta(days=train_window_days)
            train_end = cursor

            # Test period
            test_start = cursor
            test_end = cursor + timedelta(days=test_window_days)

            test_data = market_data[
                (market_data.index >= test_start) & (market_data.index < test_end)
            ]

            if len(test_data) > 5:
                result = self.run(test_data)
                results.append(result)
                periods.append({
                    "train_start": train_start,
                    "train_end": train_end,
                    "test_start": test_start,
                    "test_end": test_end,
                    "annual_return": result.annual_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "max_drawdown": result.max_drawdown,
                })

            cursor += timedelta(days=step_days)

        summary = pd.DataFrame(periods)
        return WalkForwardResult(period_results=results, summary=summary)

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    @staticmethod
    def parameter_sweep(
        strategy_cls: type,
        market_data: pd.DataFrame,
        param_grid: Dict[str, Sequence],
        execution: Optional[ExecutionSimulator] = None,
        risk_manager: Optional[RiskManager] = None,
        initial_capital: float = 100_000.0,
    ) -> pd.DataFrame:
        """Run backtests over a grid of strategy parameters.

        Args:
            strategy_cls: Strategy class to instantiate.
            market_data: OHLCV data.
            param_grid: e.g. ``{"fast_span": [10, 20, 50], "slow_span": [30, 50, 100]}``.

        Returns:
            DataFrame with one row per parameter combination and key metrics.
        """
        keys = list(param_grid.keys())
        values = list(param_grid.values())
        rows = []

        for combo in itertools.product(*values):
            params = dict(zip(keys, combo))
            try:
                strategy = strategy_cls(**params)
                bt = Backtester(
                    strategy=strategy,
                    execution=execution,
                    risk_manager=risk_manager,
                    initial_capital=initial_capital,
                )
                result = bt.run(market_data)
                rows.append({
                    **params,
                    "annual_return": result.annual_return,
                    "sharpe_ratio": result.sharpe_ratio,
                    "sortino_ratio": result.sortino_ratio,
                    "max_drawdown": result.max_drawdown,
                    "win_rate": result.win_rate,
                    "profit_factor": result.profit_factor,
                    "total_trades": result.total_trades,
                })
            except Exception as e:
                rows.append({**params, "error": str(e)})

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Multi‑strategy comparison
    # ------------------------------------------------------------------

    @staticmethod
    def compare_strategies(
        strategies: List[Strategy],
        market_data: pd.DataFrame,
        execution: Optional[ExecutionSimulator] = None,
        risk_manager: Optional[RiskManager] = None,
        initial_capital: float = 100_000.0,
    ) -> Dict[str, BacktestResult]:
        """Run multiple strategies on the same data and return results.

        Returns:
            Dict mapping strategy name to :class:`BacktestResult`.
        """
        results = {}
        for strat in strategies:
            bt = Backtester(
                strategy=strat,
                execution=execution,
                risk_manager=risk_manager,
                initial_capital=initial_capital,
            )
            results[strat.name] = bt.run(market_data)
        return results


# ============================================================================
# Walk-forward result container
# ============================================================================

@dataclass
class WalkForwardResult:
    """Results from walk‑forward analysis."""

    period_results: List[BacktestResult]
    summary: pd.DataFrame

    def get_summary_text(self) -> str:
        if self.summary.empty:
            return "No walk-forward periods generated."

        lines = [
            "Walk-Forward Analysis Summary",
            "=" * 50,
            "",
            f"  Number of Periods:     {len(self.summary)}",
            f"  Avg Annual Return:     {self.summary['annual_return'].mean():.2%}",
            f"  Avg Sharpe Ratio:      {self.summary['sharpe_ratio'].mean():.2f}",
            f"  Avg Max Drawdown:      {self.summary['max_drawdown'].mean():.2%}",
            f"  Positive Periods:      {(self.summary['annual_return'] > 0).sum()}/{len(self.summary)}",
            "=" * 50,
        ]
        return "\n".join(lines)
