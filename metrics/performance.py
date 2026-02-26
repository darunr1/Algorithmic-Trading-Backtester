"""Performance metrics suite.

Computes everything a recruiter would expect to see from a quant backtest:
total return, Sharpe, Sortino, max drawdown, win rate, profit factor,
rolling Sharpe, and monthly/annual breakdowns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# Result container
# ============================================================================

@dataclass
class BacktestResult:
    """Immutable container for all backtest outputs and metrics."""

    # -- Time series ---
    equity_curve: pd.Series
    daily_returns: pd.Series
    positions: pd.Series

    # -- Scalar metrics ---
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_win: float
    avg_loss: float
    expectancy: float

    # -- Tabular breakdowns ---
    monthly_returns: pd.DataFrame
    rolling_sharpe: pd.Series

    # -- Metadata ---
    strategy_name: str = ""
    config: Optional[dict] = None


# ============================================================================
# Compute everything
# ============================================================================

TRADING_DAYS = 252


def compute_metrics(
    daily_returns: pd.Series,
    positions: pd.Series,
    strategy_name: str = "",
    config: Optional[dict] = None,
    risk_free_rate: float = 0.0,
    initial_capital: float = 100_000.0,
) -> BacktestResult:
    """Compute the full performance metrics suite.

    Args:
        daily_returns: Strategy daily returns (after costs).
        positions: Position sizes over time.
        strategy_name: Name for labelling.
        config: Strategy config dict for reproducibility.
        risk_free_rate: Annual risk‑free rate.
        initial_capital: Starting capital.

    Returns:
        Fully‑populated :class:`BacktestResult`.
    """
    equity_curve = initial_capital * (1 + daily_returns).cumprod()

    # --- Scalar metrics ---
    total_return = float(equity_curve.iloc[-1] / initial_capital - 1)
    n_days = len(daily_returns)
    annual_return = float((1 + total_return) ** (TRADING_DAYS / max(n_days, 1)) - 1)

    daily_vol = float(daily_returns.std())
    annual_volatility = daily_vol * np.sqrt(TRADING_DAYS)

    excess = daily_returns - risk_free_rate / TRADING_DAYS
    sharpe_ratio = (
        float(excess.mean() / excess.std() * np.sqrt(TRADING_DAYS))
        if excess.std() > 0 else 0.0
    )

    sortino_ratio = _sortino(daily_returns, risk_free_rate)
    max_drawdown = _max_drawdown(equity_curve)

    # --- Trade‑level stats ---
    winning = daily_returns[daily_returns > 0]
    losing = daily_returns[daily_returns < 0]

    total_trades = int((positions.diff().abs() > 1e-9).sum())
    win_rate = float(len(winning) / max(len(winning) + len(losing), 1))
    avg_win = float(winning.mean()) if len(winning) > 0 else 0.0
    avg_loss = float(losing.mean()) if len(losing) > 0 else 0.0
    gross_profit = float(winning.sum())
    gross_loss = float(abs(losing.sum())) if len(losing) > 0 else 1.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

    # --- Rolling Sharpe (63‑day ~ 3 months) ---
    rolling_sharpe = _rolling_sharpe(daily_returns, window=63)

    # --- Monthly returns ---
    monthly_returns = _monthly_returns_table(daily_returns)

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=daily_returns,
        positions=positions,
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe_ratio=sharpe_ratio,
        sortino_ratio=sortino_ratio,
        max_drawdown=max_drawdown,
        win_rate=win_rate,
        profit_factor=profit_factor,
        total_trades=total_trades,
        avg_win=avg_win,
        avg_loss=avg_loss,
        expectancy=expectancy,
        monthly_returns=monthly_returns,
        rolling_sharpe=rolling_sharpe,
        strategy_name=strategy_name,
        config=config,
    )


# ============================================================================
# Helpers
# ============================================================================

def _sortino(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    excess = daily_returns - risk_free_rate / TRADING_DAYS
    downside = excess[excess < 0]
    downside_std = float(downside.std()) if len(downside) > 0 else 0.0
    if downside_std == 0:
        return 0.0
    return float(excess.mean() / downside_std * np.sqrt(TRADING_DAYS))


def _max_drawdown(equity_curve: pd.Series) -> float:
    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    return float(drawdown.min())


def _rolling_sharpe(
    daily_returns: pd.Series, window: int = 63
) -> pd.Series:
    rolling_mean = daily_returns.rolling(window).mean()
    rolling_std = daily_returns.rolling(window).std().replace(0.0, float("nan"))
    return (rolling_mean / rolling_std * np.sqrt(TRADING_DAYS)).fillna(0.0)


def _monthly_returns_table(daily_returns: pd.Series) -> pd.DataFrame:
    """Pivot table of monthly returns — rows = years, columns = months."""
    monthly = daily_returns.resample("ME").apply(lambda x: (1 + x).prod() - 1)
    if monthly.empty:
        return pd.DataFrame()
    table = pd.DataFrame(
        {
            "year": monthly.index.year,
            "month": monthly.index.month,
            "return": monthly.values,
        }
    )
    pivoted = table.pivot_table(index="year", columns="month", values="return")
    pivoted.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ][: len(pivoted.columns)]
    return pivoted


# ============================================================================
# Text report
# ============================================================================

def format_report(result: BacktestResult) -> str:
    """Create a human‑readable performance summary."""
    lines = [
        f"{'═' * 50}",
        f"  Performance Report — {result.strategy_name or 'Backtest'}",
        f"{'═' * 50}",
        "",
        f"  Total Return:       {result.total_return:>10.2%}",
        f"  Annual Return:      {result.annual_return:>10.2%}",
        f"  Annual Volatility:  {result.annual_volatility:>10.2%}",
        f"  Sharpe Ratio:       {result.sharpe_ratio:>10.2f}",
        f"  Sortino Ratio:      {result.sortino_ratio:>10.2f}",
        f"  Max Drawdown:       {result.max_drawdown:>10.2%}",
        "",
        f"  Total Trades:       {result.total_trades:>10d}",
        f"  Win Rate:           {result.win_rate:>10.2%}",
        f"  Profit Factor:      {result.profit_factor:>10.2f}",
        f"  Avg Win:            {result.avg_win:>10.4f}",
        f"  Avg Loss:           {result.avg_loss:>10.4f}",
        f"  Expectancy:         {result.expectancy:>10.4f}",
        f"{'═' * 50}",
    ]
    return "\n".join(lines)
