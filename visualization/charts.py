"""Professional‑grade backtesting charts.

All plots use matplotlib with a dark theme for a polished look.
Each function can optionally save to a file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")  # Non‑interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as mticker

    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from metrics.performance import BacktestResult


# ============================================================================
# Style
# ============================================================================

def _apply_style() -> None:
    """Apply a dark professional style."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": "#0d1117",
        "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d",
        "axes.labelcolor": "#c9d1d9",
        "xtick.color": "#8b949e",
        "ytick.color": "#8b949e",
        "text.color": "#c9d1d9",
        "grid.color": "#21262d",
        "grid.alpha": 0.6,
        "font.family": "sans-serif",
        "font.size": 10,
    })


def _save_or_show(fig: plt.Figure, save_path: Optional[str]) -> None:
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Equity Curve
# ============================================================================

def plot_equity_curve(
    result: BacktestResult,
    benchmark: Optional[pd.Series] = None,
    save_path: Optional[str] = None,
) -> None:
    """Plot equity curve with optional benchmark overlay.

    Args:
        result: Backtest result.
        benchmark: Optional benchmark equity series.
        save_path: If given, save PNG to this path.
    """
    if not HAS_MPL:
        print("matplotlib not installed — skipping plot.")
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(result.equity_curve.index, result.equity_curve.values,
            color="#58a6ff", linewidth=1.5, label=result.strategy_name or "Strategy")

    if benchmark is not None:
        ax.plot(benchmark.index, benchmark.values,
                color="#8b949e", linewidth=1.0, alpha=0.7, label="Benchmark")

    ax.set_title("Equity Curve", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


# ============================================================================
# Drawdown
# ============================================================================

def plot_drawdown(
    result: BacktestResult,
    save_path: Optional[str] = None,
) -> None:
    """Plot underwater (drawdown) chart."""
    if not HAS_MPL:
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 4))

    eq = result.equity_curve
    rolling_max = eq.cummax()
    dd = (eq - rolling_max) / rolling_max

    ax.fill_between(dd.index, dd.values, 0, color="#f85149", alpha=0.5)
    ax.plot(dd.index, dd.values, color="#f85149", linewidth=0.8)

    ax.set_title("Drawdown", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


# ============================================================================
# Trade Entry / Exit
# ============================================================================

def plot_trades(
    result: BacktestResult,
    prices: pd.Series,
    save_path: Optional[str] = None,
) -> None:
    """Plot price chart with trade entry/exit markers."""
    if not HAS_MPL:
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(prices.index, prices.values, color="#8b949e", linewidth=0.8, label="Price")

    # Identify trade points from position changes
    pos = result.positions
    pos_diff = pos.diff().fillna(0.0)

    buy_mask = pos_diff > 0.1
    sell_mask = pos_diff < -0.1

    ax.scatter(prices.index[buy_mask], prices[buy_mask],
               marker="^", color="#3fb950", s=50, zorder=5, label="Buy")
    ax.scatter(prices.index[sell_mask], prices[sell_mask],
               marker="v", color="#f85149", s=50, zorder=5, label="Sell")

    ax.set_title("Trade Entry / Exit", fontsize=14, fontweight="bold")
    ax.set_ylabel("Price ($)")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)

    _save_or_show(fig, save_path)


# ============================================================================
# Metrics Summary Table
# ============================================================================

def plot_metrics_summary(
    result: BacktestResult,
    save_path: Optional[str] = None,
) -> None:
    """Render a metrics summary table as a figure."""
    if not HAS_MPL:
        return

    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")

    data = [
        ["Total Return", f"{result.total_return:.2%}"],
        ["Annual Return", f"{result.annual_return:.2%}"],
        ["Annual Vol", f"{result.annual_volatility:.2%}"],
        ["Sharpe Ratio", f"{result.sharpe_ratio:.2f}"],
        ["Sortino Ratio", f"{result.sortino_ratio:.2f}"],
        ["Max Drawdown", f"{result.max_drawdown:.2%}"],
        ["Win Rate", f"{result.win_rate:.2%}"],
        ["Profit Factor", f"{result.profit_factor:.2f}"],
        ["Total Trades", f"{result.total_trades}"],
        ["Expectancy", f"{result.expectancy:.4f}"],
    ]

    table = ax.table(
        cellText=data,
        colLabels=["Metric", "Value"],
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)

    # Style
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#30363d")
        if row == 0:
            cell.set_facecolor("#21262d")
            cell.set_text_props(fontweight="bold", color="#58a6ff")
        else:
            cell.set_facecolor("#0d1117")
            cell.set_text_props(color="#c9d1d9")

    ax.set_title("Performance Metrics", fontsize=14, fontweight="bold", pad=20)
    _save_or_show(fig, save_path)


# ============================================================================
# Monthly heatmap
# ============================================================================

def plot_monthly_heatmap(
    result: BacktestResult,
    save_path: Optional[str] = None,
) -> None:
    """Plot a monthly returns heatmap."""
    if not HAS_MPL:
        return

    _apply_style()
    monthly = result.monthly_returns
    if monthly.empty:
        return

    fig, ax = plt.subplots(figsize=(12, max(4, len(monthly) * 0.6)))

    data = monthly.values * 100  # to percent
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(monthly.shape[1]))
    ax.set_xticklabels(monthly.columns, fontsize=9)
    ax.set_yticks(range(monthly.shape[0]))
    ax.set_yticklabels(monthly.index, fontsize=9)

    # Annotate cells
    for i in range(monthly.shape[0]):
        for j in range(monthly.shape[1]):
            val = data[i, j]
            if not np.isnan(val):
                color = "black" if abs(val) < 5 else "white"
                ax.text(j, i, f"{val:.1f}%", ha="center", va="center",
                        color=color, fontsize=8)

    ax.set_title("Monthly Returns (%)", fontsize=14, fontweight="bold")
    fig.colorbar(im, ax=ax, shrink=0.8)
    _save_or_show(fig, save_path)


# ============================================================================
# Full report (all charts combined)
# ============================================================================

def create_full_report(
    result: BacktestResult,
    prices: pd.Series,
    output_dir: str = "reports",
) -> None:
    """Generate all charts and save them to the output directory.

    Args:
        result: Backtest result.
        prices: Raw price series (for trade plot).
        output_dir: Directory to save charts into.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = result.strategy_name or "backtest"

    plot_equity_curve(result, save_path=str(out / f"{name}_equity.png"))
    plot_drawdown(result, save_path=str(out / f"{name}_drawdown.png"))
    plot_trades(result, prices, save_path=str(out / f"{name}_trades.png"))
    plot_metrics_summary(result, save_path=str(out / f"{name}_metrics.png"))
    plot_monthly_heatmap(result, save_path=str(out / f"{name}_monthly.png"))

    print(f"Report saved to {out.resolve()}/")
