"""Quantitative trading bot with volatility-targeted trend strategy."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class StrategyConfig:
    fast_ema_span: int = 12
    slow_ema_span: int = 48
    vol_lookback: int = 20
    target_vol: float = 0.15
    max_leverage: float = 2.0
    transaction_cost_bps: float = 1.0


@dataclass(frozen=True)
class BacktestResult:
    equity_curve: pd.Series
    daily_returns: pd.Series
    sharpe_ratio: float
    annual_return: float
    annual_volatility: float
    max_drawdown: float


def load_price_data(csv_path: str, price_column: str = "close") -> pd.DataFrame:
    """Load OHLCV data from CSV and return cleaned dataframe.

    Expected columns: date, open, high, low, close, volume (case-insensitive).
    """
    data = pd.read_csv(csv_path)
    data.columns = [col.lower() for col in data.columns]
    if "date" not in data.columns:
        raise ValueError("CSV must contain a 'date' column.")
    if price_column.lower() not in data.columns:
        raise ValueError(f"CSV must contain '{price_column}' column.")
    data["date"] = pd.to_datetime(data["date"], utc=True)
    data = data.sort_values("date").set_index("date")
    return data


def compute_strategy_returns(
    prices: pd.Series,
    config: StrategyConfig,
) -> pd.Series:
    """Compute daily strategy returns with volatility targeting and costs."""
    fast_ema = prices.ewm(span=config.fast_ema_span, adjust=False).mean()
    slow_ema = prices.ewm(span=config.slow_ema_span, adjust=False).mean()
    trend_signal = (fast_ema > slow_ema).astype(float)

    daily_returns = prices.pct_change().fillna(0.0)
    rolling_vol = daily_returns.rolling(config.vol_lookback).std().replace(0.0, pd.NA)
    vol_target = config.target_vol / (rolling_vol * (252**0.5))
    position_size = vol_target.clip(upper=config.max_leverage).fillna(0.0)

    raw_position = trend_signal * position_size
    position = raw_position.shift(1).fillna(0.0)

    turnover = position.diff().abs().fillna(0.0)
    transaction_cost = turnover * (config.transaction_cost_bps / 10_000)

    strategy_returns = position * daily_returns - transaction_cost
    return strategy_returns


def calculate_performance(strategy_returns: pd.Series) -> BacktestResult:
    """Calculate performance metrics from strategy returns."""
    equity_curve = (1 + strategy_returns).cumprod()
    daily_vol = strategy_returns.std()
    sharpe_ratio = (
        (strategy_returns.mean() / daily_vol) * (252**0.5) if daily_vol else 0.0
    )
    annual_return = equity_curve.iloc[-1] ** (252 / len(equity_curve)) - 1
    annual_volatility = daily_vol * (252**0.5)

    rolling_max = equity_curve.cummax()
    drawdown = (equity_curve - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    return BacktestResult(
        equity_curve=equity_curve,
        daily_returns=strategy_returns,
        sharpe_ratio=sharpe_ratio,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        max_drawdown=max_drawdown,
    )


def run_backtest(
    csv_path: str,
    price_column: str = "close",
    config: Optional[StrategyConfig] = None,
) -> BacktestResult:
    """Run backtest using the configured strategy."""
    config = config or StrategyConfig()
    data = load_price_data(csv_path, price_column=price_column)
    prices = data[price_column.lower()].astype(float)
    strategy_returns = compute_strategy_returns(prices, config)
    return calculate_performance(strategy_returns)


def format_report(result: BacktestResult) -> str:
    """Create a human-readable performance summary."""
    return "\n".join(
        [
            "Performance Summary",
            "-------------------",
            f"Annual Return: {result.annual_return:.2%}",
            f"Annual Volatility: {result.annual_volatility:.2%}",
            f"Sharpe Ratio: {result.sharpe_ratio:.2f}",
            f"Max Drawdown: {result.max_drawdown:.2%}",
        ]
    )


def main() -> None:
    """CLI entrypoint."""
    import argparse
    import os
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Quantitative trading bot backtest")
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file with price data",
    )
    parser.add_argument(
        "--price-column",
        dest="price_column",
        default="close",
        help="Column for price data (default: close)",
    )
    parser.add_argument("--fast-ema", type=int, default=12, help="Fast EMA span")
    parser.add_argument("--slow-ema", type=int, default=48, help="Slow EMA span")
    parser.add_argument("--vol-lookback", type=int, default=20, help="Volatility window")
    parser.add_argument("--target-vol", type=float, default=0.15, help="Target volatility")
    parser.add_argument("--max-leverage", type=float, default=2.0, help="Max leverage")
    parser.add_argument(
        "--transaction-cost-bps",
        type=float,
        default=1.0,
        help="Transaction cost in basis points",
    )

    args = parser.parse_args()
    
    # Resolve and validate the CSV path
    # Path handles both forward and backslashes on Windows automatically
    csv_path = Path(args.csv_path)
    
    # Resolve relative paths to absolute
    if not csv_path.is_absolute():
        csv_path = (Path.cwd() / csv_path).resolve()
    else:
        csv_path = csv_path.resolve()
    
    # Validate the path exists and is a file
    if not csv_path.exists():
        parser.error(f"CSV file not found: {csv_path}")
    if not csv_path.is_file():
        parser.error(f"Path exists but is not a file: {csv_path}")
    
    config = StrategyConfig(
        fast_ema_span=args.fast_ema,
        slow_ema_span=args.slow_ema,
        vol_lookback=args.vol_lookback,
        target_vol=args.target_vol,
        max_leverage=args.max_leverage,
        transaction_cost_bps=args.transaction_cost_bps,
    )

    result = run_backtest(str(csv_path), args.price_column, config)
    print(format_report(result))


if __name__ == "__main__":
    main()
