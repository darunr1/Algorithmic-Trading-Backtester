"""Algorithmic Trading Backtester — CLI entrypoint.

Usage examples::

    python main.py backtest --strategy ma_crossover --symbol AAPL --source yahoo
    python main.py backtest --strategy rsi_mean_reversion data.csv
    python main.py sweep --strategy ma_crossover --fast-span 10,20,50 --slow-span 30,50,100 --symbol AAPL
    python main.py walkforward --strategy momentum_breakout --symbol AAPL
    python main.py compare --strategies ma_crossover,rsi_mean_reversion --symbol AAPL
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


def _load_market_data(args: argparse.Namespace) -> pd.DataFrame:
    """Load market data from file or remote source."""
    if args.csv_path:
        from data.loader import load_csv
        return load_csv(args.csv_path, price_column=args.price_column)

    if args.symbol:
        source = args.source or "yahoo"
        end = datetime.now()
        start = end - timedelta(days=365 * 2)

        if source == "yahoo":
            from data.providers import YahooFinanceProvider
            provider = YahooFinanceProvider()
        elif source == "alpaca":
            if not args.alpaca_key or not args.alpaca_secret:
                print("Error: --alpaca-key and --alpaca-secret required for Alpaca.")
                sys.exit(1)
            from data.providers import AlpacaProvider
            provider = AlpacaProvider(args.alpaca_key, args.alpaca_secret)
        else:
            print(f"Error: unknown source '{source}'")
            sys.exit(1)

        print(f"Fetching data for {args.symbol} from {source}...")
        return provider.fetch_historical(args.symbol, start, end)

    print("Error: provide either a CSV path or --symbol.")
    sys.exit(1)


def _make_strategy(name: str, **kwargs):
    """Instantiate a strategy by name."""
    from strategies import STRATEGY_REGISTRY
    if name not in STRATEGY_REGISTRY:
        print(f"Error: unknown strategy '{name}'.")
        print(f"Available: {', '.join(STRATEGY_REGISTRY.keys())}")
        sys.exit(1)
    # Filter kwargs to only those the strategy accepts
    import inspect
    cls = STRATEGY_REGISTRY[name]
    sig = inspect.signature(cls.__init__)
    valid = {k: v for k, v in kwargs.items() if k in sig.parameters and v is not None}
    return cls(**valid)


def _make_backtester(strategy, args):
    """Build a Backtester with execution + risk config from CLI args."""
    from backtester.engine import Backtester
    from execution.simulator import ExecutionSimulator, FixedSlippage, VolumeSlippage, SpreadSlippage, FeeModel
    from risk.controls import RiskManager, RiskConfig

    # Slippage model
    slippage_map = {
        "fixed": FixedSlippage(bps=getattr(args, "slippage_bps", 5.0)),
        "volume": VolumeSlippage(),
        "spread": SpreadSlippage(),
    }
    slippage = slippage_map.get(getattr(args, "slippage_model", "fixed"), FixedSlippage())

    execution = ExecutionSimulator(
        slippage_model=slippage,
        fee_model=FeeModel(percentage_bps=getattr(args, "fee_bps", 1.0)),
    )

    risk_config = RiskConfig(
        stop_loss_pct=getattr(args, "stop_loss", 0.05),
        take_profit_pct=getattr(args, "take_profit", 0.10),
        max_drawdown_pct=getattr(args, "max_drawdown", 0.20),
    )

    return Backtester(
        strategy=strategy,
        execution=execution,
        risk_manager=RiskManager(risk_config),
        initial_capital=getattr(args, "capital", 100_000.0),
    )


# ============================================================================
# Commands
# ============================================================================

def cmd_backtest(args: argparse.Namespace) -> None:
    """Run a single backtest."""
    from metrics.performance import format_report
    from visualization.charts import create_full_report

    data = _load_market_data(args)
    strategy = _make_strategy(
        args.strategy,
        fast_span=getattr(args, "fast_span", None),
        slow_span=getattr(args, "slow_span", None),
        period=getattr(args, "period", None),
        overbought=getattr(args, "overbought", None),
        oversold=getattr(args, "oversold", None),
        breakout_period=getattr(args, "breakout_period", None),
    )
    bt = _make_backtester(strategy, args)
    result = bt.run(data)

    print(format_report(result))

    if not getattr(args, "no_charts", False):
        create_full_report(result, data["close"], output_dir=getattr(args, "output", "reports"))


def cmd_walkforward(args: argparse.Namespace) -> None:
    """Run walk-forward validation."""
    data = _load_market_data(args)
    strategy = _make_strategy(args.strategy)
    bt = _make_backtester(strategy, args)

    wf = bt.walk_forward(
        data,
        train_window_days=getattr(args, "train_window", 252),
        test_window_days=getattr(args, "test_window", 63),
        step_days=getattr(args, "step_days", 21),
    )
    print(wf.get_summary_text())


def cmd_sweep(args: argparse.Namespace) -> None:
    """Run parameter sweep."""
    from backtester.engine import Backtester
    from strategies import STRATEGY_REGISTRY

    data = _load_market_data(args)
    cls = STRATEGY_REGISTRY.get(args.strategy)
    if cls is None:
        print(f"Error: unknown strategy '{args.strategy}'.")
        sys.exit(1)

    # Parse param grid from CLI
    grid = {}
    if getattr(args, "fast_span", None):
        grid["fast_span"] = [int(x) for x in args.fast_span.split(",")]
    if getattr(args, "slow_span", None):
        grid["slow_span"] = [int(x) for x in args.slow_span.split(",")]
    if getattr(args, "period", None):
        grid["period"] = [int(x) for x in args.period.split(",")]
    if getattr(args, "breakout_period", None):
        grid["breakout_period"] = [int(x) for x in args.breakout_period.split(",")]

    if not grid:
        print("Error: specify at least one parameter grid (e.g. --fast-span 10,20,50)")
        sys.exit(1)

    print(f"Running parameter sweep for {args.strategy}...")
    results = Backtester.parameter_sweep(cls, data, grid)
    print(results.to_string(index=False))

    if getattr(args, "output", None):
        results.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


def cmd_compare(args: argparse.Namespace) -> None:
    """Compare multiple strategies."""
    from backtester.engine import Backtester
    from metrics.performance import format_report

    data = _load_market_data(args)
    names = [s.strip() for s in args.strategies.split(",")]
    strategies = [_make_strategy(n) for n in names]

    bt = _make_backtester(strategies[0], args)  # shared config
    results = Backtester.compare_strategies(
        strategies, data,
        execution=bt.execution,
        risk_manager=bt.risk_manager,
    )

    for name, result in results.items():
        print(f"\n{'─' * 50}")
        print(format_report(result))


# ============================================================================
# Argument parser
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="algorithmic-trading-backtester",
        description="Algorithmic Trading Backtester — a mini quant research platform",
    )
    sub = parser.add_subparsers(dest="command", help="Available commands")

    # -- Shared arguments --
    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("csv_path", nargs="?", help="Path to CSV file with OHLCV data")
        p.add_argument("--symbol", type=str, help="Ticker symbol (e.g. AAPL)")
        p.add_argument("--source", choices=["yahoo", "alpaca"], default="yahoo")
        p.add_argument("--price-column", default="close")
        p.add_argument("--alpaca-key", type=str)
        p.add_argument("--alpaca-secret", type=str)
        p.add_argument("--capital", type=float, default=100_000.0)
        p.add_argument("--slippage-model", choices=["fixed", "volume", "spread"], default="fixed")
        p.add_argument("--slippage-bps", type=float, default=5.0)
        p.add_argument("--fee-bps", type=float, default=1.0)
        p.add_argument("--stop-loss", type=float, default=0.05)
        p.add_argument("--take-profit", type=float, default=0.10)
        p.add_argument("--max-drawdown", type=float, default=0.20)

    def add_strategy_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--strategy", required=True, help="Strategy name")
        p.add_argument("--fast-span", dest="fast_span")
        p.add_argument("--slow-span", dest="slow_span")
        p.add_argument("--period", dest="period")
        p.add_argument("--overbought", type=float)
        p.add_argument("--oversold", type=float)
        p.add_argument("--breakout-period", dest="breakout_period")

    # -- backtest --
    bt_parser = sub.add_parser("backtest", help="Run a single backtest")
    add_common(bt_parser)
    add_strategy_args(bt_parser)
    bt_parser.add_argument("--output", default="reports", help="Output directory for charts")
    bt_parser.add_argument("--no-charts", action="store_true")

    # -- walkforward --
    wf_parser = sub.add_parser("walkforward", help="Walk-forward validation")
    add_common(wf_parser)
    add_strategy_args(wf_parser)
    wf_parser.add_argument("--train-window", type=int, default=252)
    wf_parser.add_argument("--test-window", type=int, default=63)
    wf_parser.add_argument("--step-days", type=int, default=21)

    # -- sweep --
    sw_parser = sub.add_parser("sweep", help="Parameter sweep")
    add_common(sw_parser)
    add_strategy_args(sw_parser)
    sw_parser.add_argument("--output", help="CSV output path")

    # -- compare --
    cmp_parser = sub.add_parser("compare", help="Compare strategies")
    add_common(cmp_parser)
    cmp_parser.add_argument("--strategies", required=True,
                            help="Comma‑separated strategy names")

    return parser


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    commands = {
        "backtest": cmd_backtest,
        "walkforward": cmd_walkforward,
        "sweep": cmd_sweep,
        "compare": cmd_compare,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
