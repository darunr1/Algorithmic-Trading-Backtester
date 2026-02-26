"""Pluggable strategy engine — abstract base and built-in strategies."""

from strategies.base import Strategy, StrategyConfig, SignalFrame
from strategies.ma_crossover import MovingAverageCrossover
from strategies.rsi_mean_reversion import RSIMeanReversion
from strategies.momentum_breakout import MomentumBreakout
from strategies.pairs_trading import PairsTrading
from strategies.ml_signal import MLSignal

STRATEGY_REGISTRY: dict[str, type[Strategy]] = {
    "ma_crossover": MovingAverageCrossover,
    "rsi_mean_reversion": RSIMeanReversion,
    "momentum_breakout": MomentumBreakout,
    "pairs_trading": PairsTrading,
    "ml_signal": MLSignal,
}

__all__ = [
    "Strategy",
    "StrategyConfig",
    "SignalFrame",
    "MovingAverageCrossover",
    "RSIMeanReversion",
    "MomentumBreakout",
    "PairsTrading",
    "MLSignal",
    "STRATEGY_REGISTRY",
]
