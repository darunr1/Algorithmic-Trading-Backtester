"""Momentum Breakout strategy.

Goes long when the price breaks above the N‑day high,
short when it breaks below the N‑day low.
Uses ATR‑based position sizing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, SignalFrame


class MomentumBreakout(Strategy):
    """Donchian‑channel breakout with ATR position sizing."""

    name = "momentum_breakout"

    def __init__(
        self,
        breakout_period: int = 20,
        atr_period: int = 14,
        risk_per_trade: float = 0.02,
    ) -> None:
        self.breakout_period = breakout_period
        self.atr_period = atr_period
        self.risk_per_trade = risk_per_trade

    def generate_signals(self, market_data: pd.DataFrame) -> SignalFrame:
        high = market_data["high"]
        low = market_data["low"]
        close = market_data["close"]

        # Donchian channel
        upper = high.rolling(self.breakout_period).max()
        lower = low.rolling(self.breakout_period).min()

        signal = pd.Series(0.0, index=close.index)
        signal[close > upper.shift(1)] = 1.0   # Breakout above
        signal[close < lower.shift(1)] = -1.0  # Breakdown below

        # Forward‑fill signal (stay in position until opposite breakout)
        signal = signal.replace(0.0, float("nan")).ffill().fillna(0.0)

        # ATR‑based position sizing
        atr = self._compute_atr(market_data)
        position_size = (self.risk_per_trade * close / atr).clip(upper=2.0).fillna(0.0)

        return pd.DataFrame(
            {"signal": signal, "position_size": position_size},
            index=close.index,
        )

    # ------------------------------------------------------------------

    def _compute_atr(self, market_data: pd.DataFrame) -> pd.Series:
        """Average True Range."""
        high = market_data["high"]
        low = market_data["low"]
        close = market_data["close"]

        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        return true_range.rolling(self.atr_period).mean()
