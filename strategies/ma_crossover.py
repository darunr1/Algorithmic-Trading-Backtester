"""Moving Average Crossover strategy.

Goes long when the fast EMA crosses above the slow EMA, flat otherwise.
Position size is determined by inverse volatility (vol targeting).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from strategies.base import Strategy, SignalFrame


class MovingAverageCrossover(Strategy):
    """EMA crossover with volatility‑targeted position sizing."""

    name = "ma_crossover"

    def __init__(self, fast_span: int = 12, slow_span: int = 48) -> None:
        self.fast_span = fast_span
        self.slow_span = slow_span

    def generate_signals(self, market_data: pd.DataFrame) -> SignalFrame:
        prices = market_data["close"]

        fast_ema = prices.ewm(span=self.fast_span, adjust=False).mean()
        slow_ema = prices.ewm(span=self.slow_span, adjust=False).mean()

        signal = pd.Series(0.0, index=prices.index)
        signal[fast_ema > slow_ema] = 1.0
        signal[fast_ema < slow_ema] = -1.0

        # Position size: inverse‑vol targeting
        daily_ret = prices.pct_change().fillna(0.0)
        rolling_vol = daily_ret.rolling(20).std().replace(0.0, float("nan"))
        ann_vol = rolling_vol * (252 ** 0.5)
        position_size = (0.15 / ann_vol).clip(upper=2.0).fillna(0.0)

        return pd.DataFrame(
            {"signal": signal, "position_size": position_size},
            index=prices.index,
        )
