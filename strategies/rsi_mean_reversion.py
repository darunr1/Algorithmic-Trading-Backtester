"""RSI Mean Reversion strategy.

Sells when RSI enters overbought territory, buys when oversold,
and stays flat in the neutral zone.
"""

from __future__ import annotations

import pandas as pd

from strategies.base import Strategy, SignalFrame


class RSIMeanReversion(Strategy):
    """Relative Strength Index mean‑reversion strategy."""

    name = "rsi_mean_reversion"

    def __init__(
        self,
        period: int = 14,
        overbought: float = 70.0,
        oversold: float = 30.0,
    ) -> None:
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    # ------------------------------------------------------------------
    # Signal generation
    # ------------------------------------------------------------------

    def generate_signals(self, market_data: pd.DataFrame) -> SignalFrame:
        prices = market_data["close"]
        rsi = self._compute_rsi(prices)

        signal = pd.Series(0.0, index=prices.index)
        signal[rsi < self.oversold] = 1.0    # Buy when oversold
        signal[rsi > self.overbought] = -1.0  # Sell when overbought

        # Strength of signal scales with RSI distance from neutral (50)
        distance = (rsi - 50).abs() / 50  # 0–1 range
        position_size = distance.clip(lower=0.2, upper=1.0).fillna(0.0)

        return pd.DataFrame(
            {"signal": signal, "position_size": position_size},
            index=prices.index,
        )

    # ------------------------------------------------------------------
    # RSI helper
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        delta = prices.diff()
        gain = delta.where(delta > 0, 0.0).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / loss.replace(0.0, float("nan"))
        return 100.0 - (100.0 / (1.0 + rs))
