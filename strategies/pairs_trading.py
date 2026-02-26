"""Statistical Arbitrage — Pairs Trading strategy.

Trades the z‑score of the spread between two co‑integrated assets.
Goes long the spread when it's below −threshold and short when above +threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.base import Strategy, SignalFrame


class PairsTrading(Strategy):
    """Z‑score pairs trading on two correlated assets."""

    name = "pairs_trading"

    def __init__(
        self,
        symbol_a: str = "close_a",
        symbol_b: str = "close_b",
        lookback: int = 60,
        entry_z: float = 2.0,
        exit_z: float = 0.5,
    ) -> None:
        self.symbol_a = symbol_a
        self.symbol_b = symbol_b
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z

    def generate_signals(self, market_data: pd.DataFrame) -> SignalFrame:
        """Generate signals from a DataFrame with two price columns.

        Expects ``market_data`` to have at least ``self.symbol_a`` and
        ``self.symbol_b`` columns, or falls back to the first two numeric
        columns.
        """
        if self.symbol_a in market_data.columns and self.symbol_b in market_data.columns:
            a = market_data[self.symbol_a]
            b = market_data[self.symbol_b]
        else:
            # Fallback: first two numeric columns
            nums = market_data.select_dtypes(include="number")
            if nums.shape[1] < 2:
                raise ValueError("PairsTrading requires at least two price columns.")
            a, b = nums.iloc[:, 0], nums.iloc[:, 1]

        # Rolling hedge ratio (OLS slope)
        cov_ab = a.rolling(self.lookback).cov(b)
        var_b = b.rolling(self.lookback).var()
        hedge_ratio = (cov_ab / var_b.replace(0.0, float("nan"))).fillna(1.0)

        # Spread
        spread = a - hedge_ratio * b

        # Z‑score
        spread_mean = spread.rolling(self.lookback).mean()
        spread_std = spread.rolling(self.lookback).std().replace(0.0, float("nan"))
        z_score = ((spread - spread_mean) / spread_std).fillna(0.0)

        # Signals
        signal = pd.Series(0.0, index=market_data.index)
        signal[z_score < -self.entry_z] = 1.0   # Spread too low → buy spread
        signal[z_score > self.entry_z] = -1.0    # Spread too high → sell spread

        # Exit when z‑score reverts
        signal[(z_score.abs() < self.exit_z)] = 0.0

        # Forward‑fill positions between entry and exit
        signal = signal.replace(0.0, float("nan")).ffill().fillna(0.0)
        # But force flat when near zero z
        signal[(z_score.abs() < self.exit_z)] = 0.0

        # Position size inversely proportional to z‑score magnitude
        position_size = (1.0 / z_score.abs().clip(lower=0.5)).clip(upper=1.0).fillna(0.5)

        return pd.DataFrame(
            {"signal": signal, "position_size": position_size},
            index=market_data.index,
        )
