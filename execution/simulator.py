"""Execution simulator with pluggable slippage models and fee structures.

This is where *most people cut corners*. A proper simulator accounts for
the fact that markets are NOT frictionless: there are transaction fees,
bid‑ask spreads, and volume‑dependent price impact.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ============================================================================
# Slippage models
# ============================================================================

class SlippageModel(ABC):
    """Abstract slippage model."""

    @abstractmethod
    def compute(
        self,
        price: pd.Series,
        trade_size: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Return the slippage cost per unit traded (positive = cost)."""


class FixedSlippage(SlippageModel):
    """Fixed percentage slippage on every trade.

    Args:
        bps: Slippage in basis points (e.g. 5.0 = 0.05 %).
    """

    def __init__(self, bps: float = 5.0) -> None:
        self.fraction = bps / 10_000

    def compute(
        self,
        price: pd.Series,
        trade_size: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        return price * self.fraction


class VolumeSlippage(SlippageModel):
    """Slippage scales with trade size relative to daily volume.

    Implements a square‑root market impact model::

        impact = eta * sigma * sqrt(trade_qty / daily_volume)

    Args:
        eta: Market‑impact coefficient (default 0.1).
    """

    def __init__(self, eta: float = 0.1) -> None:
        self.eta = eta

    def compute(
        self,
        price: pd.Series,
        trade_size: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        if volume is None or volume.sum() == 0:
            return price * 0.0005  # fallback to 0.5 bps

        daily_vol = price.pct_change().rolling(20).std().fillna(0.001)
        participation = (trade_size.abs() / volume.replace(0, float("nan"))).fillna(0.0)
        impact = self.eta * daily_vol * np.sqrt(participation.clip(lower=0))
        return price * impact


class SpreadSlippage(SlippageModel):
    """Bid‑ask spread slippage.

    Assumes the effective spread is a fixed factor of price.

    Args:
        half_spread_bps: Half the bid‑ask spread in basis points.
    """

    def __init__(self, half_spread_bps: float = 2.5) -> None:
        self.half_spread = half_spread_bps / 10_000

    def compute(
        self,
        price: pd.Series,
        trade_size: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.Series:
        return price * self.half_spread


# ============================================================================
# Fee model
# ============================================================================

@dataclass
class FeeModel:
    """Transaction fee structure.

    Args:
        per_trade_flat: Flat fee per trade (e.g. $1.00).
        per_share: Fee per share (e.g. $0.005).
        percentage_bps: Percentage fee in basis points.
    """

    per_trade_flat: float = 0.0
    per_share: float = 0.0
    percentage_bps: float = 1.0

    def compute(
        self,
        price: pd.Series,
        quantity: pd.Series,
    ) -> pd.Series:
        """Compute total fees for each bar."""
        trades_occurred = (quantity.abs() > 1e-9).astype(float)
        flat = self.per_trade_flat * trades_occurred
        per_share = self.per_share * quantity.abs()
        percentage = price * quantity.abs() * (self.percentage_bps / 10_000)
        return flat + per_share + percentage


# ============================================================================
# Execution Simulator
# ============================================================================

class ExecutionSimulator:
    """Simulates realistic trade execution.

    Combines a slippage model, fee structure, and optional latency to
    compute the *effective* return after friction.
    """

    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        fee_model: Optional[FeeModel] = None,
        latency_bars: int = 0,
    ) -> None:
        self.slippage = slippage_model or FixedSlippage(bps=5.0)
        self.fees = fee_model or FeeModel()
        self.latency_bars = latency_bars

    def apply(
        self,
        positions: pd.Series,
        prices: pd.Series,
        volume: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """Apply execution friction to raw positions.

        Args:
            positions: Target position sizes over time.
            prices: Close prices.
            volume: Daily volume (optional, used by VolumeSlippage).

        Returns:
            DataFrame with columns:

            * ``position`` — effective position after latency
            * ``trade_size`` — units traded each bar
            * ``slippage_cost`` — slippage deducted
            * ``fee_cost`` — fees deducted
            * ``total_cost`` — slippage + fees
        """
        # Latency delay
        effective_pos = positions.shift(self.latency_bars).fillna(0.0)

        # Trade sizes (change in position)
        trade_size = effective_pos.diff().fillna(0.0)

        # Slippage
        slippage_cost = self.slippage.compute(prices, trade_size, volume) * trade_size.abs()

        # Fees
        fee_cost = self.fees.compute(prices, trade_size)

        total_cost = slippage_cost + fee_cost

        return pd.DataFrame(
            {
                "position": effective_pos,
                "trade_size": trade_size,
                "slippage_cost": slippage_cost,
                "fee_cost": fee_cost,
                "total_cost": total_cost,
            },
            index=positions.index,
        )
