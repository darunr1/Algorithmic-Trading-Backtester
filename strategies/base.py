"""Abstract base class for all trading strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Type alias for signal output
# ---------------------------------------------------------------------------

SignalFrame = pd.DataFrame
"""DataFrame with at minimum a ``signal`` column (+1 / -1 / 0)
and optionally a ``position_size`` column (float, 0‒1 range)."""


# ---------------------------------------------------------------------------
# Generic config container
# ---------------------------------------------------------------------------

@dataclass
class StrategyConfig:
    """Generic key‑value configuration for any strategy.

    Strategies pull their own parameters from ``params``.
    Common parameters (vol targeting, cost model) live at top level
    so that the backtester can read them without knowing the strategy.
    """

    # -- Strategy‑specific params --
    params: Dict[str, Any] = field(default_factory=dict)

    # -- Shared config used by the backtester --
    vol_lookback: int = 20
    target_vol: float = 0.15
    max_leverage: float = 2.0
    transaction_cost_bps: float = 1.0


# ---------------------------------------------------------------------------
# Strategy ABC
# ---------------------------------------------------------------------------

class Strategy(ABC):
    """Abstract base class every strategy must implement.

    Subclasses only need to provide:

    * ``name`` — a human‑readable identifier.
    * ``generate_signals(market_data)`` — which returns a :pydata:`SignalFrame`.

    The backtester takes care of execution simulation, portfolio tracking,
    risk management, and performance measurement.
    """

    name: str = "base"

    @abstractmethod
    def generate_signals(self, market_data: pd.DataFrame) -> SignalFrame:
        """Produce trading signals from OHLCV market data.

        Args:
            market_data: DataFrame with ``[open, high, low, close, volume]``
                columns and a ``DatetimeIndex``.

        Returns:
            A :pydata:`SignalFrame` indexed identically to *market_data*
            with columns:

            * ``signal`` — directional signal (+1 long, −1 short, 0 flat).
            * ``position_size`` — suggested raw weight (before vol scaling).
        """

    # Convenience -----------------------------------------------------------

    def __repr__(self) -> str:
        return f"<Strategy: {self.name}>"
