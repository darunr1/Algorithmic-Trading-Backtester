"""Portfolio manager — tracks cash, positions, equity, and leverage over time."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    """Record of a single executed trade."""

    timestamp: pd.Timestamp
    side: str           # "buy" or "sell"
    quantity: float
    price: float
    cost: float         # total execution cost (slippage + fees)

    @property
    def notional(self) -> float:
        return abs(self.quantity) * self.price


class Portfolio:
    """Tracks portfolio state bar‑by‑bar during a backtest.

    The backtester feeds in positions and execution results each bar;
    the Portfolio records the full history of cash, equity, leverage,
    and individual trades.
    """

    def __init__(self, initial_cash: float = 100_000.0) -> None:
        self.initial_cash = initial_cash
        self._cash: float = initial_cash
        self._positions: Dict[str, float] = {}  # symbol → quantity

        # History (populated bar‑by‑bar)
        self._equity_history: List[float] = []
        self._cash_history: List[float] = []
        self._leverage_history: List[float] = []
        self._dates: List[pd.Timestamp] = []
        self._trades: List[Trade] = []

    # ------------------------------------------------------------------
    # Per‑bar update
    # ------------------------------------------------------------------

    def update(
        self,
        timestamp: pd.Timestamp,
        positions: pd.Series,
        prices: pd.Series,
        execution_costs: float = 0.0,
    ) -> None:
        """Update portfolio state for a single bar.

        Args:
            timestamp: Current bar timestamp.
            positions: Target position sizes (number of shares per symbol).
            prices: Current prices per symbol.
            execution_costs: Total cost of trades this bar.
        """
        self._cash -= execution_costs

        # Record trades for any position changes
        for symbol in positions.index:
            old_qty = self._positions.get(symbol, 0.0)
            new_qty = float(positions.get(symbol, 0.0))
            if abs(new_qty - old_qty) > 1e-9:
                trade_qty = new_qty - old_qty
                self._trades.append(
                    Trade(
                        timestamp=timestamp,
                        side="buy" if trade_qty > 0 else "sell",
                        quantity=trade_qty,
                        price=float(prices.get(symbol, 0.0)),
                        cost=execution_costs,
                    )
                )
            self._positions[symbol] = new_qty

        # Compute equity
        position_value = sum(
            qty * float(prices.get(sym, 0.0))
            for sym, qty in self._positions.items()
        )
        equity = self._cash + position_value
        gross_exposure = sum(
            abs(qty) * float(prices.get(sym, 0.0))
            for sym, qty in self._positions.items()
        )
        leverage = gross_exposure / equity if equity > 0 else 0.0

        self._dates.append(timestamp)
        self._equity_history.append(equity)
        self._cash_history.append(self._cash)
        self._leverage_history.append(leverage)

    # ------------------------------------------------------------------
    # Vectorized single‑asset update (for the common case)
    # ------------------------------------------------------------------

    @classmethod
    def from_backtest_vectors(
        cls,
        dates: pd.DatetimeIndex,
        positions: pd.Series,
        prices: pd.Series,
        daily_returns: pd.Series,
        total_costs: pd.Series,
        initial_cash: float = 100_000.0,
    ) -> "Portfolio":
        """Build portfolio history from vectorised backtest output.

        This is the fast path — avoids per‑bar Python loops for the
        single‑asset case.
        """
        port = cls(initial_cash=initial_cash)

        # Strategy returns already incorporate positions and costs
        equity_curve = initial_cash * (1 + daily_returns).cumprod()

        port._dates = list(dates)
        port._equity_history = equity_curve.tolist()
        port._cash_history = [initial_cash] * len(dates)  # Simplified
        port._leverage_history = positions.abs().tolist()

        # Record trades at position changes
        pos_diff = positions.diff().fillna(0.0)
        trade_mask = pos_diff.abs() > 1e-9
        for ts, qty, px, cost in zip(
            dates[trade_mask],
            pos_diff[trade_mask],
            prices[trade_mask],
            total_costs[trade_mask],
        ):
            port._trades.append(
                Trade(
                    timestamp=ts,
                    side="buy" if qty > 0 else "sell",
                    quantity=qty,
                    price=px,
                    cost=cost,
                )
            )

        return port

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def equity_curve(self) -> pd.Series:
        return pd.Series(self._equity_history, index=pd.DatetimeIndex(self._dates), name="equity")

    @property
    def cash_series(self) -> pd.Series:
        return pd.Series(self._cash_history, index=pd.DatetimeIndex(self._dates), name="cash")

    @property
    def leverage_series(self) -> pd.Series:
        return pd.Series(self._leverage_history, index=pd.DatetimeIndex(self._dates), name="leverage")

    @property
    def trades(self) -> List[Trade]:
        return self._trades

    @property
    def trade_count(self) -> int:
        return len(self._trades)

    @property
    def final_equity(self) -> float:
        return self._equity_history[-1] if self._equity_history else self.initial_cash
