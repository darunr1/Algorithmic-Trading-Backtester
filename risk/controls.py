"""Risk management controls.

Applies stop‑loss, take‑profit, drawdown limits, position sizing rules,
and exposure caps to raw strategy signals.  This is what makes the system
feel like *real finance*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RiskConfig:
    """Risk management parameters."""

    # Per‑trade guards
    stop_loss_pct: Optional[float] = 0.05       # 5 %
    take_profit_pct: Optional[float] = 0.10     # 10 %

    # Portfolio‑level guards
    max_drawdown_pct: float = 0.20              # 20 %
    drawdown_risk_scaling: bool = True

    # Position sizing
    sizing_method: str = "fixed_pct"            # "fixed_pct" | "kelly"
    fixed_risk_pct: float = 0.02                # Risk 2 % of equity per trade
    kelly_fraction: float = 0.5                 # Half‑Kelly (safer)

    # Exposure limits
    max_position_pct: float = 1.0               # Max 100 % of equity in one asset
    max_leverage: float = 2.0                   # Overall leverage cap


class RiskManager:
    """Applies risk controls to raw strategy positions.

    Usage::

        rm = RiskManager(RiskConfig(stop_loss_pct=0.05))
        adjusted = rm.apply(positions, prices, equity_curve)
    """

    def __init__(self, config: Optional[RiskConfig] = None) -> None:
        self.cfg = config or RiskConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def apply(
        self,
        positions: pd.Series,
        prices: pd.Series,
        equity_curve: Optional[pd.Series] = None,
    ) -> pd.Series:
        """Apply all risk controls and return adjusted positions.

        Args:
            positions: Raw (unscaled) position sizes.
            prices: Close prices (same index).
            equity_curve: Running equity (if None, built from prices).

        Returns:
            Risk‑adjusted position series.
        """
        pos = positions.copy()

        # Build equity if not provided
        if equity_curve is None:
            daily_ret = prices.pct_change().fillna(0.0)
            equity_curve = (1 + daily_ret * pos.shift(1).fillna(0.0)).cumprod()

        # 1. Stop‑loss / take‑profit
        pos = self._apply_stop_take(pos, prices)

        # 2. Drawdown scaling
        if self.cfg.drawdown_risk_scaling:
            pos = self._apply_drawdown_scaling(pos, equity_curve)

        # 3. Position sizing
        pos = self._apply_sizing(pos, prices, equity_curve)

        # 4. Leverage cap
        pos = pos.clip(lower=-self.cfg.max_leverage, upper=self.cfg.max_leverage)

        return pos

    # ------------------------------------------------------------------
    # Stop‑loss / take‑profit
    # ------------------------------------------------------------------

    def _apply_stop_take(
        self, positions: pd.Series, prices: pd.Series
    ) -> pd.Series:
        if self.cfg.stop_loss_pct is None and self.cfg.take_profit_pct is None:
            return positions

        pos = positions.copy()
        entry_price = prices.iloc[0]
        in_trade = False

        for i in range(len(pos)):
            if abs(pos.iloc[i]) > 1e-9 and not in_trade:
                entry_price = prices.iloc[i]
                in_trade = True

            if in_trade and abs(pos.iloc[i]) > 1e-9:
                ret_since_entry = (prices.iloc[i] - entry_price) / entry_price

                if pos.iloc[i] > 0:  # Long
                    if self.cfg.stop_loss_pct and ret_since_entry < -self.cfg.stop_loss_pct:
                        pos.iloc[i] = 0.0
                        in_trade = False
                    elif self.cfg.take_profit_pct and ret_since_entry > self.cfg.take_profit_pct:
                        pos.iloc[i] = 0.0
                        in_trade = False
                else:  # Short
                    if self.cfg.stop_loss_pct and ret_since_entry > self.cfg.stop_loss_pct:
                        pos.iloc[i] = 0.0
                        in_trade = False
                    elif self.cfg.take_profit_pct and ret_since_entry < -self.cfg.take_profit_pct:
                        pos.iloc[i] = 0.0
                        in_trade = False
            else:
                if abs(pos.iloc[i]) < 1e-9:
                    in_trade = False

        return pos

    # ------------------------------------------------------------------
    # Drawdown scaling
    # ------------------------------------------------------------------

    def _apply_drawdown_scaling(
        self, positions: pd.Series, equity_curve: pd.Series
    ) -> pd.Series:
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve - rolling_max) / rolling_max.replace(0.0, float("nan"))

        # Scale linearly: full size at 0 DD, zero size at max_drawdown_pct
        scale = (1.0 - drawdown.abs() / self.cfg.max_drawdown_pct).clip(0.0, 1.0)
        return positions * scale

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def _apply_sizing(
        self, positions: pd.Series, prices: pd.Series, equity_curve: pd.Series
    ) -> pd.Series:
        if self.cfg.sizing_method == "kelly":
            return self._kelly_sizing(positions, prices)
        else:
            # Fixed percentage of equity
            return positions * self.cfg.fixed_risk_pct / 0.02  # normalised to 2 %

    def _kelly_sizing(
        self, positions: pd.Series, prices: pd.Series
    ) -> pd.Series:
        """Half‑Kelly criterion sizing.

        Kelly fraction = (win_rate * avg_win - (1‑win_rate) * avg_loss) / avg_win
        """
        returns = prices.pct_change().fillna(0.0)
        wins = returns[returns > 0]
        losses = returns[returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return positions

        win_rate = len(wins) / (len(wins) + len(losses))
        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_win == 0:
            return positions

        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0.0, min(kelly * self.cfg.kelly_fraction, self.cfg.max_leverage))

        return positions * kelly
