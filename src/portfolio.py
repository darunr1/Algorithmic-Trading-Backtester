"""Multi-asset portfolio management with risk parity position sizing."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.trading_bot import StrategyConfig, compute_strategy_returns


@dataclass
class PortfolioConfig:
    """Configuration for multi-asset portfolio."""

    symbols: List[str]
    strategy_config: StrategyConfig
    target_portfolio_vol: float = 0.15
    rebalance_frequency: str = "D"  # 'D' for daily, 'W' for weekly, etc.
    risk_parity_method: str = "equal_risk"  # 'equal_risk' or 'inverse_vol'


def compute_risk_parity_weights(
    returns: pd.DataFrame,
    method: str = "equal_risk",
) -> pd.Series:
    """Compute risk parity weights for assets.

    Args:
        returns: DataFrame with asset returns (columns = symbols)
        method: 'equal_risk' or 'inverse_vol'

    Returns:
        Series with weights for each asset
    """
    if method == "inverse_vol":
        # Inverse volatility weighting
        volatilities = returns.std()
        inv_vol = 1.0 / volatilities.replace(0.0, np.inf)
        weights = inv_vol / inv_vol.sum()
    else:  # equal_risk
        # Equal risk contribution (simplified)
        # Each asset contributes equally to portfolio risk
        volatilities = returns.std()
        inv_vol = 1.0 / volatilities.replace(0.0, np.inf)
        # Normalize so sum of weights = 1
        weights = inv_vol / inv_vol.sum()

    return weights.fillna(0.0)


def compute_portfolio_returns(
    prices: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.Series:
    """Compute portfolio returns for multiple assets (Vectorized)."""
    # 1. Compute individual strategy returns for each asset
    asset_returns_dict = {}
    for symbol in config.symbols:
        if symbol not in prices.columns:
            continue
        asset_prices = prices[symbol].dropna()
        if len(asset_prices) == 0:
            continue
        asset_returns_dict[symbol] = compute_strategy_returns(
            asset_prices, config.strategy_config
        )

    if not asset_returns_dict:
        return pd.Series(dtype=float)

    returns_df = pd.DataFrame(asset_returns_dict).fillna(0.0)
    lookback = config.strategy_config.vol_lookback

    # 2. Compute dynamic weights (Risk Parity or Inverse Vol)
    if config.risk_parity_method == "inverse_vol":
        # Vectorized rolling volatility
        vols = returns_df.rolling(window=lookback).std()
        inv_vols = 1.0 / vols.replace(0.0, np.inf)
        weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)
    else:  # equal_risk (simplified to inverse vol here)
        vols = returns_df.rolling(window=lookback).std()
        inv_vols = 1.0 / vols.replace(0.0, np.inf)
        weights = inv_vols.div(inv_vols.sum(axis=1), axis=0)

    weights = weights.fillna(1.0 / len(returns_df.columns))

    # 3. Portfolio Volatility Targeting
    # Compute unscaled portfolio returns first to estimate portfolio vol
    unscaled_port_returns = (returns_df * weights).sum(axis=1)
    port_vol = unscaled_port_returns.rolling(window=lookback).std() * np.sqrt(252)
    
    vol_scale = (config.target_portfolio_vol / port_vol.replace(0.0, np.inf)).clip(
        upper=config.strategy_config.max_leverage
    )
    
    # Final scaled weights
    final_weights = weights.multiply(vol_scale, axis=0).fillna(0.0)

    # 4. Final Vectorized Portfolio Returns
    portfolio_returns = (returns_df * final_weights.shift(1)).sum(axis=1)

    return portfolio_returns


def run_portfolio_backtest(
    prices: pd.DataFrame,
    config: PortfolioConfig,
) -> pd.Series:
    """Run backtest for multi-asset portfolio.

    Args:
        prices: DataFrame with prices (columns = symbols)
        config: Portfolio configuration

    Returns:
        Series with portfolio returns
    """
    return compute_portfolio_returns(prices, config)
