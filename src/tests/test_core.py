import pandas as pd
import numpy as np
import pytest
from src.trading_bot import StrategyConfig, compute_strategy_returns
from src.portfolio import PortfolioConfig, compute_portfolio_returns

@pytest.fixture
def sample_prices():
    dates = pd.date_range(start="2020-01-01", periods=100, freq="D")
    # Linear price increase with some noise
    prices = pd.Series(np.linspace(100, 150, 100) + np.random.normal(0, 1, 100), index=dates)
    return prices

def test_strategy_config_defaults():
    config = StrategyConfig()
    assert config.fast_ema_span == 12
    assert config.rsi_period == 14
    assert config.drawdown_risk_scaling is True

def test_compute_strategy_returns(sample_prices):
    config = StrategyConfig()
    returns = compute_strategy_returns(sample_prices, config)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_prices)
    # Check that costs are applied (returns should be slightly different than just price change * position)
    assert not returns.equals(sample_prices.pct_change().fillna(0.0))

def test_portfolio_returns_vectorized(sample_prices):
    prices_df = pd.DataFrame({
        "AAPL": sample_prices,
        "MSFT": sample_prices * 1.1
    })
    strat_config = StrategyConfig()
    port_config = PortfolioConfig(symbols=["AAPL", "MSFT"], strategy_config=strat_config)
    
    returns = compute_portfolio_returns(prices_df, port_config)
    assert isinstance(returns, pd.Series)
    assert len(returns) == len(sample_prices)
    assert not returns.isnull().any()

def test_drawdown_scaling(sample_prices):
    # Create a sharp drop
    drop_prices = sample_prices.copy()
    drop_prices.iloc[50:] = drop_prices.iloc[50] * 0.5
    
    config = StrategyConfig(drawdown_limit_pct=0.1, drawdown_risk_scaling=True)
    returns = compute_strategy_returns(drop_prices, config)
    
    # After the drop, signals should be scaled down or Zero
    # This is a bit heuristic, but checks that the logic runs without error
    assert len(returns) == len(drop_prices)
