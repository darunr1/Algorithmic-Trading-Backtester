"""Sector-based analysis: strategy metrics, rankings, and invest reasoning."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.sectors import Sector, SECTORS, get_sector
from src.trading_bot import (
    StrategyConfig,
    calculate_performance,
    compute_strategy_returns,
)


@dataclass
class TickerAnalysis:
    """Analysis result for a single ticker."""
    symbol: str
    sector_id: str
    is_etf: bool
    current_price: float
    fast_ema: float
    slow_ema: float
    trend_signal: str  # "bullish" | "bearish"
    position_size: float
    annual_return: float
    sharpe_ratio: float
    annual_volatility: float
    max_drawdown: float
    n_observations: int
    reasoning: str
    investment_thesis: str
    is_good: bool
    event_score: float
    event_headlines: List[str] = field(default_factory=list)
    event_summary: str = ""
    rank_score: float = 0.0
    raw_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SectorRecommendation:
    """Sector-level recommendation with ticker-level breakdown."""

    sector: Sector
    etf_analysis: Optional[TickerAnalysis] = None
    stock_analyses: List[TickerAnalysis] = field(default_factory=list)
    sector_score: float = 0.0
    reasoning: str = ""


def _fetch_prices(symbol: str, lookback_days: int = 504) -> Optional[pd.Series]:
    """Fetch daily close prices from Yahoo Finance. Returns None on failure."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    end = datetime.now()
    start = end - timedelta(days=lookback_days)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start, end=end, interval="1d")
    if df is None or df.empty or len(df) < 60:
        return None
    return df["Close"].dropna()

def _fetch_recent_headlines(
    symbol: str,
    lookback_days: int = 14,
    max_items: int = 5,
) -> List[str]:
    """Fetch recent headlines for a symbol from Yahoo Finance."""
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    ticker = yf.Ticker(symbol)
    news_items = getattr(ticker, "news", None)
    if not news_items:
        return []
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    headlines: List[str] = []
    for item in news_items:
        if len(headlines) >= max_items:
            break
        title = item.get("title")
        published = item.get("providerPublishTime")
        if not title:
            continue
        if published:
            published_dt = datetime.fromtimestamp(published, tz=timezone.utc)
            if published_dt < cutoff:
                continue
        headlines.append(str(title))
    return headlines


def _score_event_headlines(headlines: List[str]) -> float:
    """Score headlines using simple keyword polarity."""
    if not headlines:
        return 0.0
    positives = {
        "beats",
        "beat",
        "surge",
        "record",
        "raises",
        "raise",
        "upgrade",
        "upgraded",
        "win",
        "wins",
        "contract",
        "approval",
        "approves",
        "launch",
        "partnership",
        "acquire",
        "acquisition",
        "guidance raise",
        "buyback",
    }
    negatives = {
        "miss",
        "misses",
        "downgrade",
        "downgraded",
        "lawsuit",
        "recall",
        "probe",
        "decline",
        "cuts",
        "cut",
        "warning",
        "fraud",
    }
    score = 0
    for headline in headlines:
        text = headline.lower()
        pos_hits = sum(1 for word in positives if word in text)
        neg_hits = sum(1 for word in negatives if word in text)
        score += pos_hits - neg_hits
    return score / max(len(headlines), 1)


def _summarize_events(headlines: List[str], score: float) -> str:
    if not headlines:
        return "No recent headlines available."
    label = "positive" if score > 0 else "mixed" if score == 0 else "negative"
    headline_list = "; ".join(headlines[:3])
    return f"Recent headlines are {label}: {headline_list}."


def _is_good_ticker(
    trend: str,
    sharpe: float,
    ann_ret: float,
    max_drawdown: float,
    event_score: float,
    has_headlines: bool,
) -> bool:
    if trend != "bullish":
        return False
    if sharpe < 0.5 or ann_ret <= 0:
        return False
    if max_drawdown < -0.35:
        return False
    if has_headlines and event_score < 0:
        return False
    return True


def _analyze_ticker(
    symbol: str,
    sector_id: str,
    is_etf: bool,
    config: StrategyConfig,
    prices: pd.Series,
) -> TickerAnalysis:
    """Run strategy on prices, compute metrics, build reasoning."""
    prices = prices.ffill().dropna()
    if len(prices) < config.slow_ema_span + config.vol_lookback:
        return TickerAnalysis(
            symbol=symbol,
            sector_id=sector_id,
            is_etf=is_etf,
            current_price=float(prices.iloc[-1]),
            fast_ema=0.0,
            slow_ema=0.0,
            trend_signal="unknown",
            position_size=0.0,
            annual_return=0.0,
            sharpe_ratio=0.0,
            annual_volatility=0.0,
            max_drawdown=0.0,
            n_observations=len(prices),
            reasoning="Insufficient data for analysis.",
            investment_thesis="Not enough data to form a reliable investment thesis.",
            is_good=False,
            event_score=0.0,
            event_summary="No recent headlines available.",
            rank_score=0.0,
        )

    fast_ema = prices.ewm(span=config.fast_ema_span, adjust=False).mean()
    slow_ema = prices.ewm(span=config.slow_ema_span, adjust=False).mean()
    daily_returns = prices.pct_change().fillna(0.0)
    rolling_vol = daily_returns.rolling(config.vol_lookback).std().replace(0.0, np.nan)
    vol_target = config.target_vol / (rolling_vol * np.sqrt(252))
    position_size_series = vol_target.clip(upper=config.max_leverage).fillna(0.0)

    fe = float(fast_ema.iloc[-1])
    se = float(slow_ema.iloc[-1])
    trend = "bullish" if fe > se else "bearish"
    ps_last = position_size_series.dropna()
    ps_val = float(ps_last.iloc[-1]) if len(ps_last) else 0.0
    ps_val = min(max(ps_val, 0.0), config.max_leverage)
    pos = (1.0 if fe > se else 0.0) * ps_val

    strat_returns = compute_strategy_returns(prices, config)
    perf = calculate_performance(strat_returns)

    ann_ret = perf.annual_return
    sharpe = perf.sharpe_ratio
    ann_vol = perf.annual_volatility
    mdd = perf.max_drawdown

    try:
        headlines = _fetch_recent_headlines(symbol)
    except Exception:
        headlines = []
    event_score = _score_event_headlines(headlines)
    event_summary = _summarize_events(headlines, event_score)
    is_good = _is_good_ticker(
        trend=trend,
        sharpe=sharpe,
        ann_ret=ann_ret,
        max_drawdown=mdd,
        event_score=event_score,
        has_headlines=bool(headlines),
    )
    rank_score = _market_rank_score(
        annual_return=ann_ret,
        sharpe=sharpe,
        max_drawdown=mdd,
        event_score=event_score,
        trend=trend,
    )
    investment_thesis = _build_investment_thesis(
        symbol=symbol,
        trend=trend,
        fast_ema=fe,
        slow_ema=se,
        annual_return=ann_ret,
        sharpe=sharpe,
        annual_volatility=ann_vol,
        max_drawdown=mdd,
        position_size=pos,
        target_vol=config.target_vol,
        n_days=len(prices),
        event_summary=event_summary,
        rank_score=rank_score,
    )
    # Mathematical reasoning
    reason_parts = [
        f"**Price & trend:** Latest close = ${prices.iloc[-1]:.2f}. "
        f"Fast EMA ({config.fast_ema_span}d) = {fe:.2f}, Slow EMA ({config.slow_ema_span}d) = {se:.2f}. "
        f"Trend is **{trend}** (fast {'>' if fe > se else '<'} slow).",
        f"**Volatility targeting:** Realized vol (annualized) = {ann_vol:.1%}. "
        f"Target vol = {config.target_vol:.0%}. Position size scaled to {pos:.2f} (capped at {config.max_leverage}x).",
        f"**Backtest (≈{len(prices)} days):** Annual return = {ann_ret:.1%}, "
        f"Sharpe = {sharpe:.2f}, Max drawdown = {mdd:.1%}. "
        f"Transaction costs = {config.transaction_cost_bps} bps.",
        f"**Real-world catalysts:** {event_summary}",
        f"**Investment thesis (paragraph):** {investment_thesis}",
    ]
    if is_good:
        reason_parts.append(
            "**Recommendation:** Good-stock filter passed (bullish trend, positive return, "
            "Sharpe ≥ 0.5, drawdown within limits, and no negative news skew). "
            f"Consider exposure if trend remains {trend}."
        )
    elif trend == "bearish":
        reason_parts.append(
            "**Recommendation:** No long allocation; trend is bearish. "
            "Strategy waits for fast EMA > slow EMA before going long."
        )
    else:
        reason_parts.append(
            "**Recommendation:** Mixed signals. Evaluate alongside other sectors and risk constraints."
        )

    reasoning = " ".join(reason_parts)

    return TickerAnalysis(
        symbol=symbol,
        sector_id=sector_id,
        is_etf=is_etf,
        current_price=float(prices.iloc[-1]),
        fast_ema=fe,
        slow_ema=se,
        trend_signal=trend,
        position_size=pos,
        annual_return=ann_ret,
        sharpe_ratio=sharpe,
        annual_volatility=ann_vol,
        max_drawdown=mdd,
        n_observations=len(prices),
        reasoning=reasoning,
        investment_thesis=investment_thesis,
        is_good=is_good,
        event_score=event_score,
        event_headlines=headlines,
        event_summary=event_summary,
        rank_score=rank_score,
        raw_metrics={
            "annual_return": ann_ret,
            "sharpe_ratio": sharpe,
            "annual_volatility": ann_vol,
            "max_drawdown": mdd,
            "event_score": event_score,
            "rank_score": rank_score,
        },
    )


def _market_rank_score(
    annual_return: float,
    sharpe: float,
    max_drawdown: float,
    event_score: float,
    trend: str,
) -> float:
    """Composite score for ranking across the full market universe."""
    trend_bonus = 0.25 if trend == "bullish" else -0.25
    return (
        (annual_return * 3.0)
        + (sharpe * 1.5)
        + (event_score * 0.5)
        + (max_drawdown * 0.5)
        + trend_bonus
    )


def _build_investment_thesis(
    symbol: str,
    trend: str,
    fast_ema: float,
    slow_ema: float,
    annual_return: float,
    sharpe: float,
    annual_volatility: float,
    max_drawdown: float,
    position_size: float,
    target_vol: float,
    n_days: int,
    event_summary: str,
    rank_score: float,
) -> str:
    """Create a narrative paragraph that blends math with real-world events."""
    trend_phrase = "uptrend" if trend == "bullish" else "downtrend"
    return (
        f"{symbol} is currently in a {trend_phrase}, with a fast EMA ({fast_ema:.2f}) "
        f"{'above' if trend == 'bullish' else 'below'} the slow EMA ({slow_ema:.2f}). "
        f"Over roughly {n_days} trading days, the strategy produces an annualized return of "
        f"{annual_return:.1%} with a Sharpe ratio of {sharpe:.2f}, implying "
        f"{annual_volatility:.1%} annualized volatility and a max drawdown of {max_drawdown:.1%}. "
        f"The volatility-targeting model suggests a {position_size:.2f}x position size versus a "
        f"{target_vol:.0%} risk target, indicating how much exposure the math supports. "
        f"Current events: {event_summary} This real-world backdrop, combined with a composite "
        f"market score of {rank_score:.2f}, supports the case for investment if the trend persists."
    )


def analyze_sector(
    sector: Sector,
    config: Optional[StrategyConfig] = None,
    lookback_days: int = 504,
) -> SectorRecommendation:
    """Analyze sector ETF and representative stocks; produce recommendation."""
    config = config or StrategyConfig()
    rec = SectorRecommendation(sector=sector)

    # ETF
    etf_prices = _fetch_prices(sector.etf, lookback_days)
    if etf_prices is not None:
        etf_analysis = _analyze_ticker(sector.etf, sector.id, True, config, etf_prices)
        rec.etf_analysis = etf_analysis if etf_analysis.is_good else None

    # Stocks
    for sym in sector.stocks:
        prices = _fetch_prices(sym, lookback_days)
        if prices is not None:
            analysis = _analyze_ticker(sym, sector.id, False, config, prices)
            if analysis.is_good:
                rec.stock_analyses.append(analysis)

    # Sector score: avg Sharpe of good tickers
    sharpes: List[float] = []
    if rec.etf_analysis:
        sharpes.append(rec.etf_analysis.sharpe_ratio)
    sharpes.extend(a.sharpe_ratio for a in rec.stock_analyses)
    rec.sector_score = float(np.mean(sharpes)) if sharpes else 0.0

    # Sector-level reasoning
    n_good = len(rec.stock_analyses) + (1 if rec.etf_analysis else 0)
    n_total = len(sector.stocks) + 1
    rec.reasoning = (
        f"**{sector.name}** ({sector.etf}): {n_good}/{n_total} tickers pass the "
        "good-stock filter (bullish + positive return + Sharpe ≥ 0.5 + "
        "drawdown limits + non-negative news skew). "
        f"Sector score (avg Sharpe, good only) = {rec.sector_score:.2f}. "
        f"{sector.description}"
    )

    return rec


def analyze_all_sectors(
    config: Optional[StrategyConfig] = None,
    lookback_days: int = 504,
) -> List[SectorRecommendation]:
    """Analyze all sectors and return sorted by sector score (desc)."""
    results = []
    for s in SECTORS:
        try:
            rec = analyze_sector(s, config=config, lookback_days=lookback_days)
            results.append(rec)
        except Exception:
            continue
    results.sort(key=lambda r: r.sector_score, reverse=True)
    return results


def analyze_market_top_stocks(
    config: Optional[StrategyConfig] = None,
    lookback_days: int = 504,
    top_n: int = 10,
) -> List[TickerAnalysis]:
    """Analyze the full stock universe and return top-ranked stocks."""
    config = config or StrategyConfig()
    from src.sectors import get_all_stocks, get_sector_for_symbol

    analyses: List[TickerAnalysis] = []
    for symbol in get_all_stocks():
        prices = _fetch_prices(symbol, lookback_days)
        if prices is None:
            continue
        sector = get_sector_for_symbol(symbol)
        sector_id = sector.id if sector else "market"
        analysis = _analyze_ticker(symbol, sector_id, False, config, prices)
        analyses.append(analysis)

    analyses.sort(key=lambda a: a.rank_score, reverse=True)
    return analyses[:top_n]


def analyze_ticker(
    symbol: str,
    sector_id: Optional[str] = None,
    config: Optional[StrategyConfig] = None,
    lookback_days: int = 504,
) -> Optional[TickerAnalysis]:
    """Analyze a single ticker. sector_id used only for labeling."""
    config = config or StrategyConfig()
    prices = _fetch_prices(symbol, lookback_days)
    if prices is None:
        return None
    sector_id = sector_id or "unknown"
    return _analyze_ticker(symbol, sector_id, False, config, prices)
