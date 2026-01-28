"""Streamlit frontend: sector-based invest recommendations and reasoning."""

from __future__ import annotations

import streamlit as st

from src.sector_analysis import (
    SectorRecommendation,
    TickerAnalysis,
    analyze_all_sectors,
    analyze_market_top_stocks,
    analyze_sector,
    analyze_ticker,
)
from src.sectors import SECTORS, get_sector
from src.trading_bot import StrategyConfig

st.set_page_config(
    page_title="Sector & Stock Recommendations",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar
st.sidebar.title("📈 Sector & Stock Recommendations")
st.sidebar.markdown("Explore sectors and stocks with **math-backed reasoning**.")
st.sidebar.divider()

mode = st.sidebar.radio(
    "View",
    ["All Sectors", "Top 10 Stocks", "Single Sector", "Lookup Ticker"],
    index=0,
)

sector_options = ["(select)"] + [s.name for s in SECTORS]
lookback = st.sidebar.slider("Lookback (days)", 252, 756, 504, 63)
st.sidebar.caption("More days = longer backtest, fewer updates.")

run = st.sidebar.button("Run analysis", type="primary")

# Helper: render one ticker's metrics + reasoning
def _render_ticker(a: TickerAnalysis, label: str = "") -> None:
    tag = "ETF" if a.is_etf else "Stock"
    lbl = f"{a.symbol} ({tag})" if not label else label
    with st.expander(f"**{lbl}** · ${a.current_price:.2f} · {a.trend_signal.upper()}", expanded=False):
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Annual Return", f"{a.annual_return:.1%}")
        c2.metric("Sharpe Ratio", f"{a.sharpe_ratio:.2f}")
        c3.metric("Ann. Volatility", f"{a.annual_volatility:.1%}")
        c4.metric("Max Drawdown", f"{a.max_drawdown:.1%}")
        c5.metric("Position Size", f"{a.position_size:.2f}x")
        st.markdown("---")
        st.markdown("#### Why invest?")
        st.markdown(a.reasoning)
        st.markdown("#### Investment thesis (paragraph)")
        st.markdown(a.investment_thesis)
        if a.event_headlines:
            st.markdown("#### Current events (recent headlines)")
            st.markdown("\n".join([f"- {h}" for h in a.event_headlines]))

# Main
st.title("What to invest in — and why")
st.markdown(
    "Recommendations use a **volatility-targeted EMA trend** strategy. "
    "We show **actual data**, **math**, and **reasoning** for each sector and ticker."
)
st.divider()

if not run:
    st.info("👈 Pick a view, optionally adjust lookback, then click **Run analysis**.")
    st.stop()

config = StrategyConfig()

if mode == "All Sectors":
    st.subheader("All sectors (ranked by score)")
    with st.spinner("Fetching data and running analysis…"):
        recs = analyze_all_sectors(config=config, lookback_days=lookback)

    if not recs:
        st.warning("No sector data available. Check connectivity and try again.")
        st.stop()

    rows = []
    for r in recs:
        n_good = len(r.stock_analyses) + (1 if r.etf_analysis else 0)
        n_tot = len(r.sector.stocks) + 1
        rows.append({
            "Sector": r.sector.name,
            "ETF": r.sector.etf,
            "Score": round(r.sector_score, 2),
            "Good tickers": f"{n_good}/{n_tot}",
            "Reasoning": r.reasoning[:120] + "…" if len(r.reasoning) > 120 else r.reasoning,
        })

    st.dataframe(rows, use_container_width=True, hide_index=True)

    st.subheader("Drill down by sector")
    sel = st.selectbox("Select sector", sector_options)
    if sel != "(select)":
        sec = next((s for s in SECTORS if s.name == sel), None)
        if sec:
            with st.spinner(f"Analyzing {sec.name}…"):
                rec = analyze_sector(sec, config=config, lookback_days=lookback)
            st.markdown(f"### {rec.sector.name} ({rec.sector.etf})")
            st.markdown(rec.reasoning)
            st.divider()
            if rec.etf_analysis:
                _render_ticker(rec.etf_analysis, f"{rec.sector.etf} (ETF)")
            for a in rec.stock_analyses:
                _render_ticker(a)
            if not rec.stock_analyses and not rec.etf_analysis:
                st.warning("No tickers met the good-stock filter in this sector.")

elif mode == "Top 10 Stocks":
    st.subheader("Market-wide top 10 stocks")
    st.caption(
        "We scan the full universe of representative stocks across all sectors, "
        "rank them by a composite score (returns, Sharpe, drawdown, trend, and event signal), "
        "and show the top 10 with math + current-events reasoning."
    )
    with st.spinner("Analyzing the full market universe…"):
        top_stocks = analyze_market_top_stocks(config=config, lookback_days=lookback, top_n=10)

    if not top_stocks:
        st.warning("No market data available. Check connectivity and try again.")
        st.stop()

    table_rows = []
    for idx, a in enumerate(top_stocks, start=1):
        table_rows.append(
            {
                "#": idx,
                "Ticker": a.symbol,
                "Sector": a.sector_id,
                "Score": round(a.rank_score, 2),
                "Trend": a.trend_signal,
                "Ann Return": f"{a.annual_return:.1%}",
                "Sharpe": f"{a.sharpe_ratio:.2f}",
            }
        )
    st.dataframe(table_rows, use_container_width=True, hide_index=True)

    st.subheader("Top 10 breakdown")
    for a in top_stocks:
        _render_ticker(a, a.symbol)

elif mode == "Single Sector":
    sel = st.selectbox("Select sector", sector_options)
    if sel == "(select)":
        st.warning("Pick a sector.")
        st.stop()
    sec = next((s for s in SECTORS if s.name == sel), None)
    if not sec:
        st.error("Sector not found.")
        st.stop()

    with st.spinner(f"Analyzing {sec.name}…"):
        rec = analyze_sector(sec, config=config, lookback_days=lookback)

    st.subheader(f"{rec.sector.name}")
    st.caption(rec.sector.description)
    st.markdown(rec.reasoning)
    st.divider()

    st.markdown("#### ETF")
    if rec.etf_analysis:
        _render_ticker(rec.etf_analysis, f"{rec.sector.etf}")
    else:
        st.warning(f"{rec.sector.etf} did not meet the good-stock filter or data was unavailable.")

    st.markdown("#### Stocks")
    for a in rec.stock_analyses:
        _render_ticker(a)
    if not rec.stock_analyses and not rec.etf_analysis:
        st.warning("No tickers met the good-stock filter in this sector.")
else:
    # Lookup Ticker
    sym = st.text_input("Ticker symbol", value="AAPL", max_chars=10).strip().upper()
    if not sym:
        st.warning("Enter a ticker.")
        st.stop()

    with st.spinner(f"Analyzing {sym}…"):
        a = analyze_ticker(sym, config=config, lookback_days=lookback)

    if not a:
        st.error(f"No data or insufficient history for {sym}.")
        st.stop()

    st.subheader(sym)
    _render_ticker(a, sym)
