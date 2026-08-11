from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.ai_research import build_research_prompt, generate_research
from core.analytics import backtest_sma_strategy, market_snapshot, position_size
from core.market_data import MarketBundle, fetch_market_bundle, fetch_news, validate_ticker

try:
    from groq import Groq
except Exception:
    Groq = None


PERIOD_OPTIONS = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "10 Years": "10y",
}


st.set_page_config(
    page_title="Axiom Market Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
:root { --panel:#111827; --border:#253047; --muted:#8b98ad; --accent:#79a7ff; }
[data-testid="stAppViewContainer"] { background: #080d16; color: #eef3fb; }
[data-testid="stSidebar"] { background: #0b111d; border-right: 1px solid #1d2738; }
.block-container { max-width: 1500px; padding-top: 1.5rem; padding-bottom: 3rem; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3 { letter-spacing: -0.025em; }
.ax-kicker { color:#7f8da3; font-size:.72rem; letter-spacing:.16em; text-transform:uppercase; margin-bottom:.35rem; }
.ax-title { font-size:2.15rem; font-weight:760; letter-spacing:-.04em; margin:0; }
.ax-sub { color:#8b98ad; margin-top:.45rem; }
.ax-badge { display:inline-block; border:1px solid #2b3b56; background:#101a2b; border-radius:999px; padding:.2rem .55rem; font-size:.74rem; color:#a9bad2; margin-right:.35rem; }
[data-testid="stMetric"] { background:linear-gradient(180deg,#111827,#0d1523); border:1px solid #253047; border-radius:14px; padding:1rem; }
[data-testid="stMetricLabel"] { color:#8b98ad; }
[data-testid="stMetricValue"] { font-size:1.55rem; }
.stTabs [data-baseweb="tab-list"] { gap:.35rem; background:#0a101b; border-bottom:1px solid #202c40; }
.stTabs [data-baseweb="tab"] { border-radius:8px 8px 0 0; padding:.6rem .9rem; color:#91a0b7; }
.stTabs [aria-selected="true"] { color:#fff !important; background:#111827 !important; }
div[data-testid="stDataFrame"] { border:1px solid #253047; border-radius:12px; overflow:hidden; }
.stButton > button, .stDownloadButton > button { border-radius:9px; border:1px solid #30415e; background:#101a2b; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#6e9ef5; color:#fff; }
.ax-note { color:#92a0b5; font-size:.84rem; line-height:1.55; }
.ax-section { border-top:1px solid #202c40; margin-top:1rem; padding-top:1rem; }
</style>
""",
    unsafe_allow_html=True,
)


def fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def fmt_num(value, digits: int = 2, prefix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{prefix}{float(value):,.{digits}f}"
    except Exception:
        return "N/A"


def optional_access_gate() -> None:
    try:
        allowed = list(st.secrets.get("allowed_users", []))
    except Exception:
        allowed = []
    if not allowed:
        return
    st.sidebar.markdown("### Access")
    email = st.sidebar.text_input("Email", value=st.session_state.get("user_email", ""))
    if not email:
        st.sidebar.info("Enter an approved email to continue.")
        st.stop()
    if email.strip().lower() not in {u.strip().lower() for u in allowed}:
        st.sidebar.error("This email is not on the access list.")
        st.stop()
    st.session_state.user_email = email


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_bundle(symbol: str, period: str, demo_mode: bool) -> MarketBundle:
    return fetch_market_bundle(symbol, period=period, demo_mode=demo_mode)


@st.cache_data(ttl=1200, show_spinner=False)
def cached_news(symbol: str):
    return fetch_news(symbol, max_items=8)


def price_chart(hist: pd.DataFrame, symbol: str, show_sma20: bool, show_sma50: bool, show_sma200: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"], high=hist["High"], low=hist["Low"], close=hist["Close"],
            name=symbol,
            increasing_line_color="#4bd49a",
            decreasing_line_color="#ff6b7a",
        )
    )
    overlays = [
        (show_sma20, "SMA_20", "SMA 20", "#79a7ff"),
        (show_sma50, "SMA_50", "SMA 50", "#eab464"),
        (show_sma200, "SMA_200", "SMA 200", "#b58cff"),
    ]
    for enabled, column, name, color in overlays:
        if enabled and column in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[column], name=name, line=dict(width=1.6, color=color)))
    fig.update_layout(
        height=610,
        template="plotly_dark",
        paper_bgcolor="#0b111d",
        plot_bgcolor="#0b111d",
        margin=dict(l=12, r=12, t=20, b=15),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.04, x=0),
        xaxis=dict(gridcolor="#1c2638"),
        yaxis=dict(gridcolor="#1c2638", side="right"),
    )
    return fig


def oscillator_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI_14"], name="RSI 14", line=dict(width=1.8, color="#79a7ff")))
    fig.add_hline(y=70, line_dash="dot", line_color="#ff6b7a")
    fig.add_hline(y=50, line_dash="dot", line_color="#4a5568")
    fig.add_hline(y=30, line_dash="dot", line_color="#4bd49a")
    fig.update_layout(
        height=260, template="plotly_dark", paper_bgcolor="#0b111d", plot_bgcolor="#0b111d",
        margin=dict(l=12, r=12, t=15, b=15), yaxis=dict(range=[0, 100], side="right", gridcolor="#1c2638"),
        xaxis=dict(gridcolor="#1c2638"), showlegend=False,
    )
    return fig


def equity_chart(result) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="SMA strategy", line=dict(width=2.2)))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark, name="Buy & hold", line=dict(width=1.5, dash="dot")))
    fig.update_layout(
        height=430, template="plotly_dark", paper_bgcolor="#0b111d", plot_bgcolor="#0b111d",
        margin=dict(l=10, r=10, t=20, b=10), hovermode="x unified",
        yaxis_title="Growth of $1", yaxis=dict(side="right", gridcolor="#1c2638"), xaxis=dict(gridcolor="#1c2638"),
    )
    return fig


def main() -> None:
    optional_access_gate()

    st.sidebar.markdown("## AXIOM")
    st.sidebar.caption("Market intelligence terminal")
    symbol = st.sidebar.text_input("Symbol", value="AAPL").strip().upper()
    valid, validation_message = validate_ticker(symbol)
    if not valid:
        st.sidebar.error(validation_message)
        st.stop()

    period_label = st.sidebar.selectbox("History", list(PERIOD_OPTIONS), index=3)
    period = PERIOD_OPTIONS[period_label]
    demo_mode = st.sidebar.toggle("Demo data", value=False, help="Uses deterministic synthetic data for interface testing.")

    st.sidebar.markdown("### Chart overlays")
    show_sma20 = st.sidebar.checkbox("SMA 20", True)
    show_sma50 = st.sidebar.checkbox("SMA 50", True)
    show_sma200 = st.sidebar.checkbox("SMA 200", False)

    if st.sidebar.button("Refresh market data", use_container_width=True):
        cached_market_bundle.clear()
        cached_news.clear()
        st.rerun()

    try:
        with st.spinner(f"Loading {symbol} market data..."):
            bundle = cached_market_bundle(symbol, period, demo_mode)
    except Exception as exc:
        st.error(f"Unable to load {symbol}: {exc}")
        st.info("Try another symbol or enable Demo data to test the dashboard without external market APIs.")
        st.stop()

    hist = bundle.history
    snapshot = market_snapshot(hist)
    info = bundle.info
    company_name = info.get("longName") or symbol

    st.markdown('<div class="ax-kicker">AI TRADING INSIGHT DASHBOARD · V2</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="ax-title">{company_name} <span style="color:#7f8da3;font-weight:520">{symbol}</span></div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ax-sub"><span class="ax-badge">{bundle.source}</span>'
        f'<span class="ax-badge">{period_label}</span>'
        f'<span class="ax-badge">Last bar {hist.index[-1].strftime("%d %b %Y")}</span></div>',
        unsafe_allow_html=True,
    )
    if bundle.warning:
        st.warning(bundle.warning)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", fmt_num(snapshot.get("price"), 2, "$"), fmt_pct(snapshot.get("daily_return")))
    m2.metric("1M return", fmt_pct(snapshot.get("return_1m")))
    m3.metric("RSI 14", fmt_num(snapshot.get("rsi_14"), 1))
    m4.metric("60D vol", fmt_pct(snapshot.get("annualized_volatility")))
    m5.metric("Max drawdown", fmt_pct(snapshot.get("max_drawdown")))
    m6.metric("Trend regime", snapshot.get("regime", "N/A"), f"Score {snapshot.get('trend_score', 0):+d}")

    tabs = st.tabs(["Overview", "Price & Technicals", "Signal Lab", "Backtest", "AI Research", "News", "Data"])

    with tabs[0]:
        left, right = st.columns([1.55, 1])
        with left:
            st.markdown("### Market state")
            st.plotly_chart(price_chart(hist.tail(min(len(hist), 300)), symbol, True, True, False), use_container_width=True)
        with right:
            st.markdown("### Deterministic trend model")
            st.metric("Composite score", f"{snapshot['trend_score']:+d} / 100", f"{snapshot['confidence']}% directional strength")
            factor_df = pd.DataFrame(snapshot["factors"])
            st.dataframe(factor_df, hide_index=True, use_container_width=True)
            st.caption("The score is rule-based. AI does not determine the regime classification.")

        s1, s2, s3, s4 = st.columns(4)
        s1.metric("20D support", fmt_num(snapshot.get("support_20"), 2, "$"))
        s2.metric("20D resistance", fmt_num(snapshot.get("resistance_20"), 2, "$"))
        s3.metric("ATR 14", fmt_num(snapshot.get("atr_14"), 2, "$"), fmt_pct(snapshot.get("atr_pct")))
        s4.metric("3M return", fmt_pct(snapshot.get("return_3m")))

    with tabs[1]:
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            show_sma20_tab = st.checkbox("20-day", value=show_sma20, key="tab_sma20")
        with c2:
            show_sma50_tab = st.checkbox("50-day", value=show_sma50, key="tab_sma50")
        with c3:
            show_sma200_tab = st.checkbox("200-day", value=show_sma200, key="tab_sma200")
        st.plotly_chart(price_chart(hist, symbol, show_sma20_tab, show_sma50_tab, show_sma200_tab), use_container_width=True)
        st.markdown("#### Momentum")
        st.plotly_chart(oscillator_chart(hist), use_container_width=True)

    with tabs[2]:
        st.markdown("### Signal decomposition")
        st.caption("A transparent technical state model. This is research tooling, not an execution instruction.")
        signal_left, signal_right = st.columns([1.25, 1])
        with signal_left:
            factors = pd.DataFrame(snapshot["factors"])
            st.dataframe(factors, hide_index=True, use_container_width=True)
            st.markdown("#### Key levels")
            levels = pd.DataFrame([
                {"Level": "20D support", "Price": snapshot.get("support_20")},
                {"Level": "SMA 20", "Price": snapshot.get("sma_20")},
                {"Level": "SMA 50", "Price": snapshot.get("sma_50")},
                {"Level": "SMA 200", "Price": snapshot.get("sma_200")},
                {"Level": "20D resistance", "Price": snapshot.get("resistance_20")},
            ]).dropna()
            st.dataframe(levels, hide_index=True, use_container_width=True)
        with signal_right:
            st.markdown("#### Position risk calculator")
            account_size = st.number_input("Account size", min_value=100.0, value=10000.0, step=500.0)
            risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            entry = st.number_input("Reference entry", min_value=0.01, value=float(snapshot["price"]), step=0.01)
            default_stop = max(0.01, entry - 2 * float(snapshot.get("atr_14") or entry * 0.02))
            stop = st.number_input("Reference stop", min_value=0.01, value=float(default_stop), step=0.01)
            sizing = position_size(account_size, risk_pct, entry, stop)
            r1, r2 = st.columns(2)
            r1.metric("Risk amount", fmt_num(sizing["risk_amount"], 2, "$"))
            r2.metric("Units", fmt_num(sizing["units"], 2))
            r3, r4 = st.columns(2)
            r3.metric("Risk / unit", fmt_num(sizing["risk_per_unit"], 2, "$"))
            r4.metric("Notional", fmt_num(sizing["notional"], 2, "$"))

    with tabs[3]:
        st.markdown("### Reproducible SMA backtest")
        st.caption("Long-only rule: hold from the next session when Close > SMA, exit when the condition is false. Transaction costs are deducted on position changes.")
        b1, b2 = st.columns([1, 1])
        with b1:
            sma_window = st.selectbox("SMA length", [20, 50, 100, 200], index=1)
        with b2:
            transaction_cost_bps = st.number_input("Cost per position change (bps)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        try:
            bt = backtest_sma_strategy(hist, sma_window=sma_window, transaction_cost_bps=transaction_cost_bps)
            metrics = bt.metrics()
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("Strategy return", fmt_pct(metrics["total_return"]))
            k2.metric("Buy & hold", fmt_pct(metrics["buy_hold_return"]))
            k3.metric("Max drawdown", fmt_pct(metrics["max_drawdown"]))
            k4.metric("Sharpe", "N/A" if metrics["sharpe"] is None else f"{metrics['sharpe']:.2f}")
            k5.metric("Trades", str(metrics["trades"]))
            k6.metric("Exposure", fmt_pct(metrics["exposure"]))
            st.plotly_chart(equity_chart(bt), use_container_width=True)
            st.caption("Historical simulation only. No slippage model, taxes, borrowing costs, or intraday execution assumptions are included beyond the stated transaction cost.")
            st.session_state["latest_backtest_metrics"] = metrics
        except ValueError as exc:
            st.info(str(exc))

    with tabs[4]:
        st.markdown("### Grounded AI research")
        st.caption("The language model receives the deterministic technical snapshot, optional backtest metrics, and retrieved headlines. It is instructed not to invent data or issue a buy/sell command.")
        try:
            groq_key = st.secrets.get("GROQ_API_KEY")
        except Exception:
            groq_key = None
        if not groq_key or Groq is None:
            st.info("Add GROQ_API_KEY to Streamlit secrets to enable AI research. The rest of the dashboard works without it.")
        else:
            news_items = cached_news(symbol)
            if st.button("Generate research brief", type="primary"):
                prompt = build_research_prompt(
                    symbol=symbol,
                    company_name=company_name,
                    snapshot=snapshot,
                    backtest_metrics=st.session_state.get("latest_backtest_metrics"),
                    news=news_items,
                )
                with st.spinner("Generating grounded research brief..."):
                    data, raw = generate_research(Groq(api_key=groq_key), prompt)
                if data:
                    st.session_state[f"research_data_{symbol}"] = data
                    st.session_state[f"research_raw_{symbol}"] = raw
                else:
                    st.error("The AI service did not return a valid research brief.")
            data = st.session_state.get(f"research_data_{symbol}")
            if data:
                a1, a2, a3 = st.columns(3)
                a1.metric("Regime", data.get("market_regime", "N/A"))
                a2.metric("Conviction", data.get("conviction", "N/A"))
                a3.metric("Horizon", data.get("time_horizon", "N/A"))
                st.markdown("#### Thesis")
                st.write(data.get("thesis", ""))
                bull, bear = st.columns(2)
                with bull:
                    st.markdown("#### Bullish factors")
                    for item in data.get("bullish_factors", []):
                        st.markdown(f"• {item}")
                with bear:
                    st.markdown("#### Bearish factors")
                    for item in data.get("bearish_factors", []):
                        st.markdown(f"• {item}")
                st.markdown("#### Invalidation conditions")
                for item in data.get("invalidation_conditions", []):
                    st.markdown(f"• {item}")
                st.markdown("#### Levels to watch")
                for item in data.get("levels_to_watch", []):
                    st.markdown(f"• {item}")
                st.markdown("#### News context")
                st.write(data.get("news_context", ""))
                st.markdown("#### Data caveats")
                for item in data.get("data_caveats", []):
                    st.markdown(f"• {item}")
                st.info(data.get("bottom_line", ""))
                st.download_button(
                    "Download research JSON",
                    json.dumps(data, indent=2),
                    file_name=f"{symbol}_research_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                    mime="application/json",
                )

    with tabs[5]:
        st.markdown("### Latest market headlines")
        news_items = cached_news(symbol)
        if not news_items:
            st.info("No headlines were retrieved for this symbol.")
        for item in news_items:
            source = item.get("source") or "Google News"
            published = item.get("published") or ""
            title = item.get("title", "Untitled")
            link = item.get("link", "")
            if link:
                st.markdown(f"**[{title}]({link})**")
            else:
                st.markdown(f"**{title}**")
            st.caption(f"{source} · {published}".strip(" ·"))
            st.markdown("---")

    with tabs[6]:
        st.markdown("### Market metadata")
        fields = {
            "Company": company_name,
            "Symbol": symbol,
            "Source": bundle.source,
            "Exchange": info.get("exchange"),
            "Currency": info.get("currency"),
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Market cap": info.get("marketCap"),
            "52-week low": info.get("fiftyTwoWeekLow"),
            "52-week high": info.get("fiftyTwoWeekHigh"),
            "History requested": period_label,
            "Rows loaded": len(hist),
        }
        st.dataframe(pd.DataFrame(fields.items(), columns=["Field", "Value"]), hide_index=True, use_container_width=True)
        export = hist.copy()
        export.index.name = "Date"
        if getattr(export.index, "tz", None) is not None:
            export.index = export.index.tz_localize(None)
        st.download_button(
            "Download enriched OHLCV CSV",
            export.to_csv().encode("utf-8"),
            file_name=f"{symbol}_{period}_enriched.csv",
            mime="text/csv",
        )

    st.markdown('<div class="ax-section ax-note">Research and educational use only. Market data can be delayed, incomplete, or sourced from fallback providers. Backtests describe historical rule performance and do not guarantee future results.</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
