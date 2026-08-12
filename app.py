from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

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

CURRENCY_SYMBOLS = {
    "USD": "$",
    "EUR": "€",
    "GBP": "£",
    "JPY": "¥",
    "NGN": "₦",
    "CAD": "C$",
    "AUD": "A$",
    "NZD": "NZ$",
    "HKD": "HK$",
    "SGD": "S$",
    "INR": "₹",
    "CNY": "¥",
    "KRW": "₩",
    "ZAR": "R",
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
:root { --bg:#070b12; --panel:#0d1420; --panel2:#101927; --border:#233047; --muted:#8492a8; --text:#edf3fb; --accent:#7ea6ff; }
[data-testid="stAppViewContainer"] { background: radial-gradient(circle at 70% -20%, #12203a 0%, #070b12 38%, #070b12 100%); color:var(--text); }
[data-testid="stSidebar"] { background:#090f19; border-right:1px solid #1c2738; }
.block-container { max-width:1520px; padding-top:1.15rem; padding-bottom:3rem; }
#MainMenu, footer { visibility:hidden; }
h1,h2,h3,h4 { letter-spacing:-.025em; }
.ax-kicker { color:#75849a; font-size:.69rem; letter-spacing:.18em; text-transform:uppercase; margin-bottom:.35rem; }
.ax-title { font-size:2.25rem; line-height:1.12; font-weight:760; letter-spacing:-.045em; margin:0; }
.ax-sub { color:#8b99ae; margin-top:.48rem; display:flex; gap:.42rem; flex-wrap:wrap; align-items:center; }
.ax-badge { display:inline-block; border:1px solid #293a55; background:#0d1727; border-radius:999px; padding:.24rem .58rem; font-size:.72rem; color:#a9b9d1; }
.ax-status { display:inline-flex; width:.45rem; height:.45rem; border-radius:50%; background:#55d6a0; margin-right:.25rem; }
[data-testid="stMetric"] { background:linear-gradient(180deg,#101927,#0b121e); border:1px solid #223149; border-radius:13px; padding:.92rem 1rem; min-height:108px; }
[data-testid="stMetricLabel"] { color:#8492a8; }
[data-testid="stMetricValue"] { font-size:1.48rem; }
.stTabs [data-baseweb="tab-list"] { gap:.2rem; background:transparent; border-bottom:1px solid #202d42; }
.stTabs [data-baseweb="tab"] { padding:.62rem .82rem; color:#8594aa; }
.stTabs [aria-selected="true"] { color:#fff !important; background:#0e1725 !important; border-radius:8px 8px 0 0; }
div[data-testid="stDataFrame"] { border:1px solid #223149; border-radius:11px; overflow:hidden; }
.stButton > button, .stDownloadButton > button { border-radius:9px; border:1px solid #304361; background:#0f1a2a; min-height:2.55rem; }
.stButton > button:hover, .stDownloadButton > button:hover { border-color:#719eff; color:#fff; }
.ax-note { color:#8492a8; font-size:.82rem; line-height:1.55; }
.ax-rule { border-top:1px solid #202d42; margin:1rem 0; }
.ax-mini { color:#7f8da3; font-size:.76rem; }
</style>
""",
    unsafe_allow_html=True,
)


def fmt_pct(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{value * 100:.{digits}f}%"


def fmt_num(value, digits: int = 2, prefix: str = "", suffix: str = "") -> str:
    try:
        if value is None or pd.isna(value):
            return "N/A"
        return f"{prefix}{float(value):,.{digits}f}{suffix}"
    except Exception:
        return "N/A"


def currency_prefix(currency: str | None) -> str:
    if not currency:
        return ""
    code = str(currency).upper()
    return CURRENCY_SYMBOLS.get(code, f"{code} ")


def price_digits(value: float | None) -> int:
    try:
        value = abs(float(value))
    except Exception:
        return 2
    if value < 1:
        return 5
    if value < 10:
        return 4
    return 2


def fmt_price(value: float | None, currency: str | None) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return fmt_num(value, price_digits(value), currency_prefix(currency))


def research_context_key(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:18]


@st.cache_data(ttl=900, show_spinner=False)
def cached_market_bundle(symbol: str, period: str, demo_mode: bool) -> MarketBundle:
    return fetch_market_bundle(symbol, period=period, demo_mode=demo_mode)


@st.cache_data(ttl=1200, show_spinner=False)
def cached_news(symbol: str, company_name: str):
    return fetch_news(symbol, company_name=company_name, max_items=8)


def price_chart(
    hist: pd.DataFrame,
    symbol: str,
    currency: str | None,
    show_sma20: bool,
    show_sma50: bool,
    show_sma200: bool,
    show_bollinger: bool = False,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name=symbol,
            increasing_line_color="#50d4a0",
            decreasing_line_color="#ff6f80",
        )
    )
    overlays = [
        (show_sma20, "SMA_20", "SMA 20", "#79a7ff"),
        (show_sma50, "SMA_50", "SMA 50", "#e8b465"),
        (show_sma200, "SMA_200", "SMA 200", "#b58cff"),
    ]
    for enabled, column, name, color in overlays:
        if enabled and column in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist[column], name=name, line=dict(width=1.55, color=color)))
    if show_bollinger and {"BB_UPPER", "BB_LOWER"}.issubset(hist.columns):
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_UPPER"], name="BB upper", line=dict(width=1, color="#5d6d86", dash="dot")))
        fig.add_trace(go.Scatter(x=hist.index, y=hist["BB_LOWER"], name="BB lower", line=dict(width=1, color="#5d6d86", dash="dot"), fill="tonexty", fillcolor="rgba(93,109,134,0.06)"))
    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="#0a111d",
        plot_bgcolor="#0a111d",
        margin=dict(l=10, r=10, t=18, b=10),
        xaxis_rangeslider_visible=False,
        hovermode="x unified",
        legend=dict(orientation="h", y=1.045, x=0),
        xaxis=dict(gridcolor="#1b2739"),
        yaxis=dict(gridcolor="#1b2739", side="right", title=currency or "Price"),
    )
    return fig


def rsi_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["RSI_14"], name="RSI 14", line=dict(width=1.8, color="#79a7ff")))
    fig.add_hline(y=70, line_dash="dot", line_color="#ff6f80")
    fig.add_hline(y=50, line_dash="dot", line_color="#4a5568")
    fig.add_hline(y=30, line_dash="dot", line_color="#50d4a0")
    fig.update_layout(
        height=250,
        template="plotly_dark",
        paper_bgcolor="#0a111d",
        plot_bgcolor="#0a111d",
        margin=dict(l=10, r=10, t=12, b=10),
        yaxis=dict(range=[0, 100], side="right", gridcolor="#1b2739"),
        xaxis=dict(gridcolor="#1b2739"),
        showlegend=False,
    )
    return fig


def macd_chart(hist: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD"], name="MACD", line=dict(width=1.6, color="#79a7ff")))
    fig.add_trace(go.Scatter(x=hist.index, y=hist["MACD_SIGNAL"], name="Signal", line=dict(width=1.3, color="#e8b465")))
    fig.add_trace(go.Bar(x=hist.index, y=hist["MACD_HIST"], name="Histogram", marker_color="#4f6280"))
    fig.update_layout(
        height=250,
        template="plotly_dark",
        paper_bgcolor="#0a111d",
        plot_bgcolor="#0a111d",
        margin=dict(l=10, r=10, t=12, b=10),
        yaxis=dict(side="right", gridcolor="#1b2739"),
        xaxis=dict(gridcolor="#1b2739"),
        legend=dict(orientation="h", y=1.05, x=0),
    )
    return fig


def equity_chart(result) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity.index, y=result.equity, name="SMA strategy", line=dict(width=2.2, color="#79a7ff")))
    fig.add_trace(go.Scatter(x=result.benchmark.index, y=result.benchmark, name="Buy and hold", line=dict(width=1.45, dash="dot", color="#93a1b7")))
    fig.update_layout(
        height=430,
        template="plotly_dark",
        paper_bgcolor="#0a111d",
        plot_bgcolor="#0a111d",
        margin=dict(l=10, r=10, t=16, b=10),
        hovermode="x unified",
        yaxis_title="Growth of 1 unit",
        yaxis=dict(side="right", gridcolor="#1b2739"),
        xaxis=dict(gridcolor="#1b2739"),
    )
    return fig


def data_quality(bundle: MarketBundle) -> dict[str, object]:
    hist = bundle.history
    required = ["Open", "High", "Low", "Close"]
    missing = int(hist[required].isna().sum().sum()) if not hist.empty else 0
    last_bar = pd.Timestamp(hist.index[-1]) if not hist.empty else None
    now = pd.Timestamp.now(tz=None)
    age_days = (now.normalize() - last_bar.normalize()).days if last_bar is not None else None
    return {
        "Provider": bundle.source,
        "Adjusted prices": "Yes" if bundle.adjusted_prices else "Provider-dependent",
        "Requested display period": bundle.requested_period,
        "Display rows": bundle.display_rows,
        "Analysis rows including warm-up": bundle.analysis_rows,
        "Missing OHLC cells": missing,
        "Last market bar": last_bar.strftime("%Y-%m-%d") if last_bar is not None else "N/A",
        "Calendar age of last bar": f"{age_days} days" if age_days is not None else "N/A",
        "Fetched at UTC": bundle.fetched_at.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
    }


def main() -> None:
    st.sidebar.markdown("## AXIOM")
    st.sidebar.caption("Market intelligence terminal, V3")

    symbol = st.sidebar.text_input("Symbol", value="AAPL").strip().upper()
    valid, validation_message = validate_ticker(symbol)
    if not valid:
        st.sidebar.error(validation_message)
        st.stop()

    period_label = st.sidebar.selectbox("Display history", list(PERIOD_OPTIONS), index=3)
    period = PERIOD_OPTIONS[period_label]
    demo_mode = st.sidebar.toggle("Demo data", value=False, help="Uses deterministic synthetic data for interface testing.")

    st.sidebar.markdown("### Default overlays")
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
    analysis_hist = bundle.analysis_history
    snapshot = market_snapshot(analysis_hist)
    if not snapshot:
        st.error("Market data loaded, but the analytical snapshot could not be calculated.")
        st.stop()

    info = bundle.info
    company_name = info.get("longName") or symbol
    currency = info.get("currency") or ("USD" if symbol.endswith("-USD") else None)
    latest_bar = pd.Timestamp(hist.index[-1])

    st.markdown('<div class="ax-kicker">AI TRADING INSIGHT DASHBOARD · V3 STABILITY BUILD</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="ax-title">{company_name} <span style="color:#77869b;font-weight:520">{symbol}</span></div>',
        unsafe_allow_html=True,
    )
    adjusted_badge = "Adjusted OHLC" if bundle.adjusted_prices else "Provider pricing"
    st.markdown(
        f'<div class="ax-sub"><span class="ax-badge"><span class="ax-status"></span>{bundle.source}</span>'
        f'<span class="ax-badge">{period_label}</span>'
        f'<span class="ax-badge">{adjusted_badge}</span>'
        f'<span class="ax-badge">Last bar {latest_bar.strftime("%d %b %Y")}</span></div>',
        unsafe_allow_html=True,
    )
    if bundle.warning:
        st.warning(bundle.warning)

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Price", fmt_price(snapshot.get("price"), currency), fmt_pct(snapshot.get("daily_return")))
    m2.metric("1M return", fmt_pct(snapshot.get("return_1m")))
    m3.metric("RSI 14", fmt_num(snapshot.get("rsi_14"), 1))
    m4.metric("60D volatility", fmt_pct(snapshot.get("annualized_volatility")), snapshot.get("volatility_regime", "N/A"))
    m5.metric("Max drawdown", fmt_pct(snapshot.get("max_drawdown")))
    m6.metric("Trend regime", snapshot.get("regime", "N/A"), f"Score {snapshot.get('trend_score', 0):+d}, coverage {snapshot.get('data_coverage', 0)}%")

    tabs = st.tabs(["Overview", "Technicals", "Risk", "Backtest", "AI Research", "News", "Data Quality"])
    backtest_metrics = None
    backtest_config = None

    with tabs[0]:
        left, right = st.columns([1.62, 1])
        with left:
            st.markdown("### Market state")
            st.plotly_chart(price_chart(hist, symbol, currency, True, True, False), use_container_width=True)
        with right:
            st.markdown("### Deterministic trend model")
            st.metric(
                "Normalized score",
                f"{snapshot['trend_score']:+d} / 100",
                f"Coverage {snapshot['data_coverage']}%",
            )
            factor_df = pd.DataFrame(snapshot["factors"])[["factor", "state", "score", "weight", "detail"]]
            st.dataframe(factor_df, hide_index=True, use_container_width=True)
            st.caption("The trend model is rule-based and computed from the warm-up history, so changing the visible chart range does not change the current signal state.")

        s1, s2, s3, s4, s5 = st.columns(5)
        s1.metric("20D support", fmt_price(snapshot.get("support_20"), currency))
        s2.metric("20D resistance", fmt_price(snapshot.get("resistance_20"), currency))
        s3.metric("ATR 14", fmt_price(snapshot.get("atr_14"), currency), fmt_pct(snapshot.get("atr_pct")))
        s4.metric("3M return", fmt_pct(snapshot.get("return_3m")))
        s5.metric("1Y return", fmt_pct(snapshot.get("return_1y")))

    with tabs[1]:
        controls = st.columns(4)
        with controls[0]:
            show_sma20_tab = st.checkbox("SMA 20", value=show_sma20, key="tab_sma20")
        with controls[1]:
            show_sma50_tab = st.checkbox("SMA 50", value=show_sma50, key="tab_sma50")
        with controls[2]:
            show_sma200_tab = st.checkbox("SMA 200", value=show_sma200, key="tab_sma200")
        with controls[3]:
            show_bb = st.checkbox("Bollinger bands", value=False)
        st.plotly_chart(
            price_chart(hist, symbol, currency, show_sma20_tab, show_sma50_tab, show_sma200_tab, show_bb),
            use_container_width=True,
        )
        osc1, osc2 = st.columns(2)
        with osc1:
            st.markdown("#### RSI 14")
            st.plotly_chart(rsi_chart(hist), use_container_width=True)
        with osc2:
            st.markdown("#### MACD")
            st.plotly_chart(macd_chart(hist), use_container_width=True)

    with tabs[2]:
        st.markdown("### Position risk calculator")
        st.caption("Risk sizing is constrained by both stop distance and a maximum notional leverage limit.")
        rc1, rc2 = st.columns([1, 1.2])
        with rc1:
            account_size = st.number_input("Account size", min_value=100.0, value=10000.0, step=500.0)
            risk_pct = st.number_input("Risk per trade (%)", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            direction = st.selectbox("Direction", ["Long", "Short"])
            max_leverage = st.number_input("Maximum leverage", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
            entry_default = float(snapshot["price"])
            entry = st.number_input(
                "Reference entry",
                min_value=0.000001,
                value=entry_default,
                step=max(entry_default * 0.001, 0.0001),
                format="%.6f" if entry_default < 10 else "%.2f",
            )
            atr_value = float(snapshot.get("atr_14") or entry * 0.02)
            default_stop = max(0.000001, entry - 2 * atr_value) if direction == "Long" else entry + 2 * atr_value
            stop = st.number_input(
                "Reference stop",
                min_value=0.000001,
                value=float(default_stop),
                step=max(entry * 0.001, 0.0001),
                format="%.6f" if entry < 10 else "%.2f",
            )
        with rc2:
            sizing = position_size(account_size, risk_pct, entry, stop, direction=direction, max_leverage=max_leverage)
            if not sizing.get("valid"):
                st.error(sizing.get("error", "Invalid risk configuration."))
            else:
                p1, p2 = st.columns(2)
                p1.metric("Target risk budget", fmt_num(sizing["risk_amount"], 2, currency_prefix(currency)))
                p2.metric("Actual risk", fmt_num(sizing["actual_risk"], 2, currency_prefix(currency)))
                p3, p4 = st.columns(2)
                p3.metric("Position units", fmt_num(sizing["units"], 4 if sizing["units"] < 10 else 2))
                p4.metric("Notional", fmt_num(sizing["notional"], 2, currency_prefix(currency)))
                p5, p6 = st.columns(2)
                p5.metric("Risk per unit", fmt_price(sizing["risk_per_unit"], currency))
                p6.metric("Binding constraint", sizing["limited_by"].title())
                if sizing["limited_by"] == "capital":
                    st.info("The risk-based unit count would exceed the selected leverage limit, so the position has been capped by available notional capital.")

        st.markdown("#### Reference levels")
        levels = pd.DataFrame(
            [
                {"Level": "20D support", "Price": snapshot.get("support_20")},
                {"Level": "SMA 20", "Price": snapshot.get("sma_20")},
                {"Level": "SMA 50", "Price": snapshot.get("sma_50")},
                {"Level": "SMA 100", "Price": snapshot.get("sma_100")},
                {"Level": "SMA 200", "Price": snapshot.get("sma_200")},
                {"Level": "20D resistance", "Price": snapshot.get("resistance_20")},
            ]
        ).dropna()
        levels["Formatted"] = levels["Price"].map(lambda x: fmt_price(x, currency))
        st.dataframe(levels[["Level", "Formatted"]], hide_index=True, use_container_width=True)

    with tabs[3]:
        st.markdown("### Reproducible SMA backtest")
        st.caption("Signal is observed at the session close, execution occurs at the next session open, and costs are charged on entries and exits.")
        b1, b2 = st.columns(2)
        with b1:
            sma_window = st.selectbox("SMA length", [20, 50, 100, 200], index=1)
        with b2:
            transaction_cost_bps = st.number_input("Cost per side (bps)", min_value=0.0, max_value=100.0, value=5.0, step=1.0)
        backtest_config = {"sma_window": sma_window, "transaction_cost_bps": transaction_cost_bps}
        try:
            bt = backtest_sma_strategy(
                analysis_hist,
                sma_window=sma_window,
                transaction_cost_bps=transaction_cost_bps,
                start_date=hist.index[0],
            )
            backtest_metrics = bt.metrics()
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Strategy return", fmt_pct(backtest_metrics["total_return"]))
            k2.metric("Buy and hold", fmt_pct(backtest_metrics["buy_hold_return"]))
            k3.metric("CAGR", fmt_pct(backtest_metrics["cagr"]))
            k4.metric("Max drawdown", fmt_pct(backtest_metrics["max_drawdown"]))
            k5, k6, k7, k8 = st.columns(4)
            k5.metric("Sharpe", "N/A" if backtest_metrics["sharpe"] is None else f"{backtest_metrics['sharpe']:.2f}")
            k6.metric("Sortino", "N/A" if backtest_metrics["sortino"] is None else f"{backtest_metrics['sortino']:.2f}")
            k7.metric("Win rate", fmt_pct(backtest_metrics["win_rate"]))
            k8.metric("Trades", str(backtest_metrics["trades"]))
            st.plotly_chart(equity_chart(bt), use_container_width=True)

            details = pd.DataFrame(
                {
                    "Metric": ["Exposure", "Annualized volatility", "Calmar", "Average trade", "Best trade", "Worst trade"],
                    "Value": [
                        fmt_pct(backtest_metrics["exposure"]),
                        fmt_pct(backtest_metrics["annualized_volatility"]),
                        "N/A" if backtest_metrics["calmar"] is None else f"{backtest_metrics['calmar']:.2f}",
                        fmt_pct(backtest_metrics["avg_trade_return"]),
                        fmt_pct(backtest_metrics["best_trade"]),
                        fmt_pct(backtest_metrics["worst_trade"]),
                    ],
                }
            )
            st.dataframe(details, hide_index=True, use_container_width=True)
            st.caption("Historical simulation only. Results exclude taxes, financing, market impact, and detailed slippage beyond the configured per-side cost.")
        except ValueError as exc:
            st.info(str(exc))

    with tabs[4]:
        st.markdown("### Grounded AI research")
        st.caption("AI receives the deterministic market snapshot, the current backtest configuration, provider metadata, and current retrieved headlines. It does not calculate the trading regime itself.")
        try:
            groq_key = st.secrets.get("GROQ_API_KEY")
            groq_model = st.secrets.get("GROQ_MODEL", "llama-3.3-70b-versatile")
        except Exception:
            groq_key = None
            groq_model = "llama-3.3-70b-versatile"

        news_items = cached_news(symbol, company_name)
        context_payload = {
            "symbol": symbol,
            "period": period,
            "last_bar": str(latest_bar.date()),
            "source": bundle.source,
            "adjusted_prices": bundle.adjusted_prices,
            "trend_score": snapshot.get("trend_score"),
            "backtest": backtest_config,
            "backtest_metrics": backtest_metrics,
        }
        context_id = research_context_key(context_payload)
        session_key = f"research_{context_id}"

        st.markdown(
            f'<div class="ax-mini">Research context: {symbol} · {period_label} · {latest_bar.strftime("%Y-%m-%d")} · context {context_id}</div>',
            unsafe_allow_html=True,
        )

        if not groq_key or Groq is None:
            st.info("Add GROQ_API_KEY to Streamlit secrets to enable AI research. All deterministic analytics remain available without it.")
        else:
            if st.button("Generate research brief", type="primary"):
                prompt = build_research_prompt(
                    symbol=symbol,
                    company_name=company_name,
                    snapshot=snapshot,
                    backtest_metrics=backtest_metrics,
                    news=news_items,
                    data_context={
                        "provider": bundle.source,
                        "fetched_at_utc": bundle.fetched_at.isoformat(),
                        "last_bar": str(latest_bar.date()),
                        "requested_period": period_label,
                        "adjusted_prices": bundle.adjusted_prices,
                        "currency": currency,
                    },
                )
                with st.spinner("Generating grounded research brief..."):
                    data, raw, error = generate_research(Groq(api_key=groq_key), prompt, model=groq_model)
                if data:
                    st.session_state[session_key] = {"data": data, "raw": raw, "generated_at": datetime.now(timezone.utc).isoformat()}
                else:
                    st.error(error or "The AI service did not return a valid research brief.")

            stored = st.session_state.get(session_key)
            if stored:
                data = stored["data"]
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
                    file_name=f"{symbol}_research_{context_id}.json",
                    mime="application/json",
                )

    with tabs[5]:
        st.markdown("### Latest market headlines")
        news_items = cached_news(symbol, company_name)
        if not news_items:
            st.info("No headlines were retrieved for this asset from the current news source.")
        for item in news_items:
            source = item.get("source") or "Google News"
            published = item.get("published") or ""
            title = item.get("title", "Untitled")
            link = item.get("link", "")
            st.markdown(f"**[{title}]({link})**" if link else f"**{title}**")
            st.caption(f"{source} · {published}".strip(" ·"))
            st.markdown("---")

    with tabs[6]:
        st.markdown("### Data quality and provenance")
        quality = data_quality(bundle)
        st.dataframe(pd.DataFrame(quality.items(), columns=["Check", "Value"]), hide_index=True, use_container_width=True)

        metadata = {
            "Company": company_name,
            "Symbol": symbol,
            "Exchange": info.get("exchange"),
            "Currency": currency,
            "Sector": info.get("sector"),
            "Industry": info.get("industry"),
            "Market cap": info.get("marketCap"),
            "52-week low": info.get("fiftyTwoWeekLow"),
            "52-week high": info.get("fiftyTwoWeekHigh"),
        }
        st.markdown("#### Market metadata")
        st.dataframe(pd.DataFrame(metadata.items(), columns=["Field", "Value"]), hide_index=True, use_container_width=True)

        d1, d2 = st.columns(2)
        display_export = hist.copy()
        display_export.index.name = "Date"
        analysis_export = analysis_hist.copy()
        analysis_export.index.name = "Date"
        with d1:
            st.download_button(
                "Download displayed enriched OHLCV",
                display_export.to_csv().encode("utf-8"),
                file_name=f"{symbol}_{period}_display_enriched.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with d2:
            st.download_button(
                "Download analysis history",
                analysis_export.to_csv().encode("utf-8"),
                file_name=f"{symbol}_{period}_analysis_history.csv",
                mime="text/csv",
                use_container_width=True,
            )

    st.markdown(
        '<div class="ax-rule"></div><div class="ax-note">Research and educational use only. Market data can be delayed, incomplete, or sourced from fallback providers. Historical simulations do not guarantee future performance.</div>',
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
