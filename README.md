# AI Trading Insight Dashboard V2

A Streamlit market-intelligence dashboard that combines resilient market data, transparent technical analytics, reproducible strategy backtests, risk-based position sizing, linked market headlines, and grounded AI research.

## What changed in V2

- Rebuilt the project into a modular application instead of keeping the entire system in one large `finance.py` file.
- Fixed the history selector so the selected 1M, 3M, 6M, 1Y, 2Y, 5Y, or 10Y period is actually used for market-data retrieval and analytics.
- Added a deterministic trend engine with explainable factor scores rather than letting the language model invent a trading signal.
- Added RSI, MACD, ATR, realized volatility, multi-horizon returns, support/resistance, and max-drawdown analytics.
- Added a reproducible long-only SMA backtest with next-session execution, configurable SMA length, explicit transaction costs, Sharpe ratio, drawdown, exposure, and trade statistics.
- Added a position-risk calculator based on account size, risk percentage, entry, and stop distance.
- Reworked AI output into a grounded research brief using only supplied technicals, backtest metrics, and retrieved headlines.
- AI is now optional. Missing `GROQ_API_KEY` no longer prevents the rest of the dashboard from loading.
- News headlines now preserve clickable source links and publication metadata.
- Added deterministic demo data for interface testing when external market APIs are unavailable.
- Expanded symbol validation to support common Yahoo Finance formats such as `BTC-USD`, `^GSPC`, `GC=F`, and `EURUSD=X`.
- Added enriched OHLCV CSV exports.
- Replaced the older visual treatment with a denser finance-terminal interface.

## Architecture

```text
app.py
├── core/market_data.py   # Yahoo/Stooq/demo data + news
├── core/analytics.py     # indicators, market state, risk sizing, backtest
└── core/ai_research.py   # grounded AI prompt + JSON parsing

finance.py                # compatibility entrypoint for existing deployments
```

## Run locally

```bash
git clone https://github.com/adejumotosin/ai-trading-insight-dashboard.git
cd ai-trading-insight-dashboard
python -m venv .venv
```

Activate the environment and install dependencies:

```bash
pip install -r requirements.txt
```

Optional Streamlit secrets:

```toml
GROQ_API_KEY = "your-groq-api-key"

# Optional access gate. If omitted, the dashboard is publicly usable.
allowed_users = ["you@example.com"]
```

Run either entrypoint:

```bash
streamlit run app.py
```

Existing deployments that run `streamlit run finance.py` continue to work because `finance.py` is retained as a compatibility entrypoint.

## Backtest definition

The built-in SMA strategy is intentionally simple and reproducible:

1. Compute the selected simple moving average from daily closing prices.
2. Signal long when `Close > SMA`.
3. Shift the position by one session to avoid using the same close for both signal generation and execution.
4. Deduct the configured transaction cost whenever the position changes.
5. Compare the resulting equity curve with buy-and-hold over the same sample.

This is a research baseline, not evidence that the rule will remain profitable.

## Data caveats

Yahoo Finance and Stooq are convenient research sources, not institutional execution feeds. Data may be delayed, adjusted, incomplete, or temporarily unavailable. Demo mode is synthetic and is clearly labeled in the interface.

## License

MIT

## Author

Oluwatosin Adejumo
