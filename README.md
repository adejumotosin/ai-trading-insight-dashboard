# Axiom Market Intelligence

A Streamlit market-research dashboard for transparent technical analysis, risk sizing, historical strategy testing, market news, and grounded AI commentary.

## V3 highlights

- Range-independent technical analysis with indicator warm-up history
- Corporate-action-adjusted Yahoo Finance OHLC data for cleaner return and backtest calculations
- Safer Stooq fallback that is limited to compatible equity-style symbols
- Deterministic trend regime with normalized score and explicit data coverage
- SMA 20/50/100/200, EMA 20, RSI 14, MACD, ATR, Bollinger Bands, volatility regime, support and resistance
- Currency-aware price formatting for non-USD assets
- Long and short position sizing with stop validation and leverage caps
- SMA backtest with close-signal / next-open execution, entry and exit costs, and warm-up data
- Backtest metrics including CAGR, Sharpe, Sortino, Calmar, drawdown, exposure, win rate, and trade statistics
- Context-bound AI research so a brief cannot silently carry over to another symbol, period, market bar, or backtest configuration
- Data-quality and provenance panel
- GitHub Actions CI and regression tests
- Deterministic demo data for UI testing when external providers are unavailable

## Architecture

```text
app.py
  |
  +-- core/market_data.py   -> Yahoo Finance, Stooq fallback, news, warm-up history
  +-- core/analytics.py     -> indicators, trend model, risk sizing, backtesting
  +-- core/ai_research.py   -> grounded AI prompt, JSON validation
  |
  +-- tests/                -> regression tests
```

`finance.py` remains a compatibility entrypoint for deployments that still run `streamlit run finance.py`.

## Run locally

```bash
git clone https://github.com/adejumotosin/ai-trading-insight-dashboard.git
cd ai-trading-insight-dashboard
python -m venv .venv
```

Activate the environment:

```bash
# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
streamlit run app.py
```

## Optional AI research

The deterministic dashboard works without an AI key. To enable AI research, add a Streamlit secret:

```toml
GROQ_API_KEY = "your-key"
```

You may optionally override the model:

```toml
GROQ_MODEL = "llama-3.3-70b-versatile"
```

Do not commit `.streamlit/secrets.toml` to Git.

## Backtest methodology

The included SMA strategy is intentionally simple and reproducible:

1. The signal is calculated after the daily close.
2. The position changes at the next session open.
3. Position P&L is measured from open to open while held.
4. A final open position is marked to the last available close.
5. The configured transaction cost is charged on entries and exits.
6. Indicator warm-up data is loaded before the selected evaluation window.

This design removes the previous close-to-close execution mismatch and prevents the selected chart range from deleting the history required to calculate longer moving averages.

## Data notes

Yahoo Finance is used as the primary research data source and adjusted OHLC prices are requested to reduce split-related distortions. Stooq is a fallback for compatible equity symbols only. Fallback adjustment behavior can differ from Yahoo Finance and is identified in the interface.

The dashboard is research software, not an execution system or institutional market-data terminal. Market data may be delayed, incomplete, or revised.

## Tests

```bash
pip install -r requirements-dev.txt
pytest -q
```

CI runs compilation and tests on supported Python versions for every pull request.

## Author

Oluwatosin Adejumo
