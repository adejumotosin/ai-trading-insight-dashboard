# AI Trading Insight Dashboard

A Streamlit-based financial dashboard that combines market data, candlestick visualization, financial news, sentiment analysis, and AI-generated research commentary.

## Features

- Interactive candlestick charts for supported market symbols
- Historical market data through Yahoo Finance / `yfinance`
- AI-generated market commentary using Google Gemini
- Financial news retrieval and sentiment-oriented analysis
- Multi-language output support
- Downloadable PDF and CSV reports
- Streamlit-based browser interface

## Technology

- Python
- Streamlit
- `yfinance`
- Plotly
- Google Generative AI
- Google News RSS
- BeautifulSoup
- FPDF
- Translation utilities

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

Configure the Gemini API key in `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

Run the dashboard:

```bash
streamlit run finance.py
```

## Architecture

```text
Market symbol
    |
    +------> Yahoo Finance price history
    |              |
    |              v
    |        Plotly visualizations
    |
    +------> Financial news feed
                   |
                   v
          AI analysis + sentiment
                   |
                   v
          Streamlit dashboard
                   |
                   v
             PDF / CSV export
```

## Current limitations

- AI-generated commentary is analytical assistance, not a trading signal with validated predictive performance.
- Yahoo Finance data is convenient for research but should not be treated as institutional market data.
- News sentiment quality depends on the availability and relevance of retrieved headlines.
- Any investment interpretation should be independently verified before use.

## Potential upgrades

- Add reproducible strategy backtests instead of qualitative AI commentary alone
- Add portfolio-level analytics and risk metrics
- Add economic-calendar and macro-event context
- Add source citations for generated claims
- Add caching and structured historical research storage
- Add model evaluation for sentiment and directional forecasts

## License

MIT License.

## Author

Oluwatosin Adejumo  
[tosinadejumo1997@gmail.com](mailto:tosinadejumo1997@gmail.com)
