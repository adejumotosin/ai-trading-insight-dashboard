# ai-trading-insight-dashboard
A Streamlit-based AI-powered financial dashboard for live trading insights, candlestick analysis, news sentiment, and downloadable reports.
# AI-Powered Trading & Market Insight Dashboard

This is an AI-enhanced financial dashboard built with Streamlit, integrating real-time stock data, news sentiment analysis, and Gemini AI to provide intelligent trading insights for stocks, cryptocurrencies, and other assets.

---

## Features

| Feature                   | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Candlestick Charts       | Visualize 1-year price movements of any asset (e.g., AAPL, BTC, TSLA)       |
| AI Insight Generator     | Generate professional trading insights using Google Gemini AI               |
| News Sentiment Analysis  | Get real-time financial headlines and sentiment from Google News RSS        |
| Multi-language Support   | Translate AI insights into English, French, Spanish, German, or Chinese     |
| Export Reports           | Download AI analysis as PDF and CSV files                                   |
| Simulated Login Access   | Restrict access using simple email-based input                              |

---

## Powered By

- Gemini 1.5 Flash (Google Generative AI)
- Yahoo Finance (via yfinance)
- Google News RSS
- Streamlit for UI
- Plotly for charts
- FPDF for report generation

---

## Demo

![App Screenshot](https://user-images.githubusercontent.com/yourusername/demo-screenshot.png)

> Replace with your own screenshot or GIF demo.

---

## Tech Stack

- Python
- Streamlit
- yfinance
- plotly
- google-generativeai
- fpdf
- BeautifulSoup
- googletrans

---

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-trading-insight-dashboard
cd ai-trading-insight-dashboard

# Install dependencies
pip install -r requirements.txt

# Create secrets file
mkdir -p .streamlit
echo 'GEMINI_API_KEY = "your-gemini-api-key"' > .streamlit/secrets.toml

# Run the app
streamlit run finance.py
```

---

## Secrets Configuration

Create a `.streamlit/secrets.toml` file with your Gemini API key:

```toml
GEMINI_API_KEY = "your-gemini-api-key"
```

---

## Deployment

To deploy on [Streamlit Cloud](https://streamlit.io/cloud):

1. Push your project to GitHub
2. Create a new app from the repo on Streamlit Cloud
3. Add your `GEMINI_API_KEY` in the app's Secrets Manager
4. Deploy the app

---

## License

This project is licensed under the MIT License.

---

## Contributing

Contributions are welcome. Please fork the repository, make your changes, and open a pull request. For major changes, please open an issue first to discuss what you would like to change.

---

## Contact

Developed by [Your Name]  
Email: adejumoking@gmail.com
