import os
import json
import re
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from io import BytesIO
from deep_translator import GoogleTranslator
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai

# --- Configure Gemini API Key ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("❌ Gemini API Key is missing. Please set it in your Streamlit secrets.")
    st.stop()

# --- Page Setup ---
st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 AI-Powered Trading & Market Insight Dashboard")

# --- Sidebar ---
st.sidebar.header("🔐 User Authentication")

# Clear cache button
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.success("Cache cleared! Please refresh the page.")

# User login input
user_email = st.sidebar.text_input("Enter your Google email:", value=st.session_state.get('user_email', ''))
st.session_state.user_email = user_email
allowed_users = ["your@email.com", "admin@gmail.com"]

if user_email not in allowed_users:
    st.sidebar.warning("Access is restricted to authorized users.")
    st.stop()

st.sidebar.success(f"Welcome, {user_email}!")

# Language selection
lang = st.sidebar.selectbox("🌍 Choose Output Language", ["English", "French", "Spanish", "German", "Chinese"])

# Ticker input
asset = st.sidebar.text_input("📈 Enter Ticker Symbol (e.g., AAPL, BTC-USD):", value="AAPL").upper()

if not asset:
    st.warning("Please enter a ticker symbol to begin.")
    st.stop()

# --- Caching functions ---
@st.cache_data(ttl=300)
def get_stock_data(ticker_symbol):
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        hist = ticker.history(period="1y")
        return info, hist
    except Exception as e:
        st.error(f"⚠️ Failed to fetch data for {ticker_symbol}: {e}")
        return None, None

@st.cache_data(ttl=3600)
def fetch_google_news_headlines(asset):
    try:
        url = f"https://news.google.com/rss/search?q={asset}+stock&hl=en-US&gl=US&ceid=US:en"
        page = requests.get(url)
        page.raise_for_status()
        soup = BeautifulSoup(page.content, features="xml")
        return [item.title.text for item in soup.findAll("item")[:5]]
    except Exception as e:
        st.error(f"Failed to fetch Google News headlines: {e}")
        return []

# --- Helper functions ---
def translate_text(text, dest_lang):
    """Translates text to the destination language."""
    if dest_lang.lower() == "english" or not text:
        return text
    try:
        return GoogleTranslator(source='auto', target=dest_lang.lower()[:2]).translate(text)
    except Exception as e:
        st.warning(f"Text translation failed: {e}")
        return text

def generate_gemini_insight(prompt):
    """Generates insights using the Gemini model and parses the JSON response."""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash-latest")
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        match = re.search(r"```json\s*({.*?})\s*```", response_text, re.DOTALL)
        if not match:
            match = re.search(r"({.*})", response_text, re.DOTALL)
        
        if match:
            return json.loads(match.group(1)), response_text
        else:
            st.warning("AI did not return a valid JSON format. Displaying raw text.")
            return None, response_text
    except Exception as e:
        st.error(f"An error occurred with the Gemini AI: {e}")
        return None, None

def create_pdf_report(asset, metrics, insight, sentiment):
    """Creates a PDF report from the collected data."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, f"{asset} AI Trading Report", ln=True, align="C")
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for key, value in metrics.items():
        pdf.cell(0, 10, txt=f"{key}: {value}", ln=True)
    if insight:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Executive Summary:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 5, txt=insight.get("executive_summary", "N/A"))
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "Recommendation:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 5, txt=insight.get("bottom_line", "N/A"))
    if sentiment:
        pdf.ln(5)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, "News Sentiment:", ln=True)
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 5, txt=sentiment.get("summary", "N/A"))
    return pdf.output(dest='S').encode('latin1')

# --- Fetch stock data ---
info, hist = get_stock_data(asset)
if info is None or hist.empty:
    st.error(f"Could not retrieve data for {asset}. Please check the ticker symbol.")
    st.stop()

# --- Candlestick Chart ---
st.subheader(f"Candlestick Chart for {asset}")
fig = go.Figure(data=[go.Candlestick(
    x=hist.index,
    open=hist['Open'], high=hist['High'],
    low=hist['Low'], close=hist['Close']
)])
fig.update_layout(xaxis_rangeslider_visible=False, title=f"{info.get('longName', asset)} - 1 Year Performance")
st.plotly_chart(fig, use_container_width=True)

# --- Key Metrics ---
st.markdown("### 🔧 Key Metrics (Live Data)")

# Fetch the dividend yield from the ticker info
div_yield = info.get('dividendYield')

if div_yield is not None and not pd.isna(div_yield):
    # Heuristically check if the value is likely a percentage (e.g., 0.44, 2.5)
    # or a decimal (e.g., 0.0044, 0.025).
    # A common-sense check: dividend yields rarely exceed 15-20%.
    # If the value is small, but larger than what a decimal should be, it's likely a percentage.
    if div_yield > 0.05 and div_yield < 100:
        # If it's something like 0.44 or 2.5, it's probably a percentage that wasn't converted
        value_to_format = div_yield / 100
    else:
        # Otherwise, assume it's a correctly formatted decimal
        value_to_format = div_yield
        
    dividend_yield_str = f"{value_to_format:.2%}"
else:
    dividend_yield_str = "N/A"

key_metrics = {
    "Price": f"${info.get('currentPrice', 'N/A'):,.2f}",
    "Market Cap": f"${info.get('marketCap', 0):,}",
    "Volume": f"{info.get('volume', 0):,}",
    "52 Week Range": f"${info.get('fiftyTwoWeekLow', 'N/A'):,.2f} - ${info.get('fiftyTwoWeekHigh', 'N/A'):,.2f}",
    "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A",
    "Dividend Yield": dividend_yield_str
}

metrics_df = pd.DataFrame(key_metrics.items(), columns=["Metric", "Value"])
st.table(metrics_df)


# --- Gemini AI Insight ---
st.markdown("### 🧠 AI Trading Insight")
if st.button("Generate AI Insight"):
    with st.spinner("🤖 Generating AI Analysis..."):
        prompt = f"""
        As a professional AI market analyst, analyze the asset "{asset}" using the following metrics:
        {json.dumps(key_metrics, indent=2)}
        Return a single, raw JSON object only, without any markdown formatting or code fences.
        The JSON object must contain these exact keys: "verdict", "best_for", "risk_score", "executive_summary", "pros", "cons", "strategy_suggestions", "recommendation", "bottom_line".
        """
        insight_data, raw_output = generate_gemini_insight(prompt)
        st.session_state.insight_data = insight_data
        st.session_state.raw_output = raw_output

if 'insight_data' in st.session_state and st.session_state.insight_data:
    insight_data = st.session_state.insight_data
    col1, col2, col3 = st.columns(3)
    col1.metric("Verdict", translate_text(insight_data.get("verdict", "N/A"), lang))
    col2.metric("Best For", translate_text(insight_data.get("best_for", "N/A"), lang))
    col3.metric("Risk Score", translate_text(str(insight_data.get("risk_score", "N/A")), lang))

    st.markdown("#### Executive Summary")
    st.info(translate_text(insight_data.get("executive_summary", ""), lang))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### ✅ Pros")
        for pro in insight_data.get("pros", []):
            st.markdown(f"- {translate_text(pro, lang)}")
    with c2:
        st.markdown("#### ⚠️ Cons")
        for con in insight_data.get("cons", []):
            st.markdown(f"- {translate_text(con, lang)}")

    st.markdown("#### 💡 Strategy Suggestions")
    for strat in insight_data.get("strategy_suggestions", []):
        st.success(f"📌 {translate_text(strat, lang)}")

    st.markdown("#### Recommendation")
    recommendation = f"**{translate_text(insight_data.get('recommendation', ''), lang)}** — {translate_text(insight_data.get('bottom_line', ''), lang)}"
    st.success(recommendation)

    with st.expander("Show Raw Gemini Output"):
        st.code(st.session_state.get('raw_output', 'No raw output available.'), language='json')

# --- News Sentiment ---
st.markdown("### 📰 News Sentiment Analysis")
headlines = fetch_google_news_headlines(asset)

if headlines:
    st.markdown("**Latest Headlines:**")
    for h in headlines:
        st.markdown(f"- {h}")
    
    if st.button("Analyze News Sentiment"):
        with st.spinner("Analyzing news sentiment..."):
            sentiment_prompt = f"""
            Analyze the sentiment of these headlines regarding the asset {asset}.
            Return a single, raw JSON object only, without markdown formatting.
            The JSON object must contain these exact keys: "sentiment" (e.g., "Positive", "Negative", "Neutral") and "summary".
            Headlines:
            {json.dumps(headlines, indent=2)}
            """
            sentiment_data, _ = generate_gemini_insight(sentiment_prompt)
            if sentiment_data:
                st.session_state.sentiment_data = sentiment_data

if 'sentiment_data' in st.session_state and st.session_state.sentiment_data:
    sentiment_data = st.session_state.sentiment_data
    st.info(f"🧠 **Sentiment:** {translate_text(sentiment_data.get('sentiment', 'N/A'), lang)}")
    st.markdown(f"📝 {translate_text(sentiment_data.get('summary', ''), lang)}")
else:
    st.warning("No recent news headlines were found or sentiment not yet analyzed.")

# --- Export PDF & CSV ---
st.markdown("### 📤 Export Report")
col1, col2 = st.columns(2)
with col1:
    if st.button("📄 Generate PDF Report"):
        if 'insight_data' in st.session_state and 'sentiment_data' in st.session_state:
            pdf_data = create_pdf_report(
                asset, 
                key_metrics, 
                st.session_state.insight_data, 
                st.session_state.sentiment_data
            )
            st.download_button(
                label="⬇️ Download PDF",
                data=pdf_data,
                file_name=f"{asset}_AI_Report_{datetime.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf"
            )
        else:
            st.warning("Please generate AI insight and sentiment analysis first.")

with col2:
    csv_data = metrics_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📊 Download Metrics (CSV)",
        data=csv_data,
        file_name=f"{asset}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )


