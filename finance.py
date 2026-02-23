import os
import json
import re
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
from deep_translator import GoogleTranslator
from fpdf import FPDF
import requests
from bs4 import BeautifulSoup
try:
    from groq import Groq
except ImportError:
    st.error("❌ The 'groq' library is not installed. If you are on Streamlit Cloud, please wait for the app to finish installing dependencies or click 'Reboot App' in the 'Manage app' menu.")
    st.stop()
import hashlib
import hmac

# ============================================================================
# CONFIGURATION & SETUP
# ============================================================================

st.set_page_config(
    page_title="AI Trading Dashboard",
    page_icon="📊",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2227;
        padding: 15px !important;
        border-radius: 10px !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important;
        border: 1px solid #2d3139 !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 24px !important;
        font-weight: 700 !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 45px;
        background-color: #1e2227;
        border-radius: 5px 5px 0 0;
        padding: 10px 20px;
        color: #ccd6f6;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2d3139 !important;
        border-bottom: 2px solid #4e8df5 !important;
        color: #ffffff !important;
    }
    .stButton>button {
        border-radius: 5px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .sidebar .sidebar-content {
        background-color: #1e2227;
    }
</style>
""", unsafe_allow_html=True)

# Configure Groq API Key
try:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    client = Groq(api_key=groq_api_key)
except Exception:
    st.error("❌ Groq API Key is missing. Please set it in your Streamlit secrets as 'GROQ_API_KEY'.")
    st.stop()

# Initialize session state
if 'rate_limits' not in st.session_state:
    st.session_state.rate_limits = {}
if 'user_authenticated' not in st.session_state:
    st.session_state.user_authenticated = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_ticker(ticker):
    """
    Validate ticker symbol format.
    
    Args:
        ticker (str): Ticker symbol to validate
        
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    if not ticker:
        return False, "Ticker cannot be empty"
    
    # Basic validation: 1-5 uppercase letters/numbers, optional dash
    if not re.match(r'^[A-Z0-9]{1,5}(-[A-Z0-9]{1,5})?$', ticker):
        return False, "Invalid ticker format (e.g., AAPL, BTC-USD)"
    
    return True, ""


def check_rate_limit(key, max_calls=5, window_minutes=5):
    """
    Simple rate limiting using session state.
    
    Args:
        key (str): Unique identifier for the rate limit
        max_calls (int): Maximum number of calls allowed in the window
        window_minutes (int): Time window in minutes
        
    Returns:
        tuple: (can_proceed: bool, wait_time_seconds: int)
    """
    if 'rate_limits' not in st.session_state:
        st.session_state.rate_limits = {}
    
    now = datetime.now()
    if key not in st.session_state.rate_limits:
        st.session_state.rate_limits[key] = []
    
    # Remove old timestamps outside the window
    st.session_state.rate_limits[key] = [
        ts for ts in st.session_state.rate_limits[key]
        if now - ts < timedelta(minutes=window_minutes)
    ]
    
    # Check if limit exceeded
    if len(st.session_state.rate_limits[key]) >= max_calls:
        oldest = st.session_state.rate_limits[key][0]
        wait_time = int((oldest + timedelta(minutes=window_minutes) - now).total_seconds())
        return False, wait_time
    
    # Add current timestamp
    st.session_state.rate_limits[key].append(now)
    return True, 0


def translate_text(text, dest_lang):
    """
    Translates text to the destination language.
    
    Args:
        text (str): Text to translate
        dest_lang (str): Destination language name
        
    Returns:
        str: Translated text or original if translation fails
    """
    if dest_lang.lower() == "english" or not text:
        return text
    
    try:
        lang_code = dest_lang.lower()[:2]
        return GoogleTranslator(source='auto', target=lang_code).translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text


def format_dividend_yield(div_yield):
    """
    Safely format dividend yield with proper detection.
    
    Args:
        div_yield: Dividend yield value (can be None, float, or string)
        
    Returns:
        str: Formatted dividend yield percentage or "N/A"
    """
    if div_yield is None or pd.isna(div_yield):
        return "N/A"
    
    try:
        div_yield = float(div_yield)
        
        # If value is between 0 and 1, assume it's already decimal
        if 0 < div_yield < 1:
            return f"{div_yield:.2%}"
        # If between 1 and 100, assume it's percentage points
        elif 1 <= div_yield < 100:
            return f"{div_yield / 100:.2%}"
        else:
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"


def parse_ai_response(response_text):
    """
    Safely extract and validate JSON from AI response.
    """
    if not response_text:
        return None
    
    try:
        # Try direct JSON parse first
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError:
        # Extract from code blocks using multiple patterns
        patterns = [
            r"```json\s*(\{.*?\})\s*```",  # Standard markdown with json
            r"```\s*(\{.*?\})\s*```",      # Markdown without language specifier
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"  # Raw JSON with nested objects
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    # Basic validation - check if it's a dictionary
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        
        return None


# ============================================================================
# TECHNICAL ANALYSIS ENGINE
# ============================================================================

def calculate_indicators(df):
    """
    Calculate various technical indicators for the provided historical data.
    
    Args:
        df (DataFrame): Historical price data
        
    Returns:
        DataFrame: Dataframe with added indicator columns
    """
    if df is None or df.empty:
        return df
    
    # Simple Moving Averages
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
    
    return df


# ============================================================================
# DATA FETCHING FUNCTIONS (WITH CACHING)
# ============================================================================

@st.cache_data(ttl=60, show_spinner=False)
def get_current_price_safe(ticker_symbol):
    """
    Lightning fast price fetcher using history(1d) instead of heavy .info
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return None
    except Exception:
        return None

@st.cache_data(ttl=300, show_spinner=False)
def get_stock_data_safe(ticker_symbol):
    """
    Fetch stock data with comprehensive error handling.
    
    Args:
        ticker_symbol (str): Stock ticker symbol
        
    Returns:
        tuple: (info: dict, hist: DataFrame, error: str or None)
    """
    try:
        ticker = yf.Ticker(ticker_symbol)
        info = ticker.info
        
        # Validate info contains basic data
        if not info or 'currentPrice' not in info:
            return None, None, "No price data available for this ticker"
        
        hist = ticker.history(period="1y")
        
        # Validate historical data
        required_cols = ['Open', 'High', 'Low', 'Close']
        if hist.empty:
            return info, None, "No historical data available"
        
        if not all(col in hist.columns for col in required_cols):
            return info, None, "Historical data incomplete"
        
        # Calculate technical indicators
        hist = calculate_indicators(hist)
        
        return info, hist, None
        
    except Exception as e:
        error_msg = str(e).lower()
        if "too many requests" in error_msg or "rate limited" in error_msg:
            return None, None, "⚠️ Yahoo Finance rate limit reached. Please wait a few minutes and refresh."
        if "404" in error_msg or "no data found" in error_msg:
            return None, None, f"Ticker '{ticker_symbol}' not found. Please check the symbol."
        return None, None, f"Error fetching data: {error_msg}"


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_google_news_headlines(asset):
    """
    Fetch recent news headlines from Google News RSS feed.
    
    Args:
        asset (str): Asset/ticker symbol
        
    Returns:
        list: List of headline strings (max 5)
    """
    try:
        url = f"https://news.google.com/rss/search?q={asset}+stock&hl=en-US&gl=US&ceid=US:en"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, features="xml")
        items = soup.findAll("item")
        
        headlines = [item.title.text for item in items[:5] if item.title]
        return headlines
        
    except requests.exceptions.Timeout:
        st.warning("News fetch timed out. Please try again later.")
        return []
    except Exception as e:
        st.warning(f"Could not fetch news headlines: {e}")
        return []


def generate_ai_insight(prompt, insight_type="general"):
    """
    Generates insights using the Groq model and parses the JSON response.
    
    Args:
        prompt (str): The prompt to send to Groq
        insight_type (str): Type of insight for rate limiting
        
    Returns:
        tuple: (parsed_data: dict or None, raw_output: str or None)
    """
    # Check rate limit
    can_proceed, wait_time = check_rate_limit(f'groq_{insight_type}', max_calls=5, window_minutes=5)
    
    if not can_proceed:
        st.error(f"⏳ Rate limit reached. Please wait {wait_time} seconds before trying again.")
        return None, None
    
    try:
        # Using Llama-3-70b-8192 for high-quality insights
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "user", "content": prompt}
            ],
            model="llama-3-70b-8192",
            temperature=0.1,
        )
        response_text = chat_completion.choices[0].message.content.strip()
        
        parsed_data = parse_ai_response(response_text)
        
        if parsed_data is None:
            st.warning("⚠️ AI did not return valid JSON. Showing raw response.")
        
        return parsed_data, response_text
        
    except Exception as e:
        st.error(f"❌ Groq API error: {e}")
        return None, None


# ============================================================================
# PDF GENERATION
# ============================================================================

def create_pdf_report(asset, metrics, insight, sentiment):
    """
    Creates a PDF report from the collected data.
    
    Args:
        asset (str): Ticker symbol
        metrics (dict): Key metrics dictionary
        insight (dict): AI insight data
        sentiment (dict): Sentiment analysis data
        
    Returns:
        bytes: PDF file content
    """
    try:
        pdf = FPDF()
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"{asset} AI Trading Report", ln=True, align="C")
        pdf.ln(10)
        
        # Metrics section
        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Key Metrics", ln=True)
        pdf.set_font("Arial", size=11)
        
        for key, value in metrics.items():
            # Handle long values by truncating if necessary
            value_str = str(value)[:100]
            pdf.cell(0, 8, txt=f"{key}: {value_str}", ln=True)
        
        # AI Insight section
        if insight:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "AI Analysis", ln=True)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Executive Summary:", ln=True)
            pdf.set_font("Arial", size=10)
            summary = insight.get("executive_summary", "N/A")[:500]  # Limit length
            pdf.multi_cell(0, 5, txt=summary)
            
            pdf.ln(3)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Recommendation:", ln=True)
            pdf.set_font("Arial", size=10)
            recommendation = insight.get("bottom_line", "N/A")[:300]
            pdf.multi_cell(0, 5, txt=recommendation)
        
        # Sentiment section
        if sentiment:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "News Sentiment Analysis", ln=True)
            pdf.set_font("Arial", size=11)
            
            sentiment_text = sentiment.get("sentiment", "N/A")
            pdf.cell(0, 8, f"Overall Sentiment: {sentiment_text}", ln=True)
            
            pdf.set_font("Arial", size=10)
            summary = sentiment.get("summary", "N/A")[:400]
            pdf.multi_cell(0, 5, txt=summary)
        
        # Footer
        pdf.ln(10)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")
        
        return pdf.output(dest='S').encode('latin1', errors='replace')
        
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None


# ============================================================================
# AUTHENTICATION (IMPROVED BUT STILL BASIC)
# ============================================================================

def verify_user_simple(email):
    """
    Simple email-based verification.
    NOTE: This is still not secure for production use.
    For production, implement proper OAuth or JWT authentication.
    
    Args:
        email (str): User email address
        
    Returns:
        bool: True if user is authorized
    """
    # Get allowed users from secrets (more secure than hardcoding)
    try:
        allowed_users = st.secrets.get("allowed_users", [])
    except:
        # Fallback to hardcoded list if secrets not available
        allowed_users = ["your@email.com", "admin@gmail.com"]
    
    return email.lower().strip() in [u.lower().strip() for u in allowed_users]


# ============================================================================
# MAIN APPLICATION - PART 1 (SETUP & AUTHENTICATION)
# ============================================================================

st.title("📊 AI-Powered Trading & Market Insight Dashboard")

# Sidebar - Authentication
st.sidebar.header("🔐 User Authentication")

# Clear cache button
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.success("✅ Cache cleared! Refresh the page if needed.")

# User authentication
user_email = st.sidebar.text_input(
    "Enter your email:", 
    value=st.session_state.get('user_email', ''),
    key="email_input"
)

if user_email:
    st.session_state.user_email = user_email
    
    if verify_user_simple(user_email):
        st.session_state.user_authenticated = True
        st.sidebar.success(f"✅ Welcome, {user_email}!")
    else:
        st.session_state.user_authenticated = False
        st.sidebar.error("❌ Access denied. Please contact admin.")
        st.stop()
else:
    st.sidebar.warning("⚠️ Please enter your email to continue.")
    st.stop()

# Language selection
lang = st.sidebar.selectbox(
    "🌍 Choose Output Language", 
    ["English", "French", "Spanish", "German", "Chinese"],
    key="language_selector"
)

# Ticker input with validation
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Asset Selection")

asset_input = st.sidebar.text_input(
    "Enter Ticker Symbol:", 
    value="AAPL",
    help="Examples: AAPL, MSFT, GOOGL, BTC-USD, ETH-USD"
).upper()

# Validate ticker
is_valid, error_msg = validate_ticker(asset_input)

if not is_valid:
    st.sidebar.error(f"❌ {error_msg}")
    st.stop()

asset = asset_input
st.sidebar.success(f"✅ Analyzing: **{asset}**")

# Time period selection (optional enhancement)
period_options = {
    "1 Month": "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y"
}

selected_period = st.sidebar.selectbox(
    "📅 Select Time Period",
    options=list(period_options.keys()),
    index=3  # Default to 1 Year
)

# Fetch stock data
with st.spinner(f"📊 Fetching data for {asset}..."):
    info, hist, error = get_stock_data_safe(asset)

if error:
    st.error(f"⚠️ {error}")
    st.stop()

if info is None or hist is None:
    st.error(f"❌ Could not retrieve complete data for {asset}. Please try another ticker.")
    st.stop()

# Technical Indicators Selection
st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Technical Indicators")
show_sma = st.sidebar.checkbox("Show SMA (50/200)", value=False)
show_ema = st.sidebar.checkbox("Show EMA (20)", value=True)
show_rsi = st.sidebar.checkbox("Show RSI Chart", value=True)
show_macd = st.sidebar.checkbox("Show MACD Chart", value=True)

# Comparison Ticker
st.sidebar.markdown("---")
st.sidebar.subheader("⚖️ Compare Asset")
compare_ticker = st.sidebar.text_input(
    "Compare with (Optional):",
    value="",
    help="Enter another ticker to compare performance (e.g., SPY, QQQ, BTC-USD)"
).upper()

# Fetch comparison data
compare_hist = None
if compare_ticker:
    with st.spinner(f"⚖️ Fetching comparison data for {compare_ticker}..."):
        _, compare_hist, compare_error = get_stock_data_safe(compare_ticker)
        if compare_error:
            st.sidebar.warning(f"⚠️ Comparison error: {compare_error}")
            compare_hist = None

# Success message
st.success(f"✅ Successfully loaded data for **{info.get('longName', asset)}**")

# ============================================================================
# PART 2: VISUALIZATION & ANALYSIS
# ============================================================================
# This continues from Part 1. Place this code after the Part 1 code.

# Remove the info message from Part 1 and continue with:

# ============================================================================
# DATA VISUALIZATION
# ============================================================================

st.markdown("---")
st.header(f"📈 Market Analysis for {asset}")

# Create tabs for better organization
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Charts", "📋 Metrics", "🧠 AI Insights", "📰 News Sentiment", "💼 Portfolio Simulator"])

# ============================================================================
# TAB 1: CHARTS
# ============================================================================
with tab1:
    st.subheader(f"Price Chart - {info.get('longName', asset)}")
    
    # Chart type selector
    chart_col1, chart_col2 = st.columns([3, 1])
    
    with chart_col2:
        chart_type = st.radio(
            "Chart Type:",
            ["Candlestick", "Line", "Area"],
            key="chart_type_selector"
        )
    
    with chart_col1:
        # Create the appropriate chart based on selection
        if chart_type == "Candlestick":
            fig = go.Figure(data=[go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=asset
            )])
            fig.update_layout(
                title=f"{info.get('longName', asset)} - Candlestick Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
                height=600,
                hovermode='x unified',
                template="plotly_dark"
            )
        
        elif chart_type == "Line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                mode='lines',
                name=f'{asset} Close',
                line={"color": "#4e8df5", "width": 3}
            ))
            fig.update_layout(
                title=f"{info.get('longName', asset)} - Line Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                hovermode='x unified',
                template="plotly_dark"
            )
        
        else:  # Area chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index,
                y=hist['Close'],
                fill='tozeroy',
                name=f'{asset} Close',
                line={"color": "#00d1b2", "width": 2}
            ))
            fig.update_layout(
                title=f"{info.get('longName', asset)} - Area Chart",
                xaxis_title="Date",
                yaxis_title="Price (USD)",
                height=600,
                hovermode='x unified',
                template="plotly_dark"
            )

        # Add Technical Indicators if selected
        if show_sma:
            if 'SMA_50' in hist.columns:
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50', line={"color": "rgba(255, 165, 0, 0.8)", "width": 1.5}))
            if 'SMA_200' in hist.columns:
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], name='SMA 200', line={"color": "rgba(255, 0, 0, 0.8)", "width": 1.5}))
        
        if show_ema:
            if 'EMA_20' in hist.columns:
                fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_20'], name='EMA 20', line={"color": "rgba(160, 32, 240, 0.8)", "width": 1.5}))
        
        # Add Comparison Asset
        if compare_hist is not None:
            # Scale to match the main asset starting price for visual comparison
            base_close = hist['Close'].iloc[0]
            compare_base = compare_hist['Close'].iloc[0]
            scale_factor = base_close / compare_base
            scaled_compare = compare_hist['Close'] * scale_factor
            
            fig.add_trace(go.Scatter(
                x=compare_hist.index,
                y=scaled_compare,
                name=f"{compare_ticker} (Relative)",
                line={"color": "rgba(200, 200, 200, 0.6)", "dash": "dash", "width": 2}
            ))
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Volume chart
    st.subheader("📊 Trading Volume")
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(
        x=hist.index,
        y=hist['Volume'],
        name='Volume',
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    volume_fig.update_layout(
        title="Trading Volume Over Time",
        xaxis_title="Date",
        yaxis_title="Volume",
        height=300,
        hovermode='x unified'
    )
    st.plotly_chart(volume_fig, use_container_width=True)

    # Technical Indicator Charts (RSI & MACD)
    if (show_rsi or show_macd):
        st.markdown("---")
        st.header("📉 Technical Oscillators")
        
        osc_col1, osc_col2 = st.columns(2)
        
        with osc_col1:
            if show_rsi and 'RSI' in hist.columns:
                st.subheader("Relative Strength Index (RSI)")
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(
                    x=hist.index, y=hist['RSI'], 
                    name='RSI', 
                    line={"color": "#4e8df5", "width": 2}
                ))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                rsi_fig.update_layout(
                    height=300, 
                    template="plotly_dark",
                    margin=dict(t=30, b=20, l=20, r=20),
                    yaxis=dict(range=[0, 100])
                )
                st.plotly_chart(rsi_fig, use_container_width=True)
            elif show_rsi:
                st.info("RSI requires more historical data points.")

        with osc_col2:
            if show_macd and 'MACD' in hist.columns:
                st.subheader("MACD (Trend Momentum)")
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD', line={"color": "#4e8df5"}))
                macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal_Line'], name='Signal', line={"color": "#ff4b4b"}))
                macd_fig.add_trace(go.Bar(
                    x=hist.index, y=hist['MACD_Hist'], 
                    name='Histogram',
                    marker_color=['rgba(0, 255, 0, 0.5)' if x > 0 else 'rgba(255, 0, 0, 0.5)' for x in hist['MACD_Hist']]
                ))
                macd_fig.update_layout(
                    height=300, 
                    template="plotly_dark",
                    margin=dict(t=30, b=20, l=20, r=20),
                    showlegend=False
                )
                st.plotly_chart(macd_fig, use_container_width=True)
            elif show_macd:
                st.info("MACD requires more historical data points.")
    
    # Price statistics
    st.subheader("📈 Price Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
    
    with stat_col1:
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        
        st.metric(
            label="Current Price",
            value=f"${current_price:.2f}",
            delta=f"{price_change_pct:.2f}%"
        )
    
    with stat_col2:
        avg_price = hist['Close'].mean()
        st.metric(
            label="Average Price (Period)",
            value=f"${avg_price:.2f}"
        )
    
    with stat_col3:
        max_price = hist['High'].max()
        st.metric(
            label="Highest Price",
            value=f"${max_price:.2f}"
        )
    
    with stat_col4:
        min_price = hist['Low'].min()
        st.metric(
            label="Lowest Price",
            value=f"${min_price:.2f}"
        )

# ============================================================================
# TAB 2: KEY METRICS
# ============================================================================
with tab2:
    st.subheader("🔧 Fundamental Metrics")
    
    # Create comprehensive metrics dictionary
    key_metrics = {
        "Current Price": f"${info.get('currentPrice', 0):.2f}",
        "Previous Close": f"${info.get('previousClose', 0):.2f}",
        "Open": f"${info.get('open', 0):.2f}",
        "Day Range": f"${info.get('dayLow', 0):.2f} - ${info.get('dayHigh', 0):.2f}",
        "52 Week Range": f"${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}",
        "Volume": f"{info.get('volume', 0):,}",
        "Average Volume": f"{info.get('averageVolume', 0):,}",
        "Market Cap": f"${info.get('marketCap', 0):,}",
        "Beta": f"{info.get('beta', 'N/A')}",
        "P/E Ratio": f"{info.get('trailingPE', 'N/A'):.2f}" if info.get('trailingPE') else "N/A",
        "EPS": f"${info.get('trailingEps', 'N/A'):.2f}" if info.get('trailingEps') else "N/A",
        "Dividend Yield": format_dividend_yield(info.get('dividendYield')),
        "52 Week Change": f"{info.get('52WeekChange', 0) * 100:.2f}%" if info.get('52WeekChange') else "N/A"
    }
    
    # Display metrics in organized columns
    col1, col2 = st.columns(2)
    
    metrics_list = list(key_metrics.items())
    mid_point = len(metrics_list) // 2
    
    with col1:
        st.markdown("#### 💰 Price Information")
        for key, value in metrics_list[:7]:
            st.markdown(f"**{key}:** {value}")
    
    with col2:
        st.markdown("#### 📊 Company Fundamentals")
        for key, value in metrics_list[7:]:
            st.markdown(f"**{key}:** {value}")
    
    # Additional company information
    st.markdown("---")
    st.subheader("ℹ️ Company Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"**Company Name:** {info.get('longName', 'N/A')}")
        st.markdown(f"**Sector:** {info.get('sector', 'N/A')}")
        st.markdown(f"**Industry:** {info.get('industry', 'N/A')}")
        st.markdown(f"**Country:** {info.get('country', 'N/A')}")
    
    with info_col2:
        st.markdown(f"**Website:** {info.get('website', 'N/A')}")
        st.markdown(f"**Employees:** {info.get('fullTimeEmployees', 'N/A'):,}" if info.get('fullTimeEmployees') else "**Employees:** N/A")
        st.markdown(f"**Exchange:** {info.get('exchange', 'N/A')}")
        st.markdown(f"**Currency:** {info.get('currency', 'N/A')}")
    
    # Business summary
    if info.get('longBusinessSummary'):
        with st.expander("📄 Business Summary"):
            st.write(info.get('longBusinessSummary'))
    
    # Export metrics as CSV
    st.markdown("---")
    metrics_df = pd.DataFrame(key_metrics.items(), columns=["Metric", "Value"])
    csv_data = metrics_df.to_csv(index=False).encode('utf-8')
    
    st.download_button(
        label="📥 Download Metrics as CSV",
        data=csv_data,
        file_name=f"{asset}_metrics_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
        key="download_metrics_csv"
    )

# ============================================================================
# TAB 3: AI INSIGHTS
# ============================================================================
with tab3:
    st.subheader("🧠 AI-Powered Trading Analysis")
    
    st.info("💡 **Tip:** AI analysis is rate-limited to 3 requests per 5 minutes to ensure quality and prevent abuse.")
    
    # Generate AI Insight Button
    if st.button("🚀 Generate AI Insight", key="generate_insight_btn", type="primary"):
        with st.spinner("🤖 AI is analyzing the data... This may take 10-20 seconds."):
            # Prepare comprehensive technical summary for AI
            rsi_val = hist['RSI'].iloc[-1] if 'RSI' in hist.columns else "N/A"
            macd_val = hist['MACD'].iloc[-1] if 'MACD' in hist.columns else "N/A"
            sma_50 = hist['SMA_50'].iloc[-1] if 'SMA_50' in hist.columns else "N/A"
            sma_200 = hist['SMA_200'].iloc[-1] if 'SMA_200' in hist.columns else "N/A"
            
            prompt = f"""
As a professional AI financial analyst, provide a comprehensive analysis of {asset} ({info.get('longName', 'Unknown Company')}).

Use the following data:
- Current Price: ${info.get('currentPrice', 0):.2f}
- Market Cap: ${info.get('marketCap', 0):,}
- P/E Ratio: {info.get('trailingPE', 'N/A')}
- 52 Week Range: ${info.get('fiftyTwoWeekLow', 0):.2f} - ${info.get('fiftyTwoWeekHigh', 0):.2f}
- RSI (14): {rsi_val}
- MACD: {macd_val}
- SMA 50: {sma_50}
- SMA 200: {sma_200}
- Sector: {info.get('sector', 'N/A')}
- Industry: {info.get('industry', 'N/A')}

Return ONLY a valid JSON object with these exact keys:
{{
  "verdict": "Buy/Hold/Sell",
  "best_for": "Long-term/Short-term/Day Trading/Swing Trading",
  "risk_score": "Low/Medium/High",
  "executive_summary": "A comprehensive 2-3 sentence summary of the investment opportunity incorporating technical indicators",
  "pros": ["Pro 1", "Pro 2", "Pro 3"],
  "cons": ["Con 1", "Con 2", "Con 3"],
  "strategy_suggestions": ["Strategy 1", "Strategy 2", "Strategy 3"],
  "recommendation": "Clear recommendation based on technicals and fundamentals",
  "bottom_line": "Final verdict in one sentence"
}}

Provide actionable, specific insights based on both fundamental and technical data.
"""
            
            insight_data, raw_output = generate_ai_insight(prompt, insight_type="trading_insight")
            
            if insight_data:
                st.session_state.insight_data = insight_data
                st.session_state.raw_output = raw_output
                st.success("✅ AI Analysis Complete!")
            else:
                st.error("❌ Failed to generate insights. Please try again.")
    
    # Display insights if available
    if 'insight_data' in st.session_state and st.session_state.insight_data:
        insight_data = st.session_state.insight_data
        
        st.markdown("---")
        
        # Top-level metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        verdict = insight_data.get("verdict", "N/A")
        verdict_color = {
            "Buy": "🟢",
            "Hold": "🟡",
            "Sell": "🔴"
        }.get(verdict.split()[0], "⚪")
        
        with metric_col1:
            st.metric(
                label="AI Verdict",
                value=f"{verdict_color} {translate_text(verdict, lang)}"
            )
        
        with metric_col2:
            st.metric(
                label="Best Strategy For",
                value=translate_text(insight_data.get("best_for", "N/A"), lang)
            )
        
        with metric_col3:
            risk = insight_data.get("risk_score", "N/A")
            risk_emoji = {
                "Low": "🟢",
                "Medium": "🟡",
                "High": "🔴"
            }.get(risk, "⚪")
            st.metric(
                label="Risk Level",
                value=f"{risk_emoji} {translate_text(risk, lang)}"
            )
        
        # Executive Summary
        st.markdown("### 📝 Executive Summary")
        st.info(translate_text(insight_data.get("executive_summary", ""), lang))
        
        # Pros and Cons
        st.markdown("### ⚖️ Pros & Cons Analysis")
        pros_col, cons_col = st.columns(2)
        
        with pros_col:
            st.markdown("#### ✅ Strengths")
            for pro in insight_data.get("pros", []):
                st.success(f"✓ {translate_text(pro, lang)}")
        
        with cons_col:
            st.markdown("#### ⚠️ Weaknesses")
            for con in insight_data.get("cons", []):
                st.warning(f"✗ {translate_text(con, lang)}")
        
        # Strategy Suggestions
        st.markdown("### 💡 Trading Strategy Suggestions")
        for idx, strategy in enumerate(insight_data.get("strategy_suggestions", []), 1):
            st.markdown(f"**{idx}.** {translate_text(strategy, lang)}")
        
        # Final Recommendation
        st.markdown("### 🎯 Final Recommendation")
        recommendation_text = f"**{translate_text(insight_data.get('recommendation', ''), lang)}**\n\n{translate_text(insight_data.get('bottom_line', ''), lang)}"
        st.success(recommendation_text)
        
        # Raw output expander
        with st.expander("🔍 View Raw AI Response (Technical)"):
            st.code(st.session_state.get('raw_output', 'No raw output available.'), language='json')
    
    else:
        st.info("👆 Click the button above to generate AI-powered insights for this asset.")

# ============================================================================
# TAB 4: NEWS SENTIMENT
# ============================================================================
with tab4:
    st.subheader("📰 News Sentiment Analysis")
    
    st.info("📰 Fetching latest news headlines from Google News...")
    
    with st.spinner("Searching for news..."):
        headlines = fetch_google_news_headlines(asset)
    
    if headlines:
        st.success(f"✅ Found {len(headlines)} recent headlines")
        
        # Display headlines
        st.markdown("### 📑 Latest Headlines")
        for idx, headline in enumerate(headlines, 1):
            st.markdown(f"{idx}. {headline}")
        
        st.markdown("---")
        
        # Analyze sentiment button
        if st.button("🔍 Analyze News Sentiment", key="analyze_sentiment_btn", type="primary"):
            with st.spinner("🤖 AI is analyzing news sentiment..."):
                sentiment_prompt = f"""
Analyze the sentiment of these news headlines for {asset}:

Headlines:
{json.dumps(headlines, indent=2)}

Return ONLY a valid JSON object with these exact keys:
{{
  "sentiment": "Positive/Negative/Neutral/Mixed",
  "confidence": "High/Medium/Low",
  "summary": "A 2-3 sentence analysis of the overall news sentiment and what it means for investors",
  "key_themes": ["Theme 1", "Theme 2", "Theme 3"],
  "outlook": "Short summary of short-term outlook based on news"
}}
"""
                
                sentiment_data, sentiment_raw = generate_ai_insight(sentiment_prompt, insight_type="sentiment")
                
                if sentiment_data:
                    st.session_state.sentiment_data = sentiment_data
                    st.session_state.sentiment_raw = sentiment_raw
                    st.success("✅ Sentiment Analysis Complete!")
                else:
                    st.error("❌ Failed to analyze sentiment. Please try again.")
        
        # Display sentiment analysis if available
        if 'sentiment_data' in st.session_state and st.session_state.sentiment_data:
            sentiment_data = st.session_state.sentiment_data
            
            st.markdown("---")
            st.markdown("### 🎭 Sentiment Analysis Results")
            
            # Sentiment metrics
            sent_col1, sent_col2 = st.columns(2)
            
            with sent_col1:
                sentiment = sentiment_data.get("sentiment", "N/A")
                sentiment_emoji = {
                    "Positive": "😊",
                    "Negative": "😟",
                    "Neutral": "😐",
                    "Mixed": "🤔"
                }.get(sentiment.split()[0], "❓")
                
                st.metric(
                    label="Overall Sentiment",
                    value=f"{sentiment_emoji} {translate_text(sentiment, lang)}"
                )
            
            with sent_col2:
                confidence = sentiment_data.get("confidence", "N/A")
                st.metric(
                    label="Confidence Level",
                    value=translate_text(confidence, lang)
                )
            
            # Summary
            st.markdown("#### 📊 Analysis Summary")
            st.info(translate_text(sentiment_data.get("summary", ""), lang))
            
            # Key themes
            if sentiment_data.get("key_themes"):
                st.markdown("#### 🔑 Key Themes Identified")
                for theme in sentiment_data.get("key_themes", []):
                    st.markdown(f"• {translate_text(theme, lang)}")
            
            # Outlook
            if sentiment_data.get("outlook"):
                st.markdown("#### 🔮 Short-term Outlook")
                st.success(translate_text(sentiment_data.get("outlook", ""), lang))
            
            # Raw output
            with st.expander("🔍 View Raw Sentiment Analysis"):
                st.code(st.session_state.get('sentiment_raw', 'No raw output available.'), language='json')
        
    else:
        st.warning("⚠️ No recent news headlines found for this asset. This might be due to:")
        st.markdown("- Limited news coverage")
        st.markdown("- Network issues")
        st.markdown("- The ticker symbol might be too new or obscure")

# ============================================================================
# TAB 5: PORTFOLIO SIMULATOR
# ============================================================================
with tab5:
    st.subheader("💼 Paper Trading Portfolio Simulator")
    st.markdown("Track your simulated trades and monitor potential P&L based on current market prices.")
    
    # Initialize portfolio in session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []
        
    portfolio_col1, portfolio_col2 = st.columns([1, 2])
    
    with portfolio_col1:
        st.markdown("##### 📥 Add New Position")
        with st.form("portfolio_form"):
            p_asset = st.text_input("Ticker:", value=asset).upper()
            p_qty = st.number_input("Quantity:", min_value=0.01, value=10.0, step=1.0)
            p_buy_price = st.number_input("Buy Price ($):", min_value=0.01, value=float(info.get('currentPrice', 100)), step=0.01)
            
            submit_trade = st.form_submit_button("Add Position to Portfolio")
            
            if submit_trade:
                new_pos = {
                    "asset": p_asset,
                    "qty": p_qty,
                    "buy_price": p_buy_price,
                    "date": datetime.now().strftime("%Y-%m-%d %H:%M")
                }
                st.session_state.portfolio.append(new_pos)
                st.success(f"✅ Added {p_qty} shares of {p_asset} to portfolio!")

    with portfolio_col2:
        st.markdown("##### 💹 Current Holdings")
        if st.session_state.portfolio:
            port_data = []
            total_pl = 0
            
            for i, pos in enumerate(st.session_state.portfolio):
                # Fetch current price using optimized cached function
                if pos['asset'] == asset:
                    curr_p = info.get('currentPrice', pos['buy_price'])
                else:
                    curr_price_fetched = get_current_price_safe(pos['asset'])
                    curr_p = curr_price_fetched if curr_price_fetched is not None else pos['buy_price']
                
                investment = pos['qty'] * pos['buy_price']
                current_val = pos['qty'] * curr_p
                pl = current_val - investment
                pl_pct = (pl / investment) * 100 if investment != 0 else 0
                total_pl += pl
                
                port_data.append({
                    "Date": pos['date'],
                    "Asset": pos['asset'],
                    "Qty": pos['qty'],
                    "Buy Price": f"${pos['buy_price']:.2f}",
                    "Current Price": f"${curr_p:.2f}",
                    "P&L ($)": f"${pl:.2f}",
                    "P&L (%)": f"{pl_pct:.2f}%"
                })
            
            st.table(pd.DataFrame(port_data))
            
            # Summary Metrics
            st.markdown("---")
            total_inv = sum(p['qty']*p['buy_price'] for p in st.session_state.portfolio)
            total_pl_pct = (total_pl / total_inv * 100) if total_inv != 0 else 0
            
            sum_col1, sum_col2 = st.columns(2)
            with sum_col1:
                st.metric("Total Portfolio P&L", value=f"${total_pl:.2f}", delta=f"{total_pl_pct:.2f}%")
            
            with sum_col2:
                if st.button("🗑️ Clear Portfolio"):
                    st.session_state.portfolio = []
                    st.rerun()
        else:
            st.info("Your portfolio is currently empty. Add a position using the form on the left.")

# ============================================================================
# EXPORT SECTION
# ============================================================================

st.markdown("---")
st.header("📤 Export & Download Reports")

export_col1, export_col2 = st.columns(2)

with export_col1:
    st.markdown("### 📄 PDF Report")
    st.markdown("Download a comprehensive PDF report including all metrics, AI insights, and sentiment analysis.")
    
    if st.button("📄 Generate PDF Report", key="generate_pdf_btn"):
        # Check if we have the necessary data
        has_insight = 'insight_data' in st.session_state and st.session_state.insight_data
        has_sentiment = 'sentiment_data' in st.session_state and st.session_state.sentiment_data
        
        if not has_insight and not has_sentiment:
            st.warning("⚠️ Please generate AI insights and sentiment analysis first to create a complete report.")
        else:
            with st.spinner("📝 Generating PDF report..."):
                pdf_data = create_pdf_report(
                    asset,
                    key_metrics,
                    st.session_state.get('insight_data'),
                    st.session_state.get('sentiment_data')
                )
                
                if pdf_data:
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=pdf_data,
                        file_name=f"{asset}_AI_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        key="download_pdf_btn"
                    )
                    st.success("✅ PDF report generated successfully!")
                else:
                    st.error("❌ Failed to generate PDF report.")

with export_col2:
    st.markdown("### 📊 Excel Data Export")
    st.markdown("Export historical price data and metrics to Excel format for further analysis.")
    
    # Create Excel file with multiple sheets
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # Historical data sheet
        hist_export = hist.copy()
        hist_export.index.name = 'Date'
        # Strip timezone info as xlsxwriter doesn't support it
        if hist_export.index.tz is not None:
            hist_export.index = hist_export.index.tz_localize(None)
        hist_export.to_excel(writer, sheet_name='Historical Data')
        
        # Metrics sheet
        metrics_df.to_excel(writer, sheet_name='Key Metrics', index=False)
        
        # Add AI insights if available
        if 'insight_data' in st.session_state and st.session_state.insight_data:
            insight_df = pd.DataFrame([st.session_state.insight_data])
            insight_df.to_excel(writer, sheet_name='AI Insights', index=False)
    
    excel_data = output.getvalue()
    
    st.download_button(
        label="📥 Download Excel File",
        data=excel_data,
        file_name=f"{asset}_complete_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_excel_btn"
    )

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 20px;'>
    <p>📊 <b>AI Trading Dashboard</b> | Powered by Groq AI & Yahoo Finance</p>
    <p style='font-size: 12px;'>⚠️ Disclaimer: This tool is for informational purposes only. Not financial advice. Always do your own research.</p>
    <p style='font-size: 12px;'>Generated on {}</p>
</div>
""".format(datetime.now().strftime('%Y-%m-%d %H:%M:%S')), unsafe_allow_html=True)