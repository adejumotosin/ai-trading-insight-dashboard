import os
import time
import json
import re
import streamlit as st
import pandas as pd
import yfinance as yf

# FIX 1: Replace appdirs monkey-patch with proper env var
# The old approach broke yfinance's cookie/crumb cache
os.environ["XDG_CACHE_HOME"] = "/tmp"

import plotly.graph_objects as go
from datetime import datetime, timedelta
from io import BytesIO
from deep_translator import GoogleTranslator
from fpdf import FPDF
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
try:
    from groq import Groq
except ImportError as e:
    st.error(f"❌ Groq import error: {e}")
    st.info("If you are on Streamlit Cloud, please wait for 'Processing dependencies...' to finish or click 'Reboot App'.")
    st.stop()
import hashlib
import hmac

# ============================================================================
# SESSION SETUP FOR API RESILIENCE
# ============================================================================
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)

def get_yf_session():
    """Setup a requests session with retries and browser headers for yfinance."""
    session = requests.Session()
    session.headers.update({
        'User-Agent': _BROWSER_UA,
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
    })
    retry = Retry(
        total=5,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount('https://', adapter)
    session.mount('http://', adapter)
    return session

yf_session = get_yf_session()

# ============================================================================
# GLOBAL REQUEST THROTTLE - prevents burst requests to Yahoo Finance
# ============================================================================
_last_yf_call_time = 0.0
_YF_MIN_INTERVAL = 2.0  # minimum seconds between yfinance API calls

def throttled_ticker(ticker_symbol):
    """
    Create a yf.Ticker with global throttling to avoid rate limits.
    Ensures at least _YF_MIN_INTERVAL seconds between API calls.
    """
    global _last_yf_call_time
    elapsed = time.time() - _last_yf_call_time
    if elapsed < _YF_MIN_INTERVAL:
        time.sleep(_YF_MIN_INTERVAL - elapsed)
    _last_yf_call_time = time.time()
    return yf.Ticker(ticker_symbol, session=yf_session)

# ============================================================================
# DISK CACHE - serves stale data when Yahoo Finance is rate-limited
# ============================================================================
DISK_CACHE_DIR = "/tmp/yf_cache"
os.makedirs(DISK_CACHE_DIR, exist_ok=True)

def _disk_cache_path(key):
    """Get the file path for a disk cache entry."""
    safe_key = key.replace('/', '_').replace('\\', '_')
    return os.path.join(DISK_CACHE_DIR, f"{safe_key}.json")

def save_to_disk_cache(key, info_dict, hist_df):
    """Persist successful API response to disk for fallback use."""
    try:
        cache_data = {
            "info": {k: v for k, v in (info_dict or {}).items()
                     if isinstance(v, (str, int, float, bool, type(None)))},
            "timestamp": time.time()
        }
        if hist_df is not None and not hist_df.empty:
            cols = [c for c in ['Open', 'High', 'Low', 'Close', 'Volume'] if c in hist_df.columns]
            cache_data["hist"] = hist_df[cols].tail(260).to_json()
        with open(_disk_cache_path(key), 'w') as f:
            json.dump(cache_data, f)
    except Exception:
        pass  # best-effort caching

def load_from_disk_cache(key):
    """Load previously cached data from disk. Returns (info, hist, age_minutes) or None."""
    try:
        path = _disk_cache_path(key)
        if not os.path.exists(path):
            return None
        with open(path, 'r') as f:
            cache_data = json.load(f)
        age_minutes = (time.time() - cache_data.get("timestamp", 0)) / 60
        info = cache_data.get("info", {})
        hist = None
        if "hist" in cache_data:
            hist = pd.read_json(cache_data["hist"])
        return info, hist, age_minutes
    except Exception:
        return None

# ============================================================================
# STOOQ FALLBACK - free historical data source (no API key required)
# ============================================================================
def generate_simulated_data(ticker_symbol):
    """Generate realistic-looking fake data when all APIs fail."""
    import numpy as np
    
    # Deterministic "price" based on ticker letters
    base_price = sum(ord(c) for c in ticker_symbol) % 500 + 50
    
    # Create 260 days of history
    dates = pd.date_range(end=datetime.now(), periods=260)
    
    # Random walk
    changes = np.random.normal(0, 0.02, 260)
    prices = base_price * (1 + changes).cumprod()
    
    hist = pd.DataFrame({
        'Open': prices * 0.995,
        'High': prices * 1.01,
        'Low': prices * 0.99,
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, 260)
    }, index=dates)
    
    info = {
        'currentPrice': float(prices[-1]),
        'regularMarketPrice': float(prices[-1]),
        'previousClose': float(prices[-2]),
        'open': float(prices[-1] * 0.998),
        'dayLow': float(prices[-1] * 0.985),
        'dayHigh': float(prices[-1] * 1.012),
        'fiftyTwoWeekLow': float(prices.min()),
        'fiftyTwoWeekHigh': float(prices.max()),
        'volume': int(hist['Volume'].iloc[-1]),
        'averageVolume': int(hist['Volume'].mean()),
        'longName': f"{ticker_symbol} (SIMULATED)",
        'symbol': ticker_symbol,
        'note': '⚠️ Using Simulated Data (APIs Offline)'
    }
    
    return info, hist

# ============================================================================
# STOOQ FALLBACK - free historical data source (no API key required)
# ============================================================================
@st.cache_data(ttl=7200, show_spinner=False)
def fetch_stooq_history(ticker_symbol):
    """
    Fetch historical OHLCV data from Stooq.com as a fallback when Yahoo is blocked.
    """
    try:
        from io import StringIO
        stooq_ticker = ticker_symbol.replace('-', '.').upper()
        # Common US stocks usually end in .US on Stooq
        for suffix in ['.US', '']:
            url = f"https://stooq.com/q/d/l/?s={stooq_ticker}{suffix}&i=d"
            response = requests.get(url, headers={'User-Agent': _BROWSER_UA}, timeout=10)
            if response.status_code == 200 and len(response.text) > 100:
                df = pd.read_csv(StringIO(response.text))
                if not df.empty and 'Close' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df = df.set_index('Date').sort_index()
                    return df
        return None
    except Exception as e:
        st.sidebar.error(f"Stooq Fallback Error: {e}")
        return None

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

# FIX 2: Cold-start pause to avoid hammering Yahoo Finance on app boot
if 'app_started' not in st.session_state:
    st.session_state.app_started = True
    time.sleep(1.5)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_ticker(ticker):
    """Validate ticker symbol format."""
    if not ticker:
        return False, "Ticker cannot be empty"
    if not re.match(r'^[A-Z0-9]{1,5}(-[A-Z0-9]{1,5})?$', ticker):
        return False, "Invalid ticker format (e.g., AAPL, BTC-USD)"
    return True, ""


def check_rate_limit(key, max_calls=5, window_minutes=5):
    """Simple rate limiting using session state."""
    if 'rate_limits' not in st.session_state:
        st.session_state.rate_limits = {}

    now = datetime.now()
    if key not in st.session_state.rate_limits:
        st.session_state.rate_limits[key] = []

    st.session_state.rate_limits[key] = [
        ts for ts in st.session_state.rate_limits[key]
        if now - ts < timedelta(minutes=window_minutes)
    ]

    if len(st.session_state.rate_limits[key]) >= max_calls:
        oldest = st.session_state.rate_limits[key][0]
        wait_time = int((oldest + timedelta(minutes=window_minutes) - now).total_seconds())
        return False, wait_time

    st.session_state.rate_limits[key].append(now)
    return True, 0


def translate_text(text, dest_lang):
    """Translates text to the destination language."""
    if dest_lang.lower() == "english" or not text:
        return text
    try:
        lang_code = dest_lang.lower()[:2]
        return GoogleTranslator(source='auto', target=lang_code).translate(text)
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text


def format_dividend_yield(div_yield):
    """Safely format dividend yield with proper detection."""
    if div_yield is None or pd.isna(div_yield):
        return "N/A"
    try:
        div_yield = float(div_yield)
        if 0 < div_yield < 1:
            return f"{div_yield:.2%}"
        elif 1 <= div_yield < 100:
            return f"{div_yield / 100:.2%}"
        else:
            return "N/A"
    except (ValueError, TypeError):
        return "N/A"


def parse_ai_response(response_text):
    """Safely extract and validate JSON from AI response."""
    if not response_text:
        return None
    try:
        data = json.loads(response_text)
        return data
    except json.JSONDecodeError:
        patterns = [
            r"```json\s*(\{.*?\})\s*```",
            r"```\s*(\{.*?\})\s*```",
            r"(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})"
        ]
        for pattern in patterns:
            match = re.search(pattern, response_text, re.DOTALL)
            if match:
                try:
                    data = json.loads(match.group(1))
                    if isinstance(data, dict):
                        return data
                except json.JSONDecodeError:
                    continue
        return None


# ============================================================================
# TECHNICAL ANALYSIS ENGINE
# ============================================================================

def calculate_indicators(df):
    """Calculate various technical indicators for the provided historical data."""
    if df is None or df.empty:
        return df

    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()

    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['Signal_Line']

    return df


def safe_float(val, default=0.0):
    try:
        return float(val) if val is not None else default
    except (ValueError, TypeError):
        return default

def safe_int(val, default=0):
    try:
        return int(val) if val is not None else default
    except (ValueError, TypeError):
        return default

# ============================================================================
# DATA FETCHING FUNCTIONS — SUCCESS-ONLY CACHING
# ============================================================================

def _cache_key(prefix, ticker, extra=""):
    return f"_yf_{prefix}_{ticker}_{extra}"

def _get_from_session_cache(key, max_age_seconds=1800):
    """Retrieve data from session-state cache if fresh enough."""
    if key in st.session_state:
        entry = st.session_state[key]
        if time.time() - entry.get("ts", 0) < max_age_seconds:
            return entry.get("data")
    return None

def _set_session_cache(key, data):
    """Store successful data in session-state cache."""
    st.session_state[key] = {"data": data, "ts": time.time()}


# FIX 3: Improved history fetcher — tries yf.download first (more reliable),
# then falls back to Ticker.history
def _try_yahoo_history(ticker_symbol, period="1y"):
    """Try multiple yfinance methods to fetch history. Returns DataFrame or None."""
    global _last_yf_call_time

    # Method 1: yf.download — uses a different internal code path, often works
    # when Ticker.history is blocked
    try:
        elapsed = time.time() - _last_yf_call_time
        if elapsed < _YF_MIN_INTERVAL:
            time.sleep(_YF_MIN_INTERVAL - elapsed)
        _last_yf_call_time = time.time()

        df = yf.download(
            ticker_symbol,
            period=period,
            progress=False,
            session=yf_session,
            auto_adjust=True
        )
        if df is not None and not df.empty:
            # Flatten MultiIndex columns that yf.download sometimes returns
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            return df
    except Exception:
        pass

    # Method 2: Ticker.history fallback
    try:
        ticker = throttled_ticker(ticker_symbol)
        hist = ticker.history(period=period)
        if hist is not None and not hist.empty:
            return hist
    except Exception:
        pass

    return None


def _try_yahoo_info(ticker_symbol):
    """Try to fetch info from Yahoo Finance. Returns dict or None."""
    try:
        ticker = throttled_ticker(ticker_symbol)
        info = ticker.info
        if info and isinstance(info, dict) and len(info) > 2:
            return info
    except Exception:
        pass

    # Lighter fallback via fast_info
    try:
        ticker = throttled_ticker(ticker_symbol)
        fi = ticker.fast_info
        return {
            'currentPrice': getattr(fi, 'last_price', None),
            'previousClose': getattr(fi, 'previous_close', None),
            'open': getattr(fi, 'open', None),
            'dayLow': getattr(fi, 'day_low', None),
            'dayHigh': getattr(fi, 'day_high', None),
            'fiftyTwoWeekLow': getattr(fi, 'year_low', None),
            'fiftyTwoWeekHigh': getattr(fi, 'year_high', None),
            'marketCap': getattr(fi, 'market_cap', None),
            'volume': getattr(fi, 'last_volume', None),
            'symbol': ticker_symbol,
            'longName': ticker_symbol
        }
    except Exception:
        return None


def _build_info_from_history(ticker_symbol, hist):
    """Construct a complete info dict from history data alone."""
    if hist is None or hist.empty:
        return {}
    last_row = hist.iloc[-1]
    prev_row = hist.iloc[-2] if len(hist) > 1 else last_row
    return {
        'currentPrice': float(last_row['Close']),
        'regularMarketPrice': float(last_row['Close']),
        'previousClose': float(prev_row['Close']),
        'open': float(last_row['Open']),
        'dayLow': float(last_row['Low']),
        'dayHigh': float(last_row['High']),
        'fiftyTwoWeekLow': float(hist['Low'].min()),
        'fiftyTwoWeekHigh': float(hist['High'].max()),
        'volume': int(last_row['Volume']),
        'averageVolume': int(hist['Volume'].mean()),
        'longName': ticker_symbol,
        'symbol': ticker_symbol
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_current_price_safe(ticker_symbol):
    """Price fetcher using history(1d). Cache for 15 minutes."""
    try:
        ticker = throttled_ticker(ticker_symbol)
        hist = ticker.history(period="1d")
        if not hist.empty:
            return float(hist['Close'].iloc[-1])
        return None
    except Exception:
        return None

@st.cache_data(ttl=900, show_spinner=False)
def get_portfolio_prices(tickers_tuple):
    """Fetch current prices for multiple tickers in ONE API call."""
    if not tickers_tuple:
        return {}
    tickers_list = list(tickers_tuple)
    try:
        global _last_yf_call_time
        elapsed = time.time() - _last_yf_call_time
        if elapsed < _YF_MIN_INTERVAL:
            time.sleep(_YF_MIN_INTERVAL - elapsed)
        _last_yf_call_time = time.time()

        if len(tickers_list) == 1:
            data = yf.download(tickers_list, period="1d", session=yf_session, progress=False)
            if data is not None and not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                return {tickers_list[0]: float(data['Close'].iloc[-1])}
            return {}

        data = yf.download(tickers_list, period="1d", group_by='ticker', session=yf_session, progress=False)
        prices = {}
        for t in tickers_list:
            try:
                prices[t] = float(data[t]['Close'].iloc[-1])
            except Exception:
                prices[t] = None
        return prices
    except Exception:
        return {}


def get_stock_data_safe(ticker_symbol, demo_mode=False):
    """
    Fetch stock data with multi-source resilience.
    Priority: Session cache → Fresh disk cache (< 30 min) → Yahoo Finance
              → Stooq → Stale disk cache (any age) → Simulated Data (if demo_mode)
    Only successful results are cached (failures are NEVER cached).
    """
    # === 0. DEMO MODE (Instant simulation) ===
    if demo_mode:
        return generate_simulated_data(ticker_symbol) + (None,)

    # === 1. CHECK SESSION CACHE (instant, no API call) ===
    cache_key = _cache_key("data", ticker_symbol)
    cached = _get_from_session_cache(cache_key, max_age_seconds=1800)
    if cached:
        return cached

    # === 2. CHECK FRESH DISK CACHE (serve without API calls if < 30 min old) ===
    disk_cached = load_from_disk_cache(ticker_symbol)
    if disk_cached:
        cached_info, cached_hist, age_min = disk_cached
        if cached_info and cached_info.get('currentPrice') and age_min < 30:
            if cached_hist is not None and not cached_hist.empty:
                cached_hist = calculate_indicators(cached_hist)
            result = (cached_info, cached_hist, None)
            _set_session_cache(cache_key, result)
            return result

    hist = None
    info = None
    data_source = None

    # === 3. TRY YAHOO FINANCE (download + ticker) ===
    yahoo_hist = _try_yahoo_history(ticker_symbol)
    if yahoo_hist is not None and not yahoo_hist.empty:
        hist = yahoo_hist
        data_source = "yahoo"

    yahoo_info = _try_yahoo_info(ticker_symbol)
    if yahoo_info:
        info = yahoo_info

    # === 4. TRY STOOQ FALLBACK (if Yahoo history failed) ===
    if hist is None:
        stooq_hist = fetch_stooq_history(ticker_symbol)
        if stooq_hist is not None and not stooq_hist.empty:
            hist = stooq_hist
            data_source = "stooq"

    # === 5. BUILD INFO FROM HISTORY (if Yahoo info failed but we have history) ===
    if hist is not None and not hist.empty:
        history_info = _build_info_from_history(ticker_symbol, hist)
        if info:
            for k, v in history_info.items():
                if info.get(k) is None or info.get(k) == 0:
                    info[k] = v
        else:
            info = history_info

    # === 6. CHECK IF WE HAVE USABLE DATA ===
    if info and ('currentPrice' in info or 'regularMarketPrice' in info):
        if hist is not None and not hist.empty:
            hist = calculate_indicators(hist)
        save_to_disk_cache(ticker_symbol, info, hist)

        warning = None
        if data_source == "stooq":
            warning = "⚠️ Using Stooq data (Yahoo Finance unavailable). Some features may be limited."
        result = (info, hist, warning)
        _set_session_cache(cache_key, result)
        return result

    # === 7. STALE DISK CACHE FALLBACK (any age, last resort) ===
    if disk_cached:
        cached_info, cached_hist, age_min = disk_cached
        if cached_info and cached_info.get('currentPrice'):
            if cached_hist is not None and not cached_hist.empty:
                cached_hist = calculate_indicators(cached_hist)
            stale_warning = f"⚠️ Using cached data from {age_min:.0f} minutes ago (Yahoo Finance rate-limited). Click 'Refresh' later."
            return cached_info, cached_hist, stale_warning

    return None, None, "⚠️ Could not fetch data from any source. Yahoo Finance is rate-limited and no cached data is available. Please try again in a few minutes."


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_google_news_headlines(asset):
    """Fetch recent news headlines from Google News RSS feed."""
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
    """Generates insights using the Groq model and parses the JSON response."""
    can_proceed, wait_time = check_rate_limit(f'groq_{insight_type}', max_calls=5, window_minutes=5)
    if not can_proceed:
        st.error(f"⏳ Rate limit reached. Please wait {wait_time} seconds before trying again.")
        return None, None
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.3-70b-versatile",
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
    """Creates a PDF report from the collected data."""
    try:
        pdf = FPDF()
        pdf.add_page()

        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, f"{asset} AI Trading Report", ln=True, align="C")
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 14)
        pdf.cell(0, 10, "Key Metrics", ln=True)
        pdf.set_font("Arial", size=11)
        for key, value in metrics.items():
            value_str = str(value)[:100]
            pdf.cell(0, 8, txt=f"{key}: {value_str}", ln=True)

        if insight:
            pdf.ln(5)
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(0, 10, "AI Analysis", ln=True)

            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Executive Summary:", ln=True)
            pdf.set_font("Arial", size=10)
            summary = insight.get("executive_summary", "N/A")[:500]
            pdf.multi_cell(0, 5, txt=summary)

            pdf.ln(3)
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 8, "Recommendation:", ln=True)
            pdf.set_font("Arial", size=10)
            recommendation = insight.get("bottom_line", "N/A")[:300]
            pdf.multi_cell(0, 5, txt=recommendation)

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

        pdf.ln(10)
        pdf.set_font("Arial", 'I', 8)
        pdf.cell(0, 5, f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

        return pdf.output(dest='S').encode('latin1', errors='replace')
    except Exception as e:
        st.error(f"PDF generation failed: {e}")
        return None


# ============================================================================
# AUTHENTICATION
# ============================================================================

def verify_user_simple(email):
    """Simple email-based verification."""
    try:
        allowed_users = st.secrets.get("allowed_users", [])
    except Exception:
        allowed_users = ["your@email.com", "admin@gmail.com"]
    return email.lower().strip() in [u.lower().strip() for u in allowed_users]


# ============================================================================
# MAIN APPLICATION - SETUP & AUTHENTICATION
# ============================================================================

st.title("📊 AI-Powered Trading & Market Insight Dashboard")

# Sidebar - Authentication
st.sidebar.header("🔐 User Authentication")

if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    # Also clear session-state data cache entries
    keys_to_del = [k for k in st.session_state if k.startswith("_yf_")]
    for k in keys_to_del:
        del st.session_state[k]
    st.success("✅ Cache cleared!")

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

lang = st.sidebar.selectbox(
    "🌍 Choose Output Language",
    ["English", "French", "Spanish", "German", "Chinese"],
    key="language_selector"
)

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Asset Selection")

# Demo Mode Toggle
demo_mode = st.sidebar.toggle(
    "🧪 Demo Mode (Simulated Data)", 
    value=st.session_state.get('demo_mode_active', False),
    help="Enable this to test the dashboard with simulated data if Yahoo Finance is blocked."
)
st.session_state.demo_mode_active = demo_mode

asset_input = st.sidebar.text_input(
    "Enter Ticker Symbol:",
    value="AAPL",
    help="Examples: AAPL, MSFT, GOOGL, BTC-USD, ETH-USD"
).upper()

is_valid, error_msg = validate_ticker(asset_input)
if not is_valid:
    st.sidebar.error(f"❌ {error_msg}")
    st.stop()

asset = asset_input
st.sidebar.success(f"✅ Analyzing: **{asset}**")

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
    index=3
)

# Fetch stock data
with st.spinner(f"📊 Fetching data for {asset}..."):
    info, hist, fetch_warning = get_stock_data_safe(asset, demo_mode=demo_mode)

# Ultimate fallback: if everything failed, offer Demo Mode
if info is None or hist is None:
    st.error(
        f"❌ Could not retrieve data for **{asset}** from Yahoo Finance or Stooq. "
        "This usually happens due to temporary API rate-limits."
    )
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔄 Try Again"):
            st.rerun()
    with col2:
        if st.button("🧪 Switch to Demo Mode"):
            st.session_state.demo_mode_active = True
            st.rerun()
            
    st.info("💡 **Tip:** Click 'Clear Cache' in the sidebar if you recently updated your settings.")
    st.stop()

if fetch_warning:
    if "cached data" in fetch_warning or "Stooq" in fetch_warning:
        st.warning(fetch_warning)
    else:
        st.error(fetch_warning)

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

if st.sidebar.button("🔄 Refresh Data", use_container_width=True):
    st.cache_data.clear()
    keys_to_del = [k for k in st.session_state if k.startswith("_yf_")]
    for k in keys_to_del:
        del st.session_state[k]
    st.rerun()

# Fetch comparison data
compare_hist = None
if compare_ticker:
    with st.spinner(f"⚖️ Fetching comparison data for {compare_ticker}..."):
        _, compare_hist, compare_error = get_stock_data_safe(compare_ticker, demo_mode=demo_mode)
        if compare_error and "rate-limited" in str(compare_error):
            st.sidebar.warning(f"⚠️ Comparison error: {compare_error}")
            compare_hist = None

st.success(f"✅ Successfully loaded data for **{info.get('longName', asset)}**")

# ============================================================================
# DATA VISUALIZATION
# ============================================================================

st.markdown("---")
st.header(f"📈 Market Analysis for {asset}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Charts", "📋 Metrics", "🧠 AI Insights", "📰 News Sentiment", "💼 Portfolio Simulator"])

# ============================================================================
# TAB 1: CHARTS
# ============================================================================
with tab1:
    st.subheader(f"Price Chart - {info.get('longName', asset)}")

    chart_col1, chart_col2 = st.columns([3, 1])

    with chart_col2:
        chart_type = st.radio(
            "Chart Type:",
            ["Candlestick", "Line", "Area"],
            key="chart_type_selector"
        )

    with chart_col1:
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
                xaxis_title="Date", yaxis_title="Price (USD)",
                xaxis_rangeslider_visible=False,
                height=600, hovermode='x unified', template="plotly_dark"
            )
        elif chart_type == "Line":
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Close'], mode='lines',
                name=f'{asset} Close', line={"color": "#4e8df5", "width": 3}
            ))
            fig.update_layout(
                title=f"{info.get('longName', asset)} - Line Chart",
                xaxis_title="Date", yaxis_title="Price (USD)",
                height=600, hovermode='x unified', template="plotly_dark"
            )
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=hist.index, y=hist['Close'], fill='tozeroy',
                name=f'{asset} Close', line={"color": "#00d1b2", "width": 2}
            ))
            fig.update_layout(
                title=f"{info.get('longName', asset)} - Area Chart",
                xaxis_title="Date", yaxis_title="Price (USD)",
                height=600, hovermode='x unified', template="plotly_dark"
            )

        if show_sma:
            if 'SMA_50' in hist.columns:
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_50'], name='SMA 50',
                                         line={"color": "rgba(255, 165, 0, 0.8)", "width": 1.5}))
            if 'SMA_200' in hist.columns:
                fig.add_trace(go.Scatter(x=hist.index, y=hist['SMA_200'], name='SMA 200',
                                         line={"color": "rgba(255, 0, 0, 0.8)", "width": 1.5}))

        if show_ema and 'EMA_20' in hist.columns:
            fig.add_trace(go.Scatter(x=hist.index, y=hist['EMA_20'], name='EMA 20',
                                     line={"color": "rgba(160, 32, 240, 0.8)", "width": 1.5}))

        if compare_hist is not None:
            base_close = hist['Close'].iloc[0]
            compare_base = compare_hist['Close'].iloc[0]
            scale_factor = base_close / compare_base
            scaled_compare = compare_hist['Close'] * scale_factor
            fig.add_trace(go.Scatter(
                x=compare_hist.index, y=scaled_compare,
                name=f"{compare_ticker} (Relative)",
                line={"color": "rgba(200, 200, 200, 0.6)", "dash": "dash", "width": 2}
            ))

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📊 Trading Volume")
    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(
        x=hist.index, y=hist['Volume'], name='Volume',
        marker_color='rgba(55, 128, 191, 0.7)'
    ))
    volume_fig.update_layout(
        title="Trading Volume Over Time",
        xaxis_title="Date", yaxis_title="Volume",
        height=300, hovermode='x unified'
    )
    st.plotly_chart(volume_fig, use_container_width=True)

    if show_rsi or show_macd:
        st.markdown("---")
        st.header("📉 Technical Oscillators")
        osc_col1, osc_col2 = st.columns(2)

        with osc_col1:
            if show_rsi and 'RSI' in hist.columns:
                st.subheader("Relative Strength Index (RSI)")
                rsi_fig = go.Figure()
                rsi_fig.add_trace(go.Scatter(x=hist.index, y=hist['RSI'], name='RSI',
                                              line={"color": "#4e8df5", "width": 2}))
                rsi_fig.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                rsi_fig.update_layout(height=300, template="plotly_dark",
                                       margin=dict(t=30, b=20, l=20, r=20), yaxis=dict(range=[0, 100]))
                st.plotly_chart(rsi_fig, use_container_width=True)
            elif show_rsi:
                st.info("RSI requires more historical data points.")

        with osc_col2:
            if show_macd and 'MACD' in hist.columns:
                st.subheader("MACD (Trend Momentum)")
                macd_fig = go.Figure()
                macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['MACD'], name='MACD',
                                               line={"color": "#4e8df5"}))
                macd_fig.add_trace(go.Scatter(x=hist.index, y=hist['Signal_Line'], name='Signal',
                                               line={"color": "#ff4b4b"}))
                macd_fig.add_trace(go.Bar(
                    x=hist.index, y=hist['MACD_Hist'], name='Histogram',
                    marker_color=['rgba(0, 255, 0, 0.5)' if x > 0 else 'rgba(255, 0, 0, 0.5)'
                                  for x in hist['MACD_Hist']]
                ))
                macd_fig.update_layout(height=300, template="plotly_dark",
                                        margin=dict(t=30, b=20, l=20, r=20), showlegend=False)
                st.plotly_chart(macd_fig, use_container_width=True)
            elif show_macd:
                st.info("MACD requires more historical data points.")

    st.subheader("📈 Price Statistics")
    stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)

    with stat_col1:
        current_price = hist['Close'].iloc[-1]
        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
        price_change = current_price - prev_price
        price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
        st.metric(label="Current Price", value=f"${current_price:.2f}", delta=f"{price_change_pct:.2f}%")

    with stat_col2:
        st.metric(label="Average Price (Period)", value=f"${hist['Close'].mean():.2f}")

    with stat_col3:
        st.metric(label="Highest Price", value=f"${hist['High'].max():.2f}")

    with stat_col4:
        st.metric(label="Lowest Price", value=f"${hist['Low'].min():.2f}")

# ============================================================================
# TAB 2: KEY METRICS
# ============================================================================
with tab2:
    st.subheader("🔧 Fundamental Metrics")

    key_metrics = {
        "Current Price": f"${safe_float(info.get('currentPrice')):.2f}",
        "Previous Close": f"${safe_float(info.get('previousClose')):.2f}",
        "Open": f"${safe_float(info.get('open')):.2f}",
        "Day Range": f"${safe_float(info.get('dayLow')):.2f} - ${safe_float(info.get('dayHigh')):.2f}",
        "52 Week Range": f"${safe_float(info.get('fiftyTwoWeekLow')):.2f} - ${safe_float(info.get('fiftyTwoWeekHigh')):.2f}",
        "Volume": f"{safe_int(info.get('volume')):,}",
        "Average Volume": f"{safe_int(info.get('averageVolume')):,}",
        "Market Cap": f"${safe_int(info.get('marketCap')):,}",
        "Beta": f"{info.get('beta', 'N/A')}",
        "P/E Ratio": f"{safe_float(info.get('trailingPE')):.2f}" if info.get('trailingPE') else "N/A",
        "EPS": f"${safe_float(info.get('trailingEps')):.2f}" if info.get('trailingEps') else "N/A",
        "Dividend Yield": format_dividend_yield(info.get('dividendYield')),
        "52 Week Change": f"{safe_float(info.get('52WeekChange')) * 100:.2f}%" if info.get('52WeekChange') else "N/A"
    }

    col1, col2 = st.columns(2)
    metrics_list = list(key_metrics.items())

    with col1:
        st.markdown("#### 💰 Price Information")
        for key, value in metrics_list[:7]:
            st.markdown(f"**{key}:** {value}")

    with col2:
        st.markdown("#### 📊 Company Fundamentals")
        for key, value in metrics_list[7:]:
            st.markdown(f"**{key}:** {value}")

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

    if info.get('longBusinessSummary'):
        with st.expander("📄 Business Summary"):
            st.write(info.get('longBusinessSummary'))

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
    st.info("💡 **Tip:** AI analysis is rate-limited to 5 requests per 5 minutes.")

    if st.button("🚀 Generate AI Insight", key="generate_insight_btn", type="primary"):
        with st.spinner("🤖 AI is analyzing the data... This may take 10-20 seconds."):
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
"""
            insight_data, raw_output = generate_ai_insight(prompt, insight_type="trading_insight")
            if insight_data:
                st.session_state.insight_data = insight_data
                st.session_state.raw_output = raw_output
                st.success("✅ AI Analysis Complete!")
            else:
                st.error("❌ Failed to generate insights. Please try again.")

    if 'insight_data' in st.session_state and st.session_state.insight_data:
        insight_data = st.session_state.insight_data
        st.markdown("---")

        metric_col1, metric_col2, metric_col3 = st.columns(3)
        verdict = insight_data.get("verdict", "N/A")
        verdict_color = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}.get(verdict.split()[0], "⚪")

        with metric_col1:
            st.metric(label="AI Verdict", value=f"{verdict_color} {translate_text(verdict, lang)}")
        with metric_col2:
            st.metric(label="Best Strategy For", value=translate_text(insight_data.get("best_for", "N/A"), lang))
        with metric_col3:
            risk = insight_data.get("risk_score", "N/A")
            risk_emoji = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}.get(risk, "⚪")
            st.metric(label="Risk Level", value=f"{risk_emoji} {translate_text(risk, lang)}")

        st.markdown("### 📝 Executive Summary")
        st.info(translate_text(insight_data.get("executive_summary", ""), lang))

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

        st.markdown("### 💡 Trading Strategy Suggestions")
        for idx, strategy in enumerate(insight_data.get("strategy_suggestions", []), 1):
            st.markdown(f"**{idx}.** {translate_text(strategy, lang)}")

        st.markdown("### 🎯 Final Recommendation")
        recommendation_text = (
            f"**{translate_text(insight_data.get('recommendation', ''), lang)}**\n\n"
            f"{translate_text(insight_data.get('bottom_line', ''), lang)}"
        )
        st.success(recommendation_text)

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
        st.markdown("### 📑 Latest Headlines")
        for idx, headline in enumerate(headlines, 1):
            st.markdown(f"{idx}. {headline}")

        st.markdown("---")

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

        if 'sentiment_data' in st.session_state and st.session_state.sentiment_data:
            sentiment_data = st.session_state.sentiment_data
            st.markdown("---")
            st.markdown("### 🎭 Sentiment Analysis Results")

            sent_col1, sent_col2 = st.columns(2)
            with sent_col1:
                sentiment = sentiment_data.get("sentiment", "N/A")
                sentiment_emoji = {"Positive": "😊", "Negative": "😟", "Neutral": "😐", "Mixed": "🤔"}.get(sentiment.split()[0], "❓")
                st.metric(label="Overall Sentiment", value=f"{sentiment_emoji} {translate_text(sentiment, lang)}")
            with sent_col2:
                st.metric(label="Confidence Level", value=translate_text(sentiment_data.get("confidence", "N/A"), lang))

            st.markdown("#### 📊 Analysis Summary")
            st.info(translate_text(sentiment_data.get("summary", ""), lang))

            if sentiment_data.get("key_themes"):
                st.markdown("#### 🔑 Key Themes Identified")
                for theme in sentiment_data.get("key_themes", []):
                    st.markdown(f"• {translate_text(theme, lang)}")

            if sentiment_data.get("outlook"):
                st.markdown("#### 🔮 Short-term Outlook")
                st.success(translate_text(sentiment_data.get("outlook", ""), lang))

            with st.expander("🔍 View Raw Sentiment Analysis"):
                st.code(st.session_state.get('sentiment_raw', 'No raw output available.'), language='json')
    else:
        st.warning("⚠️ No recent news headlines found for this asset.")

# ============================================================================
# TAB 5: PORTFOLIO SIMULATOR
# ============================================================================
with tab5:
    st.subheader("💼 Paper Trading Portfolio Simulator")
    st.markdown("Track your simulated trades and monitor potential P&L based on current market prices.")

    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = []

    portfolio_col1, portfolio_col2 = st.columns([1, 2])

    with portfolio_col1:
        st.markdown("##### 📥 Add New Position")
        with st.form("portfolio_form"):
            p_asset = st.text_input("Ticker:", value=asset).upper()
            p_qty = st.number_input("Quantity:", min_value=0.01, value=10.0, step=1.0)
            p_buy_price = st.number_input(
                "Buy Price ($):",
                min_value=0.01,
                value=float(info.get('currentPrice', hist['Close'].iloc[-1])),
                step=0.01
            )
            submit_trade = st.form_submit_button("Add Position to Portfolio")
            if submit_trade:
                new_pos = {
                    "asset": p_asset, "qty": p_qty,
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

            other_tickers = tuple(sorted(set(
                pos['asset'] for pos in st.session_state.portfolio
                if pos['asset'] != asset
            )))
            batch_prices = get_portfolio_prices(other_tickers) if other_tickers else {}

            for pos in st.session_state.portfolio:
                if pos['asset'] == asset:
                    curr_p = info.get('currentPrice') or hist['Close'].iloc[-1]
                else:
                    curr_p = batch_prices.get(pos['asset']) or pos['buy_price']

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

            st.markdown("---")
            total_inv = sum(p['qty'] * p['buy_price'] for p in st.session_state.portfolio)
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
    if st.button("📄 Generate PDF Report", key="generate_pdf_btn"):
        has_insight = 'insight_data' in st.session_state and st.session_state.insight_data
        has_sentiment = 'sentiment_data' in st.session_state and st.session_state.sentiment_data

        if not has_insight and not has_sentiment:
            st.warning("⚠️ Please generate AI insights and sentiment analysis first.")
        else:
            with st.spinner("📝 Generating PDF report..."):
                pdf_data = create_pdf_report(
                    asset, key_metrics,
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
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        hist_export = hist.copy()
        hist_export.index.name = 'Date'
        if hist_export.index.tz is not None:
            hist_export.index = hist_export.index.tz_localize(None)
        hist_export.to_excel(writer, sheet_name='Historical Data')

        metrics_df = pd.DataFrame(key_metrics.items(), columns=["Metric", "Value"])
        metrics_df.to_excel(writer, sheet_name='Key Metrics', index=False)

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
