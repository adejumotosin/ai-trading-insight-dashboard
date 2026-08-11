from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import StringIO
from typing import Any
from urllib.parse import quote_plus

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

from .analytics import calculate_indicators

_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

PERIOD_BARS = {
    "1mo": 23,
    "3mo": 66,
    "6mo": 132,
    "1y": 264,
    "2y": 528,
    "5y": 1320,
    "10y": 2640,
}


@dataclass
class MarketBundle:
    symbol: str
    info: dict[str, Any]
    history: pd.DataFrame
    source: str
    fetched_at: datetime
    warning: str | None = None


def validate_ticker(symbol: str) -> tuple[bool, str]:
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return False, "Ticker cannot be empty."
    if not re.fullmatch(r"[A-Z0-9^=._-]{1,20}", symbol):
        return False, "Ticker contains unsupported characters."
    return True, ""


def _flatten_yf_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        first = list(out.columns.get_level_values(0))
        if {"Open", "High", "Low", "Close"}.issubset(set(first)):
            out.columns = out.columns.get_level_values(0)
        else:
            out.columns = out.columns.get_level_values(-1)
    return out


def _normalize_history(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = _flatten_yf_columns(df)
    rename = {str(c).title(): c for c in out.columns}
    required = ["Open", "High", "Low", "Close"]
    if not all(name in out.columns for name in required):
        mapped = {}
        for target in required + ["Volume"]:
            original = rename.get(target)
            if original is not None:
                mapped[original] = target
        out = out.rename(columns=mapped)
    for col in required + ["Volume"]:
        if col not in out.columns:
            out[col] = 0.0 if col == "Volume" else np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["Close"]).sort_index()
    return out[["Open", "High", "Low", "Close", "Volume"]]


def _info_from_history(symbol: str, hist: pd.DataFrame) -> dict[str, Any]:
    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else last
    return {
        "symbol": symbol,
        "longName": symbol,
        "currentPrice": float(last["Close"]),
        "regularMarketPrice": float(last["Close"]),
        "previousClose": float(prev["Close"]),
        "open": float(last["Open"]),
        "dayLow": float(last["Low"]),
        "dayHigh": float(last["High"]),
        "fiftyTwoWeekLow": float(hist["Low"].tail(252).min()),
        "fiftyTwoWeekHigh": float(hist["High"].tail(252).max()),
        "volume": int(last.get("Volume", 0) or 0),
        "averageVolume": int(hist["Volume"].tail(20).mean()) if "Volume" in hist else 0,
    }


def _yahoo_history(symbol: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            symbol,
            period=period,
            interval="1d",
            auto_adjust=False,
            repair=True,
            progress=False,
            threads=False,
            multi_level_index=False,
            timeout=12,
        )
        return _normalize_history(df)
    except TypeError:
        try:
            df = yf.download(
                symbol,
                period=period,
                interval="1d",
                auto_adjust=False,
                progress=False,
                threads=False,
            )
            return _normalize_history(df)
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _yahoo_info(symbol: str) -> dict[str, Any]:
    info: dict[str, Any] = {}
    try:
        ticker = yf.Ticker(symbol)
        try:
            raw = ticker.info
            if isinstance(raw, dict):
                info.update(raw)
        except Exception:
            pass
        if not info:
            try:
                fast = ticker.fast_info
                info.update({
                    "symbol": symbol,
                    "longName": symbol,
                    "currentPrice": getattr(fast, "last_price", None),
                    "previousClose": getattr(fast, "previous_close", None),
                    "open": getattr(fast, "open", None),
                    "dayLow": getattr(fast, "day_low", None),
                    "dayHigh": getattr(fast, "day_high", None),
                    "fiftyTwoWeekLow": getattr(fast, "year_low", None),
                    "fiftyTwoWeekHigh": getattr(fast, "year_high", None),
                    "marketCap": getattr(fast, "market_cap", None),
                    "currency": getattr(fast, "currency", None),
                    "exchange": getattr(fast, "exchange", None),
                })
            except Exception:
                pass
    except Exception:
        pass
    return info


def _stooq_history(symbol: str) -> pd.DataFrame:
    stooq_symbol = symbol.replace("-", ".").lower()
    for suffix in (".us", ""):
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}{suffix}&i=d"
        try:
            response = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=10)
            if response.ok and len(response.text) > 100:
                df = pd.read_csv(StringIO(response.text))
                if "Date" in df.columns and "Close" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    return _normalize_history(df)
        except Exception:
            continue
    return pd.DataFrame()


def _demo_history(symbol: str, period: str) -> pd.DataFrame:
    bars = PERIOD_BARS.get(period, 264)
    seed = int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize(), periods=bars)
    base = 40 + (seed % 460)
    drift = ((seed % 9) - 3) / 100000
    shocks = rng.normal(drift, 0.017, bars)
    close = base * np.exp(np.cumsum(shocks))
    overnight = rng.normal(0, 0.003, bars)
    open_ = close * (1 + overnight)
    span = np.maximum(abs(rng.normal(0.012, 0.005, bars)), 0.002)
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)
    volume = rng.integers(500_000, 12_000_000, bars)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def fetch_market_bundle(symbol: str, period: str = "1y", demo_mode: bool = False) -> MarketBundle:
    symbol = symbol.strip().upper()
    valid, message = validate_ticker(symbol)
    if not valid:
        raise ValueError(message)

    now = datetime.now(timezone.utc)
    if demo_mode:
        hist = _demo_history(symbol, period)
        info = _info_from_history(symbol, hist)
        info["longName"] = f"{symbol} Demo Series"
        return MarketBundle(
            symbol=symbol,
            info=info,
            history=calculate_indicators(hist),
            source="Simulated",
            fetched_at=now,
            warning="Demo mode uses deterministic synthetic data and must not be interpreted as live market information.",
        )

    hist = _yahoo_history(symbol, period)
    source = "Yahoo Finance"
    warning = None

    if hist.empty:
        hist = _stooq_history(symbol)
        if not hist.empty:
            bars = PERIOD_BARS.get(period)
            if bars:
                hist = hist.tail(bars)
            source = "Stooq fallback"
            warning = "Yahoo Finance was unavailable, so historical data is being served from Stooq."

    if hist.empty:
        raise RuntimeError("No market history could be retrieved from Yahoo Finance or Stooq.")

    info = _yahoo_info(symbol)
    derived = _info_from_history(symbol, hist)
    for key, value in derived.items():
        if info.get(key) in (None, "", 0):
            info[key] = value

    return MarketBundle(
        symbol=symbol,
        info=info,
        history=calculate_indicators(hist),
        source=source,
        fetched_at=now,
        warning=warning,
    )


def fetch_news(symbol: str, max_items: int = 8) -> list[dict[str, str]]:
    query = quote_plus(f"{symbol} stock OR market")
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, features="xml")
        items = []
        for item in soup.find_all("item")[:max_items]:
            title = item.title.get_text(strip=True) if item.title else ""
            link = item.link.get_text(strip=True) if item.link else ""
            published = item.pubDate.get_text(strip=True) if item.pubDate else ""
            source = ""
            if item.source:
                source = item.source.get_text(strip=True)
            if title:
                items.append({"title": title, "link": link, "published": published, "source": source})
        return items
    except Exception:
        return []
