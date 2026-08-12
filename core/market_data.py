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

WARMUP_PERIOD = {
    "1mo": "2y",
    "3mo": "2y",
    "6mo": "2y",
    "1y": "2y",
    "2y": "5y",
    "5y": "10y",
    "10y": "max",
}


@dataclass
class MarketBundle:
    symbol: str
    info: dict[str, Any]
    history: pd.DataFrame
    analysis_history: pd.DataFrame
    source: str
    fetched_at: datetime
    requested_period: str
    adjusted_prices: bool = True
    warning: str | None = None

    @property
    def display_rows(self) -> int:
        return len(self.history)

    @property
    def analysis_rows(self) -> int:
        return len(self.analysis_history)


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
    title_lookup = {str(c).title(): c for c in out.columns}
    required = ["Open", "High", "Low", "Close"]
    if not all(name in out.columns for name in required):
        mapped = {}
        for target in required + ["Volume"]:
            original = title_lookup.get(target)
            if original is not None:
                mapped[original] = target
        out = out.rename(columns=mapped)

    for col in required + ["Volume"]:
        if col not in out.columns:
            out[col] = 0.0 if col == "Volume" else np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out = out.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out[["Open", "High", "Low", "Close", "Volume"]]


def _info_from_history(symbol: str, hist: pd.DataFrame) -> dict[str, Any]:
    last = hist.iloc[-1]
    prev = hist.iloc[-2] if len(hist) > 1 else last
    trailing = hist.tail(min(252, len(hist)))
    return {
        "symbol": symbol,
        "longName": symbol,
        "currentPrice": float(last["Close"]),
        "regularMarketPrice": float(last["Close"]),
        "previousClose": float(prev["Close"]),
        "open": float(last["Open"]),
        "dayLow": float(last["Low"]),
        "dayHigh": float(last["High"]),
        "fiftyTwoWeekLow": float(trailing["Low"].min()),
        "fiftyTwoWeekHigh": float(trailing["High"].max()),
        "volume": int(last.get("Volume", 0) or 0),
        "averageVolume": int(trailing["Volume"].tail(20).mean()) if "Volume" in trailing else 0,
    }


def _yahoo_history(symbol: str, period: str) -> pd.DataFrame:
    kwargs = dict(
        tickers=symbol,
        period=period,
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
        timeout=15,
    )
    try:
        df = yf.download(repair=True, multi_level_index=False, **kwargs)
        return _normalize_history(df)
    except TypeError:
        try:
            kwargs.pop("timeout", None)
            df = yf.download(**kwargs)
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
                info.update(
                    {
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
                    }
                )
            except Exception:
                pass
    except Exception:
        pass
    return info


def _supports_stooq_fallback(symbol: str) -> bool:
    return bool(re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", symbol))


def _stooq_history(symbol: str) -> pd.DataFrame:
    if not _supports_stooq_fallback(symbol):
        return pd.DataFrame()

    stooq_symbol = symbol.lower()
    for suffix in (".us", ""):
        url = f"https://stooq.com/q/d/l/?s={stooq_symbol}{suffix}&i=d"
        try:
            response = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=10)
            if response.ok and len(response.text) > 100:
                df = pd.read_csv(StringIO(response.text))
                if "Date" in df.columns and "Close" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
                    normalized = _normalize_history(df)
                    if not normalized.empty:
                        return normalized
        except Exception:
            continue
    return pd.DataFrame()


def _demo_history(symbol: str, bars: int) -> pd.DataFrame:
    bars = max(int(bars), 520)
    seed = int(hashlib.sha256(symbol.encode("utf-8")).hexdigest()[:16], 16) % (2**32)
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.utcnow().normalize().tz_localize(None), periods=bars)
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


def _tail_requested(enriched: pd.DataFrame, period: str) -> pd.DataFrame:
    bars = PERIOD_BARS.get(period)
    return enriched.tail(bars).copy() if bars else enriched.copy()


def fetch_market_bundle(symbol: str, period: str = "1y", demo_mode: bool = False) -> MarketBundle:
    symbol = symbol.strip().upper()
    valid, message = validate_ticker(symbol)
    if not valid:
        raise ValueError(message)
    if period not in PERIOD_BARS:
        raise ValueError(f"Unsupported period: {period}")

    now = datetime.now(timezone.utc)
    fetch_period = WARMUP_PERIOD.get(period, "2y")

    if demo_mode:
        analysis_raw = _demo_history(symbol, max(PERIOD_BARS.get(fetch_period, 0), PERIOD_BARS[period] + 260))
        analysis_history = calculate_indicators(analysis_raw)
        display_history = _tail_requested(analysis_history, period)
        info = _info_from_history(symbol, analysis_raw)
        info["longName"] = f"{symbol} Demo Series"
        info.setdefault("currency", "USD")
        return MarketBundle(
            symbol=symbol,
            info=info,
            history=display_history,
            analysis_history=analysis_history,
            source="Simulated",
            fetched_at=now,
            requested_period=period,
            adjusted_prices=True,
            warning="Demo mode uses deterministic synthetic data and must not be interpreted as live market information.",
        )

    analysis_raw = _yahoo_history(symbol, fetch_period)
    source = "Yahoo Finance"
    warning = None
    adjusted_prices = True

    if analysis_raw.empty:
        analysis_raw = _stooq_history(symbol)
        if not analysis_raw.empty:
            source = "Stooq fallback"
            adjusted_prices = False
            warning = (
                "Yahoo Finance was unavailable, so historical data is being served from Stooq. "
                "Corporate-action adjustment behavior may differ from Yahoo Finance."
            )

    if analysis_raw.empty:
        fallback_note = (
            " Stooq fallback is only attempted for ordinary equity symbols."
            if not _supports_stooq_fallback(symbol)
            else ""
        )
        raise RuntimeError(f"No market history could be retrieved from available providers.{fallback_note}")

    analysis_history = calculate_indicators(analysis_raw)
    display_history = _tail_requested(analysis_history, period)
    if display_history.empty:
        raise RuntimeError("Market data was retrieved but the selected display period is empty.")

    info = _yahoo_info(symbol)
    derived = _info_from_history(symbol, analysis_raw)
    for key, value in derived.items():
        if info.get(key) in (None, "", 0):
            info[key] = value

    return MarketBundle(
        symbol=symbol,
        info=info,
        history=display_history,
        analysis_history=analysis_history,
        source=source,
        fetched_at=now,
        requested_period=period,
        adjusted_prices=adjusted_prices,
        warning=warning,
    )


def fetch_news(symbol: str, company_name: str | None = None, max_items: int = 8) -> list[dict[str, str]]:
    subject = company_name.strip() if company_name and company_name.strip() else symbol
    query = quote_plus(f'"{subject}" market OR stock OR earnings')
    url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
    try:
        response = requests.get(url, headers={"User-Agent": _BROWSER_UA}, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, features="xml")
        items: list[dict[str, str]] = []
        seen: set[str] = set()
        for item in soup.find_all("item"):
            title = item.title.get_text(strip=True) if item.title else ""
            if not title or title in seen:
                continue
            seen.add(title)
            link = item.link.get_text(strip=True) if item.link else ""
            published = item.pubDate.get_text(strip=True) if item.pubDate else ""
            source = item.source.get_text(strip=True) if item.source else ""
            items.append({"title": title, "link": link, "published": published, "source": source})
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []
