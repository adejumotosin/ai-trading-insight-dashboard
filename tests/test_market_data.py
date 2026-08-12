import sys
import types

import pandas as pd

sys.modules.setdefault("yfinance", types.SimpleNamespace())

from core.market_data import _normalize_history, _supports_stooq_fallback, _tail_requested
from core.analytics import calculate_indicators


def test_stooq_fallback_is_not_used_for_yahoo_specific_syntax():
    assert _supports_stooq_fallback("AAPL") is True
    assert _supports_stooq_fallback("BRK.B") is True
    assert _supports_stooq_fallback("BTC-USD") is False
    assert _supports_stooq_fallback("EURUSD=X") is False
    assert _supports_stooq_fallback("^GSPC") is False


def test_normalize_history_deduplicates_and_keeps_ohlcv():
    idx = pd.to_datetime(["2026-01-02", "2026-01-02", "2026-01-03"])
    df = pd.DataFrame(
        {
            "Open": [1, 2, 3],
            "High": [2, 3, 4],
            "Low": [0.5, 1.5, 2.5],
            "Close": [1.5, 2.5, 3.5],
            "Volume": [10, 20, 30],
        },
        index=idx,
    )
    out = _normalize_history(df)
    assert len(out) == 2
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"]
    assert float(out.iloc[0]["Close"]) == 2.5


def test_tail_requested_preserves_precomputed_indicator_values():
    idx = pd.bdate_range("2024-01-01", periods=300)
    close = pd.Series(range(100, 400), index=idx, dtype=float)
    raw = pd.DataFrame({"Open": close, "High": close + 1, "Low": close - 1, "Close": close, "Volume": 1000}, index=idx)
    enriched = calculate_indicators(raw)
    display = _tail_requested(enriched, "1mo")
    assert len(display) == 23
    assert pd.notna(display["SMA_200"].iloc[-1])
