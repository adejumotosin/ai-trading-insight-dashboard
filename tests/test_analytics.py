import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

spec = importlib.util.spec_from_file_location("analytics", Path(__file__).parents[1] / "core" / "analytics.py")
analytics = importlib.util.module_from_spec(spec)
import sys
sys.modules["analytics"] = analytics
spec.loader.exec_module(analytics)


def sample(n=420, rising=True):
    idx = pd.bdate_range("2024-01-01", periods=n)
    trend = np.linspace(100, 160 if rising else 60, n)
    wiggle = np.sin(np.arange(n) / 8) * 1.5
    close = trend + wiggle
    open_ = close * (1 + np.sin(np.arange(n)) * 0.001)
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    volume = np.linspace(1_000_000, 2_000_000, n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume}, index=idx)


def test_snapshot_has_full_coverage_with_warmup():
    snap = analytics.market_snapshot(sample())
    assert snap["data_coverage"] == 100
    assert -100 <= snap["trend_score"] <= 100
    assert snap["sma_200"] is not None


def test_rsi_edge_cases_are_finite():
    df = sample(260)
    df["Close"] = np.arange(1, 261, dtype=float)
    df["Open"] = df["Close"]
    df["High"] = df["Close"] + 1
    df["Low"] = df["Close"] - 1
    out = analytics.calculate_indicators(df)
    assert float(out["RSI_14"].iloc[-1]) == 100.0


def test_position_sizing_validates_stop_side():
    invalid = analytics.position_size(10_000, 1, 100, 105, direction="Long")
    assert invalid["valid"] is False
    valid = analytics.position_size(10_000, 1, 100, 95, direction="Long", max_leverage=1)
    assert valid["valid"] is True
    assert valid["notional"] <= 10_000 + 1e-9


def test_backtest_uses_next_session_open_without_lookahead():
    df = sample(320)
    bt = analytics.backtest_sma_strategy(df, sma_window=20, transaction_cost_bps=5, start_date=df.index[100])
    m = bt.metrics()
    assert len(bt.equity) == len(df.loc[df.index[100]:])
    assert np.isfinite(m["total_return"])
    assert m["trades"] >= 1
    assert 0 <= m["exposure"] <= 1


def test_short_display_window_does_not_force_snapshot_to_short_history():
    full = sample(420)
    enriched = analytics.calculate_indicators(full)
    display = enriched.tail(23)
    assert display["SMA_200"].iloc[-1] == enriched["SMA_200"].iloc[-1]


def test_backtest_does_not_capture_gap_before_next_open_entry():
    n = 45
    idx = pd.bdate_range("2025-01-01", periods=n)
    close = np.full(n, 100.0)
    open_ = np.full(n, 100.0)
    close[20:] = np.linspace(110, 120, n - 20)
    open_[21] = 200.0
    open_[22:] = np.linspace(202, 215, n - 22)
    high = np.maximum(open_, close) * 1.01
    low = np.minimum(open_, close) * 0.99
    df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": 1_000_000}, index=idx)
    bt = analytics.backtest_sma_strategy(df, sma_window=20, transaction_cost_bps=0)
    assert bt.total_return < 0.25
