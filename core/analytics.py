from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise ValueError(f"Missing required column: {name}")
    return pd.to_numeric(df[name], errors="coerce")


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of OHLCV data enriched with deterministic indicators."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy().sort_index()
    close = _series(out, "Close")
    high = _series(out, "High")
    low = _series(out, "Low")

    for window in (20, 50, 200):
        out[f"SMA_{window}"] = close.rolling(window, min_periods=window).mean()
    out["EMA_20"] = close.ewm(span=20, adjust=False).mean()

    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((avg_loss == 0) & (avg_gain > 0), 100.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss > 0), 0.0)
    rsi = rsi.mask((avg_gain == 0) & (avg_loss == 0), 50.0)
    out["RSI_14"] = rsi.fillna(50.0)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["MACD"] = ema12 - ema26
    out["MACD_SIGNAL"] = out["MACD"].ewm(span=9, adjust=False).mean()
    out["MACD_HIST"] = out["MACD"] - out["MACD_SIGNAL"]

    prev_close = close.shift(1)
    true_range = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    out["ATR_14"] = true_range.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()

    returns = close.pct_change()
    out["VOL_20"] = returns.rolling(20).std() * np.sqrt(TRADING_DAYS)
    if "Volume" in out.columns:
        out["VOLUME_SMA_20"] = pd.to_numeric(out["Volume"], errors="coerce").rolling(20).mean()

    return out


def _period_return(close: pd.Series, sessions: int) -> float | None:
    if len(close.dropna()) <= sessions:
        return None
    start = float(close.iloc[-sessions - 1])
    end = float(close.iloc[-1])
    if not np.isfinite(start) or start == 0:
        return None
    return end / start - 1


def _max_drawdown(close: pd.Series) -> float:
    clean = close.dropna().astype(float)
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    drawdown = clean / peak - 1
    return float(drawdown.min())


def _trend_components(latest: pd.Series) -> list[dict[str, Any]]:
    price = float(latest.get("Close", np.nan))
    sma20 = latest.get("SMA_20")
    sma50 = latest.get("SMA_50")
    sma200 = latest.get("SMA_200")
    rsi = latest.get("RSI_14")
    macd = latest.get("MACD")
    macd_signal = latest.get("MACD_SIGNAL")

    factors: list[dict[str, Any]] = []

    def add(label: str, bullish: bool | None, weight: int) -> None:
        if bullish is None:
            score = 0
            state = "Unavailable"
        elif bullish:
            score = weight
            state = "Bullish"
        else:
            score = -weight
            state = "Bearish"
        factors.append({"factor": label, "state": state, "score": score})

    add("Price vs SMA 20", None if pd.isna(sma20) else price > float(sma20), 15)
    add("Price vs SMA 50", None if pd.isna(sma50) else price > float(sma50), 20)
    add("SMA 50 vs SMA 200", None if pd.isna(sma50) or pd.isna(sma200) else float(sma50) > float(sma200), 25)
    add("MACD vs signal", None if pd.isna(macd) or pd.isna(macd_signal) else float(macd) > float(macd_signal), 20)

    if pd.isna(rsi):
        factors.append({"factor": "RSI regime", "state": "Unavailable", "score": 0})
    else:
        rsi = float(rsi)
        if 50 <= rsi <= 70:
            factors.append({"factor": "RSI regime", "state": "Bullish", "score": 20})
        elif 30 <= rsi < 50:
            factors.append({"factor": "RSI regime", "state": "Bearish", "score": -20})
        elif rsi > 70:
            factors.append({"factor": "RSI regime", "state": "Extended", "score": 5})
        else:
            factors.append({"factor": "RSI regime", "state": "Oversold", "score": -5})

    return factors


def market_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """Create a deterministic market-state snapshot from enriched OHLCV data."""
    if df is None or df.empty:
        return {}
    enriched = calculate_indicators(df)
    close = enriched["Close"].astype(float)
    latest = enriched.iloc[-1]
    factors = _trend_components(latest)
    score = int(sum(item["score"] for item in factors))

    if score >= 40:
        regime = "Bullish"
    elif score <= -40:
        regime = "Bearish"
    else:
        regime = "Neutral"

    confidence = min(100, abs(score))
    current = float(close.iloc[-1])
    previous = float(close.iloc[-2]) if len(close) > 1 else current
    daily_return = current / previous - 1 if previous else 0.0
    returns = close.pct_change().dropna()
    annualized_vol = float(returns.tail(60).std() * np.sqrt(TRADING_DAYS)) if len(returns) >= 20 else None
    atr = latest.get("ATR_14")
    atr = None if pd.isna(atr) else float(atr)

    return {
        "price": current,
        "daily_return": daily_return,
        "return_1m": _period_return(close, 21),
        "return_3m": _period_return(close, 63),
        "return_6m": _period_return(close, 126),
        "return_1y": _period_return(close, 252),
        "annualized_volatility": annualized_vol,
        "max_drawdown": _max_drawdown(close),
        "rsi_14": float(latest.get("RSI_14", 50.0)),
        "atr_14": atr,
        "atr_pct": (atr / current) if atr and current else None,
        "sma_20": None if pd.isna(latest.get("SMA_20")) else float(latest.get("SMA_20")),
        "sma_50": None if pd.isna(latest.get("SMA_50")) else float(latest.get("SMA_50")),
        "sma_200": None if pd.isna(latest.get("SMA_200")) else float(latest.get("SMA_200")),
        "macd": float(latest.get("MACD", 0.0)),
        "macd_signal": float(latest.get("MACD_SIGNAL", 0.0)),
        "regime": regime,
        "trend_score": score,
        "confidence": confidence,
        "factors": factors,
        "support_20": float(enriched["Low"].tail(20).min()),
        "resistance_20": float(enriched["High"].tail(20).max()),
    }


def position_size(account_size: float, risk_pct: float, entry: float, stop: float) -> dict[str, float]:
    """Risk-based position size. risk_pct is a percentage, e.g. 1.0 means 1%."""
    account_size = max(float(account_size), 0.0)
    risk_pct = max(float(risk_pct), 0.0)
    entry = float(entry)
    stop = float(stop)
    risk_per_unit = abs(entry - stop)
    risk_amount = account_size * (risk_pct / 100)
    units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
    notional = units * entry
    return {
        "risk_amount": risk_amount,
        "risk_per_unit": risk_per_unit,
        "units": units,
        "notional": notional,
    }


@dataclass
class BacktestResult:
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    sharpe: float | None
    trades: int
    win_rate: float | None
    exposure: float
    equity: pd.Series
    benchmark: pd.Series
    position: pd.Series

    def metrics(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("equity", "benchmark", "position"):
            data.pop(key, None)
        return data


def backtest_sma_strategy(
    df: pd.DataFrame,
    sma_window: int = 50,
    transaction_cost_bps: float = 5.0,
) -> BacktestResult:
    """Long-only close>SMA strategy with next-session execution and explicit costs."""
    if df is None or df.empty or len(df) < sma_window + 5:
        raise ValueError(f"At least {sma_window + 5} rows are required for this backtest")

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    sma = close.rolling(sma_window, min_periods=sma_window).mean()
    raw_signal = (close > sma).astype(float)
    position = raw_signal.shift(1).fillna(0.0)
    returns = close.pct_change().fillna(0.0)

    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (float(transaction_cost_bps) / 10000.0)
    strategy_returns = position * returns - cost

    equity = (1 + strategy_returns).cumprod()
    benchmark = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    std = strategy_returns.std()
    sharpe = None if not np.isfinite(std) or std == 0 else float(strategy_returns.mean() / std * np.sqrt(TRADING_DAYS))

    entries = (position.diff() > 0).fillna(False)
    exits = (position.diff() < 0).fillna(False)
    entry_dates = list(position.index[entries])
    exit_dates = list(position.index[exits])
    trade_returns: list[float] = []
    for entry_date in entry_dates:
        later_exits = [d for d in exit_dates if d > entry_date]
        exit_date = later_exits[0] if later_exits else position.index[-1]
        entry_price = float(close.loc[entry_date])
        exit_price = float(close.loc[exit_date])
        if entry_price:
            trade_returns.append(exit_price / entry_price - 1 - 2 * float(transaction_cost_bps) / 10000.0)

    win_rate = None if not trade_returns else sum(r > 0 for r in trade_returns) / len(trade_returns)

    return BacktestResult(
        total_return=float(equity.iloc[-1] - 1),
        buy_hold_return=float(benchmark.iloc[-1] - 1),
        max_drawdown=float(drawdown.min()),
        sharpe=sharpe,
        trades=len(trade_returns),
        win_rate=win_rate,
        exposure=float(position.mean()),
        equity=equity,
        benchmark=benchmark,
        position=position,
    )
