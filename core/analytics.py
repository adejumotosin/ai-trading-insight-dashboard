from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def _series(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        raise ValueError(f"Missing required column: {name}")
    return pd.to_numeric(df[name], errors="coerce")


def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return OHLCV data enriched with deterministic technical indicators."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df.copy()

    out = df.copy().sort_index()
    close = _series(out, "Close")
    high = _series(out, "High")
    low = _series(out, "Low")

    for window in (20, 50, 100, 200):
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
    out["RSI_14"] = rsi

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
    out["VOL_20"] = returns.rolling(20, min_periods=20).std() * np.sqrt(TRADING_DAYS)
    out["VOL_60"] = returns.rolling(60, min_periods=20).std() * np.sqrt(TRADING_DAYS)

    mid = close.rolling(20, min_periods=20).mean()
    std20 = close.rolling(20, min_periods=20).std()
    out["BB_MID"] = mid
    out["BB_UPPER"] = mid + 2 * std20
    out["BB_LOWER"] = mid - 2 * std20

    if "Volume" in out.columns:
        volume = pd.to_numeric(out["Volume"], errors="coerce")
        out["VOLUME_SMA_20"] = volume.rolling(20, min_periods=5).mean()
        out["VOLUME_RATIO"] = volume / out["VOLUME_SMA_20"].replace(0, np.nan)

    return out


def _period_return(close: pd.Series, sessions: int) -> float | None:
    clean = close.dropna().astype(float)
    if len(clean) <= sessions:
        return None
    start = float(clean.iloc[-sessions - 1])
    end = float(clean.iloc[-1])
    if not np.isfinite(start) or start == 0:
        return None
    return end / start - 1


def _max_drawdown(values: pd.Series) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return 0.0
    peak = clean.cummax()
    drawdown = clean / peak - 1
    return float(drawdown.min())


def _safe_float(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _trend_components(latest: pd.Series) -> list[dict[str, Any]]:
    price = _safe_float(latest.get("Close"))
    sma20 = _safe_float(latest.get("SMA_20"))
    sma50 = _safe_float(latest.get("SMA_50"))
    sma200 = _safe_float(latest.get("SMA_200"))
    rsi = _safe_float(latest.get("RSI_14"))
    macd = _safe_float(latest.get("MACD"))
    macd_signal = _safe_float(latest.get("MACD_SIGNAL"))

    factors: list[dict[str, Any]] = []

    def add(label: str, state: str, score: int, weight: int, detail: str = "") -> None:
        factors.append(
            {
                "factor": label,
                "state": state,
                "score": score,
                "weight": weight,
                "detail": detail,
                "available": state != "Unavailable",
            }
        )

    if price is None or sma20 is None:
        add("Price vs SMA 20", "Unavailable", 0, 15)
    else:
        bullish = price > sma20
        add("Price vs SMA 20", "Bullish" if bullish else "Bearish", 15 if bullish else -15, 15)

    if price is None or sma50 is None:
        add("Price vs SMA 50", "Unavailable", 0, 20)
    else:
        bullish = price > sma50
        add("Price vs SMA 50", "Bullish" if bullish else "Bearish", 20 if bullish else -20, 20)

    if sma50 is None or sma200 is None:
        add("SMA 50 vs SMA 200", "Unavailable", 0, 25)
    else:
        bullish = sma50 > sma200
        add("SMA 50 vs SMA 200", "Bullish" if bullish else "Bearish", 25 if bullish else -25, 25)

    if macd is None or macd_signal is None:
        add("MACD vs signal", "Unavailable", 0, 20)
    else:
        bullish = macd > macd_signal
        add("MACD vs signal", "Bullish" if bullish else "Bearish", 20 if bullish else -20, 20)

    if rsi is None:
        add("RSI regime", "Unavailable", 0, 20)
    elif rsi >= 70:
        add("RSI regime", "Extended", 5, 20, f"RSI {rsi:.1f}")
    elif rsi >= 50:
        add("RSI regime", "Bullish", 20, 20, f"RSI {rsi:.1f}")
    elif rsi >= 30:
        add("RSI regime", "Bearish", -20, 20, f"RSI {rsi:.1f}")
    else:
        add("RSI regime", "Oversold", -5, 20, f"RSI {rsi:.1f}")

    return factors


def market_snapshot(df: pd.DataFrame) -> dict[str, Any]:
    """Create a range-independent deterministic market-state snapshot."""
    if df is None or df.empty:
        return {}

    enriched = calculate_indicators(df)
    close = pd.to_numeric(enriched["Close"], errors="coerce").dropna()
    if close.empty:
        return {}

    latest = enriched.loc[close.index[-1]]
    factors = _trend_components(latest)
    available_weight = sum(item["weight"] for item in factors if item["available"])
    raw_score = sum(item["score"] for item in factors)
    normalized_score = int(round(raw_score / available_weight * 100)) if available_weight else 0
    coverage = int(round(available_weight / sum(item["weight"] for item in factors) * 100))

    if normalized_score >= 30:
        regime = "Bullish"
    elif normalized_score <= -30:
        regime = "Bearish"
    else:
        regime = "Neutral"

    current = float(close.iloc[-1])
    previous = float(close.iloc[-2]) if len(close) > 1 else current
    daily_return = current / previous - 1 if previous else 0.0
    returns = close.pct_change().dropna()
    annualized_vol = float(returns.tail(60).std() * np.sqrt(TRADING_DAYS)) if len(returns) >= 20 else None
    atr = _safe_float(latest.get("ATR_14"))

    prior = enriched.iloc[:-1] if len(enriched) > 1 else enriched
    support_window = prior["Low"].tail(20).dropna()
    resistance_window = prior["High"].tail(20).dropna()

    vol60 = _safe_float(latest.get("VOL_60"))
    vol_history = pd.to_numeric(enriched.get("VOL_60", pd.Series(dtype=float)), errors="coerce").dropna()
    volatility_percentile = None
    volatility_regime = "Unavailable"
    if vol60 is not None and len(vol_history) >= 20:
        percentile = float((vol_history <= vol60).mean())
        volatility_percentile = percentile
        if percentile >= 0.75:
            volatility_regime = "High"
        elif percentile <= 0.25:
            volatility_regime = "Low"
        else:
            volatility_regime = "Normal"

    confidence = min(100, abs(normalized_score))

    return {
        "price": current,
        "daily_return": daily_return,
        "return_1m": _period_return(close, 21),
        "return_3m": _period_return(close, 63),
        "return_6m": _period_return(close, 126),
        "return_1y": _period_return(close, 252),
        "annualized_volatility": annualized_vol,
        "volatility_regime": volatility_regime,
        "volatility_percentile": volatility_percentile,
        "max_drawdown": _max_drawdown(close),
        "rsi_14": _safe_float(latest.get("RSI_14")),
        "atr_14": atr,
        "atr_pct": (atr / current) if atr and current else None,
        "sma_20": _safe_float(latest.get("SMA_20")),
        "sma_50": _safe_float(latest.get("SMA_50")),
        "sma_100": _safe_float(latest.get("SMA_100")),
        "sma_200": _safe_float(latest.get("SMA_200")),
        "macd": _safe_float(latest.get("MACD")),
        "macd_signal": _safe_float(latest.get("MACD_SIGNAL")),
        "regime": regime,
        "trend_score": normalized_score,
        "raw_trend_score": int(raw_score),
        "confidence": confidence,
        "data_coverage": coverage,
        "factors": factors,
        "support_20": float(support_window.min()) if not support_window.empty else None,
        "resistance_20": float(resistance_window.max()) if not resistance_window.empty else None,
    }


def position_size(
    account_size: float,
    risk_pct: float,
    entry: float,
    stop: float,
    direction: str = "Long",
    max_leverage: float = 1.0,
) -> dict[str, Any]:
    """Risk-based position sizing with stop-direction and capital constraints."""
    account_size = max(float(account_size), 0.0)
    risk_pct = max(float(risk_pct), 0.0)
    entry = float(entry)
    stop = float(stop)
    max_leverage = max(float(max_leverage), 0.0)
    direction = str(direction).strip().title()

    if entry <= 0 or stop <= 0:
        return {"valid": False, "error": "Entry and stop must be greater than zero."}
    if direction not in {"Long", "Short"}:
        return {"valid": False, "error": "Direction must be Long or Short."}
    if direction == "Long" and stop >= entry:
        return {"valid": False, "error": "For a long position, the stop must be below the entry."}
    if direction == "Short" and stop <= entry:
        return {"valid": False, "error": "For a short position, the stop must be above the entry."}

    risk_per_unit = abs(entry - stop)
    risk_amount = account_size * (risk_pct / 100)
    risk_units = risk_amount / risk_per_unit if risk_per_unit > 0 else 0.0
    capital_limit = account_size * max_leverage
    capital_units = capital_limit / entry if entry > 0 else 0.0
    units = min(risk_units, capital_units)
    notional = units * entry
    actual_risk = units * risk_per_unit

    limited_by = "capital" if capital_units < risk_units else "risk"
    return {
        "valid": True,
        "error": None,
        "direction": direction,
        "risk_amount": risk_amount,
        "risk_per_unit": risk_per_unit,
        "risk_units": risk_units,
        "capital_units": capital_units,
        "units": units,
        "notional": notional,
        "actual_risk": actual_risk,
        "limited_by": limited_by,
        "max_leverage": max_leverage,
    }


@dataclass
class BacktestResult:
    total_return: float
    buy_hold_return: float
    cagr: float | None
    max_drawdown: float
    annualized_volatility: float | None
    sharpe: float | None
    sortino: float | None
    calmar: float | None
    trades: int
    win_rate: float | None
    avg_trade_return: float | None
    best_trade: float | None
    worst_trade: float | None
    exposure: float
    equity: pd.Series
    benchmark: pd.Series
    position: pd.Series
    trade_returns: pd.Series

    def metrics(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("equity", "benchmark", "position", "trade_returns"):
            data.pop(key, None)
        return data


def _annualized_ratio(returns: pd.Series, downside_only: bool = False) -> float | None:
    clean = returns.dropna().astype(float)
    if clean.empty:
        return None
    denominator_series = clean[clean < 0] if downside_only else clean
    denominator = denominator_series.std()
    if denominator is None or not np.isfinite(denominator) or denominator == 0:
        return None
    return float(clean.mean() / denominator * np.sqrt(TRADING_DAYS))


def backtest_sma_strategy(
    df: pd.DataFrame,
    sma_window: int = 50,
    transaction_cost_bps: float = 5.0,
    start_date: Any | None = None,
) -> BacktestResult:
    """
    Long-only Close>SMA strategy.

    The signal is observed at session close and executed at the next session open.
    P&L is measured open-to-open while the position is held, with the final active
    position marked from the last open to the last close. Transaction costs are
    charged on each entry and exit.
    """
    if df is None or df.empty or len(df) < sma_window + 5:
        raise ValueError(f"At least {sma_window + 5} rows are required for this backtest")
    if "Open" not in df.columns or "Close" not in df.columns:
        raise ValueError("Backtest requires Open and Close columns")

    data = df[["Open", "Close"]].copy().sort_index()
    data["Open"] = pd.to_numeric(data["Open"], errors="coerce")
    data["Close"] = pd.to_numeric(data["Close"], errors="coerce")
    data = data.dropna()
    if len(data) < sma_window + 5:
        raise ValueError(f"At least {sma_window + 5} valid rows are required for this backtest")

    close = data["Close"]
    open_ = data["Open"]
    sma = close.rolling(sma_window, min_periods=sma_window).mean()
    signal_at_close = (close > sma).astype(float).where(sma.notna(), 0.0)
    position = signal_at_close.shift(1).fillna(0.0)

    next_open_return = open_.shift(-1) / open_ - 1

    if start_date is not None:
        start_ts = pd.Timestamp(start_date)
        mask = data.index >= start_ts
        if not mask.any():
            raise ValueError("Backtest start date is outside the available history")
        position = position.loc[mask]
        next_open_return = next_open_return.loc[mask]
        open_eval = open_.loc[mask]
        close_eval = close.loc[mask]
    else:
        open_eval = open_
        close_eval = close

    if len(position) < 2:
        raise ValueError("Not enough observations in the selected evaluation window")

    cost_rate = max(float(transaction_cost_bps), 0.0) / 10000.0
    turnover = position.diff().abs()
    turnover.iloc[0] = abs(position.iloc[0])
    strategy_returns = position * next_open_return.fillna(0.0) - turnover.fillna(0.0) * cost_rate

    if position.iloc[-1] > 0:
        final_intraday = close_eval.iloc[-1] / open_eval.iloc[-1] - 1
        strategy_returns.iloc[-1] = position.iloc[-1] * final_intraday - cost_rate

    benchmark_returns = next_open_return.fillna(0.0).copy()
    benchmark_returns.iloc[-1] = close_eval.iloc[-1] / open_eval.iloc[-1] - 1

    equity = (1 + strategy_returns).cumprod()
    benchmark = (1 + benchmark_returns).cumprod()
    drawdown = equity / equity.cummax() - 1

    pos_diff = position.diff().fillna(position)
    entry_dates = list(position.index[pos_diff > 0])
    exit_dates = list(position.index[pos_diff < 0])
    trade_values: list[float] = []
    trade_index: list[pd.Timestamp] = []
    for entry_date in entry_dates:
        later_exits = [d for d in exit_dates if d > entry_date]
        if later_exits:
            exit_date = later_exits[0]
            exit_price = float(open_eval.loc[exit_date])
        else:
            exit_date = position.index[-1]
            exit_price = float(close_eval.loc[exit_date])
        entry_price = float(open_eval.loc[entry_date])
        if entry_price > 0:
            gross = exit_price / entry_price - 1
            net = (1 + gross) * (1 - cost_rate) * (1 - cost_rate) - 1
            trade_values.append(float(net))
            trade_index.append(pd.Timestamp(entry_date))

    trade_returns = pd.Series(trade_values, index=trade_index, dtype=float)
    win_rate = float((trade_returns > 0).mean()) if not trade_returns.empty else None

    total_return = float(equity.iloc[-1] - 1)
    buy_hold_return = float(benchmark.iloc[-1] - 1)
    years = len(strategy_returns) / TRADING_DAYS
    cagr = float(equity.iloc[-1] ** (1 / years) - 1) if years > 0 and equity.iloc[-1] > 0 else None
    ann_vol = float(strategy_returns.std() * np.sqrt(TRADING_DAYS)) if strategy_returns.std() > 0 else None
    sharpe = _annualized_ratio(strategy_returns)
    sortino = _annualized_ratio(strategy_returns, downside_only=True)
    max_drawdown = float(drawdown.min())
    calmar = None if cagr is None or max_drawdown >= 0 else float(cagr / abs(max_drawdown))

    return BacktestResult(
        total_return=total_return,
        buy_hold_return=buy_hold_return,
        cagr=cagr,
        max_drawdown=max_drawdown,
        annualized_volatility=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        trades=int(len(trade_returns)),
        win_rate=win_rate,
        avg_trade_return=float(trade_returns.mean()) if not trade_returns.empty else None,
        best_trade=float(trade_returns.max()) if not trade_returns.empty else None,
        worst_trade=float(trade_returns.min()) if not trade_returns.empty else None,
        exposure=float(position.mean()),
        equity=equity,
        benchmark=benchmark,
        position=position,
        trade_returns=trade_returns,
    )
