"""
Full Market Pipeline with Binance Data and Signal Generator
===========================================================

This script integrates:
1. Binance OHLCV fetcher
2. Perception layer (features)
3. Analysis layer (day pattern recognizer)
4. Decision layer (strategy dispatcher)
5. Execution layer (sub-strategies)
6. Signal generator: `get_signals()` returns DataFrame with signals and positions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np
import pandas as pd
import requests
from datetime import datetime
import time

# ------------------------------------------------------------
# 0) Binance data source
# ------------------------------------------------------------

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 3000) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_timestamp = int(end_time.timestamp() * 1000)
    remaining = total_limit

    while remaining > 0:
        fetch_limit = min(1000, remaining)
        params = {"symbol": symbol.upper(), "interval": interval, "endTime": end_timestamp, "limit": fetch_limit}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Binance API request failed: {e}")

        data = response.json()
        if not data:
            break

        all_data = data + all_data
        end_timestamp = data[0][0] - 1
        remaining -= len(data)
        time.sleep(0.1)

    if not all_data:
        raise ValueError("Failed to fetch any K-line data.")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "open", "high", "low", "close", "volume"]]

# ------------------------------------------------------------
# 1) Input layer helpers
# ------------------------------------------------------------
REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

def _ensure_datetime_index(df: pd.DataFrame, tz: Optional[str] = None) -> pd.DataFrame:
    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamp" in df.columns:
            df = df.set_index(pd.to_datetime(df["timestamp"], utc=True))
            df = df.drop(columns=["timestamp"])
        else:
            raise ValueError("DataFrame must have a DatetimeIndex or a 'timestamp' column.")
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    if tz is not None:
        df.index = df.index.tz_convert(tz)
    return df.sort_index()

def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {missing}")
    bad = (df[["open", "high", "low", "close"]].min(axis=1) < 0) | (df["volume"] < 0)
    if bad.any():
        df = df[~bad]
    return df

# ------------------------------------------------------------
# 2) Simple Perception Layer (example: moving averages)
# ------------------------------------------------------------

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ma_fast"] = out["close"].rolling(20).mean()
    out["ma_slow"] = out["close"].rolling(50).mean()
    out["atr"] = (out["high"] - out["low"]).rolling(14).mean()
    return out

# ------------------------------------------------------------
# 3) Analysis Layer (placeholder DayPattern)
# ------------------------------------------------------------

def recognize_pattern(df: pd.DataFrame) -> str:
    if df["ma_fast"].iloc[-1] > df["ma_slow"].iloc[-1]:
        return "TrendUp"
    elif df["ma_fast"].iloc[-1] < df["ma_slow"].iloc[-1]:
        return "TrendDown"
    return "Neutral"

# ------------------------------------------------------------
# 4) Decision Layer (Dispatcher)
# ------------------------------------------------------------
@dataclass
class StrategyDispatcher:
    def decide(self, pattern: str, market_state: Dict[str, float]) -> str:
        if pattern.startswith("TrendUp"):
            return "Trend"
        if pattern.startswith("TrendDown"):
            return "Reversal"
        return "Range"

# ------------------------------------------------------------
# 5) Execution Layer (Sub-strategies)
# ------------------------------------------------------------
class ExecutionEngine:
    def __init__(self):
        self._registry: Dict[str, Callable[[pd.DataFrame], str]] = {
            "Trend": self._trend_pullback,
            "Range": self._trading_range_fade,
            "Reversal": self._major_trend_reversal,
        }

    def run(self, family: str, df: pd.DataFrame) -> str:
        fn = self._registry.get(family)
        if fn is None:
            return "NoSignal"
        return fn(df)

    def _trend_pullback(self, df: pd.DataFrame) -> str:
        return "Buy" if df["ma_fast"].iloc[-1] > df["ma_slow"].iloc[-1] else "NoSignal"

    def _trading_range_fade(self, df: pd.DataFrame) -> str:
        return "Sell" if df["close"].iloc[-1] > df["ma_slow"].iloc[-1] else "Buy"

    def _major_trend_reversal(self, df: pd.DataFrame) -> str:
        return "Sell" if df["ma_fast"].iloc[-1] < df["ma_slow"].iloc[-1] else "NoSignal"

# ------------------------------------------------------------
# 6) Signal Generator
# ------------------------------------------------------------

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 1000) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, total_limit=limit)
    df = _ensure_datetime_index(df)
    df = validate_ohlcv(df)
    df = add_indicators(df)

    dispatcher = StrategyDispatcher()
    executor = ExecutionEngine()

    signal_names = []  # keep human-readable internally
    for i in range(len(df)):
        window = df.iloc[:i+1]
        if len(window) < 50:
            signal_names.append(None)
            continue
        pattern = recognize_pattern(window)
        market_state = {"atr": window["atr"].iloc[-1]}
        family = dispatcher.decide(pattern, market_state)
        sig = executor.run(family, window)  # 'Buy' / 'Sell' / 'NoSignal'
        signal_names.append(sig)

    # Map to numeric signals: 1=Buy, -1=Sell, 0=No action
    name_to_num = {"Buy": 1, "Sell": -1, "NoSignal": 0, None: 0}
    df["signal"] = [name_to_num.get(s, 0) for s in signal_names]

    # Build running position: hold last non-zero signal until switched
    pos = 0
    positions = []
    for s in df["signal"].tolist():
        pos = pos if s == 0 else s
        positions.append(pos)
    df["position"] = positions

    # Reset index so 'timestamp' is a column in the output
    df = df.reset_index().rename(columns={"index": "timestamp"})
    return df[["timestamp", "open", "high", "low", "close", "signal", "position"]]

# ------------------------------------------------------------
# Example usage
# ------------------------------------------------------------
if __name__ == "__main__":
    out = get_signals("BTCUSDT", "1d", datetime.utcnow(), limit=500)
    print(out.tail(50))
