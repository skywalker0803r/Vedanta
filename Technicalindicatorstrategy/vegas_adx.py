import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

def compute_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    df = df.copy()
    delta_high = df["high"].diff()
    delta_low = df["low"].diff()

    plus_dm = np.where((delta_high > delta_low) & (delta_high > 0), delta_high, 0)
    minus_dm = np.where((delta_low > delta_high) & (delta_low > 0), delta_low, 0)

    tr1 = df["high"] - df["low"]
    tr2 = abs(df["high"] - df["close"].shift())
    tr3 = abs(df["low"] - df["close"].shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(period).sum() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(period).sum() / atr
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    return adx


def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 3000) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    end_timestamp = int(end_time.timestamp() * 1000)
    remaining = total_limit

    while remaining > 0:
        fetch_limit = min(1000, remaining)
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "endTime": end_timestamp,
            "limit": fetch_limit
        }

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data = data + all_data  # prepend to maintain chronological order
        end_timestamp = data[0][0] - 1  # set next end_time to the timestamp before the earliest one
        remaining -= len(data)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_vegas_signals(df: pd.DataFrame,
                                 ema_1: int = 9,
                                 ema_2: int = 21,
                                 fib_targets: list = [1.618, 2.0, 2.618],
                                 adx_threshold: float = 25.0,
                                 adx_period: int = 7,
                                 rsi_period: int = 14,
                                 rsi_lower: float = 30,
                                 rsi_upper: float = 70
                                ) -> pd.DataFrame:
    df = df.copy()
    df[f"ema_{ema_1}"] = df["close"].ewm(span=ema_1, adjust=False).mean()
    df[f"ema_{ema_2}"] = df["close"].ewm(span=ema_2, adjust=False).mean()
    tunnel_low = df[f"ema_{ema_1}"]
    tunnel_high = df[f"ema_{ema_2}"]
    tunnel_width = tunnel_high - tunnel_low

    df["adx"] = compute_adx(df, period=adx_period)
    df["rsi"] = compute_rsi(df, period=rsi_period)

    df["signal"] = 0
    df["entry_price"] = np.nan
    df["tp1"] = np.nan
    df["tp2"] = np.nan
    df["tp3"] = np.nan

    buy_condition = (
        (df["close"].shift(1) > tunnel_low.shift(1)) &
        (df["close"].shift(1) < tunnel_high.shift(1)) &
        (df["close"] > tunnel_high) &
        (df["adx"] > adx_threshold) &
        (df["rsi"] > rsi_lower) & (df["rsi"] < rsi_upper)
    )

    sell_condition = (
        (df["close"].shift(1) > tunnel_low.shift(1)) &
        (df["close"].shift(1) < tunnel_high.shift(1)) &
        (df["close"] < tunnel_low) &
        (df["adx"] > adx_threshold) &
        (df["rsi"] > rsi_lower) & (df["rsi"] < rsi_upper)
    )

    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1

    for idx in df.index:
        if df.at[idx, "signal"] == 1:
            entry = df.at[idx, "close"]
            width = tunnel_width.at[idx]
            df.at[idx, "entry_price"] = entry
            df.at[idx, "tp1"] = entry + width * fib_targets[0]
            df.at[idx, "tp2"] = entry + width * fib_targets[1]
            df.at[idx, "tp3"] = entry + width * fib_targets[2]
        elif df.at[idx, "signal"] == -1:
            entry = df.at[idx, "close"]
            width = tunnel_width.at[idx]
            df.at[idx, "entry_price"] = entry
            df.at[idx, "tp1"] = entry - width * fib_targets[0]
            df.at[idx, "tp2"] = entry - width * fib_targets[1]
            df.at[idx, "tp3"] = entry - width * fib_targets[2]

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 3000) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_vegas_signals(df)
    return df