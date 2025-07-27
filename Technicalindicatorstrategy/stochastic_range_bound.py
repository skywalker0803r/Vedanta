import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 300) -> pd.DataFrame:
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

        all_data = data + all_data
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)

def detect_stochastic_range_strategy(df: pd.DataFrame,
                                     k_period: int = 14,
                                     d_period: int = 3,
                                     oversold: int = 20,
                                     overbought: int = 80) -> pd.DataFrame:
    df = df.copy()
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()

    df["%K"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["%D"] = df["%K"].rolling(window=d_period).mean()

    df["signal"] = 0
    buy_signal = (df["%K"] > df["%D"]) & (df["%K"].shift(1) <= df["%D"].shift(1)) & (df["%K"] < oversold)
    sell_signal = (df["%K"] < df["%D"]) & (df["%K"].shift(1) >= df["%D"].shift(1)) & (df["%K"] > overbought)

    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_stochastic_range_strategy(df)
    return df
