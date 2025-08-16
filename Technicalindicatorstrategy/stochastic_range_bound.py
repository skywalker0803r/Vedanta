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

def detect_stochastic_range_strategy_optimized(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
    oversold: int = 20,
    overbought: int = 80,
    trend_ema_period: int = 200,
    atr_period: int = 14,
    atr_threshold: float = 0.5
) -> pd.DataFrame:
    df = df.copy()
    # Stochastic %K, %D
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    df["%K"] = 100 * (df["close"] - low_min) / (high_max - low_min)
    df["%D"] = df["%K"].rolling(window=d_period).mean()

    # 長期趨勢EMA
    df[f"ema_{trend_ema_period}"] = df["close"].ewm(span=trend_ema_period, adjust=False).mean()

    # ATR 計算
    df["tr"] = df["high"] - df["low"]
    df["atr"] = df["tr"].rolling(window=atr_period).mean()

    # Initialize signal and position columns
    df["signal"] = 0
    df["position"] = 0

    current_position = 0 # Track current position (1: long, -1: short, 0: flat) 
    signals = []
    positions = []

    for i in range(len(df)):
        # Ensure enough data for indicators
        max_period = max(k_period, d_period, trend_ema_period, atr_period)
        if i < max_period: 
            signals.append(0)
            positions.append(0)
            continue

        curr_close = df.loc[i, "close"]
        curr_k = df.loc[i, "%K"]
        curr_d = df.loc[i, "%D"]
        curr_ema = df.loc[i, f"ema_{trend_ema_period}"]
        curr_atr = df.loc[i, "atr"]

        prev_k = df.loc[i-1, "%K"]
        prev_d = df.loc[i-1, "%D"]

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if %K crosses below %D OR price crosses below long-term EMA
            if (curr_k < curr_d and prev_k >= prev_d) or (curr_close < curr_ema):
                current_position = 0
                current_bar_signal = -1 # Exit long signal
        elif current_position == -1: # Currently short
            # Exit if %K crosses above %D OR price crosses above long-term EMA
            if (curr_k > curr_d and prev_k <= prev_d) or (curr_close > curr_ema):
                current_position = 0
                current_bar_signal = 1 # Exit short signal

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Buy signal: Price above long-term EMA, %K crosses above %D, %K is oversold, AND ATR is above atr_threshold
            if (curr_close > curr_ema) and \
               (curr_k > curr_d and prev_k <= prev_d) and \
               (curr_k < oversold) and \
               (curr_atr > atr_threshold):
                current_position = 1
                current_bar_signal = 1 # Entry long signal
            # Sell signal: Price below long-term EMA, %K crosses below %D, %K is overbought, AND ATR is above atr_threshold
            elif (curr_close < curr_ema) and \
                 (curr_k < curr_d and prev_k >= prev_d) and \
                 (curr_k > overbought) and \
                 (curr_atr > atr_threshold):
                current_position = -1
                current_bar_signal = -1 # Entry short signal
        
        signals.append(current_bar_signal)
        positions.append(current_position)

    df["signal"] = signals
    df["position"] = positions
    return df

    # 長期趨勢EMA
    df[f"ema_{trend_ema_period}"] = df["close"].ewm(span=trend_ema_period, adjust=False).mean()

    # ATR 計算
    df["tr"] = df["high"] - df["low"]
    df["atr"] = df["tr"].rolling(window=atr_period).mean()

    df["signal"] = 0
    # 多頭趨勢且超賣區買進訊號
    buy_signal = (
        (df["close"] > df[f"ema_{trend_ema_period}"]) &
        (df["%K"] > df["%D"]) &
        (df["%K"].shift(1) <= df["%D"].shift(1)) &
        (df["%K"] < oversold) &
        (df["atr"] > atr_threshold)
    )
    # 空頭趨勢且超買區賣出訊號
    sell_signal = (
        (df["close"] < df[f"ema_{trend_ema_period}"]) &
        (df["%K"] < df["%D"]) &
        (df["%K"].shift(1) >= df["%D"].shift(1)) &
        (df["%K"] > overbought) &
        (df["atr"] > atr_threshold)
    )
    df.loc[buy_signal, "signal"] = 1
    df.loc[sell_signal, "signal"] = -1

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_stochastic_range_strategy_optimized(df)
    return df
