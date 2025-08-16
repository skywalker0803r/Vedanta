import requests
import pandas as pd
from datetime import datetime, timedelta

def get_binance_kline(symbol: str, interval: str, end_time: datetime, limit: int = 300) -> pd.DataFrame:
    base_url = "https://api.binance.com/api/v3/klines"
    end_timestamp = int(end_time.timestamp() * 1000)
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "endTime": end_timestamp,
        "limit": limit
    }
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]]

def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()

    rs = avg_gain / (avg_loss + 1e-9)  # 加一點避免除以零
    rsi = 100 - (100 / (1 + rs))
    return rsi

def detect_rsi_signal(df: pd.DataFrame, period: int = 14,
                      low_thresh: float = 40, high_thresh: float = 60) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = compute_rsi(df, period)

    # Initialize signal and position columns
    df["signal"] = 0
    df["position"] = 0

    current_position = 0 # Track current position (1: long, -1: short, 0: flat)
    signals = []
    positions = []

    for i in range(len(df)):
        # Ensure enough data for indicators
        if i < period: # Need enough bars for RSI calculation
            signals.append(0)
            positions.append(0)
            continue

        curr_rsi = df.loc[i, "rsi"]
        prev_rsi = df.loc[i-1, "rsi"]

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if RSI crosses below high_thresh
            if curr_rsi < high_thresh and prev_rsi >= high_thresh:
                current_position = 0
                current_bar_signal = -1 # Exit long signal
        elif current_position == -1: # Currently short
            # Exit if RSI crosses above low_thresh
            if curr_rsi > low_thresh and prev_rsi <= low_thresh:
                current_position = 0
                current_bar_signal = 1 # Exit short signal

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Buy signal: RSI crosses above low_thresh
            if curr_rsi > low_thresh and prev_rsi <= low_thresh:
                current_position = 1
                current_bar_signal = 1 # Entry long signal
            # Sell signal: RSI crosses below high_thresh
            elif curr_rsi < high_thresh and prev_rsi >= high_thresh:
                current_position = -1
                current_bar_signal = -1 # Entry short signal
        
        signals.append(current_bar_signal)
        positions.append(current_position)

    df["signal"] = signals
    df["position"] = positions
    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, period: int = 14, low_thresh: float = 40, high_thresh: float = 60) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_rsi_signal(df, period=period, low_thresh=low_thresh, high_thresh=high_thresh)
    return df

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    # 抓 BTCUSDT 的 1小時線，以現在時間為終點
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(),300)
    print(df_signals['signal'].value_counts())
