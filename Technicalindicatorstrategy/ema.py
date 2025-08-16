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

def detect_ema_cross(df: pd.DataFrame, n1: int = 5, n2: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["ema_1"] = df["close"].ewm(span=n1, adjust=False).mean()
    df["ema_2"] = df["close"].ewm(span=n2, adjust=False).mean()

    # Initialize signal and position columns
    df["signal"] = 0
    df["position"] = 0

    current_position = 0 # Track current position (1: long, -1: short, 0: flat)
    signals = []
    positions = []

    for i in range(len(df)):
        # Ensure enough data for indicators
        if i < n2: # Need enough bars for EMA calculation
            signals.append(0)
            positions.append(0)
            continue

        curr_ema1 = df.loc[i, "ema_1"]
        curr_ema2 = df.loc[i, "ema_2"]
        prev_ema1 = df.loc[i-1, "ema_1"]
        prev_ema2 = df.loc[i-1, "ema_2"]

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if EMA1 crosses below EMA2
            if curr_ema1 < curr_ema2 and prev_ema1 >= prev_ema2:
                current_position = 0
                current_bar_signal = -1 # Exit long signal
        elif current_position == -1: # Currently short
            # Exit if EMA1 crosses above EMA2
            if curr_ema1 > curr_ema2 and prev_ema1 <= prev_ema2:
                current_position = 0
                current_bar_signal = 1 # Exit short signal

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Buy signal: EMA1 crosses above EMA2
            if curr_ema1 > curr_ema2 and prev_ema1 <= prev_ema2:
                current_position = 1
                current_bar_signal = 1 # Entry long signal
            # Sell signal: EMA1 crosses below EMA2
            elif curr_ema1 < curr_ema2 and prev_ema1 >= prev_ema2:
                current_position = -1
                current_bar_signal = -1 # Entry short signal
        
        signals.append(current_bar_signal)
        positions.append(current_position)

    df["signal"] = signals
    df["position"] = positions
    return df


def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, n1: int = 5, n2: int = 20) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_ema_cross(df, n1=n1, n2=n2)
    return df

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    # 抓 BTCUSDT 的 1小時線，以現在時間為終點
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(),300)
    print(df_signals['signal'].value_counts())
