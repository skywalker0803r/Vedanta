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

def detect_sma_cross(df:pd.DataFrame,n1:int=1,n2:int=2) -> pd.DataFrame:
    df["sma_1"] = df["close"].rolling(window=n1).mean()
    df["sma_2"] = df["close"].rolling(window=n2).mean()

    # Initialize signal and position columns
    df["signal"] = 0
    df["position"] = 0

    current_position = 0 # Track current position (1: long, -1: short, 0: flat)
    signals = []
    positions = []

    for i in range(len(df)):
        # Ensure enough data for indicators
        if i < max(n1, n2): # Need enough bars for SMA calculation
            signals.append(0)
            positions.append(0)
            continue

        curr_sma1 = df.loc[i, "sma_1"]
        curr_sma2 = df.loc[i, "sma_2"]
        prev_sma1 = df.loc[i-1, "sma_1"]
        prev_sma2 = df.loc[i-1, "sma_2"]

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if SMA1 crosses below SMA2
            if curr_sma1 < curr_sma2 and prev_sma1 >= prev_sma2:
                current_position = 0
                current_bar_signal = -1 # Exit long signal
        elif current_position == -1: # Currently short
            # Exit if SMA1 crosses above SMA2
            if curr_sma1 > curr_sma2 and prev_sma1 <= prev_sma2:
                current_position = 0
                current_bar_signal = 1 # Exit short signal

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Buy signal: SMA1 crosses above SMA2
            if curr_sma1 > curr_sma2 and prev_sma1 <= prev_sma2:
                current_position = 1
                current_bar_signal = 1 # Entry long signal
            # Sell signal: SMA1 crosses below SMA2
            elif curr_sma1 < curr_sma2 and prev_sma1 >= prev_sma2:
                current_position = -1
                current_bar_signal = -1 # Entry short signal
        
        signals.append(current_bar_signal)
        positions.append(current_position)

    df["signal"] = signals
    df["position"] = positions
    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, n1: int = 1, n2: int = 2) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_sma_cross(df, n1=n1, n2=n2)
    return df

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    # 抓 BTCUSDT 的 1小時線，以現在時間為終點
    df_signals = get_signals("BTCUSDT", "1m", datetime.now(),300)
    print(df_signals['signal'].value_counts())
