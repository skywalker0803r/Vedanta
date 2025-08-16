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

def detect_adx_signal(df: pd.DataFrame, period: int = 14, adx_threshold: int = 25) -> pd.DataFrame:
    df = df.copy()
    high = df["high"]
    low = df["low"]
    close = df["close"]

    plus_dm = pd.Series(0.0, index=df.index)
    minus_dm = pd.Series(0.0, index=df.index)

    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm[(up_move > down_move) & (up_move > 0)] = up_move
    minus_dm[(down_move > up_move) & (down_move > 0)] = down_move
    
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-9))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-9))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx = dx.rolling(period).mean()

    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di

    # Initialize signal and position columns
    df["signal"] = 0
    df["position"] = 0

    current_position = 0 # Track current position (1: long, -1: short, 0: flat)
    signals = []
    positions = []

    for i in range(len(df)):
        # Ensure enough data for indicators
        if i < period * 2: # Need enough bars for ADX calculation
            signals.append(0)
            positions.append(0)
            continue

        curr_adx = df.loc[i, "adx"]
        curr_plus_di = df.loc[i, "plus_di"]
        curr_minus_di = df.loc[i, "minus_di"]

        prev_plus_di = df.loc[i-1, "plus_di"]
        prev_minus_di = df.loc[i-1, "minus_di"]

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if +DI crosses below -DI or ADX drops below threshold
            if (curr_plus_di < curr_minus_di and prev_plus_di >= prev_minus_di) or curr_adx < adx_threshold:
                current_position = 0
                current_bar_signal = -1 # Exit long signal
        elif current_position == -1: # Currently short
            # Exit if -DI crosses below +DI or ADX drops below threshold
            if (curr_minus_di < curr_plus_di and prev_minus_di >= prev_plus_di) or curr_adx < adx_threshold:
                current_position = 0
                current_bar_signal = 1 # Exit short signal

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Buy signal: ADX > threshold and +DI crosses above -DI
            if curr_adx > adx_threshold and curr_plus_di > curr_minus_di and prev_plus_di <= prev_minus_di:
                current_position = 1
                current_bar_signal = 1 # Entry long signal
            # Sell signal: ADX > threshold and -DI crosses above +DI
            elif curr_adx > adx_threshold and curr_plus_di < curr_minus_di and prev_plus_di >= prev_minus_di:
                current_position = -1
                current_bar_signal = -1 # Entry short signal
        
        signals.append(current_bar_signal)
        positions.append(current_position)

    df["signal"] = signals
    df["position"] = positions
    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, period: int = 14, adx_threshold: int = 25) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_adx_signal(df, period=period, adx_threshold=adx_threshold)
    return df

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    # 抓 BTCUSDT 的 1小時線，以現在時間為終點
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(),1000)
    print(df_signals['signal'].value_counts())
