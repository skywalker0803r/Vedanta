import requests
import pandas as pd
from datetime import datetime, timedelta
from binance.client import Client
import os
from dotenv import load_dotenv

load_dotenv()

# Binance API Key (可為空)
# Initialize client even if keys are empty to use get_klines and get_ticker
client = Client(api_key=os.getenv("BINANCE_API_KEY", ''), api_secret=os.getenv("BINANCE_API_SECRET", ''))

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 200) -> pd.DataFrame:
    """
    Fetches kline data from Binance using the Binance Client.
    Adjusted to fetch a 'total_limit' number of recent klines.
    """
    try:
        # Binance Client's get_klines does not use endTime directly for a paginated historical fetch
        # similar to the requests.get in sma.py. It fetches 'limit' number of most recent klines.
        # We will fetch 'total_limit' most recent klines.
        klines = client.get_klines(symbol=symbol, interval=interval, limit=total_limit)
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        df = df.sort_values("timestamp").reset_index(drop=True)
        return df
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def detect_ema_cross(df: pd.DataFrame, n1: int = 144, n2: int = 169) -> pd.DataFrame:
    """
    Detects EMA crosses (Vegas strategy) based on ema_n1 and ema_n2.
    Adds 'ema_n1', 'ema_n2', and 'signal' columns to the DataFrame.
    Signal: 1 for long condition (Vegas breakout/bounce), -1 for short condition (Vegas breakdown/fail_bounce).
    """
    df[f'ema_{n1}'] = df['close'].ewm(span=n1, adjust=False).mean()
    df[f'ema_{n2}'] = df['close'].ewm(span=n2, adjust=False).mean()
    df = df.dropna().reset_index(drop=True) # Ensure no NaN rows after EMA calculation

    # Initialize signal and position columns
    df["signal"] = 0
    df["position"] = 0
    df["signal_reason"] = ""

    current_position = 0 # Track current position (1: long, -1: short, 0: flat)
    signals = []
    positions = []
    signal_reasons = []

    for i in range(len(df)):
        # Ensure enough data for indicators
        if i < max(n1, n2): 
            signals.append(0)
            positions.append(0)
            signal_reasons.append("")
            continue

        curr_close = df.loc[i, "close"]
        curr_low = df.loc[i, "low"]
        curr_high = df.loc[i, "high"]
        curr_ema_n1 = df.loc[i, f'ema_{n1}']
        curr_ema_n2 = df.loc[i, f'ema_{n2}']

        prev_close = df.loc[i-1, "close"]
        prev_low = df.loc[i-1, "low"]
        prev_high = df.loc[i-1, "high"]
        prev_ema_n1 = df.loc[i-1, f'ema_{n1}']
        prev_ema_n2 = df.loc[i-1, f'ema_{n2}']

        vegas_low_curr = min(curr_ema_n1, curr_ema_n2)
        vegas_high_curr = max(curr_ema_n1, curr_ema_n2)
        vegas_low_prev = min(prev_ema_n1, prev_ema_n2)
        vegas_high_prev = max(prev_ema_n1, prev_ema_n2)

        current_bar_signal = 0
        current_signal_reason = ""

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if close crosses below Vegas low
            if curr_close < vegas_low_curr and prev_close >= vegas_low_prev:
                current_position = 0
                current_bar_signal = -1 # Exit long signal
                current_signal_reason = "平多 (Vegas Tunnel 下穿)"
        elif current_position == -1: # Currently short
            # Exit if close crosses above Vegas high
            if curr_close > vegas_high_curr and prev_close <= vegas_high_prev:
                current_position = 0
                current_bar_signal = 1 # Exit short signal
                current_signal_reason = "平空 (Vegas Tunnel 上穿)"

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Long Conditions
            breakout_long = (
                (prev_close < vegas_low_prev) and 
                (curr_close > vegas_high_curr)
            )
            
            bounce_long = (
                (prev_close > vegas_high_prev) and 
                (curr_close > vegas_high_curr) and 
                (min(curr_low, prev_low) <= vegas_high_curr) # Corrected bounce logic
            )

            # Short Conditions
            breakdown_short = (
                (prev_close > vegas_high_prev) and 
                (curr_close < vegas_low_curr)
            )
            
            fail_bounce_short = (
                (prev_close < vegas_low_prev) and 
                (curr_close < vegas_low_curr) and 
                (max(curr_high, prev_high) >= vegas_low_curr) # Corrected fail_bounce logic
            )

            if breakout_long:
                current_position = 1
                current_bar_signal = 1
                current_signal_reason = "開多 (突破)"
            elif bounce_long:
                current_position = 1
                current_bar_signal = 1
                current_signal_reason = "開多 (回踩反彈)"
            elif breakdown_short:
                current_position = -1
                current_bar_signal = -1
                current_signal_reason = "開空 (跌破)"
            elif fail_bounce_short:
                current_position = -1
                current_bar_signal = -1
                current_signal_reason = "開空 (反彈失敗)"
        
        signals.append(current_bar_signal)
        positions.append(current_position)
        signal_reasons.append(current_signal_reason)

    df["signal"] = signals
    df["position"] = positions
    df["signal_reason"] = signal_reasons

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 200, n1: int = 144, n2: int = 169) -> pd.DataFrame:
    """
    Combines kline fetching and EMA cross detection.
    Returns a DataFrame with kline data, EMAs, and signals.
    """
    df = get_binance_kline(symbol, interval, end_time, limit)
    if not df.empty:
        df = detect_ema_cross(df, n1=n1, n2=n2)
    return df

# Usage Example
if __name__ == '__main__':
    from datetime import datetime
    
    # Fetch BTCUSDT 1-hour klines ending now, with 200 data points
    # Using Vegas strategy EMAs (144, 169)
    df_signals = get_signals("BTCUSDT", Client.KLINE_INTERVAL_1HOUR, datetime.now(), limit=200, n1=144, n2=169)
    
    if not df_signals.empty:
        print(df_signals[['timestamp', 'close', 'ema_144', 'ema_169', 'signal', 'signal_reason']].tail(10))
        print("\nSignal counts:")
        print(df_signals['signal'].value_counts())
    else:
        print("No data or signals generated.")