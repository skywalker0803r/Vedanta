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

    df["signal"] = 0
    df["signal_reason"] = ""

    if len(df) < 2:
        return df

    # Iterate to check conditions for each point, or use vectorized operations if possible for efficiency
    # For complex Vegas conditions, iteration might be clearer initially.
    # We will apply the logic from telegram_message_bot.py.

    # Apply vectorized logic for efficiency
    # Long Conditions
    vegas_low_curr_long = df[[f'ema_{n1}', f'ema_{n2}']].min(axis=1)
    vegas_high_curr_long = df[[f'ema_{n1}', f'ema_{n2}']].max(axis=1)

    breakout_long = (df['close'].shift(1) < vegas_low_curr_long.shift(1)) & \
                    (df['close'] > vegas_high_curr_long)
    
    bounce_long = (df['close'].shift(1) > vegas_high_curr_long.shift(1)) & \
                  (df['close'] > vegas_high_curr_long) & \
                  ((df['low'].shift(1) <= vegas_high_curr_long.shift(1)) | (df['low'] <= vegas_high_curr_long)) # Corrected bounce logic

    # Short Conditions
    vegas_low_curr_short = df[[f'ema_{n1}', f'ema_{n2}']].min(axis=1)
    vegas_high_curr_short = df[[f'ema_{n1}', f'ema_{n2}']].max(axis=1)

    breakdown_short = (df['close'].shift(1) > vegas_high_curr_short.shift(1)) & \
                      (df['close'] < vegas_low_curr_short)
    
    fail_bounce_short = (df['close'].shift(1) < vegas_low_curr_short.shift(1)) & \
                        (df['close'] < vegas_low_curr_short) & \
                        ((df['high'].shift(1) >= vegas_low_curr_short.shift(1)) | (df['high'] >= vegas_low_curr_short)) # Corrected fail_bounce logic

    # Assign signals based on conditions
    df.loc[breakout_long, "signal"] = 1
    df.loc[breakout_long, "signal_reason"] = "突破 (Breakout)"

    df.loc[bounce_long, "signal"] = 1
    df.loc[bounce_long, "signal_reason"] = "回踩反彈 (Bounce)"

    df.loc[breakdown_short, "signal"] = -1
    df.loc[breakdown_short, "signal_reason"] = "跌破 (Breakdown)"

    df.loc[fail_bounce_short, "signal"] = -1
    df.loc[fail_bounce_short, "signal_reason"] = "反彈失敗 (Failed Bounce)"

    # Prioritize signals if both long and short conditions are met (though unlikely for Vegas)
    # For simplicity, if both are true for a row, the last assigned will stick or we can set a priority.
    # In this case, we'll let them overwrite or add a check if both are true to pick one.
    # For now, it's sequential.

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