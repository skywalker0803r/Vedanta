"""
This script provides a function to generate trading signals based on the mixed strategy
(Donchian/SMA/RSI for long, EMA/SMA/RSI for short) without running a full backtest.
"""

import ccxt
import time
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ------------------------- UTILS / INDICATORS -------------------------

def timeframe_to_ms(timeframe):
    """Converts a timeframe string (e.g., '1h', '4h') to milliseconds."""
    return ccxt.Timeframe.parse(timeframe) * 1000

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    import time
    import requests
    import pandas as pd

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

        response = requests.get(base_url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data = data + all_data  # prepend older data
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

        time.sleep(0.5)  # sleep after request to avoid rate limits

    if not all_data:
        raise ValueError("No data fetched")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    # 轉成 UTC 時區
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]

def sma(series, n):
    return series.rolling(n, min_periods=1).mean()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

def get_signals(symbol: str, timeframe: str, end_time: datetime, n_bars: int):
    """
    Generates trading signals for a given symbol and timeframe.

    Args:
        symbol (str): The trading symbol (e.g., 'ADA/USDT').
        timeframe (str): The timeframe (e.g., '4h').
        end_time (datetime): The end time for the data fetch.
        n_bars (int): The number of bars to fetch.

    Returns:
        pandas.DataFrame: A DataFrame with OHLCV data and signal columns:
            - timestamp, open, high, low, close
            - position: 1 for long, -1 for short, 0 for flat.
            - entry_price: The price of the entry signal.
            - stop_loss: The calculated stop loss for the active position.
            - signal: A string describing the signal (e.g., 'Long Entry').
            - reason: A string describing the reason for the signal.
    """
    print(f"Fetching {n_bars} bars for {symbol} on timeframe {timeframe} until {end_time.isoformat()}...")
    df = get_binance_kline(symbol, timeframe, end_time,n_bars)
    print(f"Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}\n")

    # ------------------------- STRATEGY PARAMETERS -------------------------
    donchianLength = 12
    longTermSmaLen = 150
    rsiLenLong = 30
    rsiThLong = 60.0
    emaFastLength = 6
    smaSlowLength = 65
    rsiLenShort = 65
    rsiShortThresh = 50
    shortTPPct = 10.0
    shortSLPct = 5.0
    trailTriggerPct = 8.0
    trailOffsetPct = 4.0
    maxConsecLosses = 1
    cooldownBars = 12

    # ------------------------- INDICATOR CALCULATION -------------------------
    df['longTermSma'] = sma(df['close'], longTermSmaLen)
    df['rsiLong'] = rsi(df['close'], rsiLenLong)
    df['upperBand'] = df['high'].rolling(donchianLength).max().shift(1)
    df['lowerBand'] = df['low'].rolling(donchianLength).min().shift(1)
    df['emaFast'] = ema(df['close'], emaFastLength)
    df['smaSlow'] = sma(df['close'], smaSlowLength)
    df['rsiShort'] = rsi(df['close'], rsiLenShort)

    min_required = max(longTermSmaLen, donchianLength, smaSlowLength, rsiLenLong, rsiLenShort)
    df = df.iloc[min_required:].copy()

    # ------------------------- SIGNAL GENERATION LOOP -------------------------
    position_col = []
    entry_price_col = []
    stop_loss_col = []
    signal_col = []
    reason_col = []

    current_pos = 0
    current_entry_price = 0.0
    trail_stop_price_short = 0.0
    shortLossCount = 0
    shortCooldownUntilBarIdx = -1

    for i, row in enumerate(df.itertuples()):
        close, high, low = row.close, row.high, row.low
        inShortCooldown = i < shortCooldownUntilBarIdx
        
        signal_text, reason_text, sl_price = '', '', np.nan

        # --- Exit Logic ---
        if current_pos == 1 and low <= row.lowerBand:
            signal_text, reason_text = 'Long Exit', 'Stop Loss Hit (Lower Band)'
            current_pos = 0
            current_entry_price = 0.0
        
        elif current_pos == -1:
            exit_price, exit_type = 0, ''
            prev = df.iloc[i-1]
            if prev.emaFast <= prev.smaSlow and row.emaFast > row.smaSlow:
                exit_price, exit_type = close, 'Reversal'
            else:
                shortTP = current_entry_price * (1 - shortTPPct / 100.0)
                shortSL = current_entry_price * (1 + shortSLPct / 100.0)
                stop_price_for_bar = trail_stop_price_short if trail_stop_price_short > 0 else shortSL
                if low <= current_entry_price * (1 - trailTriggerPct / 100.0):
                    new_trail_stop = close * (1 + trailOffsetPct / 100.0)
                    stop_price_for_bar = min(stop_price_for_bar, new_trail_stop)
                    trail_stop_price_short = stop_price_for_bar
                
                if high >= stop_price_for_bar: exit_price, exit_type = stop_price_for_bar, 'Stop'
                elif low <= shortTP: exit_price, exit_type = shortTP, 'TP'
            
            if exit_price > 0:
                signal_text, reason_text = 'Short Exit', exit_type
                pnl = current_entry_price - exit_price
                if pnl < 0:
                    shortLossCount += 1
                    if shortLossCount >= maxConsecLosses: shortCooldownUntilBarIdx = i + cooldownBars
                else: shortLossCount = 0
                current_pos, current_entry_price, trail_stop_price_short = 0, 0.0, 0.0

        # --- Entry Logic ---
        if current_pos == 0:
            prev = df.iloc[i-1]
            if (close > row.longTermSma) and (prev.close <= prev.upperBand) and (close > row.upperBand) and (row.rsiLong > rsiThLong):
                signal_text, reason_text = 'Long Entry', 'Donchian Breakout + SMA/RSI Filter'
                current_pos = 1
                current_entry_price = close
            elif not inShortCooldown and (prev.emaFast > prev.smaSlow) and (row.emaFast <= row.smaSlow) and (row.rsiShort < rsiShortThresh):
                signal_text, reason_text = 'Short Entry', 'EMA/SMA Crossunder + RSI Filter'
                current_pos = -1
                current_entry_price = close

        # --- Set Stop Loss for active positions ---
        if current_pos == 1:
            sl_price = row.lowerBand
        elif current_pos == -1:
            sl_price = trail_stop_price_short if trail_stop_price_short > 0 else current_entry_price * (1 + shortSLPct / 100.0)

        position_col.append(current_pos)
        entry_price_col.append(current_entry_price if current_pos != 0 else np.nan)
        stop_loss_col.append(sl_price)
        signal_col.append(signal_text)
        reason_col.append(reason_text)

    df['position'] = position_col
    df['entry_price'] = entry_price_col
    df['stop_loss'] = stop_loss_col
    df['signal'] = signal_col
    df['reason'] = reason_col
    
    final_df = df[['timestamp', 'open', 'high', 'low', 'close', 'position', 'entry_price', 'stop_loss', 'signal', 'reason']].copy()
    return final_df

if __name__ == '__main__':
    from datetime import datetime
    
    # Example usage: Get signals for ADA/USDT 4h from 2021-01-01 up to now.
    # Note: 10257 was the number of 4h bars from 2021-01-01 to Sep 2024. Adjust if needed.
    signals_df = get_signals(
        symbol='ADA/USDT',
        timeframe='4h',
        end_time=datetime.now(),
        n_bars=10257 
    )
    
    print("\n--- Last 5 Signals ---")
    print(signals_df[signals_df['signal'] != ''].tail())

    print("\n--- Current Position Status ---")
    print(signals_df.tail(1))