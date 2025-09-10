import pandas as pd
import requests
from datetime import datetime
import numpy as np
import time

# =======================
# 1. Data Fetching
# =======================
def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    """
    Fetches K-line data from Binance API (UTC timezone).
    """
    time.sleep(0.2) # To avoid hitting rate limits
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
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"Binance API request failed: {e}")

        data = response.json()
        if not data:
            break

        all_data = data + all_data
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

    if not all_data:
        raise ValueError("No data fetched")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[[ "open", "high", "low", "close"]] = df[[ "open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[[ "timestamp", "open", "high", "low", "close"]]

# =======================
# 2. Indicator Calculation
# =======================
def calculate_indicators(df: pd.DataFrame, rsi_len: int, atr_len: int, ema_lower_len: int, ema_upper_len: int):
    """Calculates all necessary indicators and FVG signals."""
    df_copy = df.copy()

    # RSI
    delta = df_copy['close'].diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/rsi_len, adjust=False).mean()
    ma_down = down.ewm(alpha=1/rsi_len, adjust=False).mean()
    rs = ma_up / ma_down
    df_copy['rsi'] = 100 - (100 / (1 + rs))

    # ATR
    df_copy['H-L'] = df_copy['high'] - df_copy['low']
    df_copy['H-PC'] = abs(df_copy['high'] - df_copy['close'].shift(1))
    df_copy['L-PC'] = abs(df_copy['low'] - df_copy['close'].shift(1))
    tr = df_copy[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df_copy['atr'] = tr.rolling(window=atr_len).mean()

    # EMA
    df_copy['ema_lower'] = df_copy['close'].ewm(span=ema_lower_len, adjust=False).mean()
    df_copy['ema_upper'] = df_copy['close'].ewm(span=ema_upper_len, adjust=False).mean()

    # FVG (Fair Value Gap)
    # fvg_up (bearish): high of 12 bars ago is lower than the current low
    df_copy['fvg_up'] = df_copy['high'].shift(12) < df_copy['low']
    # fvg_down (bullish): low of 12 bars ago is higher than the current high
    df_copy['fvg_down'] = df_copy['low'].shift(12) > df_copy['high']

    return df_copy

# =======================
# 3. Signal Generation
# =======================
def generate_signals(df: pd.DataFrame, rsi_overbought: int, rsi_oversold: int, atr_tp_multiplier: float, sl_pct: float):
    """Generates trading signals based on the calculated indicators."""
    df_copy = df.copy()

    signals = []
    positions = []
    reasons = []
    
    in_trade = False
    is_long = False
    entry_price = 0.0
    take_profit_price = 0.0
    stop_loss_price = 0.0

    for i, row in df_copy.iterrows():
        # Skip initial rows where indicators are not yet valid
        if i < max(20, 100): # Based on EMA lengths
            signals.append(0)
            positions.append(0)
            reasons.append("")
            continue

        signal = 0
        reason = ""

        # --- Exit Logic ---
        if in_trade:
            if is_long:
                if row['close'] >= take_profit_price:
                    signal, reason = -1, "Long TP"
                    in_trade = False
                    is_long = False
                elif row['close'] <= stop_loss_price:
                    signal, reason = -1, "Long SL"
                    in_trade = False
                    is_long = False
            else: # is_short
                if row['close'] <= take_profit_price:
                    signal, reason = 1, "Short TP"
                    in_trade = False
                elif row['close'] >= stop_loss_price:
                    signal, reason = 1, "Short SL"
                    in_trade = False
        
        # --- Entry Logic (only if not in a trade) ---
        if not in_trade:
            # Bullish Entry
            is_bullish_signal = (
                row['fvg_down'] and
                row['close'] > row['open'] and
                row['rsi'] < rsi_oversold and
                row['close'] < row['ema_lower'] and
                row['close'] < row['ema_upper']
            )
            if is_bullish_signal:
                signal, reason = 1, "Long Entry"
                in_trade = True
                is_long = True
                entry_price = row['close']
                take_profit_price = entry_price + row['atr'] * atr_tp_multiplier
                stop_loss_price = entry_price * (1 - sl_pct / 100)

            # Bearish Entry
            is_bearish_signal = (
                row['fvg_up'] and
                row['close'] < row['open'] and
                row['rsi'] > rsi_overbought and
                row['close'] > row['ema_lower'] and
                row['close'] > row['ema_upper']
            )
            if is_bearish_signal:
                signal, reason = -1, "Short Entry"
                in_trade = True
                is_long = False
                entry_price = row['close']
                take_profit_price = entry_price - row['atr'] * atr_tp_multiplier
                stop_loss_price = entry_price * (1 + sl_pct / 100)

        signals.append(signal)
        positions.append(1 if is_long else -1 if in_trade and not is_long else 0)
        reasons.append(reason)

    df_copy['signal'] = signals
    df_copy['position'] = positions
    df_copy['reason'] = reasons
    return df_copy

# =======================
# 4. Main Flow
# =======================
def get_signals(
    symbol: str,
    interval: str,
    end_time: datetime,
    limit: int = 2000,
    rsi_len: int = 19,
    rsi_overbought: int = 74,
    rsi_oversold: int = 26,
    atr_len: int = 30,
    atr_tp_multiplier: float = 4.30,
    sl_pct: float = 2.7,
    ema_lower_len: int = 49,
    ema_upper_len: int = 135
):
    """
    Main function to get trading signals for the FVG RSI strategy.
    """
    # 1. Fetch data
    df = get_binance_kline(symbol, interval, end_time, limit)
    
    # 2. Calculate indicators
    df_indicators = calculate_indicators(df, rsi_len, atr_len, ema_lower_len, ema_upper_len)
    
    # 3. Generate signals
    df_signals = generate_signals(df_indicators, rsi_overbought, rsi_oversold, atr_tp_multiplier, sl_pct)
    
    return df_signals

# =======================
# 5. Example Usage
# =======================
if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1m"
    end_time = datetime.utcnow()
    
    print(f"Fetching data for {symbol} on {interval} timeframe...")
    
    try:
        signals_df = get_signals(
            symbol=symbol,
            interval=interval,
            end_time=end_time,
            limit=5000 # Fetch more data for 1m timeframe
        )
        
        # Print last 30 rows with signals
        print("\n--- Last 30 Signals ---")
        print(signals_df[signals_df['signal'] != 0][['timestamp', 'close', 'signal', 'position', 'reason']].tail(30))

        # Print current status
        print("\n--- Current Status ---")
        print(signals_df[['timestamp', 'close', 'position', 'reason']].tail(1))

    except (ValueError, RuntimeError) as e:
        print(f"An error occurred: {e}")
