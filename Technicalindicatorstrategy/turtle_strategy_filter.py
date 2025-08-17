import pandas as pd
import requests
from datetime import datetime
import numpy as np
import time

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

def calculate_atr(df, period=20):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# --- 新增手動計算 MACD 的函數 ---
def calculate_macd(close_series, fast_period=12, slow_period=26, signal_period=9):
    # 計算 EMA
    exp1 = close_series.ewm(span=fast_period, adjust=False).mean()
    exp2 = close_series.ewm(span=slow_period, adjust=False).mean()
    
    # 計算 MACD 線
    macd_line = exp1 - exp2
    
    # 計算 Signal 線
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    
    # 計算 MACD 柱狀體
    macd_hist = macd_line - signal_line
    
    return macd_hist

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 100):
    df = get_binance_kline(symbol, interval, end_time, limit)

    # 計算高低點，並移位避免用到當根
    df['20_high'] = df['high'].rolling(window=20).max().shift(1)
    df['20_low'] = df['low'].rolling(window=20).min().shift(1)

    # 計算 ATR，並移位避免看未來
    df['ATR'] = calculate_atr(df, 20).shift(1)

    # 手動計算 MACD 柱狀體並加入 DataFrame
    df['MACDh'] = calculate_macd(df['close'])

    # 初始狀態
    current_position = 0 # 追蹤當前倉位
    entry_price = np.nan
    stop_loss = np.nan
    prev_macd_hist = np.nan  # 追蹤前一期的 MACD 柱狀體

    # 記錄訊號和倉位
    signals = []
    positions = []
    entry_prices = []
    stop_losses = []
    reasons = [] # 新增原因欄位

    for i in range(len(df)):
        # 需等待足夠的 K 棒來計算所有指標 (EMA 至少需要26期)
        if i < 26:
            signals.append(0)
            positions.append(0)
            entry_prices.append(np.nan)
            stop_losses.append(np.nan)
            reasons.append("") # 初始狀態無原因
            continue

        current_close = df.loc[i, 'close']
        atr = df.loc[i, 'ATR']
        high_20 = df.loc[i, '20_high']
        low_20 = df.loc[i, '20_low']
        macd_hist = df.loc[i, 'MACDh']
        
        current_signal = 0
        current_reason = ""

        # --- 判斷出場 ---
        if current_position == 1:  # 多單持倉
            # 柱狀體反轉向下或觸發停損
            if macd_hist < prev_macd_hist:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                current_signal = -1  # 訊號改為 -1，代表平倉
                current_reason = "多單平倉"
            elif current_close < stop_loss:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                current_signal = -1  # 訊號改為 -1，代表平倉
                current_reason = "多單平倉"
        elif current_position == -1:  # 空單持倉
            # 柱狀體反轉向上或觸發停損
            if macd_hist > prev_macd_hist:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                current_signal = 1  # 訊號改為 1，代表平倉
                current_reason = "空單平倉"
            elif current_close > stop_loss:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                current_signal = 1  # 訊號改為 1，代表平倉
                current_reason = "空單平倉"

        # --- 判斷進場 ---
        if current_position == 0 and current_signal == 0:
            if current_close > high_20:  # 突破 20 日高
                current_position = 1
                entry_price = current_close
                stop_loss = entry_price - 2 * atr
                current_signal = 1  # 訊號改為 1，代表開多
                current_reason = "多單進場"
            elif current_close < low_20:  # 跌破 20 日低
                current_position = -1
                entry_price = current_close
                stop_loss = entry_price + 2 * atr
                current_signal = -1  # 訊號改為 -1，代表開空
                current_reason = "空單進場"
        
        # 更新 MACD 柱狀體的前一個值
        prev_macd_hist = macd_hist

        signals.append(current_signal)
        positions.append(current_position) # Append the actual current_position
        entry_prices.append(entry_price)
        stop_losses.append(stop_loss)
        reasons.append(current_reason)

    df['position'] = positions
    df['entry_price'] = entry_prices
    df['stop_loss'] = stop_losses
    df['signal'] = signals
    df['reason'] = reasons # 將原因欄位加入回傳的 DataFrame
    return df

if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    end_time = datetime.utcnow()
    df_signals = get_signals(symbol, interval, end_time, limit=500)
    print(df_signals[['timestamp', 'close', 'position', 'signal', 'MACDh']].tail(30))