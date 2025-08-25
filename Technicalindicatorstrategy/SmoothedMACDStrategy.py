import pandas as pd
import requests
from datetime import datetime, timedelta, timezone
import time

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 3000) -> pd.DataFrame:
    time.sleep(1)
    """
    從幣安 API 獲取 K 線數據。
    ... (此處的程式碼與之前相同，故省略) ...
    """
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
        except Exception as e:
            print(f"Error fetching data: {e}")
            break

        data = response.json()

        if not data:
            print("No more data returned from API.")
            break

        all_data = data + all_data
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

    if not all_data:
        raise ValueError("Failed to fetch any K-line data.")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)

    return df[["timestamp", "open", "high", "low", "close"]]

### 使用 $\alpha = 1/N$ 的 `get_signals_macd` 函數

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 3000) -> pd.DataFrame:
    """
    從幣安 API 獲取 K 線數據，並根據 MACD 指標產生買賣訊號。
    本函數使用 alpha = 1/N 的方式計算 EMA。

    Args:
        symbol (str): 交易對符號 (例如 "BTCUSDT")。
        interval (str): K 線間隔 (例如 "1h", "4h", "1d")。
        end_time (datetime): 結束時間。
        limit (int): 要獲取的 K 線總數。

    Returns:
        pd.DataFrame: 包含 MACD 數據和買賣訊號的 DataFrame。
    """
    df = get_binance_kline(symbol, interval, end_time, total_limit=limit + 300)

    # 計算 EMA，使用 alpha = 1/N 的邏輯
    # pandas 的 ewm 方法可以透過 alpha 參數手動指定平滑常數
    alpha_12 = 1 / 12
    alpha_26 = 1 / 26
    alpha_9 = 1 / 9
    
    df['ema12'] = df['close'].ewm(alpha=alpha_12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(alpha=alpha_26, adjust=False).mean()
    
    # 計算 MACD 線
    df['macd'] = df['ema12'] - df['ema26']
    
    # 計算信號線 (MACD 的 9 日 EMA)
    df['signal_line'] = df['macd'].ewm(alpha=alpha_9, adjust=False).mean()
    
    # 計算 MACD 柱狀圖 (Histogram)
    df['histogram'] = df['macd'] - df['signal_line']
    
    # 初始化訊號和倉位欄位
    df["signal"] = 0
    df["action"] = ""
    df["position"] = 0 # 新增 position 欄位

    current_position = 0 # 追蹤當前倉位
    signals = []
    positions = []
    actions = []

    # 判斷買賣訊號 (金叉與死叉)
    for i in range(1, len(df)):
        current_bar_signal = 0
        current_bar_action = ""

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if MACD crosses below Signal Line
            if df.at[i-1, 'macd'] > df.at[i-1, 'signal_line'] and \
               df.at[i, 'macd'] < df.at[i, 'signal_line']:
                current_position = 0
                current_bar_signal = -1 # Exit long signal
                current_bar_action = "平多"
        elif current_position == -1: # Currently short
            # Exit if MACD crosses above Signal Line
            if df.at[i-1, 'macd'] < df.at[i-1, 'signal_line'] and \
               df.at[i, 'macd'] > df.at[i, 'signal_line']:
                current_position = 0
                current_bar_signal = 1 # Exit short signal
                current_bar_action = "平空"

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # MACD 金叉 (買入訊號)
            if df.at[i-1, 'macd'] < df.at[i-1, 'signal_line'] and \
               df.at[i, 'macd'] > df.at[i, 'signal_line']:
                current_position = 1
                current_bar_signal = 1
                if df.at[i, 'macd'] > 0:
                    current_bar_action = "買入 (多頭金叉)"
                else:
                    current_bar_action = "買入 (空頭金叉)"
            
            # MACD 死叉 (賣出訊號)
            elif df.at[i-1, 'macd'] > df.at[i-1, 'signal_line'] and \
                 df.at[i, 'macd'] < df.at[i, 'signal_line']:
                current_position = -1
                current_bar_signal = -1
                if df.at[i, 'macd'] < 0:
                    current_bar_action = "賣出 (空頭死叉)"
                else:
                    current_bar_action = "賣出 (多頭死叉)"

        signals.append(current_bar_signal)
        positions.append(current_position)
        actions.append(current_bar_action)

    return df.iloc[300:].reset_index(drop=True)[[
        "timestamp", "open", "high", "low", "close",
        "macd", "signal_line", "histogram", "signal", "action", "position"
    ]]

    return df.iloc[300:].reset_index(drop=True)[[
        "timestamp", "open", "high", "low", "close",
        "macd", "signal_line", "histogram", "signal", "action"
    ]]

# 測試範例
if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    end_time = datetime.now(timezone.utc)
    df_signals = get_signals_macd(symbol, interval, end_time, limit=1000)
    print(df_signals.tail(10))