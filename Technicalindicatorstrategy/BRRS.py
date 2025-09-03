import requests
import pandas as pd
from datetime import datetime, timedelta
import time

# ----------------------------
# Binance Kline 抓取
# ----------------------------
def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
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

        time.sleep(0.5)

    if not all_data:
        raise ValueError("No data fetched")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]

# ----------------------------
# RSI 計算
# ----------------------------
def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    delta = df["close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period-1, adjust=False).mean()
    avg_loss = loss.ewm(com=period-1, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ----------------------------
# 布林通道計算
# ----------------------------
def compute_bollinger(df: pd.DataFrame, period: int = 20, dev: float = 2) -> pd.DataFrame:
    df['ma'] = df['close'].rolling(period).mean()
    df['std'] = df['close'].rolling(period).std()
    df['upper'] = df['ma'] + dev * df['std']
    df['lower'] = df['ma'] - dev * df['std']
    return df

# ----------------------------
# RSI + 布林通道訊號判斷
# ----------------------------
def detect_rsi_bollinger_signal(df: pd.DataFrame, rsi_period: int = 14,
                                low_thresh: float = 40, high_thresh: float = 60,
                                bb_period: int = 20, bb_dev: float = 2) -> pd.DataFrame:
    df = df.copy()
    df['rsi'] = compute_rsi(df, rsi_period)
    df = compute_bollinger(df, bb_period, bb_dev)

    df['signal'] = 0
    df['position'] = 0
    current_position = 0

    for i in range(len(df)):
        if i < max(rsi_period, bb_period):
            continue

        curr_rsi = df.loc[i, 'rsi']
        prev_rsi = df.loc[i-1, 'rsi']
        close = df.loc[i, 'close']
        upper = df.loc[i, 'upper']
        lower = df.loc[i, 'lower']

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1:  # Long exit
            if curr_rsi < high_thresh and prev_rsi >= high_thresh:
                current_position = 0
                current_bar_signal = -1
        elif current_position == -1:  # Short exit
            if curr_rsi > low_thresh and prev_rsi <= low_thresh:
                current_position = 0
                current_bar_signal = 1

        # --- Entry Conditions ---
        if current_position == 0:
            # Long entry: RSI突破低位 + 價格碰布林下軌
            if curr_rsi > low_thresh and prev_rsi <= low_thresh and close <= lower:
                current_position = 1
                current_bar_signal = 1
            # Short entry: RSI跌破高位 + 價格碰布林上軌
            elif curr_rsi < high_thresh and prev_rsi >= high_thresh and close >= upper:
                current_position = -1
                current_bar_signal = -1

        df.loc[i, 'signal'] = current_bar_signal
        df.loc[i, 'position'] = current_position

    return df

# ----------------------------
# 一步抓取與生成訊號
# ----------------------------
def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300,
                rsi_period: int = 14, low_thresh: float = 40, high_thresh: float = 60,
                bb_period: int = 20, bb_dev: float = 2) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_rsi_bollinger_signal(df, rsi_period, low_thresh, high_thresh, bb_period, bb_dev)
    return df

# ----------------------------
# 使用範例
# ----------------------------
if __name__ == '__main__':
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(), limit=300)
    print(df_signals.tail(20))
    print(df_signals['signal'].value_counts())
