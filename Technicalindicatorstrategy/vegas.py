import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 3000) -> pd.DataFrame:
    """
    從幣安 API 獲取 K 線數據。

    Args:
        symbol (str): 交易對符號 (例如 "BTCUSDT")。
        interval (str): K 線間隔 (例如 "1h", "4h", "1d")。
        end_time (datetime): 結束時間。
        total_limit (int): 要獲取的 K 線總數。

    Returns:
        pd.DataFrame: 包含 K 線數據的 DataFrame。
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

        all_data = data + all_data  # 按時間順序前置
        end_timestamp = data[0][0] - 1  # 下一輪抓更舊的資料
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


def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 1000) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    
    # 計算 EMA
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()

    # 初始化訊號欄位
    df["signal"] = 0
    df["long_type"] = ""
    df["short_type"] = ""

    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]

        vegas_low = min(curr['ema144'], curr['ema169'])
        vegas_high = max(curr['ema144'], curr['ema169'])

        # 多頭條件
        is_breakout = (
            prev['close'] < vegas_low and
            curr['close'] > vegas_high and
            curr['ema12'] > vegas_high
        )

        is_bounce = (
            prev['close'] > vegas_high and
            curr['close'] > vegas_high and
            min(curr['low'], prev['low']) <= vegas_high and
            curr['ema12'] > vegas_high
        )

        # 空頭條件
        is_breakdown = (
            prev['close'] > vegas_high and
            curr['close'] < vegas_low and
            curr['ema12'] < vegas_low
        )

        is_failed_bounce = (
            prev['close'] < vegas_low and
            curr['close'] < vegas_low and
            max(curr['high'], prev['high']) >= vegas_low and
            curr['ema12'] < vegas_low
        )

        if is_breakout:
            df.at[i, "signal"] = 1
            df.at[i, "long_type"] = "突破"
        elif is_bounce:
            df.at[i, "signal"] = 1
            df.at[i, "long_type"] = "回踩反彈"
        elif is_breakdown:
            df.at[i, "signal"] = -1
            df.at[i, "short_type"] = "跌破"
        elif is_failed_bounce:
            df.at[i, "signal"] = -1
            df.at[i, "short_type"] = "反彈失敗"

    return df[[
        "timestamp", "open", "high", "low", "close",
        "ema12", "ema144", "ema169", "signal", "long_type", "short_type"
    ]]

# 測試範例
if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    end_time = datetime.utcnow()
    df_signals = get_signals(symbol, interval, end_time, limit=1000)
    print(df_signals.tail(10))
