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
            break

        all_data = data + all_data  # 按時間順序前置
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)

    df = df.sort_values("timestamp").reset_index(drop=True)  # 確保時間排序

    return df[["timestamp", "open", "high", "low", "close"]]


def check_vegas_conditions(df: pd.DataFrame) -> tuple[bool, str]:
    """
    檢查 Vegas 多頭條件。

    Args:
        df (pd.DataFrame): 包含 K 線數據的 DataFrame。

    Returns:
        tuple[bool, str]: 如果滿足條件，則為 (True, "突破") 或 (True, "回踩反彈")，否則為 (False, "")。
    """
    # 計算 EMA 指標
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()
    df = df.dropna()
    
    if len(df) < 2:
        return False, ""

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    vegas_low = min(curr['ema144'], curr['ema169'])
    vegas_high = max(curr['ema144'], curr['ema169'])

    # 突破條件：前一根在區間下方，當前收盤在區間上方，且 ema12 也高於 vegas_high
    breakout = (
        prev['close'] < vegas_low and
        curr['close'] > vegas_high and
        curr['ema12'] > vegas_high
    )

    # 回踩反彈條件：兩根 K 棒收盤都在區間上方，且最低價觸及或跌破 vegas_high，並且 ema12 在區間上方
    bounce = (
        prev['close'] > vegas_high and
        curr['close'] > vegas_high and
        min(curr['low'], prev['low']) <= vegas_high and
        curr['ema12'] > vegas_high
    )

    if breakout:
        return True, "突破"
    elif bounce:
        return True, "回踩反彈"
    
    return False, ""


def check_vegas_short_conditions(df: pd.DataFrame) -> tuple[bool, str]:
    """
    檢查 Vegas 空頭條件。

    Args:
        df (pd.DataFrame): 包含 K 線數據的 DataFrame。

    Returns:
        tuple[bool, str]: 如果滿足條件，則為 (True, "跌破") 或 (True, "反彈失敗")，否則為 (False, "")。
    """
    # 計算 EMA 指標
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()
    df = df.dropna()

    if len(df) < 2:
        return False, ""

    prev = df.iloc[-2]
    curr = df.iloc[-1]
    
    vegas_low = min(curr['ema144'], curr['ema169'])
    vegas_high = max(curr['ema144'], curr['ema169'])

    # 跌破條件：前一根在區間上方，當前收盤在區間下方，且 ema12 也低於 vegas_low
    breakdown = (
        prev['close'] > vegas_high and
        curr['close'] < vegas_low and
        curr['ema12'] < vegas_low
    )

    # 反彈失敗條件：兩根收盤都在區間下方，期間最高價有觸及 vegas_low，且 ema12 也低於 vegas_low
    fail_bounce = (
        prev['close'] < vegas_low and
        curr['close'] < vegas_low and
        max(curr['high'], prev['high']) >= vegas_low and
        curr['ema12'] < vegas_low
    )

    if breakdown:
        return True, "跌破"
    elif fail_bounce:
        return True, "反彈失敗"
    
    return False, ""

# 產生訊號
def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 3000) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    
    # 先算好所有 EMA
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema144'] = df['close'].ewm(span=144, adjust=False).mean()
    df['ema169'] = df['close'].ewm(span=169, adjust=False).mean()
    
    # 初始化信號欄位
    df["signal"] = 0
    df["long_type"] = ""
    df["short_type"] = ""

    # 從第二根 K 線開始判斷（因為需要前一根資料）
    for i in range(1, len(df)):
        prev = df.iloc[i-1]
        curr = df.iloc[i]

        vegas_low = min(curr['ema144'], curr['ema169'])
        vegas_high = max(curr['ema144'], curr['ema169'])

        # 多頭條件判斷
        breakout = (
            prev['close'] < vegas_low and
            curr['close'] > vegas_high and
            curr['ema12'] > vegas_high
        )

        bounce = (
            prev['close'] > vegas_high and
            curr['close'] > vegas_high and
            min(curr['low'], prev['low']) <= vegas_high and
            curr['ema12'] > vegas_high
        )

        # 空頭條件判斷
        breakdown = (
            prev['close'] > vegas_high and
            curr['close'] < vegas_low and
            curr['ema12'] < vegas_low
        )

        fail_bounce = (
            prev['close'] < vegas_low and
            curr['close'] < vegas_low and
            max(curr['high'], prev['high']) >= vegas_low and
            curr['ema12'] < vegas_low
        )

        if breakout:
            df.at[i, "signal"] = 1
            df.at[i, "long_type"] = "突破"
        elif bounce:
            df.at[i, "signal"] = 1
            df.at[i, "long_type"] = "回踩反彈"
        elif breakdown:
            df.at[i, "signal"] = -1
            df.at[i, "short_type"] = "跌破"
        elif fail_bounce:
            df.at[i, "signal"] = -1
            df.at[i, "short_type"] = "反彈失敗"
    
    return df