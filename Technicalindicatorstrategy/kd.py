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

def detect_kd_signal(df: pd.DataFrame, k_period=14, d_period=3, buy_threshold=30, sell_threshold=70) -> pd.DataFrame:
    df = df.copy()
    low_min = df["low"].rolling(window=k_period).min()
    high_max = df["high"].rolling(window=k_period).max()
    df["%K"] = 100 * (df["close"] - low_min) / (high_max - low_min + 1e-9)
    df["%D"] = df["%K"].rolling(window=d_period).mean()
    df["signal"] = 0
    cross_up = (df["%K"] > df["%D"]) & (df["%K"].shift(1) <= df["%D"].shift(1)) & (df["%K"] < buy_threshold)
    cross_down = (df["%K"] < df["%D"]) & (df["%K"].shift(1) >= df["%D"].shift(1)) & (df["%K"] > sell_threshold)
    df.loc[cross_up, "signal"] = 1
    df.loc[cross_down, "signal"] = -1
    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, k_period: int = 14, d_period: int = 3, buy_threshold: int = 30, sell_threshold: int = 70) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_kd_signal(df, k_period=k_period, d_period=d_period, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    return df

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    # 抓 BTCUSDT 的 1小時線，以現在時間為終點
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(),300)
    print(df_signals['signal'].value_counts())
