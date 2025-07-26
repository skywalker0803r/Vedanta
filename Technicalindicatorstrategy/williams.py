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

def detect_willr_signal(df: pd.DataFrame, period: int = 14, buy_threshold: float = -80, sell_threshold: float = -20) -> pd.DataFrame:
    df = df.copy()
    highest_high = df["high"].rolling(period).max()
    lowest_low = df["low"].rolling(period).min()
    df["%R"] = -100 * (highest_high - df["close"]) / (highest_high - lowest_low + 1e-9)
    df["signal"] = 0
    df.loc[(df["%R"] > buy_threshold) & (df["%R"].shift(1) <= buy_threshold), "signal"] = 1
    df.loc[(df["%R"] < sell_threshold) & (df["%R"].shift(1) >= sell_threshold), "signal"] = -1
    return df


def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, period: int = 14, buy_threshold: float = -80, sell_threshold: float = -20) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_willr_signal(df, period=period, buy_threshold=buy_threshold, sell_threshold=sell_threshold)
    return df

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    # 抓 BTCUSDT 的 1小時線，以現在時間為終點
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(),300)
    print(df_signals['signal'].value_counts())
