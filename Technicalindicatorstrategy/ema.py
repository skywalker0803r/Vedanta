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

def detect_ema_cross(df: pd.DataFrame, n1: int = 5, n2: int = 20) -> pd.DataFrame:
    df = df.copy()
    df["ema_1"] = df["close"].ewm(span=n1, adjust=False).mean()
    df["ema_2"] = df["close"].ewm(span=n2, adjust=False).mean()
    df["signal"] = 0
    df.loc[(df["ema_1"] > df["ema_2"]) & (df["ema_1"].shift(1) <= df["ema_2"].shift(1)), "signal"] = 1
    df.loc[(df["ema_1"] < df["ema_2"]) & (df["ema_1"].shift(1) >= df["ema_2"].shift(1)), "signal"] = -1
    return df


def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, n1: int = 5, n2: int = 20) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_ema_cross(df, n1=n1, n2=n2)
    return df

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    # 抓 BTCUSDT 的 1小時線，以現在時間為終點
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(),300)
    print(df_signals['signal'].value_counts())
