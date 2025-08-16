import requests
import pandas as pd
from datetime import datetime

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

def atr_wilder(df: pd.DataFrame, period: int) -> pd.Series:
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

def calculate_supertrend(df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    df = df.copy()

    hl2 = (df['high'] + df['low']) / 2
    atr = atr_wilder(df, period)

    upperband = hl2 + multiplier * atr
    lowerband = hl2 - multiplier * atr

    supertrend = pd.Series(index=df.index)
    direction = pd.Series(index=df.index)

    for i in range(len(df)):
        if i < period:
            supertrend.iat[i] = 0
            direction.iat[i] = 1
            continue

        if i == period:
            supertrend.iat[i] = upperband.iat[i]
            direction.iat[i] = 1
            continue

        prev_supertrend = supertrend.iat[i-1]
        prev_direction = direction.iat[i-1]

        # 當前的upper與lower band
        curr_upper = upperband.iat[i]
        curr_lower = lowerband.iat[i]
        close = df['close'].iat[i]

        if close > prev_supertrend:
            # 多頭趨勢
            if prev_direction == -1:
                supertrend.iat[i] = curr_lower
            else:
                supertrend.iat[i] = max(curr_lower, prev_supertrend)
            direction.iat[i] = 1
        else:
            # 空頭趨勢
            if prev_direction == 1:
                supertrend.iat[i] = curr_upper
            else:
                supertrend.iat[i] = min(curr_upper, prev_supertrend)
            direction.iat[i] = -1

    df['supertrend'] = supertrend
    df['direction'] = direction

    # 訊號：direction從-1轉1是買訊，從1轉-1是賣訊
    df['signal'] = 0
    df.loc[(df['direction'] == 1) & (df['direction'].shift(1) == -1), 'signal'] = 1
    df.loc[(df['direction'] == -1) & (df['direction'].shift(1) == 1), 'signal'] = -1

    df['position'] = df['direction'] # SuperTrend direction directly represents the position

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, period: int = 3, multiplier: float = 1) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = calculate_supertrend(df, period=period, multiplier=multiplier)
    return df

if __name__ == "__main__":
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(), 300, 10, 3.0)
    print(df_signals[['timestamp', 'close', 'supertrend', 'direction', 'signal']].tail(10))
    print("Signal counts:\n", df_signals['signal'].value_counts())
