import pandas as pd
import requests
from datetime import datetime
import numpy as np

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

        all_data = data + all_data  # 新資料往前插
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

    if not all_data:
        raise ValueError("No data fetched")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
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

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 1000):
    df = get_binance_kline(symbol, interval, end_time, limit)

    # 計算高低點，並移位避免用到當根
    df['20_high'] = df['high'].rolling(window=20).max().shift(1)
    df['20_low']  = df['low'].rolling(window=20).min().shift(1)
    df['10_high'] = df['high'].rolling(window=10).max().shift(1)
    df['10_low']  = df['low'].rolling(window=10).min().shift(1)

    # 計算 ATR，並移位避免看未來
    df['ATR'] = calculate_atr(df, 20).shift(1)

    position = 0  # 0空倉, 1多單, -1空單
    entry_price = 0
    stop_loss = 0

    signals = []
    stop_losses = []  # 用來存每根的停損值

    for i in range(len(df)):
        if i < 20:
            signals.append(0)
            stop_losses.append(stop_loss)
            continue

        close = df.loc[i, 'close']
        atr = df.loc[i, 'ATR']
        high_20 = df.loc[i, '20_high']
        low_20 = df.loc[i, '20_low']
        high_10 = df.loc[i, '10_high']
        low_10 = df.loc[i, '10_low']

        if position == 0:
            if close > high_20:  # 突破 20 日高
                position = 1
                entry_price = close
                stop_loss = entry_price - 2 * atr
                signals.append(1)  # 多單進場
            elif close < low_20:  # 跌破 20 日低
                position = -1
                entry_price = close
                stop_loss = entry_price + 2 * atr
                signals.append(-1)  # 空單進場
            else:
                signals.append(0)

        elif position == 1:
            # 多單出場條件：跌破 10 日低或停損
            if close < low_10 or close < stop_loss:
                position = 0
                signals.append(-1)  # 多單平倉
            else:
                signals.append(0)

        elif position == -1:
            # 空單出場條件：突破 10 日高或停損
            if close > high_10 or close > stop_loss:
                position = 0
                signals.append(1)  # 空單平倉
            else:
                signals.append(0)

        stop_losses.append(stop_loss)

    df['stop_loss'] = stop_losses
    df['signal'] = signals
    return df

if __name__ == "__main__":
    symbol = "BTCUSDT"
    interval = "1h"
    end_time = datetime.utcnow()
    df_signals = get_signals(symbol, interval, end_time, limit=500)
    print(df_signals[['timestamp', 'close', 'signal']].tail(30))
