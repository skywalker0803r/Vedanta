import pandas as pd
import requests
from datetime import datetime, timedelta
import numpy as np

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 3000) -> pd.DataFrame:
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

        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            break

        all_data = data + all_data  # prepend to maintain chronological order
        end_timestamp = data[0][0] - 1  # set next end_time to the timestamp before the earliest one
        remaining -= len(data)

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close"]].reset_index(drop=True)


def detect_vegas_signals(df: pd.DataFrame,
                         ema_1: int = 144,
                         ema_2: int = 169,
                         fib_targets: list = [1.618, 2.0, 2.618]
                        ) -> pd.DataFrame:
    df = df.copy()
    df[f"ema_{ema_1}"] = df["close"].ewm(span=ema_1, adjust=False).mean()
    df[f"ema_{ema_2}"] = df["close"].ewm(span=ema_2, adjust=False).mean()

    tunnel_low = df[f"ema_{ema_1}"]
    tunnel_high = df[f"ema_{ema_2}"]
    tunnel_width = tunnel_high - tunnel_low

    df["signal"] = 0
    df["entry_price"] = np.nan
    df["tp1"] = np.nan
    df["tp2"] = np.nan
    df["tp3"] = np.nan

    # Buy signal
    buy_condition = (
        (df["close"].shift(1) > tunnel_low.shift(1)) &
        (df["close"].shift(1) < tunnel_high.shift(1)) &
        (df["close"] > tunnel_high)
    )

    # Sell signal
    sell_condition = (
        (df["close"].shift(1) > tunnel_low.shift(1)) &
        (df["close"].shift(1) < tunnel_high.shift(1)) &
        (df["close"] < tunnel_low)
    )

    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1

    # 計算目標價位（斐波那契延伸）
    for idx in df.index:
        if df.at[idx, "signal"] == 1:
            entry = df.at[idx, "close"]
            width = tunnel_width.at[idx]
            df.at[idx, "entry_price"] = entry
            df.at[idx, "tp1"] = entry + width * fib_targets[0]
            df.at[idx, "tp2"] = entry + width * fib_targets[1]
            df.at[idx, "tp3"] = entry + width * fib_targets[2]
        elif df.at[idx, "signal"] == -1:
            entry = df.at[idx, "close"]
            width = tunnel_width.at[idx]
            df.at[idx, "entry_price"] = entry
            df.at[idx, "tp1"] = entry - width * fib_targets[0]
            df.at[idx, "tp2"] = entry - width * fib_targets[1]
            df.at[idx, "tp3"] = entry - width * fib_targets[2]

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 3000) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_vegas_signals(df)
    return df

#我剛剛測試上面這個vegas策略 
#使用df_signals = strategy.get_signals("ETHUSDT", "1h", datetime.now(),1000)回測效果很好年化報酬率	235788.87
#但是用df_signals = strategy.get_signals("ETHUSDT", "1h", datetime.now(),10000)回測效果很糟年化報酬率	-25.49
#可能就是因為Vegas交易法是典型的「順勢交易策略」不一定永遠work 能不能基於上面這版本做一個 全天候版本的vegas策略
# 我的思路是既然 這個策略在特定行情有用 那是否做個過濾判斷當前行情如果不適合就空手 這樣就好了 蠻簡單的
