import pandas as pd
import requests
from datetime import datetime
import numpy as np
import time

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    time.sleep(0.2)
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

        all_data = data + all_data
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


# =======================
# 2. 指標計算
# =======================
def calc_indicators(df: pd.DataFrame, bb_length, mult
                    , lookback
                    , ATR_period):
    df = df.copy()
    
    # Bollinger Bands
    df["ma"] = df["close"].rolling(bb_length).mean()
    df["std"] = df["close"].rolling(bb_length).std()
    df["upper"] = df["ma"] + mult * df["std"]
    df["lower"] = df["ma"] - mult * df["std"]

    # BB Rank（標準差 / 收盤價，再做百分位排名）
    stdev_series = df["std"] / df["close"]
    df["bb_rank"] = stdev_series.rolling(lookback).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100, raw=False
    )

    # ATR
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    tr = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df["atr"] = tr.rolling(window=ATR_period).mean()
    # 計算 ATR，並移位避免看未來
    df["ATR"] = df["atr"].shift(1)
    return df


# =======================
# 3. 訊號判斷
# =======================
def generate_signals(df: pd.DataFrame, rank_th
                     , ATR_multi_SL, ATR_multi_TP):
    df = df.copy()
    
    # df["signal"] = 0
    # df["position"] = 0


    # 初始狀態
    current_position = 0 # 追蹤當前倉位
    entry_price = np.nan
    stop_loss = np.nan
    take_profit = np.nan
    take_profit = np.nan

    # 記錄訊號和倉位
    signals = []
    positions = []
    entry_prices = []
    stop_losses = []
    take_profits = []
    reasons = [] # 新增原因欄位


    for i in range(len(df)):
        if (
            np.isnan(df.loc[i, "ma"]) 
            or np.isnan(df.loc[i, "upper"]) 
            or np.isnan(df.loc[i, "lower"]) 
            or np.isnan(df.loc[i, "bb_rank"])
            or np.isnan(df.loc[i, "ATR"])
        ):
            signals.append(0)
            positions.append(0)
            entry_prices.append(np.nan)
            stop_losses.append(np.nan)
            reasons.append("") # 初始狀態無原因
            continue

        close = df.loc[i, "close"]
        ma = df.loc[i, "ma"]
        upper = df.loc[i, "upper"]
        lower = df.loc[i, "lower"]
        rank = df.loc[i, "bb_rank"]
        ATR = df.loc[i, "ATR"]
        
        current_signal = 0
        current_reason = ""

        # === 進場條件 ===
        if current_position == 0 and current_signal == 0:
            # 多頭進場條件：BBRank>閾值
            if rank > rank_th and close >= ma and close <= upper:
                current_position = 1
                entry_price = close
                stop_loss = entry_price - ATR_multi_SL * ATR
                take_profit = entry_price + ATR_multi_TP * ATR
                current_signal = 1  # 訊號改為 1，代表開多
                current_reason = "多單進場"

            # 空頭進場條件：BBRank
            elif rank > rank_th and close <= ma and close >= lower:
                current_position = -1
                entry_price = close
                stop_loss = entry_price + ATR_multi_SL * ATR
                take_profit = entry_price - ATR_multi_TP * ATR
                current_signal = -1  # 訊號改為 -1，代表開空
                current_reason = "空單進場"

        # === 出場條件===
        elif current_position == 1:
            if close < stop_loss:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                take_profit = np.nan
                current_signal = -1  # 訊號改為 -1，代表平倉
                current_reason = "多單SL"
            elif close > take_profit:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                take_profit = np.nan
                current_signal = -1  # 訊號改為 -1，代表平倉
                current_reason = "多單TP"
        elif current_position == -1:
            if close < stop_loss:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                take_profit = np.nan
                current_signal = 1  # 訊號改為 -1，代表平倉
                current_reason = "空單SL"
            elif close > take_profit:
                current_position = 0
                entry_price = np.nan
                stop_loss = np.nan
                take_profit = np.nan
                current_signal = 1  # 訊號改為 -1，代表平倉
                current_reason = "空單TP"

        signals.append(current_signal)
        positions.append(current_position) # Append the actual current_position
        entry_prices.append(entry_price)
        stop_losses.append(stop_loss)
        take_profits.append(take_profit)
        reasons.append(current_reason)

    df['position'] = positions
    df['entry_price'] = entry_prices
    df['stop_loss'] = stop_losses
    df['signal'] = signals
    df['reason'] = reasons # 將原因欄位加入回傳的 DataFrame

    return df



# =======================
# 4. 主流程
# =======================
def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300
                , bb_length: int = 20, mult: float = 2.0, lookback: int = 100
                , rank_th: float = 85, ATR_period: int = 20
                , ATR_multi_SL: float = 5.0, ATR_multi_TP: float = 8.0):
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = calc_indicators(df, bb_length, mult, lookback, ATR_period)
    df = generate_signals(df, rank_th, ATR_multi_SL, ATR_multi_TP)
    return df


# =======================
# 測試範例
# =======================
if __name__ == "__main__":
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(), 300)
    print(df_signals[['timestamp', 'close', 'position', 'signal', 'ATR']].tail(30))



from Technicalindicatorstrategy import bbrank
import warnings 
warnings.filterwarnings('ignore')
from Backtest.backtest import backtest_signals
from Plot.plot import display_trades_log_as_html,plot_backtest_result
from IPython.display import HTML
import pandas as pd
from datetime import datetime,timedelta
import warnings 
warnings.filterwarnings('ignore')
import numpy as np
np.random.seed(42)  # ✅ 固定隨機性（可重現性）
import random
random.seed(42)

df_signals = bbrank.get_signals('ETHUSDT','1h',datetime.now(),3000, 
                                lookback = 300, rank_th = 90,
                                ATR_multi_SL = 2.0, ATR_multi_TP = 10.0)

result = backtest_signals(
    df_signals.copy(),
    initial_capital = 1000000, # 1000台幣
    fee_rate = 0.000, # 合約手續費
    leverage = 2, # 槓桿
    allow_short = True, # 是否做空
    stop_loss = None,       # 停損閾值，例如0.05代表5%
    take_profit = None,     # 停利閾值
    capital_ratio = 1, # 每次使用的資金佔比
    max_hold_bars = 100000,# 最大持有K棒數
    delay_entry=False,
    risk_free_rate=0
    )  
display(pd.DataFrame(result['Overview performance'],index=['Overview performance']))
display(pd.DataFrame(result['Trades analysis'],index=['Trades analysis']))
display(pd.DataFrame(result['Risk/performance ratios'],index=['Risk/performance ratios']))
html_output = display_trades_log_as_html(result['trades_log'][-10:])
plot_backtest_result(result)
display(HTML(html_output))
display(df_signals.loc[df_signals['signal']!=0,['timestamp','signal','close','reason','stop_loss']].tail(10).style.background_gradient())