import pandas as pd
import requests
from datetime import datetime
import numpy as np
import time

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    time.sleep(0.1)
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
def generate_signals(df: pd.DataFrame, rank_th,
                    ATR_multi_SL, ATR_multi_TP,
                    rank_th_2, ATR_multi_SL_2, ATR_multi_TP_2,
                    allow_dual_position: bool = True
                    ):
    df = df.copy()

    # === 趨勢 & 盤整獨立倉位 ===
    trend_position = 0
    trend_entry = np.nan
    trend_sl = np.nan
    trend_tp = np.nan

    consolid_position = 0
    consolid_entry = np.nan
    consolid_sl = np.nan
    consolid_tp = np.nan

  
    # === 記錄 ===
    signals, positions, reasons = [], [], []
    trend_positions, consolid_positions = [], []
    trend_entries, trend_stops = [], []
    consolid_entries, consolid_stops = [], []
    
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
            reasons.append("")
            trend_positions.append(0)
            consolid_positions.append(0)
            trend_entries.append(np.nan)
            trend_stops.append(np.nan)
            consolid_entries.append(np.nan)
            consolid_stops.append(np.nan)
            continue

        close, ma, upper, lower = df.loc[i, ["close", "ma", "upper", "lower"]]
        rank, ATR = df.loc[i, ["bb_rank", "ATR"]]

        current_signal, current_reason = 0, ""

        # ============= 趨勢策略處理 =============
        if trend_position == 0:
            if rank > rank_th and close >= ma and close <= upper:
                # 如果不允許雙倉，需平掉盤整倉位
                if not allow_dual_position and consolid_position != 0:
                    consolid_position = 0
                    current_reason = "切換趨勢多，先平盤整倉"
                trend_position = 1
                trend_entry = close
                trend_sl = trend_entry - ATR_multi_SL * ATR
                trend_tp = trend_entry + ATR_multi_TP * ATR
                current_signal, current_reason = 1, "趨勢多單進場"
            elif rank > rank_th and close <= ma and close >= lower:
                if not allow_dual_position and consolid_position != 0:
                    consolid_position = 0
                    current_reason = "切換趨勢空，先平盤整倉"
                trend_position = -1
                trend_entry = close
                trend_sl = trend_entry + ATR_multi_SL * ATR
                trend_tp = trend_entry - ATR_multi_TP * ATR
                current_signal, current_reason = -1, "趨勢空單進場"
        else:
            if trend_position == 1:
                if close < trend_sl:
                    trend_position = 0
                    trend_entry = np.nan
                    trend_sl = np.nan
                    trend_tp = np.nan
                    current_signal, current_reason = -1, "趨勢多單SL"
                elif close > trend_tp:
                    trend_position = 0
                    trend_entry = np.nan
                    trend_sl = np.nan
                    trend_tp = np.nan
                    current_signal, current_reason = -1, "趨勢多單TP"
            elif trend_position == -1:
                if close > trend_sl:
                    trend_position = 0
                    trend_entry = np.nan
                    trend_sl = np.nan
                    trend_tp = np.nan
                    current_signal, current_reason = 1, "趨勢空單SL"
                elif close < trend_tp:
                    trend_position = 0
                    trend_entry = np.nan
                    trend_sl = np.nan
                    trend_tp = np.nan
                    current_signal, current_reason = 1, "趨勢空單TP"

        # ============= 盤整策略處理 =============
        if consolid_position == 0:
            if rank < rank_th_2 and close <= ma and close <= upper:
                if not allow_dual_position and trend_position != 0:
                    trend_position = 0
                    current_reason = "切換盤整多，先平趨勢倉"
                consolid_position = 1
                consolid_entry = close
                consolid_sl = consolid_entry - ATR_multi_SL_2 * ATR
                consolid_tp = consolid_entry + ATR_multi_TP_2 * ATR
                current_signal, current_reason = 1, "盤整多單進場"
            elif rank < rank_th_2 and close >= ma and close >= lower:
                if not allow_dual_position and trend_position != 0:
                    trend_position = 0
                    current_reason = "切換盤整空，先平趨勢倉"
                consolid_position = -1
                consolid_entry = close
                consolid_sl = consolid_entry + ATR_multi_SL_2 * ATR
                consolid_tp = consolid_entry - ATR_multi_TP_2 * ATR
                current_signal, current_reason = -1, "盤整空單進場"
        else:
            if consolid_position == 1:
                if close < consolid_sl:
                    consolid_position = 0
                    consolid_entry = np.nan
                    consolid_sl = np.nan
                    consolid_tp = np.nan
                    current_signal, current_reason = -1, "盤整多單SL"
                elif close > consolid_tp:
                    consolid_position = 0
                    consolid_entry = np.nan
                    consolid_sl = np.nan
                    consolid_tp = np.nan
                    current_signal, current_reason = -1, "盤整多單TP"
            elif consolid_position == -1:
                if close > consolid_sl:
                    consolid_position = 0
                    consolid_entry = np.nan
                    consolid_sl = np.nan
                    consolid_tp = np.nan
                    current_signal, current_reason = 1, "盤整空單SL"
                elif close < consolid_tp:
                    consolid_position = 0
                    consolid_entry = np.nan
                    consolid_sl = np.nan
                    consolid_tp = np.nan
                    current_signal, current_reason = 1, "盤整空單TP"

        # === 記錄當前行的數據 ===
        total_position = trend_position + consolid_position  # 總倉位
        signals.append(current_signal)
        positions.append(total_position)
        reasons.append(current_reason)
        trend_positions.append(trend_position)
        consolid_positions.append(consolid_position)
        
        # 記錄進場價格和停損價格（每行一個值）
        trend_entries.append(trend_entry)
        trend_stops.append(trend_sl)
        consolid_entries.append(consolid_entry)
        consolid_stops.append(consolid_sl)

    # === 將結果添加到DataFrame ===
    df["position"] = positions        # 總倉位 (回測套件讀取)
    df["signal"] = signals            # 總訊號
    df["reason"] = reasons
    df["trend_position"] = trend_positions      # 額外追蹤
    df["consolid_position"] = consolid_positions
    
    # 分別記錄趨勢和盤整的進場價格和停損價格
    df['trend_entry_price'] = trend_entries
    df['trend_stop_loss'] = trend_stops
    df['consolid_entry_price'] = consolid_entries
    df['consolid_stop_loss'] = consolid_stops

    return df



# =======================
# 4. 主流程
# =======================
def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300,
                bb_length: int = 20, mult: float = 2.0, lookback: int = 100,
                rank_th: float = 85, ATR_period: int = 20,
                ATR_multi_SL: float = 5.0, ATR_multi_TP: float = 8.0,rank_th_2: int = 50,
                ATR_multi_SL_2: float = 2.0, ATR_multi_TP_2: float = 1.0,
                allow_dual_position: bool = True
                ):
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = calc_indicators(df, bb_length, mult, lookback, ATR_period)
    df = generate_signals(df, rank_th, ATR_multi_SL, ATR_multi_TP,
                          rank_th_2, ATR_multi_SL_2, ATR_multi_TP_2,
                            allow_dual_position)
    return df

# =======================
# 測試範例
# =======================
if __name__ == "__main__":
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(), 1000)
    print(df_signals[['timestamp', 'close', 'position', 'signal', 'ATR']].tail(30))

# =======================
# 測試使用backtest_usage
# =======================
# from Technicalindicatorstrategy import bbrank,bbrank_1,bbrank_2
# import warnings 
# warnings.filterwarnings('ignore')
# from Backtest.backtest import backtest_signals
# from Plot.plot import display_trades_log_as_html,plot_backtest_result
# from IPython.display import HTML
# import pandas as pd
# from datetime import datetime,timedelta
# import warnings 
# warnings.filterwarnings('ignore')
# import numpy as np
# np.random.seed(42)  # ✅ 固定隨機性（可重現性）
# import random
# random.seed(42)

# df_signals = bbrank.get_signals('ETHUSDT','1h',datetime.now(),5000, 
#                                 lookback = 300, 
#                                 rank_th = 90,
#                                 ATR_multi_SL = 1.75, ATR_multi_TP = 5.0,
#                                 rank_th_2 = 50,
#                                 ATR_multi_SL_2 = 1.0, ATR_multi_TP_2 = 2.0,
#                                 allow_dual_position = True)

# result = backtest_signals(
#     df_signals.copy(),
#     initial_capital = 1000000, # 1000台幣
#     fee_rate = 0.0004, # 合約手續費
#     leverage = 1, # 槓桿
#     allow_short = True, # 是否做空
#     stop_loss = None,       # 停損閾值，例如0.05代表5%
#     take_profit = None,     # 停利閾值
#     capital_ratio = 1, # 每次使用的資金佔比
#     max_hold_bars = 10000,# 最大持有K棒數
#     delay_entry=False,
#     risk_free_rate=0
#     )  
# display(pd.DataFrame(result['Overview performance'],index=['Overview performance']))
# display(pd.DataFrame(result['Trades analysis'],index=['Trades analysis']))
# display(pd.DataFrame(result['Risk/performance ratios'],index=['Risk/performance ratios']))
# html_output = display_trades_log_as_html(result['trades_log'][:10])
# plot_backtest_result(result)
# display(HTML(html_output))
# display(df_signals.loc[df_signals['signal']!=0,['timestamp','signal','close','reason']].head(100).style.background_gradient())
