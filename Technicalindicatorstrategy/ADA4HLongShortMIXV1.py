import requests
import pandas as pd
import numpy as np
from datetime import datetime

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    import time
    import requests
    import pandas as pd

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

        all_data = data + all_data  # prepend older data
        end_timestamp = data[0][0] - 1
        remaining -= len(data)

        time.sleep(0.5)  # sleep after request to avoid rate limits

    if not all_data:
        raise ValueError("No data fetched")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    # 轉成 UTC 時區
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]

# --- 指標計算函數 (與前一個回答相同) ---
def ta_highest(series, length):
    return series.rolling(window=length).max().shift(1)

def ta_lowest(series, length):
    return series.rolling(window=length).min().shift(1)

def ta_sma(series, length):
    return series.rolling(window=length).mean()

def ta_ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

def ta_rsi(series, length):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=length).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=length).mean()
    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def ta_crossover(series1, series2):
    return (series1.shift(1) <= series2.shift(1)) & (series1 > series2)

def ta_crossunder(series1, series2):
    return (series1.shift(1) >= series2.shift(1)) & (series1 < series2)

def detect_ada_signal(df: pd.DataFrame, long_params: dict, short_params: dict) -> pd.DataFrame:
    df = df.copy()

    # 初始化信號與倉位追蹤
    df["signal"] = 0
    df["position"] = 0
    df["cooldown"] = False

    # 長單指標
    df['upperBand'] = ta_highest(df['high'], long_params['donchianLength'])
    df['lowerBand'] = ta_lowest(df['low'], long_params['donchianLength'])
    df['longTermSma'] = ta_sma(df['close'], long_params['longTermSmaLen'])
    df['rsiLong'] = ta_rsi(df['close'], long_params['rsiLenLong'])

    # 空單指標
    df['emaFast'] = ta_ema(df['close'], short_params['emaFastLength'])
    df['smaSlow'] = ta_sma(df['close'], short_params['smaSlowLength'])
    df['rsiShort'] = ta_rsi(df['close'], short_params['rsiLenShort'])

    # 狀態變數
    current_position = 0  # 1: long, -1: short, 0: flat
    entry_price = 0.0
    entry_index = -1
    short_loss_count = 0
    short_cooldown_until = -1
    
    # 初始化新的欄位
    df["entry_price"] = np.nan
    df["exit_price"] = np.nan
    df["exit_reason"] = ""
    df["stop_loss_level"] = np.nan
    df["take_profit_level"] = np.nan
    df["trailing_stop_level"] = np.nan

    # 這裡的迴圈會模擬策略的逐K棒回測
    for i in range(len(df)):
        
        # 檢查冷卻期是否結束
        if i >= short_cooldown_until:
            short_cooldown_until = -1
        in_short_cooldown = (short_cooldown_until != -1)

        long_signal = (df.loc[i, 'close'] > df.loc[i, 'longTermSma'] and 
                       ta_crossover(df['close'], df['upperBand']).iloc[i] and
                       df.loc[i, 'rsiLong'] > long_params['rsiThLong'])
        
        short_signal = (ta_crossunder(df['emaFast'], df['smaSlow']).iloc[i] and
                        df.loc[i, 'rsiShort'] < short_params['rsiShortThresh'])

        current_bar_signal = 0
        exit_triggered = False
        exit_reason = ""
        exit_price_val = np.nan

        # 平倉邏輯
        if current_position == 1:  # 多單
            # 止損：低於唐奇安下軌
            if df.loc[i, 'low'] <= df.loc[i, 'lowerBand']:
                current_position = 0
                current_bar_signal = -1
                exit_triggered = True
                exit_reason = "Stop Loss (Lower Band)"
                exit_price_val = df.loc[i, 'lowerBand'] # 假設在下軌觸發
            # 反向平倉：EMA 上穿 SMA
            elif ta_crossover(df['emaFast'], df['smaSlow']).iloc[i]:
                current_position = 0
                current_bar_signal = -1
                exit_triggered = True
                exit_reason = "Reverse Close (EMA Cross SMA)"
                exit_price_val = df.loc[i, 'close'] # 假設在收盤價平倉
        
        elif current_position == -1:  # 空單
            # 計算空單止盈止損和移動止損
            shortTP = entry_price * (1 - short_params['shortTPPct'] / 100.0)
            shortSL = entry_price * (1 + short_params['shortSLPct'] / 100.0)
            triggerPrice = entry_price * (1 - short_params['trailTriggerPct'] / 100.0)
            
            trailStopPriceShort = shortSL # 預設為固定止損

            if df.loc[i, 'low'] <= triggerPrice: # 觸發移動止損
                trailStopPriceShort = df.loc[i, 'close'] * (1 + short_params['trailOffsetPct'] / 100.0)
            
            # 止盈
            if df.loc[i, 'high'] >= shortTP: # 空單止盈是價格下跌
                current_position = 0
                current_bar_signal = 1
                exit_triggered = True
                exit_reason = "Take Profit"
                exit_price_val = shortTP # 假設在止盈價觸發
            # 止損/移動止損
            elif df.loc[i, 'high'] >= trailStopPriceShort:
                current_position = 0
                current_bar_signal = 1
                exit_triggered = True
                exit_reason = "Stop Loss / Trailing Stop"
                exit_price_val = trailStopPriceShort # 假設在止損價觸發
            # 反向平倉：EMA 上穿 SMA
            elif ta_crossover(df['emaFast'], df['smaSlow']).iloc[i]:
                current_position = 0
                current_bar_signal = 1
                exit_triggered = True
                exit_reason = "Reverse Close (EMA Cross SMA)"
                exit_price_val = df.loc[i, 'close'] # 假設在收盤價平倉

        # 進場邏輯
        if current_position == 0 and not exit_triggered: # 只有在沒有部位且沒有觸發出場時才考慮進場
            if long_signal:
                current_position = 1
                current_bar_signal = 1
                entry_price = df.loc[i, 'close'] # 假設在收盤價進場
                entry_index = i
            elif short_signal and not in_short_cooldown:
                current_position = -1
                current_bar_signal = -1
                entry_price = df.loc[i, 'close'] # 假設在收盤價進場
                entry_index = i
        
        # 記錄信號和部位
        df.loc[i, "signal"] = current_bar_signal
        df.loc[i, "position"] = current_position
        df.loc[i, "cooldown"] = in_short_cooldown
        
        if exit_triggered:
            df.loc[i, "exit_price"] = exit_price_val
            df.loc[i, "exit_reason"] = exit_reason
            entry_price = 0.0 # 重置進場價格
            entry_index = -1 # 重置進場索引
        elif current_position != 0: # 如果有部位，記錄進場價格和當前止損止盈水平
            df.loc[i, "entry_price"] = entry_price
            if current_position == 1: # 多單
                df.loc[i, "stop_loss_level"] = df.loc[i, 'lowerBand']
            elif current_position == -1: # 空單
                shortTP = entry_price * (1 - short_params['shortTPPct'] / 100.0)
                shortSL = entry_price * (1 + short_params['shortSLPct'] / 100.0)
                triggerPrice = entry_price * (1 - short_params['trailTriggerPct'] / 100.0)
                
                trailStopPriceShort = shortSL
                if df.loc[i, 'low'] <= triggerPrice:
                    trailStopPriceShort = df.loc[i, 'close'] * (1 + short_params['trailOffsetPct'] / 100.0)
                
                df.loc[i, "take_profit_level"] = shortTP
                df.loc[i, "stop_loss_level"] = shortSL # 初始止損
                df.loc[i, "trailing_stop_level"] = trailStopPriceShort # 移動止損
    
    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300, 
                    long_params: dict = None, short_params: dict = None) -> pd.DataFrame:
    
    # 使用預設參數
    if long_params is None:
        long_params = {
            'donchianLength': 12, 'longTermSmaLen': 150, 'rsiLenLong': 30, 'rsiThLong': 60.0
        }
    if short_params is None:
        short_params = {
            'emaFastLength': 6, 'smaSlowLength': 65, 'rsiLenShort': 65, 'rsiShortThresh': 50,
            'shortTPPct': 10, 'shortSLPct': 5, 'trailTriggerPct': 8, 'trailOffsetPct': 4
        }
    
    df = get_binance_kline(symbol, interval, end_time, limit)
    
    # 這裡呼叫新的 detect_ada_signal 函數
    df_signals = detect_ada_signal(df, long_params, short_params)
    
    return df_signals

def get_ada_signals_optimized(symbol: str, interval: str, end_time: datetime, limit: int = 300,
                              donchianLength: int = 12, longTermSmaLen: int = 150, rsiLenLong: int = 30, rsiThLong: float = 60.0,
                              emaFastLength: int = 6, smaSlowLength: int = 65, rsiLenShort: int = 65, rsiShortThresh: int = 50,
                              shortTPPct: float = 10, shortSLPct: float = 5, trailTriggerPct: float = 8, trailOffsetPct: float = 4) -> pd.DataFrame:
    
    long_params = {
        'donchianLength': donchianLength,
        'longTermSmaLen': longTermSmaLen,
        'rsiLenLong': rsiLenLong,
        'rsiThLong': rsiThLong
    }
    
    short_params = {
        'emaFastLength': emaFastLength,
        'smaSlowLength': smaSlowLength,
        'rsiLenShort': rsiLenShort,
        'rsiShortThresh': rsiShortThresh,
        'shortTPPct': shortTPPct,
        'shortSLPct': shortSLPct,
        'trailTriggerPct': trailTriggerPct,
        'trailOffsetPct': trailOffsetPct
    }
    
    return get_signals(symbol, interval, end_time, limit, long_params, short_params)

# 使用範例
if __name__ == '__main__':
    from datetime import datetime
    
    # 獲取 ETHUSDT 的 4 小時線
    df_signals = get_signals("ETHUSDT", "4h", datetime.now(), 300)
    
    print("信號統計：")
    print(df_signals['signal'].value_counts())
    
    print("\n最後的 10 根 K 棒信號與倉位：")
    print(df_signals[['timestamp', 'open', 'high', 'low', 'close', 'signal', 'position', 'cooldown']].tail(10))