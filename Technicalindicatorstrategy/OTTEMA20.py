import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# ==================== 幣安數據獲取函式 ====================
def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    """
    從幣安 API 獲取指定 K 線數據。

    Args:
        symbol (str): 交易對，例如 'BTCUSDT'。
        interval (str): K 線週期，例如 '1h', '4h', '1d'。
        end_time (datetime): 獲取數據的結束時間（UTC）。
        total_limit (int): 總共要獲取的 K 線數量，最大為 10000。

    Returns:
        pd.DataFrame: 包含 'timestamp', 'open', 'high', 'low', 'close' 的 DataFrame。
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
            data = response.json()

            if not data:
                break

            all_data = data + all_data
            end_timestamp = data[0][0] - 1
            remaining -= len(data)
            time.sleep(0.1)

        except requests.RequestException as e:
            print(f"Error fetching data: {e}")
            break

    if not all_data:
        raise ValueError("No data fetched")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.set_index("timestamp").drop_duplicates().sort_index().reset_index()
    return df[["timestamp", "open", "high", "low", "close"]]


# ==================== 核心交易邏輯函式 ====================
def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 1000, **settings) -> pd.DataFrame:
    """
    從幣安獲取數據，並計算所有指標和交易訊號，返回完整的 DataFrame 以供回測。

    Args:
        symbol (str): 交易對，例如 'BTCUSDT'。
        interval (str): K 線週期，例如 '1h', '4h', '1d'。
        end_time (datetime): 獲取數據的結束時間（UTC）。
        limit (int): 總共要獲取的 K 線數量。
        **settings: 策略參數，例如 length=3, rr=1.5 等。

    Returns:
        pd.DataFrame: 包含 'open', 'high', 'low', 'close', 'timestamp', 'signal', 'position', 'reason' 的 DataFrame。
    """
    # 1. 調用 get_binance_kline 獲取數據
    print(f"正在從幣安獲取 {limit} 根 {interval} K線數據...")
    try:
        df = get_binance_kline(symbol, interval, end_time, total_limit=limit)
        print(f"成功獲取 {len(df)} 根 K線數據。")
    except ValueError as e:
        print(f"數據獲取失敗: {e}")
        return pd.DataFrame()

    # 2. 將 timestamp 欄位設為索引，方便計算
    df = df.set_index("timestamp")
    
    # 3. 策略邏輯開始
    strategy_settings = {
        'length': settings.get('length', 3),
        'percent': settings.get('percent', 1.4),
        'rr': settings.get('rr', 1.5),
        'swing_len': settings.get('swing_len', 20),
        'buy_ema_len': settings.get('buy_ema_len', 20),
        'sell_ema_len': settings.get('sell_ema_len', 37),
        'gmma_threshold': settings.get('gmma_threshold', 0.005),
        'ema_out_len': settings.get('ema_out_len', 30),
        'need_close_cross': settings.get('need_close_cross', True)
    }

    # 計算所有必要的技術指標
    df['emabuy'] = df['close'].ewm(span=strategy_settings['buy_ema_len'], adjust=False).mean()
    df['emasell'] = df['close'].ewm(span=strategy_settings['sell_ema_len'], adjust=False).mean()
    df['ema_out'] = df['close'].ewm(span=strategy_settings['ema_out_len'], adjust=False).mean()
    df['prev_low'] = df['low'].rolling(strategy_settings['swing_len']).min().shift(1)
    df['prev_high'] = df['high'].rolling(strategy_settings['swing_len']).max().shift(1)
    ema_lengths = [3, 5, 8, 10, 12, 15, 30, 35, 40, 45, 50, 60]
    ema_df = pd.DataFrame({f'ema{l}': df['close'].ewm(span=l, adjust=False).mean() for l in ema_lengths})
    gmma_high = ema_df.max(axis=1)
    gmma_low = ema_df.min(axis=1)
    gmma_spread = (gmma_high - gmma_low) / df['close']
    df['is_consolidating'] = gmma_spread < strategy_settings['gmma_threshold']
    df['is_weekend'] = (df.index.dayofweek == 5) | (df.index.dayofweek == 6)

    # 計算 OTT 指標
    src = df['close']
    length = strategy_settings['length']
    percent = strategy_settings['percent']
    valpha = 2 / (length + 1)
    vud1 = np.where(src > src.shift(1), src - src.shift(1), 0)
    vdd1 = np.where(src < src.shift(1), src.shift(1) - src, 0)
    vud = pd.Series(vud1, index=df.index).rolling(9).sum()
    vdd = pd.Series(vdd1, index=df.index).rolling(9).sum()
    v_cmo = ((vud - vdd) / (vud + vdd)).fillna(0)
    var = pd.Series(0.0, index=df.index)
    for i in range(1, len(df)):
        var.iloc[i] = (valpha * abs(v_cmo.iloc[i]) * src.iloc[i]) + (1 - valpha * abs(v_cmo.iloc[i])) * var.iloc[i-1]
    fark = var * percent * 0.01
    long_stop = var - fark
    short_stop = var + fark
    ott = pd.Series(0.0, index=df.index)
    current_dir = 1
    for i in range(1, len(df)):
        long_stop_prev = long_stop.iloc[i-1]
        short_stop_prev = short_stop.iloc[i-1]
        long_stop.iloc[i] = max(long_stop.iloc[i], long_stop_prev) if var.iloc[i] > long_stop_prev else long_stop.iloc[i]
        short_stop.iloc[i] = min(short_stop.iloc[i], short_stop_prev) if var.iloc[i] < short_stop_prev else short_stop.iloc[i]
        if current_dir == -1 and var.iloc[i] > short_stop_prev:
            current_dir = 1
        elif current_dir == 1 and var.iloc[i] < long_stop_prev:
            current_dir = -1
        mt = long_stop.iloc[i] if current_dir == 1 else short_stop.iloc[i]
        ott.iloc[i] = mt * (200 + percent) / 200 if var.iloc[i] > mt else mt * (200 - percent) / 200
    df['mavg'] = var
    df['ott'] = ott

    # 4. 生成回測所需的核心欄位
    df['signal'] = 0  # 訊號：1=買, -1=賣, 0=持有
    df['position'] = 0 # 倉位：1=多頭, -1=空頭, 0=無倉位
    df['reason'] = np.nan # 記錄進出場原因

    # 產生多單和空單進場訊號
    df.loc[(df['mavg'] > df['ott']) & (df['mavg'].shift(1) <= df['ott'].shift(1)) & (df['close'] > df['emabuy']) & (~df['is_consolidating']) & (~df['is_weekend']), 'signal'] = 1
    df.loc[(df['mavg'] < df['ott']) & (df['mavg'].shift(1) >= df['ott'].shift(1)) & (df['close'] < df['emasell']) & (~df['is_consolidating']) & (~df['is_weekend']), 'signal'] = -1

    # 5. 處理倉位狀態機 (迭代) - 模擬Pine腳本的完整邏輯
    # 初始化狀態變數
    current_position = 0  # 0: 空手, 1: 持有多單, -1: 持有空單
    entry_price = 0.0
    sl_level = 0.0
    tp1_level = 0.0
    trailing_active = False

    # 迭代每一根K棒來處理狀態
    for i in range(1, len(df)):
        # 從DataFrame獲取當前K棒的數據
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        long_signal = row['signal'] == 1
        short_signal = row['signal'] == -1

        # 預設繼承上一個狀態
        df.at[df.index[i], 'position'] = df.at[df.index[i-1], 'position']
        df.at[df.index[i], 'reason'] = ''

        # A. 如果當前是空手狀態，檢查是否要進場
        if current_position == 0:
            if long_signal:
                # --- 進場做多 ---
                current_position = 1
                entry_price = row['close']
                sl_level = row['prev_low']
                tp1_level = entry_price + (entry_price - sl_level) * strategy_settings['rr']
                trailing_active = False
                df.at[df.index[i], 'position'] = 1
                df.at[df.index[i], 'reason'] = 'Enter Long'
            elif short_signal:
                # --- 進場做空 ---
                current_position = -1
                entry_price = row['close']
                sl_level = row['prev_high']
                tp1_level = entry_price - (sl_level - entry_price) * strategy_settings['rr']
                trailing_active = False
                df.at[df.index[i], 'position'] = -1
                df.at[df.index[i], 'reason'] = 'Enter Short'

        # B. 如果當前持有多單，檢查出場條件
        elif current_position == 1:
            # 檢查信號反轉
            if short_signal:
                current_position = -1
                entry_price = row['close']
                sl_level = row['prev_high']
                tp1_level = entry_price - (sl_level - entry_price) * strategy_settings['rr']
                trailing_active = False
                df.at[df.index[i], 'position'] = -1
                df.at[df.index[i], 'reason'] = 'Exit Long, Enter Short'
                continue # 進入下一個循環

            # 檢查止損
            if row['low'] <= sl_level:
                current_position = 0
                df.at[df.index[i], 'position'] = 0
                df.at[df.index[i], 'reason'] = 'Stop Loss'
                continue

            # 檢查是否觸發移動止盈
            if not trailing_active and row['high'] >= tp1_level:
                trailing_active = True
                df.at[df.index[i], 'reason'] = 'TP1 Hit, Trailing Activated'

            # 如果移動止盈已觸發，檢查EMA出場
            if trailing_active:
                # Pine腳本的 ta.crossunder(close, emaOut)
                if row['close'] < row['ema_out'] and prev_row['close'] >= prev_row['ema_out']:
                    current_position = 0
                    df.at[df.index[i], 'position'] = 0
                    df.at[df.index[i], 'reason'] = 'Trailing Stop (EMA)'
                    continue

        # C. 如果當前持有空單，檢查出場條件
        elif current_position == -1:
            # 檢查信號反轉
            if long_signal:
                current_position = 1
                entry_price = row['close']
                sl_level = row['prev_low']
                tp1_level = entry_price + (entry_price - sl_level) * strategy_settings['rr']
                trailing_active = False
                df.at[df.index[i], 'position'] = 1
                df.at[df.index[i], 'reason'] = 'Exit Short, Enter Long'
                continue

            # 檢查止損
            if row['high'] >= sl_level:
                current_position = 0
                df.at[df.index[i], 'position'] = 0
                df.at[df.index[i], 'reason'] = 'Stop Loss'
                continue

            # 檢查是否觸發移動止盈
            if not trailing_active and row['low'] <= tp1_level:
                trailing_active = True
                df.at[df.index[i], 'reason'] = 'TP1 Hit, Trailing Activated'

            # 如果移動止盈已觸發，檢查EMA出場
            if trailing_active:
                # Pine腳本的 ta.crossover(close, emaOut)
                if row['close'] > row['ema_out'] and prev_row['close'] <= prev_row['ema_out']:
                    current_position = 0
                    df.at[df.index[i], 'position'] = 0
                    df.at[df.index[i], 'reason'] = 'Trailing Stop (EMA)'
                    continue

    # 最後，確保所有回測所需的欄位都存在
    required_cols = ['open', 'high', 'low', 'close', 'timestamp', 'signal', 'position', 'reason']
    final_df = df.reset_index()[required_cols]

    return final_df

# ==================== 範例使用方式 ====================
if __name__ == '__main__':
    # 你只需要呼叫這個函式，它會返回一個完整的回測 DataFrame
    signals_df = get_signals(
        symbol='BTCUSDT',
        interval='4h',
        end_time=datetime.now(),
        limit=2000,
        # 策略參數
        length=3,
        percent=1.4,
        rr=1.5,
        swing_len=20,
        buy_ema_len=20,
        sell_ema_len=37,
        gmma_threshold=0.005,
        ema_out_len=30,
        need_close_cross=True
    )
    
    # 輸出 DataFrame，檢查欄位是否齊全
    if not signals_df.empty:
        print("\n生成的完整回測數據如下：")
        print(signals_df.head())
        print("\n欄位列表：")
        print(signals_df.columns)
    else:
        print("\n未生成任何數據。")