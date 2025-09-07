import requests
import pandas as pd
import numpy as np
from datetime import datetime

# ====== 神經網路元件 ======
def relu(x):
    return np.maximum(0, x)

class TwoLayerNN_MACD:
    def __init__(self, W1=None, b1=None, W2=None, b2=None, activation=relu, threshold=0.0):
        # 如果沒給，使用預設 MACD 交叉檢測權重
        self.W1 = W1 if W1 is not None else np.array([
            [1,  0],   # delta_t > 0 ?
            [-1, 0],   # delta_t < 0 ?
            [0,  1],   # delta_t_prev > 0 ?
            [0, -1]    # delta_t_prev < 0 ?
        ])
        self.b1 = b1 if b1 is not None else np.zeros(4)
        self.W2 = W2 if W2 is not None else np.array([[1, -1, -1, 1]])  # shape (1,4)
        self.b2 = b2 if b2 is not None else np.array([0.0])
        self.activation = activation
        self.threshold = threshold

    def forward(self, delta_t, delta_t_prev):
        x = np.array([delta_t, delta_t_prev])
        h = self.activation(np.dot(self.W1, x) + self.b1)  # shape (4,)
        out = np.dot(self.W2, h) + self.b2                 # scalar
        if out > self.threshold:
            return 1   # 黃金交叉
        elif out < -self.threshold:
            return -1  # 死亡交叉
        else:
            return 0   # 無事件

# ====== 技術指標計算 ======
def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    import time

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

        time.sleep(0.5)  # sleep to avoid rate limits

    if not all_data:
        raise ValueError("No data fetched")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True).dt.tz_convert("Asia/Taipei")
    df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
    df = df.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    return df[["timestamp", "open", "high", "low", "close"]]

def detect_macd_signal(df: pd.DataFrame, fast, slow, signal_period, nn_params) -> pd.DataFrame:
    df = df.copy()
    ema_fast = df["close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=signal_period, adjust=False).mean()
    df["delta"] = df["macd"] - df["macd_signal"]

    df["signal"] = 0
    df["position"] = 0
    current_position = 0
    signals = []
    positions = []

    nn = TwoLayerNN_MACD(**nn_params)

    for i in range(len(df)):
        if i < slow + signal_period:
            signals.append(0)
            positions.append(0)
            continue

        delta_t = df.loc[i, "delta"]
        delta_t_prev = df.loc[i - 1, "delta"]

        nn_signal = nn.forward(delta_t, delta_t_prev)

        current_bar_signal = 0

        # === 出場邏輯 ===
        if current_position == 1 and nn_signal == -1:
            current_position = 0
            current_bar_signal = -1
        elif current_position == -1 and nn_signal == 1:
            current_position = 0
            current_bar_signal = 1

        # === 進場邏輯 ===
        if current_position == 0 and nn_signal != 0:
            current_position = nn_signal
            current_bar_signal = nn_signal

        signals.append(current_bar_signal)
        positions.append(current_position)

    df["signal"] = signals
    df["position"] = positions
    return df

# ====== 對外接口 ======
def get_signals(symbol: str,
                interval: str,
                end_time: datetime,
                limit: int = 300,
                fast: int = 12,
                slow: int = 26,
                signal_period: int = 9,
                W1=None, b1=None, W2=None, b2=None,
                activation=relu, threshold: float = 0.0) -> pd.DataFrame:
    """
    對外接口，所有可調整參數都集中於此
    """
    nn_params = dict(W1=W1, b1=b1, W2=W2, b2=b2, activation=activation, threshold=threshold)
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_macd_signal(df, fast=fast, slow=slow, signal_period=signal_period, nn_params=nn_params)
    return df

# ====== 測試 ======
if __name__ == '__main__':
    df_signals = get_signals("BTCUSDT", "15m", datetime.now(), 300)
    print(df_signals[["timestamp", "close", "signal", "position"]].tail(20))
