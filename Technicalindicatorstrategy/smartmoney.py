import requests
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

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

def _wma(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Weighted Moving Average (WMA).
    """
    weights = np.arange(1, period + 1)
    # Use a lambda function with apply to calculate WMA for each rolling window
    return series.rolling(period).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def calculate_hull_moving_average(series: pd.Series, period: int) -> pd.Series:
    """
    Calculates the Hull Moving Average (HMA).
    Requires the 'close' price series and the period.
    """
    # WMA(n/2)
    wma1 = _wma(series, period // 2)
    # WMA(n)
    wma2 = _wma(series, period)
    # 2 * WMA(n/2) - WMA(n)
    diff = 2 * wma1 - wma2
    # HMA = WMA(diff, sqrt(n))
    hma_period = int(np.sqrt(period))
    hma = _wma(diff, hma_period)
    return hma

def detect_smart_money_signals(df: pd.DataFrame,
                                vegas1_period: int = 144,
                                vegas2_period: int = 169,
                                ema_periods: list = [288, 338, 576, 676],
                                main_hull_period: int = 50,
                                second_hull_period: int = 100,
                                hma_trend_threshold: float = 0.0
                               ) -> pd.DataFrame:
    df = df.copy()

    # Calculate Vegas EMAs
    df[f"ema_{vegas1_period}"] = df["close"].ewm(span=vegas1_period, adjust=False).mean()
    df[f"ema_{vegas2_period}"] = df["close"].ewm(span=vegas2_period, adjust=False).mean()

    # Calculate other specified EMAs
    for period in ema_periods:
        df[f"ema_{period}"] = df["close"].ewm(span=period, adjust=False).mean()

    # Calculate Hull MAs (PLACEHOLDER - REQUIRES ACTUAL HMA LOGIC)
    df["main_hull"] = calculate_hull_moving_average(df["close"], main_hull_period)
    df["second_hull"] = calculate_hull_moving_average(df["close"], second_hull_period)

    # Initialize signal and position columns
    df["signal"] = 0
    df["position"] = 0

    current_position = 0 # Track current position (1: long, -1: short, 0: flat)
    signals = []
    positions = []

    for i in range(len(df)):
        # Ensure enough data for indicators
        # This strategy uses multiple periods, so we need to ensure all are calculated
        max_period = max(vegas1_period, vegas2_period, max(ema_periods), main_hull_period, second_hull_period)
        if i < max_period + 1: # +1 for shift/diff operations
            signals.append(0)
            positions.append(0)
            continue

        curr_close = df.loc[i, "close"]
        curr_vegas1_ema = df.loc[i, f"ema_{vegas1_period}"]
        curr_vegas2_ema = df.loc[i, f"ema_{vegas2_period}"]
        curr_main_hull = df.loc[i, "main_hull"]

        prev_close = df.loc[i-1, "close"]
        prev_vegas1_ema = df.loc[i-1, f"ema_{vegas1_period}"]
        prev_vegas2_ema = df.loc[i-1, f"ema_{vegas2_period}"]
        prev_main_hull = df.loc[i-1, "main_hull"]

        current_bar_signal = 0

        # --- Exit Conditions ---
        if current_position == 1: # Currently long
            # Exit if price crosses below Vegas1 EMA OR Main Hull is no longer trending up
            if (curr_close < curr_vegas1_ema and prev_close >= prev_vegas1_ema) or (curr_main_hull - prev_main_hull <= hma_trend_threshold):
                current_position = 0
                current_bar_signal = -1 # Exit long signal
        elif current_position == -1: # Currently short
            # Exit if price crosses above Vegas2 EMA OR Main Hull is no longer trending down
            if (curr_close > curr_vegas2_ema and prev_close <= prev_vegas2_ema) or (curr_main_hull - prev_main_hull >= -hma_trend_threshold):
                current_position = 0
                current_bar_signal = 1 # Exit short signal

        # --- Entry Conditions ---
        if current_position == 0: # Only enter if currently flat
            # Buy signal: Price crosses above Vegas2 EMA AND Main Hull is trending up
            if (curr_close > curr_vegas2_ema and prev_close <= prev_vegas2_ema) and (curr_main_hull - prev_main_hull > hma_trend_threshold):
                current_position = 1
                current_bar_signal = 1 # Entry long signal
            # Sell signal: Price crosses below Vegas1 EMA AND Main Hull is trending down
            elif (curr_close < curr_vegas1_ema and prev_close >= prev_vegas1_ema) and (curr_main_hull - prev_main_hull < -hma_trend_threshold):
                current_position = -1
                current_bar_signal = -1 # Entry short signal
        
        signals.append(current_bar_signal)
        positions.append(current_position)

    df["signal"] = signals
    df["position"] = positions
    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 1000,
                            vegas1_period: int = 144,
                            vegas2_period: int = 169,
                            ema_periods: list = [288, 338, 576, 676],
                            main_hull_period: int = 50,
                            second_hull_period: int = 100,
                            hma_trend_threshold: float = 0.0
                           ) -> pd.DataFrame:
    df = get_binance_kline(symbol, interval, end_time, limit)
    df = detect_smart_money_signals(df,
                                     vegas1_period=vegas1_period,
                                     vegas2_period=vegas2_period,
                                     ema_periods=ema_periods,
                                     main_hull_period=main_hull_period,
                                     second_hull_period=second_hull_period,
                                     hma_trend_threshold=hma_trend_threshold)
    return df

if __name__ == '__main__':
    from datetime import datetime
    # Example Usage
    df_smart_money_signals = get_signals("BTCUSDT", "15m", datetime.now(), 500)
    print(df_smart_money_signals.tail())
    # You might want to inspect the 'signal' column's value counts:
    # print(df_smart_money_signals['signal'].value_counts())