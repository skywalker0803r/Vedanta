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

    # --- Signal Logic (Based on Vegas EMAs and HMA) ---
    # Buy signal: Price crosses above Vegas tunnel AND Main Hull is trending up
    buy_condition = (
        (df["close"] > df[f"ema_{vegas2_period}"]) & 
        (df["close"].shift(1) <= df[f"ema_{vegas2_period}"].shift(1)) & 
        (df["main_hull"].diff() > hma_trend_threshold)
    )

    # Sell signal: Price crosses below Vegas tunnel AND Main Hull is trending down
    sell_condition = (
        (df["close"] < df[f"ema_{vegas1_period}"]) & 
        (df["close"].shift(1) >= df[f"ema_{vegas1_period}"].shift(1)) & 
        (df["main_hull"].diff() < -hma_trend_threshold)
    )

    df.loc[buy_condition, "signal"] = 1
    df.loc[sell_condition, "signal"] = -1

    # --- End of Signal Logic ---

    return df

def get_signals(symbol: str, interval: str, end_time: datetime, limit: int = 300,
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