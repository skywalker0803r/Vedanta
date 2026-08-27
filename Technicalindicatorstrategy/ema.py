import pandas as pd

from .base_strategy import BaseStrategy


class EMACrossStrategy(BaseStrategy):
    def __init__(self, fast_window: int = 5, slow_window: int = 20):
        if fast_window >= slow_window:
            raise ValueError("fast_window must be smaller than slow_window")
        self.fast_window = fast_window
        self.slow_window = slow_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        result = df.copy()
        result["ema_fast"] = result["Close"].ewm(
            span=self.fast_window, adjust=False
        ).mean()
        result["ema_slow"] = result["Close"].ewm(
            span=self.slow_window, adjust=False
        ).mean()
        result["signal"] = 0
        bullish_cross = (
            (result["ema_fast"] > result["ema_slow"])
            & (result["ema_fast"].shift(1) <= result["ema_slow"].shift(1))
        )
        bearish_cross = (
            (result["ema_fast"] < result["ema_slow"])
            & (result["ema_fast"].shift(1) >= result["ema_slow"].shift(1))
        )
        result.loc[bullish_cross, "signal"] = 1
        result.loc[bearish_cross, "signal"] = -1
        return result