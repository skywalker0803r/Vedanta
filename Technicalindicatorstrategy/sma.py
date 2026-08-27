import pandas as pd

from .base_strategy import BaseStrategy


class SMACrossStrategy(BaseStrategy):
    def __init__(self, short_window: int = 5, long_window: int = 10):
        if short_window >= long_window:
            raise ValueError("short_window must be smaller than long_window")
        self.short_window = short_window
        self.long_window = long_window

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        result = df.copy()
        result["sma_short"] = result["Close"].rolling(self.short_window).mean()
        result["sma_long"] = result["Close"].rolling(self.long_window).mean()
        result["signal"] = 0
        bullish_cross = (
            (result["sma_short"] > result["sma_long"])
            & (result["sma_short"].shift(1) <= result["sma_long"].shift(1))
        )
        bearish_cross = (
            (result["sma_short"] < result["sma_long"])
            & (result["sma_short"].shift(1) >= result["sma_long"].shift(1))
        )
        result.loc[bullish_cross, "signal"] = 1
        result.loc[bearish_cross, "signal"] = -1
        return result