import pandas as pd

from .base_strategy import BaseStrategy


class MACDStrategy(BaseStrategy):
    def __init__(self, fast: int = 12, slow: int = 26, signal_period: int = 9):
        if fast >= slow:
            raise ValueError("fast must be smaller than slow")
        self.fast = fast
        self.slow = slow
        self.signal_period = signal_period

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        result = df.copy()
        ema_fast = result["Close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = result["Close"].ewm(span=self.slow, adjust=False).mean()
        result["macd"] = ema_fast - ema_slow
        result["macd_signal"] = result["macd"].ewm(
            span=self.signal_period, adjust=False
        ).mean()
        result["signal"] = 0
        bullish_cross = (
            (result["macd"] > result["macd_signal"])
            & (result["macd"].shift(1) <= result["macd_signal"].shift(1))
        )
        bearish_cross = (
            (result["macd"] < result["macd_signal"])
            & (result["macd"].shift(1) >= result["macd_signal"].shift(1))
        )
        result.loc[bullish_cross, "signal"] = 1
        result.loc[bearish_cross, "signal"] = -1
        return result