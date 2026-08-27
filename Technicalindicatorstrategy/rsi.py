import pandas as pd

from .base_strategy import BaseStrategy


class RSIStrategy(BaseStrategy):
    def __init__(self, period: int = 14, low_threshold: float = 40,
                 high_threshold: float = 60):
        self.period = period
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        result = df.copy()
        delta = result["Close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        average_gain = gain.ewm(com=self.period - 1, adjust=False).mean()
        average_loss = loss.ewm(com=self.period - 1, adjust=False).mean()
        relative_strength = average_gain / (average_loss + 1e-9)
        result["rsi"] = 100 - (100 / (1 + relative_strength))
        result["signal"] = 0
        long_signal = (
            (result["rsi"] > self.low_threshold)
            & (result["rsi"].shift(1) <= self.low_threshold)
        )
        short_signal = (
            (result["rsi"] < self.high_threshold)
            & (result["rsi"].shift(1) >= self.high_threshold)
        )
        result.loc[long_signal, "signal"] = 1
        result.loc[short_signal, "signal"] = -1
        return result