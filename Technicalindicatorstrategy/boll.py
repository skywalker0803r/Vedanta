import pandas as pd

from .base_strategy import BaseStrategy


class BollingerBandStrategy(BaseStrategy):
    def __init__(self, period: int = 20, num_std: float = 2):
        self.period = period
        self.num_std = num_std

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        result = df.copy()
        result["middle_band"] = result["Close"].rolling(self.period).mean()
        standard_deviation = result["Close"].rolling(self.period).std()
        result["upper_band"] = (
            result["middle_band"] + self.num_std * standard_deviation
        )
        result["lower_band"] = (
            result["middle_band"] - self.num_std * standard_deviation
        )
        result["signal"] = 0
        result.loc[result["Close"] < result["lower_band"], "signal"] = 1
        result.loc[result["Close"] > result["upper_band"], "signal"] = -1
        return result