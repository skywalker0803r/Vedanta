from abc import ABC, abstractmethod

import pandas as pd


class BaseStrategy(ABC):
    REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}

    def _validate_input(self, df: pd.DataFrame) -> None:
        if not isinstance(df, pd.DataFrame):
            raise TypeError("df must be a pandas DataFrame")

        missing_columns = self.REQUIRED_COLUMNS - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"DataFrame is missing required columns: {sorted(missing_columns)}"
            )

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate a DataFrame containing 1, -1, or 0 in ``signal``."""
        raise NotImplementedError