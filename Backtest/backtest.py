import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def backtest_signals(df: pd.DataFrame,
                     initial_capital=100,
                     fee_rate=0.001,
                     leverage=1,
                     allow_short=True,
                     ):
    """
    回測交易訊號策略，支援槓桿與放空。

    參數：
        df (pd.DataFrame): 必須包含 'close', 'signal', 'timestamp' 欄位。
        initial_capital (float): 初始資金。
        fee_rate (float): 交易手續費費率。
        leverage (int): 使用的槓桿倍數。
        allow_short (bool): 是否允許放空。

    回傳：
        dict: 包含績效統計數據與策略時間序列。
    """
    
    df = df.copy().reset_index(drop=True)

    # 建立部位欄位
    temp_position = pd.Series(np.nan, index=df.index)
    temp_position[df["signal"] == 1] = leverage
    temp_position[df["signal"] == -1] = -leverage if allow_short else 0
    df["position"] = temp_position.ffill().fillna(0)
    df.loc[0, "position"] = 0  # 確保第一筆資料為空倉

    # 計算報酬與手續費
    df["return"] = df["close"].pct_change()
    df["strategy_return"] = df["return"] * df["position"].shift(1).fillna(0)
    df["trade"] = df["position"] != df["position"].shift(1)
    df["strategy_return_with_fee"] = df["strategy_return"]
    df.loc[df["trade"], "strategy_return_with_fee"] -= fee_rate * abs(df["position"])

    # 填補nan，第一筆策略報酬改為0
    df["strategy_return_with_fee"].fillna(0, inplace=True)

    # 計算資產曲線
    df["equity"] = initial_capital * (1 + df["strategy_return_with_fee"]).cumprod()
    df["buy_and_hold"] = initial_capital * (1 + df["return"].fillna(0)).cumprod()

    # 計算最大回撤
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1
    max_drawdown = df["drawdown"].min()

    # 計算總報酬與年化報酬率
    total_return = df["equity"].iloc[-1] / initial_capital - 1
    days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).days
    annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0

    # 計算每筆交易的報酬與持倉天數
    trade_returns, hold_days = [], []
    entry_price = entry_time = None
    entry_position = 0

    for i, row in df.iterrows():
        if row["trade"]:
            if entry_price is not None and entry_position != 0:
                exit_price = row["close"] * (1 - fee_rate)
                rtn = ((exit_price / entry_price) - 1 if entry_position > 0
                       else (entry_price / exit_price) - 1) * leverage
                trade_returns.append(rtn)
                hold_days.append((row["timestamp"] - entry_time).days)
            # 建立新部位
            entry_price = row["close"] * (1 + fee_rate)
            entry_time = row["timestamp"]
            entry_position = row["position"]

    # 計算交易統計
    num_trades = len(trade_returns)
    win_rate = np.mean([r > 0 for r in trade_returns]) if num_trades else 0
    avg_profit = np.mean(trade_returns) if num_trades else 0
    avg_days = np.mean(hold_days) if hold_days else 0

    return {
        "metric":{
        "總報酬率": round(total_return * 100, 2),
        "年化報酬率": round(annual_return * 100, 2),
        "最大回撤": round(max_drawdown * 100, 2),
        "交易次數": num_trades,
        "勝率": round(win_rate * 100, 2),
        "平均持有天數": round(avg_days, 2),
        "平均每筆報酬率": round(avg_profit * 100, 2),
    },
    "fig":{
        "timestamp": df["timestamp"].values,
        "equity": df["equity"].values,
        "buy_and_hold": df["buy_and_hold"].values.tolist(),
        "trade_returns": np.array(trade_returns) * 100,
        "close": df["close"].values,
        "signal": df["signal"].values,
        "position": df["position"].values,
    }
    }