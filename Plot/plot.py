import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_backtest_result(result):
    timestamp = pd.to_datetime(result["timestamp"])
    close = result["close"]
    equity = np.array(result["equity"])
    buy_and_hold = np.array(result["buy_and_hold"])
    trade_returns = result["trade_returns"]
    position = np.array(result.get("position", np.zeros_like(close)))

    equity_pct = (equity / equity[0] - 1) * 100
    buy_and_hold_pct = (buy_and_hold / buy_and_hold[0] - 1) * 100

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), gridspec_kw={"height_ratios": [2, 2, 1]})

    # 策略報酬率曲線
    axs[0].plot(timestamp, equity_pct, label="Strategy Return (%)", color="blue", linewidth=2)
    axs[0].plot(timestamp, buy_and_hold_pct, label="Buy and Hold (%)", linestyle="--", alpha=0.7, color="orange")
    axs[0].set_title("Equity Curve (%)")
    axs[0].set_ylabel("Return (%)")
    axs[0].legend()
    axs[0].grid()

    # 收盤價 + 持倉顏色
    position_changes = np.where(np.diff(position) != 0)[0] + 1
    split_points = np.concatenate(([0], position_changes, [len(timestamp)]))

    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]

        if start_idx >= len(timestamp) or end_idx > len(timestamp) or start_idx == end_idx:
            continue

        current_position = position[start_idx]
        color = "green" if current_position > 0 else "red" if current_position < 0 else "gray"
        axs[1].plot(timestamp[start_idx:end_idx + 1], close[start_idx:end_idx + 1], color=color, linewidth=1.5)

    signal = result["signal"]
    buy_idx = np.where(signal == 1)[0]
    sell_idx = np.where(signal == -1)[0]

    axs[1].scatter(timestamp[buy_idx], close[buy_idx], marker="^", color="green", label="Buy Signal", zorder=5)
    axs[1].scatter(timestamp[sell_idx], close[sell_idx], marker="v", color="red", label="Sell Signal", zorder=5)

    axs[1].set_title("Price with Positions and Signals")
    axs[1].legend()
    axs[1].grid()

    # 單筆報酬分布
    axs[2].hist(np.array(trade_returns), bins=20, color="skyblue", edgecolor="k")
    axs[2].set_title("Trade Return Distribution (%)")
    axs[2].set_xlabel("Return (%)")
    axs[2].grid()

    plt.tight_layout()
    plt.show()
