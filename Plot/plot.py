import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_backtest_result(result):
    fig_data = result["fig"]
    trades = result.get("trades_log", [])

    timestamp = pd.to_datetime(fig_data["timestamp"])
    close = np.array(fig_data["close"])
    equity = np.array(fig_data["equity"])
    buy_and_hold = np.array(fig_data["buy_and_hold"])
    position = np.array(fig_data["position"])
    signal = np.array(fig_data["signal"])

    trade_returns = [t['return'] for t in trades]

    equity_pct = (equity / equity[0] - 1) * 100
    buy_and_hold_pct = (buy_and_hold / buy_and_hold[0] - 1) * 100

    fig, axs = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [2.5, 2, 1]})

    # === 策略報酬率曲線 ===
    axs[0].plot(timestamp, equity_pct, label="Strategy Return (%)", color="blue", linewidth=2)
    axs[0].plot(timestamp, buy_and_hold_pct, label="Buy and Hold (%)", linestyle="--", alpha=0.7, color="orange")
    
    # 最大回撤區塊標記
    dd = equity / np.maximum.accumulate(equity) - 1
    min_dd_idx = np.argmin(dd)
    peak_idx = np.argmax(equity[:min_dd_idx + 1])
    axs[0].axvspan(timestamp[peak_idx], timestamp[min_dd_idx], color='red', alpha=0.1, label='Max Drawdown')

    axs[0].set_title("Equity Curve (%)")
    axs[0].set_ylabel("Return (%)")
    axs[0].legend()
    axs[0].grid()

    # === 價格走勢與部位區段 ===
    position_changes = np.where(np.diff(position) != 0)[0] + 1
    split_points = np.concatenate(([0], position_changes, [len(timestamp)]))

    for i in range(len(split_points) - 1):
        start_idx = split_points[i]
        end_idx = split_points[i + 1]
        if start_idx >= len(timestamp) or end_idx > len(timestamp):
            continue
        pos = position[start_idx]
        color = 'green' if pos > 0 else 'red' if pos < 0 else 'gray'
        axs[1].plot(timestamp[start_idx:end_idx + 1], close[start_idx:end_idx + 1], color=color, linewidth=1.5)

    # === 訊號點 ===
    buy_idx = np.where(signal == 1)[0]
    sell_idx = np.where(signal == -1)[0]
    axs[1].scatter(timestamp[buy_idx], close[buy_idx], marker="^", color="green", label="Buy Signal", zorder=5)
    axs[1].scatter(timestamp[sell_idx], close[sell_idx], marker="v", color="red", label="Sell Signal", zorder=5)

    # === 真實進出場點與箭頭 ===
    for trade in trades:
        entry_time = pd.to_datetime(trade["entry_time"])
        exit_time = pd.to_datetime(trade["exit_time"])
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]

        color = "green" if trade["side"] == "long" else "red"
        axs[1].scatter(entry_time, entry_price, color=color, marker='o', edgecolor='black', zorder=6, label='Entry' if trade == trades[0] else "")
        axs[1].scatter(exit_time, exit_price, color=color, marker='x', edgecolor='black', zorder=6, label='Exit' if trade == trades[0] else "")

        # 箭頭
        axs[1].annotate("", xy=(exit_time, exit_price), xytext=(entry_time, entry_price),
                        arrowprops=dict(arrowstyle="->", color=color, lw=1.5), zorder=4)

    axs[1].set_title("Price Chart with Positions & Trades")
    axs[1].legend()
    axs[1].grid()

    # === 單筆交易報酬分布 ===
    axs[2].hist(np.array(trade_returns) * 100, bins=20, color="skyblue", edgecolor="k")
    axs[2].set_title("Trade Return Distribution (%)")
    axs[2].set_xlabel("Return (%)")
    axs[2].grid()

    plt.tight_layout()
    plt.show()
