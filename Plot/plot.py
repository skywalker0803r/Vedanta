import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_backtest_result(result, max_trades_to_draw=10, max_points=3000):
    fig_data = result["fig"]
    trades = result.get("trades_log", [])

    # 原始資料
    timestamp = pd.to_datetime(fig_data["timestamp"])
    close = np.array(fig_data["close"])
    equity = np.array(fig_data["equity"])
    buy_and_hold = np.array(fig_data["buy_and_hold"])
    position = np.array(fig_data["position"])
    signal = np.array(fig_data["signal"])
    trade_returns = [t['return'] for t in trades]

    # Downsampling（避免繪圖過慢）
    if len(timestamp) > max_points:
        step = len(timestamp) // max_points
        timestamp_ds = timestamp[::step]
        close = close[::step]
        equity = equity[::step]
        buy_and_hold = buy_and_hold[::step]
        position = position[::step]
        signal = signal[::step]
    else:
        timestamp_ds = timestamp

    # 計算報酬率
    equity_pct = (equity / equity[0] - 1) * 100
    buy_and_hold_pct = (buy_and_hold / buy_and_hold[0] - 1) * 100

    # 最大回撤計算（使用 downsample 後的 equity）
    dd = equity / np.maximum.accumulate(equity) - 1
    min_dd_idx = np.argmin(dd)
    peak_idx = np.argmax(equity[:min_dd_idx + 1])

    # --- 畫圖開始 ---
    fig, axs = plt.subplots(3, 1, figsize=(15, 11), gridspec_kw={"height_ratios": [2.5, 2, 1]})

    # === [1] Equity Curve ===
    axs[0].plot(timestamp_ds, equity_pct, label="Strategy Return (%)", color="blue", linewidth=2)
    axs[0].plot(timestamp_ds, buy_and_hold_pct, label="Buy and Hold (%)", linestyle="--", alpha=0.7, color="orange")
    axs[0].axvspan(timestamp_ds[peak_idx], timestamp_ds[min_dd_idx], color='red', alpha=0.1, label='Max Drawdown')

    axs[0].set_title("Equity Curve (%)")
    axs[0].set_ylabel("Return (%)")
    axs[0].legend()
    axs[0].grid()

    # === [2] 價格走勢與部位區間 ===
    axs[1].plot(timestamp_ds, close, color='black', linewidth=1)
    axs[1].fill_between(timestamp_ds, close, where=position > 0, color='green', alpha=0.15, label="Long")
    axs[1].fill_between(timestamp_ds, close, where=position < 0, color='red', alpha=0.15, label="Short")

    # 訊號點
    buy_idx = np.where(signal == 1)[0]
    sell_idx = np.where(signal == -1)[0]
    if len(buy_idx) > 0:
        axs[1].scatter(timestamp_ds[buy_idx], close[buy_idx], marker="^", color="green", label="Buy Signal", zorder=5)
    if len(sell_idx) > 0:
        axs[1].scatter(timestamp_ds[sell_idx], close[sell_idx], marker="v", color="red", label="Sell Signal", zorder=5)

    # Entry/Exit 點（最多畫 N 筆）
    for i, trade in enumerate(trades[:max_trades_to_draw]):
        entry_time = pd.to_datetime(trade["entry_time"])
        exit_time = pd.to_datetime(trade["exit_time"])
        entry_price = trade["entry_price"]
        exit_price = trade["exit_price"]
        color = "green" if trade["side"] == "long" else "red"
        axs[1].scatter(entry_time, entry_price, color=color, marker='o', edgecolor='black', zorder=6,
                       label="Entry" if i == 0 else None)
        axs[1].scatter(exit_time, exit_price, color=color, marker='x', edgecolor='black', zorder=6,
                       label="Exit" if i == 0 else None)
        # （可選）畫箭頭會變慢
        # axs[1].annotate("", xy=(exit_time, exit_price), xytext=(entry_time, entry_price),
        #                 arrowprops=dict(arrowstyle="->", color=color, lw=1.2), zorder=4)

    axs[1].set_title("Price Chart with Positions & Trades")
    axs[1].legend()
    axs[1].grid()

    # === [3] 單筆交易報酬分布 ===
    axs[2].hist(np.array(trade_returns) * 100, bins=20, color="skyblue", edgecolor="k")
    axs[2].set_title("Trade Return Distribution (%)")
    axs[2].set_xlabel("Return (%)")
    axs[2].grid()

    plt.tight_layout()
    plt.show()
