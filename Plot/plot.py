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
    trade_returns = [float(t['P&L (%)'].replace('%', '').replace(',', '')) for t in trades]

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
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={"height_ratios": [2.5, 2, 1]})

    # === [1] Equity Curve ===
    # === [1] Equity Curve ===
    # Conditional coloring for Strategy Return
    axs[0].plot(timestamp_ds, buy_and_hold_pct, label="Buy and Hold (%)", linestyle="--", alpha=0.7, color="blue")

    # Plot strategy equity curve with conditional coloring and fill
    # Find points where equity crosses the 0-axis
    equity_pct_green = np.where(equity_pct >= 0, equity_pct, np.nan)
    equity_pct_red = np.where(equity_pct < 0, equity_pct, np.nan)

    # Plot green line and fill for positive equity
    axs[0].plot(timestamp_ds, equity_pct_green, label="Strategy Return (%)", color="green", linewidth=2)
    axs[0].fill_between(timestamp_ds, 0, equity_pct, where=equity_pct >= 0, facecolor='green', alpha=0.1)

    # Plot red line and fill for negative equity
    axs[0].plot(timestamp_ds, equity_pct_red, color="red", linewidth=2)
    axs[0].fill_between(timestamp_ds, 0, equity_pct, where=equity_pct < 0, facecolor='red', alpha=0.1)

    # Draw a horizontal line at 0 for reference
    axs[0].axhline(0, color='gray', linestyle='--', linewidth=0.8)

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
        entry_time = pd.to_datetime(trade["Date/Time (Entry)"], errors='coerce')
        exit_time = pd.to_datetime(trade["Date/Time (Exit)"], errors='coerce')
        entry_price = float(trade["Price (Entry)"].replace(',', ''))
        exit_price = float(trade["Price (Exit)"].replace(',', ''))
        color = "green" if trade["Type"] == "Long" else "red"
        axs[1].scatter(entry_time, entry_price, color=color, marker='o', edgecolor='black', zorder=6,
                       label="Entry" if i == 0 else None)
        # Only plot exit if exit_time is a valid datetime (not NaT)
        if pd.notna(exit_time):
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

def display_trades_log_as_html(trades_log):
    html_output = """
    <style>
      body {
        background-color: #1a1a1a;
        color: #e0e0e0;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      }
      .trade-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 14px;
      }
      .trade-table th, .trade-table td {
        padding: 12px;
        border-bottom: 1px solid #333;
        text-align: left;
        white-space: nowrap;
        vertical-align: top;
      }
      .trade-table th {
        color: #999;
        font-weight: normal;
        text-align: right;
      }
      .trade-table td {
        color: #e0e0e0;
        font-weight: 500;
      }
      .trade-table .header-row th {
        border-bottom: 2px solid #555;
        padding-bottom: 15px;
        text-align: left;
      }
      .trade-table .trade-id {
        color: #007bff;
      }
      .trade-table .short-trade-id {
        color: #ff3333;
      }
      .trade-table .value-col {
        text-align: right;
      }
      .trade-table .positive-value {
        color: #00ff00;
      }
      .trade-table .negative-value {
        color: #ff3333;
      }
      .trade-table .merged-cell {
        border-bottom: none;
      }
    </style>
    <table class="trade-table">
      <thead>
        <tr class="header-row">
          <th>Trade #</th>
          <th>Type</th>
          <th>Date/Time</th>
          <th>Signal</th>
          <th>Price</th>
          <th>Position size</th>
          <th>P&L</th>
          <th>Run-up</th>
          <th>Drawdown</th>
          <th>Cumulative P&L</th>
        </tr>
      </thead>
      <tbody>
    """

    for i, trade in enumerate(reversed(trades_log)): # Iterate in reverse to show latest trades first
        trade_number = len(trades_log) - i
        trade_type_class = "trade-id" if trade['Type'] == 'Long' else "short-trade-id"
        trade_type_text = f"{trade_number} {trade['Type']}"

        # P&L formatting
        pnl_usdt = float(trade['P&L (USDT)'].replace(',', ''))
        pnl_class = "positive-value" if pnl_usdt >= 0 else "negative-value"
        pnl_text = f"<div>{trade['P&L (USDT)']} usdt</div><div>{trade['P&L (%)']}</div>"

        # Run-up formatting
        run_up_pct = float(trade['Run-up (%)'].replace('%', '').replace(',', ''))
        run_up_class = "positive-value" if run_up_pct >= 0 else "negative-value" # Drawdown is usually negative, so positive value means less negative
        run_up_text = f"<div>{trade['Run-up (%)']}</div>"

        # Drawdown formatting
        draw_down_pct = float(trade['Drawdown (%)'].replace('%', '').replace(',', ''))
        draw_down_class = "positive-value" if draw_down_pct >= 0 else "negative-value" # Drawdown is usually negative, so positive value means less negative
        draw_down_text = f"<div>{trade['Drawdown (%)']}</div>"

        # Cumulative P&L formatting
        cumulative_pnl_usdt = float(trade['Cumulative P&L'].replace(',', ''))
        cumulative_pnl_class = "positive-value" if cumulative_pnl_usdt >= 0 else "negative-value"
        cumulative_pnl_text = f"<div>{trade['Cumulative P&L']} usdt</div><div>{trade['Cumulative P&L (%)']}</div>"


        html_output += f"""
        <tr>
          <td rowspan="2" class="{trade_type_class} merged-cell">{trade_type_text}</td>
          <td class="merged-cell">Exit</td>
          <td class="merged-cell">{trade['Date/Time (Exit)']}</td>
          <td class="merged-cell">{trade['Signal (Exit)']}</td>
          <td class="merged-cell">{trade['Price (Exit)']} usdt</td>
          <td rowspan="2">
            <div>{trade['Position size']}</div>
          </td>
          <td rowspan="2" class="{pnl_class}">
            {pnl_text}
          </td>
          <td rowspan="2" class="{run_up_class}">
            {run_up_text}
          </td>
          <td rowspan="2" class="{draw_down_class}">
            {draw_down_text}
          </td>
          <td rowspan="2" class="{cumulative_pnl_class}">
            {cumulative_pnl_text}
          </td>
        </tr>
        <tr>
          <td style="border-top: 1px solid #333;">Entry</td>
          <td style="border-top: 1px solid #333;">{trade['Date/Time (Entry)']}</td>
          <td style="border-top: 1px solid #333;">{trade['Signal (Entry)']}</td>
          <td style="border-top: 1px solid #333;">{trade['Price (Entry)']} usdt</td>
        </tr>
        <tr>
          <td colspan="10" style="padding: 0; height: 20px;"></td>
        </tr>
        """
    html_output += """
      </tbody>
    </table>
    """
    return html_output
