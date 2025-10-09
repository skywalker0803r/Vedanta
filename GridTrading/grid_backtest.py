
"""
Gate.io 網格交易回測腳本
- 功能：
  1. 從 Gate.io 獲取歷史 K 線數據。
  2. 根據設定的參數（價格區間、網格數、總投資）執行回測。
  3. 計算並輸出網格交易的績效指標。
- 使用方式：
  1. 修改下方的 `BACKTEST_CONFIG` 來設定回測參數。
  2. 執行 `python 網格交易/grid_backtest.py`。
"""
import gate_api
import pandas as pd
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_DOWN
import time
from datetime import datetime
import requests

# --- 回測設定 ---
#
# --- 數據獲取設定 (資料來源: 幣安 Binance) ---
# 程式會獲取在 `end_date` 結束時，往前 `data_limit` 根 K 線。
# 將 end_date 設為 None 可使用當前最新時間。
#
BACKTEST_CONFIG = {
    "currency_pair": "XRPUSDT",      # 交易對 (注意：幣安格式為 BTCUSDT，無底線)
    "interval": "1h",                # K線週期 (e.g., "15m", "4h", "1d")
    "end_date": None,                # 結束日期 (格式: "YYYY-MM-DD" 或 None)
    "data_limit": 1000,              # 要獲取的 K 線數量

    # --- 網格策略參數 ---
    # 重要：當 upper_price 和 lower_price 皆為 0 時，會啟用「動態範圍模式」
    "upper_price": Decimal("0"), # 網格上限價格 (設為0以啟用動態範圍)
    "lower_price": Decimal("0"), # 網格下限價格 (設為0以啟用動態範圍)
    "grid_num": 100,                  # 網格數量
    "total_investment": Decimal("10000"), # 總投資額 (USDT)
    "fee_rate": Decimal("0.0005")     # 交易手續費率 (幣安現貨為 0.05%)
}

# --- Gate.io API 設定 ---
# 回測只需要公開數據，所以不需要 API Key
# LIVE_HOST = "https://api.gateio.ws/api/v4"
TESTNET_HOST = "https://api-testnet.gateapi.io/api/v4" # 使用測試網獲取數據
HOST = TESTNET_HOST

def get_binance_kline(symbol: str, interval: str, end_time: datetime, total_limit: int = 1000) -> pd.DataFrame:
    """使用 requests 直接從幣安 API 獲取 K 線數據"""
    base_url = "https://api.binance.com/api/v3/klines"
    all_data = []
    # 幣安 API 的 endTime 是包含的，所以我們直接用傳入的時間
    end_timestamp = int(end_time.timestamp() * 1000)
    remaining = total_limit

    print(f"\n正在從 Binance 獲取 {symbol} 的 {total_limit} 條 {interval} K線數據...")
    print(f"結束時間點: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    while remaining > 0:
        # 幣安 API 單次請求上限為 1000 條
        fetch_limit = min(1000, remaining)
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "endTime": end_timestamp,
            "limit": fetch_limit
        }

        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status() # 如果請求失敗則拋出異常
            data = response.json()
        except requests.exceptions.RequestException as e:
            print(f"API 請求錯誤: {e}")
            break

        if not data:
            break # 沒有更多數據了

        all_data = data + all_data  # 將舊的數據拼接到前面
        
        # 下一次請求的結束時間是本次獲取到的第一條數據的時間戳再減 1 毫秒
        end_timestamp = data[0][0] - 1
        remaining -= len(data)
        
        print(f"  已獲取 {len(data)} 條數據，目前最早時間點: {datetime.fromtimestamp(data[0][0]/1000).strftime('%Y-%m-%d %H:%M:%S')}")
        time.sleep(0.3)  # 友善對待 API

    if not all_data:
        raise ValueError("從幣安獲取數據失敗，請檢查參數。")

    df = pd.DataFrame(all_data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])

    # --- 數據格式化 ---
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    
    # 移除重複並排序
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(ascending=True, inplace=True)

    print(f"數據獲取成功！總共 {len(df)} 條不重複數據。")
    return df

def run_grid_backtest(df_klines, config):
    """執行網格交易回測"""
    if df_klines.empty:
        print("K線數據為空，無法進行回測。")
        return None

    print("\n--- 開始進行網格回測 ---")
    
    # --- 1. 初始化網格 ---
    upper_price = config["upper_price"]
    lower_price = config["lower_price"]
    grid_num = config["grid_num"]
    total_investment = config["total_investment"]
    fee_rate = config["fee_rate"]

    grid_lines = [lower_price + (upper_price - lower_price) * i / grid_num for i in range(grid_num + 1)]
    grid_lines = [Decimal(str(p)) for p in grid_lines]
    grid_step = grid_lines[1] - grid_lines[0] # 每格的價差
    
    quote_per_grid = total_investment / grid_num
    
    print(f"網格區間: {lower_price.quantize(Decimal('0.01'))} - {upper_price.quantize(Decimal('0.01'))} USDT")
    print(f"網格數量: {grid_num}")
    print(f"每格價差: {grid_step.quantize(Decimal('0.01'))} USDT")
    print(f"每格資金: {quote_per_grid.quantize(Decimal('0.01'))} USDT")

    # --- 2. 初始化帳戶和統計 ---
    base_currency_balance = Decimal("0")
    quote_currency_balance = total_investment
    trades = []
    grid_profit = Decimal("0")
    # 轉為字典，key是網格線價格，value是買入時的實際數量
    open_positions = {}
    portfolio_history = []

    # --- 3. 核心回測迴圈 (重寫) ---
    for timestamp, row in df_klines.iterrows():
        price = Decimal(str(row['close']))

        # 檢查買入：價格是否跌破了某個沒有掛買單的網格線
        for i in range(grid_num):
            buy_price = grid_lines[i]
            if price < buy_price and buy_price not in open_positions:
                if quote_currency_balance >= quote_per_grid:
                    amount_to_buy = quote_per_grid / buy_price
                    fee = amount_to_buy * fee_rate
                    actual_amount = amount_to_buy - fee

                    base_currency_balance += actual_amount
                    quote_currency_balance -= quote_per_grid
                    
                    open_positions[buy_price] = actual_amount # 記錄倉位
                    trades.append({"timestamp": timestamp, "type": "buy", "price": buy_price, "amount": actual_amount})

        # 檢查賣出：價格是否漲破了某個已掛買單的網格線的「上一級」
        positions_to_close = []
        for buy_price, bought_amount in open_positions.items():
            # 賣出價 = 對應的買入價 + 一個網格間距
            sell_price = buy_price + grid_step
            if price > sell_price:
                sell_value = bought_amount * sell_price
                fee = sell_value * fee_rate

                base_currency_balance -= bought_amount
                quote_currency_balance += (sell_value - fee)

                # 計算本次交易對的利潤
                profit = (sell_value - fee) - (bought_amount * buy_price)
                grid_profit += profit

                trades.append({"timestamp": timestamp, "type": "sell", "price": sell_price, "amount": bought_amount, "profit": profit})
                positions_to_close.append(buy_price)

        # 從持倉中移除已平倉的訂單
        for p in positions_to_close:
            del open_positions[p]

        # 記錄當前時間點的總資產
        current_portfolio_value = quote_currency_balance + (base_currency_balance * price)
        portfolio_history.append(current_portfolio_value)

    # --- 4. 計算最終結果 ---
    print("\n--- 回測結果 ---")
    
    final_price = Decimal(str(df_klines['close'].iloc[-1]))
    
    # 1. 計算最終資產總價值
    final_portfolio_value = quote_currency_balance + (base_currency_balance * final_price)
    
    # 2. 計算總盈虧
    total_pnl = final_portfolio_value - total_investment
    
    # 3. 計算浮動盈虧
    floating_pnl = Decimal("0")
    if open_positions:
        current_value_of_open_positions = base_currency_balance * final_price
        cost_of_open_positions = sum(open_positions.keys()) * (quote_per_grid / sum(open_positions.keys())) # 簡化估算
        # 估算持倉成本
        cost_of_open_positions = Decimal(str(len(open_positions))) * quote_per_grid
        floating_pnl = current_value_of_open_positions - cost_of_open_positions

    # 4. 計算已實現盈虧
    realized_profit = grid_profit
    
    # 計算年化報酬率
    days = (df_klines.index[-1] - df_klines.index[0]).days if len(df_klines) > 1 else 1
    annualized_return = (total_pnl / total_investment) * (Decimal("365") / Decimal(str(days))) if days > 0 else 0

    print(f"回測時間: {df_klines.index[0].date()} to {df_klines.index[-1].date()} ({days} 天)")
    print(f"初始總價值: {total_investment:.2f} USDT")
    print(f"最終總價值: {final_portfolio_value:.2f} USDT")
    print("-" * 20)
    print(f"網格已實現利潤: {realized_profit:.2f} USDT")
    print(f"持倉浮動盈虧: {floating_pnl:.2f} USDT")
    print(f"總盈虧: {total_pnl:.2f} USDT")
    print("-" * 20)
    print(f"總報酬率: {total_pnl / total_investment:.2%}")
    if days > 0:
        print(f"年化報酬率: {annualized_return:.2%}")
    print(f"總交易次數 (買+賣): {len(trades)}")
    print(f"當前持倉格數: {len(open_positions)}")

    return portfolio_history, trades


import matplotlib.animation as animation





def animate_performance_comparison(df_klines, grid_equity_curve, trades, investment):


    """繪製網格交易與買入持有策略的績效對比動畫"""


    if not grid_equity_curve or len(grid_equity_curve) != len(df_klines):


        print("無法繪製動畫：數據長度不匹配或無效。")


        return





    # --- 準備數據 ---


    buy_and_hold_equity = (float(investment) / df_klines['close'].iloc[0]) * df_klines['close']


    buy_trades = [t for t in trades if t['type'] == 'buy']


    sell_trades = [t for t in trades if t['type'] == 'sell']





    # --- 建立圖表和座標軸 ---


    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={'height_ratios': [1, 2]})


    fig.suptitle('Grid Trading vs. Buy and Hold', fontsize=16)





    # 上圖: 權益曲線


    ax1.set_ylabel('Portfolio Value (USDT)')


    ax1.grid(True)


    line_grid, = ax1.plot([], [], lw=2, label='Grid Strategy')


    line_bh, = ax1.plot([], [], lw=2, label='Buy and Hold', linestyle='--')


    ax1.legend()





    # 下圖: 價格與交易點


    ax2.set_ylabel('Price (USDT)')


    ax2.set_xlabel('Date')


    ax2.grid(True)


    line_price, = ax2.plot([], [], lw=1.5, color='#6c757d', label='Price')


    scatter_buys, = ax2.plot([], [], '^', color='#28a745', markersize=8, label='Buy')


    scatter_sells, = ax2.plot([], [], 'v', color='#dc3545', markersize=8, label='Sell')


    ax2.legend()





    # --- 動畫核心函式 ---


    def init():


        # 設定座標軸範圍


        ax1.set_xlim(df_klines.index[0], df_klines.index[-1])


        min_equity = min(buy_and_hold_equity.min(), min(grid_equity_curve)) * 0.98


        max_equity = max(buy_and_hold_equity.max(), max(grid_equity_curve)) * 1.02


        ax1.set_ylim(min_equity, max_equity)





        ax2.set_ylim(df_klines['low'].min() * 0.98, df_klines['high'].max() * 1.02)


        return line_grid, line_bh, line_price, scatter_buys, scatter_sells





    def update(frame):


        # 每次更新一幀 (一個時間點)


        current_time = df_klines.index[frame]


        x_data = df_klines.index[:frame+1]





        # 更新權益曲線


        line_grid.set_data(x_data, grid_equity_curve[:frame+1])


        line_bh.set_data(x_data, buy_and_hold_equity[:frame+1])





        # 更新價格曲線


        line_price.set_data(x_data, df_klines['close'][:frame+1])





        # 更新交易點


        frame_buys_x = [t['timestamp'] for t in buy_trades if t['timestamp'] <= current_time]


        frame_buys_y = [t['price'] for t in buy_trades if t['timestamp'] <= current_time]


        scatter_buys.set_data(frame_buys_x, frame_buys_y)





        frame_sells_x = [t['timestamp'] for t in sell_trades if t['timestamp'] <= current_time]


        frame_sells_y = [t['price'] for t in sell_trades if t['timestamp'] <= current_time]


        scatter_sells.set_data(frame_sells_x, frame_sells_y)


        


        return line_grid, line_bh, line_price, scatter_buys, scatter_sells





    # --- 建立並顯示動畫 ---


    # frames: 總幀數, interval: 每幀之間的毫秒數


    ani = animation.FuncAnimation(fig, update, frames=len(df_klines), init_func=init, blit=True, interval=10)


    plt.tight_layout(rect=[0, 0, 1, 0.96]) # 調整佈局以容納主標題


    plt.show()





def main():


    """主執行函數"""


    # --- 處理日期設定 ---


    end_time = datetime.now() # 預設為當前時間


    if BACKTEST_CONFIG["end_date"]:


        try:


            end_time = datetime.strptime(BACKTEST_CONFIG["end_date"], "%Y-%m-%d")


        except ValueError:


            print(f"錯誤：結束日期格式不正確，請使用 'YYYY-MM-DD' 格式。將使用當前時間代替。")





    # --- 獲取數據 ---


    df_klines = get_binance_kline(


        symbol=BACKTEST_CONFIG["currency_pair"],


        interval=BACKTEST_CONFIG["interval"],


        end_time=end_time,


        total_limit=BACKTEST_CONFIG["data_limit"]


    )





    if df_klines.empty:


        return





    # --- 檢查並設定網格區間 ---


    if BACKTEST_CONFIG["lower_price"] == Decimal("0") and BACKTEST_CONFIG["upper_price"] == Decimal("0"):


        print("\n未設定手動價格區間，啟用動態範圍模式...")


        low_quantile = df_klines['low'].quantile(0.02)


        high_quantile = df_klines['high'].quantile(0.98)





        data_min_price = Decimal(str(low_quantile))


        data_max_price = Decimal(str(high_quantile))


        


        print(f"使用分位數設定網格範圍：")


        print(f"價格上界 (98%): {data_max_price.quantize(Decimal('0.01'))}")


        print(f"價格下界 (2%): {data_min_price.quantize(Decimal('0.01'))}")





        BACKTEST_CONFIG["upper_price"] = data_max_price


        BACKTEST_CONFIG["lower_price"] = data_min_price


    else:


        print("\n檢測到手動設定的價格區間，將使用該設定進行回測。")





    


    # --- 執行回測 ---


    portfolio_history, trades = run_grid_backtest(df_klines, BACKTEST_CONFIG)





    # --- 繪製績效動畫 ---


    if portfolio_history and trades is not None:


        animate_performance_comparison(df_klines, portfolio_history, trades, BACKTEST_CONFIG["total_investment"])


if __name__ == "__main__":
    main()
