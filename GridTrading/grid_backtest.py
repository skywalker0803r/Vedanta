
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

# --- 回測設定 ---
# 重要：當 upper_price 和 lower_price 皆為 0 時，會啟用「動態範圍模式」
#       程式會自動根據歷史數據的 2% 和 98% 分位數來設定網格範圍。
#       若要使用手動範圍，請填入非零值。
BACKTEST_CONFIG = {
    "currency_pair": "BTC_USDT", # 交易對
    "interval": "4h",            # K線週期 (e.g., "15m", "4h", "1d")
    "data_limit": 500,           # 要獲取的 K 線數量

    "upper_price": Decimal("0"), # 網格上限價格 (設為0以啟用動態範圍)
    "lower_price": Decimal("0"), # 網格下限價格 (設為0以啟用動態範圍)
    "grid_num": 100,                  # 網格數量
    "total_investment": Decimal("10000"), # 總投資額 (USDT)
    "fee_rate": Decimal("0.002")     # 交易手續費率 (0.2%)
}

# --- Gate.io API 設定 ---
# 回測只需要公開數據，所以不需要 API Key
# LIVE_HOST = "https://api.gateio.ws/api/v4"
TESTNET_HOST = "https://api-testnet.gateapi.io/api/v4" # 使用測試網獲取數據
HOST = TESTNET_HOST

def fetch_gateio_klines(spot_api, currency_pair, interval, limit) -> pd.DataFrame:
    """使用 gate-api 獲取 K 線數據並轉換為 pandas DataFrame"""
    print(f"正在從 Gate.io 獲取 {currency_pair} 的 {limit} 條 {interval} K線數據...")
    try:
        # list_candlesticks(currency_pair, limit, interval)
        candlesticks = spot_api.list_candlesticks(currency_pair=currency_pair, limit=limit, interval=interval)
        
        if not candlesticks:
            print("錯誤：未獲取到任何數據。 সন")
            return pd.DataFrame()

        # 轉換為 DataFrame
        # 根據錯誤訊息，API回傳了8個欄位。我們為所有欄位命名，然後選取我們需要的。
        # Gate.io API v4 文件說明格式為: [timestamp, quote_volume, close, high, low, open]
        # 我們假設多的兩個欄位是 base_volume 和 turnover 或其他未使用到的欄位。
        all_columns = ['timestamp', 'quote_volume', 'close', 'high', 'low', 'open', 'base_volume', 'turnover']
        df = pd.DataFrame(candlesticks, columns=all_columns)

        # 為了後續處理，我們重新排列並選取標準的 OHLCV 格式
        # 注意：Gate.io 的 'volume' 通常指 'quote_volume' (e.g., USDT)
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'quote_volume']]
        df.rename(columns={'quote_volume': 'volume'}, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        df.set_index('timestamp', inplace=True)
        
        # 轉換欄位為數值類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        print("數據獲取成功！")
        return df.sort_index(ascending=True)

    except gate_api.exceptions.GateApiException as ex:
        print(f"Gate API 錯誤: {ex}")
        return pd.DataFrame()
    except Exception as e:
        print(f"發生未知錯誤: {e}")
        return pd.DataFrame()

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
    
    quote_per_grid = total_investment / grid_num # 每格分配的報價貨幣 (USDT)
    
    print(f"網格區間: {lower_price.quantize(Decimal('0.01'))} - {upper_price.quantize(Decimal('0.01'))} USDT")
    print(f"網格數量: {grid_num}")
    print(f"每格資金: {quote_per_grid.quantize(Decimal('0.01'))} USDT")

    # --- 2. 初始化帳戶和統計 ---
    base_currency_balance = Decimal("0")  # e.g., BTC
    quote_currency_balance = total_investment # e.g., USDT
    
    trades = []
    grid_profit = Decimal("0")
    buy_records = {} # 記錄每個網格的買入價格
    portfolio_history = [] # 用於記錄每個時間點的總資產

    # --- 3. 遍歷歷史數據進行模擬 ---
    last_grid_index = -1

    for timestamp, row in df_klines.iterrows():
        price = Decimal(str(row['close']))

        # 找出當前價格所在的網格索引
        current_grid_index = -1
        for i in range(len(grid_lines) - 1):
            if grid_lines[i] <= price < grid_lines[i+1]:
                current_grid_index = i
                break
        
        if last_grid_index == -1: # 首次運行
            last_grid_index = current_grid_index
            portfolio_history.append(total_investment) # 初始資產
            continue

        if current_grid_index != -1 and current_grid_index != last_grid_index:
            # 價格穿越網格
            start_grid = min(last_grid_index, current_grid_index)
            end_grid = max(last_grid_index, current_grid_index)

            for grid_idx in range(start_grid, end_grid + 1):
                grid_price = grid_lines[grid_idx]

                # 價格下跌，穿越網格線，執行買入
                if price < grid_price and grid_idx not in buy_records:
                    if quote_currency_balance >= quote_per_grid:
                        amount_to_buy = quote_per_grid / grid_price
                        fee = amount_to_buy * fee_rate
                        base_currency_balance += (amount_to_buy - fee)
                        quote_currency_balance -= quote_per_grid
                        
                        buy_records[grid_idx] = grid_price
                        trades.append({
                            "timestamp": timestamp, "type": "buy", "price": grid_price, 
                            "amount": amount_to_buy, "profit": 0
                        })

                # 價格上漲，穿越網格線，執行賣出
                elif price > grid_price and grid_idx in buy_records:
                    amount_to_sell = quote_per_grid / buy_records[grid_idx] # 用記錄的買價計算賣出數量
                    
                    if base_currency_balance >= amount_to_sell:
                        buy_price = buy_records.pop(grid_idx) # 交易成功後才移除紀錄
                        sell_value = amount_to_sell * grid_price
                        fee = sell_value * fee_rate
                        quote_currency_balance += (sell_value - fee)
                        base_currency_balance -= amount_to_sell
                        
                        profit = sell_value - (amount_to_sell * buy_price)
                        grid_profit += profit
                        
                        trades.append({
                            "timestamp": timestamp, "type": "sell", "price": grid_price, 
                            "amount": amount_to_sell, "profit": profit
                        })

            last_grid_index = current_grid_index

        # 記錄當前時間點的總資產
        current_portfolio_value = quote_currency_balance + (base_currency_balance * price)
        portfolio_history.append(current_portfolio_value)

    # --- 4. 計算最終結果 ---
    print("\n--- 回測結果 ---")
    
    final_price = Decimal(str(df_klines['close'].iloc[-1]))
    
    # 1. 計算最終資產總價值
    final_portfolio_value = quote_currency_balance + (base_currency_balance * final_price)
    
    # 2. 計算總盈虧 (最準確的方式)
    total_pnl = final_portfolio_value - total_investment
    
    # 3. 計算浮動盈虧
    # 浮動盈虧 = 持有資產的現值 - 持有資產的成本
    cost_of_open_positions = Decimal(str(len(buy_records))) * quote_per_grid
    floating_pnl = (base_currency_balance * final_price) - cost_of_open_positions

    # 4. 計算已實現盈虧
    # 已實現盈虧 = 總盈虧 - 浮動盈虧
    realized_profit = total_pnl - floating_pnl
    
    # 計算年化報酬率
    days = (df_klines.index[-1] - df_klines.index[0]).days
    annualized_return = (total_pnl / total_investment) * (Decimal("365") / Decimal(str(days))) if days > 0 else 0

    print(f"回測時間: {df_klines.index[0]} to {df_klines.index[-1]} ({days} 天)")
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
    print(f"當前持倉格數: {len(buy_records)}")

    return portfolio_history


def plot_performance_comparison(df_klines, grid_equity_curve, investment):
    """繪製網格交易與買入持有策略的績效對比圖"""
    if not grid_equity_curve:
        print("無法繪圖：無有效的權益曲線數據。")
        return

    # 確保權益曲線長度與K線數據長度一致
    if len(grid_equity_curve) != len(df_klines):
        print(f"警告：權益曲線長度({len(grid_equity_curve)})與K線數據長度({len(df_klines)})不匹配，無法繪圖。")
        return

    # 計算買入並持有策略的權益曲線
    initial_price = df_klines['close'].iloc[0]
    buy_and_hold_equity = (float(investment) / initial_price) * df_klines['close']

    # 繪圖
    plt.style.use('seaborn-v0_8-darkgrid') # 使用一個好看的樣式
    fig, ax = plt.subplots(figsize=(15, 8))

    ax.plot(df_klines.index, grid_equity_curve, label='Grid Trading Strategy', color='#1f77b4', linewidth=2)
    ax.plot(df_klines.index, buy_and_hold_equity, label='Buy and Hold Strategy', color='#ff7f0e', linestyle='--')

    # 格式化圖表
    ax.set_title('Grid Trading vs. Buy and Hold Performance', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Portfolio Value (USDT)', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True)
    
    # 格式化Y軸標籤為貨幣格式
    from matplotlib.ticker import FuncFormatter
    formatter = FuncFormatter(lambda x, p: f'${x:,.0f}')
    ax.yaxis.set_major_formatter(formatter)

    fig.tight_layout() # 自動調整邊距
    plt.show()


def main():
    """主執行函數"""
    # --- 初始化 API 客戶端 ---
    conf = gate_api.Configuration(host=HOST)
    api_client = gate_api.ApiClient(conf)
    spot_api = gate_api.SpotApi(api_client)

    # --- 獲取數據 ---
    df_klines = fetch_gateio_klines(
        spot_api,
        currency_pair=BACKTEST_CONFIG["currency_pair"],
        interval=BACKTEST_CONFIG["interval"],
        limit=BACKTEST_CONFIG["data_limit"]
    )

    if df_klines.empty:
        return

    # --- 檢查並設定網格區間 ---
    # 如果使用者沒有設定價格區間 (預設為0)，則動態計算
    if BACKTEST_CONFIG["lower_price"] == Decimal("0") and BACKTEST_CONFIG["upper_price"] == Decimal("0"):
        print("\n未設定手動價格區間，啟用動態範圍模式...")
        # 使用分位數來避免極端值對網格範圍的影響
        low_quantile = df_klines['low'].quantile(0.02)   # 取2%分位數為下界
        high_quantile = df_klines['high'].quantile(0.98)  # 取98%分位數為上界

        # 使用過濾後的價格範圍來設定網格
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
    # 注意：回測本身仍然在完整的原始數據上運行
    portfolio_history = run_grid_backtest(df_klines, BACKTEST_CONFIG)

    # --- 繪製績效圖 ---
    if portfolio_history:
        plot_performance_comparison(df_klines, portfolio_history, BACKTEST_CONFIG["total_investment"])


if __name__ == "__main__":
    main()
