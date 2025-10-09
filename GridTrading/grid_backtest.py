
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
#
# --- 模式一：使用日期範圍 (推薦) ---
# 填寫 start_date 和 end_date (格式: "YYYY-MM-DD") 來指定回測期間。
# 此模式下，data_limit 會被忽略。
#
# --- 模式二：使用最新K線 ---
# 將 start_date 和 end_date 設為 None，程式會改用 data_limit 來獲取最新的 N 條K線。
#
BACKTEST_CONFIG = {
    "currency_pair": "BTC_USDT", # 交易對
    "interval": "4h",            # K線週期 (e.g., "15m", "4h", "1d")

    "start_date": "2025-09-01",      # 開單日期 (設為 None 則不使用)
    "end_date": "2025-10-08",        # 關單日期 (設為 None 則不使用)
    "data_limit": 500,               # 僅在未指定日期時生效

    # --- 網格策略參數 ---
    # 重要：當 upper_price 和 lower_price 皆為 0 時，會啟用「動態範圍模式」
    #       程式會自動根據歷史數據的 2% 和 98% 分位數來設定網格範圍。
    #       若要使用手動範圍，請填入非零值。
    "upper_price": Decimal("0"), # 網格上限價格 (設為0以啟用動態範圍)
    "lower_price": Decimal("0"), # 網格下限價格 (設為0以啟用動態範圍)
    "grid_num": 100,                  # 網格數量 (增加網格數以提高交易頻率)
    "total_investment": Decimal("10000"), # 總投資額 (USDT)
    "fee_rate": Decimal("0")#Decimal("0.0005")     # 交易手續費率 (0.05%)
}

# --- Gate.io API 設定 ---
# 回測只需要公開數據，所以不需要 API Key
# LIVE_HOST = "https://api.gateio.ws/api/v4"
TESTNET_HOST = "https://api-testnet.gateapi.io/api/v4" # 使用測試網獲取數據
HOST = TESTNET_HOST

from datetime import datetime

def fetch_gateio_klines(spot_api, currency_pair, interval, limit, start_date_str=None, end_date_str=None) -> pd.DataFrame:
    """使用 gate-api 獲取 K 線數據並轉換為 pandas DataFrame"""
    print("\n正在從 Gate.io 獲取 K線數據...")
    all_candlesticks = []

    try:
        if start_date_str and end_date_str:
            # --- 日期範圍模式 ---
            from_ts = int(datetime.strptime(start_date_str, "%Y-%m-%d").timestamp())
            to_ts = int(datetime.strptime(end_date_str, "%Y-%m-%d").timestamp())
            print(f"模式: 日期範圍 | 從 {start_date_str} 到 {end_date_str}")

            current_from = from_ts
            while current_from < to_ts:
                # Gate.io API 單次請求上限為 1000 條
                candlesticks = spot_api.list_candlesticks(
                    currency_pair=currency_pair, 
                    _from=current_from, 
                    to=to_ts, 
                    interval=interval,
                    limit=1000 
                )
                if not candlesticks:
                    break # 該範圍內已無更多數據

                all_candlesticks.extend(candlesticks)
                
                # 獲取最後一根K棒的時間戳，並加1秒作為下一次請求的起點，以避免重疊
                last_ts = int(candlesticks[-1][0])
                print(f"  已獲取 {len(candlesticks)} 條數據，目前時間點: {datetime.fromtimestamp(last_ts).strftime('%Y-%m-%d %H:%M:%S')}")

                if last_ts >= to_ts:
                    break # 已達到或超過結束日期

                current_from = last_ts + 1 
                time.sleep(0.3) # 友善對待 API，避免請求過於頻繁
        else:
            # --- 最新K線數量模式 ---
            print(f"模式: 最新K線 | 獲取最新的 {limit} 條數據")
            all_candlesticks = spot_api.list_candlesticks(currency_pair=currency_pair, limit=limit, interval=interval)

        if not all_candlesticks:
            print("錯誤：未獲取到任何數據。")
            return pd.DataFrame()

        # --- 數據處理 (通用) ---
        # 根據錯誤訊息，API回傳了8個欄位。我們為所有欄位命名，然後選取我們需要的。
        all_columns = ['timestamp', 'quote_volume', 'close', 'high', 'low', 'open', 'base_volume', 'turnover']
        df = pd.DataFrame(all_candlesticks, columns=all_columns)

        # 為了後續處理，我們重新排列並選取標準的 OHLCV 格式
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'quote_volume']]
        df.rename(columns={'quote_volume': 'volume'}, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # 移除重複數據並按時間排序
        df.drop_duplicates(subset=['timestamp'], inplace=True)
        df.set_index('timestamp', inplace=True)
        df.sort_index(ascending=True, inplace=True)
        
        # 轉換欄位為數值類型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])
            
        print(f"數據獲取成功！總共 {len(df)} 條不重複數據。")
        return df

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
        limit=BACKTEST_CONFIG["data_limit"],
        start_date_str=BACKTEST_CONFIG["start_date"],
        end_date_str=BACKTEST_CONFIG["end_date"]
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
