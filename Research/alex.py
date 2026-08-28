# 1. 安裝必要的第三方套件 (ccxt 用於抓取歷史 K 線數據)
!pip install ccxt pandas numpy matplotlib

import ccxt
import pandas as pd
import numpy as np
import math
from datetime import datetime, timedelta, timezone
import matplotlib.pyplot as plt

# ==========================================
# 2. 核心幾何網格公式 (Math Core)
# ==========================================
def seq_order(i: int, j: int, price_r: float, vol_r: float, r: float):
    """
    計算第 i 層網格的價格與數量
    i: 層數 (1, 2, 3...)
    j: 方向 (1 為賣單/向上, -1 為買單/向下)
    """
    price = price_r * math.pow(1 + r, i * j)
    sng_vol_step = (1 + 0.5 * r) / (1 + r)
    volume = vol_r * math.pow(sng_vol_step, i * j) * (1 - math.pow(sng_vol_step, -j)) * (-j)
    return abs(price), abs(volume)

# ==========================================
# 3. 抓取 Binance 過去一年歷史 K 線 (OHLCV)
# ==========================================
def fetch_btc_historical_data():
    print("正在從 Binance 擷取過去 365 天的 BTC/USDT 每日 K 線數據...")
    exchange = ccxt.binance()
    
    # 計算 365 天前的時間戳 (UTC)
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=365)
    since = int(start_time.timestamp() * 1000)
    
    ohlcv = []
    limit = 1000
    while since < int(now.timestamp() * 1000):
        data = exchange.fetch_ohlcv('BTC/USDT', timeframe='1d', since=since, limit=limit)
        if not data:
            break
        ohlcv.extend(data)
        since = data[-1][0] + 1
        
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.sort_values('date').reset_index(drop=True)
    print(f"成功擷取 {len(df)} 筆 K 線數據 (區間: {df['date'].iloc[0].strftime('%Y-%m-%d')} 至 {df['date'].iloc[-1].strftime('%Y-%m-%d')})")
    return df

# ==========================================
# 4. 網格回測引擎 (Backtest Engine)
# ==========================================
def run_grid_backtest(df, r=0.01, grid_levels=10, initial_usdt=10000, fee_rate=0.001):
    """
    r: 網格間距 (1%)
    grid_levels: 單邊掛單層數 (預設上下各 10 層)
    initial_usdt: 起始資金 (USDT)
    fee_rate: 手續費 (預設 0.1%)
    """
    start_price = df['close'].iloc[0]
    vol_r = 0.001 # 基準交易數量 (BTC)
    
    # 建立雙向網格
    # key: price, value: dict(side, amount, level)
    grid_orders = {}
    
    def reset_grid(center_price):
        grid = {}
        # 建立向上賣單 (j=1) 與 向下買單 (j=-1)
        for j in [1, -1]:
            side = 'SELL' if j == 1 else 'BUY'
            for i in range(1, grid_levels + 1):
                p, v = seq_order(i, j, center_price, vol_r, r)
                grid[round(p, 2)] = {'side': side, 'amount': v, 'level': i, 'j': j}
        return grid

    # 初始帳戶資產狀態
    usdt_balance = initial_usdt
    btc_balance = 0.0
    
    # 為了實現現貨網格，預先劃分一部分 USDT 購買初始 BTC 以便發送賣單
    # 假設預先劃分 50% USDT 買入 BTC
    initial_btc_spend = initial_usdt * 0.5
    btc_balance += (initial_btc_spend * (1 - fee_rate)) / start_price
    usdt_balance -= initial_btc_spend
    
    current_center_price = start_price
    grid_orders = reset_grid(current_center_price)
    
    portfolio_history = []
    trade_count = 0
    grid_profit_usdt = 0.0
    
    # 開始逐日回測
    for idx, row in df.iterrows():
        high = row['high']
        low = row['low']
        close = row['close']
        
        # 檢查當天最高/最低價是否觸發網格掛單
        triggered_prices = []
        for p, order in list(grid_orders.items()):
            # 買單觸發: 當日最低價低於等於掛單價
            if order['side'] == 'BUY' and low <= p:
                cost = p * order['amount']
                if usdt_balance >= cost:
                    usdt_balance -= cost
                    btc_acquired = order['amount'] * (1 - fee_rate)
                    btc_balance += btc_acquired
                    trade_count += 1
                    triggered_prices.append(p)
                    # 每成交一單，重新以成交價為基準點重新掛單
                    current_center_price = p
            
            # 賣單觸發: 當日最高價高於等於掛單價
            elif order['side'] == 'SELL' and high >= p:
                if btc_balance >= order['amount']:
                    btc_balance -= order['amount']
                    revenue = p * order['amount'] * (1 - fee_rate)
                    usdt_balance += revenue
                    trade_count += 1
                    triggered_prices.append(p)
                    current_center_price = p

        # 若有觸發交易，根據最新的基準價更新網格
        if triggered_prices:
            grid_orders = reset_grid(current_center_price)
            
        # 計算當日總資產價值 (Total Portfolio Value)
        total_value = usdt_balance + (btc_balance * close)
        portfolio_history.append({
            'date': row['date'],
            'close': close,
            'usdt_balance': usdt_balance,
            'btc_balance': btc_balance,
            'total_value': total_value
        })

    res_df = pd.DataFrame(portfolio_history)
    return res_df, trade_count, initial_usdt

# ==========================================
# 5. 執行回測與數據視覺化 (Run & Plot)
# ==========================================
df_btc = fetch_btc_historical_data()
results_df, total_trades, initial_capital = run_grid_backtest(
    df_btc, 
    r=0.01,          # 1% 網格間距
    grid_levels=10,  # 上下各 10 層
    initial_usdt=10000 # 起始資金 10,000 USDT
)

# 計算績效指標
final_value = results_df['total_value'].iloc[-1]
net_profit = final_value - initial_capital
roi = (net_profit / initial_capital) * 100

# 計算 BTC Buy & Hold (買入持有) 績效做為基準對比
btc_initial_price = results_df['close'].iloc[0]
btc_final_price = results_df['close'].iloc[-1]
hold_roi = ((btc_final_price - btc_initial_price) / btc_initial_price) * 100

print("\n" + "="*40)
print(" 📊 幾何網格回測績效報告 (過去 365 天)")
print("="*40)
print(f"起始資金    : {initial_capital:,.2f} USDT")
print(f"最終資產淨值: {final_value:,.2f} USDT")
print(f"淨獲利      : {net_profit:,.2f} USDT")
print(f"網格策略 ROI: {roi:.2f}%")
print(f"買入持有 ROI: {hold_roi:.2f}% (基準對比)")
print(f"總成交筆數  : {total_trades} 次")
print("="*40)

# 繪製回測資產曲線圖
plt.figure(figsize=(12, 6))
plt.plot(results_df['date'], results_df['total_value'], label='Grid Strategy Portfolio (USDT)', color='green', linewidth=2)
plt.axhline(y=initial_capital, color='red', linestyle='--', label='Initial Capital')
plt.title('BTC Geometric Grid Strategy Backtest (1 Year)', fontsize=14)
plt.xlabel('Date')
plt.ylabel('Portfolio Value (USDT)')
plt.legend()
plt.grid(True)
plt.show()