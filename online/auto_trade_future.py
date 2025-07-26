import ccxt
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# 載入環境變數
load_dotenv()

# ✅ 建立 Binance Futures 客戶端
def create_binance_futures_client():
    testnet = os.getenv("BINANCE_TESTNET_MODE", "True") == "True"
    client = ccxt.binance({
        'apiKey': os.getenv("BINANCE_API_KEY_FUTURE"),
        'secret': os.getenv("BINANCE_SECRET_FUTURE"),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    client.set_sandbox_mode(testnet)
    print(f"✅ 使用 {'Testnet' if testnet else '主網'} 模式")
    return client

# ✅ 設定槓桿
def set_leverage(client, symbol, leverage):
    try:
        client.set_leverage(leverage, symbol)
        print(f"✅ 槓桿設為 {leverage}x")
    except Exception as e:
        print(f"❌ 槓桿設定失敗: {e}")

# ✅ 取得倉位資訊
def get_position(client, symbol):
    try:
        positions = client.fetch_positions([symbol])
        pos = positions[0]
        amt = float(pos['contracts'])
        side = 'long' if amt > 0 else 'short' if amt < 0 else 'none'
        return amt, side
    except Exception as e:
        print(f"❌ 查持倉錯誤: {e}")
        return 0, 'none'

# ✅ 查詢 USDT 可用餘額
def get_usdt_balance(client):
    try:
        return client.fetch_balance()['USDT']['free']
    except Exception as e:
        print(f"❌ 查餘額錯誤: {e}")
        return 0

# ✅ 取得最小下單量與精度
def get_order_precision(client, symbol):
    try:
        market = client.load_markets()[symbol]
        step_size = float(market['precision']['amount'])
        min_amount = float(market['limits']['amount']['min'])
        return min_amount, step_size
    except Exception as e:
        print(f"❌ 無法取得精度資訊: {e}")
        return 0.01, 0.001

# ✅ 四捨五入到對應精度
def round_step_size(amount, step_size):
    return round(round(amount / step_size) * step_size, 8)

# ✅ 自動交易主程序
def auto_trade_futures(symbol="ETH/USDT", 
                       interval="1m", 
                       usdt_per_order=50, 
                       leverage=5, 
                       strategy=None):

    client = create_binance_futures_client()
    set_leverage(client, symbol, leverage)

    min_amount, step_size = get_order_precision(client, symbol)
    print(f"✅ 最小下單量：{min_amount}, 數量精度：{step_size}")

    interval_sec = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200,
        "4h": 14400, "1d": 86400
    }.get(interval, 60)

    while True:
        try:
            now = datetime.utcnow()
            df = strategy.get_signals(symbol.replace("/", ""), interval, now)
            latest = df.iloc[-1]
            close_price = latest['close']
            signal = latest['signal']
            print(f"[{now:%Y-%m-%d %H:%M:%S}] Close: {close_price:.2f}, Signal: {signal}")

            position_amt, position_side = get_position(client, symbol)
            usdt_balance = get_usdt_balance(client)
            print(f"目前持倉：{position_amt:.6f}（{position_side}）, USDT 餘額：{usdt_balance:.2f}")

            order_amt = (usdt_per_order * leverage) / close_price
            order_amt = max(order_amt, min_amount)
            order_amt = round_step_size(order_amt, step_size)

            # 1️⃣ 平倉階段
            if position_side == 'long' and signal == -1:
                print("📉 平多單中...")
                client.create_order(symbol=symbol, type='market', side='sell', amount=position_amt, params={"reduceOnly": True})
                time.sleep(1)
            elif position_side == 'short' and signal == 1:
                print("📈 平空單中...")
                client.create_order(symbol=symbol, type='market', side='buy', amount=abs(position_amt), params={"reduceOnly": True})
                time.sleep(1)

            # 2️⃣ 更新倉位
            position_amt, position_side = get_position(client, symbol)

            # 3️⃣ 開倉階段
            if signal == 1 and position_side == 'none':
                print(f"✅ 開多單 {order_amt}")
                client.create_order(symbol=symbol, type='market', side='buy', amount=order_amt)
            elif signal == -1 and position_side == 'none':
                print(f"✅ 開空單 {order_amt}")
                client.create_order(symbol=symbol, type='market', side='sell', amount=order_amt)
            else:
                print("⏸ 訊號未變或已有倉位，無操作")

        except Exception as e:
            print(f"❌ 執行錯誤: {e}")

        time.sleep(interval_sec)
