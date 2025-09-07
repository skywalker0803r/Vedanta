import ccxt
import time
import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
import os

# 載入 .env 中的環境變數
load_dotenv()

# ✅ 建立 Binance 客戶端（Testnet 或主網現貨）
def create_binance_client():
    api_key = os.getenv('BINANCE_API_KEY')
    secret = os.getenv('BINANCE_SECRET')
    testnet_mode = os.getenv("BINANCE_TESTNET_MODE", "True") == "True"

    client = ccxt.binance({
        'apiKey': api_key,
        'secret': secret,
        'enableRateLimit': True,
        'options': {'defaultType': 'spot'}
    })

    client.set_sandbox_mode(testnet_mode)
    print(f"✅ 已啟用 {'Testnet' if testnet_mode else '主網'} 模式")
    return client

def get_binance_latest_price(symbol: str) :
    """
    從幣安 API 獲取 成交價格。
    """
    base_url = "https://api.binance.com/api/v3/trades"

    params = {
        "symbol": symbol.upper(),
        "limit": 1
    }
    try:
        response = requests.get(base_url, params=params, timeout=100)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching data: {e}")

    data = response.json()


    if not data:
        print("No more data returned from API.")
    return data[0]['price']


# ✅ 取得該交易對最小下單數量
def get_min_trade_amount(client, symbol):
    markets = client.load_markets()
    return markets[symbol]["limits"]["amount"]["min"]

# ✅ 自動交易主程序
def auto_trade(symbol="ETHUSDT", interval="1m", usdt_per_order=50, strategy=None, run_once=True):
    client = create_binance_client()
    min_amount = get_min_trade_amount(client, symbol)
    print(f"✅ {symbol} 最小下單量為 {min_amount}")

    interval_sec = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200,
        "4h": 14400, "1d": 86400
    }[interval]

    def process_once():
        try:
            # 每次重新從帳戶餘額判斷目前持倉狀態
            balance = client.fetch_balance()
            coin = symbol.split("/")[0]
            free_coin = balance['free'].get(coin, 0)
            # 判斷是否持有多單（只要大於最小下單量就視為有持倉）
            last_position = 1 if free_coin >= min_amount else 0

            now = datetime.now(timezone.utc)
            df = strategy.get_signals(symbol.replace("/", ""), interval, now)
            latest = df.iloc[-1]
            close = latest["close"]
            signal = latest["signal"]
            print(f"[{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}] Close: {close:.2f}, Signal: {signal}")

            # 多單信號且目前無多單，則買入
            if signal == 1 and last_position == 0:
                amount = usdt_per_order / close
                if amount >= min_amount:
                    print(f"🟢 黃金交叉 → 市價買入 {amount:.6f} {symbol}")
                    # client.create_market_buy_order(symbol, amount)
                    # 使用實盤成交價格，在並用限價單送出
                    now_price = get_binance_latest_price(symbol)
                    client.createOrder(symbol, type = 'limit',amount = amount,side='buy',price = now_price)
                else:
                    print(f"⚠️ 買入失敗，數量 {amount:.6f} 小於最小下單量 {min_amount}")

            # 空單信號且目前有多單，則賣出
            elif signal == -1 and last_position == 1:
                amount = free_coin
                if amount >= min_amount:
                    print(f"🔴 死亡交叉 → 市價賣出 {amount:.6f} {coin}")
                    # client.create_market_sell_order(symbol, amount)
                    # 使用實盤成交價格，在並用限價單送出
                    now_price = get_binance_latest_price(symbol)
                    client.createOrder(symbol, type = 'limit',amount = amount,side='sell',price = now_price)
                else:
                    print(f"⚠️ 賣出失敗，數量 {amount:.6f} 小於最小下單量 {min_amount}")

            else:
                print("⏸ 無操作")

            # 顯示餘額
            balance = client.fetch_balance()
            print(f"{coin} 餘額：{balance['total'].get(coin, 0)}")
            print(f"USDT 餘額：{balance['total'].get('USDT', 0)}")

        except Exception as e:
            print(f"❌ 發生錯誤：{e}")

    if run_once:
        process_once()
    else:
        while True:
            process_once()
            time.sleep(interval_sec)
