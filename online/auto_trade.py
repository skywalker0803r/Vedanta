import ccxt
import time
import datetime
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

# ✅ 取得該交易對最小下單數量
def get_min_trade_amount(client, symbol):
    markets = client.load_markets()
    return markets[symbol]["limits"]["amount"]["min"]

# ✅ 自動交易主程序
def auto_trade(symbol="ETH/USDT", interval="1m", usdt_per_order=50, strategy=None):
    client = create_binance_client()
    min_amount = get_min_trade_amount(client, symbol)
    print(f"✅ {symbol} 最小下單量為 {min_amount}")

    last_position = 0  # -1: 空單, 0: 無單, 1: 多單

    interval_sec = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200,
        "4h": 14400, "1d": 86400
    }[interval]

    while True:
        try:
            now = datetime.datetime.utcnow()
            df = strategy.get_signals(symbol.replace("/", ""), interval, now)
            latest = df.iloc[-1]
            close = latest["close"]
            signal = latest["signal"]
            print(f"[{now:%Y-%m-%d %H:%M:%S}] Close: {close:.2f}, Signal: {signal}")

            # ✅ 多單信號：買入
            if signal == 1 and last_position <= 0:
                amount = usdt_per_order / close
                if amount >= min_amount:
                    print(f"🟢 黃金交叉 → 市價買入 {amount:.6f} {symbol}")
                    client.create_market_buy_order(symbol, amount)
                    last_position = 1
                else:
                    print(f"⚠️ 買入失敗，數量 {amount:.6f} 小於最小下單量 {min_amount}")

            # ✅ 空單信號：賣出
            elif signal == -1 and last_position >= 0:
                coin = symbol.split("/")[0]
                amount = client.fetch_balance()[coin]["free"]
                if amount >= min_amount:
                    print(f"🔴 死亡交叉 → 市價賣出 {amount:.6f} {coin}")
                    client.create_market_sell_order(symbol, amount)
                    last_position = -1
                else:
                    print(f"⚠️ 賣出失敗，數量 {amount:.6f} 小於最小下單量 {min_amount}")

            else:
                print("⏸ 無操作")

            # ✅ 顯示餘額
            balance = client.fetch_balance()
            coin = symbol.split("/")[0]
            print(f"{coin} 餘額：{balance['total'].get(coin, 0)}")
            print(f"USDT 餘額：{balance['total'].get('USDT', 0)}")

        except Exception as e:
            print(f"❌ 發生錯誤：{e}")

        time.sleep(interval_sec)
