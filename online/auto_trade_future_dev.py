import ccxt
import os
import time
from datetime import datetime, timezone
from dotenv import load_dotenv
import pandas as pd
import sys
import io
import requests
import math

load_dotenv()

# Telegram config
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        return
    apiURL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage'
    try:
        requests.post(apiURL, json={'chat_id': TELEGRAM_CHAT_ID, 'text': message, 'parse_mode': 'Markdown'})
    except Exception as e:
        print(f"Error sending Telegram message: {e}")

class TelegramStream(io.StringIO):
    def __init__(self, original_stdout):
        super().__init__()
        self.original_stdout = original_stdout
        self.buffer = []

    def write(self, s):
        super().write(s)
        self.original_stdout.write(s)
        self.buffer.append(s)

    def flush(self):
        super().flush()
        self.original_stdout.flush()
        message = "".join(self.buffer).strip()
        if message:
            send_telegram_message(message)
        self.buffer = []

# Binance futures client with safe headers
def create_binance_futures_client():
    testnet = os.getenv("BINANCE_TESTNET_MODE", "True") == "True"
    client = ccxt.binance({
        'apiKey': os.getenv("BINANCE_API_KEY_FUTURE"),
        'secret': os.getenv("BINANCE_SECRET_FUTURE"),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
        'headers': {'User-Agent': 'Mozilla/5.0'}
    })
    client.set_sandbox_mode(testnet)
    client.load_markets()
    return client

# 設定槓桿
def set_leverage(client, symbol, leverage):
    try:
        client.set_leverage(leverage, symbol)
    except Exception as e:
        print(f"❗ 槓桿設定失敗: {e}")

# 取得倉位資訊
def get_position(client, symbol):
    try:
        positions = client.fetch_positions([symbol], params={"type": "future"})
        for pos in positions:
            amt = float(pos.get('contracts', 0))
            if amt == 0:
                return 0.0, 'none', None, None
            side_raw = pos.get('side')
            side = side_raw.lower() if side_raw else 'unknown'
            entry_price = float(pos['entryPrice']) if pos.get('entryPrice') else None
            timestamp = pos.get('timestamp')
            return amt, side, entry_price, timestamp
        return 0.0, 'none', None, None
    except Exception as e:
        print(f"❌ 讀取持倉錯誤: {e}")
        return 0.0, 'none', None, None

# 取得 USDT 可用餘額
def get_usdt_balance(client):
    try:
        return client.fetch_balance()['USDT']['free']
    except Exception as e:
        print(f"❌ 查詢餘額錯誤: {e}")
        return 0

# 取得精度資訊
def get_order_precision(client, symbol):
    try:
        market = client.load_markets()[symbol]
        step_size = float(market['precision']['amount'])
        min_amount = float(market['limits']['amount']['min'])
        return min_amount, step_size
    except Exception as e:
        print(f"❌ 取得交易精度失敗: {e}")
        return 0.01, 0.001

def round_step_size(amount, step_size):
    return math.floor(amount / step_size) * step_size

# 關閉所有持倉
def close_all_positions(client, symbol):
    amt, side, _, _ = get_position(client, symbol)
    if amt == 0:
        print("✅ 無持倉")
        return
    order_side = 'sell' if side == 'long' else 'buy'
    try:
        client.create_order(symbol=symbol, type='market', side=order_side, amount=amt)
        print(f"✅ 已成功關閉 {symbol} 持倉")
    except Exception as e:
        print(f"❌ 關閉持倉失敗: {e}")
    time.sleep(0.5)

# 取消所有掛單
def cancel_all_open_orders(client, symbol):
    try:
        client.cancel_all_orders(symbol)
    except Exception as e:
        print(f"❌ 取消掛單失敗: {e}")

# 向下對齊到最近 interval
def align_to_interval(dt, interval_sec):
    ts = int(dt.timestamp())
    aligned_ts = ts - (ts % interval_sec)
    return datetime.fromtimestamp(aligned_ts, tz=timezone.utc)

# 安全抓取 K 線
def fetch_klines_safe(client, symbol, interval, limit=100):
    try:
        now_ms = int(time.time() * 1000)
        data = client.fetch_ohlcv(symbol, timeframe=interval, limit=limit, params={"endTime": now_ms})
        df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        print(f"❌ K線抓取失敗: {e}")
        time.sleep(0.5)
        return None

# 主程序
def auto_trade_safe(symbol="ETH/USDT", interval="1h",
                     usdt_percent_per_order=0.1, leverage=5,
                     strategy=None, run_once=True,
                     stop_loss=None, take_profit=None,
                     max_hold_bars=1000):
    original_stdout = sys.stdout
    sys.stdout = TelegramStream(original_stdout)

    try:
        client = create_binance_futures_client()
        set_leverage(client, symbol, leverage)
        min_amount, step_size = get_order_precision(client, symbol)

        interval_sec = {
            "1m": 60, "3m": 180, "5m": 300,
            "15m": 900, "30m": 1800, "1h": 3600,
            "2h": 7200, "4h": 14400, "1d": 86400
        }.get(interval, 60)

        def process_once():
            try:
                now = datetime.now(timezone.utc)
                print(f"🧠 使用策略: {os.path.basename(strategy.__file__)}，交易標的: {symbol}")

                df = strategy.get_signals(symbol.replace("/", ""), interval, now)
                latest = df.iloc[-1]
                close_price = latest['close']
                signal = latest['signal']
                print(f"📈 最新收盤價: {close_price:.2f}, 訊號: {signal}")

                position_amt, position_side, entry_price, entry_time = get_position(client, symbol)
                usdt_balance = get_usdt_balance(client)
                human_time = datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S") if entry_time else "N/A"
                print(f"💼 持倉狀況: {position_amt:.6f} ({position_side})，入場價: {entry_price}, 入場時間: {human_time} UTC")

                usdt_per_order = usdt_balance * usdt_percent_per_order
                order_amt = max(round_step_size((usdt_per_order * leverage)/close_price, step_size), min_amount)

                # 持倉時間檢查
                if entry_time:
                    entry_time_dt = datetime.fromtimestamp(entry_time / 1000, tz=timezone.utc)
                    aligned_entry_time = align_to_interval(entry_time_dt, interval_sec)
                    filtered_df = df[df['timestamp'] <= aligned_entry_time]
                    if not filtered_df.empty:
                        entry_index = df.index.get_loc(filtered_df.iloc[-1].name)
                        held_bars = len(df)-1 - entry_index
                        print(f"⏳ 持倉時間: {held_bars} 根 K 棒, 最大允許: {max_hold_bars}")
                        if held_bars >= max_hold_bars:
                            print("⏰ 超過最大持有K棒數，執行強制平倉")
                            close_all_positions(client, symbol)

                # 多空訊號切換 & 開倉邏輯
                if position_side == 'long' and signal == -1:
                    print("🔻 訊號切換做空，平多單...")
                    close_all_positions(client, symbol)
                elif position_side == 'short' and signal == 1:
                    print("🔺 訊號切換做多，平空單...")
                    close_all_positions(client, symbol)

                if signal == 1 and position_side == 'none':
                    print(f"🚀 開多單 {order_amt} 張")
                    cancel_all_open_orders(client, symbol)
                    order = client.create_order(symbol=symbol, type='market', side='buy', amount=order_amt)
                    entry_price = float(order.get('average'))
                    if stop_loss is not None and take_profit is not None:
                        trigger_sl = entry_price*(1-stop_loss)
                        trigger_tp = entry_price*(1+take_profit)
                        client.create_order(symbol=symbol, type='stop_market', side='sell', amount=order_amt,
                                            params={"stopPrice": trigger_sl, "reduceOnly": True, "priceProtect": True})
                        client.create_order(symbol=symbol, type='take_profit_market', side='sell', amount=order_amt,
                                            params={"stopPrice": trigger_tp, "reduceOnly": True, "priceProtect": True})
                        print(f"✅ 多單建立完成，入場價: {entry_price:.4f}，止損: {trigger_sl:.4f}，止盈: {trigger_tp:.4f}")
                    else:
                        print(f"✅ 多單建立完成（無止損止盈），入場價: {entry_price:.4f}")

                elif signal == -1 and position_side == 'none':
                    print(f"🛑 開空單 {order_amt} 張")
                    cancel_all_open_orders(client, symbol)
                    order = client.create_order(symbol=symbol, type='market', side='sell', amount=order_amt)
                    entry_price = float(order.get('average'))
                    if stop_loss is not None and take_profit is not None:
                        trigger_sl = entry_price*(1+stop_loss)
                        trigger_tp = entry_price*(1-take_profit)
                        client.create_order(symbol=symbol, type='stop_market', side='buy', amount=order_amt,
                                            params={"stopPrice": trigger_sl, "reduceOnly": True, "priceProtect": True})
                        client.create_order(symbol=symbol, type='take_profit_market', side='buy', amount=order_amt,
                                            params={"stopPrice": trigger_tp, "reduceOnly": True, "priceProtect": True})
                        print(f"✅ 空單建立完成，入場價: {entry_price:.4f}，止損: {trigger_sl:.4f}，止盈: {trigger_tp:.4f}")
                    else:
                        print(f"✅ 空單建立完成（無止損止盈），入場價: {entry_price:.4f}")

            except Exception as e:
                print(f"❌ 執行錯誤: {e}")
            finally:
                sys.stdout.flush()
                time.sleep(0.5)  # 幣種間延遲，降低 418 風險

        if run_once:
            process_once()
        else:
            while True:
                process_once()
                print(f"⏳ 等待下一次執行 ({interval_sec} 秒)...")
                time.sleep(interval_sec)

    finally:
        sys.stdout = original_stdout
