import ccxt
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
load_dotenv()

# 創立幣安客端
def create_binance_futures_client():
    testnet = os.getenv("BINANCE_TESTNET_MODE", "True") == "True"
    client = ccxt.binance({
        'apiKey': os.getenv("BINANCE_API_KEY_FUTURE"),
        'secret': os.getenv("BINANCE_SECRET_FUTURE"),
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })
    client.set_sandbox_mode(testnet)
    client.load_markets()
    print(f"🛠️ 連線設定完成，使用 {'🧪 測試網路' if testnet else '🚀 主網路'} 模式")
    return client

# 設定槓桿
def set_leverage(client, symbol, leverage):
    try:
        client.set_leverage(leverage, symbol)
        print(f"⚙️ 槓桿已設定為 {leverage}x")
    except Exception as e:
        print(f"❗ 槓桿設定失敗: {e}")

# 取得倉位資訊
def get_position(client, symbol):
    try:
        positions = client.fetch_positions([symbol], params={"type": "future"})
        for pos in positions:
            amt = float(pos.get('contracts', 0))
            if amt == 0:
                print("📭 無持倉")
                return 0.0, 'none', None, None

            side_raw = pos.get('side')
            if not side_raw:
                print("⚠️ 讀取持倉方向失敗（ccxt版本問題？）")
                return amt, 'unknown', None, None

            side = side_raw.lower()
            entry_price = float(pos['entryPrice']) if pos.get('entryPrice') else None
            timestamp = pos.get('timestamp')
            print(f"📊 持倉偵測: {amt} 張，方向: {side}，入場價: {entry_price}")
            return amt, side, entry_price, timestamp

        print("📭 無持倉")
        return 0.0, 'none', None, None
    except Exception as e:
        print(f"❌ 讀取持倉錯誤: {e}")
        return 0.0, 'none', None, None

# 取得餘額
def get_usdt_balance(client):
    try:
        balance = client.fetch_balance()['USDT']['free']
        print(f"💰 可用餘額: {balance:.2f} USDT")
        return balance
    except Exception as e:
        print(f"❌ 查詢餘額錯誤: {e}")
        return 0

# 取得精度資訊
def get_order_precision(client, symbol):
    try:
        market = client.load_markets()[symbol]
        step_size = float(market['precision']['amount'])
        min_amount = float(market['limits']['amount']['min'])
        print(f"📐 交易精度: 最小數量 {min_amount}, 單位步長 {step_size}")
        return min_amount, step_size
    except Exception as e:
        print(f"❌ 取得交易精度失敗: {e}")
        return 0.01, 0.001

# 計算step_size
def round_step_size(amount, step_size):
    import math
    rounded = math.floor(amount / step_size) * step_size
    print(f"🔢 數量經過精度對齊: 原始 {amount} → 對齊後 {rounded}")
    return rounded

# 關閉所有持倉
def close_all_positions(client, symbol):
    amt, side, _, _ = get_position(client, symbol)
    if amt == 0:
        print("✅ 無持倉，無需平倉")
        return
    order_side = 'sell' if side == 'long' else 'buy'
    print(f"🔒 關閉持倉中: {amt} 張 {side}，下 {order_side} 市價單...")
    try:
        client.create_order(
            symbol=symbol,
            type='market',
            side=order_side,
            amount=amt,
        )
        print(f"✅ 已成功關閉 {symbol} 持倉")
    except Exception as e:
        print(f"❌ 關閉持倉失敗: {e}")
    time.sleep(1)

# 取消所有掛單
def cancel_all_open_orders(client, symbol):
    try:
        client.cancel_all_orders(symbol)
        print(f"🗑️ 已取消 {symbol} 的所有掛單")
    except Exception as e:
        print(f"❌ 取消掛單失敗: {e}")

# 向下對齊到最近的 interval 開始
def align_to_interval(dt, interval_sec):
    ts = int(dt.timestamp())
    aligned_ts = ts - (ts % interval_sec)
    return datetime.utcfromtimestamp(aligned_ts)

# 主程序
def auto_trade_futures(symbol="ETH/USDT", interval="1h",
                       usdt_percent_per_order=0.1,  # 每次用餘額的百分比（0.1=10%）
                       leverage=5, strategy=None,
                       run_once=True,
                       stop_loss=None, take_profit=None,
                       max_hold_bars=1000):

    client = create_binance_futures_client()
    set_leverage(client, symbol, leverage)
    min_amount, step_size = get_order_precision(client, symbol)

    interval_sec = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200,
        "4h": 14400, "1d": 86400
    }.get(interval, 60)

    def process_once():
        try:
            print(f"\n🔔 【策略執行】時間: {datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC")
            print(f"🧠 使用策略: {strategy.__class__.__name__}，交易標的: {symbol}")

            now = datetime.utcnow()
            df = strategy.get_signals(symbol.replace("/", ""), interval, now)
            latest = df.iloc[-1]
            close_price = latest['close']
            signal = latest['signal']
            
            # 如果沒設止損則看策略本身是否帶止損 有的話就用策略止損替換None
            if stop_loss == None:
                if pd.notna(latest['stop_loss']):
                    stop_loss = latest['stop_loss']
                else:
                    stop_loss = None
                    print('沒設止損 策略也沒有止損')
            
            print(f"📈 最新收盤價: {close_price:.2f}, 訊號: {signal}")

            position_amt, position_side, entry_price, entry_time = get_position(client, symbol)
            usdt_balance = get_usdt_balance(client)
            human_time = datetime.utcfromtimestamp(entry_time / 1000).strftime("%Y-%m-%d %H:%M:%S") if entry_time else "N/A"
            print(f"💼 持倉狀況: {position_amt:.6f} ({position_side})，入場價: {entry_price}，入場時間: {human_time} UTC")

            usdt_per_order = usdt_balance * usdt_percent_per_order
            order_amt = (usdt_per_order * leverage) / close_price
            order_amt = max(order_amt, min_amount)
            order_amt = round_step_size(order_amt, step_size)

            if entry_time:
                entry_time_dt = datetime.utcfromtimestamp(entry_time / 1000)
                aligned_entry_time = align_to_interval(entry_time_dt, interval_sec)
                filtered_df = df[df['timestamp'] <= aligned_entry_time]
                if not filtered_df.empty:
                    entry_index = df.index.get_loc(filtered_df.iloc[-1].name)
                    current_index = len(df) - 1
                    held_bars = current_index - entry_index
                    print(f"⏳ 持倉時間: {held_bars} 根 K 棒, 最大允許: {max_hold_bars}")
                    if held_bars >= max_hold_bars:
                        print(f"⏰ 超過最大持有K棒數({held_bars}/{max_hold_bars})，執行強制平倉")
                        close_all_positions(client, symbol)

            # 多單持倉但訊號做空，平多單(單純平倉不進場)
            if position_side == 'long' and signal == -1:
                print("🔻 訊號切換做空，準備平多單...")
                close_all_positions(client, symbol)
                position_amt, position_side, entry_price, entry_time = get_position(client, symbol)
                print(f"♻️ 持倉更新: {position_amt:.6f} ({position_side})")

            # 空單持倉但訊號做多，平空單(單純平倉不進場)
            elif position_side == 'short' and signal == 1:
                print("🔺 訊號切換做多，準備平空單...")
                close_all_positions(client, symbol)
                position_amt, position_side, entry_price, entry_time = get_position(client, symbol)
                print(f"♻️ 持倉更新: {position_amt:.6f} ({position_side})")

            # 無持倉且訊號做多，開多單並設定止損止盈
            if signal == 1 and position_side == 'none':
                print(f"🚀 開多單 {order_amt} 張")
                cancel_all_open_orders(client, symbol)
                order = client.create_order(symbol=symbol, type='market', side='buy', amount=order_amt)
                entry_price = float(order.get('average'))

                if stop_loss is not None and take_profit is not None:
                    trigger_sl = entry_price * (1 - stop_loss)
                    trigger_tp = entry_price * (1 + take_profit)
                    client.create_order(symbol=symbol, type='stop_market', side='sell', amount=order_amt,
                                        params={"stopPrice": trigger_sl, "reduceOnly": True, "priceProtect": True})
                    client.create_order(symbol=symbol, type='take_profit_market', side='sell', amount=order_amt,
                                        params={"stopPrice": trigger_tp, "reduceOnly": True, "priceProtect": True})
                    print(f"✅ 多單建立完成，入場價: {entry_price:.4f}，止損: {trigger_sl:.4f}，止盈: {trigger_tp:.4f}")
                else:
                    print(f"✅ 多單建立完成（無止損止盈），入場價: {entry_price:.4f}")

            # 無持倉且訊號做空，開空單並設定止損止盈
            elif signal == -1 and position_side == 'none':
                print(f"🛑 開空單 {order_amt} 張")
                cancel_all_open_orders(client, symbol)
                order = client.create_order(symbol=symbol, type='market', side='sell', amount=order_amt)
                entry_price = float(order.get('average'))

                if stop_loss is not None and take_profit is not None:
                    trigger_sl = entry_price * (1 + stop_loss)
                    trigger_tp = entry_price * (1 - take_profit)
                    client.create_order(symbol=symbol, type='stop_market', side='buy', amount=order_amt,
                                        params={"stopPrice": trigger_sl, "reduceOnly": True, "priceProtect": True})
                    client.create_order(symbol=symbol, type='take_profit_market', side='buy', amount=order_amt,
                                        params={"stopPrice": trigger_tp, "reduceOnly": True, "priceProtect": True})
                    print(f"✅ 空單建立完成，入場價: {entry_price:.4f}，止損: {trigger_sl:.4f}，止盈: {trigger_tp:.4f}")
                else:
                    print(f"✅ 空單建立完成（無止損止盈），入場價: {entry_price:.4f}")

        except Exception as e:
            print(f"❌ 執行錯誤: {e}")

    if run_once:
        process_once()
    else:
        while True:
            process_once()
            print(f"⏳ 等待下一次執行（{interval_sec}秒）...\n")
            time.sleep(interval_sec)