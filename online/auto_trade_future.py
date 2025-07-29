import ccxt
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

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
    print(f"✅ 使用 {'Testnet' if testnet else '主網'} 模式")
    return client

def set_leverage(client, symbol, leverage):
    try:
        client.set_leverage(leverage, symbol)
        print(f"✅ 槓桿設為 {leverage}x")
    except Exception as e:
        print(f"❌ 槓桿設定失敗: {e}")

def get_position(client, symbol):
    try:
        balance_info = client.fetch_balance()
        positions = balance_info.get('info', {}).get('positions', [])
        symbol_id = client.market(symbol)['id']
        for pos in positions:
            if pos['symbol'] == symbol_id:
                amt = float(pos['positionAmt'])
                side = 'long' if amt > 0 else 'short' if amt < 0 else 'none'
                return abs(amt), side
        return 0.0, 'none'
    except Exception as e:
        print(f"❌ 查持倉錯誤: {e}")
        return 0, 'none'

def get_usdt_balance(client):
    try:
        return client.fetch_balance()['USDT']['free']
    except Exception as e:
        print(f"❌ 查餘額錯誤: {e}")
        return 0

def get_order_precision(client, symbol):
    try:
        market = client.load_markets()[symbol]
        step_size = float(market['precision']['amount'])
        min_amount = float(market['limits']['amount']['min'])
        return min_amount, step_size
    except Exception as e:
        print(f"❌ 無法取得精度資訊: {e}")
        return 0.01, 0.001

def round_step_size(amount, step_size):
    # 向下取整數量，避免超出交易所限制
    import math
    return math.floor(amount / step_size) * step_size

def close_all_positions(client, symbol):
    try:
        amt, side = get_position(client, symbol)
        if amt == 0:
            print("✅ 無持倉需平倉")
            return
        order_side = 'sell' if side == 'long' else 'buy'
        print(f"嘗試關閉持倉: {amt} {side}，下 {order_side} 市價單")
        client.create_order(symbol=symbol, type='market', side=order_side, amount=amt, params={"reduceOnly": True})
        print(f"✅ 成功關閉所有 {symbol} 持倉")
        time.sleep(1)
    except Exception as e:
        print(f"❌ 關閉所有持倉失敗: {e}")

def cancel_all_open_orders(client, symbol):
    try:
        client.cancel_all_orders(symbol)
        print(f"🧹 已取消 {symbol} 所有掛單")
    except Exception as e:
        print(f"⚠️ 取消掛單失敗: {e}")

def auto_trade_futures(symbol="ETH/USDT", interval="1h",
                       usdt_percent_per_order=0.1,  # 每次用餘額的百分比（0.1=10%）
                       leverage=5, strategy=None,
                       max_retries=3, run_once=True,
                       stop_loss=0.005, take_profit=0.05,
                       max_hold_bars=1000):

    trigger_price_buffer = 0.005  # 0.5%

    client = create_binance_futures_client()
    set_leverage(client, symbol, leverage)
    min_amount, step_size = get_order_precision(client, symbol)
    print(f"✅ 最小下單量: {min_amount}, 數量精度: {step_size}")

    interval_sec = {
        "1m": 60, "3m": 180, "5m": 300, "15m": 900,
        "30m": 1800, "1h": 3600, "2h": 7200,
        "4h": 14400, "1d": 86400
    }.get(interval, 60)

    hold_info = {'entry_index': None, 'entry_price': None}

    def process_once():
        try:
            print(f"📊 正在使用策略: {strategy.__class__.__name__}，交易標的: {symbol}")
            now = datetime.utcnow()
            df = strategy.get_signals(symbol.replace("/", ""), interval, now)
            latest = df.iloc[-1]
            close_price = latest['close']
            signal = latest['signal']
            print(f"[{now:%Y-%m-%d %H:%M:%S}] Close: {close_price:.2f}, Signal: {signal}")

            position_amt, position_side = get_position(client, symbol)
            usdt_balance = get_usdt_balance(client)
            print(f"目前持倉: {position_amt:.6f} ({position_side}), USDT 餘額: {usdt_balance:.2f}")

            # 根據百分比計算每次開倉的USDT金額
            usdt_per_order = usdt_balance * usdt_percent_per_order

            order_amt = (usdt_per_order * leverage) / close_price
            order_amt = max(order_amt, min_amount)
            order_amt = round_step_size(order_amt, step_size)
            
            if position_side != 'none' and hold_info['entry_index'] is not None:
                current_index = len(df) - 1
                held_bars = current_index - hold_info['entry_index']
                if held_bars >= max_hold_bars:
                    print(f"⏰ 超過最大持有K棒數({held_bars}/{max_hold_bars})，平倉")
                    close_all_positions(client, symbol)
                    hold_info['entry_index'] = None
                    hold_info['entry_price'] = None
                    return

            if position_side == 'long' and signal == -1:
                print("📉 平多單中...")
                close_all_positions(client, symbol)
                hold_info['entry_index'] = None
                hold_info['entry_price'] = None

            elif position_side == 'short' and signal == 1:
                print("📈 平空單中...")
                close_all_positions(client, symbol)
                hold_info['entry_index'] = None
                hold_info['entry_price'] = None

            time.sleep(1)
            position_amt, position_side = get_position(client, symbol)

            ticker = client.fetch_ticker(symbol)
            last_price = ticker['last']
            min_diff_ratio = 0.005  # 0.5% 安全距離

            if signal == 1 and position_side == 'none':
                print(f"✅ 開多單 {order_amt}")
                try:
                    cancel_all_open_orders(client, symbol)
                    client.create_order(symbol=symbol, type='market', side='buy', amount=order_amt)
                    entry_price = close_price
                    hold_info['entry_price'] = entry_price
                    hold_info['entry_index'] = len(df) - 1

                    sl = entry_price * (1 - stop_loss)
                    tp = entry_price * (1 + take_profit)
                    trigger_sl = sl
                    trigger_tp = tp

                    if trigger_tp <= last_price or abs(trigger_tp - last_price) / last_price < min_diff_ratio:
                        trigger_tp = last_price * (1 + min_diff_ratio)
                    if trigger_sl >= last_price or abs(trigger_sl - last_price) / last_price < min_diff_ratio:
                        trigger_sl = last_price * (1 - min_diff_ratio)

                    # 取得價格精度，這邊以2位小數為例，實務可改為動態取得
                    trigger_sl = round(trigger_sl, 2)
                    trigger_tp = round(trigger_tp, 2)

                    retries = 0
                    while retries < max_retries:
                        try:
                            client.create_order(symbol=symbol, type='stop_market', side='sell', amount=order_amt,
                                                params={"stopPrice": trigger_sl, "reduceOnly": True, "priceProtect": True})
                            client.create_order(symbol=symbol, type='take_profit_market', side='sell', amount=order_amt,
                                                params={"stopPrice": trigger_tp, "reduceOnly": True, "priceProtect": True})
                            print(f"✅ 多單建立完成，止損: {trigger_sl}, 止盈: {trigger_tp}")
                            break
                        except Exception as e:
                            print(f"⚠️ 掛單失敗，嘗試第 {retries + 1} 次: {e}")
                            retries += 1
                            time.sleep(1)
                    if retries >= max_retries:
                        print("❌ 多單掛單最終失敗，建議檢查市價與觸發價距離")
                except Exception as e:
                    print(f"❌ 開多單失敗: {e}")

            elif signal == -1 and position_side == 'none':
                print(f"✅ 開空單 {order_amt}")
                try:
                    cancel_all_open_orders(client, symbol)
                    client.create_order(symbol=symbol, type='market', side='sell', amount=order_amt)
                    entry_price = close_price
                    hold_info['entry_price'] = entry_price
                    hold_info['entry_index'] = len(df) - 1

                    sl = entry_price * (1 + stop_loss)
                    tp = entry_price * (1 - take_profit)
                    trigger_sl = sl
                    trigger_tp = tp

                    if trigger_sl <= last_price or abs(trigger_sl - last_price) / last_price < min_diff_ratio:
                        trigger_sl = last_price * (1 + min_diff_ratio)
                    if trigger_tp >= last_price or abs(trigger_tp - last_price) / last_price < min_diff_ratio:
                        trigger_tp = last_price * (1 - min_diff_ratio)

                    trigger_sl = round(trigger_sl, 2)
                    trigger_tp = round(trigger_tp, 2)

                    client.create_order(symbol=symbol, type='stop_market', side='buy', amount=order_amt,
                                        params={"stopPrice": trigger_sl, "reduceOnly": True, "priceProtect": True})
                    client.create_order(symbol=symbol, type='take_profit_market', side='buy', amount=order_amt,
                                        params={"stopPrice": trigger_tp, "reduceOnly": True, "priceProtect": True})
                    print(f"✅ 空單建立完成，止損: {trigger_sl}, 止盈: {trigger_tp}")
                except Exception as e:
                    print(f"❌ 開空單失敗: {e}")
            else:
                print("⏸ 無開倉條件或已有持倉")

        except Exception as e:
            print(f"❌ 執行錯誤: {e}")

    if run_once:
        process_once()
    else:
        while True:
            process_once()
            time.sleep(interval_sec)
