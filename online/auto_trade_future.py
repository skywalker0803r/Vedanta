import ccxt
import os
import time
from datetime import datetime
from dotenv import load_dotenv

# 載入環境變數 (.env 需有 BINANCE_API_KEY_FUTURE, BINANCE_SECRET_FUTURE, BINANCE_TESTNET_MODE)
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
    print(f"✅ 使用 {'Testnet' if testnet else '主網'} 模式")
    return client

# 設定持倉模式（One-Way 或 Hedge）
def set_position_mode(client, hedge_mode=False, symbol="ETH/USDT"):
    try:
        market = client.market(symbol)
        market_id = market['id']
        # marginType 設成 cross，全倉
        client.fapiPrivate_post_margintype({'symbol': market_id, 'marginType': 'CROSS'})
        # positionSide (dualSidePosition) true 為 hedge，false 為 one-way
        client.fapiPrivate_post_positionside_dual({'dualSidePosition': hedge_mode})
        print(f"✅ 持倉模式設為 {'Hedge' if hedge_mode else 'One-Way'}")
    except Exception as e:
        print(f"❌ 設定持倉模式失敗: {e}")

def set_leverage(client, symbol, leverage):
    try:
        client.set_leverage(leverage, symbol)
        print(f"✅ 槓桿設為 {leverage}x")
    except Exception as e:
        print(f"❌ 槓桿設定失敗: {e}")

# 查持倉，回傳 (數量絕對值, 'long'/'short'/'none')
def get_position(client, symbol):
    try:
        positions = client.fapiPrivateGetPositionRisk()
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
    return round(round(amount / step_size) * step_size, 8)

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

def auto_trade_futures(symbol="ETH/USDT", interval="1m", usdt_per_order=50, leverage=5, strategy=None, max_retries=3):

    client = create_binance_futures_client()
    set_position_mode(client, hedge_mode=False, symbol=symbol)  # One-Way 模式
    set_leverage(client, symbol, leverage)

    min_amount, step_size = get_order_precision(client, symbol)
    print(f"✅ 最小下單量: {min_amount}, 數量精度: {step_size}")

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
            print(f"目前持倉: {position_amt:.6f} ({position_side}), USDT 餘額: {usdt_balance:.2f}")

            order_amt = (usdt_per_order * leverage) / close_price
            order_amt = max(order_amt, min_amount)
            order_amt = round_step_size(order_amt, step_size)

            # 平倉判斷
            if position_side == 'long' and signal == -1:
                print("📉 平多單中...")
                close_amt = round_step_size(position_amt, step_size)
                if close_amt >= min_amount:
                    for i in range(max_retries):
                        try:
                            # 重新取得持倉防止異常
                            pos_amt, pos_side = get_position(client, symbol)
                            if pos_side != 'long' or pos_amt == 0:
                                print("⚠️ 無多單可平，跳過")
                                break
                            client.create_order(symbol=symbol, type='market', side='sell', amount=close_amt, params={"reduceOnly": True})
                            print(f"✅ 平多單成功: {close_amt}")
                            time.sleep(1)
                            break
                        except Exception as e:
                            print(f"❌ 平多單失敗 (嘗試 {i+1}/{max_retries}): {e}")
                            time.sleep(2)
                    else:
                        print("⛔ 平多單達最大重試，嘗試關閉持倉")
                        close_all_positions(client, symbol)

            elif position_side == 'short' and signal == 1:
                print("📈 平空單中...")
                close_amt = round_step_size(position_amt, step_size)
                if close_amt >= min_amount:
                    for i in range(max_retries):
                        try:
                            pos_amt, pos_side = get_position(client, symbol)
                            if pos_side != 'short' or pos_amt == 0:
                                print("⚠️ 無空單可平，跳過")
                                break
                            client.create_order(symbol=symbol, type='market', side='buy', amount=close_amt, params={"reduceOnly": True})
                            print(f"✅ 平空單成功: {close_amt}")
                            time.sleep(1)
                            break
                        except Exception as e:
                            print(f"❌ 平空單失敗 (嘗試 {i+1}/{max_retries}): {e}")
                            time.sleep(2)
                    else:
                        print("⛔ 平空單達最大重試，嘗試關閉持倉")
                        close_all_positions(client, symbol)

            # 更新倉位狀態
            time.sleep(1)
            position_amt, position_side = get_position(client, symbol)

            # 開倉判斷
            if signal == 1 and position_side == 'none':
                print(f"✅ 開多單 {order_amt}")
                try:
                    client.create_order(symbol=symbol, type='market', side='buy', amount=order_amt)
                    print(f"✅ 開多單成功: {order_amt}")
                    time.sleep(1)
                except Exception as e:
                    print(f"❌ 開多單失敗: {e}")
            elif signal == -1 and position_side == 'none':
                print(f"✅ 開空單 {order_amt}")
                try:
                    client.create_order(symbol=symbol, type='market', side='sell', amount=order_amt)
                    print(f"✅ 開空單成功: {order_amt}")
                    time.sleep(1)
                except Exception as e:
                    print(f"❌ 開空單失敗: {e}")
            else:
                print("⏸ 訊號未變或已有倉位，無操作")

        except Exception as e:
            print(f"❌ 執行錯誤: {e}")

        time.sleep(interval_sec)


if __name__ == "__main__":
    # 請自行準備 Technicalindicatorstrategy 模組與 testsma 策略
    from Technicalindicatorstrategy import testsma

    auto_trade_futures(
        symbol="ETH/USDT",
        interval="1m",
        usdt_per_order=500,
        leverage=5,
        strategy=testsma
    )
