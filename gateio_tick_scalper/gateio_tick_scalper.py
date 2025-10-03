"""
Gate.io 現貨：自動刷交易量機器人
- 需求：設定目標交易量、每次投入的本金、交易對
- 邏輯：不斷重複買入-賣出循環，直到達到目標交易量
- 注意：此腳本為交易量而生，不保證獲利，並可能因手續費和市場滑點產生虧損。
         請先在 Testnet 測試，並確認 API key 權限與可用資金。
"""

import time
import sys
from decimal import Decimal, ROUND_DOWN
import os
from dotenv import load_dotenv

import gate_api
from gate_api.exceptions import ApiException, GateApiException

# --------- 設定區 ----------
load_dotenv()
API_KEY = os.getenv("GATE_API_KEY")
API_SECRET = os.getenv("GATE_API_SECRET")

# 選擇 host：測試網請改為 testnet host（推薦先測試）
# TESTNET_HOST = "https://api-testnet.gateapi.io/api/v4"
LIVE_HOST = "https://api.gateio.ws/api/v4"
# HOST = TESTNET_HOST # <- 測試網
HOST = LIVE_HOST      # <- 正式上線

# --- 交易參數 ---
CURRENCY_PAIR = "SOL_USDT"       # 交易對
PRINCIPAL_USDT = Decimal("10")   # 每次投入的本金 (USDT)
TARGET_VOLUME_USDT = Decimal("100") # 目標總交易量 (USDT)
ORDER_TIMEOUT_SECONDS = 60       # 訂單等待成交的超時時間 (秒)
# ----------------------------

ACCOUNT = "spot" # 一般現貨

def decimal_round_down(value: Decimal, quantize: Decimal) -> Decimal:
    """向下捨去到 quantize 的小數位"""
    return (value // quantize) * quantize

def print_progress(current_volume, target_volume):
    """打印進度條和詳細資訊"""
    progress = float(current_volume / target_volume)
    bar_length = 40
    block = int(round(bar_length * progress))
    
    text = f"\r進度: [{'#' * block + '-' * (bar_length - block)}] {progress:.2%} | "
    text += f"目前交易量: {current_volume:,.2f} / {target_volume:,.2f} USDT"
    
    sys.stdout.write(text)
    sys.stdout.flush()

def wait_for_order_fill(spot_api, order_id, currency_pair, timeout_seconds):
    """等待訂單成交，帶有超時和取消邏輯"""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            order_status = spot_api.get_order(order_id, currency_pair)
            
            # 在同一行更新等待訊息
            elapsed_time = int(time.time() - start_time)
            sys.stdout.write(f"\r等待訂單 {order_id} 成交... 目前狀態: {order_status.status} (已等待 {elapsed_time}s)")
            sys.stdout.flush()

            if order_status.status in ['finished', 'closed']:
                # 清除等待訊息並換行
                sys.stdout.write("\r" + " " * 80 + "\r") 
                print(f"訂單 {order_id} 已完全成交！")
                return order_status
            elif order_status.status in ['cancelled', 'expired', 'ioc']:
                print(f"\n訂單 {order_id} 未成交，狀態為 {order_status.status} ويعمل.")
                return None
            
            # 訂單仍在 open，繼續等待
            time.sleep(2)

        except GateApiException as ex:
            print(f"\n查詢訂單狀態時出錯: {ex}")
            time.sleep(5) # 發生錯誤時，等待久一點再試
        except Exception as e:
            print(f"\n發生未知錯誤: {e}")
            return None

    # 如果超時
    print(f"\n訂單 {order_id} 在 {timeout_seconds} 秒內未成交，正在取消訂單...")
    try:
        spot_api.cancel_order(order_id, currency_pair)
        print(f"訂單 {order_id} 已取消。")
    except GateApiException as e:
        print(f"\n取消訂單 {order_id} 失敗: {e}")
    return None


def main():
    # --- 初始化 ---
    conf = gate_api.Configuration(host=HOST, key=API_KEY, secret=API_SECRET)
    api_client = gate_api.ApiClient(conf)
    spot_api = gate_api.SpotApi(api_client)
    
    accumulated_volume = Decimal("0")
    cycle_count = 0

    print("--- 自動交易量機器人已啟動 ---")
    print(f"交易對: {CURRENCY_PAIR}")
    print(f"本金: {PRINCIPAL_USDT} USDT")
    print(f"目標交易量: {TARGET_VOLUME_USDT:,.2f} USDT")
    print("---------------------------------")

    try:
        # --- 取得交易對精度 ---
        pair_info = spot_api.get_currency_pair(CURRENCY_PAIR)
        price_precision = getattr(pair_info, 'precision', None)
        amount_precision = getattr(pair_info, 'amount_precision', None)
        if price_precision is None or amount_precision is None:
            raise RuntimeError("無法取得交易對的精度資訊。 সন")
            
        price_quantizer = Decimal('1').scaleb(-int(price_precision))
        amount_quantizer = Decimal('1').scaleb(-int(amount_precision))

        # --- 主循環 ---
        while accumulated_volume < TARGET_VOLUME_USDT:
            cycle_count += 1
            print(f"\n--- 循環 {cycle_count} | 目前交易量: {accumulated_volume:,.2f} USDT ---")

            # --- 買入 ---
            print("步驟 1: 執行買入")
            # 取得最新報價
            tickers = spot_api.list_tickers(currency_pair=CURRENCY_PAIR)
            buy_price = Decimal(tickers[0].highest_bid)
            buy_price_aligned = decimal_round_down(buy_price, price_quantizer)
            
            # 計算數量
            amount = PRINCIPAL_USDT / buy_price_aligned
            amount_aligned = decimal_round_down(amount, amount_quantizer)

            if amount_aligned <= 0:
                print("錯誤：計算出的購買數量為 0，請檢查本金和市場價格。")
                break

            # 下買單
            buy_order = gate_api.Order(
                currency_pair=CURRENCY_PAIR, account=ACCOUNT, side='buy',
                amount=str(amount_aligned), price=str(buy_price_aligned),
                time_in_force='gtc', type='limit'
            )
            created_buy_order = spot_api.create_order(buy_order)
            print(f"已送出買單，ID: {created_buy_order.id}，價格: {buy_price_aligned}, 數量: {amount_aligned}")

            # 等待買單成交
            filled_buy_order = wait_for_order_fill(spot_api, created_buy_order.id, CURRENCY_PAIR, ORDER_TIMEOUT_SECONDS)
            if not filled_buy_order:
                print("買單未成功，終止本次循環。")
                continue
            
            # 更新交易量
            buy_volume = Decimal(filled_buy_order.filled_total)
            accumulated_volume += buy_volume
            print_progress(accumulated_volume, TARGET_VOLUME_USDT)

            # --- 賣出 ---
            print("\n步驟 2: 執行賣出")
            asset_sold = False
            sell_attempt = 0

            # 賣出迴圈，直到成功賣出為止
            while not asset_sold:
                sell_attempt += 1
                print(f"\n第 {sell_attempt} 次嘗試賣出...")

                # 第一次嘗試以 (買價+1 tick) 的價格賣出，後續嘗試以當前市場最高買價賣出
                if sell_attempt == 1:
                    sell_price = (buy_price_aligned + price_quantizer)
                    print("首次嘗試：以 (買價+1 tick) 的價格掛賣單。")
                else:
                    print("後續嘗試：獲取當前市場最高買價掛賣單。")
                    try:
                        tickers = spot_api.list_tickers(currency_pair=CURRENCY_PAIR)
                        sell_price = Decimal(tickers[0].highest_bid)
                    except Exception as e:
                        print(f"獲取最新價格失敗: {e}，等待 5 秒後重試...")
                        time.sleep(5)
                        continue

                sell_price_aligned = decimal_round_down(sell_price, price_quantizer)
                
                # 賣出數量等於實際買入數量
                sell_amount = Decimal(filled_buy_order.amount)
                sell_amount_aligned = decimal_round_down(sell_amount, amount_quantizer)

                # 下賣單
                sell_order = gate_api.Order(
                    currency_pair=CURRENCY_PAIR, account=ACCOUNT, side='sell',
                    amount=str(sell_amount_aligned), price=str(sell_price_aligned),
                    time_in_force='gtc', type='limit'
                )
                created_sell_order = spot_api.create_order(sell_order)
                print(f"已送出賣單，ID: {created_sell_order.id}，價格: {sell_price_aligned}, 數量: {sell_amount_aligned}")

                # 等待賣單成交
                filled_sell_order = wait_for_order_fill(spot_api, created_sell_order.id, CURRENCY_PAIR, ORDER_TIMEOUT_SECONDS)
                
                if filled_sell_order:
                    asset_sold = True # 標記為已售出，跳出 while 迴圈
                    # 更新交易量
                    sell_volume = Decimal(filled_sell_order.filled_total)
                    accumulated_volume += sell_volume
                    print_progress(accumulated_volume, TARGET_VOLUME_USDT)
                else:
                    print("本次賣單嘗試未成功，將自動重新嘗試...")
                    time.sleep(2) # 短暫等待後，進入下一次 while 迴圈


    except (GateApiException, ApiException) as ex:
        print(f"\n\nAPI 錯誤: {ex}")
    except Exception as e:
        print(f"\n\n發生非預期錯誤: {e}")
    finally:
        print("\n\n--- 機器人執行完畢 ---")
        print(f"最終總交易量: {accumulated_volume:,.2f} / {TARGET_VOLUME_USDT:,.2f} USDT")

if __name__ == "__main__":
    main()