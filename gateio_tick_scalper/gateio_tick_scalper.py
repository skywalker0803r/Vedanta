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
CURRENCY_PAIR = "STRIKE_USDT"       # 交易對
PRINCIPAL_USDT = Decimal("100")   # 每次投入的本金 (USDT)
TARGET_VOLUME_USDT = Decimal("3000") # 目標總交易量 (USDT)
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
        # 取消訂單並獲取最終訂單狀態
        cancelled_order = spot_api.cancel_order(order_id, currency_pair)
        print(f"訂單 {order_id} 已取消。")
        # 即使取消，也返回最終訂單狀態，以便主循環可以檢查部分成交
        return cancelled_order
    except GateApiException as e:
        print(f"\n取消訂單 {order_id} 失敗: {e}")
        # 如果取消失敗，嘗試重新獲取訂單狀態作為最後手段
        try:
            return spot_api.get_order(order_id, currency_pair)
        except GateApiException as ex:
            print(f"\n獲取最終訂單狀態也失敗: {ex}")
            return None


def main():
    # --- 初始化 ---
    conf = gate_api.Configuration(host=HOST, key=API_KEY, secret=API_SECRET)
    api_client = gate_api.ApiClient(conf)
    spot_api = gate_api.SpotApi(api_client)
    
    accumulated_volume = Decimal("0")
    cycle_count = 0

    # --- 查詢初始餘額 ---
    initial_usdt_balance = Decimal("0")
    try:
        usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
        if usdt_accounts:
            initial_usdt_balance = Decimal(usdt_accounts[0].available)
    except GateApiException as e:
        print(f"查詢初始餘額失敗: {e}")

    print("--- 自動交易量機器人已啟動 ---")
    print(f"交易對: {CURRENCY_PAIR}")
    print(f"本金: {PRINCIPAL_USDT} USDT")
    print(f"目標交易量: {TARGET_VOLUME_USDT:,.2f} USDT")
    print(f"初始 USDT 餘額: {initial_usdt_balance:,.4f} USDT")
    print("---------------------------------")

    try:
        # --- 取得交易對精度 ---
        pair_info = spot_api.get_currency_pair(CURRENCY_PAIR)
        price_precision = getattr(pair_info, 'precision', None)
        amount_precision = getattr(pair_info, 'amount_precision', None)
        min_quote_amount = getattr(pair_info, 'min_quote_amount', None)
        if price_precision is None or amount_precision is None or min_quote_amount is None:
            raise RuntimeError("無法取得交易對的精度或最小下單金額資訊。")
        
        min_quote_amount = Decimal(min_quote_amount)
        print(f"最小下單金額: {min_quote_amount} USDT")
            
        price_quantizer = Decimal('1').scaleb(-int(price_precision))
        amount_quantizer = Decimal('1').scaleb(-int(amount_precision))

        # --- 主循環 ---
        while accumulated_volume < TARGET_VOLUME_USDT:
            cycle_count += 1
            print(f"\n--- 循環 {cycle_count} | 目前交易量: {accumulated_volume:,.2f} USDT ---")

            # --- 記錄循環前餘額 ---
            usdt_balance_before_cycle = Decimal("0")
            try:
                usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
                if usdt_accounts:
                    usdt_balance_before_cycle = Decimal(usdt_accounts[0].available)
            except GateApiException as e:
                print(f"查詢循環前餘額失敗: {e}")

            # --- 買入 ---
            print("步驟 1: 執行買入")

            # --- 使用當前可用餘額或設定的本金（取較小者）進行購買 ---
            available_usdt = Decimal("0")
            try:
                usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
                if usdt_accounts:
                    available_usdt = Decimal(usdt_accounts[0].available)
            except GateApiException as e:
                print(f"查詢可用 USDT 餘額失敗: {e}，終止循環。")
                break # 中斷主循環

            buy_principal = min(available_usdt, PRINCIPAL_USDT)
            
            if buy_principal < min_quote_amount:
                print(f"可用資金 ({buy_principal:.4f} USDT) 不足，小於最小下單金額 {min_quote_amount} USDT。")
                print("機器人停止。")
                break # 資金不足，中斷主循環

            # 取得最新報價
            tickers = spot_api.list_tickers(currency_pair=CURRENCY_PAIR)
            buy_price = Decimal(tickers[0].highest_bid)
            buy_price_aligned = decimal_round_down(buy_price, price_quantizer)
            
            # 計算數量
            amount = buy_principal / buy_price_aligned
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
                print("買單處理失敗，無法獲取最終狀態，終止本次循環。")
                continue

            # 檢查是否有任何成交量
            buy_volume = Decimal(filled_buy_order.filled_total)
            if buy_volume <= 0:
                print("買單最終無成交，終止本次循環。")
                continue
            
            # 更新交易量
            print(f"\n買單部分或完全成交，成交金額: {buy_volume:.4f} USDT")
            accumulated_volume += buy_volume
            print_progress(accumulated_volume, TARGET_VOLUME_USDT)

            # --- 賣出 ---
            # print("\n步驟 2: 執行賣出") # 改為在迴圈內打印
            
            # 為確保賣出數量精確 (考慮手續費等因素)，直接查詢帳戶可用餘額
            base_currency = CURRENCY_PAIR.split('_')[0]
            try:
                spot_accounts = spot_api.list_spot_accounts(currency=base_currency)
                if not spot_accounts:
                    print(f"\n錯誤：無法查詢到 {base_currency} 的餘額。")
                    continue # 跳過此循環
                
                available_balance = Decimal(spot_accounts[0].available)
                sell_amount_aligned = decimal_round_down(available_balance, amount_quantizer)

            except GateApiException as e:
                print(f"\n查詢 {base_currency} 餘額時出錯: {e}，終止本次循環。")
                continue

            if sell_amount_aligned <= 0:
                print(f"\n基礎貨幣 {base_currency} 可用餘額為 0 或過少，無法賣出。")
                continue

            asset_sold = False
            sell_attempt = 0
            # 賣出迴圈，直到成功賣出為止
            while not asset_sold:
                sell_attempt += 1
                print(f"\n第 {sell_attempt} 次嘗試賣出 {sell_amount_aligned} {base_currency}...")

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

                # --- 檢查是否滿足最小下單金額 ---
                sell_value = sell_amount_aligned * sell_price_aligned
                if sell_value < min_quote_amount:
                    print(f"\n賣出價值 ({sell_value:.4f} USDT) 過低，小於最小金額 {min_quote_amount} USDT，無法賣出。")
                    print("此部分資產將留在帳戶中，本循環結束。")
                    asset_sold = True # 標記為已處理，以跳出賣出迴圈
                    continue

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
                
                # 檢查賣單是否真的有成交
                if filled_sell_order and Decimal(filled_sell_order.filled_total) > 0:
                    asset_sold = True # 標記為已售出，跳出 while 迴圈
                    # 更新交易量
                    sell_volume = Decimal(filled_sell_order.filled_total)
                    accumulated_volume += sell_volume
                    print_progress(accumulated_volume, TARGET_VOLUME_USDT)

                    # --- 計算本次循環磨損 ---
                    usdt_balance_after_cycle = Decimal("0")
                    try:
                        usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
                        if usdt_accounts:
                            usdt_balance_after_cycle = Decimal(usdt_accounts[0].available)
                        
                        if usdt_balance_before_cycle > 0:
                            cycle_slippage = usdt_balance_before_cycle - usdt_balance_after_cycle
                            if cycle_slippage >= 0:
                                sys.stdout.write(f" | 本次循環磨損: {cycle_slippage:,.4f} USDT")
                            else:
                                sys.stdout.write(f" | 本次循環獲利: {-cycle_slippage:,.4f} USDT")
                            sys.stdout.flush()

                    except GateApiException as e:
                        print(f" | 查詢循環後餘額失敗: {e}")

                else:
                    print("\n本次賣單嘗試未成功或無成交，將自動重新嘗試...")
                    # 在重試之前，最好再次檢查餘額，因為訂單可能部分成交後被取消
                    try:
                        spot_accounts = spot_api.list_spot_accounts(currency=base_currency)
                        if spot_accounts:
                            available_balance = Decimal(spot_accounts[0].available)
                            sell_amount_aligned = decimal_round_down(available_balance, amount_quantizer)
                            if sell_amount_aligned <= 0:
                                print(f"{base_currency} 餘額已為 0，停止賣出嘗試。")
                                asset_sold = True # 餘額為0，也算“處理完畢”，跳出迴圈
                        else:
                            print(f"無法重新查詢 {base_currency} 餘額，停止賣出嘗試。")
                            asset_sold = True # 避免無限迴圈
                    except GateApiException as e:
                        print(f"重新查詢餘額時出錯: {e}，停止賣出嘗試。")
                        asset_sold = True # 避免無限迴圈
                    
                    time.sleep(2) # 短暫等待後，進入下一次 while 迴圈


    except (GateApiException, ApiException) as ex:
        print(f"\n\nAPI 錯誤: {ex}")
    except Exception as e:
        print(f"\n\n發生非預期錯誤: {e}")
    finally:
        print("\n\n--- 機器人執行完畢 ---")
        print(f"最終總交易量: {accumulated_volume:,.2f} / {TARGET_VOLUME_USDT:,.2f} USDT")
        
        # --- 顯示最終結果 ---
        final_usdt_balance = Decimal("0")
        try:
            usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
            if usdt_accounts:
                final_usdt_balance = Decimal(usdt_accounts[0].available)
            print(f"最終 USDT 餘額: {final_usdt_balance:,.4f} USDT")
        except GateApiException as e:
            print(f"查詢最終餘額失敗: {e}")

        if initial_usdt_balance > 0:
            total_slippage = initial_usdt_balance - final_usdt_balance
            if total_slippage >= 0:
                print(f"總磨損: {total_slippage:,.4f} USDT")
            else:
                print(f"總獲利: {-total_slippage:,.4f} USDT")

if __name__ == "__main__":
    main()