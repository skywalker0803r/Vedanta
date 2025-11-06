"""
Gate.io 現貨：自動刷交易量機器人 (v3.3 - 支援雙策略模式 & P&L 校正)
- 需求：設定目標交易量、每次投入的本金、交易對
- 策略：
    - 模式 1: (Maker掛單買) -> (等待10秒) -> (若成交) -> (等待1秒) -> (Taker市價賣)
    - 模式 2: (Taker市價買) -> (若成交) -> (Taker市價賣) -> (等待2秒)
- P&L 校正：
    - 啟動時會自動偵測帳戶中已有的基礎貨幣（如 ON）價值。
    - 在計算累計 P&L 時會扣除此初始價值，以隔離機器人本身的磨損。
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

# 選擇 host
LIVE_HOST = "https://api.gateio.ws/api/v4"
HOST = LIVE_HOST

# --- 【*** 策略模式選擇 ***】 ---
# 模式 1: Maker-Taker (Maker掛單買, Taker市價賣)
# 模式 2: Taker-Taker (Taker限價買, Taker市價賣)
STRATEGY_MODE = 2 # ★★★ 請在此選擇 1 或 2 ★★★
# --------------------------------

# --- 交易參數 ---
CURRENCY_PAIR = "ON_USDT"       # 交易對
PRINCIPAL_USDT = Decimal("5")    # 每次投入的本金 (USDT)
TARGET_VOLUME_USDT = Decimal("1000") # 目標總交易量 (USDT)

# --- 策略 1 (Maker-Taker) 專用參數 ---
MAKER_BUY_TIMEOUT = 10      # 模式 1: Maker 掛單等待成交的超時時間 (秒)
MAKER_POST_BUY_DELAY = 1    # 模式 1: Maker 買單成交後，等待 N 秒再賣出

# --- 策略 2 (Taker-Taker) 專用參數 ---
TAKER_POST_SELL_DELAY = 2   # 模式 2: Taker 賣出成交後，額外冷卻 N 秒

# --- 通用參數 ---
GENERAL_TIMEOUT = 5         # Taker單/市價單/取消單的通用超時 (秒)
API_CYCLE_COOLDOWN = 1      # 每個循環末尾的基礎冷卻 (秒)，防止 429
# ----------------------------

ACCOUNT = "spot" # 一般現貨

def decimal_round_down(value: Decimal, quantize: Decimal) -> Decimal:
    """向下捨去到 quantize 的小數位"""
    return (value // quantize) * quantize

def print_progress(current_volume, target_volume):
    """打印進度條和詳細資訊"""
    progress = 0.0
    if target_volume > 0:
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
            
            elapsed_time = int(time.time() - start_time)
            sys.stdout.write(f"\r等待訂單 {order_id} 成交... 目前狀態: {order_status.status} (已等待 {elapsed_time}s / {timeout_seconds}s)")
            sys.stdout.flush()

            if order_status.status in ['finished', 'closed']:
                sys.stdout.write("\r" + " " * 80 + "\r") 
                print(f"訂單 {order_id} 已完全成交！")
                return order_status
            elif order_status.status in ['cancelled', 'expired', 'ioc']:
                sys.stdout.write("\r" + " " * 80 + "\r")
                print(f"訂單 {order_id} 未成交或已關閉，狀態為 {order_status.status}。")
                return order_status

            time.sleep(0.5)

        except GateApiException as ex:
            if "ORDER_NOT_FOUND" in str(ex):
                print(f"\n訂單 {order_id} 尚未在 API 中找到，可能是延遲，稍後重試...")
                time.sleep(1)
                continue
            
            print(f"\n查詢訂單狀態時出錯: {ex}")
            time.sleep(1)
        except Exception as e:
            print(f"\n發生未知錯誤: {e}")
            return None

    # 如果超時
    print(f"\n訂單 {order_id} 在 {timeout_seconds} 秒內未成交，正在取消訂單...")
    try:
        cancelled_order = spot_api.cancel_order(order_id, currency_pair)
        print(f"訂單 {order_id} 已取消。")
        return cancelled_order
    except GateApiException as e:
        print(f"\n取消訂單 {order_id} 失敗: {e}")
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

    # --- 查詢初始 USDT 餘額 ---
    initial_usdt_balance = Decimal("0")
    try:
        usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
        if usdt_accounts:
            initial_usdt_balance = Decimal(usdt_accounts[0].available)
    except GateApiException as e:
        print(f"查詢初始 USDT 餘額失敗: {e}")
        
    # --- 【*** v3.3 P&L 校正：檢查初始基礎貨幣餘額 ***】 ---
    initial_base_value = Decimal("0")
    base_currency = CURRENCY_PAIR.split('_')[0]
    try:
        spot_accounts = spot_api.list_spot_accounts(currency=base_currency)
        if spot_accounts:
            initial_base_balance = Decimal(spot_accounts[0].available)
            if initial_base_balance > 0:
                print("--- P&L 校正 ---")
                print(f"警告：偵測到 {initial_base_balance} {base_currency} 初始餘額。")
                # 獲取當前價格來估算價值
                tickers = spot_api.list_tickers(currency_pair=CURRENCY_PAIR)
                initial_price = Decimal(tickers[0].highest_bid)
                initial_base_value = initial_base_balance * initial_price
                print(f"此餘額價值約: {initial_base_value:,.4f} USDT。")
                print("後續「累計磨損」將自動扣除此價值，以反映機器人真實 P&L。")
                print("-------------------")
    except Exception as e:
        print(f"查詢初始 {base_currency} 餘額失敗: {e}，P&L 統計可能不準確。")
    # --- 【*** P&L 校正結束 ***】 ---


    print("\n--- 自動交易量機器人已啟動 ---")
    if STRATEGY_MODE == 1:
        print(f"策略模式: 1 (Maker-Taker / 掛單買 - 市價賣)")
        print(f"掛單超時: {MAKER_BUY_TIMEOUT} 秒")
    elif STRATEGY_MODE == 2:
        print(f"策略模式: 2 (Taker-Taker / 限價買 - 市價賣)")
        print(f"賣後冷卻: {TAKER_POST_SELL_DELAY} 秒")
    else:
        print(f"錯誤：未知的 STRATEGY_MODE ({STRATEGY_MODE})")
        return

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

            # --- 步驟 1: 買入 (根據策略模式決定) ---

            # --- 檢查可用餘額 (通用) ---
            available_usdt = Decimal("0")
            try:
                usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
                if usdt_accounts:
                    available_usdt = Decimal(usdt_accounts[0].available)
            except GateApiException as e:
                print(f"查詢可用 USDT 餘額失敗: {e}，稍後重試...")
                time.sleep(10)
                continue

            buy_principal = min(available_usdt, PRINCIPAL_USDT)
            
            if buy_principal < min_quote_amount:
                print(f"可用資金 ({buy_principal:.4f} USDT) 不足... 等待 10 秒後重試...")
                time.sleep(10)
                continue

            # --- 獲取報價 (通用) ---
            try:
                tickers = spot_api.list_tickers(currency_pair=CURRENCY_PAIR)
            except Exception as e:
                print(f"獲取報價失敗: {e}，稍後重試...")
                time.sleep(5)
                continue

            # --- 根據策略設定買入參數 ---
            buy_price = Decimal("0")
            buy_time_in_force = ""
            buy_order_timeout = 0

            if STRATEGY_MODE == 1:
                # --- 模式 1: Maker 掛單買 ---
                print("步驟 1: 執行買入 (Maker 掛單)")
                buy_price = Decimal(tickers[0].highest_bid)
                buy_time_in_force = 'gtc'
                buy_order_timeout = MAKER_BUY_TIMEOUT # 10 秒
            
            elif STRATEGY_MODE == 2:
                # --- 模式 2: Taker 限價買 ---
                print("步驟 1: 執行買入 (Taker 限價單)")
                buy_price = Decimal(tickers[0].lowest_ask) # Taker 邏輯
                buy_time_in_force = 'ioc' # Taker 邏輯
                buy_order_timeout = GENERAL_TIMEOUT # 5 秒

            buy_price_aligned = decimal_round_down(buy_price, price_quantizer)
            
            if buy_price_aligned <= 0:
                print(f"錯誤：市場價格 ({buy_price}) 為 0，無法計算數量。")
                time.sleep(5)
                continue
            
            amount = buy_principal / buy_price_aligned
            amount_aligned = decimal_round_down(amount, amount_quantizer)

            if amount_aligned <= 0:
                print("錯誤：計算出的購買數量為 0。")
                time.sleep(5)
                continue

            # --- 下買單 (通用) ---
            try:
                buy_order = gate_api.Order(
                    currency_pair=CURRENCY_PAIR, account=ACCOUNT, side='buy',
                    amount=str(amount_aligned), price=str(buy_price_aligned),
                    time_in_force=buy_time_in_force, # 根據策略
                    type='limit'
                )
                created_buy_order = spot_api.create_order(buy_order)
                print(f"已送出買單 (模式 {STRATEGY_MODE})，ID: {created_buy_order.id}，價格: {buy_price_aligned}, 數量: {amount_aligned}")
            except GateApiException as e:
                print(f"下買單失敗: {e}，稍後重試...")
                time.sleep(5)
                continue

            # --- 等待買單成交 (通用) ---
            filled_buy_order = wait_for_order_fill(spot_api, created_buy_order.id, CURRENCY_PAIR, buy_order_timeout) # 根據策略

            if not filled_buy_order:
                print("買單處理失敗，無法獲取最終狀態，終止本次循環。")
                continue

            buy_volume = Decimal(filled_buy_order.filled_total)
            if buy_volume <= 0:
                print(f"買單 (模式 {STRATEGY_MODE}) 最終無成交 (已取消或ioc未成交)，終止本次循環。")
                continue
            
            # --- 買單成功 ---
            print(f"買單部分或完全成交，成交金額: {buy_volume:.4f} USDT")
            accumulated_volume += buy_volume
            print_progress(accumulated_volume, TARGET_VOLUME_USDT)

            # --- 策略 1: 買後等待 ---
            if STRATEGY_MODE == 1:
                print(f" | 買單已成交，等待 {MAKER_POST_BUY_DELAY} 秒後市價賣出...")
                time.sleep(MAKER_POST_BUY_DELAY)


            # --- 步驟 2: 市價賣出 (Taker) (通用) ---
            
            # base_currency 已在 v3.3 頂部定義
            sell_amount_aligned = Decimal("0")
            
            try:
                spot_accounts = spot_api.list_spot_accounts(currency=base_currency)
                if not spot_accounts:
                    print(f"\n錯誤：無法查詢到 {base_currency} 的餘額。")
                    continue
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
            
            while not asset_sold and sell_amount_aligned > 0:
                sell_attempt += 1
                print(f"\n步驟 2: 第 {sell_attempt} 次嘗試『市價賣出』 {sell_amount_aligned} {base_currency}...")
                
                try:
                    tickers = spot_api.list_tickers(currency_pair=CURRENCY_PAIR)
                    estimated_price = Decimal(tickers[0].highest_bid)
                    if estimated_price <= 0:
                        print("錯誤：市場賣價為 0，無法估算價值。")
                        time.sleep(5)
                        continue
                        
                    sell_value = sell_amount_aligned * estimated_price
                    if sell_value < min_quote_amount:
                        print(f"\n估算賣出價值 ({sell_value:.4f} USDT) 過低...資產將留在帳戶中。")
                        asset_sold = True
                        continue
                except Exception as e:
                    print(f"獲取價格以估算價值時出錯: {e}...")
                    time.sleep(5)
                    continue

                sell_order = gate_api.Order(
                    currency_pair=CURRENCY_PAIR, account=ACCOUNT, side='sell',
                    amount=str(sell_amount_aligned), type='market', time_in_force='ioc'
                )
                
                try:
                    created_sell_order = spot_api.create_order(sell_order)
                    print(f"已送出市價賣單，ID: {created_sell_order.id}")
                    
                    filled_sell_order = wait_for_order_fill(spot_api, created_sell_order.id, CURRENCY_PAIR, GENERAL_TIMEOUT)

                    if filled_sell_order and Decimal(filled_sell_order.filled_total) > 0:
                        asset_sold = True
                        sell_volume = Decimal(filled_sell_order.filled_total)
                        print(f"市價單已成交，成交金額: {sell_volume:.4f} USDT")
                        accumulated_volume += sell_volume
                        print_progress(accumulated_volume, TARGET_VOLUME_USDT)
                        
                        # --- 【*** v3.3 P&L 校正：計算磨損 ***】 ---
                        try:
                            usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
                            if usdt_accounts:
                                usdt_balance_after_cycle = Decimal(usdt_accounts[0].available)
                                
                                # 顯示 "本次循環磨損"
                                if usdt_balance_before_cycle > 0:
                                    cycle_slippage = usdt_balance_before_cycle - usdt_balance_after_cycle
                                    if cycle_slippage >= 0:
                                        sys.stdout.write(f" | 本次循環磨損: {cycle_slippage:,.4f} USDT")
                                    else:
                                        sys.stdout.write(f" | 本次循環獲利: {-cycle_slippage:,.4f} USDT")

                                # 顯示 "累計磨損" (使用 P&L 校正)
                                if initial_usdt_balance > 0:
                                    # 原始 P&L = 初始 USDT - 當前 USDT
                                    raw_cumulative_loss = initial_usdt_balance - usdt_balance_after_cycle
                                    # 校正 P&L = 原始 P&L + 初始基礎貨幣價值
                                    cumulative_loss = raw_cumulative_loss + initial_base_value
                                    
                                    if cumulative_loss >= 0:
                                        sys.stdout.write(f" | 累計磨損 (校正): {cumulative_loss:,.4f} USDT")
                                    else:
                                        sys.stdout.write(f" | 累計獲利 (校正): {-cumulative_loss:,.4f} USDT")
                                    sys.stdout.flush()
                                    
                        except GateApiException as e:
                            print(f" | 查詢循環後餘額失敗: {e}")
                        # --- 【*** P&L 校正結束 ***】 ---
                        
                        # --- 策略 2: 賣後等待 ---
                        if STRATEGY_MODE == 2:
                            print(f"\n | Taker 賣出完畢，冷卻 {TAKER_POST_SELL_DELAY} 秒...")
                            time.sleep(TAKER_POST_SELL_DELAY)

                    else:
                        print("\n市價單狀態未知或無成交，等待 5 秒後重試...")
                        time.sleep(5)
                except GateApiException as e:
                    print(f"\n送出市價賣單時發生 API 錯誤: {e}，等待 5 秒後重試...")
                    time.sleep(5)
                except Exception as e:
                    print(f"\n市價賣單發生未知錯誤: {e}，等待 5 秒後重試...")
                    time.sleep(5)

                # 重試前檢查餘額
                if not asset_sold:
                    try:
                        spot_accounts = spot_api.list_spot_accounts(currency=base_currency)
                        if spot_accounts:
                            available_balance = Decimal(spot_accounts[0].available)
                            sell_amount_aligned = decimal_round_down(available_balance, amount_quantizer)
                            if sell_amount_aligned <= 0:
                                print(f"{base_currency} 餘額已為 0，停止賣出嘗試。")
                                asset_sold = True
                        else:
                            print(f"無法重新查詢 {base_currency} 餘額，停止賣出嘗試。")
                            asset_sold = True
                    except GateApiException as e:
                        print(f"重新查詢餘額時出錯: {e}，停止賣出嘗試。")
                        asset_sold = True
            
            # --- 循環末尾冷卻 (通用) ---
            if accumulated_volume < TARGET_VOLUME_USDT:
                print(f" | 循環 {cycle_count} 結束，基礎冷卻 {API_CYCLE_COOLDOWN} 秒...")
                time.sleep(API_CYCLE_COOLDOWN)

    except (GateApiException, ApiException) as ex:
        print(f"\n\nAPI 錯誤導致機器人終止: {ex}")
    except KeyboardInterrupt:
        print("\n\n偵測到手動中斷 (Ctrl+C)...")
    except Exception as e:
        print(f"\n\n發生非預期錯誤導致機器人終止: {e}")
    finally:
        print("\n\n--- 機器人執行完畢 ---")
        print(f"最終總交易量: {accumulated_volume:,.2f} / {TARGET_VOLUME_USDT:,.2f} USDT")
        
        try:
            usdt_accounts = spot_api.list_spot_accounts(currency='USDT')
            if usdt_accounts:
                final_usdt_balance = Decimal(usdt_accounts[0].available)
                print(f"最終 USDT 餘額: {final_usdt_balance:,.4f} USDT")
                
                if initial_usdt_balance > 0:
                    # 最終 P&L 也使用校正
                    raw_total_slippage = initial_usdt_balance - final_usdt_balance
                    total_slippage = raw_total_slippage + initial_base_value
                    
                    if total_slippage >= 0:
                        print(f"總磨損 (校正): {total_slippage:,.4f} USDT")
                    else:
                        print(f"總獲利 (校正): {-total_slippage:,.4f} USDT")
        except GateApiException as e:
            print(f"查詢最終餘額失敗: {e}")

if __name__ == "__main__":
    main()