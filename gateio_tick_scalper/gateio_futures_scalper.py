"""
Gate.io 合約：自動刷交易量機器人
- 需求：設定目標交易量、每次開倉的倉位價值、交易對、槓桿
- 邏輯：不斷重複開倉-平倉循環，直到達到目標交易量
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
# TESTNET_HOST = "https://fx-api-testnet.gateio.ws/api/v4" # 合約測試網
LIVE_HOST = "https://api.gateio.ws/api/v4"
# HOST = TESTNET_HOST # <- 測試網
HOST = LIVE_HOST      # <- 正式上線

# --- 交易參數 ---
SETTLE = "usdt"                     # 結算貨幣 (usdt / btc)
CONTRACT = "ON_USDT"               # 交易對
LEVERAGE = "10"                      # 槓桿倍數
POSITION_SIZE_USDT = Decimal("160") # 每次開倉的倉位價值 (USDT)
TARGET_VOLUME_USDT = Decimal("12000")# 目標總交易量 (USDT)
ORDER_TIMEOUT_SECONDS = 10          # 訂單等待成交的超時時間 (秒)
# ----------------------------

ACCOUNT = "futures" # 合約帳戶

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

def wait_for_order_fill(futures_api, settle, order_id, timeout_seconds):
    """等待合約訂單成交，帶有超時和取消邏輯"""
    start_time = time.time()
    while time.time() - start_time < timeout_seconds:
        try:
            # 合約API查詢訂單狀態
            order_status = futures_api.get_futures_order(settle, order_id)
            
            elapsed_time = int(time.time() - start_time)
            sys.stdout.write(f"\r等待訂單 {order_id} 成交... 目前狀態: {order_status.status} (已等待 {elapsed_time}s)")
            sys.stdout.flush()

            if order_status.status == 'finished':
                sys.stdout.write("\r" + " " * 80 + "\r") 
                print(f"訂單 {order_id} 已完全成交！")
                return order_status
            elif order_status.status == 'cancelled':
                print(f"\n訂單 {order_id} 未成交，狀態為 {order_status.status}。")
                # 即使取消，也返回最終訂單狀態，以便主循環可以檢查部分成交
                return order_status
            
            time.sleep(2)

        except GateApiException as ex:
            # 在某些情況下，已完全成交的訂單可能會立即從活躍訂單列表中移除，導致查詢時出現 ORDER_NOT_FOUND
            # 我們可以將這種情況視為成功，因為後續的平倉邏輯是基於實際倉位而非訂單回傳
            if "ORDER_NOT_FOUND" in str(ex):
                 sys.stdout.write("\r" + " " * 80 + "\r") 
                 print(f"訂單 {order_id} 狀態為 ORDER_NOT_FOUND，可能已成交。將其視為成功。")
                 # 返回一個模擬的已成交訂單對象。重要的是 left=0，表示沒有剩餘未成交。
                 return gate_api.FuturesOrder(id=order_id, status='finished', size=0, left=0)
            print(f"\n查詢訂單狀態時出錯: {ex}")
            time.sleep(5)
        except Exception as e:
            print(f"\n發生未知錯誤: {e}")
            return None

    # 超時處理
    print(f"\n訂單 {order_id} 在 {timeout_seconds} 秒內未成交，正在取消訂單...")
    try:
        cancelled_order = futures_api.cancel_futures_order(settle, order_id)
        print(f"訂單 {order_id} 已取消。")
        return cancelled_order # 返回被取消的訂單，主循環會檢查其中的 left 屬性
    except GateApiException as e:
        print(f"\n取消訂單 {order_id} 失敗: {e}")
        try:
            # 作為最後手段，再次嘗試獲取訂單狀態
            return futures_api.get_futures_order(settle, order_id)
        except GateApiException as ex:
            print(f"\n獲取最終訂單狀態也失敗: {ex}")
            return None

def main():
    # --- 初始化 ---
    conf = gate_api.Configuration(host=HOST, key=API_KEY, secret=API_SECRET)
    api_client = gate_api.ApiClient(conf)
    futures_api = gate_api.FuturesApi(api_client)
    
    accumulated_volume = Decimal("0")
    cycle_count = 0

    # --- 查詢初始餘額 ---
    initial_usdt_balance = Decimal("0")
    try:
        # 修正: list_futures_accounts 返回的是單一物件，不是列表
        futures_account = futures_api.list_futures_accounts(settle=SETTLE)
        initial_usdt_balance = Decimal(futures_account.available)
    except GateApiException as e:
        print(f"查詢初始合約帳戶餘額失敗: {e}")
        return

    print("--- 合約自動交易量機器人已啟動 ---")
    print(f"結算貨幣: {SETTLE}")
    print(f"交易對: {CONTRACT}")
    print(f"槓桿: {LEVERAGE}x")
    print(f"每次開倉價值: {POSITION_SIZE_USDT} USDT")
    print(f"目標交易量: {TARGET_VOLUME_USDT:,.2f} USDT")
    print(f"初始 {SETTLE.upper()} 帳戶可用餘額: {initial_usdt_balance:,.4f} USDT")
    print("---------------------------------")

    try:
        # --- 設定槓桿 ---
        print(f"正在為 {CONTRACT} 設定槓桿為 {LEVERAGE}x...")
        futures_api.update_position_leverage(settle=SETTLE, contract=CONTRACT, leverage=LEVERAGE)
        print("槓桿設定成功。 সন")

        # --- 取得合約精度 ---
        contract_info = futures_api.get_futures_contract(settle=SETTLE, contract=CONTRACT)
        price_precision = len(contract_info.order_price_round.split('.')[1]) if '.' in contract_info.order_price_round else 0
        price_quantizer = Decimal('1').scaleb(-price_precision)
        # 合約的數量 (size) 是整數 (代表倉位價值 USDT)
        min_order_size = Decimal(contract_info.order_size_min)
        print(f"最小下單價值: {min_order_size} USDT")

        # --- 主循環 ---
        while accumulated_volume < TARGET_VOLUME_USDT:
            cycle_count += 1
            print(f"\n--- 循環 {cycle_count} | 目前交易量: {accumulated_volume:,.2f} USDT ---")

            # --- 記錄循環前餘額 ---
            usdt_balance_before_cycle = Decimal("0")
            try:
                # 修正: list_futures_accounts 返回的是單一物件，不是列表
                account_before = futures_api.list_futures_accounts(settle=SETTLE)
                usdt_balance_before_cycle = Decimal(account_before.available)
            except GateApiException as e:
                print(f"查詢循環前餘額失敗: {e}")

            # --- 步驟 1: 開多倉 ---
            print("步驟 1: 執行開多倉")
            
            # 自動調整開倉價值
            order_size_dec = POSITION_SIZE_USDT
            required_margin = POSITION_SIZE_USDT / Decimal(LEVERAGE)

            if usdt_balance_before_cycle < required_margin:
                print(f"可用餘額 ({usdt_balance_before_cycle:.4f} USDT) 不足預設倉位所需保證金 ({required_margin:.4f} USDT)，將自動縮小倉位。")
                # 使用 99% 的可用餘額來計算倉位價值，留出一些緩衝
                order_size_dec = (usdt_balance_before_cycle * Decimal('0.99')) * Decimal(LEVERAGE)
                order_size_dec = order_size_dec.quantize(Decimal('1'), rounding=ROUND_DOWN)

            if order_size_dec < min_order_size:
                print(f"計算出的倉位價值 ({order_size_dec} USDT) 小於最小下單價值 {min_order_size} USDT。")
                print("機器人停止。 সন")
                break
            
            # 再次檢查調整後的保證金是否足夠
            final_required_margin = order_size_dec / Decimal(LEVERAGE)
            if usdt_balance_before_cycle < final_required_margin:
                 print(f"可用餘額 ({usdt_balance_before_cycle:.4f} USDT) 仍不足以開立調整後的倉位 (需要 {final_required_margin:.4f} USDT)。")
                 print("機器人停止。 সন")
                 break

            # 獲取最新報價 (吃單策略: 使用賣一價)
            try:
                order_book = futures_api.list_futures_order_book(settle=SETTLE, contract=CONTRACT, limit=1)
                if not order_book.asks or not order_book.asks[0].p:
                    print("訂單簿賣方為空，無法獲取開倉價格，跳過此循環。")
                    time.sleep(5)
                    continue
                open_price = Decimal(order_book.asks[0].p)
                open_price_aligned = decimal_round_down(open_price, price_quantizer)
            except GateApiException as e:
                print(f"獲取訂單簿失敗: {e}，跳過此循環。")
                time.sleep(5)
                continue
            
            # 下開倉單
            order_size = int(order_size_dec)
            open_order = gate_api.FuturesOrder(
                contract=CONTRACT,
                size=order_size, # 正數為開多
                price=str(open_price_aligned),
                tif='gtc', # 修正: time_in_force -> tif
                close=False # 明確指定為開倉
            )
            created_open_order = futures_api.create_futures_order(settle=SETTLE, futures_order=open_order)
            print(f"已送出開多倉訂單，ID: {created_open_order.id}，價格: {open_price_aligned}, 價值: {order_size} USDT")

            # 等待開倉單成交
            filled_open_order = wait_for_order_fill(futures_api, SETTLE, created_open_order.id, ORDER_TIMEOUT_SECONDS)

            # 檢查訂單是否成功以及是否完全成交
            if not filled_open_order or filled_open_order.left != 0:
                print("\n開倉單處理失敗或未完全成交，終止本次循環。 সন")
                # 確保所有殘留訂單都被取消
                try:
                    print("正在取消此合約的所有剩餘掛單...")
                    futures_api.cancel_futures_orders(settle=SETTLE, contract=CONTRACT)
                    print("已取消該合約所有掛單。 সন")
                except GateApiException as e:
                    print(f"取消殘留掛單失敗: {e}")
                continue

            # 更新交易量 (size 是合約張數，對於USDT本位合約，1張=1 USD價值)
            open_volume = Decimal(abs(filled_open_order.size))
            accumulated_volume += open_volume
            print(f"\n開倉成功，成交價值: {open_volume:.4f} USDT")
            print_progress(accumulated_volume, TARGET_VOLUME_USDT)

            # --- 步驟 2: 平多倉 ---
            time.sleep(0.5) # 短暫等待，確保倉位狀態更新

            # 查詢當前倉位
            current_position = None
            try:
                # 修正: list_positions 不能用 contract 參數過濾，需在程式碼中自行過濾
                all_positions = futures_api.list_positions(settle=SETTLE)
                # 篩選出指定合約的多頭倉位
                long_positions = [p for p in all_positions if p.contract == CONTRACT and p.size > 0]
                if long_positions:
                    current_position = long_positions[0]
            except GateApiException as e:
                print(f"\n查詢倉位失敗: {e}，終止本次循環。 সন")
                continue

            if not current_position:
                print("\n未找到需要平倉的多頭倉位，可能已被手動平倉或API延遲。 সন")
                continue
            
            position_to_close_size = current_position.size
            print(f"\n步驟 2: 執行平多倉，數量: {position_to_close_size}")

            asset_closed = False
            close_attempt = 0
            while not asset_closed:
                close_attempt += 1
                print(f"\n第 {close_attempt} 次嘗試平倉...")

                # 獲取最新報價 (吃單策略: 使用買一價)
                try:
                    order_book = futures_api.list_futures_order_book(settle=SETTLE, contract=CONTRACT, limit=1)
                    if not order_book.bids or not order_book.bids[0].p:
                        print("訂單簿買方為空，無法獲取平倉價格，等待 5 秒後重試...")
                        time.sleep(5)
                        continue
                    close_price = Decimal(order_book.bids[0].p)
                    close_price_aligned = decimal_round_down(close_price, price_quantizer)
                except GateApiException as e:
                    print(f"獲取訂單簿失敗: {e}，等待 5 秒後重試...")
                    time.sleep(5)
                    continue

                # 下平倉單
                close_order = gate_api.FuturesOrder(
                    contract=CONTRACT,
                    size=-position_to_close_size, # 負數為平多
                    price=str(close_price_aligned),
                    tif='gtc', # 修正: time_in_force -> tif
                    reduce_only=True # 只減倉，避免意外開空倉
                )
                created_close_order = futures_api.create_futures_order(settle=SETTLE, futures_order=close_order)
                print(f"已送出平多倉訂單，ID: {created_close_order.id}，價格: {close_price_aligned}, 數量: {-position_to_close_size}")

                # 等待平倉單成交
                filled_close_order = wait_for_order_fill(futures_api, SETTLE, created_close_order.id, ORDER_TIMEOUT_SECONDS)

                if filled_close_order and filled_close_order.left == 0:
                    asset_closed = True
                    close_volume = Decimal(abs(filled_close_order.size))
                    accumulated_volume += close_volume
                    print(f"\n平倉成功，成交價值: {close_volume:.4f} USDT")
                    print_progress(accumulated_volume, TARGET_VOLUME_USDT)

                    # --- 計算本次循環磨損 ---
                    try:
                        # 修正: list_futures_accounts 返回的是單一物件，不是列表
                        account_after = futures_api.list_futures_accounts(settle=SETTLE)
                        usdt_balance_after_cycle = Decimal(account_after.available)
                        
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
                    print("\n本次平倉嘗試未成功或未完全成交，將自動重新嘗試...")
                    # 檢查是否還有倉位
                    try:
                        # 修正: list_positions 不能用 contract 參數過濾，需在程式碼中自行過濾
                        all_positions = futures_api.list_positions(settle=SETTLE)
                        long_positions = [p for p in all_positions if p.contract == CONTRACT and p.size > 0]
                        if not long_positions:
                            print("倉位已為0，停止平倉嘗試。 সন")
                            asset_closed = True
                        else:
                            position_to_close_size = long_positions[0].size # 更新剩餘倉位數量
                    except GateApiException as e:
                        print(f"重新查詢倉位時出錯: {e}，停止平倉嘗試。 সন")
                        asset_closed = True
                    time.sleep(2)

    except (GateApiException, ApiException) as ex:
        print(f"\n\nAPI 錯誤: {ex}")
    except Exception as e:
        print(f"\n\n發生非預期錯誤: {e}")
    finally:
        print("\n\n--- 機器人執行完畢 ---")
        print(f"最終總交易量: {accumulated_volume:,.2f} / {TARGET_VOLUME_USDT:,.2f} USDT")
        
        # --- 顯示最終結果 ---
        try:
            # 修正: list_futures_accounts 返回的是單一物件，不是列表
            final_futures_account = futures_api.list_futures_accounts(settle=SETTLE)
            final_usdt_balance = Decimal(final_futures_account.available)
            print(f"最終 {SETTLE.upper()} 帳戶可用餘額: {final_usdt_balance:,.4f} USDT")

            if initial_usdt_balance > 0:
                total_slippage = initial_usdt_balance - final_usdt_balance
                if total_slippage >= 0:
                    print(f"總磨損: {total_slippage:,.4f} USDT")
                else:
                    print(f"總獲利: {-total_slippage:,.4f} USDT")
        except GateApiException as e:
            print(f"查詢最終餘額失敗: {e}")

if __name__ == "__main__":
    main()
