import numpy as np
import pandas as pd

def backtest_signals(df: pd.DataFrame,
                        initial_capital=100,
                        fee_rate=0.001,
                        leverage=1,
                        allow_short=True,
                        stop_loss=None,      # 停損閾值，例如0.05代表5%
                        take_profit=None,    # 停利閾值
                        max_hold_bars=None,  # 最大持有K棒數
                        slippage_rate=0.0005, # 新增: 模擬滑點比率，例如0.0005代表0.05%
                        execution_price_type='open'): # 新增: 進出場價格類型 ('open', 'close', 'high_low')
    """
    回測交易訊號策略，支援槓桿與放空，並考慮雙邊手續費、滑點。
    新增停損、停利與固定持有時間機制，並優化K棒內部停損/停利判斷。

    參數：
        df (pd.DataFrame): 必須包含 'open', 'high', 'low', 'close', 'signal', 'timestamp' 欄位。
        initial_capital (float): 初始資金。
        fee_rate (float): 每次交易的單邊手續費費率。
        leverage (int): 使用的槓桿倍數。
        allow_short (bool): 是否允許放空。
        stop_loss (float or None): 停損點，負報酬率門檻，如0.05表示5%。
        take_profit (float or None): 停利點。
        max_hold_bars (int or None): 最大持有時間（K棒數）。
        slippage_rate (float): 模擬的滑點比率，實際滑點會在 -slippage_rate 到 +slippage_rate 之間隨機產生。
                                正值表示對您不利的滑點（買入價更高，賣出價更低）。
        execution_price_type (str): 指定進出場價格的類型。
                                    'open': 在下一個K棒的開盤價進出場。
                                    'close': 在當前K棒的收盤價進出場（原邏輯，但停損停利會檢查高低價）。
                                    'high_low': 停損停利會精確使用高低價觸發，入場使用開盤價。

    回傳：
        dict: 包含績效統計數據與策略時間序列。
    """

    if initial_capital <= 0:
        raise ValueError("initial_capital 必須大於 0")
    if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'signal', 'timestamp']):
        raise ValueError("df 必須包含 'open', 'high', 'low', 'close', 'signal', 'timestamp' 欄位。")

    df = df.copy().reset_index(drop=True)

    # 設定部位：1為多，-1為空(若允許放空)，0為空倉
    temp_position = pd.Series(np.nan, index=df.index)
    temp_position[df["signal"] == 1] = leverage
    temp_position[df["signal"] == -1] = -leverage if allow_short else 0
    df["position"] = temp_position.ffill().fillna(0)
    df.loc[0, "position"] = 0

    df["return"] = df["close"].pct_change().fillna(0)

    trade_returns = []
    hold_bars = []

    entry_price = None
    entry_index = None
    entry_position = 0
    
    # 每日策略報酬紀錄，初始化為0
    strategy_returns = np.zeros(len(df))

    # Initialize these variables before the loop to prevent UnboundLocalError
    should_exit = False
    exit_trade_return = 0

    # 為了實現'open'或'high_low'邏輯，需要將信號平移一個單位，因為決策是基於前一個K棒結束時的信號
    # 除非設定為 'close'，否則當前K棒的信號會在下一個K棒的開盤時執行
    df['next_open'] = df['open'].shift(-1)
    df['next_high'] = df['high'].shift(-1)
    df['next_low'] = df['low'].shift(-1)
    df['next_close'] = df['close'].shift(-1)
    df['prev_position'] = df['position'].shift(1).fillna(0)

    for i in range(len(df)):
        current_bar = df.loc[i]
        
        # 今天的實際開盤/高/低/收盤價 (用於計算當日報酬或出場價)
        # 如果是最後一根K棒，則用當前K棒的資料
        open_price_today = current_bar['open']
        high_price_today = current_bar['high']
        low_price_today = current_bar['low']
        close_price_today = current_bar['close']

        # 前一個K棒的信號決策導致今天的行為
        signal_at_prev_close = current_bar['prev_position'] # 前一個K棒結束時的預期部位
        current_position_target = current_bar['position'] # 當前K棒結束時的目標部位

        # 生成隨機滑點 (每次交易都會有)
        # 滑點方向與交易方向相反，例如買入時價格更高，賣出時價格更低
        buy_slippage = 1 + np.random.uniform(0, slippage_rate)
        sell_slippage = 1 - np.random.uniform(0, slippage_rate)

        # Reset for each iteration if not already exited
        should_exit = False
        exit_trade_return = 0

        # ====== 處理出場邏輯 (優先於入場) ======
        if entry_position != 0: # 如果目前有持倉
            holding_period = i - entry_index

            # 預設出場標誌
            exit_reason = None
            exit_price_actual = None # 實際的出場價格

            # 計算未實現報酬 (多空分別)
            # 這裡用K棒的收盤價作為參考點計算浮動損益，但觸發點會檢查高低價
            if entry_position > 0: # 多頭
                unrealized_ret_close = (close_price_today / entry_price) - 1
            else: # 空頭
                unrealized_ret_close = (entry_price / close_price_today) - 1
            unrealized_ret_close *= leverage

            # 檢查停損/停利 (使用K棒高低價來判斷是否觸發)
            if entry_position > 0: # 多頭
                # 檢查停損：如果當前K棒的最低價達到或跌破停損點
                # Note: / leverage here is incorrect for SL/TP price calculation, should be direct ratio of entry price
                # Corrected:
                if stop_loss is not None and low_price_today <= entry_price * (1 - stop_loss):
                    should_exit = True
                    exit_reason = "Stop Loss"
                    exit_price_actual = entry_price * (1 - stop_loss) # Assume executed at stop loss price
                # 檢查停利：如果當前K棒的最高價達到或超過停利點
                elif take_profit is not None and high_price_today >= entry_price * (1 + take_profit):
                    should_exit = True
                    exit_reason = "Take Profit"
                    exit_price_actual = entry_price * (1 + take_profit) # Assume executed at take profit price
            else: # 空頭
                # 檢查停損：如果當前K棒的最高價達到或超過停損點
                # Corrected:
                if stop_loss is not None and high_price_today >= entry_price * (1 + stop_loss):
                    should_exit = True
                    exit_reason = "Stop Loss"
                    exit_price_actual = entry_price * (1 + stop_loss) # Assume executed at stop loss price
                # 檢查停利：如果當前K棒的最低價達到或跌破停利點
                elif take_profit is not None and low_price_today <= entry_price * (1 - take_profit):
                    should_exit = True
                    exit_reason = "Take Profit"
                    exit_price_actual = entry_price * (1 - take_profit) # Assume executed at take profit price
            
            # 檢查最大持有時間
            if max_hold_bars is not None and holding_period >= max_hold_bars:
                should_exit = True
                exit_reason = "Max Hold Bars"
                # 無論何種情況，如果因為時間止損出場，則在當前K棒的收盤價出場
                exit_price_actual = close_price_today 
            
            # 檢查信號切換 (如果目標部位與當前持有部位不符)
            # This check should apply only if not already exited by SL/TP/MaxHold
            if not should_exit and current_position_target != entry_position:
                should_exit = True
                exit_reason = "Signal Change"
                # 假設在當前K棒的收盤價出場
                exit_price_actual = close_price_today


            if should_exit:
                # 應用出場手續費和滑點
                final_exit_price = exit_price_actual
                if entry_position > 0: # 多頭平倉 (賣出)
                    final_exit_price = final_exit_price * (1 - fee_rate) * sell_slippage
                else: # 空頭平倉 (買回)
                    final_exit_price = final_exit_price * (1 + fee_rate) * buy_slippage

                rtn = ((final_exit_price / entry_price) - 1 if entry_position > 0
                       else (entry_price / final_exit_price) - 1) * leverage

                trade_returns.append(rtn)
                hold_bars.append(holding_period)
                exit_trade_return = rtn # 記錄本次出場的實際報酬

                # 重置持倉狀態
                entry_price = None
                entry_index = None
                entry_position = 0

        # ====== 處理入場邏輯 (在出場之後，若有新信號且已空倉) ======
        # 如果當前已無持倉，且今天的目標部位與前一刻（signal_at_prev_close）不同，則考慮開新倉
        if entry_position == 0 and current_position_target != 0:
            
            trade_entry_price = open_price_today # 預設在當前K棒的開盤價入場
            if execution_price_type == 'close':
                trade_entry_price = close_price_today # 如果指定用收盤價入場

            # 應用入場手續費和滑點
            if current_position_target > 0: # 多頭開倉 (買入)
                entry_price_with_fees_slippage = trade_entry_price * (1 + fee_rate) * buy_slippage
            else: # 空頭開倉 (賣出)
                entry_price_with_fees_slippage = trade_entry_price * (1 - fee_rate) * sell_slippage
            
            entry_price = entry_price_with_fees_slippage
            entry_index = i
            entry_position = current_position_target # 記錄當前持倉方向與槓桿

            # 當天策略報酬：出場的真實報酬 + 新倉當天未計入的報酬 (設為0)
            strategy_returns[i] = exit_trade_return if should_exit else 0
            continue # 開倉當天不計算當日市場波動報酬

        # ====== 計算每日未出場的策略報酬 ======
        if entry_position != 0:
            # 持倉中，當日策略報酬為當日K棒報酬乘以部位槓桿
            # 注意：這裡使用當前K棒的'return'，這相當於(current_close - prev_close)/prev_close
            # 但如果'execution_price_type'是'open'，且入場是在當天開盤，則報酬應從開盤價計算
            if i > 0: # 確保不是第一根K棒
                # Calculate daily return based on open position's performance from previous close to current close
                daily_raw_return = (close_price_today / df.loc[i-1, 'close']) - 1
                strategy_returns[i] = daily_raw_return * entry_position
            else: # First bar, if a position starts immediately (unlikely with prev_position logic)
                strategy_returns[i] = 0 # Or handle specifically if first bar can have a position
        else:
            # 空倉時，策略報酬為0 (除非有出場行為)
            strategy_returns[i] = exit_trade_return # If there was an exit today but no new entry, use the exit return

    # 計算資產曲線
    df["strategy_return_with_fee"] = pd.Series(strategy_returns, index=df.index).fillna(0)
    df["equity"] = initial_capital * (1 + df["strategy_return_with_fee"]).cumprod()
    df["buy_and_hold"] = initial_capital * (1 + df["return"]).cumprod()

    # 最大回撤計算
    df["peak"] = df["equity"].cummax()
    df["drawdown"] = df["equity"] / df["peak"] - 1
    max_drawdown = df["drawdown"].min()

    # 總報酬率
    total_return = df["equity"].iloc[-1] / initial_capital - 1

    time_diff_days = (df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]).total_seconds() / (3600 * 24)
    daily_return = (1 + total_return) ** (1 / time_diff_days) - 1 if time_diff_days > 0 else 0

    # 計算交易次數、勝率、平均報酬等
    num_trades = len(trade_returns)
    win_rate = np.mean([r > 0 for r in trade_returns]) if num_trades else 0
    avg_profit = np.mean(trade_returns) if num_trades else 0

    if num_trades > 0:
        profits = [r for r in trade_returns if r > 0]
        losses = [r for r in trade_returns if r <= 0]
        avg_profit_win = np.mean(profits) if profits else 0
        avg_loss_loss = np.mean(losses) if losses else 0
        max_profit = np.max(trade_returns)
        max_loss = np.min(trade_returns)
    else:
        avg_profit_win = avg_loss_loss = max_profit = max_loss = 0

    return {
        "metric": {
            "回測K棒數量": int(len(df)),
            "總報酬率": f"{total_return * 100:.2f}",
            "日報酬率": f"{daily_return * 100:.4f}",
            "最大回撤": f"{max_drawdown * 100:.2f}",
            "交易次數": int(num_trades),
            "勝率": f"{win_rate * 100:.2f}",
            "平均持有K棒數": f"{np.mean(hold_bars):.2f}" if hold_bars else "0",
            "平均每筆報酬率": f"{avg_profit * 100:.2f}",
            "平均獲利時報酬": f"{avg_profit_win * 100:.2f}",
            "平均虧損時報酬": f"{avg_loss_loss * 100:.2f}",
            "最大單筆報酬": f"{max_profit * 100:.2f}",
            "最大單筆虧損": f"{max_loss * 100:.2f}",
        },
        "fig": {
            "timestamp": df["timestamp"].values,
            "equity": df["equity"].values,
            "buy_and_hold": df["buy_and_hold"].values.tolist(),
            "trade_returns": np.array(trade_returns) * 100,
            "close": df["close"].values,
            "signal": df["signal"].values,
            "position": df["position"].values,
        }
    }