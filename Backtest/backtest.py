import numpy as np
import pandas as pd

def backtest_signals(df: pd.DataFrame,
                                initial_capital=1000000,
                                fee_rate=0.000,
                                leverage=1, # Adjusted to match Pine Script's 10x margin
                                allow_short=True,
                                stop_loss=None,
                                take_profit=None,
                                max_hold_bars=None,
                                slippage_rate=0.0000,
                                capital_ratio=1,
                                maintenance_margin_ratio=0.005,
                                liquidation_penalty=1.0,
                                delay_entry=True):
    """
    Simulates a trading strategy with a "worst-case" scenario for stop-loss and take-profit.
    If both conditions are met within the same bar, the stop-loss is always triggered.
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing ['open', 'high', 'low', 'close', 'signal', 'timestamp'].
    - initial_capital (float): Starting capital for the backtest.
    - fee_rate (float): Transaction fee rate (e.g., 0.001 for 0.1%).
    - leverage (float): Leverage multiple.
    - allow_short (bool): Whether to allow short selling.
    - stop_loss (float): Stop-loss percentage.
    - take_profit (float): Take-profit percentage.
    - max_hold_bars (int): Maximum number of bars to hold a position.
    - slippage_rate (float): Slippage percentage.
    - capital_ratio (float): Percentage of capital to use per trade.
    - maintenance_margin_ratio (float): Margin call/liquidation ratio.
    - liquidation_penalty (float): Penalty for liquidation (1.0 means losing all used capital).
    - delay_entry (bool): If True, a signal triggers a trade on the next bar's open.
    
    Returns:
    - dict: A dictionary containing performance metrics, figure data, and a trade log.
    """

    required_cols = ['open', 'high', 'low', 'close', 'signal', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df 必須包含欄位: {required_cols}")
    if not (0 < capital_ratio <= 1):
        raise ValueError("capital_ratio 必須介於 0 與 1 之間")

    df = df.copy().reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    if delay_entry:
        # Use the position from get_signals, shifted by 1 to avoid lookahead
        # The 'position' column in the input df should be 1 (long), -1 (short), or 0 (flat)
        df['target_position'] = df['position'].shift(1).fillna(0)
    else:
        # Use the position from get_signals directly
        df['target_position'] = df['position'].fillna(0)

    # The 'position' column in the input df is now the source for target_position.
    # The previous line that re-derived 'df['position']' from 'used_signal' is removed.
    df.loc[0, 'target_position'] = 0 # Ensure first bar has no target position

    equity_curve = [initial_capital]
    trade_returns, hold_bars, trades_log = [], [], []
    cumulative_pnl_usdt = 0

    entry_price = None
    entry_index = None
    entry_position = 0
    current_equity = initial_capital

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        target_position = row['target_position']

        open_ = row['open']
        high_ = row['high']
        low_ = row['low']
        close_ = row['close']
        
        slip_pct = np.random.uniform(0, slippage_rate)
        buy_slip = 1 + slip_pct
        sell_slip = 1 - slip_pct

        should_exit = False
        exit_reason = None
        exit_price = None
        rtn = 0

        # ===== 出場判斷 (如果目前有部位) =====
        if entry_position != 0:
            holding_period = i - entry_index

            # Update max/min price during hold
            max_price_during_hold = max(max_price_during_hold, high_)
            min_price_during_hold = min(min_price_during_hold, low_)

            # --- 計算止盈/止損價格 ---
            sl_price_long = entry_price * (1 - stop_loss) if stop_loss is not None else None
            tp_price_long = entry_price * (1 + take_profit) if take_profit is not None else None
            sl_price_short = entry_price * (1 + stop_loss) if stop_loss is not None else None
            tp_price_short = entry_price * (1 - take_profit) if take_profit is not None else None
            
            # --- 檢查觸發條件 ---
            hit_sl = False
            hit_tp = False

            if entry_position > 0: # 多單
                if sl_price_long is not None and low_ <= sl_price_long:
                    hit_sl = True
                if tp_price_long is not None and high_ >= tp_price_long:
                    hit_tp = True
            else: # 空單
                if sl_price_short is not None and high_ >= sl_price_short:
                    hit_sl = True
                if tp_price_short is not None and low_ <= tp_price_short:
                    hit_tp = True
            
            # --- 決定出場價格 (最壞情況) ---
            if hit_sl and hit_tp:
                should_exit = True
                exit_reason = 'Stop Loss (Worst Case)'
                if entry_position > 0: # 多單
                    exit_price = sl_price_long * (1 - fee_rate) * sell_slip
                else: # 空單
                    exit_price = sl_price_short * (1 + fee_rate) * buy_slip
            elif hit_sl:
                should_exit = True
                exit_reason = 'Stop Loss'
                if entry_position > 0: # 多單
                    exit_price = sl_price_long * (1 - fee_rate) * sell_slip
                else: # 空單
                    exit_price = sl_price_short * (1 + fee_rate) * buy_slip
            elif hit_tp:
                should_exit = True
                exit_reason = 'Take Profit'
                if entry_position > 0: # 多單
                    exit_price = tp_price_long * (1 - fee_rate) * sell_slip
                else: # 空單
                    exit_price = tp_price_short * (1 + fee_rate) * buy_slip

            # --- 檢查其他出場條件 ---
            if not should_exit:
                if max_hold_bars is not None and holding_period >= max_hold_bars:
                    should_exit = True
                    exit_reason = 'Max Hold Bars'
                    exit_price = close_ * (1 - fee_rate) if entry_position > 0 else close_ * (1 + fee_rate)
                elif target_position != entry_position:
                    should_exit = True
                    exit_reason = 'Signal Change'
                    exit_price = close_ * (1 - fee_rate) if entry_position > 0 else close_ * (1 + fee_rate)

            # --- 執行出場 ---
            if should_exit:
                capital_used = current_equity * capital_ratio
                maintenance_margin = capital_used * maintenance_margin_ratio

                if entry_position > 0: # 多單
                    rtn = (exit_price / entry_price - 1) * leverage
                    run_up_pct = (max_price_during_hold / entry_price - 1) * leverage * 100
                    draw_down_pct = (min_price_during_hold / entry_price - 1) * leverage * 100
                else: # 空單
                    rtn = (entry_price / exit_price - 1) * leverage
                    run_up_pct = (entry_price / min_price_during_hold - 1) * leverage * 100
                    draw_down_pct = (entry_price / max_price_during_hold - 1) * leverage * 100

                profit = capital_used * rtn

                if current_equity + profit < maintenance_margin:
                    loss = capital_used * liquidation_penalty
                    current_equity -= loss
                    rtn = -liquidation_penalty
                    exit_reason = 'Liquidated'
                    run_up_pct = 0 # Reset if liquidated
                    draw_down_pct = -100 # Reset if liquidated
                else:
                    current_equity += profit

                trade_returns.append(rtn)
                hold_bars.append(holding_period)
                # Calculate P&L (USDT)
                pnl_usdt = capital_used * rtn
                cumulative_pnl_usdt += pnl_usdt # Update cumulative P&L
                cumulative_pnl_pct = (cumulative_pnl_usdt / initial_capital) * 100 # Calculate cumulative P&L %

                # Determine Type
                trade_type = 'Long' if entry_position > 0 else 'Short'

                # Determine Signal (Entry)
                signal_entry = '多單開倉' if entry_position > 0 else '空單開倉'

                # Determine Signal (Exit)
                signal_exit = ''
                if exit_reason == 'Liquidated' or exit_reason == 'Final Liquidated':
                    signal_exit = 'Margin call'
                elif entry_position > 0: # Long exit
                    signal_exit = '多單平倉'
                else: # Short exit
                    signal_exit = '空單平倉'

                # Calculate Position Size
                position_size = capital_used * leverage

                trades_log.append({
                    'Type': trade_type,
                    'Date/Time (Entry)': df.iloc[entry_index]['timestamp'].strftime('%Y/%m/%d, %H:%M'),
                    'Date/Time (Exit)': row['timestamp'].strftime('%Y/%m/%d, %H:%M'),
                    'Signal (Entry)': signal_entry,
                    'Signal (Exit)': signal_exit,
                    'Price (Entry)': f'{entry_price:,.2f}',
                    'Price (Exit)': f'{exit_price:,.2f}',
                    'Position size': f'{position_size:,.2f}',
                    'P&L (USDT)': f'{pnl_usdt:,.2f}',
                    'P&L (%)': f'{rtn * 100:,.2f}%',
                    'Run-up (%)': f'{run_up_pct:,.2f}%', # New
                    'Drawdown (%)': f'{draw_down_pct:,.2f}%', # New
                    'Cumulative P&L': f'{cumulative_pnl_usdt:,.2f}',
                    'Cumulative P&L (%)': f'{cumulative_pnl_pct:,.2f}%', # New
                })

                entry_price = None
                entry_index = None
                entry_position = 0
                max_price_during_hold = -np.inf # Reset for next trade
                min_price_during_hold = np.inf # Reset for next trade

        # ===== 進場判斷 =====
        if entry_position == 0 and target_position != 0:
            entry_price = close_ * (1 + fee_rate) * buy_slip if target_position > 0 else close_ * (1 - fee_rate) * sell_slip
            entry_index = i
            entry_position = target_position
            # Initialize for new trade
            max_price_during_hold = high_
            min_price_during_hold = low_

        equity_curve.append(current_equity)

    # If there's an open position at the end, log it as an open trade
    if entry_position != 0:
        # Calculate unrealized P&L for the open position
        current_price = df.iloc[-1]['close'] # Use close price for unrealized P&L
        unrealized_rtn = 0
        unrealized_pnl_usdt = 0
        unrealized_run_up_pct = 0
        unrealized_draw_down_pct = 0

        capital_used = current_equity * capital_ratio
        position_size = capital_used * leverage

        if entry_position > 0: # Long position
            unrealized_rtn = (current_price / entry_price - 1) * leverage
            unrealized_run_up_pct = (max_price_during_hold / entry_price - 1) * leverage * 100
            unrealized_draw_down_pct = (min_price_during_hold / entry_price - 1) * leverage * 100
        else: # Short position
            unrealized_rtn = (entry_price / current_price - 1) * leverage
            unrealized_run_up_pct = (entry_price / min_price_during_hold - 1) * leverage * 100
            unrealized_draw_down_pct = (entry_price / max_price_during_hold - 1) * leverage * 100
        
        unrealized_pnl_usdt = capital_used * unrealized_rtn

        trade_type = 'Long' if entry_position > 0 else 'Short'
        signal_entry = '多單開倉' if entry_position > 0 else '空單開倉'

        trades_log.append({
            'Type': trade_type,
            'Date/Time (Entry)': df.iloc[entry_index]['timestamp'].strftime('%Y/%m/%d, %H:%M'),
            'Date/Time (Exit)': 'Open', # Indicate open position
            'Signal (Entry)': signal_entry,
            'Signal (Exit)': 'Open', # Indicate open position
            'Price (Entry)': f'{entry_price:,.2f}',
            'Price (Exit)': f'{current_price:,.2f}', # Current price for open position
            'Position size': f'{position_size:,.2f}',
            'P&L (USDT)': f'{unrealized_pnl_usdt:,.2f}', # Unrealized P&L
            'P&L (%)': f'{unrealized_rtn * 100:,.2f}%', # Unrealized P&L %
            'Run-up (%)': f'{unrealized_run_up_pct:,.2f}%',
            'Drawdown (%)': f'{unrealized_draw_down_pct:,.2f}%',
            'Cumulative P&L': f'{cumulative_pnl_usdt:,.2f}', # Cumulative P&L up to this point
            'Cumulative P&L (%)': f'{cumulative_pnl_pct:,.2f}%', # Cumulative P&L % up to this point
        })

    if len(equity_curve) < len(df):
        equity_curve += [equity_curve[-1]] * (len(df) - len(equity_curve))

    df['equity'] = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    df['buy_and_hold'] = initial_capital * (1 + df['close'].pct_change().fillna(0)).cumprod()
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1

    final_equity = df['equity'].iloc[-1]
    if pd.isna(final_equity) or initial_capital <= 0:
        total_return = 0
    else:
        total_return = final_equity / initial_capital - 1

    time_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    if time_days <= 0:
        time_days = max(1, len(df) / 24)

    if pd.isna(final_equity) or initial_capital <= 0:
        daily_return = 0
    else:
        daily_return = (final_equity / initial_capital) ** (1 / time_days) - 1

    max_dd = df['drawdown'].min()
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    
    # Calculate metrics for USDT
    trade_pnls_usdt = [float(trade['P&L (USDT)'].replace(',', '')) for trade in trades_log if 'Open' not in trade['Date/Time (Exit)'] ]
    net_profit_usdt = sum(trade_pnls_usdt)
    gross_profit_usdt = sum(pnl for pnl in trade_pnls_usdt if pnl > 0)
    gross_loss_usdt = sum(pnl for pnl in trade_pnls_usdt if pnl <= 0)
    
    if abs(gross_loss_usdt) < 1e-9: # Avoid division by zero
        profit_factor = np.inf
    else:
        profit_factor = gross_profit_usdt / abs(gross_loss_usdt)

    buy_and_hold_return = (df['buy_and_hold'].iloc[-1] / initial_capital - 1) * 100

    return {
        'metric': {
            'Net Profit (%)': f'{(net_profit_usdt / initial_capital) * 100:,.2f}%',
            'Total Closed Trades': len(trade_returns),
            'Percent Profitable': f'{(np.mean([r > 0 for r in trade_returns]) * 100):.2f}%' if trade_returns else '0.00%',
            'Profit Factor': f'{profit_factor:.2f}',
            'Max Drawdown': f'{max_dd * 100:.2f}%',
            'Avg Trade': f'{np.mean(trade_returns) * 100:.2f}%' if trade_returns else '0.00%',
            'Avg Win Trade': f'{np.mean(wins) * 100:.2f}%' if wins else '0.00%',
            'Avg Loss Trade': f'{np.mean(losses) * 100:.2f}%' if losses else '0.00%',
            'Avg Bars in Trade': f'{np.mean(hold_bars):.2f}' if hold_bars else '0.00',
            'Buy & Hold Return': f'{buy_and_hold_return:.2f}%',
        },
        'fig': {
            'timestamp': df['timestamp'].tolist(),
            'equity': df['equity'].tolist(),
            'buy_and_hold': df['buy_and_hold'].tolist(),
            'close': df['close'].tolist(),
            'signal': df['signal'].tolist(),
            'position': df['position'].tolist(),
            'trade_returns': np.array(trade_returns) * 100,
        },
        'trades_log': trades_log
    }