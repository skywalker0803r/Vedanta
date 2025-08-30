import numpy as np
import pandas as pd
import warnings

def round_price(price, precision=8):
    """
    Round price to specified precision to avoid floating point errors
    
    Args:
        price (float): Price to round
        precision (int): Number of decimal places (default: 8)
    
    Returns:
        float: Rounded price
    """
    return round(price, precision)

def determine_exit_order(open_, high_, low_, close_, sl_price, tp_price, is_long):
    """
    根據OHLC順序判斷止損止盈觸發順序
    假設順序：Open → (High/Low依市場方向) → Close
    
    Args:
        open_, high_, low_, close_: OHLC價格
        sl_price, tp_price: 止損止盈價格
        is_long: 是否為多單
    
    Returns:
        tuple: (exit_type, exit_price) 或 (None, None)
    """
    if sl_price is None and tp_price is None:
        return None, None
    
    if is_long:
        # 多單邏輯
        sl_hit = sl_price is not None and low_ <= sl_price
        tp_hit = tp_price is not None and high_ >= tp_price
        
        if sl_hit and tp_hit:
            # 兩個都觸發，判斷順序
            # 如果開盤價已經突破止損，"優先止損"
            if open_ <= sl_price:
                return 'stop_loss', sl_price
            # 如果開盤價已經達到止盈，優先止盈
            elif open_ >= tp_price:
                return 'take_profit', tp_price
            else:
                # 根據市場方向判斷：下跌市場先觸及止損
                if close_ < open_:  # 下跌K線
                    return 'stop_loss', sl_price
                else:  # 上漲K線
                    return 'take_profit', tp_price
        elif sl_hit:
            return 'stop_loss', sl_price
        elif tp_hit:
            return 'take_profit', tp_price
    else:
        # 空單邏輯
        sl_hit = sl_price is not None and high_ >= sl_price
        tp_hit = tp_price is not None and low_ <= tp_price
        
        if sl_hit and tp_hit:
            # 兩個都觸發，判斷順序
            if open_ >= sl_price:
                return 'stop_loss', sl_price
            elif open_ <= tp_price:
                return 'take_profit', tp_price
            else:
                # 根據市場方向判斷：上漲市場先觸及止損
                if close_ > open_:  # 上漲K線
                    return 'stop_loss', sl_price
                else:  # 下跌K線
                    return 'take_profit', tp_price
        elif sl_hit:
            return 'stop_loss', sl_price
        elif tp_hit:
            return 'take_profit', tp_price
    
    return None, None

def backtest_signals(df: pd.DataFrame,
                     initial_capital=1000000,
                     fee_rate=0.000,
                     leverage=1,
                     allow_short=True,
                     stop_loss=None,
                     take_profit=None,
                     max_hold_bars=None,
                     slippage_rate=0.0000,
                     capital_ratio=1,
                     maintenance_margin_ratio=0.005,
                     liquidation_penalty=1.0,
                     risk_free_rate=0.02,
                     interval: str = '',
                     price_precision=8):
    """
    改進的回測函數，與TradingView邏輯對齊
    
    主要改進：
    1. 移除delay_entry參數，統一使用標準執行邏輯
    2. 改進止損止盈觸發順序判斷
    3. 修正手續費計算方式
    
    Parameters:
    - df (pd.DataFrame): DataFrame containing ['open', 'high', 'low', 'close', 'signal', 'timestamp', 'position'].
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
    - risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).
    - interval (str): Data interval (e.g., '1h' for hourly). If provided, uses predefined periods_per_year; else infers from timestamps.
      Supported: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d'.
    - price_precision (int): Number of decimal places for price rounding (default: 8).
    
    Returns:
    - dict: A dictionary containing performance metrics, figure data, and a trade log.
    """

    required_cols = ['open', 'high', 'low', 'close', 'signal', 'timestamp', 'position']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df 必需包含欄位: {required_cols}")
    if not (0 < capital_ratio <= 1):
        raise ValueError("capital_ratio 必需介於 0 與 1 之間")

    df = df.copy().reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 修正1: 移除delay_entry，統一邏輯
    # position欄位代表當根K線收盤後產生的訊號，下一根K線開盤執行
    df['target_position'] = df['position'].fillna(0)

    if not allow_short:
        df.loc[df['target_position'] == -1, 'target_position'] = 0

    df.loc[0, 'target_position'] = 0  # 確保第一筆無部位

    equity_curve = [initial_capital]
    trade_returns, hold_bars, trades_log = [], [], []
    cumulative_pnl_usdt = 0

    entry_price = None
    entry_index = None
    entry_position = 0
    current_equity = initial_capital
    max_price_during_hold = -np.inf
    min_price_during_hold = np.inf
    total_fees_paid = 0

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i-1]
        target_position = prev_row['target_position']  # 使用前一根的訊號

        open_ = row['open']
        high_ = row['high']
        low_ = row['low']
        close_ = row['close']
        
        # 滑點計算
        slip_pct = np.random.uniform(0, slippage_rate)
        buy_slip = 1 + slip_pct
        sell_slip = 1 - slip_pct

        should_exit = False
        exit_reason = None
        exit_price = None
        exit_fees = 0
        rtn = 0

        # ===== 出場判斷 (如果目前有部位) =====
        if entry_position != 0:
            holding_period = i - entry_index

            # 更新 max/min 價格
            max_price_during_hold = max(max_price_during_hold, high_)
            min_price_during_hold = min(min_price_during_hold, low_)

            # --- 計算止盈/止損價格 ---
            sl_price_long = entry_price * (1 - stop_loss) if stop_loss is not None else None
            tp_price_long = entry_price * (1 + take_profit) if take_profit is not None else None
            sl_price_short = entry_price * (1 + stop_loss) if stop_loss is not None else None
            tp_price_short = entry_price * (1 - take_profit) if take_profit is not None else None
            
            # 修正2: 使用改進的觸發順序判斷
            if entry_position > 0:  # 多單
                exit_type, triggered_price = determine_exit_order(
                    open_, high_, low_, close_, sl_price_long, tp_price_long, True
                )
            else:  # 空單
                exit_type, triggered_price = determine_exit_order(
                    open_, high_, low_, close_, sl_price_short, tp_price_short, False
                )
            
            if exit_type is not None:
                should_exit = True
                exit_reason = 'Stop Loss' if exit_type == 'stop_loss' else 'Take Profit'
                # 修正3: 手續費分離計算
                if entry_position > 0:
                    exit_price = round_price(triggered_price * sell_slip, price_precision)
                else:
                    exit_price = round_price(triggered_price * buy_slip, price_precision)

            # --- 檢查其他出場條件 ---
            if not should_exit:
                if max_hold_bars is not None and holding_period >= max_hold_bars:
                    should_exit = True
                    exit_reason = 'Max Hold Bars'
                    # 使用下一根開盤價出場（如果有下一根的話）
                    if i + 1 < len(df):
                        next_open = df.iloc[i + 1]['open']
                        exit_price = round_price(next_open * sell_slip, price_precision) if entry_position > 0 else round_price(next_open * buy_slip, price_precision)
                    else:
                        # 如果沒有下一根，使用當前收盤價
                        exit_price = round_price(close_ * sell_slip, price_precision) if entry_position > 0 else round_price(close_ * buy_slip, price_precision)
                elif target_position != entry_position:
                    should_exit = True
                    exit_reason = 'Signal Change'
                    # 訊號變化時使用當前開盤價出場
                    exit_price = round_price(open_ * sell_slip, price_precision) if entry_position > 0 else round_price(open_ * buy_slip, price_precision)

            # --- 執行出場 ---
            if should_exit:
                capital_used = current_equity * capital_ratio
                position_value = capital_used * leverage
                
                # 修正3: 分離手續費計算
                exit_fees = position_value * fee_rate
                
                maintenance_margin = capital_used * maintenance_margin_ratio

                if entry_position > 0:  # 多單
                    gross_rtn = (exit_price / entry_price - 1) * leverage
                    run_up_pct = (max_price_during_hold / entry_price - 1) * leverage * 100
                    draw_down_pct = (min_price_during_hold / entry_price - 1) * leverage * 100
                else:  # 空單
                    gross_rtn = (entry_price / exit_price - 1) * leverage
                    run_up_pct = (entry_price / min_price_during_hold - 1) * leverage * 100
                    draw_down_pct = (entry_price / max_price_during_hold - 1) * leverage * 100

                # 計算淨收益（扣除進場和出場手續費）
                entry_fees = position_value * fee_rate  # 進場手續費
                total_trade_fees = entry_fees + exit_fees
                gross_profit = capital_used * gross_rtn
                net_profit = gross_profit - total_trade_fees
                rtn = net_profit / capital_used

                # 檢查是否觸發強制平倉
                if current_equity + net_profit < maintenance_margin:
                    loss = capital_used * liquidation_penalty
                    current_equity -= loss
                    current_equity = max(0, current_equity)  # 避免負值
                    rtn = -liquidation_penalty
                    exit_reason = 'Liquidated'
                    run_up_pct = 0
                    draw_down_pct = -100
                    net_profit = loss * -1
                else:
                    current_equity += net_profit

                total_fees_paid += total_trade_fees
                trade_returns.append(rtn)
                hold_bars.append(holding_period)
                pnl_usdt = net_profit
                cumulative_pnl_usdt += pnl_usdt
                cumulative_pnl_pct = (cumulative_pnl_usdt / initial_capital) * 100

                trade_type = 'Long' if entry_position > 0 else 'Short'
                signal_entry = '多單開倉' if entry_position > 0 else '空單開倉'
                signal_exit = 'Margin call' if 'Liquidated' in exit_reason else '多單平倉' if entry_position > 0 else '空單平倉'

                trades_log.append({
                    'Type': trade_type,
                    'Date/Time (Entry)': df.iloc[entry_index]['timestamp'].strftime('%Y/%m/%d, %H:%M'),
                    'Date/Time (Exit)': row['timestamp'].strftime('%Y/%m/%d, %H:%M'),
                    'Signal (Entry)': signal_entry,
                    'Signal (Exit)': signal_exit,
                    'Price (Entry)': f'{entry_price:,.2f}',
                    'Price (Exit)': f'{exit_price:,.2f}',
                    'Position size': f'{position_value:,.2f}',
                    'P&L (USDT)': f'{pnl_usdt:,.2f}',
                    'P&L (%)': f'{rtn * 100:,.2f}%',
                    'Gross P&L': f'{gross_profit:,.2f}',
                    'Fees': f'{total_trade_fees:,.2f}',
                    'Run-up (%)': f'{run_up_pct:,.2f}%', 
                    'Drawdown (%)': f'{draw_down_pct:,.2f}%', 
                    'Cumulative P&L': f'{cumulative_pnl_usdt:,.2f}',
                    'Cumulative P&L (%)': f'{cumulative_pnl_pct:,.2f}%',
                })

                # 重置部位狀態
                entry_price = None
                entry_index = None
                entry_position = 0
                max_price_during_hold = -np.inf
                min_price_during_hold = np.inf

        # ===== 進場判斷 =====
        if entry_position == 0 and target_position != 0:
            # 使用當前開盤價進場
            capital_used = current_equity * capital_ratio
            position_value = capital_used * leverage
            
            # 修正3: 分離手續費計算
            entry_fees = position_value * fee_rate
            
            if target_position > 0:
                entry_price = round_price(open_ * buy_slip, price_precision)
            else:
                entry_price = round_price(open_ * sell_slip, price_precision)
            
            entry_index = i
            entry_position = target_position
            max_price_during_hold = high_
            min_price_during_hold = low_
            
            # 扣除進場手續費（但不影響entry_price）
            current_equity -= entry_fees
            total_fees_paid += entry_fees

        equity_curve.append(current_equity)

    # 處理最後開放部位
    if entry_position != 0:
        current_price = df.iloc[-1]['close']
        unrealized_rtn = 0
        unrealized_pnl_usdt = 0
        unrealized_run_up_pct = 0
        unrealized_draw_down_pct = 0

        capital_used = (current_equity + total_fees_paid) * capital_ratio  # 回推原始資金
        position_value = capital_used * leverage

        if entry_position > 0:
            gross_unrealized_rtn = (current_price / entry_price - 1) * leverage
            unrealized_run_up_pct = (max_price_during_hold / entry_price - 1) * leverage * 100
            unrealized_draw_down_pct = (min_price_during_hold / entry_price - 1) * leverage * 100
        else:
            gross_unrealized_rtn = (entry_price / current_price - 1) * leverage
            unrealized_run_up_pct = (entry_price / min_price_during_hold - 1) * leverage * 100
            unrealized_draw_down_pct = (entry_price / max_price_during_hold - 1) * leverage * 100
        
        # 計算未實現盈虧（扣除潛在出場手續費）
        gross_unrealized_pnl = capital_used * gross_unrealized_rtn
        exit_fees = position_value * fee_rate
        unrealized_pnl_usdt = gross_unrealized_pnl - exit_fees
        unrealized_rtn = unrealized_pnl_usdt / capital_used

        trade_type = 'Long' if entry_position > 0 else 'Short'
        signal_entry = '多單開倉' if entry_position > 0 else '空單開倉'

        trades_log.append({
            'Type': trade_type,
            'Date/Time (Entry)': df.iloc[entry_index]['timestamp'].strftime('%Y/%m/%d, %H:%M'),
            'Date/Time (Exit)': 'Open',
            'Signal (Entry)': signal_entry,
            'Signal (Exit)': 'Open',
            'Price (Entry)': f'{entry_price:,.2f}',
            'Price (Exit)': f'{current_price:,.2f}',
            'Position size': f'{position_value:,.2f}',
            'P&L (USDT)': f'{unrealized_pnl_usdt:,.2f}',
            'P&L (%)': f'{unrealized_rtn * 100:,.2f}%',
            'Gross P&L': f'{gross_unrealized_pnl:,.2f}',
            'Fees': f'{exit_fees:,.2f}',
            'Run-up (%)': f'{unrealized_run_up_pct:,.2f}%',
            'Drawdown (%)': f'{unrealized_draw_down_pct:,.2f}%',
            'Cumulative P&L': f'{cumulative_pnl_usdt:,.2f}',
            'Cumulative P&L (%)': f'{cumulative_pnl_pct:,.2f}%',
        })

    # 補齊equity_curve長度
    if len(equity_curve) < len(df):
        equity_curve += [equity_curve[-1]] * (len(df) - len(equity_curve))

    df['equity'] = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    df['buy_and_hold'] = initial_capital * (1 + df['close'].pct_change().fillna(0)).cumprod()
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1

    final_equity = df['equity'].iloc[-1]
    total_return = final_equity / initial_capital - 1 if initial_capital > 0 else 0

    time_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    if time_days <= 0:
        time_days = max(1, len(df) / 24)

    daily_return = (final_equity / initial_capital) ** (1 / time_days) - 1 if initial_capital > 0 else 0

    max_dd = df['drawdown'].min()
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    
    trade_pnls_usdt = [float(trade['P&L (USDT)'].replace(',', '')) for trade in trades_log if 'Open' not in trade['Date/Time (Exit)']]
    net_profit_usdt = sum(trade_pnls_usdt)
    gross_profit_usdt = sum(pnl for pnl in trade_pnls_usdt if pnl > 0)
    gross_loss_usdt = sum(pnl for pnl in trade_pnls_usdt if pnl <= 0)
    
    profit_factor = np.inf if abs(gross_loss_usdt) < 1e-9 else gross_profit_usdt / abs(gross_loss_usdt)

    buy_and_hold_return = (df['buy_and_hold'].iloc[-1] / initial_capital - 1) * 100

    # ----- 計算 Sharpe 和 Sortino (改用簡單收益率) -----
    equity_returns = df['equity'].pct_change().dropna()
    
    # 間隔字典
    interval_to_periods = {
        '1m': 365 * 24 * 60,   # 525600
        '5m': 365 * 24 * 12,   # 105120
        '15m': 365 * 24 * 4,   # 35040
        '30m': 365 * 24 * 2,   # 17520
        '1h': 365 * 24,        # 8760
        '2h': 365 * 12,        # 4380
        '4h': 365 * 6,         # 2190
        '8h': 365 * 3,         # 1095
        '12h': 365 * 2,        # 730
        '1d': 365              # 365
    }

    # 決定 periods_per_year
    if interval and interval in interval_to_periods:
        periods_per_year = interval_to_periods[interval]
    else:
        # 從 timestamp 推斷（使用中位數）
        try:
            avg_seconds = df['timestamp'].diff().dt.total_seconds().dropna().median()
            if np.isnan(avg_seconds) or avg_seconds <= 0:
                periods_per_year = 365.0  # 回退每日
            else:
                periods_per_year = (365.0 * 24.0 * 3600.0) / avg_seconds
        except Exception:
            periods_per_year = 365.0

    # 處理空回報
    if equity_returns.empty:
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    else:
        mean_return = equity_returns.mean()
        std_return = equity_returns.std(ddof=1)

        # 年化指標
        annualized_return = (1 + mean_return) ** periods_per_year - 1
        annualized_volatility = std_return * np.sqrt(periods_per_year)

        # Sharpe ratio
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = float('inf') if annualized_return > risk_free_rate else float('-inf') if annualized_return < risk_free_rate else 0.0

        # Sortino ratio
        rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        downside_returns = equity_returns[equity_returns < rf_per_period]
        if len(downside_returns) > 1:
            downside_std = downside_returns.std(ddof=1)
            downside_deviation = downside_std * np.sqrt(periods_per_year)
        else:
            downside_deviation = 0.0
            
        if downside_deviation > 0:
            sortino_ratio = (annualized_return - risk_free_rate) / downside_deviation
        else:
            sortino_ratio = float('inf') if annualized_return > risk_free_rate else float('-inf') if annualized_return < risk_free_rate else 0.0
    
    # Calmar Ratio
    calmar_ratio = annualized_return/(-1*float(max_dd)) if max_dd < 0 else float('inf')
    
    # Expectancy 等其他指標
    if trade_returns:
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        win_rate = len(wins) / len(trade_returns)
        loss_rate = len(losses) / len(trade_returns)
        expectancy = (avg_win * win_rate) - (abs(avg_loss) * loss_rate)
    else:
        expectancy = 0

    largest_win = max(wins) * 100 if wins else 0
    largest_loss = min(losses) * 100 if losses else 0

    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_consecutive_wins = 0
    current_consecutive_losses = 0

    for r in trade_returns:
        if r > 0:
            current_consecutive_wins += 1
            current_consecutive_losses = 0
            max_consecutive_wins = max(max_consecutive_wins, current_consecutive_wins)
        else:
            current_consecutive_losses += 1
            current_consecutive_wins = 0
            max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)

    return {
        'Overview performance': {
            'Total P&L': f'{(net_profit_usdt / initial_capital) * 100:,.2f}%',
            'Max Drawdown': f'{max_dd * 100:.2f}%',
            'Total Trades': len(trade_returns),
            'Percent Profitable': f'{(np.mean([r > 0 for r in trade_returns]) * 100):.2f}%' if trade_returns else '0.00%',
            'Profit Factor': f'{profit_factor:.2f}',
            'Expectancy': f'{expectancy: .4f}',
            'Total Fees Paid': f'{total_fees_paid:,.2f}'
        },
        'Trades analysis': {
            'Total Trades': len(trade_returns),
            'Number of Winning Trades': len(wins),
            'Number of Losing Trades': len(losses),
            'Average Trade (%)': f'{np.mean(trade_returns) * 100:.2f}%' if trade_returns else '0.00%',
            'Average Win (%)': f'{np.mean(wins) * 100:.2f}%' if wins else '0.00%',
            'Average Loss (%)': f'{np.mean(losses) * 100:.2f}%' if losses else '0.00%',
            'Largest Win (%)': f'{largest_win:.2f}%',
            'Largest Loss (%)': f'{largest_loss:.2f}%',
        },
        'Risk/performance ratios': {
            'Sharpe Ratio': f'{sharpe_ratio:.2f}',
            'Sortino Ratio': f'{sortino_ratio:.2f}',
            'Calmar Ratio' : f'{calmar_ratio:.2f}',
            'Profit Factor': f'{profit_factor:.2f}',
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
        'trades_log': trades_log,
        'float_type_metrics': {  
            'Expectancy': float(expectancy) if expectancy is not None else 0.0000,
            'Max Drawdown': float(max_dd) if max_dd is not None else 0.0000,
            'Sharpe Ratio': float(sharpe_ratio) if sharpe_ratio is not None else 0.00,
            'Sortino Ratio': float(sortino_ratio) if sortino_ratio is not None else 0.00,
            'Calmar Ratio' : float(calmar_ratio) if calmar_ratio is not None else 0.00,
            'Total Fees': float(total_fees_paid),
        }
    }
