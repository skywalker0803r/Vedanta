import numpy as np
import pandas as pd
import warnings
from datetime import timedelta

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

def backtest_signals(df: pd.DataFrame,
                     initial_capital=1000000,
                     fee_rate=0.000,
                     leverage=1,
                     allow_short=True,
                     stop_loss=None,
                     take_profit=None,
                     max_hold_bars=None,
                     slippage_rate=0.0000,
                     capital_ratio = 1, # 每次使用的資金佔比
                     long_capital_ratio=1.0,  # 新增：多單倉位比例
                     short_capital_ratio=1.0, # 新增：空單倉位比例
                     maintenance_margin_ratio=0.005,
                     liquidation_penalty=1.0,
                     delay_entry=True,
                     risk_free_rate=0.02,
                     interval: str = '',
                     price_precision=8):
    """
    Simulates a trading strategy with a "worst-case" scenario for stop-loss and take-profit.
    If both conditions are met within the same bar, the stop-loss is always triggered.
    
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
    - long_capital_ratio (float): Percentage of capital to use for long positions.
    - short_capital_ratio (float): Percentage of capital to use for short positions.
    - maintenance_margin_ratio (float): Margin call/liquidation ratio.
    - liquidation_penalty (float): Penalty for liquidation (1.0 means losing all used capital).
    - delay_entry (bool): If True, a signal triggers a trade on the next bar's open.
    - risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%).
    - interval (str): Data interval (e.g., '1h' for hourly). If provided, uses predefined periods_per_year; else infers from timestamps.
      Supported: '1m', '5m', '15m', '30m', '1h', '2h', '4h', '8h', '12h', '1d'.
    - price_precision (int): Number of decimal places for price rounding (default: 8).
    
    Returns:
    - dict: A dictionary containing performance metrics, figure data, and a trade log.
    """

    required_cols = ['open', 'high', 'low', 'close', 'signal', 'timestamp', 'position']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df 必須包含欄位: {required_cols}")
    if not (0 < long_capital_ratio <= 1) or not (0 < short_capital_ratio <= 1):
        raise ValueError("long_capital_ratio 和 short_capital_ratio 必須介於 0 與 1 之間")

    df = df.copy().reset_index(drop=True)

    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # === 新增: 計算 K 棒間隔 ===
    if len(df) > 1:
        bar_interval = df['timestamp'].iloc[1] - df['timestamp'].iloc[0]
    else:
        bar_interval = timedelta(hours=4) # 預設為 4 小時

    if delay_entry:
        df['target_position'] = df['position'].shift(1).fillna(0)
    else:
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

    for i in range(1, len(df)):
        row = df.iloc[i]
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
        capital_used = 0

        # ===== 出場判斷 (如果目前有部位) =====
        if entry_position != 0:
            holding_period = i - entry_index

            # 更新 max/min 價格
            max_price_during_hold = max(max_price_during_hold, high_)
            min_price_during_hold = min(min_price_during_hold, low_)

            # --- 檢查策略預設的出場信號 ---
            if pd.notna(row.get('exit_price')) and row.get('exit_reason'):
                should_exit = True
                exit_reason = row['exit_reason']
                exit_price = row['exit_price']
            else:
                # --- 計算止盈/止損價格 (如果策略沒有預設) ---
                sl_price_long = row.get('stop_loss_level') if pd.notna(row.get('stop_loss_level')) else (entry_price * (1 - stop_loss) if stop_loss is not None else None)
                tp_price_long = row.get('take_profit_level') if pd.notna(row.get('take_profit_level')) else (entry_price * (1 + take_profit) if take_profit is not None else None)
                sl_price_short = row.get('stop_loss_level') if pd.notna(row.get('stop_loss_level')) else (entry_price * (1 + stop_loss) if stop_loss is not None else None)
                tp_price_short = row.get('take_profit_level') if pd.notna(row.get('take_profit_level')) else (entry_price * (1 - take_profit) if take_profit is not None else None)
                
                # 處理移動止損 (如果策略有提供)
                if entry_position == -1 and pd.notna(row.get('trailing_stop_level')):
                    sl_price_short = row['trailing_stop_level']

                # --- 檢查觸發條件 ---
                hit_sl = False
                hit_tp = False

                if entry_position > 0:  # 多單
                    if sl_price_long is not None and low_ <= sl_price_long:
                        hit_sl = True
                    if tp_price_long is not None and high_ >= tp_price_long:
                        hit_tp = True
                else:  # 空單
                    if sl_price_short is not None and high_ >= sl_price_short:
                        hit_sl = True
                    if tp_price_short is not None and low_ <= tp_price_short:
                        hit_tp = True
                
                # --- 決定出場價格 (最壞情況) ---
                if hit_sl and hit_tp:
                    should_exit = True
                    exit_reason = 'Stop Loss (Worst Case)'
                    if entry_position > 0:
                        exit_price = round_price(sl_price_long * (1 - fee_rate) * sell_slip, price_precision)
                    else:
                        exit_price = round_price(sl_price_short * (1 + fee_rate) * buy_slip, price_precision)
                elif hit_sl:
                    should_exit = True
                    exit_reason = 'Stop Loss'
                    if entry_position > 0:
                        exit_price = round_price(sl_price_long * (1 - fee_rate) * sell_slip, price_precision)
                    else:
                        exit_price = round_price(sl_price_short * (1 + fee_rate) * buy_slip, price_precision)
                elif hit_tp:
                    should_exit = True
                    exit_reason = 'Take Profit'
                    if entry_position > 0:
                        exit_price = round_price(tp_price_long * (1 - fee_rate) * sell_slip, price_precision)
                    else:
                        exit_price = round_price(tp_price_short * (1 + fee_rate) * buy_slip, price_precision)

            # --- 檢查其他出場條件 ---
            if not should_exit:
                if max_hold_bars is not None and holding_period >= max_hold_bars:
                    should_exit = True
                    exit_reason = 'Max Hold Bars'
                    exit_price = round_price(close_ * (1 - fee_rate) * sell_slip, price_precision) if entry_position > 0 else round_price(close_ * (1 + fee_rate) * buy_slip, price_precision)
                elif target_position != entry_position:
                    should_exit = True
                    exit_reason = 'Signal Change'
                    exit_price = round_price(close_ * (1 - fee_rate) * sell_slip, price_precision) if entry_position > 0 else round_price(close_ * (1 + fee_rate) * buy_slip, price_precision)

            # --- 執行出場 ---
            if should_exit:
                capital_used = current_equity * long_capital_ratio if entry_position > 0 else current_equity * short_capital_ratio
                maintenance_margin = capital_used * maintenance_margin_ratio

                if entry_position > 0:  # 多單
                    rtn = (exit_price / entry_price - 1) * leverage
                    run_up_pct = (max_price_during_hold / entry_price - 1) * leverage * 100
                    draw_down_pct = (min_price_during_hold / entry_price - 1) * leverage * 100
                else:  # 空單
                    rtn = (entry_price / exit_price - 1) * leverage
                    run_up_pct = (entry_price / min_price_during_hold - 1) * leverage * 100
                    draw_down_pct = (entry_price / max_price_during_hold - 1) * leverage * 100

                profit = capital_used * rtn

                if current_equity + profit < maintenance_margin:
                    loss = capital_used * liquidation_penalty
                    current_equity -= loss
                    current_equity = max(0, current_equity)  # 避免負值
                    rtn = -liquidation_penalty
                    exit_reason = 'Liquidated'
                    run_up_pct = 0
                    draw_down_pct = -100
                else:
                    current_equity += profit

                trade_returns.append(rtn)
                hold_bars.append(holding_period)
                pnl_usdt = capital_used * rtn
                cumulative_pnl_usdt += pnl_usdt
                cumulative_pnl_pct = (cumulative_pnl_usdt / initial_capital) * 100

                trade_type = 'Long' if entry_position > 0 else 'Short'
                signal_entry = '多單開倉' if entry_position > 0 else '空單開倉'
                signal_exit = 'Margin call' if 'Liquidated' in exit_reason else '多單平倉' if entry_position > 0 else '空單平倉'
                position_size = capital_used * leverage

                # === 修正: 根據 delay_entry 調整進場時間 ===
                entry_timestamp = df.iloc[entry_index]['timestamp']
                if not delay_entry:
                    entry_timestamp += bar_interval # 立即進場則使用 K 棒收盤時間

                trades_log.append({
                    'Type': trade_type,
                    'Date/Time (Entry)': entry_timestamp.strftime('%Y/%m/%d, %H:%M'),
                    'Date/Time (Exit)': row['timestamp'].strftime('%Y/%m/%d, %H:%M'),
                    'Signal (Entry)': signal_entry,
                    'Signal (Exit)': signal_exit,
                    'Price (Entry)': f'{entry_price:,.2f}',
                    'Price (Exit)': f'{exit_price:,.2f}',
                    'Position size': f'{position_size:,.2f}',
                    'P&L (USDT)': f'{pnl_usdt:,.2f}',
                    'P&L (%)': f'{rtn * 100:,.2f}%',
                    'Run-up (%)': f'{run_up_pct:,.2f}%', 
                    'Drawdown (%)': f'{draw_down_pct:,.2f}%', 
                    'Cumulative P&L': f'{cumulative_pnl_usdt:,.2f}',
                    'Cumulative P&L (%)': f'{cumulative_pnl_pct:,.2f}%',
                })

                entry_price = None
                entry_index = None
                entry_position = 0
                max_price_during_hold = -np.inf
                min_price_during_hold = np.inf

        # ===== 進場判斷 =====
        if entry_position == 0 and target_position != 0:
            entry_price = round_price(close_ * (1 + fee_rate) * buy_slip, price_precision) if target_position > 0 else round_price(close_ * (1 - fee_rate) * sell_slip, price_precision)
            entry_index = i
            entry_position = target_position
            max_price_during_hold = high_
            min_price_during_hold = low_

        equity_curve.append(current_equity)

    # 處理最後開放部位
    if entry_position != 0:
        current_price = df.iloc[-1]['close']
        unrealized_rtn = 0
        unrealized_pnl_usdt = 0
        unrealized_run_up_pct = 0
        unrealized_draw_down_pct = 0

        capital_used = current_equity * long_capital_ratio if entry_position > 0 else current_equity * short_capital_ratio
        position_size = capital_used * leverage

        if entry_position > 0:
            unrealized_rtn = (current_price / entry_price - 1) * leverage
            unrealized_run_up_pct = (max_price_during_hold / entry_price - 1) * leverage * 100
            unrealized_draw_down_pct = (min_price_during_hold / entry_price - 1) * leverage * 100
        else:
            unrealized_rtn = (entry_price / current_price - 1) * leverage
            unrealized_run_up_pct = (entry_price / min_price_during_hold - 1) * leverage * 100
            unrealized_draw_down_pct = (entry_price / max_price_during_hold - 1) * leverage * 100
        
        unrealized_pnl_usdt = capital_used * unrealized_rtn

        trade_type = 'Long' if entry_position > 0 else 'Short'
        signal_entry = '多單開倉' if entry_position > 0 else '空單開倉'

        # === 修正: 根據 delay_entry 調整最後一個部位的進場時間 ===
        entry_timestamp = df.iloc[entry_index]['timestamp']
        if not delay_entry:
            entry_timestamp += bar_interval # 立即進場則使用 K 棒收盤時間

        trades_log.append({
            'Type': trade_type,
            'Date/Time (Entry)': entry_timestamp.strftime('%Y/%m/%d, %H:%M'),
            'Date/Time (Exit)': 'Open',
            'Signal (Entry)': signal_entry,
            'Signal (Exit)': 'Open',
            'Price (Entry)': f'{entry_price:,.2f}',
            'Price (Exit)': f'{current_price:,.2f}',
            'Position size': f'{position_size:,.2f}',
            'P&L (USDT)': f'{unrealized_pnl_usdt:,.2f}',
            'P&L (%)': f'{unrealized_rtn * 100:,.2f}%',
            'Run-up (%)': f'{unrealized_run_up_pct:,.2f}%',
            'Drawdown (%)': f'{unrealized_draw_down_pct:,.2f}%',
            'Cumulative P&L': f'{cumulative_pnl_usdt:,.2f}',
            'Cumulative P&L (%)': f'{cumulative_pnl_pct:,.2f}%',
        })

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

    # ----- 計算 Sharpe 和 Sortino (加入改進) -----
    equity_curve_series = pd.Series(equity_curve)
    if (equity_curve_series <= 0).any():
        raise ValueError("Equity curve contains non-positive values, invalid for log returns.")
    if len(equity_curve) < 2:
        raise ValueError("Equity curve has fewer than 2 data points, cannot compute returns.")

    log_returns = np.log(equity_curve_series / equity_curve_series.shift(1)).dropna()

    # 自相關檢查（修正 .abs() 為 np.abs()）
    if not log_returns.empty and np.abs(log_returns.autocorr()) > 0.1:
        warnings.warn("Log returns show significant autocorrelation, annualized volatility may be biased.")

    # 間隔字典
    interval_to_periods = {
        '1m': 365 * 24 * 60,  # 525600
        '5m': 365 * 24 * 12,  # 105120
        '15m': 365 * 24 * 4,  # 35040
        '30m': 365 * 24 * 2,  # 17520
        '1h': 365 * 24,       # 8760
        '2h': 365 * 12,       # 4380
        '4h': 365 * 6,        # 2190
        '8h': 365 * 3,        # 1095
        '12h': 365 * 2,       # 730
        '1d': 365             # 365
    }

    # 決定 periods_per_year
    if interval:
        if interval in interval_to_periods:
            periods_per_year = interval_to_periods[interval]
        else:
            warnings.warn(f"Unsupported interval '{interval}', falling back to timestamp inference.")
            periods_per_year = None  # 觸發推斷
    else:
        periods_per_year = None  # 觸發推斷

    if periods_per_year is None:
        # 從 timestamp 推斷（使用中位數）
        try:
            avg_seconds = df['timestamp'].diff().dt.total_seconds().dropna().median()
            if np.isnan(avg_seconds) or avg_seconds <= 0:
                periods_per_year = 365.0  # 回退每日
            else:
                periods_per_year = (365.0 * 24.0 * 3600.0) / avg_seconds
                # 驗證是否接近小時級別（選用）
                if interval.startswith(('1m', '5m', '15m', '30m', '1h')) and not (3000 <= avg_seconds <= 4200 * 60):
                    warnings.warn(f"Timestamp intervals (median={avg_seconds:.0f}s) deviate from expected for '{interval}', check data consistency.")
        except Exception:
            periods_per_year = 365.0

    # 處理空回報
    if log_returns.empty:
        annualized_return = 0.0
        annualized_volatility = 0.0
        sharpe_ratio = 0.0
        sortino_ratio = 0.0
    else:
        mean_log_return = log_returns.mean()
        std_log_return = log_returns.std(ddof=1)

        # 溢出檢查
        annualized_log = mean_log_return * periods_per_year
        if abs(annualized_log) > 700:
            raise OverflowError("Annualized log return too large, numerical overflow.")
        annualized_return = np.exp(annualized_log) - 1.0

        annualized_volatility = std_log_return * np.sqrt(periods_per_year)

        # Sharpe ratio
        if annualized_volatility > 0:
            sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
        else:
            sharpe_ratio = float('inf') if annualized_return > risk_free_rate else float('-inf') if annualized_return < risk_free_rate else 0.0

        # --- Sortino ratio ---
        daily_simple = np.expm1(log_returns)  # 轉換為簡單回報
        rf_per_period = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
        downside_diff = np.minimum(0.0, daily_simple - rf_per_period)
        if len(downside_diff) > 1:
            downside_variance = np.sum(downside_diff ** 2) / (len(downside_diff) - 1)
        else:
            downside_variance = 0.0
        downside_deviation = np.sqrt(downside_variance) * np.sqrt(periods_per_year)
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
        expectancy = (avg_win * win_rate) - (avg_loss * loss_rate)
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
            'Expectancy': f'{expectancy: .4f}'
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
        'float_type_metrics': {  # 新增的字典，存儲浮點數
            'Expectancy': float(expectancy) if expectancy is not None else 0.0000,
            'Max Drawdown': float(max_dd) if max_dd is not None else 0.0000,
            'Sharpe Ratio': float(sharpe_ratio) if sharpe_ratio is not None else 0.00,
            'Sortino Ratio': float(sortino_ratio) if sortino_ratio is not None else 0.00,
            'Calmar Ratio' : float(calmar_ratio) if calmar_ratio is not None else 0.00,
        }
    }