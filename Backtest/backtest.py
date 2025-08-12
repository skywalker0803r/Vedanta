import numpy as np
import pandas as pd

def backtest_signals(df: pd.DataFrame,
                     initial_capital=100,
                     fee_rate=0.001,
                     leverage=1,
                     allow_short=True,
                     stop_loss=None,
                     take_profit=None,
                     max_hold_bars=None,
                     slippage_rate=0.0005,
                     capital_ratio=1,
                     maintenance_margin_ratio=0.005,
                     liquidation_penalty=1.0,
                     delay_entry=True):  # 新增 delay_entry 參數

    required_cols = ['open', 'high', 'low', 'close', 'signal', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df 必須包含欄位: {required_cols}")
    if not (0 < capital_ratio <= 1):
        raise ValueError("capital_ratio 必須介於 0 與 1 之間")

    df = df.copy().reset_index(drop=True)

    # 確保 timestamp 為 datetime 格式
    if not np.issubdtype(df['timestamp'].dtype, np.datetime64):
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 根據 delay_entry 來決定是否延遲訊號
    if delay_entry:
        df['used_signal'] = df['signal'].shift(1).fillna(0)
    else:
        df['used_signal'] = df['signal'].fillna(0)

    df['position'] = np.where(df['used_signal'] == 1, leverage,
                       np.where(df['used_signal'] == -1, -leverage if allow_short else 0, np.nan))
    df['position'] = df['position'].ffill().fillna(0)
    df.loc[0, 'position'] = 0

    equity_curve = [initial_capital]
    trade_returns, hold_bars, trades_log = [], [], []

    entry_price = None
    entry_index = None
    entry_position = 0
    current_equity = initial_capital

    for i in range(1, len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        target_position = row['position']

        open_, close = row['open'], row['close']
        slip_pct = np.random.uniform(0, slippage_rate)
        buy_slip = 1 + slip_pct
        sell_slip = 1 - slip_pct

        should_exit = False
        exit_reason = None
        exit_price = None
        rtn = 0

        # ===== 出場判斷 =====
        if entry_position != 0:
            holding_period = i - entry_index

            # 止盈止損（用已結束的 K 棒判斷，下一根開盤平倉）
            if entry_position > 0:  # 多單
                if stop_loss is not None and row['low'] <= entry_price * (1 - stop_loss):
                    should_exit = True
                    exit_reason = 'Stop Loss'
                    exit_price = next_row['open'] * (1 - fee_rate) * sell_slip
                elif take_profit is not None and row['high'] >= entry_price * (1 + take_profit):
                    should_exit = True
                    exit_reason = 'Take Profit'
                    exit_price = next_row['open'] * (1 - fee_rate) * sell_slip
            else:  # 空單
                if stop_loss is not None and row['high'] >= entry_price * (1 + stop_loss):
                    should_exit = True
                    exit_reason = 'Stop Loss'
                    exit_price = next_row['open'] * (1 + fee_rate) * buy_slip
                elif take_profit is not None and row['low'] <= entry_price * (1 - take_profit):
                    should_exit = True
                    exit_reason = 'Take Profit'
                    exit_price = next_row['open'] * (1 + fee_rate) * buy_slip

            # 最大持倉K棒數
            if max_hold_bars is not None and holding_period >= max_hold_bars:
                should_exit = True
                exit_reason = 'Max Hold Bars'
                exit_price = next_row['open'] * (1 - fee_rate if entry_position > 0 else 1 + fee_rate)

            # 信號反轉
            if not should_exit and target_position != entry_position:
                should_exit = True
                exit_reason = 'Signal Change'
                exit_price = next_row['open'] * (1 - fee_rate if entry_position > 0 else 1 + fee_rate)

            # 執行出場
            if should_exit:
                capital_used = current_equity * capital_ratio
                maintenance_margin = capital_used * maintenance_margin_ratio

                if entry_position > 0:
                    rtn = (exit_price / entry_price - 1) * leverage
                else:
                    rtn = (entry_price / exit_price - 1) * leverage

                profit = capital_used * rtn

                if current_equity + profit < maintenance_margin:
                    loss = capital_used * liquidation_penalty
                    current_equity -= loss
                    rtn = -liquidation_penalty
                    exit_reason = 'Liquidated'
                else:
                    current_equity += profit

                trade_returns.append(rtn)
                hold_bars.append(holding_period)
                trades_log.append({
                    'entry_time': df.iloc[entry_index]['timestamp'],
                    'exit_time': next_row['timestamp'],
                    'side': 'long' if entry_position > 0 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'bars_held': holding_period,
                    'return': rtn,
                    'reason': exit_reason,
                })

                entry_price = None
                entry_index = None
                entry_position = 0

        # ===== 進場判斷 =====
        if entry_position == 0 and target_position != 0:
            entry_price = open_ * (1 + fee_rate) * buy_slip if target_position > 0 else open_ * (1 - fee_rate) * sell_slip
            entry_index = i
            entry_position = target_position

        equity_curve.append(current_equity)

    # ===== 最後強制平倉 =====
    if entry_position != 0:
        final_price = df.iloc[-1]['open']
        if entry_position > 0:
            final_price *= (1 - fee_rate) * (1 - np.random.uniform(0, slippage_rate))
            rtn = (final_price / entry_price - 1) * leverage
        else:
            final_price *= (1 + fee_rate) * (1 + np.random.uniform(0, slippage_rate))
            rtn = (entry_price / final_price - 1) * leverage

        capital_used = current_equity * capital_ratio
        maintenance_margin = capital_used * maintenance_margin_ratio
        profit = capital_used * rtn

        if current_equity + profit < maintenance_margin:
            loss = capital_used * liquidation_penalty
            current_equity -= loss
            rtn = -liquidation_penalty
            exit_reason = 'Final Liquidated'
        else:
            current_equity += profit
            exit_reason = 'Final Force Exit'

        trade_returns.append(rtn)
        hold_bars.append(len(df) - entry_index)
        trades_log.append({
            'entry_time': df.iloc[entry_index]['timestamp'],
            'exit_time': df.iloc[-1]['timestamp'],
            'side': 'long' if entry_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': final_price,
            'bars_held': len(df) - entry_index,
            'return': rtn,
            'reason': exit_reason,
        })

        equity_curve[-1] = current_equity

    # ===== 確保 equity_curve 長度與 df 一致 =====
    if len(equity_curve) < len(df):
        equity_curve += [equity_curve[-1]] * (len(df) - len(equity_curve))

    df['equity'] = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    df['buy_and_hold'] = initial_capital * (1 + df['close'].pct_change().fillna(0)).cumprod()
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1

    # ===== 安全計算報酬率 =====
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

    return {
        'metric': {
            '回測K棒數量': len(df),
            '總報酬率': f'{total_return * 100:.2f}',
            '日報酬率': f'{daily_return * 100:.4f}',
            '最大回撤': f'{max_dd * 100:.2f}',
            '交易次數': len(trade_returns),
            '勝率': f'{(np.mean([r > 0 for r in trade_returns]) * 100):.2f}' if trade_returns else '0.00',
            '平均每筆報酬率': f'{np.mean(trade_returns) * 100:.2f}' if trade_returns else '0.00',
            '平均獲利時報酬': f'{np.mean(wins) * 100:.2f}' if wins else '0.00',
            '平均虧損時報酬': f'{np.mean(losses) * 100:.2f}' if losses else '0.00',
            '盈虧比': f'{(np.mean(wins) / abs(np.mean(losses))):.2f}' if wins and losses and np.mean(losses) != 0 else 'N/A',
            '最大單筆報酬': f'{np.max(trade_returns) * 100:.2f}' if trade_returns else '0.00',
            '最大單筆虧損': f'{np.min(trade_returns) * 100:.2f}' if trade_returns else '0.00',
            '平均持有K棒數': f'{np.mean(hold_bars):.2f}' if hold_bars else '0.00',
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
