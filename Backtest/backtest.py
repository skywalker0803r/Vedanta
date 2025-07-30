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
                            execution_price_type='open',
                            capital_ratio=1):
    """
    修正版本：以動態資產追蹤方式回測，避免報酬高估。
    """
    required_cols = ['open', 'high', 'low', 'close', 'signal', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df 必須包含欄位: {required_cols}")
    if not (0 < capital_ratio <= 1):
        raise ValueError("capital_ratio 必須介於 0 與 1 之間")

    df = df.copy().reset_index(drop=True)
    df['position'] = np.where(df['signal'] == 1, leverage,
                       np.where(df['signal'] == -1, -leverage if allow_short else 0, np.nan))
    df['position'] = df['position'].ffill().fillna(0)
    df.loc[0, 'position'] = 0

    equity = initial_capital
    equity_curve = [equity]
    trade_returns = []
    hold_bars = []
    trades_log = []

    entry_price = None
    entry_index = None
    entry_position = 0
    current_equity = initial_capital

    for i in range(1, len(df)):
        row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        target_position = row['position']

        open_, high, low, close = row['open'], row['high'], row['low'], row['close']
        slip_pct = np.random.uniform(0, slippage_rate)
        buy_slip = 1 + slip_pct
        sell_slip = 1 - slip_pct

        should_exit = False
        exit_reason = None
        exit_price = None
        rtn = 0

        if entry_position != 0:
            holding_period = i - entry_index

            if entry_position > 0:
                if stop_loss is not None and low <= entry_price * (1 - stop_loss):
                    should_exit = True
                    exit_reason = 'Stop Loss'
                    exit_price = entry_price * (1 - stop_loss)
                elif take_profit is not None and high >= entry_price * (1 + take_profit):
                    should_exit = True
                    exit_reason = 'Take Profit'
                    exit_price = entry_price * (1 + take_profit)
            else:
                if stop_loss is not None and high >= entry_price * (1 + stop_loss):
                    should_exit = True
                    exit_reason = 'Stop Loss'
                    exit_price = entry_price * (1 + stop_loss)
                elif take_profit is not None and low <= entry_price * (1 - take_profit):
                    should_exit = True
                    exit_reason = 'Take Profit'
                    exit_price = entry_price * (1 - take_profit)

            if max_hold_bars is not None and holding_period >= max_hold_bars:
                should_exit = True
                exit_reason = 'Max Hold Bars'
                exit_price = close

            if not should_exit and target_position != entry_position:
                should_exit = True
                exit_reason = 'Signal Change'
                exit_price = close

            if should_exit:
                if entry_position > 0:
                    exit_price *= (1 - fee_rate) * sell_slip
                    rtn = (exit_price / entry_price - 1) * leverage
                else:
                    exit_price *= (1 + fee_rate) * buy_slip
                    rtn = (entry_price / exit_price - 1) * leverage

                capital_used = current_equity * capital_ratio
                profit = capital_used * rtn
                current_equity += profit

                trade_returns.append(rtn)
                hold_bars.append(holding_period)
                trades_log.append({
                    'entry_time': df.iloc[entry_index]['timestamp'],
                    'exit_time': row['timestamp'],
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

        if entry_position == 0 and target_position != 0:
            entry_price_raw = open_ if execution_price_type == 'open' else close
            entry_price = entry_price_raw * (1 + fee_rate) * buy_slip if target_position > 0 else entry_price_raw * (1 - fee_rate) * sell_slip
            entry_index = i
            entry_position = target_position

        equity_curve.append(current_equity)

    # 強制平倉
    if entry_position != 0:
        final_price = df.iloc[-1]['close']
        if entry_position > 0:
            final_price *= (1 - fee_rate) * (1 - np.random.uniform(0, slippage_rate))
            rtn = (final_price / entry_price - 1) * leverage
        else:
            final_price *= (1 + fee_rate) * (1 + np.random.uniform(0, slippage_rate))
            rtn = (entry_price / final_price - 1) * leverage

        profit = current_equity * capital_ratio * rtn
        current_equity += profit
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
            'reason': 'Final Force Exit',
        })

        equity_curve[-1] = current_equity

    equity_series = pd.Series(equity_curve, index=df.index)
    df['equity'] = equity_series
    df['buy_and_hold'] = initial_capital * (1 + df['close'].pct_change().fillna(0)).cumprod()
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1

    total_return = df['equity'].iloc[-1] / initial_capital - 1
    max_dd = df['drawdown'].min()
    time_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400
    daily_return = (1 + total_return) ** (1 / time_days) - 1 if time_days > 0 else 0

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