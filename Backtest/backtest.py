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
                     execution_price_type='open'):
    """
    補強版策略回測函式，含滑點、停損停利、持倉限制與最後平倉邏輯。
    """

    required_cols = ['open', 'high', 'low', 'close', 'signal', 'timestamp']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"df 必須包含欄位: {required_cols}")
    if initial_capital <= 0:
        raise ValueError("initial_capital 必須大於 0")

    df = df.copy().reset_index(drop=True)
    df['position'] = np.where(df['signal'] == 1, leverage,
                       np.where(df['signal'] == -1, -leverage if allow_short else 0, np.nan))
    df['position'] = df['position'].ffill().fillna(0)
    df.loc[0, 'position'] = 0

    df['return'] = df['close'].pct_change().fillna(0)
    df['prev_position'] = df['position'].shift(1).fillna(0)

    strategy_returns = np.zeros(len(df))
    trade_returns = []
    hold_bars = []
    trades_log = []

    entry_price = None
    entry_index = None
    entry_position = 0

    for i in range(len(df)):
        row = df.iloc[i]
        open_, high, low, close = row['open'], row['high'], row['low'], row['close']
        target_position = row['position']
        prev_position = row['prev_position']

        # 產生滑點（對你不利）
        slip_pct = np.random.uniform(0, slippage_rate)
        buy_slip = 1 + slip_pct
        sell_slip = 1 - slip_pct

        exit_trade_return = 0
        should_exit = False
        exit_reason = None

        if entry_position != 0:
            holding_period = i - entry_index
            exit_price = None

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

                trade_returns.append(rtn)
                hold_bars.append(holding_period)
                strategy_returns[i] += rtn
                exit_trade_return = rtn

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
            if target_position > 0:
                entry_price = entry_price_raw * (1 + fee_rate) * buy_slip
            else:
                entry_price = entry_price_raw * (1 - fee_rate) * sell_slip

            entry_index = i
            entry_position = target_position

            strategy_returns[i] = exit_trade_return
            continue  # 當日新倉，不計入當日報酬

        if entry_position != 0 and i > 0:
            if i != entry_index:
                day_rtn = (close / df.iloc[i - 1]['close']) - 1
                strategy_returns[i] = day_rtn * entry_position
            else:
                strategy_returns[i] = exit_trade_return  # 若有出場報酬則保留

    # 強制最後一根平倉（如仍有持倉）
    if entry_position != 0:
        last = df.iloc[-1]
        final_price = last['close']
        if entry_position > 0:
            final_price *= (1 - fee_rate) * (1 - np.random.uniform(0, slippage_rate))
            rtn = (final_price / entry_price - 1) * leverage
        else:
            final_price *= (1 + fee_rate) * (1 + np.random.uniform(0, slippage_rate))
            rtn = (entry_price / final_price - 1) * leverage

        trade_returns.append(rtn)
        hold_bars.append(len(df) - entry_index)
        strategy_returns[-1] += rtn

        trades_log.append({
            'entry_time': df.iloc[entry_index]['timestamp'],
            'exit_time': last['timestamp'],
            'side': 'long' if entry_position > 0 else 'short',
            'entry_price': entry_price,
            'exit_price': final_price,
            'bars_held': len(df) - entry_index,
            'return': rtn,
            'reason': 'Final Force Exit',
        })

    # 資產曲線與績效
    df['strategy_return_with_fee'] = strategy_returns
    df['equity'] = initial_capital * (1 + df['strategy_return_with_fee']).cumprod()
    df['buy_and_hold'] = initial_capital * (1 + df['return']).cumprod()
    df['drawdown'] = df['equity'] / df['equity'].cummax() - 1

    total_return = df['equity'].iloc[-1] / initial_capital - 1
    max_dd = df['drawdown'].min()
    time_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / (3600 * 24)
    daily_return = (1 + total_return) ** (1 / time_days) - 1 if time_days > 0 else 0

    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]

    result = {
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
            "trade_returns": np.array(trade_returns) * 100,
        },
        'trades_log': trades_log
    }

    return result
