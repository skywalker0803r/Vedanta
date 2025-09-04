import pandas as pd
import numpy as np

def rsi(series, period):
    delta = series.diff()
    up = np.where(delta > 0, delta, 0)
    down = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(up).rolling(period).mean()
    roll_down = pd.Series(down).rolling(period).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))

def run_backtest(df: pd.DataFrame,
                 donchian_len=12, long_sma_len=150,
                 rsi_len_long=30, rsi_th_long=60,
                 ema_fast_len=6, sma_slow_len=65,
                 rsi_len_short=65, rsi_th_short=50,
                 short_tp_pct=10, short_sl_pct=5,
                 trail_trigger=8, trail_offset=4,
                 max_consec_loss=1, cooldown_bars=12,
                 initial_capital=100):
    
    # === 指標計算 ===
    df = df.copy()
    df["donchian_high"] = df["high"].rolling(donchian_len).max().shift(1)
    df["donchian_low"]  = df["low"].rolling(donchian_len).min().shift(1)
    df["sma_long"]      = df["close"].rolling(long_sma_len).mean()
    df["ema_fast"]      = df["close"].ewm(span=ema_fast_len).mean()
    df["sma_slow"]      = df["close"].rolling(sma_slow_len).mean()
    df["rsi_long"]      = rsi(df["close"], rsi_len_long)
    df["rsi_short"]     = rsi(df["close"], rsi_len_short)

    equity = initial_capital
    pos = 0
    entry_price = 0
    short_loss_count = 0
    cooldown_until = -1
    trades = []

    for i in range(max(donchian_len, long_sma_len), len(df)):
        price = df["close"].iloc[i]

        # === 多單進場 ===
        if pos == 0:
            if (price > df["sma_long"].iloc[i]) and \
               (price > df["donchian_high"].iloc[i]) and \
               (df["rsi_long"].iloc[i] > rsi_th_long):
                pos = 1
                entry_price = price
                trades.append(("LONG_ENTRY", df.index[i], price))

        # === 多單出場 ===
        if pos > 0 and price < df["donchian_low"].iloc[i]:
            profit = (price - entry_price) / entry_price * equity
            equity += profit
            trades.append(("LONG_EXIT", df.index[i], price, profit))
            pos = 0

        # === 空單進場 ===
        in_cooldown = i < cooldown_until
        crossunder = (df["ema_fast"].iloc[i-1] > df["sma_slow"].iloc[i-1]) and (df["ema_fast"].iloc[i] < df["sma_slow"].iloc[i])
        if pos == 0 and crossunder and (df["rsi_short"].iloc[i] < rsi_th_short) and not in_cooldown:
            pos = -1
            entry_price = price
            trades.append(("SHORT_ENTRY", df.index[i], price))

        # === 空單止盈止損 ===
        if pos < 0:
            tp_price = entry_price * (1 - short_tp_pct/100)
            sl_price = entry_price * (1 + short_sl_pct/100)
            trigger  = entry_price * (1 - trail_trigger/100)
            trail_stop = sl_price
            if df["low"].iloc[i] <= trigger:
                trail_stop = price * (1 + trail_offset/100)

            if price <= tp_price or price >= trail_stop:
                profit = (entry_price - price) / entry_price * equity
                equity += profit
                trades.append(("SHORT_EXIT", df.index[i], price, profit))
                if profit < 0:
                    short_loss_count += 1
                    if short_loss_count >= max_consec_loss:
                        cooldown_until = i + cooldown_bars
                else:
                    short_loss_count = 0
                pos = 0

        # === 空單反向平倉 ===
        if pos < 0 and (df["ema_fast"].iloc[i-1] < df["sma_slow"].iloc[i-1]) and (df["ema_fast"].iloc[i] > df["sma_slow"].iloc[i]):
            profit = (entry_price - price) / entry_price * equity
            equity += profit
            trades.append(("SHORT_CLOSE_REV", df.index[i], price, profit))
            pos = 0

    # 計算最大回撤
    equity_curve = pd.Series([t[3] for t in trades if len(t) == 4], index=[t[2] for t in trades if len(t) == 4])
    equity_curve = equity_curve.cumsum() + initial_capital
    
    if equity_curve.empty:
        max_drawdown = 0.0
    else:
        peak = equity_curve.expanding(min_periods=1).max()
        drawdown = (equity_curve - peak) / peak
        max_drawdown = drawdown.min() * -1 # Convert to positive value

    return equity, trades, max_drawdown
