"""
This script runs a single backtest using the default parameters from the original pine.txt strategy.
Its purpose is to evaluate the performance of the baseline strategy before optimization.
"""

import ccxt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# ------------------------- STATIC PARAMETERS -------------------------
EXCHANGE = 'binance'
SYMBOL = 'ADA/USDT'
TIMEFRAME = '4h'
SINCE = '2021-01-01'
INITIAL_CAPITAL = 100.0
COMMISSION_PCT = 0.2

# ------------------------- UTILS / INDICATORS -------------------------

def to_ms(dt_str):
    dt = datetime.fromisoformat(dt_str)
    return int(dt.replace(tzinfo=timezone.utc).timestamp() * 1000)

def fetch_ohlcv_ccxt(exchange_id, symbol, timeframe, since_ms):
    ex = getattr(ccxt, exchange_id)({'enableRateLimit': True})
    out = []
    limit = 1000
    since = since_ms
    while True:
        candles = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not candles:
            break
        out += candles
        since = candles[-1][0] + 1
        if len(candles) < limit:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(out, columns=['timestamp','open','high','low','close','volume'])
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('datetime', inplace=True)
    return df

def sma(series, n):
    return series.rolling(n, min_periods=1).mean()

def ema(series, n):
    return series.ewm(span=n, adjust=False).mean()

def rsi(series, n):
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/n, adjust=False).mean()
    ma_down = down.ewm(alpha=1/n, adjust=False).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))

# ------------------------- DATA LOADING -------------------------
print('Fetching OHLCV data...')
since_ms = to_ms(SINCE)
df = fetch_ohlcv_ccxt(EXCHANGE, SYMBOL, TIMEFRAME, since_ms)
print(f'Loaded {len(df)} bars from {df.index[0]} to {df.index[-1]}\n')

# ------------------------- BACKTEST PARAMETERS (from pine.txt defaults) -------------------------
# Long params
donchianLength = 25
longTermSmaLen = 118
rsiLenLong = 26
rsiThLong = 71.06
# Short params
emaFastLength = 14
smaSlowLength = 80
rsiLenShort = 80
rsiShortThresh = 52
shortTPPct = 7.38
shortSLPct = 4.93
trailTriggerPct = 8.0
trailOffsetPct = 4.0
# Cooldown params
maxConsecLosses = 1
cooldownBars = 12
# Sizing params
longPosPct = 50.0
shortPosPct = 100.0

# ------------------------- INDICATOR CALCULATION -------------------------
df['longTermSma'] = sma(df['close'], longTermSmaLen)
df['rsiLong'] = rsi(df['close'], rsiLenLong)
df['upperBand'] = df['high'].rolling(donchianLength).max().shift(1)
df['lowerBand'] = df['low'].rolling(donchianLength).min().shift(1)
df['emaFast'] = ema(df['close'], emaFastLength)
df['smaSlow'] = sma(df['close'], smaSlowLength)
df['rsiShort'] = rsi(df['close'], rsiLenShort)

min_required = max(longTermSmaLen, donchianLength, smaSlowLength, rsiLenLong, rsiLenShort)
df = df.iloc[min_required:]

# ------------------------- BACKTEST LOOP -------------------------
cash = INITIAL_CAPITAL
position = 0
pos_qty = 0.0
pos_entry_price = 0.0
entry_comm = 0.0
commission_pct = COMMISSION_PCT / 100.0
shortLossCount = 0
shortCooldownUntilBarIdx = -1
equity_curve = []
trade_log = []

for i, row in enumerate(df.itertuples()):
    close, high, low = row.close, row.high, row.low
    inShortCooldown = i < shortCooldownUntilBarIdx
    
    equity = cash
    if position == 1: equity = cash + pos_qty * close
    elif position == -1: equity = cash + pos_qty * (pos_entry_price - close)

    # --- Exit Logic ---
    if position == 1 and low <= row.lowerBand:
        exit_price = row.lowerBand
        pnl = pos_qty * (exit_price - pos_entry_price)
        exit_comm = pos_qty * exit_price * commission_pct
        cash += pos_qty * exit_price - exit_comm
        trade_log.append({'pnl': pnl - entry_comm - exit_comm})
        position, pos_qty = 0, 0.0
    elif position == -1:
        exit_price, exit_type = 0, ''
        prev = df.iloc[i-1]
        if prev['emaFast'] <= prev['smaSlow'] and row.emaFast > row.smaSlow:
            exit_price, exit_type = close, 'Reversal'
        else:
            shortTP = pos_entry_price * (1 - shortTPPct / 100.0)
            shortSL = pos_entry_price * (1 + shortSLPct / 100.0)
            stop_price_for_bar = shortSL
            if low <= pos_entry_price * (1 - trailTriggerPct / 100.0):
                stop_price_for_bar = close * (1 + trailOffsetPct / 100.0)
            if high >= stop_price_for_bar: exit_price, exit_type = stop_price_for_bar, 'Stop'
            elif low <= shortTP: exit_price, exit_type = shortTP, 'TP'
        
        if exit_price > 0:
            pnl = pos_qty * (pos_entry_price - exit_price)
            exit_comm = pos_qty * exit_price * commission_pct
            cash += pnl - exit_comm
            trade_log.append({'pnl': pnl - entry_comm - exit_comm})
            if pnl < 0:
                shortLossCount += 1
                if shortLossCount >= maxConsecLosses: shortCooldownUntilBarIdx = i + cooldownBars
            else: shortLossCount = 0
            position, pos_qty = 0, 0.0

    # --- Entry Logic ---
    if position == 0:
        prev = df.iloc[i-1]
        if (close > row.longTermSma) and (prev['close'] <= prev['upperBand']) and (close > row.upperBand) and (row.rsiLong > rsiThLong):
            qty = (equity * longPosPct / 100.0) / close
            entry_comm = qty * close * commission_pct
            cash -= qty * close + entry_comm
            pos_qty, pos_entry_price, position = qty, close, 1
        elif not inShortCooldown and (prev['emaFast'] > prev['smaSlow']) and (row.emaFast <= row.smaSlow) and (row.rsiShort < rsiShortThresh):
            qty = (equity * shortPosPct / 100.0) / close
            entry_comm = qty * close * commission_pct
            cash -= entry_comm
            pos_qty, pos_entry_price, position = qty, close, -1
    
    final_equity_on_bar = cash
    if position == 1: final_equity_on_bar = cash + pos_qty * close
    elif position == -1: final_equity_on_bar = cash + pos_qty * (pos_entry_price - close)
    equity_curve.append(final_equity_on_bar)

# ------------------------- POST-BACKTEST & METRIC CALCULATION -------------------------
print("="*30 + " PERFORMANCE (Default Params) " + "="*30)

final_equity = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
total_pnl_pct = (final_equity / INITIAL_CAPITAL - 1.0) * 100.0
print(f"Initial Capital: {INITIAL_CAPITAL:.2f}")
print(f"Final Equity:    {final_equity:.2f}")
print(f"Total PNL:       {total_pnl_pct:.2f}%")

if equity_curve and len(df) > 0:
    equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    peak = equity_series.expanding(min_periods=1).max()
    drawdown = (equity_series - peak) / peak
    max_drawdown = abs(drawdown.min())
    max_drawdown_pct = max_drawdown * 100.0
    print(f"Max Drawdown:    {max_drawdown_pct:.2f}%")

    total_days = (df.index[-1] - df.index[0]).days
    if total_days > 30:
        years = total_days / 365.25
        cagr = (final_equity / INITIAL_CAPITAL)**(1 / years) - 1

        # Calculate Calmar Ratio
        if max_drawdown > 0:
            calmar_ratio = cagr / max_drawdown
            print(f"Calmar Ratio:    {calmar_ratio:.3f}")
        else:
            print("Calmar Ratio:    inf")

        # Calculate Sortino Ratio
        daily_equity = equity_series.resample('D').last()
        daily_returns = daily_equity.pct_change().dropna()
        negative_returns = daily_returns[daily_returns < 0]
        if len(negative_returns) > 1:
            downside_deviation = negative_returns.std()
            annualized_downside_deviation = downside_deviation * np.sqrt(365)
            if annualized_downside_deviation > 0:
                sortino_ratio = cagr / annualized_downside_deviation
                print(f"Sortino Ratio:   {sortino_ratio:.3f}")
            else:
                print("Sortino Ratio:   inf") # No downside risk
        else:
            print("Sortino Ratio:   N/A (Not enough data)")
    else:
        print("Calmar/Sortino:  N/A (Backtest too short)")
else:
    print("Max Drawdown:    0.00%")
    print("Calmar Ratio:    0.000")
    print("Sortino Ratio:   0.000")

trades_df = pd.DataFrame(trade_log)
total_trades = len(trades_df)
if total_trades > 0:
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    win_rate_pct = (winning_trades / total_trades) * 100.0
    gross_profit = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
    gross_loss = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    print(f"Total Trades:    {total_trades}")
    print(f"Win Rate:        {win_rate_pct:.2f}%")
    print(f"Profit Factor:   {profit_factor:.3f}")
else:
    print("Total Trades:    0")

print("="*87 + "\n")

# ------------------------- PLOTTING -------------------------
plt.figure(figsize=(12, 6))
plt.plot(df.index[:len(equity_curve)], equity_curve)
plt.title('Equity Curve (Default Parameters)')
plt.ylabel('Equity')
plt.xlabel('Date')
plt.grid(True)
plt.show()