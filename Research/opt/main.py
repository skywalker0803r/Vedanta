"""
Reproduce the Pine Script "ADA 4H Long+Short MIX V1" strategy in Python.
This version is updated to be logically consistent with the provided Pine Script,
including all entry, exit, and risk management rules.
"""

import ccxt
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone

# ------------------------- USER PARAMETERS (match Pine inputs) -------------------------
EXCHANGE = 'binance'
SYMBOL = 'ADA/USDT'
TIMEFRAME = '4h'
SINCE = '2021-01-01'   # inclusive start date
INITIAL_CAPITAL = 100.0
COMMISSION_PCT = 0.2    # percent per trade (entry or exit)

# Position sizing
longPosPct = 50.0
shortPosPct = 100.0

# Long params
donchianLength = 12
longTermSmaLen = 150
rsiLenLong = 30
rsiThLong = 60.0

# Short params
emaFastLength = 6
smaSlowLength = 65
rsiLenShort = 65
rsiShortThresh = 50

shortTPPct = 10.0
shortSLPct = 5.0
trailTriggerPct = 8.0
trailOffsetPct = 4.0

# Short cooldown
maxConsecLosses = 1
cooldownBars = 12

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

# ------------------------- FETCH DATA -------------------------

since_ms = to_ms(SINCE)
print('Fetching OHLCV from Binance since', SINCE)
df = fetch_ohlcv_ccxt(EXCHANGE, SYMBOL, TIMEFRAME, since_ms)
print('Loaded', len(df), 'bars. From', df.index[0], 'to', df.index[-1])

# ------------------------- INDICATORS -------------------------
df['longTermSma'] = sma(df['close'], longTermSmaLen)
df['rsiLong'] = rsi(df['close'], rsiLenLong)
df['upperBand'] = df['high'].rolling(donchianLength).max().shift(1)
df['lowerBand'] = df['low'].rolling(donchianLength).min().shift(1)

df['emaFast'] = ema(df['close'], emaFastLength)
df['smaSlow'] = sma(df['close'], smaSlowLength)
df['rsiShort'] = rsi(df['close'], rsiLenShort)

min_required = max(longTermSmaLen, donchianLength, smaSlowLength, rsiLenLong, rsiLenShort)
df = df.iloc[min_required:]

# ------------------------- BACKTEST -------------------------

capital = INITIAL_CAPITAL
cash = capital
position = 0  # 0: flat, 1: long, -1: short
pos_qty = 0.0
pos_entry_price = 0.0
entry_comm = 0.0
commission_pct = COMMISSION_PCT / 100.0
trade_log = []
shortLossCount = 0
shortCooldownUntilBarIdx = -1
equity_curve = []

for i, row in enumerate(df.itertuples()):
    timestamp = row.Index
    close = row.close
    high = row.high
    low = row.low
    inShortCooldown = i < shortCooldownUntilBarIdx

    # Calculate equity at the start of the bar, consistent with Pine's strategy.equity
    equity = cash
    if position == 1:
        equity = cash + pos_qty * close
    elif position == -1:
        # For short, equity is cash + unrealized PnL
        equity = cash + pos_qty * (pos_entry_price - close)

    # -------------------- EXIT LOGIC --------------------
    # --- LONG EXIT ---
    if position == 1 and low <= row.lowerBand:
        exit_price = row.lowerBand  # Exit at the stop price
        pnl = pos_qty * (exit_price - pos_entry_price)
        exit_comm = pos_qty * exit_price * commission_pct
        cash += pos_qty * exit_price - exit_comm # Add back asset value and subtract exit commission
        
        trade_log.append({'type':'CloseLong', 'exit_price': exit_price, 'pnl': pnl - entry_comm - exit_comm, 'timestamp': timestamp})
        position = 0
        pos_qty = 0.0

    # --- SHORT EXIT ---
    elif position == -1:
        exit_price = 0
        exit_type = ''

        # 1. Reversal signal exit
        prev = df.iloc[i-1]
        if prev['emaFast'] <= prev['smaSlow'] and row.emaFast > row.smaSlow:
            exit_price = close
            exit_type = 'Reversal'
        
        # 2. TP/SL/Trail Exit (only if not already exited by reversal)
        if exit_price == 0:
            shortTP = pos_entry_price * (1 - shortTPPct / 100.0)
            shortSL = pos_entry_price * (1 + shortSLPct / 100.0)
            triggerPrice = pos_entry_price * (1 - trailTriggerPct / 100.0)

            # Determine stop price for this bar (conditional stop, not a true sticky trail)
            stop_price_for_bar = shortSL
            if low <= triggerPrice:
                stop_price_for_bar = close * (1 + trailOffsetPct / 100.0)

            if high >= stop_price_for_bar:
                exit_price = stop_price_for_bar
                exit_type = 'Stop'
            elif low <= shortTP:
                exit_price = shortTP
                exit_type = 'TP'

        if exit_price > 0:
            pnl = pos_qty * (pos_entry_price - exit_price)
            exit_comm = pos_qty * exit_price * commission_pct
            cash += pnl - exit_comm # For shorts, cash was not tied up, so just add PnL and subtract exit comm

            # Update cooldown counter
            if pnl < 0:
                shortLossCount += 1
                if shortLossCount >= maxConsecLosses:
                    shortCooldownUntilBarIdx = i + cooldownBars
            else:
                shortLossCount = 0  # Reset on profit

            trade_log.append({'type':f'CloseShort ({exit_type})', 'exit_price': exit_price, 'pnl': pnl - entry_comm - exit_comm, 'timestamp': timestamp})
            position = 0
            pos_qty = 0.0

    # -------------------- ENTRY LOGIC --------------------
    if position == 0:
        # Long signal
        isMacroUptrend = close > row.longTermSma
        prev = df.iloc[i-1]
        isBreakout = (prev['close'] <= prev['upperBand']) and (close > row.upperBand)
        isRsiOKLong = row.rsiLong > rsiThLong
        longSignal = isMacroUptrend and isBreakout and isRsiOKLong

        if longSignal:
            qty = (equity * longPosPct / 100.0) / close
            entry_comm = qty * close * commission_pct
            cash -= qty * close + entry_comm # Cash is reduced by purchase cost and commission
            pos_qty = qty
            pos_entry_price = close
            position = 1
            trade_log.append({'type':'OpenLong','entry_price':pos_entry_price,'qty':pos_qty,'entry_index':i, 'timestamp': timestamp})

        # Short signal
        shortSignal = (prev['emaFast'] > prev['smaSlow']) and (row.emaFast <= row.smaSlow) and (row.rsiShort < rsiShortThresh)
        
        if shortSignal and not inShortCooldown:
            qty = (equity * shortPosPct / 100.0) / close
            entry_comm = qty * close * commission_pct
            cash -= entry_comm  # For shorts, cash is only reduced by commission
            pos_qty = qty
            pos_entry_price = close
            position = -1
            trade_log.append({'type':'OpenShort','entry_price':pos_entry_price,'qty':pos_qty,'entry_index':i, 'timestamp': timestamp})

    # Record equity for this bar
    # Recalculate final equity for the record
    final_equity_on_bar = cash
    if position == 1:
        final_equity_on_bar = cash + pos_qty * close
    elif position == -1:
        final_equity_on_bar = cash + pos_qty * (pos_entry_price - close)
    equity_curve.append(final_equity_on_bar)


# ------------------------- POST-BACKTEST -------------------------
# Close any open positions at the last price
if position == 1:
    exit_price = df['close'].iloc[-1]
    pnl = pos_qty * (exit_price - pos_entry_price)
    exit_comm = pos_qty * exit_price * commission_pct
    cash += pos_qty * exit_price - exit_comm
    trade_log.append({'type':'CloseLong (EOD)', 'exit_price': exit_price, 'pnl': pnl - entry_comm - exit_comm, 'timestamp': df.index[-1]})
    position = 0
elif position == -1:
    exit_price = df['close'].iloc[-1]
    pnl = pos_qty * (pos_entry_price - exit_price)
    exit_comm = pos_qty * exit_price * commission_pct
    cash += pnl - exit_comm
    trade_log.append({'type':'CloseShort (EOD)', 'exit_price': exit_price, 'pnl': pnl - entry_comm - exit_comm, 'timestamp': df.index[-1]})
    position = 0

if equity_curve:
    equity_curve[-1] = cash # Final equity is the final cash after closing all positions

# ------------------------- RESULTS -------------------------
trades_df = pd.DataFrame(trade_log)
final_equity = equity_curve[-1] if equity_curve else INITIAL_CAPITAL
total_pnl_pct = (final_equity / INITIAL_CAPITAL - 1.0) * 100.0
print(f"Initial capital: {INITIAL_CAPITAL}, Final equity: {final_equity:.6f}, TOTAL PNL% = {total_pnl_pct:.2f}%")

plt.figure(figsize=(10,5))
plt.plot(df.index[:len(equity_curve)], equity_curve)
plt.title('Equity Curve')
plt.ylabel('Equity')
plt.xlabel('Date')
plt.grid(True)
plt.show()

trades_df.to_csv('ada_strategy_trades_fixed.csv', index=False)
print('Saved trades to ada_strategy_trades_fixed.csv')