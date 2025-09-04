"""
Strategy backtester for "ADA 4H Long+Short MIX V1" with Optuna optimization.
This script finds the best parameter set by maximizing the Sortino Ratio.
Progress and best trials are logged to 'optuna_progress_log.txt'.
"""

import ccxt
import time
import pandas as pd
import numpy as np
import optuna
from datetime import datetime, timezone

# Suppress Optuna's trial info logging
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ------------------------- STATIC PARAMETERS -------------------------
EXCHANGE = 'binance'
SYMBOL = 'ADA/USDT'
TIMEFRAME = '4h'
SINCE = '2021-01-01'
INITIAL_CAPITAL = 100.0
COMMISSION_PCT = 0.2
LOG_FILE = 'optuna_progress_log.txt'

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

# ------------------------- DATA LOADING (run once) -------------------------
print('Fetching OHLCV data...')
since_ms = to_ms(SINCE)
ohlcv_df = fetch_ohlcv_ccxt(EXCHANGE, SYMBOL, TIMEFRAME, since_ms)
print(f'Loaded {len(ohlcv_df)} bars from {ohlcv_df.index[0]} to {ohlcv_df.index[-1]}')

# ------------------------- OPTIMIZATION CALLBACK -------------------------
def save_best_trial_callback(study, trial):
    """Callback to save best trial results to a file."""
    if study.best_trial.number == trial.number:
        with open(LOG_FILE, 'a') as f:
            f.write(f"---")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Trial Number: {trial.number}\n")
            f.write(f"Value (Sortino Ratio): {trial.value:.4f}\n")
            f.write("Params: \n")
            for key, value in trial.params.items():
                f.write(f"  {key}: {value}\n")

# ------------------------- OPTIMIZATION OBJECTIVE FUNCTION -------------------------
def objective(trial):
    # --- Define parameter search space ---
    donchianLength = trial.suggest_int('donchianLength', 10, 30)
    longTermSmaLen = trial.suggest_int('longTermSmaLen', 100, 200)
    rsiLenLong = trial.suggest_int('rsiLenLong', 10, 40)
    rsiThLong = trial.suggest_float('rsiThLong', 55.0, 75.0)
    emaFastLength = trial.suggest_int('emaFastLength', 5, 15)
    smaSlowLength = trial.suggest_int('smaSlowLength', 50, 80)
    rsiLenShort = trial.suggest_int('rsiLenShort', 50, 80)
    rsiShortThresh = trial.suggest_int('rsiShortThresh', 40, 60)
    shortTPPct = trial.suggest_float('shortTPPct', 5.0, 15.0)
    shortSLPct = trial.suggest_float('shortSLPct', 3.0, 10.0)
    
    # --- Backtest with suggested parameters ---
    df = ohlcv_df.copy()
    
    # Indicators
    df['longTermSma'] = sma(df['close'], longTermSmaLen)
    df['rsiLong'] = rsi(df['close'], rsiLenLong)
    df['upperBand'] = df['high'].rolling(donchianLength).max().shift(1)
    df['lowerBand'] = df['low'].rolling(donchianLength).min().shift(1)
    df['emaFast'] = ema(df['close'], emaFastLength)
    df['smaSlow'] = sma(df['close'], smaSlowLength)
    df['rsiShort'] = rsi(df['close'], rsiLenShort)

    min_required = max(longTermSmaLen, donchianLength, smaSlowLength, rsiLenLong, rsiLenShort)
    df = df.iloc[min_required:]
    if len(df) < 6: # Need at least a day of 4h bars
        return 0.0

    # Backtest loop variables (omitted for brevity, logic is the same)
    cash = INITIAL_CAPITAL; position = 0; pos_qty = 0.0; pos_entry_price = 0.0; entry_comm = 0.0
    commission_pct = COMMISSION_PCT / 100.0; shortLossCount = 0; shortCooldownUntilBarIdx = -1
    equity_curve = []
    longPosPct = 50.0; shortPosPct = 100.0; maxConsecLosses = 1; cooldownBars = 12
    trailTriggerPct = 8.0; trailOffsetPct = 4.0

    for i, row in enumerate(df.itertuples()):
        close, high, low = row.close, row.high, row.low
        inShortCooldown = i < shortCooldownUntilBarIdx
        equity = cash
        if position == 1: equity = cash + pos_qty * close
        elif position == -1: equity = cash + pos_qty * (pos_entry_price - close)
        if position == 1 and low <= row.lowerBand:
            exit_price = row.lowerBand; pnl = pos_qty * (exit_price - pos_entry_price)
            cash += pos_qty * exit_price - (pos_qty * exit_price * commission_pct); position, pos_qty = 0, 0.0
        elif position == -1:
            exit_price = 0; prev = df.iloc[i-1]
            if prev['emaFast'] <= prev['smaSlow'] and row.emaFast > row.smaSlow: exit_price = close
            else:
                shortTP = pos_entry_price * (1 - shortTPPct / 100.0); shortSL = pos_entry_price * (1 + shortSLPct / 100.0)
                stop_price_for_bar = shortSL
                if low <= pos_entry_price * (1 - trailTriggerPct / 100.0): stop_price_for_bar = close * (1 + trailOffsetPct / 100.0)
                if high >= stop_price_for_bar: exit_price = stop_price_for_bar
                elif low <= shortTP: exit_price = shortTP
            if exit_price > 0:
                pnl = pos_qty * (pos_entry_price - exit_price); cash += pnl - (pos_qty * exit_price * commission_pct)
                if pnl < 0: 
                    shortLossCount += 1
                    if shortLossCount >= maxConsecLosses: shortCooldownUntilBarIdx = i + cooldownBars
                else: shortLossCount = 0
                position, pos_qty = 0, 0.0
        if position == 0:
            prev = df.iloc[i-1]
            if (close > row.longTermSma) and (prev['close'] <= prev['upperBand']) and (close > row.upperBand) and (row.rsiLong > rsiThLong):
                qty = (equity*longPosPct/100.0)/close; entry_comm=qty*close*commission_pct; cash-=qty*close+entry_comm; pos_qty,pos_entry_price,position=qty,close,1
            elif not inShortCooldown and (prev['emaFast'] > prev['smaSlow']) and (row.emaFast <= row.smaSlow) and (row.rsiShort < rsiShortThresh):
                qty=(equity*shortPosPct/100.0)/close; entry_comm=qty*close*commission_pct; cash-=entry_comm; pos_qty,pos_entry_price,position=qty,close,-1
        final_equity_on_bar = cash
        if position == 1: final_equity_on_bar = cash + pos_qty * close
        elif position == -1: final_equity_on_bar = cash + pos_qty * (pos_entry_price - close)
        equity_curve.append(final_equity_on_bar)

    # --- Post-backtest calculation for Sortino Ratio ---
    if not equity_curve: return 0.0
    
    # 1. Calculate Compound Annual Return (CAGR)
    total_days = (df.index[-1] - df.index[0]).days
    if total_days < 30: return 0.0
    years = total_days / 365.25
    cagr = (equity_curve[-1] / INITIAL_CAPITAL)**(1 / years) - 1
    if cagr <= 0: return 0.0

    # 2. Calculate Annualized Downside Deviation
    equity_series = pd.Series(equity_curve, index=df.index[:len(equity_curve)])
    daily_equity = equity_series.resample('D').last()
    daily_returns = daily_equity.pct_change().dropna()
    negative_returns = daily_returns[daily_returns < 0]
    
    if len(negative_returns) < 2:
        return 0.0 # Not enough data to calculate downside deviation
        
    downside_deviation = negative_returns.std()
    annualized_downside_deviation = downside_deviation * np.sqrt(365)

    if annualized_downside_deviation == 0: 
        return 1000.0 # Return a large number if there's no downside risk

    # 3. Calculate Sortino Ratio
    sortino_ratio = cagr / annualized_downside_deviation
    return sortino_ratio

# ------------------------- MAIN EXECUTION -------------------------
if __name__ == "__main__":
    n_trials = 100 # You can change this value
    
    # Clear log file at the start of a new study
    with open(LOG_FILE, 'w') as f:
        f.write(f"Optuna optimization log started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Objective: Maximize Sortino Ratio\n")
        f.write(f"Number of Trials: {n_trials}\n")

    print(f"\nStarting Optuna optimization for {n_trials} trials...")
    print(f"Progress will be logged to: {LOG_FILE}")
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, callbacks=[save_best_trial_callback], show_progress_bar=True)

    print("\n" + "="*30 + " OPTIMIZATION FINISHED " + "="*30)
    print(f"Number of finished trials: {len(study.trials)}")
    print("Best trial found:")
    
    best = study.best_trial
    print(f"  Value (Sortino Ratio): {best.value:.4f}")
    print("  Params: ")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
    print(f"\nBest parameters and results are saved in '{LOG_FILE}'")
    print("="*82 + "\n")
