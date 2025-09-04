import optuna
from strategy import run_backtest

def optimize(df):
    def objective(trial):
        params = {
            "donchian_len": trial.suggest_int("donchian_len", 5, 30),
            "long_sma_len": trial.suggest_int("long_sma_len", 100, 200),
            "rsi_len_long": trial.suggest_int("rsi_len_long", 10, 50),
            "rsi_th_long":  trial.suggest_int("rsi_th_long", 40, 70),
            "ema_fast_len": trial.suggest_int("ema_fast_len", 5, 20),
            "sma_slow_len": trial.suggest_int("sma_slow_len", 30, 100),
            "rsi_len_short": trial.suggest_int("rsi_len_short", 20, 80),
            "rsi_th_short":  trial.suggest_int("rsi_th_short", 30, 70),
        }
        equity, trades, max_drawdown = run_backtest(df, **params)
        return max_drawdown

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=100)
    return study
