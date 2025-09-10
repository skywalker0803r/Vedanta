from datetime import datetime
from Multi_parameter_optimization_analysis.MPOA import FlexibleStrategyOptimizer
from Technicalindicatorstrategy import fvg_rsi_strategy_optimized

symbols = ["BTCUSDT"]#, "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT", "HYPEUSDT", "LINKUSDT", "SUIUSDT"]

# Define common optimization parameters
optimize_params = {
    'rsi_len': {'type': 'int', 'min': 5, 'max': 30, 'step': 1},
    'rsi_overbought': {'type': 'int', 'min': 70, 'max': 90, 'step': 1},
    'rsi_oversold': {'type': 'int', 'min': 10, 'max': 30, 'step': 1},
    'atr_len': {'type': 'int', 'min': 10, 'max': 30, 'step': 1},
    'atr_tp_multiplier': {'type': 'float', 'min': 1.0, 'max': 5.0, 'step': 0.1},
    'sl_pct': {'type': 'float', 'min': 0.5, 'max': 3.0, 'step': 0.1},
    'ema_lower_len': {'type': 'int', 'min': 10, 'max': 50, 'step': 1},
    'ema_upper_len': {'type': 'int', 'min': 50, 'max': 200, 'step': 5}
}

# Define common backtest config
backtest_config = {
    'initial_capital': 1000000,
    'fee_rate': 0.0005,
    'leverage': 1,
    'allow_short': True,
    'stop_loss': None,
    'take_profit': None,
    'capital_ratio': 1,
    'max_hold_bars': 10000,
    'delay_entry': False,
    'risk_free_rate': 0.02,
    'interval': '1m'
}

# Define composite config
composite_config = {
    'method': 'weighted',
    'weights': {
        'Sharpe Ratio': 0.4,
        'Sortino Ratio': 0.4,
        'Calmar Ratio': 0.2
    },
    'thresholds': {
        'Sharpe Ratio': 1.0,
        'Sortino Ratio': 1.5,
        'Calmar Ratio': 0.5
    }
}

results = {}

for symbol in symbols:
    print(f"\n--- Optimizing for {symbol} ---")
    strategy_config = {
        'strategy_module': fvg_rsi_strategy_optimized,
        'strategy_function': 'get_signals',
        'fixed_params': {
            'symbol': symbol,
            'interval': '1m',
            'end_time': datetime.utcnow(), # Use UTC for consistency with Binance API
            'limit': 12000
        },
        'optimize_params': optimize_params,
        'target_metrics': ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio'],
        'composite_config': composite_config,
        'backtest_config': backtest_config,
        'optimize_config': {
            'sampler': 'tpe', # You can change this to 'random', 'cmaes', 'nsga2', 'qmc'
            'seed': 42
        }
    }

    optimizer = FlexibleStrategyOptimizer(strategy_config)
    
    # Run optimization with a reasonable number of trials. 
    # For multiple symbols, consider reducing n_trials per symbol if total time is a concern.
    # For demonstration, let's use 50 trials per symbol.
    try:
        study = optimizer.run_optimization(n_trials=50, n_jobs=-1, study_name=f"fvg_rsi_optimization_{symbol.replace('USDT', '')}")
        best_trial = study.best_trial
        
        results[symbol] = {
            'best_composite_metric': best_trial.value,
            'best_params': best_trial.params,
            'all_metrics': {k: v for k, v in best_trial.user_attrs.items() if k.startswith('metric_')}
        }
        print(f"Optimization for {symbol} completed. Best composite metric: {best_trial.value:.4f}")
        print(f"Best parameters for {symbol}: {best_trial.params}")

    except Exception as e:
        print(f"Error optimizing for {symbol}: {e}")
        results[symbol] = {'error': str(e)}

print("\n--- Optimization Summary ---")
for symbol, data in results.items():
    if 'error' in data:
        print(f"{symbol}: Error - {data['error']}")
    else:
        print(f"{symbol}: Best Composite Metric = {data['best_composite_metric']:.4f}, Best Params = {data['best_params']}")
