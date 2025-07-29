from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import testsma

class StrategyWrapper:
    def __init__(self, strategy_module):
        self.strategy_module = strategy_module
        self.name = "testsma"  # 你想顯示的策略名稱

    def get_signals(self, symbol, interval, end_time, limit=3000):
        return self.strategy_module.get_signals(symbol, interval, end_time, limit)

auto_trade_futures(
    symbol="ETH/USDT",
    interval="1m",
    usdt_per_order=100,
    leverage=3,
    strategy=StrategyWrapper(testsma),
    stop_loss=0.005,
    take_profit=0.05,
    max_hold_bars=1000,
    run_once=False
)

