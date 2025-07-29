from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import testsma

auto_trade_futures(
    symbol="ETH/USDT",
    interval="1m",
    usdt_percent_per_order=0.1,  # 10%
    leverage=3,
    strategy=testsma,
    stop_loss=0.005,
    take_profit=0.05,
    max_hold_bars=1000,
    run_once=False
)

