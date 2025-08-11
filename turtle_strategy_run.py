from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import turtle_strategy

auto_trade_futures(
    symbol="ETH/USDT",#幣種
    interval="15m",#週期
    usdt_percent_per_order=1,# 每次使用的資金佔比
    leverage=1,#槓桿
    strategy=turtle_strategy,# 策略
    stop_loss=None,# 停損閾值
    take_profit=None, # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)


