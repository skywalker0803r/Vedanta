from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import vegas

auto_trade_futures(
    symbol="ETH/USDT",#幣種
    interval="1h",#週期
    usdt_percent_per_order=1,# 每次使用的資金佔比
    leverage=10, # 槓桿
    strategy=vegas,# 策略
    stop_loss=0.001,# 停損閾值
    take_profit = 0.017, # 停利閾值
    max_hold_bars=1000,# 最大持有K棒數
    run_once=True #單次執行模式
)