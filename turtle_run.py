from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import turtle_strategy

# 1
auto_trade_futures(
    symbol="ETH/USDT",#幣種
    interval="4h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=10,#槓桿
    strategy=turtle_strategy,# 策略
    stop_loss = 0.01,       # 停損閾值，例如0.05代表5%
    take_profit = 0.17,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)

# 2
auto_trade_futures(
    symbol="BTC/USDT",#幣種
    interval="4h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=10,#槓桿
    strategy=turtle_strategy,# 策略
    stop_loss = 0.01,       # 停損閾值，例如0.05代表5%
    take_profit = 0.17,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)

# 3
auto_trade_futures(
    symbol="XRP/USDT",#幣種
    interval="4h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=10,#槓桿
    strategy=turtle_strategy,# 策略
    stop_loss = 0.01,       # 停損閾值，例如0.05代表5%
    take_profit = 0.17,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)