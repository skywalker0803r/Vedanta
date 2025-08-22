from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import TurtleMACDTimeZoneFilter

# 1
auto_trade_futures(
    symbol="ETH/USDT",#幣種
    interval="2h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=7,#槓桿
    strategy=TurtleMACDTimeZoneFilter,# 策略
    stop_loss = None,       # 停損閾值，例如0.05代表5%
    take_profit = None,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)

# 2
auto_trade_futures(
    symbol="BTC/USDT",#幣種
    interval="2h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=7,#槓桿
    strategy=TurtleMACDTimeZoneFilter,# 策略
    stop_loss = None,       # 停損閾值，例如0.05代表5%
    take_profit = None,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)

# 3
auto_trade_futures(
    symbol="XRP/USDT",#幣種
    interval="2h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=7,#槓桿
    strategy=TurtleMACDTimeZoneFilter,# 策略
    stop_loss = None,       # 停損閾值，例如0.05代表5%
    take_profit = None,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)


# 4
auto_trade_futures(
    symbol="BNB/USDT",#幣種
    interval="2h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=7,#槓桿
    strategy=TurtleMACDTimeZoneFilter,# 策略
    stop_loss = None,       # 停損閾值，例如0.05代表5%
    take_profit = None,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)


# 5
auto_trade_futures(
    symbol="ADA/USDT",#幣種
    interval="2h",#週期
    usdt_percent_per_order=0.1,# 每次使用的資金佔比
    leverage=7,#槓桿
    strategy=TurtleMACDTimeZoneFilter,# 策略
    stop_loss = None,       # 停損閾值，例如0.05代表5%
    take_profit = None,     # 停利閾值
    max_hold_bars=100000,# 最大持有K棒數
    run_once=True #單次執行模式
)