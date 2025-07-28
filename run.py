from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import vegas

auto_trade_futures(symbol="ETH/USDT",interval="1h",usdt_per_order=500,leverage=5,strategy=vegas)

