from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import sma

class StrategyWrapper:
    def __init__(self, n1=60, n2=35, limit=1000):
        self.n1 = n1
        self.n2 = n2
        self.limit = limit

    def get_signals(self, symbol, interval, end_time):
        print(
            f"⚙️ 呼叫策略："
            f"symbol={symbol}, interval={interval}, end_time={end_time}, "
            f"n1={self.n1}, n2={self.n2}"
        )
        return sma.get_signals(
            symbol=symbol,
            interval=interval,
            end_time=end_time,
            limit=self.limit,
            n1=self.n1,
            n2=self.n2
        )

# 初始化策略實例
strategy = StrategyWrapper()


auto_trade_futures(symbol="ETH/USDT",interval="15m",usdt_per_order=500,leverage=10,strategy=strategy)