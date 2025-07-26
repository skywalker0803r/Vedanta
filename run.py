from online.auto_trade_future import auto_trade_futures
from Technicalindicatorstrategy import testsma

class StrategyWrapper:
    def __init__(self,n1=5,n2=10):
        self.n1 = n1
        self.n2 = n2
        self.limit = 300

    def get_signals(self, symbol, interval, end_time):
        print(f"⚙️ 呼叫策略：symbol={symbol}, interval={interval}, end_time={end_time}, "
              f"n1={self.n1}, n2={self.n2}")
        return testsma.get_signals(symbol, interval, end_time, limit=self.limit, n1=self.n1,n2=self.n2)

strategy = StrategyWrapper()

auto_trade_futures(symbol="ETH/USDT",interval="15m",usdt_per_order=500,leverage=1,strategy=strategy)