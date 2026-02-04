from StockCalendar import StockCalendar as sc
class Strategy:
    def __init__(self, param):
        self.param = param

    def pick(self):
        pass

    def buy(self):
        pass

    def sell(self):
        pass

class UpStrategy(Strategy):
    def __init__(self, param):
        super().__init__(param)

    def pick(self):
        print("UpStrategy pick stocks")
        pass

    def buy(self):
        pass

    def sell(self):
        pass




class Chain:
    def __init__(self, param=None):
        #回测框架参数
        self.strategies = []
        self.date_arr = []

        if param and "strategy" in param:
            self.strategies = param["strategy"]
        if param and "date_arr" in param:
            self.date_arr = param["date_arr"]

    def execute(self):
        # 遍历策略
        for strategy in self.strategies:
            # 遍历日期区间
            for start_date, end_date in self.date_arr:
                self.execute_one_strategy(strategy, start_date, end_date)

    def execute_one_strategy(self, strategy, start_date, end_date):
        scalendar = sc()
        current_date = scalendar.start(start_date)

        while current_date is not None and current_date <= end_date:
            print(f"Processing date: {current_date}")
            current_date = scalendar.next()
            strategy.buy()
            strategy.sell()
            strategy.pick()
        
if __name__ == "__main__":

    param={
        "strategy":[UpStrategy({})],
        "date_arr":[["20250101","20250630"],["20250701","20251231"]]
    }
    chain = Chain(param=param)
    chain.execute()
