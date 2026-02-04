class Strategy:
    def __init__(self, param):
        self.param = param

    def pick(self):
        pass

    def buy(self):
        pass

    def sell(self):
        pass

# class up_strategy(Strategy):
#     def __init__(self, param):
#         super().__init__(param)

#     def pick(self):
        

#     def buy(self):
        

#     def sell(self):
        




class Chain:
    def __init__(self):
        self.strategies = []

    def add_strategy(self, strategy):
        self.strategies.append(strategy)

    def execute(self, start_date=None,end_date=None):
        for strategy in self.strategies:
            strategy.pick()
            strategy.buy()
            strategy.sell()
        
if __name__ == "__main__":

    chain = Chain()
    # chain.add_strategy(Strategy("Strategy A"))
    chain.execute(start_date="20250101", end_date="20251231")
