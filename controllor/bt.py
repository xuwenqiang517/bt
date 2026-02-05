from StockCalendar import StockCalendar as sc
from StockData import StockData as sd
import pandas as pd
import numpy as np
from tqdm import tqdm
class HoldStock:
    def __init__(self, code, name, buy_price, buy_count):
        self.code = code
        self.name = name
        self.buy_price = buy_price
        self.buy_count = buy_count

class Strategy:
    def __init__(self, param):
        self.param = param
        self.init_amount = param.get("init_amount", 1000000)
        # 最大持仓数
        self.max_hold_count = param.get("max_hold_count")
        # 可用资金
        self.free_amount = self.init_amount
        # 持仓列表
        self.hold=[]
        # 数据源
        self.data=None
        # 当前日期
        self.today=None
        # 挑出来的待买的票
        self.picked_data=None


    def pick(self): self.picked_data=self.doPick(self.data.get_data_by_date(self.today))
    def doPick(self):pass

    def buy(self):
        # 没选出来票，不买
        if self.picked_data is None or self.picked_data.empty:
            print(f"日期 {self.today} 没有选出股票，跳过买入")
            return
        # 达到最大持仓了，不买
        if len(self.hold) >= self.max_hold_count:
            print(f"日期 {self.today} 达到最大持仓数 {self.max_hold_count}，跳过买入")
            return
        #计算每个股票买入金额
        buy_amount_per_stock = self.free_amount / (self.max_hold_count - len(self.hold))
        #计算买入的票的数量 按今天的开盘价买
        for _, row in self.picked_data.iterrows():
            #持仓数量够了，跳过买入
            if len(self.hold) >= self.max_hold_count:
                print(f"日期 {self.today} 达到最大持仓数 {self.max_hold_count}，跳过后续买入")
                break
            next_open=row["next_open"]
            buy_count= int(buy_amount_per_stock / next_open)
            if buy_count <=0:
                print(f"日期 {self.today} 股票 {row['code']} 开盘价 {next_open} 太高，买入数量为0，跳过买入")
                continue
            else:
                self.hold.append(HoldStock(row["code"], row["name"], next_open, buy_count))
                cost = buy_count * next_open
                self.free_amount -= cost
                print(f"日期 {self.today} 买入股票 {row['code']} {row['name']}，买入价格 {next_open}，买入数量 {buy_count}，花费金额 {cost}，剩余资金 {self.free_amount}")

    def sell(self):pass
    def doSell(self):pass
    
    def update_today(self, today): self.today = today
    def bind_data(self, data): self.data=data

class UpStrategy(Strategy):
    def doPick(self,today_stock_df:pd.DataFrame)->pd.DataFrame:
        # 选出涨幅前10的股票
        if today_stock_df.empty:
            print(f"日期 {self.today} 没有股票数据，跳过选股")
            return
        top_stocks = today_stock_df.nlargest(10, 'change_pct')
        print(f"日期 {self.today} 涨幅前10的股票：")
        print(top_stocks[['code', 'change_pct']])
        return top_stocks
    def doSell(self):
        return super().doSell()

        




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
        # 数据
        stock_data = sd()
        # 遍历策略
        for strategy in self.strategies:
            strategy.bind_data(stock_data)
            # 遍历日期区间
            for start_date, end_date in self.date_arr:
                self.execute_one_strategy(strategy, start_date, end_date)

    def execute_one_strategy(self, strategy, start_date, end_date):
        scalendar = sc()
        current_date = scalendar.start(start_date)

        while current_date is not None and current_date <= end_date:
            current_date = scalendar.next()
            strategy.update_today(current_date)
            strategy.buy()
            strategy.sell()
            strategy.pick()
        
if __name__ == "__main__":

    param={
        "strategy":[UpStrategy({})]
        ,"1date_arr":[["20250101","20250630"],["20250701","20251231"]]
        ,"date_arr":[["20250101","20250230"]]
    }
    chain = Chain(param=param)
    chain.execute()
