from calendar import c
import sys
from StockCalendar import StockCalendar as sc
from StockData import StockData as sd
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import NamedTuple


HoldStock=NamedTuple("HoldStock", [
    ("code", str),
    ("buy_price", float),
    ("buy_count", int),
    ("buy_day", str),
    ("sell_price", float),
    ("sell_day", str)
])
    # def __init__(self, code, name, buy_price, buy_count, buy_day, sell_price=None, sell_day=None):
    #     self.code = code
    #     self.name = name
    #     self.buy_price = buy_price
    #     self.buy_count = buy_count
    #     self.buy_day = buy_day
    #     self.sell_price = sell_price
    #     self.sell_day = sell_day

class Strategy:
    def __init__(self, param):
        self.param = param
        self.init_amount = param.get("init_amount")
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

        #未配置项做检查
        if self.max_hold_count is None:
            print(f"策略未配置最大持仓数max_hold_count,结束任务")
            sys.exit(1)
        if self.init_amount is None:
            print(f"策略未配置初始资金init_amount,结束任务")
            sys.exit(1)
        


    def pick(self): self.picked_data=self.doPick(self.data.get_data_by_date(self.today))
    def doPick(self):pass

    def buy(self):
        # 没选出来票,不买
        if self.picked_data is None or self.picked_data.empty:
            print(f"日期 {self.today} 没有选出股票,跳过买入")
            return
        # 达到最大持仓了,不买
        if len(self.hold) >= self.max_hold_count:
            return
        #计算每个股票买入金额
        buy_amount_per_stock = self.free_amount / (self.max_hold_count - len(self.hold))
        #计算买入的票的数量 按今天的开盘价买
        for _, row in self.picked_data.iterrows():
            #持仓数量够了,跳过买入
            if len(self.hold) >= self.max_hold_count:
                break
            next_open=row["next_open"]
            #只能买100的整数
            buy_count= int(buy_amount_per_stock / next_open) // 100 * 100
            if buy_count <=0:
                print(f"日期 {self.today} 股票 {row['code']} 开盘价 {next_open} 太高,买入数量为0,跳过买入")
                continue
            else:
                hold_stock = HoldStock(row["code"], next_open, buy_count, self.today, None, None)
                self.hold.append(hold_stock)
                cost = buy_count * next_open
                self.free_amount -= cost
                print(f"日期 {self.today} 买入股票 {row['code']} , 买入价格 {next_open},买入数量 {buy_count},花费金额 {cost},剩余资金 {self.free_amount}")

    def sell(self):
        if not self.hold:
            return
        sells_info = self.doSell()
        if len(sells_info) == 0:
            return
        
        for code, sell_price , sell_reason in sells_info:
            hold_to_remove = None
            for hold in self.hold:
                if hold.code == code:
                    hold_to_remove = hold
                    break
            if hold_to_remove:
                profit = (sell_price - hold_to_remove.buy_price) * hold_to_remove.buy_count
                self.free_amount += sell_price * hold_to_remove.buy_count
                profit_rate = profit / (hold_to_remove.buy_price * hold_to_remove.buy_count) if hold_to_remove.buy_price * hold_to_remove.buy_count > 0 else 0
                print(f"日期 {self.today} 卖出股票 {code} {hold_to_remove.buy_day}->{self.today} 买入价 {hold_to_remove.buy_price}  卖出价格 {sell_price}  卖出原因 {sell_reason}  利润 {profit} 盈亏比例{profit_rate:.2%}, 剩余资金 {self.free_amount}")
                self.hold.remove(hold_to_remove)
    def doSell(self):pass
    
    def update_today(self, today): self.today = today
    def bind_data(self, data): self.data=data

class UpStrategy(Strategy):
    def doPick(self,today_stock_df:pd.DataFrame)->pd.DataFrame:
        # 选出涨幅前10的股票
        if today_stock_df.empty:
            print(f"日期 {self.today} 没有股票数据,跳过选股")
            return
        top_stocks = today_stock_df.nlargest(10, 'change_pct')
        # print(f"日期 {self.today} 涨幅前10的股票：")
        # print(top_stocks[['code', 'change_pct']])
        return top_stocks
    
    def doSell(self):
        today=self.today
        # 止损率 从配置param取
        stop_loss_rate= self.param.get("stop_loss_rate") 
        # 判断止损率是否配置
        if stop_loss_rate is None:
            print(f"策略未配置止损率stop_loss_rate,结束任务")
            sys.exit(1)

        sells_info=[]
        for hold in self.hold:
            code=hold.code
            buy_price=hold.buy_price
            if hold.buy_day==today:
                continue
            stock_data=self.data.get_data_by_date_code(today,code)
            if stock_data is None:
                print(f"日期 {today} 没有找到股票 {code} 的数据,跳过卖出")
                continue
            #计算止损价
            stop_loss_price=buy_price * (1 + stop_loss_rate)
            # 最低价达到止损价,卖出
            if stock_data.open<= stop_loss_price:
                sells_info.append((code, stock_data.open,f"开盘价{stock_data.open}低于止损价{stop_loss_price}"))
                continue
            if stock_data.low <= stop_loss_price:
                sells_info.append((code, stop_loss_price,f"最低价{stock_data.low}低于止损价{stop_loss_price}"))
                continue
            if stock_data.open >= buy_price * 1.15:
                sells_info.append((code, stock_data.open,f"{today}开盘价{stock_data.open}值盈利15%卖出"))
                continue
        return sells_info

        




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
        "strategy":[UpStrategy({"init_amount":100000,"max_hold_count":5,"stop_loss_rate":-0.03})]
        ,"1date_arr":[["20250101","20250630"],["20250701","20251231"]]
        ,"date_arr":[["20250101","20250111"]]
    }
    chain = Chain(param=param)
    chain.execute()
