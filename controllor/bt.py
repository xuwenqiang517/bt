import sys
from StockCalendar import StockCalendar as sc
from StockData import StockData as sd
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import NamedTuple, Callable
from datetime import date, datetime


HoldStock=NamedTuple("HoldStock", [
    ("code", str),
    ("buy_price", float),
    ("buy_count", int),
    ("buy_day", str),
    ("sell_price", float),
    ("sell_day", str)
])

SellStrategy=NamedTuple("SellStrategy", [
    ("name", str),
    ("params", object)  # 改为object类型，支持不同类型的参数对象
])

StopLossParams=NamedTuple("StopLossParams", [
    ("rate", float)
])

StopProfitParams=NamedTuple("StopProfitParams", [
    ("rate", float)
])

CumulativeSellParams=NamedTuple("CumulativeSellParams", [
    ("days", int),
    ("min_return", float)
])
class Strategy:
    def __init__(self, param):
        self.param = param
        self.init_amount = param.get("init_amount")
        self.max_hold_count = param.get("max_hold_count") # 最大持仓数
        self.print_log= param.get("print_log")
        self.free_amount = self.init_amount # 可用资金
        self.hold=[] # 持仓列表
        self.data=None # 数据源
        self.today=None # 当前日期
        self.picked_data=None # 挑出来的待买的票
        self.sell_chain_list = [] #卖出策略链
        self.pick_condition=None
        self.pick_sort_function=None
        
        self.trades_history = []  # 存储所有交易历史
        self.daily_values = []    # 存储每日总资产
        self.calendar = None  # 交易日历实例，由外部传入
        
        # 在初始化时一次性创建好排序函数，避免重复创建
        # 如果子类实现了get_sort_function则调用，否则设为None
        self._sort_function = self.get_sort_function() if hasattr(self, 'get_sort_function') else None
        
        # 在初始化时预创建过滤参数（如果子类需要）
        self._filter_params = self._get_filter_params() if hasattr(self, '_get_filter_params') else None
        
        #未配置项做检查
        if self.max_hold_count is None:
            print(f"策略未配置最大持仓数max_hold_count,结束任务")
            sys.exit(1)
        if self.init_amount is None:
            print(f"策略未配置初始资金init_amount,结束任务")
            sys.exit(1)
        


    def pick(self)->pd.DataFrame: 
        # 获取当日股票数据
        today_stock_df = self.data.get_data_by_date(self.today)
        # 执行筛选
        filtered_stocks = today_stock_df[self.pick_condition(today_stock_df)]
        # 处理空结果
        if filtered_stocks is None or filtered_stocks.empty:
            self.picked_data = pd.DataFrame()
            if self.print_log:
                print(f"日期 {self.today} 无符合条件股票")
        else:
            self.picked_data = filtered_stocks
            if self.print_log:
                print(f"日期 {self.today} 选出股票 {len(filtered_stocks)} 只")
        # 如果筛选结果为空，返回空DataFrame
        if filtered_stocks.empty:
            return pd.DataFrame()
        
        return self.pick_sort_function(filtered_stocks)
    

    def buy(self):
        # 没选出来票,不买
        if self.picked_data is None or self.picked_data.empty:
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
            # 已持有的股票不能重复购买
            if any(hold.code == row["code"] for hold in self.hold):
                continue
            next_open=row["next_open"]
            #只能买100的整数
            buy_count= int(buy_amount_per_stock / next_open) // 100 * 100
            if buy_count <=0:
                continue
            
            hold_stock = HoldStock(row["code"], next_open, buy_count, self.today, None, None)
            self.hold.append(hold_stock)
            cost = round(buy_count * next_open, 2)
            self.free_amount = round(self.free_amount - cost, 2)
            if self.print_log:
                print(f"日期 {self.today} 买入 {row['code']} , {next_open} * {buy_count} = {cost} ,剩余资金 {self.free_amount}")

    def sell(self):
        if not self.hold:
            return
        today=self.today
        sells_info=[]
        for hold in self.hold:
            code=hold.code
            if hold.buy_day==today:
                continue
            stock_data=self.data.get_data_by_date_code(today,code)
            if stock_data is None or (hasattr(stock_data, 'empty') and stock_data.empty):
                if self.print_log:
                    print(f"日期 {today} 没有找到股票 {code} 的数据,跳过卖出")
                continue
            # 策略决定判断是否要卖掉这个票
            need_sell, sell_price, reason = False, 0, ""
            for sell_strategy in self.sell_chain_list:
                sell_name = sell_strategy.name
                params = sell_strategy.params
                
                if sell_name=="静态止损": # 绿色原因
                    need_sell, sell_price, reason = self.stop_loss(hold, stock_data, params)
                    reason=f"\033[92m{reason}\033[0m"
                elif sell_name=="静态止盈": #红色原因
                    need_sell, sell_price, reason = self.stop_profit(hold, stock_data, params)
                    reason=f"\033[91m{reason}\033[0m"
                elif sell_name=="累计涨幅卖出": #黄色原因
                    need_sell, sell_price, reason = self.cumulative_return_sell(hold, stock_data, params)
                    reason=f"\033[93m{reason}\033[0m"
                else:
                    if self.print_log:
                        print(f"日期 {today} 未知的卖出策略 {sell_name},跳过")
                    continue
                
                # 如果某个策略触发卖出，则不再检查其他策略
                if need_sell:
                    break
                    
            if need_sell:
                sells_info.append((code, sell_price, reason))
                continue

        if len(sells_info) == 0:
            return
        
        for code, sell_price , sell_reason in sells_info:
            hold_to_remove = None
            for hold in self.hold:
                if hold.code == code:
                    hold_to_remove = hold
                    break
            if hold_to_remove:
                profit = round((sell_price - hold_to_remove.buy_price) * hold_to_remove.buy_count, 2)
                self.hold.remove(hold_to_remove)
                self.free_amount = round(self.free_amount + sell_price * hold_to_remove.buy_count, 2)
                profit_rate = profit / (hold_to_remove.buy_price * hold_to_remove.buy_count) if hold_to_remove.buy_price * hold_to_remove.buy_count > 0 else 0
                
                # 记录交易历史
                self.trades_history.append({
                    'date': self.today,
                    'code': code,
                    'buy_date': hold_to_remove.buy_day,
                    'buy_price': hold_to_remove.buy_price,
                    'sell_price': sell_price,
                    'quantity': hold_to_remove.buy_count,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'reason': sell_reason
                })
                
                if self.print_log:
                    print(f"日期 {self.today} 卖出 {code} {hold_to_remove.buy_day}->{self.today} {hold_to_remove.buy_price} -> {sell_price} 原因:{sell_reason} 盈亏 {profit}({profit_rate:.2%}), 剩余资金 {self.free_amount}")
                

    def stop_loss(self, hold:HoldStock, stock_data:pd.DataFrame,params:StopLossParams)->tuple:
        """
        触发固定阈值的止损卖出
        """
        #计算止损价
        stop_loss_price=round(hold.buy_price * (1 + params.rate), 2)
        if stock_data.open <= stop_loss_price:
            return True, stock_data.open,f"开盘价{stock_data.open}<{stop_loss_price}({params.rate:.2%}),以开盘价{stock_data.open}卖出"
        elif stock_data.low <= stop_loss_price:
            return True, stop_loss_price,f"盘中最低价{stock_data.low}<{stop_loss_price}({params.rate:.2%}),以止损价{stop_loss_price}卖出"
        return False, 0, ""
    
    def stop_profit(self, hold:HoldStock, stock_data:pd.DataFrame,params:StopProfitParams)->tuple:
        """
        触发固定阈值的止盈卖出
        """
        #计算止盈价
        stop_profit_price=round(hold.buy_price * (1 + params.rate), 2)
        if stock_data.open >= stop_profit_price:
            return True, stock_data.open,f"开盘价{stock_data.open}>止盈价{stop_profit_price}({params.rate:.2%}),以开盘价{stock_data.open}卖出"
        elif stock_data.high >= stop_profit_price:
            return True, stop_profit_price,f"盘中最高价{stock_data.high}>止盈价{stop_profit_price}({params.rate:.2%}),以止盈价{stop_profit_price}卖出"
        return False, 0, ""

    def cumulative_return_sell(self, hold:HoldStock, stock_data:pd.DataFrame, params:CumulativeSellParams)->tuple:
        """
        x天累计涨幅没到y，以持仓最后一天的开盘价卖掉
        params: CumulativeSellParams(days=x, min_return=y)
        """
        # 直接使用StockCalendar的gap函数计算持仓天数
        hold_days = self.calendar.gap(hold.buy_day, self.today) if self.calendar else 0
        
        # 如果持仓天数小于x天，不触发卖出
        if hold_days < params.days:
            return False, 0, ""
        
        # 计算累计涨幅
        cumulative_return = (stock_data.close - hold.buy_price) / hold.buy_price
        
        # 如果累计涨幅没达到最小要求，以开盘价卖出
        if cumulative_return < params.min_return:
            return True, stock_data.open, f"持仓{params.days}天 累计涨幅{cumulative_return:.2%}<{params.min_return:.2%}，以开盘价{stock_data.open}卖出"
        
        return False, 0, ""


    def print_daily(self): 
        # 计算每日总资产（用于性能计算，无论是否打印日志）
        hold_amount = 0
        for hold in self.hold:
            stock_data = self.data.get_data_by_date_code(self.today, hold.code)
            # Check if stock_data is empty (DataFrame with no rows) or None
            if stock_data is None or (hasattr(stock_data, 'empty') and stock_data.empty):
                if self.print_log:
                    print(f"日期 {self.today} 持有 {hold.code} 日期:{self.today} 无数据")
            else:
                if self.print_log:
                    print(f"日期 {self.today} 持有 {hold.code} 日期:{hold.buy_day}->{self.today} 价格{hold.buy_price}->{stock_data.close} 累计: {(stock_data.close-hold.buy_price)*hold.buy_count:.2f} ({(stock_data.close - hold.buy_price)/hold.buy_price:.2%})")
                hold_amount += stock_data.close * hold.buy_count
        
        total_value = hold_amount + self.free_amount
        # 记录每日总资产（始终记录，用于性能计算）
        self.daily_values.append({'date': self.today, 'value': total_value})
        
        if self.print_log:
            print(f"日期 {self.today} 持有股票总市值 {hold_amount}, 可用资金 {self.free_amount}, 总资产 {total_value}")
            print("\n")
    
    def update_today(self, today): self.today = today
    def bind_data(self, data, calendar=None): 
        self.data = data
        self.calendar = calendar
    
    def calculate_performance(self):
        """计算并返回策略性能指标"""
        if not self.trades_history and not self.hold:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'trade_count': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0
            }
        
        # 计算最终投资组合价值（当前持有股票价值 + 现金）
        final_holdings_value = 0
        for hold in self.hold:
            stock_data = self.data.get_data_by_date_code(self.today, hold.code)
            # Check if stock_data is empty (DataFrame with no rows) or None
            if stock_data is None or (hasattr(stock_data, 'empty') and stock_data.empty):
                # 如果今天没有数据，使用买入价格作为估值
                final_holdings_value += hold.buy_price * hold.buy_count
            else:
                final_holdings_value += stock_data.close * hold.buy_count
        
        # 最终总价值 = 持仓价值 + 可用现金
        final_total_value = final_holdings_value + self.free_amount
        
        # 总收益 = 最终总价值 - 初始资本
        total_return = final_total_value - self.init_amount
        total_return_pct = total_return / self.init_amount
        
        # 计算胜率
        winning_trades = [trade for trade in self.trades_history if trade['profit'] > 0]
        win_rate = len(winning_trades) / len(self.trades_history) if self.trades_history else 0
        
        # 计算盈亏比
        winning_profits = [trade['profit'] for trade in winning_trades]
        losing_losses = [abs(trade['profit']) for trade in self.trades_history if trade['profit'] < 0]
        
        avg_win = sum(winning_profits) / len(winning_profits) if winning_profits else 0
        avg_loss = sum(losing_losses) / len(losing_losses) if losing_losses else 1  # Avoid division by zero
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0
        
        # 交易次数
        trade_count = len(self.trades_history)
        
        # 计算夏普比率
        if len(self.daily_values) > 1:
            daily_returns = []
            for i in range(1, len(self.daily_values)):
                prev_value = self.daily_values[i-1]['value']
                curr_value = self.daily_values[i]['value']
                daily_return = (curr_value - prev_value) / prev_value if prev_value != 0 else 0
                daily_returns.append(daily_return)
            
            if daily_returns:
                daily_std = np.std(daily_returns)
                avg_daily_return = np.mean(daily_returns)
                # 年化夏普比率 = (日均收益率 / 日收益率标准差) * sqrt(252)
                sharpe_ratio = (avg_daily_return / daily_std * np.sqrt(252)) if daily_std != 0 else 0
            else:
                sharpe_ratio = 0
        else:
            sharpe_ratio = 0
        
        # 计算最大回撤
        if self.daily_values:
            values = [dv['value'] for dv in self.daily_values]
            peak = values[0]
            max_dd = 0
            for value in values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak if peak != 0 else 0
                if dd > max_dd:
                    max_dd = dd
            max_drawdown = max_dd
        else:
            max_drawdown = 0
        
        return {
            'init_amount': self.init_amount,
            'final_amount': final_total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'trade_count': trade_count,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    

class UpStrategy(Strategy):
    def __init__(self, param):
        super().__init__(param)
        # 初始化卖出策略链
        self.sell_chain_list = self.init_sell_strategy_chain()
        self.pick_condition = self.get_filter_condition
        self.pick_sort_function = self.get_sort_function()
        
        
    def get_filter_condition(self, today_stock_df:pd.DataFrame) -> pd.Series:
        buy_up_day_min = self.param.get("buy_up_day_min")
        buy_day3_min = self.param.get("buy_day3_min")
        buy_day5_min = self.param.get("buy_day5_min")
        if buy_up_day_min is None or buy_day3_min is None or buy_day5_min is None:
            print("买入参数配置不完整")
            sys.exit(1)
        # 定义过滤条件
        return (
            (today_stock_df["consecutive_up_days"] >= buy_up_day_min)
            & (today_stock_df["change_3d"] >= buy_day3_min)
            & (today_stock_df["change_5d"] >= buy_day5_min)
        )
    
    def get_sort_function(self) -> Callable:
        """定义排序函数，基类会自动缓存"""
        def sort_by_vol_rank(df: pd.DataFrame) -> pd.DataFrame:
            return df.nsmallest(self.max_hold_count, "vol_rank")
        return sort_by_vol_rank
    
    def init_sell_strategy_chain(self):
        # 直接在方法内部处理卖出参数，避免在init中存储中间变量
        sell_strategies = []
        
        # 静态止损策略
        sell_stop_loss_rate = self.param.get("sell_stop_loss_rate")
        if sell_stop_loss_rate is not None:
            stop_loss_params = StopLossParams(rate=sell_stop_loss_rate / 100.0)
            sell_strategies.append(SellStrategy("静态止损", stop_loss_params))
        
        # 静态止盈策略
        sell_stop_profit_rate = self.param.get("sell_stop_profit_rate")
        if sell_stop_profit_rate is not None:
            stop_profit_params = StopProfitParams(rate=sell_stop_profit_rate / 100.0)
            sell_strategies.append(SellStrategy("静态止盈", stop_profit_params))
        
        # 累计涨幅卖出策略
        cumulative_sell_days = self.param.get("sell_cumulative_days")
        cumulative_sell_min_return = self.param.get("sell_cumulative_min_return")
        if cumulative_sell_days is not None and cumulative_sell_min_return is not None:
            cumulative_params = CumulativeSellParams(
                days=cumulative_sell_days,
                min_return=cumulative_sell_min_return / 100.0
            )
            sell_strategies.append(SellStrategy("累计涨幅卖出", cumulative_params))
        
        return sell_strategies
    
        
        




class Chain:
    def __init__(self, param=None):
        #回测框架参数
        self.strategies = []
        self.date_arr = []
        self.print_report = param.get("print_report", False) if param else False

        if param and "strategy" in param:
            self.strategies = param["strategy"]
        if param and "date_arr" in param:
            self.date_arr = param["date_arr"]

    def execute(self,stock_data,scalendar):
        # 遍历策略
        for strategy in self.strategies:
            strategy.bind_data(stock_data, scalendar)
            # 遍历日期区间
            for start_date, end_date in self.date_arr:
                self.execute_one_strategy(strategy, start_date, end_date,scalendar)

    def execute_one_strategy(self, strategy, start_date, end_date,scalendar):
        current_date = scalendar.start(start_date)
        while current_date is not None and current_date <= end_date:
            current_date = scalendar.next()
            strategy.update_today(current_date)
            strategy.buy()
            strategy.sell()
            strategy.pick()
            strategy.print_daily()
        
        # 在策略执行结束后打印性能报告
        perf = strategy.calculate_performance()
        # 打印策略报告
        if not self.print_report:
            return
        print("=" * 50)
        print(f"时间周期: {start_date} 至 {end_date}")
        print(f"资金: {perf['init_amount']:.2f} - > {perf['final_amount']:.2f}")
        print(f"总收益: {perf['total_return']:.2f} ({perf['total_return_pct']:.2%})")
        print(f"胜率: {perf['win_rate']:.2%}")
        print(f"盈亏比: {perf['profit_loss_ratio']:.2f}")
        print(f"交易次数: {perf['trade_count']}")
        print(f"夏普比率: {perf['sharpe_ratio']:.2f}")
        print(f"最大回撤: {perf['max_drawdown']:.2%}")


        
if __name__ == "__main__":


    param={
        "strategy":[UpStrategy({"init_amount":100000,"max_hold_count":5
                                # 卖出参数：统一使用sell_前缀
                                ,"sell_stop_loss_rate":-8  # 统一使用整数百分比：-8表示-8%
                                ,"sell_stop_profit_rate":15  # 统一使用整数百分比：15表示15%
                                ,"sell_cumulative_days":3
                                ,"sell_cumulative_min_return":5  # 统一使用整数百分比：5表示5%
                                # 买入参数：统一使用buy_前缀
                                ,"buy_up_day_min":2
                                ,"buy_day3_min":5  # 统一使用整数百分比：5表示5%
                                ,"buy_day3_max":10  # 统一使用整数百分比：10表示10%
                                ,"buy_day5_min":5  # 统一使用整数百分比：5表示5%
                                ,"buy_day5_max":10  # 统一使用整数百分比：10表示10%
                                ,"print_log":True
                                })]
        # ,"date_arr":[["20250101","20250630"],["20250701","20251231"]]
        ,"date_arr":[["20250101","20250111"]]
        ,"print_report":True
    }
    
    start_time=datetime.now().timestamp()*1000
    chain = Chain(param=param)
    # 数据
    stock_data = sd()
    scalendar = sc()
    for i in tqdm(range(1)):
        chain.execute(stock_data,scalendar)
    end_time=datetime.now().timestamp()*1000
    print(f"回测完成 耗时{(end_time-start_time):.2f}ms")