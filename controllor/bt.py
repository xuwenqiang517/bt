import sys
from StockCalendar import StockCalendar as sc
from StockData import StockData as sd
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import NamedTuple, Callable
from datetime import date, datetime
from LocalCache import LocalCache

# 回测结果CSV列名
RESULT_COLS = ['周期胜率', '平均胜率', '平均收益率', '平均最大回撤', '平均交易次数', '平均资金使用率', '配置']


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

BacktestResult=NamedTuple("BacktestResult", [
    ("起始日期", str),
    ("结束日期", str),
    ("初始资金", float),
    ("最终资金", float),
    ("总收益", float),
    ("总收益率", float),
    ("胜率", float),
    ("盈亏比", float),
    ("交易次数", int),
    ("最大回撤", float),
    ("平均资金使用率", float)
])

StrategyBacktestResult=NamedTuple("StrategyBacktestResult", [
    ("策略配置", dict),
    ("平均收益率", float),
    ("平均胜率", float),
    ("周期胜率", float),
    ("平均最大回撤", float)
])
class Strategy:
    def __init__(self, base_param_arr, sell_param_arr, buy_param_arr, debug):
        # 基础参数：[初始资金, 最大持仓数]
        self.base_param_arr = base_param_arr  # 保存原始参数数组
        self.sell_param_arr = sell_param_arr
        self.buy_param_arr = buy_param_arr
        self.init_amount, self.max_hold_count = base_param_arr[0], base_param_arr[1]
        self.free_amount = self.init_amount  # 可用资金
        self.hold = []  # 持仓列表
        self.data = None  # 数据源
        self.calendar = None  # 交易日历实例，由外部传入
        self.today = None  # 当前日期
        self.picked_data = None  # 挑出来的待买的票
        self.pick_condition = None  # 选股条件函数
        self.pick_sort_function = None  # 选股排序函数
        self.trades_history = []  # 存储所有交易历史
        self.daily_values = []  # 存储每日总资产
        self.debug = debug  # 调试模式开关
        

        if self.max_hold_count is None or self.init_amount is None:
            print(f"策略未配置最大持仓数max_hold_count或初始资金init_amount,结束任务")
            sys.exit(1)
    
    def bind(self,data:sd,calendar:sc):
        self.data=data
        self.calendar=calendar


    def pick(self)->pd.DataFrame: 
        # 获取当日股票数据
        today_stock_df = self.data.get_data_by_date(self.today)
        # 执行筛选
        filtered_stocks = today_stock_df[self.pick_condition(today_stock_df)]
        # 处理空结果
        if filtered_stocks is None or filtered_stocks.empty:
            self.picked_data = pd.DataFrame()
            if self.debug:
                print(f"日期 {self.today} 无符合条件股票")
        else:
            self.picked_data = filtered_stocks
            if self.debug:
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
        # 计算每个股票买入金额
        buy_amount_per_stock = self.free_amount / (self.max_hold_count - len(self.hold))
        # 计算买入的票的数量 按今天的开盘价买
        for _, row in self.picked_data.iterrows():
            # 持仓数量够了,跳过买入
            if len(self.hold) >= self.max_hold_count:
                break
            # 已持有的股票不能重复购买
            if any(hold.code == row["code"] for hold in self.hold):
                continue
            next_open = row["next_open"]
            # 只能买100的整数
            buy_count = int(buy_amount_per_stock / next_open) // 100 * 100
            if buy_count <= 0:
                continue
            
            hold_stock = HoldStock(row["code"], next_open, buy_count, self.today, None, None)
            self.hold.append(hold_stock)
            cost = round(buy_count * next_open, 2)
            self.free_amount = round(self.free_amount - cost, 2)
            if self.debug:
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
                if self.debug:
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
                    if self.debug:
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
                
                if self.debug:
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
                if self.debug:
                    print(f"日期 {self.today} 持有 {hold.code} 日期:{self.today} 无数据")
            else:
                if self.debug:
                    print(f"日期 {self.today} 持有 {hold.code} 日期:{hold.buy_day}->{self.today} 价格{hold.buy_price}->{stock_data.close} 累计: {(stock_data.close-hold.buy_price)*hold.buy_count:.2f} ({(stock_data.close - hold.buy_price)/hold.buy_price:.2%})")
                hold_amount += stock_data.close * hold.buy_count
        
        total_value = hold_amount + self.free_amount
        # 记录每日总资产和可用资金（始终记录，用于性能计算）
        self.daily_values.append({'date': self.today, 'value': total_value, 'free_amount': self.free_amount})
        
        if self.debug:
            print(f"日期 {self.today} 持有股票总市值 {hold_amount}, 可用资金 {self.free_amount}, 总资产 {total_value}")
            print("\n")
    
    def update_today(self, today): self.today = today
    # def bind_data(self, data, calendar=None): 
    #     self.data = data
    #     self.calendar = calendar
    
    def calculate_performance(self):
        """计算并返回策略性能指标"""
        if not self.trades_history and not self.hold:
            return {
                'total_return': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'trade_count': 0,
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
        profits = [t['profit'] for t in self.trades_history]
        winning_trades = [p for p in profits if p > 0]
        win_rate = len(winning_trades) / len(profits) if profits else 0

        # 计算盈亏比
        avg_win = sum(winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(-p for p in profits if p < 0) / len([p for p in profits if p < 0]) if any(p < 0 for p in profits) else 1
        profit_loss_ratio = avg_win / avg_loss if avg_loss != 0 else 0

        # 交易次数
        trade_count = len(profits)

        # 计算资金使用率（每日持仓市值 / 总资产）
        values = [dv['value'] for dv in self.daily_values]
        hold_values = [dv['value'] - dv['free_amount'] for dv in self.daily_values]
        avg_utilization = np.mean([h/v for v, h in zip(values, hold_values) if v != 0]) if values else 0

        # 计算最大回撤
        peak, max_drawdown = values[0] if values else 0, 0
        for v in values[1:]:
            if v > peak:
                peak = v
            dd = (peak - v) / peak if peak != 0 else 0
            max_drawdown = max(max_drawdown, dd)
        
        return {
            'init_amount': self.init_amount,
            'final_amount': final_total_value,
            'total_return': total_return,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio,
            'trade_count': trade_count,
            'max_drawdown': max_drawdown,
            'avg_utilization': avg_utilization
        }
    

class UpStrategy(Strategy):
    def __init__(self, base_param_arr, sell_param_arr, buy_param_arr, debug):
        super().__init__(base_param_arr, sell_param_arr, buy_param_arr, debug)
        # 卖出策略链列表
        self.sell_chain_list = self.init_sell_strategy_chain()
        # 选股条件函数
        self.pick_condition = self.get_filter_condition
        # 选股排序函数
        self.pick_sort_function = self.get_sort_function()
        
    def get_filter_condition(self, today_stock_df:pd.DataFrame) -> pd.Series:
        buy_up_day_min = self.buy_param_arr[0]
        buy_day3_min = self.buy_param_arr[1]
        buy_day3_max = self.buy_param_arr[2]
        buy_day5_min = self.buy_param_arr[3]
        buy_day5_max = self.buy_param_arr[4]
        
        if buy_up_day_min is None or buy_day3_min is None or buy_day3_max is None or buy_day5_min is None or buy_day5_max is None:
            print("买入参数配置不完整")
            sys.exit(1)
        # 定义过滤条件
        return (
            (today_stock_df["consecutive_up_days"] >= buy_up_day_min)
            & (today_stock_df["change_3d"] >= buy_day3_min)
            & (today_stock_df["change_3d"] <= buy_day3_max)
            & (today_stock_df["change_5d"] >= buy_day5_min)
            & (today_stock_df["change_5d"] <= buy_day5_max)
        )
    
    def get_sort_function(self) -> Callable:
        """定义排序函数，基类会自动缓存"""
        def sort_by_vol_rank(df: pd.DataFrame) -> pd.DataFrame:
            return df.nsmallest(self.max_hold_count, "vol_rank")
        return sort_by_vol_rank
    
    def init_sell_strategy_chain(self):
        # 从参数中解包：静态止损率、静态止盈率、累计卖出天数、累计卖出最小收益率
        sl, sp, cd, cr = self.sell_param_arr
        strategies = []
        # 静态止损策略
        if sl is not None:
            strategies.append(SellStrategy("静态止损", StopLossParams(rate=sl/100.0)))
        # 静态止盈策略
        if sp is not None:
            strategies.append(SellStrategy("静态止盈", StopProfitParams(rate=sp/100.0)))
        # 累计涨幅卖出策略
        if cd is not None and cr is not None:
            strategies.append(SellStrategy("累计涨幅卖出", CumulativeSellParams(days=cd, min_return=cr/100.0)))
        return strategies
    
        
        

class Chain:
    def __init__(self, param=None):
        self.strategies = param["strategy"]  # 策略列表
        self.date_arr = param["date_arr"]  # 回测时间周期列表
        self.print_report = param.get("print_report", False)  # 是否打印报告
        self.cache = LocalCache()  # 本地缓存
        self.param = param  # 原始参数
        self.stock_data = sd()  # 股票数据源
        self.calendar = sc()  # 交易日历

    def execute(self) -> list:
        cached_df = self.cache.get_csv("a_strategy_results")
        if cached_df is None:
            cached_df = pd.DataFrame(columns=RESULT_COLS)
        executed_keys = set(cached_df['配置'].tolist()) if '配置' in cached_df.columns else set()
        print("已缓存执行策略数:", len(executed_keys)   )
        # 定义个临时pd.DataFrame，用于存储新的行
        temp_df = pd.DataFrame(columns=RESULT_COLS)
        for idx, strategy_params in tqdm(enumerate(self.strategies), desc="执行策略"):
            # 根据参数动态创建策略对象
            strategy = UpStrategy(**strategy_params)
            # 参数用 分隔符 拼在一起作为hash可以唯一标识一个策略，再把3个参数也拼在一起
            param_join_str = "||".join(",".join(map(str, arr)) for arr in [
                strategy.base_param_arr, strategy.buy_param_arr, strategy.sell_param_arr
            ])

            if param_join_str in executed_keys:
                continue

            # 执行策略在不同时间周期上的回测
            strategy_results = [self.execute_one_strategy(strategy, s, e) for s, e in self.date_arr]
            if not strategy_results:
                continue

            r = strategy_results
            win_count = sum(1 for x in r if x.总收益率 > 0)
            total_count = len(r)
            win_rate = win_count / total_count
            # 构建结果行
            new_row = {
                "周期胜率": f"{int(win_rate * 100)}%({win_count}/{total_count})",
                "平均胜率": f"{int(np.mean([x.胜率 for x in r]) * 100)}%",
                "平均收益率": f"{np.mean([x.总收益率 for x in r]) * 100:.2f}%",
                "平均最大回撤": f"{np.mean([x.最大回撤 for x in r]) * 100:.2f}%",
                "平均交易次数": round(np.mean([x.交易次数 for x in r]), 1),
                "平均资金使用率": f"{np.mean([x.平均资金使用率 for x in r]) * 100:.2f}%",
                "配置": param_join_str
            }
            temp_df.loc[len(temp_df)] = new_row
            executed_keys.add(param_join_str)

            if idx % 100 == 0:
                # 每100个策略，合并一次temp_df到cached_df
                if cached_df.empty:
                    cached_df = temp_df.copy()
                else:
                    # 旧缓存数据可能没有新字段，补充NaN
                    for col in ['平均交易次数', '平均资金使用率']:
                        if col not in cached_df.columns:
                            cached_df[col] = np.nan
                    cached_df = pd.concat([cached_df, temp_df], ignore_index=True)
                temp_df = pd.DataFrame(columns=RESULT_COLS)
                self.cache.set_csv("a_strategy_results", cached_df)


    def execute_one_strategy(self, strategy, start_date, end_date) -> BacktestResult:
        scalendar=self.calendar
        current_date = scalendar.start(start_date)
        strategy.bind(self.stock_data,self.calendar)
        
        while current_date is not None and current_date <= end_date:
            current_date = scalendar.next()
            strategy.update_today(current_date)
            strategy.buy()
            strategy.sell()
            strategy.pick()
            strategy.print_daily()
        
        perf = strategy.calculate_performance()
        
        result = BacktestResult(
             起始日期=start_date,
             结束日期=end_date,
             初始资金=perf['init_amount'],
             最终资金=perf['final_amount'],
             总收益=perf['total_return'],
             总收益率=perf['total_return_pct'],
             胜率=perf['win_rate'],
             盈亏比=perf['profit_loss_ratio'],
             交易次数=perf['trade_count'],
             最大回撤=perf['max_drawdown'],
             平均资金使用率=perf['avg_utilization']
         )
        
        if self.print_report:
            print("=" * 50)
            print(f"时间周期: {start_date} 至 {end_date}")
            print(f"资金: {perf['init_amount']:.2f} - > {perf['final_amount']:.2f}")
            print(f"总收益: {perf['total_return']:.2f} ({perf['total_return_pct']:.2%})")
            print(f"胜率: {perf['win_rate']:.2%}")
            print(f"盈亏比: {perf['profit_loss_ratio']:.2f}")
            print(f"交易次数: {perf['trade_count']}")
            print(f"最大回撤: {perf['max_drawdown']:.2%}")
            print(f"资金使用率: {perf['avg_utilization']:.2%}")
        
        return result


        
if __name__ == "__main__":
    start_time=datetime.now().timestamp()*1000
    # 定义策略参数字典列表（不创建对象，省内存）
    strategy_params_list=[]
    for a in range(2,6,1): # 持仓数量
        for buy1 in range(1,4,1): # 连涨天数
            for buy2 in range(1,5,1): # 3日涨幅最低
                for buy3 in range(5,15,1): # 3日涨幅最高
                    for buy4 in range(5,15,5): # 5日涨幅最低
                        for buy5 in range(15,45,5): # 5日涨幅最高
                            for sell1 in range(5,20,3): # 止损率
                                for sell2 in range(15,100,5): # 止盈率
                                    for sell3 in range(3,6,1): # 止盈持有时间
                                        for sell4 in range(10,40,5): # 止盈持有收益率
                                            strategy_params_list.append({
                                                "base_param_arr": [100000, a],
                                                "buy_param_arr": [buy1, buy2, buy3, buy4, buy5],
                                                "sell_param_arr": [sell1, sell2, sell3, sell4],
                                                "debug": 0
                                            })
    print(f"策略参数数量: {len(strategy_params_list)}")
    param={
        "strategy":strategy_params_list
        ,"date_arr":[["20250101","20250201"],["20250201","20250301"],["20250301","20250401"],["20250401","20250501"],["20250501","20250601"]]
        # ,"date_arr":[["20250101","20260101"]]
        ,"print_report":0
    }
    
    chain = Chain(param=param)
    for i in tqdm(range(1)):
        # 执行回测
        chain.execute()


    end_time=datetime.now().timestamp()*1000
    print(f"回测完成 耗时{(end_time-start_time):.2f}ms")