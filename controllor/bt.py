import random

import sys
from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import NamedTuple, Callable
from datetime import date, datetime
from local_cache import LocalCache

# 回测结果CSV列名
RESULT_COLS_A = ['周期胜率', '平均胜率', '平均收益率', '平均最大回撤', '平均交易次数', '平均资金使用率', '配置']
RESULT_COLS_B = ['配置']


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
        self.base_param_arr = base_param_arr
        self.sell_param_arr = sell_param_arr
        self.buy_param_arr = buy_param_arr
        self.init_amount, self.max_hold_count = base_param_arr[0], base_param_arr[1]
        self.free_amount = self.init_amount
        self.hold = []
        self.hold_codes = set()
        self.data = None
        self.calendar = None
        self.today = None
        self.picked_data = None
        self.trades_history = []
        self.daily_values = []
        self.debug = debug
        self._today_data_cache = {}
        self._init_pick_filter()
        self._init_pick_sorter()
        
        if self.max_hold_count is None or self.init_amount is None:
            print(f"策略未配置最大持仓数max_hold_count或初始资金init_amount,结束任务")
            sys.exit(1)
    
    def _init_pick_filter(self):
        """初始化筛选函数，子类重写"""
        self._pick_filter = self._default_pick_filter
    
    def _default_pick_filter(self, df: pd.DataFrame) -> np.ndarray:
        """默认筛选：返回所有股票"""
        return np.ones(len(df), dtype=bool)
    
    def _init_pick_sorter(self):
        """初始化排序函数，子类重写"""
        self._pick_sorter = self._default_pick_sorter
    
    def _default_pick_sorter(self, df: pd.DataFrame) -> pd.DataFrame:
        """默认排序：返回原数据"""
        return df
    
    def bind(self,data:sd,calendar:sc):
        self.data=data
        self.calendar=calendar

    def reset(self):
        self.free_amount = self.init_amount
        self.hold = []
        self.hold_codes = set()
        self.picked_data = None
        self.trades_history = []
        self.daily_values = []
        self.today = None
        self._today_data_cache = {}
        self._new_day()

    def _new_day(self):
        self._today_data_cache = {}
    
    def _ensure_today_data_loaded(self):
        if not self.hold:
            return
        today = self.today
        for hold in self.hold:
            if hold.code not in self._today_data_cache:
                self._today_data_cache[hold.code] = self.data.get_data_by_date_code(today, hold.code)
    
    def _add_hold(self, hold_stock: HoldStock):
        self.hold.append(hold_stock)
        self.hold_codes.add(hold_stock.code)
    
    def _remove_hold(self, code: str) -> HoldStock:
        for i, hold in enumerate(self.hold):
            if hold.code == code:
                self.hold.pop(i)
                self.hold_codes.discard(code)
                return hold
        return None
    
    def pick(self)->pd.DataFrame: 
        today_stock_df = self.data.get_data_by_date(self.today)
        
        mask = self._pick_filter(today_stock_df)
        filtered_stocks = today_stock_df[mask]
        
        if filtered_stocks is None or filtered_stocks.empty:
            self.picked_data = pd.DataFrame()
            if self.debug:
                print(f"日期 {self.today} 无符合条件股票")
            return pd.DataFrame()
        
        self.picked_data = filtered_stocks
        result = self._pick_sorter(filtered_stocks)
        
        if self.debug:
            print(f"日期 {self.today} 选出股票 {len(filtered_stocks)} 只")
            print(f"前5只 {result.head(5)}")

        
        return result
    

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
            # 已持有的股票不能重复购买（O(1)查找）
            if row["code"] in self.hold_codes:
                continue
            next_open = row["next_open"]
            # 只能买100的整数
            buy_count = int(buy_amount_per_stock / next_open) // 100 * 100
            if buy_count <= 0:
                continue
            
            hold_stock = HoldStock(row["code"], next_open, buy_count, self.today, None, None)
            self._add_hold(hold_stock)
            cost = round(buy_count * next_open, 2)
            self.free_amount = round(self.free_amount - cost, 2)
            if self.debug:
                print(f"日期 {self.today} 买入 {row['code']} , {next_open} * {buy_count} = {cost} ,剩余资金 {self.free_amount}")

    def sell(self):
        if not self.hold:
            return
        
        today=self.today
        
        # 预加载当日所有持仓股票数据（消除重复调用）
        self._ensure_today_data_loaded()
        
        sells_info=[]
        for hold in self.hold:
            code=hold.code
            if hold.buy_day==today:
                continue
            stock_data = self._today_data_cache.get(code)
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
        
        if len(sells_info) == 0:
            return
        
        # 批量处理卖出
        sell_codes = {s[0] for s in sells_info}
        for code, sell_price, sell_reason in sells_info:
            hold = self._remove_hold(code)
            if hold:
                profit = round((sell_price - hold.buy_price) * hold.buy_count, 2)
                self.free_amount = round(self.free_amount + sell_price * hold.buy_count, 2)
                profit_rate = profit / (hold.buy_price * hold.buy_count) if hold.buy_price * hold.buy_count > 0 else 0
                
                # 记录交易历史
                self.trades_history.append({
                    'date': self.today,
                    'code': code,
                    'buy_date': hold.buy_day,
                    'buy_price': hold.buy_price,
                    'sell_price': sell_price,
                    'quantity': hold.buy_count,
                    'profit': profit,
                    'profit_rate': profit_rate,
                    'reason': sell_reason
                })
                
                if self.debug:
                    print(f"日期 {self.today} 卖出 {code} {hold.buy_day}->{self.today} {hold.buy_price} -> {sell_price} 原因:{sell_reason} 盈亏 {profit}({profit_rate:.2%}), 剩余资金 {self.free_amount}")
                

    def stop_loss(self, hold:HoldStock, stock_data:pd.DataFrame,params:StopLossParams)->tuple:
        """
        触发固定阈值的止损卖出
        """
        #计算止损价
        stop_loss_price=round(hold.buy_price * (1 + params.rate), 2)
        if stock_data.open <= stop_loss_price:
            return True, stock_data.open,f"开盘价{stock_data.open}<{stop_loss_price}({abs(params.rate):.2%}),以开盘价{stock_data.open}卖出"
        elif stock_data.low <= stop_loss_price:
            return True, stop_loss_price,f"盘中最低价{stock_data.low}<{stop_loss_price}({abs(params.rate):.2%}),以止损价{stop_loss_price}卖出"
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
        # 计算每日总资产（使用缓存，避免重复获取数据）
        hold_amount = 0
        
        for hold in self.hold:
            stock_data = self._today_data_cache.get(hold.code)
            if stock_data is None or (hasattr(stock_data, 'empty') and stock_data.empty):
                if self.debug:
                    print(f"日期 {self.today} 持有 {hold.code} 日期:{self.today} 无数据")
            else:
                if self.debug:
                    print(f"日期 {self.today} 持有 {hold.code} 日期:{hold.buy_day}->{self.today} 价格{hold.buy_price}->{stock_data.close} 累计: {(stock_data.close-hold.buy_price)*hold.buy_count:.2f} ({(stock_data.close - hold.buy_price)/hold.buy_price:.2%})")
                hold_amount += stock_data.close * hold.buy_count
        
        total_value = hold_amount + self.free_amount
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
                'init_amount': self.init_amount,
                'final_amount': self.init_amount,
                'total_return': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'profit_loss_ratio': 0,
                'trade_count': 0,
                'max_drawdown': 0,
                'avg_utilization': 0
            }
        
        # 预加载当日持仓数据到缓存
        self._ensure_today_data_loaded()
        
        # 计算最终投资组合价值（当前持有股票价值 + 现金）
        final_holdings_value = 0
        for hold in self.hold:
            stock_data = self._today_data_cache.get(hold.code)
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
        self.sell_chain_list = self.init_sell_strategy_chain()
        
    def _init_pick_filter(self):
        buy_up_day_min = self.buy_param_arr[0]
        buy_day3_min = self.buy_param_arr[1]
        buy_day3_max = self.buy_param_arr[2]
        buy_day5_min = self.buy_param_arr[3]
        buy_day5_max = self.buy_param_arr[4]
        
        def filter_func(df: pd.DataFrame) -> np.ndarray:
            col_consecutive = df["consecutive_up_days"].values
            col_change3d = df["change_3d"].values
            col_change5d = df["change_5d"].values
            return (
                (col_consecutive >= buy_up_day_min)
                & (col_change3d >= buy_day3_min)
                & (col_change3d <= buy_day3_max)
                & (col_change5d >= buy_day5_min)
                & (col_change5d <= buy_day5_max)
            )
        self._pick_filter = filter_func
    
    def _init_pick_sorter(self):
        max_hold = self.max_hold_count
        
        def sorter_func(df: pd.DataFrame) -> pd.DataFrame:
            n = min(max_hold, len(df))
            if n <= 0:
                return pd.DataFrame()
            
            vol_rank_values = df["vol_rank"].values
            top_n_indices = np.argpartition(vol_rank_values, n-1)[:n]
            sorted_indices = top_n_indices[np.argsort(vol_rank_values[top_n_indices])]
            return df.iloc[sorted_indices].reset_index(drop=True)
        self._pick_sorter = sorter_func
    
    def init_sell_strategy_chain(self):
        sl, sp, cd, cr = self.sell_param_arr
        strategies = []
        if sl is not None:
            strategies.append(SellStrategy("静态止损", StopLossParams(rate=sl/100.0)))
        if sp is not None:
            strategies.append(SellStrategy("静态止盈", StopProfitParams(rate=sp/100.0)))
        if cd is not None and cr is not None:
            strategies.append(SellStrategy("累计涨幅卖出", CumulativeSellParams(days=cd, min_return=cr/100.0)))
        return strategies
    
        
        

class Chain:
    def __init__(self, param=None):
        self.strategies = param["strategy"]  # 策略列表
        self.date_arr = param["date_arr"]  # 回测时间周期列表
        self.chain_debug = param.get("chain_debug", False)  # 是否打印报告
        self.cache = LocalCache()  # 本地缓存
        self.param = param  # 原始参数
        self.stock_data = sd()  # 股票数据源
        self.calendar = sc()  # 交易日历
        self.result_file = param.get("result_file", None)  # 结果文件

    def execute(self) -> list:
        cached_a_df = self.cache.get_csv(f"a_{self.result_file}")
        if cached_a_df is None:
            cached_a_df = pd.DataFrame(columns=RESULT_COLS_A)
        
        cached_b_df = self.cache.get_csv(f"b_{self.result_file}")
        if cached_b_df is None:
            cached_b_df = pd.DataFrame(columns=RESULT_COLS_B)
        
        executed_keys = set(cached_a_df['配置'].tolist()) if '配置' in cached_a_df.columns else set()
        b_keys = set(cached_b_df['配置'].tolist()) if '配置' in cached_b_df.columns else set()
        executed_keys.update(b_keys)
        
        print("已缓存执行策略数:", len(executed_keys))
        
        temp_a_df = pd.DataFrame(columns=RESULT_COLS_A)
        temp_b_df = pd.DataFrame(columns=RESULT_COLS_B)
        
        total_strategies = len(self.strategies)
        print(f"总策略数: {total_strategies}, 已缓存: {len(executed_keys)}")
        
        for idx, strategy_params in tqdm(enumerate(self.strategies), desc="执行策略", total=total_strategies):
            strategy = UpStrategy(**strategy_params)

            param_join_str = "|".join(",".join(map(str, arr)) for arr in [
                [strategy.base_param_arr[1]], strategy.buy_param_arr, strategy.sell_param_arr
            ])

            if param_join_str in executed_keys and not self.chain_debug:
                continue

            results = []
            all_win = True
            for s, e in self.date_arr:
                result = self.execute_one_strategy(strategy, s, e)
                if result.总收益率 <= 0 and not self.chain_debug:
                    all_win = False
                    break
                results.append(result)
            
            # if not results:
            #     continue
            
            if all_win:
                win_count = len(results)
                total_count = len(results)
                win_rate = win_count / total_count
                new_row = {
                    "周期胜率": f"{int(win_rate * 100)}%({win_count}/{total_count})",
                    "平均胜率": f"{int(np.mean([x.胜率 for x in results]) * 100)}%",
                    "平均收益率": f"{np.mean([x.总收益率 for x in results]) * 100:.2f}%",
                    "平均最大回撤": f"{np.mean([x.最大回撤 for x in results]) * 100:.2f}%",
                    "平均交易次数": round(np.mean([x.交易次数 for x in results]), 1),
                    "平均资金使用率": f"{np.mean([x.平均资金使用率 for x in results]) * 100:.2f}%",
                    "配置": param_join_str
                }
                temp_a_df.loc[len(temp_a_df)] = new_row
            else:
                temp_b_df.loc[len(temp_b_df)] = {"配置": param_join_str}
            if self.chain_debug:
                print(new_row)
            executed_keys.add(param_join_str)

            # 跑完了也写一下
            if idx % 10000 == 0 or idx == total_strategies - 1:
                if cached_a_df.empty:
                    cached_a_df = temp_a_df.copy()
                else:
                    for col in ['平均交易次数', '平均资金使用率']:
                        if col not in cached_a_df.columns:
                            cached_a_df[col] = np.nan
                    cached_a_df = pd.concat([cached_a_df, temp_a_df], ignore_index=True)
                temp_a_df = pd.DataFrame(columns=RESULT_COLS_A)

                # cached_a_df 按平均胜率、平均收益率 排序
                cached_a_df = cached_a_df.sort_values(by=['平均胜率', '平均收益率'], ascending=[False, False])
                self.cache.set_csv(f"a_{self.result_file}", cached_a_df)
                
                if cached_b_df.empty:
                    cached_b_df = temp_b_df.copy()
                else:
                    cached_b_df = pd.concat([cached_b_df, temp_b_df], ignore_index=True)
                temp_b_df = pd.DataFrame(columns=RESULT_COLS_B)
                self.cache.set_csv(f"b_{self.result_file}", cached_b_df)


    def execute_one_strategy(self, strategy, start_date, end_date) -> BacktestResult:
        scalendar = self.calendar
        current_idx = scalendar.start(start_date)
        end_idx = scalendar.start(end_date)
        if current_idx == -1 or end_idx == -1:
            return BacktestResult(
                起始日期=start_date,
                结束日期=end_date,
                初始资金=0,
                最终资金=0,
                总收益=0,
                总收益率=0,
                胜率=0,
                盈亏比=0,
                交易次数=0,
                最大回撤=0,
                平均资金使用率=0
            )
        strategy.bind(self.stock_data, self.calendar)
        strategy.reset()

        while current_idx != -1 and current_idx <= end_idx:
            current_date = scalendar.get_date(current_idx)
            strategy.update_today(current_date)
            strategy._new_day()
            strategy.buy()
            strategy.sell()
            strategy.pick()
            strategy.print_daily()
            current_idx = scalendar.next(current_idx)
        
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
        
        if self.chain_debug:
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



def bt_all(day_array,result_file):
    # 定义策略参数字典列表（不创建对象，省内存）
    strategy_params_list=[]
    for a in range(2,6,1): # 持仓数量
        for buy1 in range(2,4,1): # 连涨天数
            for buy2 in range(3,10,2): # 3日涨幅最低
                for buy3 in range(5,15,5): # 3日涨幅最高
                    for buy4 in range(5,15,3): # 5日涨幅最低
                        for buy5 in range(15,45,5): # 5日涨幅最高
                            for sell1 in range(-15,-4,1): # 止损率（负数，如-5表示-5%）
                                for sell2 in range(15,100,5): # 止盈率
                                    for sell3 in range(3,6,1): # 止盈持有时间
                                        for sell4 in range(5,40,5): # 止盈持有收益率
                                            strategy_params_list.append({
                                                "base_param_arr": [100000, a],
                                                "buy_param_arr": [buy1, buy2, buy3, buy4, buy5],
                                                "sell_param_arr": [sell1, sell2, sell3, sell4],
                                                "debug": 0
                                            })
    print(f"策略参数数量: {len(strategy_params_list)}")
    
    # 随机打散
    random.shuffle(strategy_params_list)
    random.shuffle(strategy_params_list)
    random.shuffle(strategy_params_list)

    # 测试 先用1000个策略
    # strategy_params_list = strategy_params_list[:1]
    

    
    param={
        "strategy":strategy_params_list
        ,"date_arr":day_array
        ,"chain_debug":0
        ,"result_file":result_file
    }
        # ,"date_arr":[
        #     ["20240701","20240801"],["20240801","20240901"],["20240901","20241001"],["20241001","20241101"],["20241101","20241201"],["20241201","20250101"]
        #     ,["20250101","20250201"],["20250201","20250301"],["20250301","20250401"],["20250401","20250501"],["20250501","20250601"],["20250601","20250701"]
        #     ,["20250701","20250801"],["20250801","20250901"],["20250901","20251001"],["20251001","20251101"],["20251101","20251201"],["20251201","20260101"]
        #     ,["20260101","20260201"]
        #              ]
        # ,"date_arr":[["20250101","20260101"]]
    chain = Chain(param=param)
    chain.execute()

def bt_one(strategy_params,day_array,result_file):
    base_arr, buy_arr, sell_arr = strategy_params.split("|")
    a= int(base_arr.split(",")[0])
    buy1, buy2, buy3, buy4, buy5 = map(int, buy_arr.split(","))
    sell1, sell2, sell3, sell4 = map(int, sell_arr.split(","))
    
    strategy_params_list=[]
    strategy_params_list.append({
        "base_param_arr": [100000, a],
        "buy_param_arr": [buy1, buy2, buy3, buy4, buy5],
        "sell_param_arr": [sell1, sell2, sell3, sell4],
        "debug": 1
    })
    param={
        "strategy":strategy_params_list
        ,"date_arr":day_array
        ,"chain_debug":1
        ,"result_file":result_file
    }
    chain = Chain(param=param)
    chain.execute()


if __name__ == "__main__":
    start_time=datetime.now().timestamp()*1000
    
    start_date="20240701"
    end_date="20260206"
    part=6
    day_array=sc().build_day_array(start_date,end_date,part)

    result_file=f"连涨{start_date}-{end_date}-{part}-vol_rank正排"

    bt_all(day_array,result_file)

    # 100%(6/6),48%,17.70%,23.14%,124.0,49.20%,
    # bt_one("4|2,5,10,8,15|-11,60,5,25",day_array,result_file)


    end_time=datetime.now().timestamp()*1000
    print(f"回测完成 耗时{(end_time-start_time):.2f}ms")
