import sys
from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from dto import *

class Strategy:
    # 策略映射，避免重复的条件判断
    STRATEGY_MAP = {
        "静态止损": "stop_loss",
        "静态止盈": "stop_profit",
        "累计涨幅卖出": "cumulative_return_sell"
    }
    
    def __init__(self, base_param_arr, sell_param_arr, buy_param_arr, debug):
        self.base_param_arr = base_param_arr
        self.sell_param_arr = sell_param_arr
        self.buy_param_arr = buy_param_arr
        self.init_amount, self.max_hold_count = base_param_arr[0], base_param_arr[1]
        self.data = None
        self.calendar = None
        self.debug = debug
        self._init_pick_filter()
        self._init_pick_sorter()
        self.reset()
        
        if self.max_hold_count is None or self.init_amount is None:
            print(f"策略未配置最大持仓数max_hold_count或初始资金init_amount,结束任务")
            sys.exit(1)
    
    def _init_pick_filter(self):
        """初始化筛选函数，子类重写"""
        def default_filter(df: pd.DataFrame) -> np.ndarray:
            """默认筛选：返回所有股票"""
            return np.ones(len(df), dtype=bool)
        self._pick_filter = default_filter

    def _init_pick_sorter(self):
        """初始化排序函数，子类重写"""
        def default_sorter(df: pd.DataFrame) -> pd.DataFrame:
            """默认排序：返回原数据"""
            return df
        self._pick_sorter = default_sorter
    
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
                # 使用类属性策略映射查找方法名
                method_name = self.STRATEGY_MAP.get(sell_name)
                if not method_name:
                    raise ValueError(f"未知的卖出策略: {sell_name}")
                # 获取对应的方法对象
                strategy_func = getattr(self, method_name)
                # 调用对应的策略函数
                need_sell, sell_price, reason = strategy_func(hold, stock_data, params)
                # 设置颜色
                reason=f"\033[91m{reason}\033[0m" if sell_price>hold.buy_price else f"\033[92m{reason}\033[0m"
                # 如果某个策略触发卖出，则不再检查其他策略
                if need_sell:
                    break
            if need_sell:
                sells_info.append((code, sell_price, reason))
        
        if len(sells_info) == 0:
            return
        
        # 批量处理卖出
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

    def settle_amount(self): 
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
    
    def update_today(self, today):
        self.today = today
        self._today_data_cache = {}
    
    def calculate_performance(self, start_date, end_date):
        """计算并返回策略性能指标"""
        
        if not self.trades_history and not self.hold:
            return BacktestResult(
                起始日期=start_date,
                结束日期=end_date,
                初始资金=self.init_amount,
                最终资金=self.init_amount,
                总收益=0,
                总收益率=0,
                胜率=0,
                交易次数=0,
                最大资金=self.init_amount,
                最小资金=self.init_amount,
                夏普比率=0,
                平均资金使用率=0
            )
        
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

        # 交易次数
        trade_count = len(profits)

        # 计算资金使用率（每日持仓市值 / 总资产）
        values = [dv['value'] for dv in self.daily_values]
        hold_values = [dv['value'] - dv['free_amount'] for dv in self.daily_values]
        avg_utilization = np.mean([h/v for v, h in zip(values, hold_values) if v != 0]) if values else 0

        # 计算最大资金和最小资金
        max_value = max(values) if values else self.init_amount
        min_value = min(values) if values else self.init_amount
        
        # 计算夏普比率
        sharpe_ratio = 0
        if len(values) > 1:
            # 计算每日收益率
            daily_returns = [(values[i] - values[i-1]) / values[i-1] for i in range(1, len(values))]
            if daily_returns:
                # 假设无风险利率为0
                risk_free_rate = 0
                avg_daily_return = np.mean(daily_returns)
                std_daily_return = np.std(daily_returns)
                if std_daily_return > 0:
                    # 年化夏普比率（假设一年252个交易日）
                    sharpe_ratio = (avg_daily_return - risk_free_rate) / std_daily_return * np.sqrt(252)
        

        
        return BacktestResult(
            起始日期=start_date,
            结束日期=end_date,
            初始资金=self.init_amount,
            最终资金=final_total_value,
            总收益=total_return,
            总收益率=total_return_pct,
            胜率=win_rate,
            交易次数=trade_count,
            最大资金=max_value,
            最小资金=min_value,
            夏普比率=sharpe_ratio,
            平均资金使用率=avg_utilization
        )
    
