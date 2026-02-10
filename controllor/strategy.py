import sys
from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import matplotlib.pyplot as plt
import polars as pl

from dto import *

# 全局常量
EMPTY_STRING = ""

pl.Config.set_tbl_cols(-1)          # -1 表示显示所有列（默认是有限数量）

class Strategy:
    # 策略映射，避免重复的条件判断
    STRATEGY_MAP = {
        "静态止损": "stop_loss",
        "静态止盈": "stop_profit",
        "累计涨幅卖出": "cumulative_return_sell",
        "移动止盈": "trailing_stop_profit"
    }
    
    def __init__(self, base_param_arr: list, sell_param_arr: list, buy_param_arr: list, debug: bool):
        """初始化策略
        
        Args:
            base_param_arr: 基础参数数组，包含初始资金和最大持仓数
            sell_param_arr: 卖出参数数组
            buy_param_arr: 买入参数数组
            debug: 是否开启调试模式
        """
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
    
    def _init_pick_filter(self) -> None:
        """初始化筛选函数，子类重写"""
        def default_filter(df: pl.DataFrame) -> pl.Series:
            """默认筛选：返回所有股票"""
            return pl.Series([True] * len(df))
        self._pick_filter = default_filter

    def _init_pick_sorter(self) -> None:
        """初始化排序函数，子类重写"""
        def default_sorter(df: pl.DataFrame) -> pl.DataFrame:
            """默认排序：返回原数据"""
            return df
        self._pick_sorter = default_sorter
    
    def bind(self, data: sd, calendar: sc) -> None:
        """绑定数据和日历对象"""
        self.data = data
        self.calendar = calendar

    def reset(self) -> None:
        """重置策略状态"""
        self.free_amount = self.init_amount
        self.hold: list[HoldStock] = []
        self.hold_codes: set[str] = set()
        self.picked_data: pl.DataFrame | None = None
        self.trades_history: list[dict] = []
        self.daily_values: list[dict] = []
        self.today: str | None = None
        self._today_data_cache: dict[str, pl.DataFrame] = {}
    
    def _ensure_today_data_loaded(self) -> None:
        """确保当日持仓股票数据已加载"""
        if not self.hold:
            return
        today = self.today
        for hold in self.hold:
            if hold.code not in self._today_data_cache:
                self._today_data_cache[hold.code] = self.data.get_data_by_date_code(today, hold.code)
    
    def _add_hold(self, hold_stock: HoldStock) -> None:
        """添加持仓股票"""
        self.hold.append(hold_stock)
        self.hold_codes.add(hold_stock.code)
    
    def _remove_hold(self, code: str) -> HoldStock | None:
        """移除持仓股票"""
        for i, hold in enumerate(self.hold):
            if hold.code == code:
                self.hold.pop(i)
                self.hold_codes.discard(code)
                return hold
        return None
    
    def pick(self) -> pl.DataFrame: 
        """选择符合条件的股票"""
        today = self.today
        
        # 获取当日股票数据（返回类型为 pl.DataFrame）
        today_stock_df = self.data.get_data_by_date(today)
        
        # 检查数据是否为空
        if today_stock_df.is_empty():
            self.picked_data = pl.DataFrame()
            if self.debug:
                print(f"日期 {today} 无符合条件股票")
            return pl.DataFrame()
        
        # 应用筛选函数生成掩码
        mask = self._pick_filter(today_stock_df)
        
        # 根据掩码筛选股票
        filtered_stocks = today_stock_df.filter(mask)
        
        # 检查筛选结果是否为空
        if filtered_stocks.is_empty():
            self.picked_data = pl.DataFrame()
            if self.debug:
                print(f"日期 {today} 无符合条件股票")
            return pl.DataFrame()
        
        # 应用排序函数对筛选结果进行排序
        result = self._pick_sorter(filtered_stocks)
        
        # 记录筛选结果
        self.picked_data = filtered_stocks
        
        # 调试信息
        if self.debug:
            print(f"日期 {today} 选出股票 {len(filtered_stocks)} 只")
            print(f"前5只 {result.head(5)}")
        
        return result
    

    def buy(self) -> None:
        """执行买入操作"""
        # 没选出来票,不买
        picked_data = self.picked_data
        if picked_data is None or picked_data.is_empty():
            return
        
        # 达到最大持仓了,不买
        hold = self.hold
        hold_codes = self.hold_codes
        max_hold_count = self.max_hold_count
        current_hold_count = len(hold)
        if current_hold_count >= max_hold_count:
            return
        
        # 计算每个股票买入金额（分）
        remaining_hold_count = max_hold_count - current_hold_count
        free_amount = self.free_amount
        buy_amount_per_stock_cents = free_amount // remaining_hold_count
        
        # 计算买入的票的数量 按今天的开盘价买
        today = self.today
        for row in picked_data.iter_rows(named=True):
            # 持仓数量够了,跳过买入
            if len(hold) >= max_hold_count:
                break
            
            # 已持有的股票不能重复购买（O(1)查找）
            code = row["code"]
            if code in hold_codes:
                continue
            
            next_open = row["next_open"]
            # 只能买100的整数
            buy_count = buy_amount_per_stock_cents // next_open // 100 * 100
            if buy_count <= 0:
                continue
            
            hold_stock = HoldStock(code, next_open, buy_count, today)
            self._add_hold(hold_stock)
            cost_cents = buy_count * next_open
            self.free_amount -= cost_cents
            
            if self.debug:
                free_amount = self.free_amount / 100
                cost = cost_cents / 100
                print(f"日期 {today} 买入 {code} , {next_open/100:.2f} * {buy_count} = {cost:.2f} ,剩余资金 {free_amount:.2f}")

    def sell(self) -> None:
        is_debug=self.debug
        """执行卖出操作"""
        hold = self.hold
        if not hold:
            return
        
        today = self.today
        # 预加载当日所有持仓股票数据（消除重复调用）
        self._ensure_today_data_loaded()
        data_cache = self._today_data_cache
        sell_chain_list = getattr(self, 'sell_chain_list', [])
        
        sells_info: list[tuple[str, int, str]] = []
        for hold_stock in hold:
            code = hold_stock.code
            buy_day = hold_stock.buy_day
            if buy_day == today:
                continue
            
            stock_data = data_cache.get(code)
            # 检查数据是否为空或无效
            is_empty = False
            if stock_data is None:
                is_empty = True
            elif hasattr(stock_data, 'is_empty'):
                is_empty = stock_data.is_empty()
            # StockDataTuple 不会为空，因为它是一个包含数据的命名元组
            
            if is_empty:
                if is_debug:
                    print(f"日期 {today} 没有找到股票 {code} 的数据,跳过卖出")
                continue
            
            # 策略决定判断是否要卖掉这个票
            need_sell, sell_price, reason = False, 0, ""
            for sell_strategy in sell_chain_list:
                sell_name = sell_strategy.name
                params = sell_strategy.params
                # 使用类属性策略映射查找方法名
                method_name = self.STRATEGY_MAP.get(sell_name)
                if not method_name:
                    raise ValueError(f"未知的卖出策略: {sell_name}")
                # 获取对应的方法对象
                strategy_func = getattr(self, method_name)
                # 调用对应的策略函数
                need_sell, sell_price, reason = strategy_func(hold_stock, stock_data, params)
                # 设置颜色
                if is_debug:
                    reason = f"\033[91m{reason}\033[0m" if sell_price > hold_stock.buy_price else f"\033[92m{reason}\033[0m"
                # 如果某个策略触发卖出，则不再检查其他策略
                if need_sell:
                    break
            
            if need_sell:
                sells_info.append((code, sell_price, reason))
        
        if not sells_info:
            return
        
        # 批量处理卖出
        trades_history = self.trades_history
        for code, sell_price, sell_reason in sells_info:
            hold_stock = self._remove_hold(code)
            if hold_stock:
                buy_price = hold_stock.buy_price
                buy_count = hold_stock.buy_count
                # 计算盈亏
                profit_cents = (sell_price - buy_price) * buy_count
                self.free_amount += sell_price * buy_count
                # 计算盈亏率
                cost_cents = buy_price * buy_count
                profit_rate = profit_cents / cost_cents if cost_cents > 0 else 0
                
                # 记录交易历史
                trades_history.append({
                    'date': today,
                    'code': code,
                    'buy_date': hold_stock.buy_day,
                    'buy_price': buy_price,
                    'sell_price': sell_price,
                    'quantity': buy_count,
                    'profit': profit_cents,
                    'profit_rate': profit_rate,
                    'reason': sell_reason
                })
                
                if is_debug:
                    free_amount = self.free_amount / 100
                    profit = profit_cents // 100
                    print(f"日期 {today} 卖出 {code} {hold_stock.buy_day}->{today} {buy_price/100:.2f} -> {sell_price/100:.2f} 原因:{sell_reason} 盈亏 {profit}({profit_rate:.2%}), 剩余资金 {free_amount:.2f}")
                

    def stop_loss(self, hold: HoldStock, stock_data: pl.DataFrame, params: StopLossParams) -> tuple[bool, int, str]:
        """
        触发固定阈值的止损卖出
        """
        #计算止损价
        stop_loss_price = int(hold.buy_price * (1 + params.rate))
        # 缓存常用值
        open_price = stock_data['open'][0]
        low_price = stock_data['low'][0]
        is_debug=self.debug
        if open_price <= stop_loss_price:
            if is_debug:
                return True, open_price, f"开盘价{open_price/100:.2f}<{stop_loss_price/100:.2f}({abs(params.rate):.2%}),以开盘价{open_price/100:.2f}卖出"
            return True, open_price, EMPTY_STRING
        elif low_price <= stop_loss_price:
            if is_debug:
                return True, stop_loss_price, f"盘中最低价{low_price/100:.2f}<{stop_loss_price/100:.2f}({abs(params.rate):.2%}),以止损价{stop_loss_price/100:.2f}卖出"
            return True, stop_loss_price, EMPTY_STRING
        return False, 0, EMPTY_STRING
    
    def stop_profit(self, hold: HoldStock, stock_data: pl.DataFrame, params: StopProfitParams) -> tuple[bool, int, str]:
        """
        触发固定阈值的止盈卖出
        """
        #计算止盈价
        stop_profit_price = int(hold.buy_price * (1 + params.rate))
        # 缓存常用值
        open_price = stock_data['open'][0]
        high_price = stock_data['high'][0]
        is_debug=self.debug
        if open_price >= stop_profit_price:
            if is_debug:
                return True, open_price, f"开盘价{open_price/100:.2f}>止盈价{stop_profit_price/100:.2f}({params.rate:.2%}),以开盘价{open_price/100:.2f}卖出"
            return True, open_price, EMPTY_STRING
        elif high_price >= stop_profit_price:
            if is_debug:
                return True, stop_profit_price, f"盘中最高价{high_price/100:.2f}>止盈价{stop_profit_price/100:.2f}({params.rate:.2%}),以止盈价{stop_profit_price/100:.2f}卖出"
            return True, stop_profit_price, EMPTY_STRING
        return False, 0, EMPTY_STRING

    def cumulative_return_sell(self, hold: HoldStock, stock_data: pl.DataFrame, params: CumulativeSellParams) -> tuple[bool, int, str]:
        """
        x天累计涨幅没到y，以持仓最后一天的开盘价卖掉
        params: CumulativeSellParams(days=x, min_return=y)
        """
        # 直接使用StockCalendar的gap函数计算持仓天数
        hold_days = self.calendar.gap(hold.buy_day, self.today) if self.calendar else 0
        # 如果持仓天数小于x天，不触发卖出
        if hold_days < params.days:
            return False, 0, EMPTY_STRING
        # 缓存常用值
        close_price = stock_data['close'][0]
        open_price = stock_data['open'][0]
        is_debug=self.debug
        # 计算累计涨幅
        cumulative_return = (close_price - hold.buy_price) / hold.buy_price
        # 如果累计涨幅没达到最小要求，以开盘价卖出
        if cumulative_return < params.min_return:
            if is_debug:
                return True, open_price, f"持仓{params.days}天 累计涨幅{cumulative_return:.2%}<{params.min_return:.2%}，以开盘价{open_price/100:.2f}卖出"
            return True, open_price, EMPTY_STRING
        
        return False, 0, EMPTY_STRING

    def trailing_stop_profit(self, hold: HoldStock, stock_data: pl.DataFrame | dict, params: TrailingStopProfitParams) -> tuple[bool, int, str]:
        """
        移动止盈策略：从最高持仓价格回撤指定百分比后卖出
        """
        # 计算止盈价（最高价格 * (1 - 回撤率)）
        trailing_stop_price = int(hold.highest_price * (1 - params.rate))
        
        # 缓存常用值
        open_price = stock_data['open'][0]
        low_price = stock_data['low'][0]
        is_debug=self.debug
        if open_price <= trailing_stop_price:
            if is_debug:
                return True, open_price, f"开盘价{open_price/100:.2f}<=移动止盈价{trailing_stop_price/100:.2f}(最高{hold.highest_price/100:.2f}回撤{params.rate:.2%}),以开盘价{open_price/100:.2f}卖出"
            return True, open_price, EMPTY_STRING
        elif low_price <= trailing_stop_price:
            if is_debug:
                return True, trailing_stop_price, f"盘中最低价{low_price/100:.2f}<=移动止盈价{trailing_stop_price/100:.2f}(最高{hold.highest_price/100:.2f}回撤{params.rate:.2%}),以移动止盈价{trailing_stop_price/100:.2f}卖出"
            return True, trailing_stop_price, EMPTY_STRING
        
        return False, 0, EMPTY_STRING

    def settle_amount(self) -> None:
        """计算每日总资产并记录"""
        # 计算每日总资产（使用缓存，避免重复获取数据）
        hold_amount = 0
        today = self.today
        hold = self.hold
        data_cache = self._today_data_cache
        is_debug = self.debug
        
        for hold_stock in hold:
            code = hold_stock.code
            stock_data = data_cache.get(code)
            
            # 检查数据是否为空或无效
            is_empty = False
            if stock_data is None:
                is_empty = True
            elif hasattr(stock_data, 'is_empty'):
                is_empty = stock_data.is_empty()
            # StockDataTuple 不会为空，因为它是一个包含数据的命名元组
            
            if is_empty:
                if is_debug:
                    print(f"日期 {today} 持有 {code} 日期:{today} 无数据")
            else:
                close_price = stock_data['close'][0]
                high_price = stock_data['high'][0]
                buy_price = hold_stock.buy_price
                buy_count = hold_stock.buy_count
                
                if high_price > hold_stock.highest_price:
                    # 更新最高持仓价格
                    hold_stock.update_highest_price(high_price)
                
                if is_debug:
                    profit_cents = (close_price - buy_price) * buy_count
                    profit_rate = (close_price - buy_price) / buy_price
                    highest_price = hold_stock.highest_price
                    print(f"日期 {today} 持有 {code} 日期:{hold_stock.buy_day}->{today} 价格{buy_price/100:.2f}->{close_price/100:.2f} 最高:{highest_price/100:.2f} 累计: {profit_cents/100:.2f} ({profit_rate:.2%})")
                
                hold_amount += close_price * buy_count
        
        free_amount = self.free_amount
        total_value = hold_amount + free_amount
        
        # 记录每日资产
        self.daily_values.append({'date': today, 'value': total_value, 'free_amount': free_amount})
        
        if is_debug:
            free_amount_元 = free_amount / 100
            hold_amount_元 = hold_amount / 100
            total_value_元 = total_value / 100
            print(f"日期 {today} 持有股票总市值 {hold_amount_元:.2f}, 可用资金 {free_amount_元:.2f}, 总资产 {total_value_元:.2f}")
            print("\n")
    
    def update_today(self, today: str) -> None:
        """更新当前日期"""
        self.today = today
        self._today_data_cache = {}
    
    def calculate_performance(self, start_date: str, end_date: str) -> BacktestResult:
        """计算并返回策略性能指标"""
        # 缓存实例变量
        init_amount = self.init_amount
        trades_history = self.trades_history
        hold = self.hold
        daily_values = self.daily_values
        
        if not trades_history and not hold:
            return BacktestResult(
                起始日期=start_date,
                结束日期=end_date,
                初始资金=init_amount,
                最终资金=init_amount,
                总收益=0,
                总收益率=0,
                胜率=0,
                交易次数=0,
                最大资金=init_amount,
                最小资金=init_amount,
                夏普比率=0,
                平均资金使用率=0
            )
        
        # 预加载当日持仓数据到缓存
        self._ensure_today_data_loaded()
        data_cache = self._today_data_cache
        
        # 计算最终投资组合价值（当前持有股票价值 + 现金）
        final_holdings_value = 0
        for hold_stock in hold:
            code = hold_stock.code
            stock_data = data_cache.get(code)
            # 检查数据是否为空或无效
            is_empty = False
            if stock_data is None:
                is_empty = True
            elif hasattr(stock_data, 'is_empty'):
                is_empty = stock_data.is_empty()
            # StockDataTuple 不会为空，因为它是一个包含数据的命名元组
            
            if is_empty:
                # 如果今天没有数据，使用买入价格作为估值
                final_holdings_value += hold_stock.buy_price * hold_stock.buy_count
            else:
                final_holdings_value += stock_data['close'][0] * hold_stock.buy_count
        
        # 最终总价值（分）
        free_amount = self.free_amount
        final_total_value = final_holdings_value + free_amount
        
        # 总收益 = 最终总价值 - 初始资本
        total_return = final_total_value - init_amount
        total_return_pct = total_return / init_amount if init_amount > 0 else 0

        # 计算胜率
        trade_count = len(trades_history)
        win_rate = 0
        if trade_count > 0:
            winning_trades = sum(1 for t in trades_history if t['profit'] > 0)
            win_rate = winning_trades / trade_count

        # 计算资金使用率（每日持仓市值 / 总资产）
        avg_utilization = 0
        if daily_values:
            utilization_sum = 0
            count = 0
            for dv in daily_values:
                value = dv['value']
                if value > 0:
                    utilization = (value - dv['free_amount']) / value
                    utilization_sum += utilization
                    count += 1
            if count > 0:
                avg_utilization = utilization_sum / count

        # 计算最大资金和最小资金
        if daily_values:
            values = [dv['value'] for dv in daily_values]
            max_value = max(values)
            min_value = min(values)
        else:
            max_value = init_amount
            min_value = init_amount
        
        # 计算夏普比率
        sharpe_ratio = 0
        if len(daily_values) > 1:
            # 计算每日收益率
            daily_returns = []
            for i in range(1, len(daily_values)):
                prev_value = daily_values[i-1]['value']
                curr_value = daily_values[i]['value']
                if prev_value > 0:
                    return_rate = (curr_value - prev_value) / prev_value
                    daily_returns.append(return_rate)
            
            if daily_returns:
                # 假设无风险利率为0
                risk_free_rate = 0
                # 计算平均每日收益率
                n_returns = len(daily_returns)
                avg_daily_return = sum(daily_returns) / n_returns
                # 计算标准差
                variance = sum((r - avg_daily_return) ** 2 for r in daily_returns) / n_returns
                std_daily_return = variance ** 0.5
                
                if std_daily_return > 0:
                    # 年化夏普比率（假设一年252个交易日）
                    import math
                    sharpe_ratio = (avg_daily_return - risk_free_rate) / std_daily_return * math.sqrt(252)
        
        return BacktestResult(
            起始日期=start_date,
            结束日期=end_date,
            初始资金=init_amount,
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
    
