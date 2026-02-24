import sys
from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import polars as pl
import numpy as np
from numba import njit

from dto import *

# 导入日志配置
import logger_config

@njit(cache=True)
def _filter_numba(up_days, change_3d, change_5d, change_pct, limit_up_count_10d, volume_ratio,
                  buy_up_day_min, buy_day3_min, buy_day5_min, change_pct_max, limit_up_count_min, volume_ratio_min):
    """Numba JIT 编译的筛选函数
    limit_up_count_min: 涨停次数要求，-1表示不限制，0表示10天内0涨停，1表示10天内≥1次涨停
    buy_up_day_min: 连涨天数要求，-1表示不限制
    buy_day3_min: 3日涨幅要求，-1表示不限制
    buy_day5_min: 5日涨幅要求，-1表示不限制
    change_pct_max: 当日涨幅上限，-1表示不限制
    volume_ratio_min: 量比要求，-1表示不限制
    """
    n = len(up_days)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        # 连涨天数条件（-1表示不限制）
        up_ok = (buy_up_day_min == -1) or (up_days[i] >= buy_up_day_min)
        # 3日涨幅条件（-1表示不限制）
        day3_ok = (buy_day3_min == -1) or (change_3d[i] > buy_day3_min)
        # 5日涨幅条件（-1表示不限制）
        day5_ok = (buy_day5_min == -1) or (change_5d[i] > buy_day5_min)
        # 当日涨幅上限条件（-1表示不限制）
        pct_ok = (change_pct_max == -1) or (change_pct[i] < change_pct_max)
        # 量比条件（-1表示不限制）
        volume_ok = (volume_ratio_min == -1) or (volume_ratio[i] > volume_ratio_min)

        base_ok = up_ok and day3_ok and day5_ok and pct_ok and volume_ok

        # 涨停次数条件（-1表示不限制，0表示10天内0涨停，1表示10天内≥1次涨停）
        if limit_up_count_min == -1:
            result[i] = base_ok
        elif limit_up_count_min == 0:
            # 0表示要求0次涨停（即排除有涨停的股票）
            result[i] = base_ok and (limit_up_count_10d[i] == 0)
        else:
            # 1表示要求至少1次涨停
            result[i] = base_ok and (limit_up_count_10d[i] >= 1)
    return result


@njit(cache=True)
def _calc_buy_counts_numba(codes, next_opens, price_limit_status, hold_codes_arr, base_amount):
    """Numba JIT 编译的买入股数计算函数 - 固定比例模式
    base_amount: 固定买入金额（分）
    """
    n = len(codes)
    buy_counts = np.zeros(n, dtype=np.int64)

    for i in range(n):
        # 跳过涨停股票
        if price_limit_status[i] == 1:
            continue

        # 检查是否已持仓
        already_hold = False
        for j in range(len(hold_codes_arr)):
            if codes[i] == hold_codes_arr[j]:
                already_hold = True
                break
        if already_hold:
            continue

        # 计算股数（100股整数倍）
        buy_counts[i] = base_amount // next_opens[i] // 100 * 100

    return buy_counts


@njit(cache=True)
def _check_sell_numba(buy_price, highest_price, hold_days,
                      open_price, close_price, high_price, low_price,
                      stop_loss_price, target_return_price, trailing_stop_price,
                      hold_days_limit):
    """Numba JIT 编译的卖出检查函数
    返回: (need_sell, sell_price)
    """
    # 1. 止损检查（开盘价或最低价触及止损价）
    if open_price <= stop_loss_price:
        return True, open_price
    if low_price <= stop_loss_price:
        return True, stop_loss_price

    # 2. 贪婪止盈
    # 阶段1：已达到目标收益，启用移动止盈（不再限制持仓天数）
    if highest_price >= target_return_price:
        if open_price <= trailing_stop_price:
            return True, open_price
        if low_price <= trailing_stop_price:
            return True, trailing_stop_price
    
    # 阶段2：持仓到期且未达到目标，强制卖出
    if hold_days >= hold_days_limit:
        return True, open_price

    return False, 0

# 全局常量
EMPTY_STRING = ""

# 全局选票缓存结构（两级缓存）：
# _global_filter_cache: {filter_key -> {date -> filtered_data}} 筛选缓存（只依赖买入参数）
# _global_sort_cache: {sort_key -> {date -> sorted_data}} 排序缓存（依赖选股参数）
_global_filter_cache: dict[tuple, dict[int, dict]] = {}
_global_sort_cache: dict[tuple, dict[int, dict]] = {}
_global_current_filter_key: tuple = ()
_global_current_sort_key: tuple = ()

pl.Config.set_tbl_cols(-1)          # -1 表示显示所有列（默认是有限数量）

class Strategy:
    
    def __init__(self, base_param_arr: list, sell_param_arr: list, buy_param_arr: list, pick_param_arr: list, debug: bool):
        """初始化策略

        Args:
            base_param_arr: 基础参数数组，包含初始资金和最大持仓数
            sell_param_arr: 卖出参数数组
            buy_param_arr: 买入参数数组
            pick_param_arr: 选股排序参数数组 [排序字段, 排序方式]
            debug: 是否开启调试模式
        """
        self.base_param_arr = base_param_arr
        self.sell_param_arr = sell_param_arr
        self.buy_param_arr = buy_param_arr
        self.pick_param_arr = pick_param_arr if pick_param_arr else [0, 1]
        self.init_amount, self.max_hold_count = base_param_arr[0], base_param_arr[1]
        # 预计算卖出参数（避免每次重复计算）
        stop_loss_rate, hold_days_limit, target_return, trailing_rate = sell_param_arr
        self._stop_loss_rate = stop_loss_rate / 100.0
        self._hold_days_limit = hold_days_limit
        self._target_return = target_return / 100.0
        self._trailing_rate = trailing_rate / 100.0
        # 交易费用参数
        self._commission_rate = 0.0001   # 手续费率 0.01% (万1)
        self._slippage_rate = 0.001      # 滑点率 0.1%
        self.data = None
        self.calendar = None
        self.debug = debug
        # 预计算两级缓存参数标识
        self._filter_key = self._compute_filter_key()
        self._sort_key = self._compute_sort_key()
        self._init_pick_filter()
        self._init_pick_sorter()
        self.reset()
        
        if self.max_hold_count is None or self.init_amount is None:
            print(f"策略未配置最大持仓数max_hold_count或初始资金init_amount,结束任务")
            sys.exit(1)
    
    def _init_pick_filter(self) -> None:
        """初始化连涨策略筛选函数"""
        buy_up_day_min = self.buy_param_arr[0]
        buy_day3_min = self.buy_param_arr[1]
        buy_day5_min = self.buy_param_arr[2]
        change_pct_max = self.buy_param_arr[3] if len(self.buy_param_arr) > 3 else 5
        limit_up_count_min = self.buy_param_arr[4] if len(self.buy_param_arr) > 4 else -1
        volume_ratio_min = self.buy_param_arr[5] if len(self.buy_param_arr) > 5 else -1

        def filter_func(numpy_data: dict):
            mask = _filter_numba(
                numpy_data['consecutive_up_days'],
                numpy_data['change_3d'],
                numpy_data['change_5d'],
                numpy_data['change_pct'],
                numpy_data['limit_up_count_10d'],
                numpy_data['volume_ratio'],
                buy_up_day_min, buy_day3_min, buy_day5_min, change_pct_max, limit_up_count_min, volume_ratio_min
            )
            return mask
        self._pick_filter = filter_func

    def _init_pick_sorter(self) -> None:
        """初始化排序函数
        数据已按成交量升序排列，直接使用无需排序
        """
        def sorter_func(filtered_data: dict) -> np.ndarray:
            """返回排序后的索引数组（输入已是筛选后的数据）"""
            n = len(filtered_data['code'])
            # 数据已按成交量升序排列，直接使用
            return np.arange(n, dtype=np.int64)
        self._pick_sorter = sorter_func
    
    def bind(self, data: sd, calendar: sc) -> None:
        """绑定数据和日历对象"""
        self.data = data
        self.calendar = calendar

    def reset(self) -> None:
        """重置策略状态"""
        self.free_amount = self.init_amount
        self.hold: list[HoldStock] = []
        self.hold_codes: set[int] = set()
        self.picked_numpy_data: dict | None = None
        self.trades_history: list[dict] = []
        self.daily_values: list[dict] = []
        self.today: int | None = None
    
    def _add_hold(self, hold_stock: HoldStock) -> None:
        """添加持仓股票"""
        self.hold.append(hold_stock)
        self.hold_codes.add(hold_stock.code)
    
    def _remove_hold(self, code: int) -> HoldStock | None:
        """移除持仓股票"""
        for i, hold in enumerate(self.hold):
            if hold.code == code:
                self.hold.pop(i)
                self.hold_codes.discard(code)
                return hold
        return None
    
    def _compute_filter_key(self) -> tuple:
        """计算筛选参数key（只包含买入参数）"""
        return tuple(self.buy_param_arr)

    def _compute_sort_key(self) -> tuple:
        """计算排序参数key（包含买入+选股参数）"""
        return tuple(self.buy_param_arr + self.pick_param_arr)

    def pick(self) -> None:
        """选择符合条件的股票，使用两级缓存优化性能
        第一级：筛选缓存（按买入参数）
        第二级：排序缓存（按买入+选股参数）
        """
        global _global_filter_cache, _global_sort_cache
        global _global_current_filter_key, _global_current_sort_key

        today = self.today
        filter_key = self._filter_key
        sort_key = self._sort_key

        # 检查排序缓存（第二级缓存，命中率最高）
        if today in _global_sort_cache.get(sort_key, {}):
            self.picked_numpy_data = _global_sort_cache[sort_key][today]
            if self.debug:
                count = len(self.picked_numpy_data['code']) if self.picked_numpy_data else 0
                print(f"日期 {today} 选出股票 {count} 只 (排序缓存)")
            return

        # 检查筛选缓存（第一级缓存）
        if today in _global_filter_cache.get(filter_key, {}):
            filtered_data = _global_filter_cache[filter_key][today]
            if self.debug:
                print(f"日期 {today} 使用筛选缓存，重新排序")
        else:
            # 筛选参数变化，清理所有缓存
            if filter_key != _global_current_filter_key:
                _global_filter_cache.clear()
                _global_sort_cache.clear()
                _global_current_filter_key = filter_key
                _global_current_sort_key = ()
                if self.debug:
                    print(f"筛选参数变化，清理所有缓存: {filter_key}")

            numpy_data = self.data.get_numpy_data_by_date(today)
            if numpy_data is None or len(numpy_data['code']) == 0:
                self.picked_numpy_data = None
                if sort_key not in _global_sort_cache:
                    _global_sort_cache[sort_key] = {}
                _global_sort_cache[sort_key][today] = None
                if self.debug:
                    print(f"日期 {today} 无符合条件股票")
                return

            mask = self._pick_filter(numpy_data)
            if not mask.any():
                self.picked_numpy_data = None
                if sort_key not in _global_sort_cache:
                    _global_sort_cache[sort_key] = {}
                _global_sort_cache[sort_key][today] = None
                if self.debug:
                    print(f"日期 {today} 无符合条件股票")
                return

            # 保存筛选结果（未排序的原始数据，包含所有可能用于排序的字段）
            filtered_data = {
                'code': numpy_data['code'][mask],
                'next_open': numpy_data['next_open'][mask],
                'price_limit_status': numpy_data['price_limit_status'][mask],
                'amount': numpy_data['amount'][mask],
                'change_pct': numpy_data['change_pct'][mask],
                'change_3d': numpy_data['change_3d'][mask],
                'change_5d': numpy_data['change_5d'][mask],
                'consecutive_up_days': numpy_data['consecutive_up_days'][mask],
                'volume': numpy_data['volume'][mask],
                'close': numpy_data['close'][mask],
                'open': numpy_data['open'][mask],
                'high': numpy_data['high'][mask],
                'low': numpy_data['low'][mask],
            }

            # 存入筛选缓存
            if filter_key not in _global_filter_cache:
                _global_filter_cache[filter_key] = {}
            _global_filter_cache[filter_key][today] = filtered_data

        # 使用排序器对筛选结果排序
        sorted_indices = self._pick_sorter(filtered_data)

        self.picked_numpy_data = {
            'code': filtered_data['code'][sorted_indices],
            'next_open': filtered_data['next_open'][sorted_indices],
            'price_limit_status': filtered_data['price_limit_status'][sorted_indices],
        }

        # 存入排序缓存
        if sort_key not in _global_sort_cache:
            _global_sort_cache[sort_key] = {}
        _global_sort_cache[sort_key][today] = self.picked_numpy_data
        _global_current_sort_key = sort_key

        if self.debug:
            codes_list = list(self.picked_numpy_data['code'])
            codes_str = ','.join([str(c) for c in codes_list[:5]])
            if len(codes_list) > 5:
                codes_str += f",...({len(codes_list)-5} more)"
            print(f"日期 {today} 选出股票 {len(codes_list)} 只: {codes_str}")
    

    def buy(self) -> None:
        """执行买入操作 - 优化仓位分配"""
        numpy_data = self.picked_numpy_data
        if numpy_data is None or len(numpy_data['code']) == 0:
            return

        hold = self.hold
        hold_codes = self.hold_codes
        max_hold_count = self.max_hold_count
        current_hold_count = len(hold)
        if current_hold_count >= max_hold_count:
            return

        remaining_hold_count = max_hold_count - current_hold_count
        free_amount = self.free_amount
        today = self.today

        # 直接使用 NumPy 数组
        codes = numpy_data['code']
        next_opens = numpy_data['next_open']
        price_limit_status = numpy_data['price_limit_status']
        hold_codes_arr = np.array(list(hold_codes), dtype=codes.dtype) if hold_codes else np.array([], dtype=codes.dtype)

        # 计算买入金额 - 基于剩余资金和剩余持仓数动态分配
        # 每只买入金额 = 剩余资金 / 剩余持仓数量
        # 这样确保资金充分利用，每只仓位均匀
        base_amount = int(free_amount / remaining_hold_count)

        # 统计过滤原因
        if self.debug:
            total_candidates = len(codes)
            limit_up_count = np.sum(price_limit_status == 1)
            already_hold = sum(1 for c in codes if c in hold_codes_arr)
            print(f"日期 {today} 买入筛选: 候选{total_candidates}只|涨停{limit_up_count}只|已持仓{already_hold}只|剩余{remaining_hold_count}仓位")

        # 使用 Numba 计算买入股数（包含涨停过滤和持仓检查）
        buy_counts = _calc_buy_counts_numba(codes, next_opens, price_limit_status, hold_codes_arr, base_amount)

        # 过滤有效买入（股数>0）
        valid_mask = buy_counts > 0
        valid_codes = codes[valid_mask]
        valid_opens = next_opens[valid_mask]
        valid_counts = buy_counts[valid_mask]

        # 批量买入
        buy_limit = min(len(valid_codes), remaining_hold_count)
        if self.debug and buy_limit == 0 and len(codes) > 0:
            print(f"日期 {today} 无有效买入: 全部涨停或已持仓")
        for i in range(buy_limit):
            code = int(valid_codes[i])
            next_open = int(valid_opens[i])
            buy_count = int(valid_counts[i])
            # 加入滑点：买入价格上浮
            buy_price_with_slippage = int(next_open * (1 + self._slippage_rate))
            hold_stock = HoldStock(code, buy_price_with_slippage, buy_count, today)
            self._add_hold(hold_stock)
            # 计算成本：股票价格 + 手续费
            stock_cost = buy_count * buy_price_with_slippage
            commission = int(stock_cost * self._commission_rate)
            total_cost = stock_cost + commission
            self.free_amount -= total_cost

            if self.debug:
                free_amount = self.free_amount / 100
                cost = stock_cost / 100
                comm = commission / 100
                print(f"日期 {today} 买入 {code} , {buy_price_with_slippage/100:.2f} * {buy_count} = {cost:.2f} (含滑点), 手续费{comm:.2f}, 剩余资金 {free_amount:.2f}")

    def sell(self) -> None:
        is_debug=self.debug
        """执行卖出操作，使用统一的卖出策略"""
        hold = self.hold
        hold_length = len(hold)
        if hold_length == 0:
            return

        today = self.today
        sells_info: list[tuple[int, int, str]] = []

        # 统计卖出检查
        if is_debug:
            limit_down_count = 0
            no_data_count = 0
            skip_today_count = 0

        for i in range(hold_length):
            hold_stock = hold[i]
            code = hold_stock.code
            buy_day = hold_stock.buy_day
            if buy_day == today:
                if is_debug:
                    skip_today_count += 1
                continue

            stock_data_dict = self.data.get_data_by_date_code(today, code)
            if stock_data_dict is None:
                if is_debug:
                    no_data_count += 1
                    print(f"日期 {today} 股票 {code} 无数据,跳过卖出")
                continue

            if stock_data_dict['price_limit_status'] == 2:
                if is_debug:
                    limit_down_count += 1
                    print(f"日期 {today} 股票 {code} 跌停,无法卖出")
                continue

            # 调用统一的卖出策略
            need_sell, sell_price, reason = self._check_sell(hold_stock, stock_data_dict)
            if is_debug:
                buy_price = hold_stock.buy_price
                reason = f"\033[91m{reason}\033[0m" if sell_price > buy_price else f"\033[92m{reason}\033[0m"

            if need_sell:
                sells_info.append((code, sell_price, reason))

        if is_debug and hold_length > 0:
            print(f"日期 {today} 卖出检查: 持仓{hold_length}只|跌停{limit_down_count}只|无数据{no_data_count}只|当日买入{skip_today_count}只|触发卖出{len(sells_info)}只")
        
        sells_info_length = len(sells_info)
        if sells_info_length == 0:
            return
        
        # 批量处理卖出
        trades_history = self.trades_history
        for i in range(sells_info_length):
            code, sell_price_raw, sell_reason = sells_info[i]
            hold_stock = self._remove_hold(code)
            if hold_stock:
                buy_price = hold_stock.buy_price
                buy_count = hold_stock.buy_count
                # 加入滑点：卖出价格下浮
                sell_price = int(sell_price_raw * (1 - self._slippage_rate))
                # 计算卖出收入
                sell_revenue = sell_price * buy_count
                # 计算卖出手续费
                sell_commission = int(sell_revenue * self._commission_rate)
                # 实际到账金额
                net_revenue = sell_revenue - sell_commission
                # 计算盈亏
                cost_cents = buy_price * buy_count
                profit_cents = net_revenue - cost_cents
                self.free_amount += net_revenue
                # 计算盈亏率
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
                    comm = sell_commission // 100
                    print(f"日期 {today} 卖出 {code} {hold_stock.buy_day}->{today} {buy_price/100:.2f} -> {sell_price/100:.2f} 原因:{sell_reason} 盈亏 {profit}({profit_rate:.2%}), 手续费{comm}, 剩余资金 {free_amount:.2f}")
                

    def _check_sell(self, hold: HoldStock, stock_data_dict: dict) -> tuple[bool, int, str]:
        """
        统一卖出策略：止损 + 贪婪止盈
        使用Numba加速核心计算
        """
        is_debug = self.debug

        # 提取数据
        open_price = stock_data_dict['open']
        close_price = stock_data_dict['close']
        high_price = stock_data_dict['high']
        low_price = stock_data_dict['low']
        buy_price = hold.buy_price
        highest_price = hold.highest_price

        # 预计算关键价格
        stop_loss_price = int(buy_price * (1 + self._stop_loss_rate))
        target_return_price = int(buy_price * (1 + self._target_return))
        trailing_stop_price = int(highest_price * (1 - self._trailing_rate)) if highest_price > 0 else 0

        # 获取持仓天数
        hold_days = self.calendar.gap(hold.buy_day, self.today) if self.calendar else 0

        # 使用Numba进行核心计算
        need_sell, sell_price = _check_sell_numba(
            buy_price, highest_price, hold_days,
            open_price, close_price, high_price, low_price,
            stop_loss_price, target_return_price, trailing_stop_price,
            self._hold_days_limit
        )

        if not need_sell:
            return False, 0, EMPTY_STRING

        # 生成原因字符串（仅在debug模式）
        if is_debug:
            buy_price_yuan = buy_price / 100
            if sell_price == stop_loss_price:
                stop_loss_rate = self._stop_loss_rate * 100
                reason = f"止损|买价{buy_price_yuan:.2f}|止损率{stop_loss_rate:.0f}%|止损价{stop_loss_price/100:.2f}|开{open_price/100:.2f}|低{low_price/100:.2f}"
            elif hold_days >= self._hold_days_limit:
                target_return_pct = self._target_return * 100
                close_yuan = close_price / 100
                target_yuan = target_return_price / 100
                reason = f"到期|持仓{hold_days}天/限{self._hold_days_limit}天|目标{target_return_pct:.0f}%|目标价{target_yuan:.2f}|收{close_yuan:.2f}"
            else:
                # 移动止盈 - 区分开盘回落还是盘中回落
                trailing_rate_pct = self._trailing_rate * 100
                highest_yuan = highest_price / 100
                trailing_yuan = trailing_stop_price / 100
                high_yuan = high_price / 100
                if open_price <= trailing_stop_price:
                    reason = f"止盈|开盘回落|回撤率{trailing_rate_pct:.0f}%|最高{highest_yuan:.2f}|止盈价{trailing_yuan:.2f}|开{open_price/100:.2f}"
                else:
                    reason = f"止盈|盘中回落|回撤率{trailing_rate_pct:.0f}%|最高{highest_yuan:.2f}|止盈价{trailing_yuan:.2f}|低{low_price/100:.2f}|高{high_yuan:.2f}"
            return True, sell_price, reason

        return True, sell_price, EMPTY_STRING

    def settle_amount(self) -> None:
        """计算每日总资产并记录"""
        hold_amount = 0
        today = self.today
        hold = self.hold
        is_debug = self.debug

        for hold_stock in hold:
            code = hold_stock.code
            stock_data = self.data.get_data_by_date_code(today, code)

            if stock_data is None:
                if is_debug:
                    print(f"日期 {today} 持有 {code} 日期:{today} 无数据")
                continue

            close_price = stock_data['close']
            high_price = stock_data['high']
            buy_price = hold_stock.buy_price
            buy_count = hold_stock.buy_count

            if high_price > hold_stock.highest_price:
                hold_stock.update_highest_price(high_price)

            if is_debug:
                profit_cents = (close_price - buy_price) * buy_count
                profit_rate = (close_price - buy_price) / buy_price
                highest_price = hold_stock.highest_price
                hold_days = self.calendar.gap(hold_stock.buy_day, today) if self.calendar else 0
                # 计算关键价格用于验证止盈逻辑
                target_price = int(buy_price * (1 + self._target_return))
                stop_loss_price = int(buy_price * (1 + self._stop_loss_rate))
                trailing_stop_price = int(highest_price * (1 - self._trailing_rate)) if highest_price > 0 else 0
                print(f"日期 {today} 持有 {code} 日期:{hold_stock.buy_day}->{today} 持仓{hold_days}天 价格{buy_price/100:.2f}->{close_price/100:.2f} 最高:{highest_price/100:.2f} 累计:{profit_cents/100:.2f}({profit_rate:.2%}) 目标价{target_price/100:.2f} 止损价{stop_loss_price/100:.2f} 止盈价{trailing_stop_price/100:.2f}")

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
    
    def update_today(self, today: int) -> None:
        """更新当前日期"""
        self.today = today
        self._today_data_cache = {}
    
    def calculate_performance(self, start_date: int, end_date: int) -> BacktestResult:
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
        
        # 结算时强制卖出所有持仓，使用最后一天的收盘价
        final_holdings_value = 0
        if hold:
            # 使用最后一天的日期获取数据
            last_date = self.today
            for hold_stock in hold:
                code = hold_stock.code
                # 从数据源获取最后一天的收盘价
                stock_data_dict = self.data.get_data_by_date_code(last_date, code)
                if stock_data_dict is None:
                    # 如果没有数据，使用买入价格作为估值
                    final_holdings_value += hold_stock.buy_price * hold_stock.buy_count
                else:
                    # 使用收盘价估值
                    close_price = stock_data_dict['close']
                    final_holdings_value += close_price * hold_stock.buy_count
        
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

        # 计算资金使用率（每日持仓市值 / 总资产），限制在0-100%之间
        avg_utilization = 0
        if daily_values:
            utilization_sum = 0
            count = 0
            for dv in daily_values:
                value = dv['value']
                if value > 0:
                    # 持仓市值 = 总资产 - 可用资金，但可用资金可能为负（透支）
                    hold_value = max(0, value - dv['free_amount'])
                    utilization = min(hold_value / value, 1.0)  # 限制最大100%
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
    
