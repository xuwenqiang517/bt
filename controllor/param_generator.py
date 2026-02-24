"""
参数生成器模块 - 动态生成策略参数，避免内存爆炸
"""
from itertools import product
import math


class ParamGenerator:
    """参数生成器 - 支持动态生成和分片"""

    def __init__(self):
        # 基础参数范围 - 动态仓位，只保留持仓数量
        # 扩大持仓数量范围
        self.hold_count_range = [1, 2, 3]                   # 持仓数量: 2-5只

        # 买入参数范围（6个参数）
        # 连涨天数: -1表示不限，增加4,5天更长的连涨趋势
        self.buy_up_day_range = [-1, 1, 2, 3, 4, 5]
        # 3日涨幅: -1表示不限，补充中间值
        self.buy_day3_range = [-1, 3, 4, 5, 6, 7, 8]
        # 5日涨幅: -1表示不限，补充中间值和更高涨幅
        self.buy_day5_range = [-1, 5, 6, 8, 10, 12, 15]
        # 当日涨幅上限: -1表示不限，更精细的买入时机控制
        self.change_pct_max_range = [-1, 1, 2, 3, 4, 5]
        # 涨停条件: -1表示不限，0表示10天内0涨停（排除涨停股），1表示10天内≥1次涨停
        self.limit_up_count_range = [-1, 0, 1]
        # 量比: -1表示不限，增加更极端的放量情况
        self.volume_ratio_range = [-1, 1, 1.5, 2, 2.5, 3]

        # 选股排序参数（1个参数）
        # 排序方向: 0=成交量升序(冷门股)，1=成交量降序(热门股)
        self.sort_desc_range = [0, 1]

        # 卖出参数
        # 止损率: 更精细的梯度
        self.sell_stop_loss_range = [-5, -6, -7, -8, -10]
        # 持仓天数: 增加5天更长持仓周期
        self.sell_hold_days_range = [2, 3, 4, 5]
        # 目标涨幅: 提高最低目标到5%，增加更高收益目标
        self.sell_target_return_range = [5, 8, 10, 12, 15, 20]
        # 回撤率: 增加更宽松的止盈回撤
        self.sell_trailing_range = [2, 3, 4, 5, 6, 7, 8]

        # 计算总组合数（动态仓位，买入参数6个+排序参数1个+卖出参数4个+持仓数量1个）
        self.total_count = (
            len(self.hold_count_range) *
            len(self.buy_up_day_range) *
            len(self.buy_day3_range) *
            len(self.buy_day5_range) *
            len(self.change_pct_max_range) *
            len(self.limit_up_count_range) *
            len(self.volume_ratio_range) *
            len(self.sort_desc_range) *
            len(self.sell_stop_loss_range) *
            len(self.sell_hold_days_range) *
            len(self.sell_target_return_range) *
            len(self.sell_trailing_range)
        )

    def get_total_count(self) -> int:
        """获取总参数组合数"""
        return self.total_count

    def _index_to_params(self, index: int) -> dict:
        """将索引转换为参数组合
        优化顺序：买入参数放前面（提高缓存命中率），基础参数和卖出参数放后面
        """
        # 买入参数（影响选股缓存）放前面，这样相同买入参数会连续执行
        buy_ranges = [
            self.buy_up_day_range,      # 连涨天数
            self.buy_day3_range,        # 3日涨幅
            self.buy_day5_range,        # 5日涨幅
            self.change_pct_max_range,  # 涨幅上限
            self.limit_up_count_range,  # 涨停条件
            self.volume_ratio_range,    # 量比
        ]
        # 选股排序参数（影响排序缓存）
        pick_ranges = [
            self.sort_desc_range,       # 排序方向
        ]
        # 基础参数（影响仓位）放中间 - 动态仓位，只保留持仓数量
        base_ranges = [
            self.hold_count_range,      # 持仓数量
        ]
        # 卖出参数放最后
        sell_ranges = [
            self.sell_stop_loss_range,
            self.sell_hold_days_range,
            self.sell_target_return_range,
            self.sell_trailing_range
        ]

        all_ranges = buy_ranges + pick_ranges + base_ranges + sell_ranges

        indices = []
        remaining = index
        for r in reversed(all_ranges):
            size = len(r)
            indices.append(remaining % size)
            remaining //= size
        indices.reverse()

        # 买入参数（0-5）
        buy1 = self.buy_up_day_range[indices[0]]
        buy2 = self.buy_day3_range[indices[1]]
        buy3 = self.buy_day5_range[indices[2]]
        buy4 = self.change_pct_max_range[indices[3]]
        buy5 = self.limit_up_count_range[indices[4]]
        buy6 = self.volume_ratio_range[indices[5]]

        # 选股排序参数（6）
        sort_desc = self.sort_desc_range[indices[6]]

        # 基础参数（7）- 只保留持仓数量
        hold_count = self.hold_count_range[indices[7]]

        # 卖出参数（8-11）
        sell1 = self.sell_stop_loss_range[indices[8]]
        sell2 = self.sell_hold_days_range[indices[9]]
        sell3 = self.sell_target_return_range[indices[10]]
        sell4 = self.sell_trailing_range[indices[11]]

        return {
            "base_param_arr": [10000000, hold_count],
            "buy_param_arr": [buy1, buy2, buy3, buy4, buy5, buy6],
            "pick_param_arr": [sort_desc],
            "sell_param_arr": [sell1, sell2, sell3, sell4],
            "debug": 0
        }

    def get_slice_params(self, start_idx: int, end_idx: int):
        """获取指定索引范围的参数生成器"""
        for i in range(start_idx, min(end_idx, self.total_count)):
            yield self._index_to_params(i)

    def get_worker_slice(self, worker_id: int, total_workers: int):
        """
        为指定工作进程获取参数切片
        按买入参数组分配，确保每个进程内的买入参数连续，提高缓存命中率
        """
        # 计算买入参数组合数
        buy_combo_size = (
            len(self.buy_up_day_range) *
            len(self.buy_day3_range) *
            len(self.buy_day5_range) *
            len(self.change_pct_max_range) *
            len(self.limit_up_count_range) *
            len(self.volume_ratio_range)
        )
        # 每个买入参数对应的基础+卖出参数组合数
        base_sell_combo_size = self.total_count // buy_combo_size

        # 将买入参数组均匀分配给各进程
        buy_base_size = buy_combo_size // total_workers
        buy_remainder = buy_combo_size % total_workers

        if worker_id < buy_remainder:
            start_buy = worker_id * (buy_base_size + 1)
            end_buy = start_buy + buy_base_size + 1
        else:
            start_buy = buy_remainder * (buy_base_size + 1) + (worker_id - buy_remainder) * buy_base_size
            end_buy = start_buy + buy_base_size

        # 转换为实际索引（每个买入参数对应 base_sell_combo_size 个策略）
        start_idx = start_buy * base_sell_combo_size
        end_idx = end_buy * base_sell_combo_size

        return start_idx, end_idx

    def get_worker_param_count(self, worker_id: int, total_workers: int) -> int:
        """获取指定工作进程应处理的参数数量"""
        base_size = self.total_count // total_workers
        remainder = self.total_count % total_workers
        return base_size + 1 if worker_id < remainder else base_size

    def get_all_indices(self) -> range:
        """获取所有参数索引的迭代器"""
        return range(self.total_count)
