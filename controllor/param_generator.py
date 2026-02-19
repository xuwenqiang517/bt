"""
参数生成器模块 - 动态生成策略参数，避免内存爆炸
"""
from itertools import product
import math


class ParamGenerator:
    """参数生成器 - 支持动态生成和分片"""

    def __init__(self):
        # 基础参数范围 - 动态仓位，只保留持仓数量
        self.hold_count_range = [2, 3]                         # 持仓数量: 2-3只

        # 买入参数范围 - 根据回测结果优化，支持-1表示禁用
        # 连涨天数: 1天(62%)、2天(37%)有效，3天极少，保留1-3天+禁用
        self.buy_up_day_range = [-1, 1, 2, 3]
        # 3日涨幅: 10%(30%)、5%(27%)、3%(24%)、8%(15%)有效
        self.buy_day3_range = [-1, 3, 5, 8, 10]
        # 5日涨幅: 3%(39%)、5%(32%)、8%(20%)有效
        self.buy_day5_range = [-1, 3, 5, 8]
        # 当日涨幅上限: 2%(47%)、3%(32%)、4%(21%)有效
        self.change_pct_max_range = [-1, 2, 3, 4]
        # 涨停天数: 15天(42%)、20天(32%)、30天(27%)都有效
        self.limit_up_count_idx_range = [0, 1, 2]
        # 涨停次数: 1次(40%)、0次(37%)、2次(23%)有效
        self.limit_up_count_min_range = [-1, 0, 1, 2]
        # 排序字段: 20日涨幅(49%)绝对主导，保留前几个有效的
        self.sort_field_range = [0, 4, 5, 6, 7]
        # 排序方式: 升序(64%)多于降序(36%)
        self.sort_desc_range = [0, 1]

        # 卖出参数 - 丰富参数范围（止损率合理范围-8%到-15%）
        self.sell_stop_loss_range = [-8, -10, -12, -15]   # 止损率: -8%, -10%, -12%, -15%
        self.sell_hold_days_range = [2, 3]                # 持仓天数: 2天, 3天
        self.sell_target_return_range = [3, 4, 5, 6]      # 目标涨幅: 3%, 4%, 5%, 6%
        self.sell_trailing_range = [2, 3, 4]              # 回撤率: 2%, 3%, 4%

        # 计算总组合数（动态仓位，去掉position_value参数）
        self.total_count = (
            len(self.hold_count_range) *
            len(self.buy_up_day_range) *
            len(self.buy_day3_range) *
            len(self.buy_day5_range) *
            len(self.change_pct_max_range) *
            len(self.limit_up_count_idx_range) *
            len(self.limit_up_count_min_range) *
            len(self.sort_field_range) *
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
            self.limit_up_count_idx_range,  # 涨停天数选择
            self.limit_up_count_min_range,  # 涨停次数
            self.sort_field_range,      # 排序字段
            self.sort_desc_range,       # 排序方式
        ]
        # 基础参数（影响仓位）放中间 - 动态仓位，只保留持仓数量
        base_ranges = [
            self.hold_count_range,      # 持仓数量
        ]
        # 卖出参数（固定1个）放最后
        sell_ranges = [
            self.sell_stop_loss_range,
            self.sell_hold_days_range,
            self.sell_target_return_range,
            self.sell_trailing_range
        ]

        all_ranges = buy_ranges + base_ranges + sell_ranges

        indices = []
        remaining = index
        for r in reversed(all_ranges):
            size = len(r)
            indices.append(remaining % size)
            remaining //= size
        indices.reverse()

        # 买入参数（0-7）
        buy1 = self.buy_up_day_range[indices[0]]
        buy2 = self.buy_day3_range[indices[1]]
        buy3 = self.buy_day5_range[indices[2]]
        buy4 = self.change_pct_max_range[indices[3]]
        buy5 = self.limit_up_count_idx_range[indices[4]]
        buy6 = self.limit_up_count_min_range[indices[5]]
        sort_field = self.sort_field_range[indices[6]]
        sort_desc = self.sort_desc_range[indices[7]]

        # 基础参数（8）- 动态仓位，只保留持仓数量
        hold_count = self.hold_count_range[indices[8]]
        buy_first = 0       # 固定先卖后买

        # 卖出参数（9-12）
        sell1 = self.sell_stop_loss_range[indices[9]]
        sell2 = self.sell_hold_days_range[indices[10]]
        sell3 = self.sell_target_return_range[indices[11]]
        sell4 = self.sell_trailing_range[indices[12]]

        return {
            "base_param_arr": [10000000, hold_count, buy_first],
            "buy_param_arr": [buy1, buy2, buy3, buy4, buy5, buy6],
            "pick_param_arr": [sort_field, sort_desc],
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
            len(self.limit_up_count_idx_range) *
            len(self.limit_up_count_min_range) *
            len(self.sort_field_range) *
            len(self.sort_desc_range)
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
