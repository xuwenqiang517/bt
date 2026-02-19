"""
参数生成器模块 - 动态生成策略参数，避免内存爆炸
"""
from itertools import product
import math


class ParamGenerator:
    """参数生成器 - 支持动态生成和分片"""

    def __init__(self):
        # 基础参数范围
        self.hold_count_range = [2, 3]                         # 持仓数量: 2-3只
        self.position_value_range = [25, 30, 35, 40, 45, 50]   # 仓位比例: 25-50%，精细测试

        # 买入参数范围 - 扩大范围精细测试
        self.buy_up_day_range = [1, 2, 3, 4, 5]                # 连涨天数: 1-5天
        self.buy_day3_range = [3, 5, 8, 10, 13, 15]            # 3日涨幅: 3-15%，更细粒度
        self.buy_day5_range = [3, 5, 8, 10, 13, 15, 18, 20]    # 5日涨幅: 3-20%，更细粒度
        self.change_pct_max_range = [2, 3, 4, 5, 6, 7]         # 当日涨幅上限: 2-7%
        self.limit_up_count_idx_range = [0, 1, 2]              # 涨停天数选择: 0=15天, 1=20天, 2=30天
        self.limit_up_count_min_range = [0, 1, 2]              # 涨停次数: 0-2次
        self.sort_field_range = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # 排序字段: 全字段测试
        self.sort_desc_range = [0, 1]                          # 排序方式: 升序/降序

        # 卖出参数范围 - 根据持仓天数优化止损范围
        self.sell_stop_loss_range = list(range(-12, -4, 1))    # 止损率: -12到-5%，适配短线持仓
        self.sell_hold_days_range = [1, 2, 3, 4, 5]            # 持仓天数: 1-5天
        self.sell_target_return_range = [3, 4, 5, 6, 7, 8, 9, 10, 12, 15]  # 目标涨幅: 3-15%
        self.sell_trailing_range = [2, 3, 4, 5, 8, 10, 13, 15, 18, 20]     # 回撤率: 2-20%

        # 计算总组合数（去掉固定的2个参数）
        self.total_count = (
            len(self.hold_count_range) *
            len(self.position_value_range) *
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
        """将索引转换为参数组合"""
        ranges = [
            self.hold_count_range,
            self.position_value_range,
            self.buy_up_day_range,
            self.buy_day3_range,
            self.buy_day5_range,
            self.change_pct_max_range,
            self.limit_up_count_idx_range,
            self.limit_up_count_min_range,
            self.sort_field_range,
            self.sort_desc_range,
            self.sell_stop_loss_range,
            self.sell_hold_days_range,
            self.sell_target_return_range,
            self.sell_trailing_range
        ]

        indices = []
        remaining = index
        for r in reversed(ranges):
            size = len(r)
            indices.append(remaining % size)
            remaining //= size
        indices.reverse()

        hold_count = self.hold_count_range[indices[0]]
        buy_first = 0       # 固定先卖后买
        position_mode = 1   # 固定比例模式
        position_value = self.position_value_range[indices[1]]
        buy1 = self.buy_up_day_range[indices[2]]
        buy2 = self.buy_day3_range[indices[3]]
        buy3 = self.buy_day5_range[indices[4]]
        buy4 = self.change_pct_max_range[indices[5]]
        buy5 = self.limit_up_count_idx_range[indices[6]]   # 涨停天数选择
        buy6 = self.limit_up_count_min_range[indices[7]]   # 涨停次数
        sort_field = self.sort_field_range[indices[8]]
        sort_desc = self.sort_desc_range[indices[9]]
        sell1 = self.sell_stop_loss_range[indices[10]]
        sell2 = self.sell_hold_days_range[indices[11]]
        sell3 = self.sell_target_return_range[indices[12]]
        sell4 = self.sell_trailing_range[indices[13]]

        # 转换仓位值：固定金额模式时，将万转换为分
        if position_mode == 2 and position_value > 0:
            position_value_converted = position_value * 1000000  # 万->分
        else:
            position_value_converted = position_value

        return {
            "base_param_arr": [10000000, hold_count, buy_first, position_mode, position_value_converted],
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
        均匀分配任务，避免重复
        """
        base_size = self.total_count // total_workers
        remainder = self.total_count % total_workers

        # 前 remainder 个进程多分配一个任务
        if worker_id < remainder:
            start_idx = worker_id * (base_size + 1)
            end_idx = start_idx + base_size + 1
        else:
            start_idx = remainder * (base_size + 1) + (worker_id - remainder) * base_size
            end_idx = start_idx + base_size

        return start_idx, end_idx

    def get_worker_param_count(self, worker_id: int, total_workers: int) -> int:
        """获取指定工作进程应处理的参数数量"""
        base_size = self.total_count // total_workers
        remainder = self.total_count % total_workers
        return base_size + 1 if worker_id < remainder else base_size

    def get_all_indices(self) -> range:
        """获取所有参数索引的迭代器"""
        return range(self.total_count)
