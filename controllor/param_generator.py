"""
参数生成器模块 - 动态生成策略参数，避免内存爆炸
"""
from itertools import product
import math


class ParamGenerator:
    """参数生成器 - 支持动态生成和分片"""

    def __init__(self):
        # 买入参数范围
        self.hold_count_range = list(range(2, 11, 1))          # 持仓数量
        self.buy_up_day_range = list(range(1, 5, 1))           # 连涨天数
        self.buy_day3_range = list(range(3, 15, 5))            # 3日涨幅最低
        self.buy_day5_range = list(range(3, 20, 5))            # 5日涨幅最低
        self.change_pct_max_range = list(range(3, 11, 2))      # 当日涨幅上限: 3,5,7,9
        self.limit_up_count_min_range = [0, 1, 2]              # 20天内涨停次数最低要求: 0=不限制, 1=至少1次, 2=至少2次
        self.sort_field_range = list(range(0, 10, 1))          # 排序字段
        self.sort_desc_range = [0, 1]                          # 排序方式: 0=升序(小到大), 1=降序(大到小)

        # 卖出参数范围
        self.sell_stop_loss_range = list(range(-20, -4, 1))    # 止损率
        self.sell_hold_days_range = list(range(2, 6, 1))       # 持仓天数
        self.sell_target_return_range = list(range(5, 20, 1))  # 目标涨幅
        self.sell_trailing_range = list(range(3, 20, 5))       # 移动止盈回撤率

        # 计算总组合数
        self.total_count = (
            len(self.hold_count_range) *
            len(self.buy_up_day_range) *
            len(self.buy_day3_range) *
            len(self.buy_day5_range) *
            len(self.change_pct_max_range) *
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
            self.buy_up_day_range,
            self.buy_day3_range,
            self.buy_day5_range,
            self.change_pct_max_range,
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

        a = self.hold_count_range[indices[0]]
        buy1 = self.buy_up_day_range[indices[1]]
        buy2 = self.buy_day3_range[indices[2]]
        buy3 = self.buy_day5_range[indices[3]]
        buy4 = self.change_pct_max_range[indices[4]]
        buy5 = self.limit_up_count_min_range[indices[5]]
        sort_field = self.sort_field_range[indices[6]]
        sort_desc = self.sort_desc_range[indices[7]]
        sell1 = self.sell_stop_loss_range[indices[8]]
        sell2 = self.sell_hold_days_range[indices[9]]
        sell3 = self.sell_target_return_range[indices[10]]
        sell4 = self.sell_trailing_range[indices[11]]

        return {
            "base_param_arr": [10000000, a],
            "buy_param_arr": [buy1, buy2, buy3, buy4, buy5],
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
