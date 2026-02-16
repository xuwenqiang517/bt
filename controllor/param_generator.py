"""
参数生成器模块 - 动态生成策略参数，避免内存爆炸
"""
from itertools import product
import math


class ParamGenerator:
    """参数生成器 - 支持动态生成和分片"""

    def __init__(self):
        # 基础参数范围 - 根据历史结果优化
        self.hold_count_range = [2, 3, 4, 5, 7, 10]            # 持仓数量: 2-5只便于操作，7-10只参考高收益
        self.buy_first_range = [0, 1]                          # 买卖顺序: 先卖后买可能提高资金利用率
        self.position_mode_range = [0, 1]                      # 仓位模式: 主要测试剩余均分和固定比例
        self.position_value_range = [0, 15, 20, 25]            # 仓位值: 15-25%比例较合理

        # 买入参数范围 - 根据100%胜率策略特征优化
        self.buy_up_day_range = [2, 3, 4]                      # 连涨天数: 2-4天，4天表现最好
        self.buy_day3_range = [3, 8, 13]                       # 3日涨幅: 3,8,13，13出现频率高
        self.buy_day5_range = [8, 13, 18]                      # 5日涨幅: 8,13,18，13和18表现好
        self.change_pct_max_range = [3, 5, 7]                  # 当日涨幅上限: 3-7%，3出现频率最高
        self.limit_up_count_min_range = [0, 1]                 # 涨停次数: 0或1次，0表现更好
        self.sort_field_range = [3, 4, 5]                      # 排序字段: 5日涨幅、连续上涨天数、成交量
        self.sort_desc_range = [0, 1]                          # 排序方式: 0=升序, 1=降序

        # 卖出参数范围 - 根据结果优化
        self.sell_stop_loss_range = list(range(-20, -12, 1))   # 止损率: -20到-13，-15到-20表现好
        self.sell_hold_days_range = [2, 3, 4]                  # 持仓天数: 2-4天，2天表现最好
        self.sell_target_return_range = [5, 6, 7, 8, 9]        # 目标涨幅: 5-9%，5-6表现好
        self.sell_trailing_range = [3, 8, 13, 18]              # 回撤率: 3,8,13,18，3和8出现频率高

        # 计算总组合数
        self.total_count = (
            len(self.hold_count_range) *
            len(self.buy_first_range) *
            len(self.position_mode_range) *
            len(self.position_value_range) *
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
            self.buy_first_range,
            self.position_mode_range,
            self.position_value_range,
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

        hold_count = self.hold_count_range[indices[0]]
        buy_first = self.buy_first_range[indices[1]]
        position_mode = self.position_mode_range[indices[2]]
        position_value = self.position_value_range[indices[3]]
        buy1 = self.buy_up_day_range[indices[4]]
        buy2 = self.buy_day3_range[indices[5]]
        buy3 = self.buy_day5_range[indices[6]]
        buy4 = self.change_pct_max_range[indices[7]]
        buy5 = self.limit_up_count_min_range[indices[8]]
        sort_field = self.sort_field_range[indices[9]]
        sort_desc = self.sort_desc_range[indices[10]]
        sell1 = self.sell_stop_loss_range[indices[11]]
        sell2 = self.sell_hold_days_range[indices[12]]
        sell3 = self.sell_target_return_range[indices[13]]
        sell4 = self.sell_trailing_range[indices[14]]

        # 转换仓位值：固定金额模式时，将万转换为分
        if position_mode == 2 and position_value > 0:
            position_value_converted = position_value * 1000000  # 万->分
        else:
            position_value_converted = position_value

        return {
            "base_param_arr": [10000000, hold_count, buy_first, position_mode, position_value_converted],
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
