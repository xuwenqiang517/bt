"""
参数生成器模块 - 动态生成策略参数，避免内存爆炸
"""
from itertools import product
from param_config import DEFAULT_PARAM_RANGES, ParamRanges


class ParamGenerator:
    """参数生成器 - 支持动态生成和分片"""

    def __init__(self, param_ranges: ParamRanges = None):
        """
        初始化参数生成器
        Args:
            param_ranges: 参数范围配置，默认使用 DEFAULT_PARAM_RANGES
        """
        self.config = param_ranges or DEFAULT_PARAM_RANGES
        
        # 从配置中提取参数范围
        self.hold_count_range = self.config.hold_count_range
        self.buy_up_day_range = self.config.buy_up_day_range
        self.buy_day3_range = self.config.buy_day3_range
        self.buy_day5_range = self.config.buy_day5_range
        self.change_pct_max_range = self.config.change_pct_max_range
        self.limit_up_count_range = self.config.limit_up_count_range
        self.volume_ratio_range = self.config.volume_ratio_range
        self.sort_desc_range = self.config.sort_desc_range
        self.sell_stop_loss_range = self.config.sell_stop_loss_range
        self.sell_hold_days_range = self.config.sell_hold_days_range
        self.sell_target_return_range = self.config.sell_target_return_range
        self.sell_trailing_range = self.config.sell_trailing_range
        
        self.total_count = self.config.get_total_count()

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
