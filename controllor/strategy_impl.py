import numpy as np
from numba import njit

from strategy import Strategy

# 排序字段映射
SORT_FIELD_MAP = {
    0: 'amount',              # 成交金额
    1: 'change_pct',          # 当日涨跌幅
    2: 'change_3d',           # 3日累计涨跌幅
    3: 'change_5d',           # 5日累计涨跌幅
    4: 'consecutive_up_days', # 连续上涨天数
    5: 'volume',              # 成交量
    6: 'close',               # 收盘价
    7: 'open',                # 开盘价
    8: 'high',                # 最高价
    9: 'low'                  # 最低价
}


@njit(cache=True)
def _filter_numba(up_days, change_3d, change_5d, change_pct, limit_up_count_20d,
                  buy_up_day_min, buy_day3_min, buy_day5_min, change_pct_max, limit_up_count_min):
    """Numba JIT 编译的筛选函数"""
    n = len(up_days)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        # 基础条件
        base_ok = (up_days[i] >= buy_up_day_min and
                   change_3d[i] > buy_day3_min and
                   change_5d[i] > buy_day5_min and
                   change_pct[i] < change_pct_max)
        # 20天涨停次数条件（0表示不限制）
        if limit_up_count_min > 0:
            result[i] = base_ok and (limit_up_count_20d[i] >= limit_up_count_min)
        else:
            result[i] = base_ok
    return result


@njit(cache=True)
def _argsort_numba(arr, desc):
    """Numba实现的argsort，返回排序后的索引"""
    n = len(arr)
    indices = np.arange(n)
    # 简单的冒泡排序（对于小数组足够快）
    for i in range(n):
        for j in range(i + 1, n):
            if desc:
                if arr[indices[i]] < arr[indices[j]]:
                    indices[i], indices[j] = indices[j], indices[i]
            else:
                if arr[indices[i]] > arr[indices[j]]:
                    indices[i], indices[j] = indices[j], indices[i]
    return indices


class UpStrategy(Strategy):
    pass

    def _init_pick_filter(self):
        buy_up_day_min = self.buy_param_arr[0]
        buy_day3_min = self.buy_param_arr[1]
        buy_day5_min = self.buy_param_arr[2]
        change_pct_max = self.buy_param_arr[3] if len(self.buy_param_arr) > 3 else 5  # 当日涨幅上限，默认5%
        limit_up_count_min = self.buy_param_arr[4] if len(self.buy_param_arr) > 4 else 0  # 20天内涨停次数最低要求，0表示不限制

        def filter_func(numpy_data: dict):
            mask = _filter_numba(
                numpy_data['consecutive_up_days'],
                numpy_data['change_3d'],
                numpy_data['change_5d'],
                numpy_data['change_pct'],
                numpy_data['limit_up_count_20d'],
                buy_up_day_min, buy_day3_min, buy_day5_min, change_pct_max, limit_up_count_min
            )
            return mask
        self._pick_filter = filter_func

    def _init_pick_sorter(self):
        """初始化排序函数，根据参数动态排序"""
        sort_field_idx = self.pick_param_arr[0] if len(self.pick_param_arr) > 0 else 0
        sort_desc = self.pick_param_arr[1] if len(self.pick_param_arr) > 1 else 1  # 0=升序(小到大), 1=降序(大到小)

        sort_field = SORT_FIELD_MAP.get(sort_field_idx, 'amount')
        desc = bool(sort_desc)

        def sorter_func(numpy_data: dict, mask: np.ndarray) -> np.ndarray:
            """返回排序后的索引数组"""
            if not mask.any():
                return np.array([], dtype=np.int64)

            row_indices = np.nonzero(mask)[0]
            if len(row_indices) <= 1:
                return row_indices

            # 获取排序字段的值
            sort_values = numpy_data[sort_field][row_indices]

            # 使用Numba排序
            sorted_local_indices = _argsort_numba(sort_values, desc)
            return row_indices[sorted_local_indices]

        self._pick_sorter = sorter_func
