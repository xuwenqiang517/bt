import polars as pl
import numpy as np
from numba import njit, int8, int32, float32, boolean

from dto import *
from strategy import Strategy

@njit(cache=True, parallel=True)
def _filter_numba(up_days, change_3d, change_5d, change_pct,
                  buy_up_day_min, buy_day3_min, buy_day5_min):
    """Numba JIT 编译的筛选函数，并行执行"""
    n = len(up_days)
    result = np.empty(n, dtype=np.bool_)
    for i in range(n):
        result[i] = (up_days[i] >= buy_up_day_min and
                     change_3d[i] > buy_day3_min and
                     change_5d[i] > buy_day5_min and
                     change_pct[i] < 5)
    return result

class UpStrategy(Strategy):
    def __init__(self, base_param_arr, sell_param_arr, buy_param_arr, debug):
        super().__init__(base_param_arr, sell_param_arr, buy_param_arr, debug)
        self.sell_chain_list = self.init_sell_strategy_chain()

    def _init_pick_filter(self):
        buy_up_day_min = self.buy_param_arr[0]
        buy_day3_min = self.buy_param_arr[1]
        buy_day5_min = self.buy_param_arr[2]

        def filter_func(numpy_data: dict):
            # 使用 Numba JIT 编译的筛选函数
            mask = _filter_numba(
                numpy_data['consecutive_up_days'],
                numpy_data['change_3d'],
                numpy_data['change_5d'],
                numpy_data['change_pct'],
                buy_up_day_min, buy_day3_min, buy_day5_min
            )
            return mask
        self._pick_filter = filter_func
    
    # def _init_pick_sorter(self):
    #     max_hold = self.max_hold_count
        
    #     def sorter_func(df: pl.DataFrame) -> pl.DataFrame:
    #         n = min(max_hold, len(df))
    #         if n <= 0:
    #             return pl.DataFrame()
            
    #         return df.sort("vol_rank").head(n).with_row_count().drop("row_nr")
    #     self._pick_sorter = sorter_func
    
    def init_sell_strategy_chain(self):
        sell1, sell2, sell3, sell4 = self.sell_param_arr
        strategies = []
        # if sell1 is not None:
        strategies.append(SellStrategy("静态止损", sell1/100.0))
        # 贪婪止盈：需要days, min_return, trailing_stop_rate三个参数
        strategies.append(SellStrategy("贪婪止盈", (sell2, sell3/100.0, sell4/100.0)))
        # 假设sell_param_arr的第五个参数用于时间止盈
        # if len(self.sell_param_arr) >= 5 and self.sell_param_arr[4] is not None:
        #     time_days = 
        # strategies.append(SellStrategy("时间止盈", sell2))
        return strategies
    
        
        