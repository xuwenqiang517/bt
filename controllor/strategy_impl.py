import polars as pl

from dto import *
from strategy import Strategy

class UpStrategy(Strategy):
    def __init__(self, base_param_arr, sell_param_arr, buy_param_arr, debug):
        super().__init__(base_param_arr, sell_param_arr, buy_param_arr, debug)
        self.sell_chain_list = self.init_sell_strategy_chain()
        
    def _init_pick_filter(self):
        buy_up_day_min = self.buy_param_arr[0]
        buy_day3_min = self.buy_param_arr[1]
        buy_day5_min = self.buy_param_arr[2]
        
        def filter_func(df: pl.DataFrame) -> pl.Series:
            # 使用列缓存减少重复访问
            consecutive_up_days = df["consecutive_up_days"]
            change_3d = df["change_3d"]
            change_5d = df["change_5d"]
            change_pct = df["change_pct"]
            
            # 应用筛选条件
            return (
                      (consecutive_up_days >= buy_up_day_min)
                    & (change_3d > buy_day3_min)
                    & (change_5d > buy_day5_min)
                    & (change_pct < 5)
                )
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
    
        
        