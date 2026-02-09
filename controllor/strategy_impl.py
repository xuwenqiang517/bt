import pandas as pd
import numpy as np

from dto import *
from strategy import Strategy

class UpStrategy(Strategy):
    def __init__(self, base_param_arr, sell_param_arr, buy_param_arr, debug):
        super().__init__(base_param_arr, sell_param_arr, buy_param_arr, debug)
        self.sell_chain_list = self.init_sell_strategy_chain()
        
    def _init_pick_filter(self):
        buy_up_day_min = self.buy_param_arr[0]
        buy_day3_min = self.buy_param_arr[1]
        buy_day3_max = self.buy_param_arr[2]
        buy_day5_min = self.buy_param_arr[3]
        buy_day5_max = self.buy_param_arr[4]
        
        def filter_func(df: pd.DataFrame) -> np.ndarray:
            col_consecutive = df["consecutive_up_days"].values
            col_change3d = df["change_3d"].values
            col_change5d = df["change_5d"].values
            return (
                (col_consecutive >= buy_up_day_min)
                & (col_change3d >= buy_day3_min)
                & (col_change3d <= buy_day3_max)
                & (col_change5d >= buy_day5_min)
                & (col_change5d <= buy_day5_max)
            )
        self._pick_filter = filter_func
    
    def _init_pick_sorter(self):
        max_hold = self.max_hold_count
        
        def sorter_func(df: pd.DataFrame) -> pd.DataFrame:
            n = min(max_hold, len(df))
            if n <= 0:
                return pd.DataFrame()
            
            vol_rank_values = df["vol_rank"].values
            top_n_indices = np.argpartition(vol_rank_values, n-1)[:n]
            sorted_indices = top_n_indices[np.argsort(vol_rank_values[top_n_indices])]
            return df.iloc[sorted_indices].reset_index(drop=True)
        self._pick_sorter = sorter_func
    
    def init_sell_strategy_chain(self):
        sl, sp, cd, cr = self.sell_param_arr
        strategies = []
        if sl is not None:
            strategies.append(SellStrategy("静态止损", StopLossParams(rate=sl/100.0)))
        if sp is not None:
            strategies.append(SellStrategy("静态止盈", StopProfitParams(rate=sp/100.0)))
        if cd is not None and cr is not None:
            strategies.append(SellStrategy("累计涨幅卖出", CumulativeSellParams(days=cd, min_return=cr/100.0)))
        return strategies
    
        
        