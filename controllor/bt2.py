import random

import sys
from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import NamedTuple, Callable
from datetime import date, datetime
from local_cache import LocalCache


from dto import *
from strategy import Strategy
from chain import Chain

        




def bt_all(day_array,result_file):
    # 定义策略参数字典列表（不创建对象，省内存）
    strategy_params_list=[]
    for a in range(2,6,1): # 持仓数量
        for buy1 in range(2,4,1): # 连涨天数
            for buy2 in range(3,10,2): # 3日涨幅最低
                for buy3 in range(5,15,5): # 3日涨幅最高
                    for buy4 in range(5,15,3): # 5日涨幅最低
                        for buy5 in range(15,45,5): # 5日涨幅最高
                            for sell1 in range(-15,-4,1): # 止损率（负数，如-5表示-5%）
                                for sell2 in range(15,100,5): # 止盈率
                                    for sell3 in range(3,6,1): # 止盈持有时间
                                        for sell4 in range(5,40,5): # 止盈持有收益率
                                            strategy_params_list.append({
                                                "base_param_arr": [100000, a],
                                                "buy_param_arr": [buy1, buy2, buy3, buy4, buy5],
                                                "sell_param_arr": [sell1, sell2, sell3, sell4],
                                                "debug": 0
                                            })
    print(f"策略参数数量: {len(strategy_params_list)}")
    
    # 随机打散
    random.shuffle(strategy_params_list)
    random.shuffle(strategy_params_list)
    random.shuffle(strategy_params_list)

    # 测试 先用1000个策略
    # strategy_params_list = strategy_params_list[:1]
    

    
    param={
        "strategy":strategy_params_list
        ,"date_arr":day_array
        ,"chain_debug":0
        ,"result_file":result_file
    }
        # ,"date_arr":[
        #     ["20240701","20240801"],["20240801","20240901"],["20240901","20241001"],["20241001","20241101"],["20241101","20241201"],["20241201","20250101"]
        #     ,["20250101","20250201"],["20250201","20250301"],["20250301","20250401"],["20250401","20250501"],["20250501","20250601"],["20250601","20250701"]
        #     ,["20250701","20250801"],["20250801","20250901"],["20250901","20251001"],["20251001","20251101"],["20251101","20251201"],["20251201","20260101"]
        #     ,["20260101","20260201"]
        #              ]
        # ,"date_arr":[["20250101","20260101"]]
    chain = Chain(param=param)
    chain.execute()

def bt_one(strategy_params,day_array,result_file):
    base_arr, buy_arr, sell_arr = strategy_params.split("|")
    a= int(base_arr.split(",")[0])
    buy1, buy2, buy3, buy4, buy5 = map(int, buy_arr.split(","))
    sell1, sell2, sell3, sell4 = map(int, sell_arr.split(","))
    
    strategy_params_list=[]
    strategy_params_list.append({
        "base_param_arr": [100000, a],
        "buy_param_arr": [buy1, buy2, buy3, buy4, buy5],
        "sell_param_arr": [sell1, sell2, sell3, sell4],
        "debug": 1
    })
    param={
        "strategy":strategy_params_list
        ,"date_arr":day_array
        ,"chain_debug":1
        ,"result_file":result_file
    }
    chain = Chain(param=param)
    chain.execute()


if __name__ == "__main__":
    start_time=datetime.now().timestamp()*1000
    
    start_date="20240701"
    end_date="20260206"
    part=6
    day_array=sc().get_date_arr()
    day_array=[["20250101","20260101"]]

    result_file=f"连涨{start_date}-{end_date}-{part}-vol_rank正排"

    # bt_all(day_array,result_file)

    # 100%(6/6),48%,17.70%,23.14%,124.0,49.20%,
    bt_one("4|2,5,10,8,15|-11,60,5,25",day_array,result_file)


    end_time=datetime.now().timestamp()*1000
    print(f"回测完成 耗时{(end_time-start_time):.2f}ms")
