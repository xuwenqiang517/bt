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

# 回测结果CSV列名
RESULT_COLS_A = ['周期胜率', '平均胜率', '平均收益率', '平均最大回撤', '平均交易次数', '平均资金使用率', '配置']
RESULT_COLS_B = ['配置']


HoldStock=NamedTuple("HoldStock", [
    ("code", str),
    ("buy_price", float),
    ("buy_count", int),
    ("buy_day", str),
    ("sell_price", float),
    ("sell_day", str)
])

SellStrategy=NamedTuple("SellStrategy", [
    ("name", str),
    ("params", object)  # 改为object类型，支持不同类型的参数对象
])

StopLossParams=NamedTuple("StopLossParams", [
    ("rate", float)
])

StopProfitParams=NamedTuple("StopProfitParams", [
    ("rate", float)
])

CumulativeSellParams=NamedTuple("CumulativeSellParams", [
    ("days", int),
    ("min_return", float)
])

BacktestResult=NamedTuple("BacktestResult", [
    ("起始日期", str),
    ("结束日期", str),
    ("初始资金", float),
    ("最终资金", float),
    ("总收益", float),
    ("总收益率", float),
    ("胜率", float),
    ("盈亏比", float),
    ("交易次数", int),
    ("最大回撤", float),
    ("平均资金使用率", float)
])

StrategyBacktestResult=NamedTuple("StrategyBacktestResult", [
    ("策略配置", dict),
    ("平均收益率", float),
    ("平均胜率", float),
    ("周期胜率", float),
    ("平均最大回撤", float)
])
