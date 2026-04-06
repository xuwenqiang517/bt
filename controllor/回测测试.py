from stock_calendar import StockCalendar as sc
from chain import Chain
import random
from param_config import parse_strategy_string, build_strategy_string
from 回测 import bt_all, bt_one

# 导入日志配置（会自动重定向 stdout 到文件）
import logger_config



if __name__ == "__main__":
    s = """
    1|-1,-1,8,15,15,20,2|0|-9,6,6,5
    """

    #批量回测
    bt_all(processor_count=1, fail_count=29, strategy_params=None, max_strategy_count=100)
    #单策略回测
    bt_one(s,[[20250101,20260401]])
    