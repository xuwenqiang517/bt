import logger_config  # 导入日志配置，重定向stdout到log.txt
from chain import Chain
from stock_calendar import StockCalendar as sc
from param_config import parse_strategy_string

def str2dict(strategy_params: str) -> list:
    """
    将策略参数字符串解析为字典列表（单策略）
    使用 param_config 中的统一解析函数
    """
    params = parse_strategy_string(strategy_params)
    params["debug"] = 1  # 单策略模式开启调试
    print([params])
    return [params]


def bt_one(strategy_params, day_array, run_year=False):
    param = {
        "strategy": str2dict(strategy_params),
        "date_arr": day_array,
        "chain_debug": 1,
        "run_year": run_year
    }
    chain = Chain(param=param)
    chain.execute()


if __name__ == "__main__":
    s = """
    1|-1,-1,10,30,15,-1,2|1|-9,8,10,3
    """
    bt_one(s,sc().get_date_arr())
    # bt_one(s,[[20250101,20260101]])
    # bt_one(s,[[20250101,20260302]])
    # bt_one(s,[[20260101,20260302]])
    