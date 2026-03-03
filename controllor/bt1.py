from chain import Chain
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
    # from datetime import datetime
    # current_time = datetime.now().strftime("%m%d_%H%M")
    # result_file = f"one_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-{current_time}"
    param = {
        "strategy": str2dict(strategy_params),
        "date_arr": day_array,
        "chain_debug": 1,
        # "result_file": result_file,
        "run_year": run_year
    }
    chain = Chain(param=param)
    chain.execute()


if __name__ == "__main__":
    s = """
    1|-1,-1,8,15,15,20,2|0|-9,6,6,5
    """
    bt_one(s,[[20250101,20260101]])
    # bt_one(s,[[20250101,20260301]])
    