from stock_calendar import StockCalendar as sc
from chain import Chain
import random
from itertools import product

# """
#     3|2,10,15|-15,2,8,3
#     时间周期: 20250101 至 20260101
#     时间周期: 20250101 至 20260101
#     资金: 100000.00 - > 314042.00
#     总收益率: 214.04%
#     胜率: 44.64%
#     交易次数: 336
#     最大资金: 314232.00
#     最小资金: 91406.00
#     夏普比率: 2.87
#     平均资金使用率: 53.00%
# """

def bt_all(processor_count, fail_count, strategy_params=None, max_strategy_count=1000000000, force_refresh=False):
    from datetime import datetime
    day_array = sc().get_date_arr()
    current_time = datetime.now().strftime("%m%d_%H%M")
    result_file = f"all_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-最多失败{fail_count}-{current_time}"

    if strategy_params is not None:
        # 指定参数模式 - 直接解析
        strategy_params_list = str2dict(strategy_params)
        strategy_params_list = strategy_params_list[:max_strategy_count]
        print(f"策略参数数量: {len(strategy_params_list)}")
        random.shuffle(strategy_params_list)

        param = {
            "strategy": strategy_params_list,
            "date_arr": day_array,
            "chain_debug": 1,
            "result_file": result_file,
            "processor_count": processor_count,
            "fail_count": fail_count,
            "force_refresh": force_refresh
        }
        chain = Chain(param=param)
        chain.execute()
    else:
        # 参数搜索模式 - 使用生成器，不预生成列表
        from param_generator import ParamGenerator
        gen = ParamGenerator()
        total_count = min(gen.get_total_count(), max_strategy_count)
        print(f"总策略参数数量: {total_count}")

        param = {
            "strategy": None,  # 不使用预生成列表
            "date_arr": day_array,
            "chain_debug": 0,
            "result_file": result_file,
            "processor_count": processor_count,
            "fail_count": fail_count,
            "force_refresh": force_refresh,
            "use_param_generator": True,
            "param_generator": gen,
            "total_strategy_count": total_count
        }
        chain = Chain(param=param)
        chain.execute_generator_mode()


def str2dict(strategy_params):
    # 格式: 持仓数量,买卖顺序,仓位模式,仓位值|连涨天数,3日涨幅,5日涨幅,涨幅上限,涨停次数|排序字段,排序方式|止损率,持仓天数,目标涨幅,回撤率
    # 示例: 3,1,0,0|2,10,15,5,0|0,1|-15,2,8,3
    parts = strategy_params.strip().split("|")
    base_arr = parts[0]
    buy_arr = parts[1]
    pick_arr = parts[2] if len(parts) > 2 else "0,1"
    sell_arr = parts[3] if len(parts) > 3 else "-15,2,8,3"

    # 解析基础参数，新格式：持仓数量,仓位比例（买卖顺序和仓位模式已固定）
    base_params = list(map(int, base_arr.split(",")))
    hold_count = base_params[0]
    position_value = base_params[1] if len(base_params) > 1 else 30  # 仓位比例，默认30%
    buy_first = 0  # 固定先卖后买
    position_mode = 1  # 固定比例模式

    # 解析买入参数，新格式6个参数：连涨天数,3日涨幅,5日涨幅,涨幅上限,涨停天数选择,涨停次数
    buy_params = list(map(int, buy_arr.split(",")))
    if len(buy_params) < 6:
        # 补齐默认值
        defaults = [2, 8, 8, 3, 1, 0]  # 连涨2天,3日8%,5日8%,涨幅上限3%,20天,0次涨停
        buy_params.extend(defaults[len(buy_params):])

    strategy_params_list = []
    strategy_params_list.append({
        "base_param_arr": [10000000, hold_count, buy_first, position_mode, position_value],
        "buy_param_arr": buy_params,
        "pick_param_arr": list(map(int, pick_arr.split(","))),
        "sell_param_arr": list(map(int, sell_arr.split(","))),
        "debug": 1
    })
    print(strategy_params_list)
    return strategy_params_list


def bt_one(strategy_params, day_array):
    from datetime import datetime
    current_time = datetime.now().strftime("%m%d_%H%M")
    result_file = f"one_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-{current_time}"
    param = {
        "strategy": str2dict(strategy_params),
        "date_arr": day_array,
        "chain_debug": 1,
        "result_file": result_file
    }
    chain = Chain(param=param)
    chain.execute()


if __name__ == "__main__":
    s = """
    2,0,1,40|2,8,8,3,1|5,0|-20,2,5,3
    """

    bt_all(processor_count=4, fail_count=0, strategy_params=None, max_strategy_count=1000000000)
    # bt_all(processor_count=4,fail_count=2,strategy_params=s,max_strategy_count=1000000000)
    # bt_one(s,sc().get_date_arr())
    # bt_one(s,[[20250101,20250201]])
    # bt_one(s,[[20250101,20260101]])
    

# /Users/JDb/miniconda3/envs/py311/bin/python /Users/JDb/Desktop/github/bt/controllor/bt2.py
