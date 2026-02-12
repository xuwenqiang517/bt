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
    day_array = sc().get_date_arr()
    result_file = f"all_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-最多失败{fail_count}"

    strategy_params_list = []
    if strategy_params is not None:
        strategy_params_list = str2dict(strategy_params)
    else:
        # 买入参数范围
        hold_count_range = range(2, 11, 1)          # 持仓数量
        buy_up_day_range = range(1, 5, 1)           # 连涨天数
        buy_day3_range = range(3, 15, 5)            # 3日涨幅最低
        buy_day5_range = range(3, 20, 5)            # 5日涨幅最低
        sort_field_range = range(0, 10, 1)          # 排序字段: 0=amount, 1=change_pct, 2=change_3d, 3=change_5d, 4=consecutive_up_days, 5=volume, 6=close, 7=open, 8=high, 9=low
        sort_desc_range = [0, 1]                    # 排序方式: 0=升序, 1=降序

        # 卖出参数范围
        sell_stop_loss_range = range(-20, -4, 5)    # 止损率
        sell_hold_days_range = range(2, 6, 1)       # 持仓天数
        sell_target_return_range = range(5, 21, 2)  # 目标涨幅
        sell_trailing_range = range(2, 10, 2)       # 移动止盈回撤率

        # 使用 itertools.product 生成所有参数组合
        for (a, buy1, buy2, buy3, sort_field, sort_desc,
             sell1, sell2, sell3, sell4) in product(
            hold_count_range, buy_up_day_range, buy_day3_range, buy_day5_range,
            sort_field_range, sort_desc_range,
            sell_stop_loss_range, sell_hold_days_range, sell_target_return_range, sell_trailing_range
        ):
            strategy_params_list.append({
                "base_param_arr": [10000000, a],
                "buy_param_arr": [buy1, buy2, buy3],
                "pick_param_arr": [sort_field, sort_desc],
                "sell_param_arr": [sell1, sell2, sell3, sell4],
                "debug": 0
            })

    strategy_params_list = strategy_params_list[:max_strategy_count]
    print(f"策略参数数量: {len(strategy_params_list)}")
    random.shuffle(strategy_params_list)

    param = {
        "strategy": strategy_params_list,
        "date_arr": day_array,
        "chain_debug": 0 if strategy_params is None else 1,
        "result_file": result_file,
        "processor_count": processor_count,
        "fail_count": fail_count,
        "force_refresh": force_refresh
    }
    chain = Chain(param=param)
    chain.execute()


def str2dict(strategy_params):
    # 格式: 持仓数量|连涨天数,3日涨幅,5日涨幅|排序字段,排序方式|止损率,持仓天数,目标涨幅,回撤率
    parts = strategy_params.strip().split("|")
    base_arr = parts[0]
    buy_arr = parts[1]
    pick_arr = parts[2] if len(parts) > 2 else "0,1"
    sell_arr = parts[3] if len(parts) > 3 else "-15,2,8,3"

    strategy_params_list = []
    strategy_params_list.append({
        "base_param_arr": [10000000, int(base_arr.split(",")[0])],
        "buy_param_arr": list(map(int, buy_arr.split(","))),
        "pick_param_arr": list(map(int, pick_arr.split(","))),
        "sell_param_arr": list(map(int, sell_arr.split(","))),
        "debug": 1
    })
    print(strategy_params_list)
    return strategy_params_list


def bt_one(strategy_params, day_array):
    result_file = f"one_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}"
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
    3|2,10,15|0,1|-15,2,8,3
    """

    bt_all(processor_count=4, fail_count=1, strategy_params=None, max_strategy_count=1000000000)
    # bt_all(processor_count=4,fail_count=2,strategy_params=s,max_strategy_count=1000000000)
    # bt_one(s,sc().get_date_arr())
    # bt_one(s,[[20250101,20250201]])
    # bt_one(s,[[20250101,20260101]])
