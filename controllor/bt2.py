from stock_calendar import StockCalendar as sc
from chain import Chain
import random

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
def bt_all(processor_count,fail_count,strategy_params=None,max_strategy_count=1000000000):
    day_array=sc().get_date_arr()
    result_file=f"all_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-最多失败{fail_count}"
    # 定义策略参数字典列表（不创建对象，省内存）

    strategy_params_list=[]
    if strategy_params is not None:
        strategy_params_list=str2dict(strategy_params)
    else:
        for a in range(2,5,1): # 持仓数量
            for buy1 in range(1,5,1): # 连涨天数
                for buy2 in range(3,15,1): # 3日涨幅最低
                    for buy3 in range(3,20,1): # 5日涨幅最低
                        # for sell1 in range(-15,-4,2): # 止损率（负数，如-5表示-5%）
                        #     for sell2 in range(1,6,1): # 持仓天数
                        #         for sell3 in range(5,21,1): # 目标涨幅
                        #             for sell4 in range(3,15,1): # 移动止盈回撤率
                        strategy_params_list.append({
                            "base_param_arr": [10000000, a],
                            "buy_param_arr": [buy1, buy2, buy3],
                            # "sell_param_arr": [sell1, sell2, sell3, sell4],
                            "sell_param_arr": [-15, 2, 8, 3],
                            "debug": 0
                        })
    strategy_params_list=strategy_params_list[:max_strategy_count]
    print(f"策略参数数量: {len(strategy_params_list)}")
    # 随机打散
    random.shuffle(strategy_params_list)

    param={
        "strategy":strategy_params_list
        ,"date_arr":day_array
        ,"chain_debug":0 if strategy_params is None else 1
        ,"result_file":result_file
        ,"processor_count":processor_count
        ,"fail_count":fail_count
    }
    chain = Chain(param=param)
    chain.execute()

def str2dict(strategy_params):
    base_arr, buy_arr, sell_arr = strategy_params.split("|")
    strategy_params_list=[]
    strategy_params_list.append({
        "base_param_arr": [10000000, int(base_arr.split(",")[0])],
        "buy_param_arr": list(map(int, buy_arr.split(","))),
        "sell_param_arr": list(map(int, sell_arr.split(","))),
        "debug": 1
    })
    print(strategy_params_list)
    return strategy_params_list

def bt_one(strategy_params,day_array):
    result_file=f"one_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}"
    param={
        "strategy":str2dict(strategy_params)
        ,"date_arr":day_array
        ,"chain_debug":1
        ,"result_file":result_file
    }
    chain = Chain(param=param)
    chain.execute()


if __name__ == "__main__":
    s="""
    3|2,10,15|-15,2,8,3
    """

    bt_all(processor_count=4,fail_count=2,strategy_params=None,max_strategy_count=1000000000)
    # bt_all(processor_count=4,fail_count=2,strategy_params=s,max_strategy_count=1000000000)
    # bt_one(s,sc().get_date_arr())
    # bt_one(s,[[20250101,20250201]])
    # bt_one(s,[[20250101,20260101]])