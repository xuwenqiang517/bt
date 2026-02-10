from stock_calendar import StockCalendar as sc
from chain import Chain
import random
def bt_all():
    day_array=sc().get_date_arr()
    result_file=f"all_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-vol_rank正排"
    # 定义策略参数字典列表（不创建对象，省内存）
    strategy_params_list=[]
    for a in range(2,6,1): # 持仓数量
        for buy1 in range(2,4,1): # 连涨天数
            for buy2 in range(3,10,2): # 3日涨幅最低
                for buy3 in range(5,15,2): # 5日涨幅最低
                    for sell1 in range(-10,-4,1): # 止损率（负数，如-5表示-5%）
                        for sell2 in range(3,6,1): # 持仓天数
                            for sell3 in range(2,15,1): # 目标涨幅
                                for sell4 in range(3,10,1): # 最低盈利阈值
                                    for sell5 in range(3,10,2): # 移动止盈回撤率
                                        strategy_params_list.append({
                                            "base_param_arr": [100000, a],
                                            "buy_param_arr": [buy1, buy2, buy3],
                                            "sell_param_arr": [sell1, sell2, sell3, sell4, sell5],
                                            "debug": 0
                                        })
    print(f"策略参数数量: {len(strategy_params_list)}")
    # 随机打散
    random.shuffle(strategy_params_list)

    param={
        "strategy":strategy_params_list
        ,"date_arr":day_array
        ,"chain_debug":0
        ,"result_file":result_file
        ,"processor_count":4
    }
    chain = Chain(param=param)
    chain.execute()

def bt_one(strategy_params,day_array):
    result_file=f"one_连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-vol_rank正排"

    base_arr, buy_arr, sell_arr = strategy_params.split("|")
    strategy_params_list=[]
    strategy_params_list.append({
        "base_param_arr": [10000000, int(base_arr.split(",")[0])],
        "buy_param_arr": list(map(int, buy_arr.split(","))),
        "sell_param_arr": list(map(int, sell_arr.split(","))),
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
    bt_all()
    # bt_one("2|3,5,10|-5,3,10,2,5",[["20250101","20250201"]])
    # bt_one("2|3,5,10|-5,10",[["20250101","20250201"]])
    # bt_one("2|3,5,10|-5,10",[["20250101","20260101"]])
