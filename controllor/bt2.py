import random

from stock_calendar import StockCalendar as sc

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
                                for sell2 in range(2,5,1): # 时间止盈
                                    # for sell3 in range(2,5,1): # 止盈持有时间
                                    #     for sell4 in range(3,10,1): # 止盈持有收益率
                                    strategy_params_list.append({
                                        "base_param_arr": [100000, a],
                                        "buy_param_arr": [buy1, buy2, buy3, buy4, buy5],
                                        "sell_param_arr": [sell1, sell2],
                                        "debug": 0
                                    })
    print(f"策略参数数量: {len(strategy_params_list)}")
    # 随机打散
    random.shuffle(strategy_params_list)

    # 测试 先用1000个策略
    # strategy_params_list = strategy_params_list[:1]
    
    param={
        "strategy":strategy_params_list
        ,"date_arr":day_array
        ,"chain_debug":0
        ,"result_file":result_file
        ,"processor_count":1
    }
    chain = Chain(param=param)
    chain.execute()

def bt_one(strategy_params,day_array,result_file):
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
    
    day_array=sc().get_date_arr()
    # day_array=[["20250101","20250201"]]

    result_file=f"连涨{day_array[0][0]}-{day_array[-1][1]}-{len(day_array)}-vol_rank正排"

    bt_all(day_array,result_file)
    # bt_one("4|3,3,10,8,25|-3,10",day_array,result_file)
