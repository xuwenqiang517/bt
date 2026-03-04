from stock_calendar import StockCalendar as sc
from chain import Chain
import random
from param_config import parse_strategy_string, build_strategy_string

# 导入日志配置（会自动重定向 stdout 到文件）
import logger_config


def bt_all(processor_count, fail_count, strategy_params=None, max_strategy_count=1000000000, force_refresh=False):
    from datetime import datetime
    day_array = sc().get_date_arr()
    current_time = datetime.now().strftime("%m%d_%H%M")
    result_file = f"all_{current_time}"

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
            "total_strategy_count": total_count,
            "run_year": True  # 确保传递run_year参数
        }
        chain = Chain(param=param)
        chain.execute_generator_mode()


def str2dict(strategy_params: str) -> list:
    """
    将策略参数字符串解析为字典列表（单策略）
    使用 param_config 中的统一解析函数
    """
    params = parse_strategy_string(strategy_params)
    params["debug"] = 1  # 单策略模式开启调试
    print([params])
    return [params]



if __name__ == "__main__":
    s = """
    1|-1,-1,8,15,15,20,2|0|-9,6,6,5
    """

    bt_all(processor_count=1, fail_count=40, strategy_params=None, max_strategy_count=100)
    # bt_all(processor_count=4,fail_count=2,strategy_params=s,max_strategy_count=1000000000)
