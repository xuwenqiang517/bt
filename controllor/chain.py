import random

from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import pandas as pd
import numpy as np
from tqdm import tqdm
from local_cache import LocalCache

from dto import *
from strategy_impl import *
# 回测结果CSV列名
RESULT_COLS_A = ['周期胜率', '平均胜率', '平均收益率', '平均最大回撤', '平均交易次数', '平均资金使用率', '配置']
RESULT_COLS_B = ['配置']


class Chain:
    def __init__(self, param=None):
        self.strategies = param["strategy"]  # 策略列表
        self.date_arr = param["date_arr"]  # 回测时间周期列表
        self.chain_debug = param.get("chain_debug", False)  # 是否打印报告
        self.cache = LocalCache()  # 本地缓存
        self.param = param  # 原始参数
        self.stock_data = sd()  # 股票数据源
        self.calendar = sc()  # 交易日历
        self.result_file = param.get("result_file", None)  # 结果文件

    def execute(self) -> list:
        cached_a_df = self.cache.get_csv(f"a_{self.result_file}")
        if cached_a_df is None:
            cached_a_df = pd.DataFrame(columns=RESULT_COLS_A)
        
        cached_b_df = self.cache.get_csv(f"b_{self.result_file}")
        if cached_b_df is None:
            cached_b_df = pd.DataFrame(columns=RESULT_COLS_B)
        
        executed_keys = set(cached_a_df['配置'].tolist()) if '配置' in cached_a_df.columns else set()
        b_keys = set(cached_b_df['配置'].tolist()) if '配置' in cached_b_df.columns else set()
        executed_keys.update(b_keys)
        
        print("已缓存执行策略数:", len(executed_keys))
        
        temp_a_df = pd.DataFrame(columns=RESULT_COLS_A)
        temp_b_df = pd.DataFrame(columns=RESULT_COLS_B)
        
        total_strategies = len(self.strategies)
        print(f"总策略数: {total_strategies}, 已缓存: {len(executed_keys)}")
        
        for idx, strategy_params in tqdm(enumerate(self.strategies), desc="执行策略", total=total_strategies):
            strategy = UpStrategy(**strategy_params)

            param_join_str = "|".join(",".join(map(str, arr)) for arr in [
                [strategy.base_param_arr[1]], strategy.buy_param_arr, strategy.sell_param_arr
            ])

            if param_join_str in executed_keys and not self.chain_debug:
                continue

            results = []
            all_win = True
            for s, e in self.date_arr:
                result = self.execute_one_strategy(strategy, s, e)
                if result.总收益率 <= 0 and not self.chain_debug:
                    all_win = False
                    break
                results.append(result)
            
            # if not results:
            #     continue
            
            if all_win:
                win_count = len(results)
                total_count = len(results)
                win_rate = win_count / total_count
                new_row = {
                    "周期胜率": f"{int(win_rate * 100)}%({win_count}/{total_count})",
                    "平均胜率": f"{int(np.mean([x.胜率 for x in results]) * 100)}%",
                    "平均收益率": f"{np.mean([x.总收益率 for x in results]) * 100:.2f}%",
                    "平均最大回撤": f"{np.mean([x.最大回撤 for x in results]) * 100:.2f}%",
                    "平均交易次数": round(np.mean([x.交易次数 for x in results]), 1),
                    "平均资金使用率": f"{np.mean([x.平均资金使用率 for x in results]) * 100:.2f}%",
                    "配置": param_join_str
                }
                temp_a_df.loc[len(temp_a_df)] = new_row
            else:
                temp_b_df.loc[len(temp_b_df)] = {"配置": param_join_str}
            if self.chain_debug:
                print(new_row)
            executed_keys.add(param_join_str)

            # 跑完了也写一下
            if idx % 10000 == 0 or idx == total_strategies - 1:
                if cached_a_df.empty:
                    cached_a_df = temp_a_df.copy()
                else:
                    for col in ['平均交易次数', '平均资金使用率']:
                        if col not in cached_a_df.columns:
                            cached_a_df[col] = np.nan
                    cached_a_df = pd.concat([cached_a_df, temp_a_df], ignore_index=True)
                temp_a_df = pd.DataFrame(columns=RESULT_COLS_A)

                # cached_a_df 按平均胜率、平均收益率 排序
                cached_a_df = cached_a_df.sort_values(by=['平均胜率', '平均收益率'], ascending=[False, False])
                self.cache.set_csv(f"a_{self.result_file}", cached_a_df)
                
                if cached_b_df.empty:
                    cached_b_df = temp_b_df.copy()
                else:
                    cached_b_df = pd.concat([cached_b_df, temp_b_df], ignore_index=True)
                temp_b_df = pd.DataFrame(columns=RESULT_COLS_B)
                self.cache.set_csv(f"b_{self.result_file}", cached_b_df)


    def execute_one_strategy(self, strategy, start_date, end_date) -> BacktestResult:
        scalendar = self.calendar
        current_idx = scalendar.start(start_date)
        end_idx = scalendar.start(end_date)
        if current_idx == -1 or end_idx == -1:
            return BacktestResult(
                起始日期=start_date,
                结束日期=end_date,
                初始资金=0,
                最终资金=0,
                总收益=0,
                总收益率=0,
                胜率=0,
                盈亏比=0,
                交易次数=0,
                最大回撤=0,
                平均资金使用率=0
            )
        strategy.bind(self.stock_data, self.calendar)
        strategy.reset()

        while current_idx != -1 and current_idx <= end_idx:
            current_date = scalendar.get_date(current_idx)
            strategy.update_today(current_date)
            strategy._new_day()
            strategy.buy()
            strategy.sell()
            strategy.pick()
            strategy.print_daily()
            current_idx = scalendar.next(current_idx)
        
        perf = strategy.calculate_performance()
        
        result = BacktestResult(
             起始日期=start_date,
             结束日期=end_date,
             初始资金=perf['init_amount'],
             最终资金=perf['final_amount'],
             总收益=perf['total_return'],
             总收益率=perf['total_return_pct'],
             胜率=perf['win_rate'],
             盈亏比=perf['profit_loss_ratio'],
             交易次数=perf['trade_count'],
             最大回撤=perf['max_drawdown'],
             平均资金使用率=perf['avg_utilization']
         )
        
        if self.chain_debug:
            print("=" * 50)
            print(f"时间周期: {start_date} 至 {end_date}")
            print(f"资金: {perf['init_amount']:.2f} - > {perf['final_amount']:.2f}")
            print(f"总收益: {perf['total_return']:.2f} ({perf['total_return_pct']:.2%})")
            print(f"胜率: {perf['win_rate']:.2%}")
            print(f"盈亏比: {perf['profit_loss_ratio']:.2f}")
            print(f"交易次数: {perf['trade_count']}")
            print(f"最大回撤: {perf['max_drawdown']:.2%}")
            print(f"资金使用率: {perf['avg_utilization']:.2%}")
        
        return result
