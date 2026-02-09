import random
import concurrent.futures
import time
from pathlib import Path
import os
from typing import List, Tuple, Dict, Any

from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import pandas as pd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
# 设置 Matplotlib 后端为非交互式的 'agg'，以支持在多线程环境中使用
plt.switch_backend('agg')
from tqdm import tqdm
from local_cache import LocalCache

from dto import *
from strategy_impl import *
# 回测结果CSV列名
RESULT_COLS_A = ['周期胜率', '平均胜率', '平均收益率', '平均交易次数', '最大资金', '最小资金', '夏普比率', '平均资金使用率', '配置']
RESULT_COLS_B = ['配置']


class Chain:
    def __init__(self, param=None):
        self.strategies = param["strategy"]  # 策略列表
        self.date_arr = param["date_arr"]  # 回测时间周期列表
        self.chain_debug = param.get("chain_debug", False)  # 是否打印报告
        self.win_rate_threshold = param.get("win_rate_threshold", 0.75)  # 胜率阈值，默认65%
        self.thread_count = param.get("thread_count", 1)  # 线程数，默认1
        
        self.param = param  # 原始参数
        self.stock_data = sd()  # 股票数据源
        self.calendar = sc()  # 交易日历
        self.result_file = param.get("result_file", None)  # 结果文件

    def _draw_fund_trend(self, daily_values, title):
        """绘制资金变化趋势图表并保存到data目录"""
        if not daily_values:
            if self.chain_debug:
                print("没有资金数据，跳过绘图")
            return
        
        from pathlib import Path
        import os
        
        # 计算保存路径（与local_cache.py中的cache_url保持一致）
        data_dir = Path(__file__).resolve().parent.parent / "./data"
        os.makedirs(data_dir, exist_ok=True)
        
        # 生成唯一的文件名
        import re
        # 移除标题中的特殊字符，确保文件名安全
        safe_title = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '_', title)
        # 截取前50个字符，避免文件名过长
        safe_title = safe_title[:50]
        file_name = f"fund_trend_{safe_title}.png"
        save_path = data_dir / file_name
        
        # 提取日期和资金值（将分转换为元）
        dates = [dv['date'] for dv in daily_values]
        values = [dv['value'] / 100 for dv in daily_values]
        
        # 调试信息
        if self.chain_debug:
            print(f"绘图数据: 日期数量={len(dates)}, 资金数量={len(values)}")
            if dates:
                print(f"起始日期: {dates[0]}, 结束日期: {dates[-1]}")
            if values:
                print(f"起始资金: {values[0]:.2f}, 结束资金: {values[-1]:.2f}")
                print(f"最大资金: {max(values):.2f}, 最小资金: {min(values):.2f}")
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(dates, values, label='总资产', linewidth=2)
        ax.set_xlabel('日期')
        ax.set_ylabel('资金')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()
        
        # 设置y轴刻度格式为普通数字，不使用科学计数法
        ax.ticklabel_format(style='plain', axis='y')
        
        # 调整日期标签
        if len(dates) > 10:
            # 只显示部分日期标签，避免重叠
            step = len(dates) // 10
            ax.set_xticks(dates[::step])
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存图表
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图表，释放内存
        
        if self.chain_debug:
            print(f"资金变化图表已保存到: {save_path}")
            print(f"图表保存成功，文件大小: {os.path.getsize(save_path) if os.path.exists(save_path) else 0} bytes")

    def _process_strategy_group(self, strategy_group: List[Dict[str, Any]], thread_id: int) -> None:
        """处理一组策略"""
        print(f"进程 {thread_id} 开始处理，策略数: {len(strategy_group)}")
        # 为每个线程创建独立的缓存
        cache = LocalCache()
        thread_result_file = f"{self.result_file}_thread_{thread_id}"
        
        # 加载线程本地缓存
        cached_a_df = cache.get_csv(f"a_{thread_result_file}")
        if cached_a_df is None:
            # 明确指定列类型，避免类型不兼容问题
            cached_a_df = pl.DataFrame({
                '周期胜率': pl.Series([], dtype=pl.String),
                '平均胜率': pl.Series([], dtype=pl.String),
                '平均收益率': pl.Series([], dtype=pl.String),
                '平均交易次数': pl.Series([], dtype=pl.Float64),
                '最大资金': pl.Series([], dtype=pl.Float64),
                '最小资金': pl.Series([], dtype=pl.Float64),
                '夏普比率': pl.Series([], dtype=pl.Float64),
                '平均资金使用率': pl.Series([], dtype=pl.String),
                '配置': pl.Series([], dtype=pl.String)
            })
        else:
            cached_a_df = pl.from_pandas(cached_a_df)
        
        cached_b_df = cache.get_csv(f"b_{thread_result_file}")
        if cached_b_df is None:
            cached_b_df = pl.DataFrame({col: pl.Series([], dtype=pl.String) for col in RESULT_COLS_B})
        else:
            cached_b_df = pl.from_pandas(cached_b_df)
        
        # 获取已执行的策略键
        executed_keys = set()
        if '配置' in cached_a_df.columns:
            executed_keys.update(cached_a_df['配置'].to_list())
        if '配置' in cached_b_df.columns:
            executed_keys.update(cached_b_df['配置'].to_list())
        
        temp_a_data = []
        temp_b_data = []
        processed_count = 0
        batch_size = 1000
        
        # 处理策略组，添加进度条
        for params in tqdm(strategy_group, desc=f"进程 {thread_id} 执行策略", total=len(strategy_group)):
            cache_key = "|".join(",".join(map(str, arr)) for arr in [[params.get('base_param_arr')[1]], params.get('buy_param_arr'), params.get('sell_param_arr')])
            if cache_key in executed_keys and not self.chain_debug:
                continue
            
            strategy = UpStrategy(**params)
            results = []
            total_periods = len(self.date_arr)
            max_failures_allowed = int(total_periods * (1 - self.win_rate_threshold))
            failure_count = 0
            successful_count = 0
            all_daily_values = []
            
            for s, e in self.date_arr:
                result = self.execute_one_strategy(strategy, s, e)
                if hasattr(strategy, 'daily_values') and strategy.daily_values:
                    all_daily_values.extend(strategy.daily_values)
                if result.总收益率 > 0:
                    successful_count += 1
                else:
                    failure_count += 1
                    if failure_count > max_failures_allowed and not self.chain_debug:
                        break
                results.append(result)
            
            actual_win_rate = successful_count / len(results) if results else 0
            if actual_win_rate >= self.win_rate_threshold and len(results) == total_periods:
                new_row = {
                    "周期胜率": f"{int(actual_win_rate * 100)}%({successful_count}/{total_periods})",
                    "平均胜率": f"{int(np.mean([x.胜率 for x in results]) * 100)}%",
                    "平均收益率": f"{np.mean([x.总收益率 for x in results]) * 100:.2f}%",
                    "平均交易次数": float(round(np.mean([x.交易次数 for x in results]), 1)),
                    "最大资金": float(round(max([x.最大资金 for x in results]), 1)),
                    "最小资金": float(round(min([x.最小资金 for x in results]), 1)),
                    "夏普比率": float(round(np.mean([x.夏普比率 for x in results]), 2)),
                    "平均资金使用率": f"{np.mean([x.平均资金使用率 for x in results]) * 100:.2f}%",
                    "配置": cache_key
                }
                temp_a_data.append(new_row)
            else:
                temp_b_data.append({"配置": cache_key})
            
            if self.chain_debug:
                if 'new_row' in locals():
                    print(f"进程 {thread_id}: {new_row}")
                self._draw_fund_trend(all_daily_values, f'进程 {thread_id} - 策略资金变化趋势 - {cache_key}')
            
            executed_keys.add(cache_key)
            processed_count += 1
            
            # 每处理10000个策略保存一次缓存
            if processed_count % batch_size == 0:
                # 转换为Polars DataFrame
                if temp_a_data:
                    temp_a_df = pl.DataFrame(temp_a_data)
                else:
                    # 明确指定列类型，避免类型不兼容问题
                    temp_a_df = pl.DataFrame({
                        '周期胜率': pl.Series([], dtype=pl.String),
                        '平均胜率': pl.Series([], dtype=pl.String),
                        '平均收益率': pl.Series([], dtype=pl.String),
                        '平均交易次数': pl.Series([], dtype=pl.Float64),
                        '最大资金': pl.Series([], dtype=pl.Float64),
                        '最小资金': pl.Series([], dtype=pl.Float64),
                        '夏普比率': pl.Series([], dtype=pl.Float64),
                        '平均资金使用率': pl.Series([], dtype=pl.String),
                        '配置': pl.Series([], dtype=pl.String)
                    })
                
                if temp_b_data:
                    temp_b_df = pl.DataFrame(temp_b_data)
                else:
                    temp_b_df = pl.DataFrame({col: pl.Series([], dtype=pl.String) for col in RESULT_COLS_B})
                
                # 保存线程本地缓存
                if not temp_a_df.is_empty():
                    if cached_a_df.is_empty():
                        cached_a_df = temp_a_df.clone()
                    else:
                        # 确保必要的列存在
                        for col in ['平均交易次数', '平均资金使用率']:
                            if col not in cached_a_df.columns:
                                cached_a_df = cached_a_df.with_columns(pl.lit(None).alias(col))
                        # 合并数据
                        cached_a_df = pl.concat([cached_a_df, temp_a_df], rechunk=True)
                    # 排序
                    cached_a_df = cached_a_df.sort(
                        by=['平均胜率', '平均收益率'], 
                        descending=[True, True]
                    )
                    # 转换为pandas DataFrame以保存到缓存
                    cache.set_csv(f"a_{thread_result_file}", cached_a_df.to_pandas())
                
                if not temp_b_df.is_empty():
                    if cached_b_df.is_empty():
                        cached_b_df = temp_b_df.clone()
                    else:
                        cached_b_df = pl.concat([cached_b_df, temp_b_df], rechunk=True)
                    # 转换为pandas DataFrame以保存到缓存
                    cache.set_csv(f"b_{thread_result_file}", cached_b_df.to_pandas())
                
                # 清空临时数据，准备下一批
                temp_a_data = []
                temp_b_data = []
        
        # 处理剩余的策略数据
        if temp_a_data or temp_b_data:
            # 转换为Polars DataFrame
            if temp_a_data:
                temp_a_df = pl.DataFrame(temp_a_data)
            else:
                # 明确指定列类型，避免类型不兼容问题
                temp_a_df = pl.DataFrame({
                    '周期胜率': pl.Series([], dtype=pl.String),
                    '平均胜率': pl.Series([], dtype=pl.String),
                    '平均收益率': pl.Series([], dtype=pl.String),
                    '平均交易次数': pl.Series([], dtype=pl.Float64),
                    '最大资金': pl.Series([], dtype=pl.Float64),
                    '最小资金': pl.Series([], dtype=pl.Float64),
                    '夏普比率': pl.Series([], dtype=pl.Float64),
                    '平均资金使用率': pl.Series([], dtype=pl.String),
                    '配置': pl.Series([], dtype=pl.String)
                })
            
            if temp_b_data:
                temp_b_df = pl.DataFrame(temp_b_data)
            else:
                temp_b_df = pl.DataFrame({col: pl.Series([], dtype=pl.String) for col in RESULT_COLS_B})
            
            # 保存线程本地缓存
            if not temp_a_df.is_empty():
                if cached_a_df.is_empty():
                    cached_a_df = temp_a_df.clone()
                else:
                    # 确保必要的列存在
                    for col in ['平均交易次数', '平均资金使用率']:
                        if col not in cached_a_df.columns:
                            cached_a_df = cached_a_df.with_columns(pl.lit(None).alias(col))
                    # 合并数据
                    cached_a_df = pl.concat([cached_a_df, temp_a_df], rechunk=True)
                # 排序
                cached_a_df = cached_a_df.sort(
                    by=['平均胜率', '平均收益率'], 
                    descending=[True, True]
                )
                # 转换为pandas DataFrame以保存到缓存
                cache.set_csv(f"a_{thread_result_file}", cached_a_df.to_pandas())
            
            if not temp_b_df.is_empty():
                if cached_b_df.is_empty():
                    cached_b_df = temp_b_df.clone()
                else:
                    cached_b_df = pl.concat([cached_b_df, temp_b_df], rechunk=True)
                # 转换为pandas DataFrame以保存到缓存
                cache.set_csv(f"b_{thread_result_file}", cached_b_df.to_pandas())

    def _merge_thread_caches(self) -> None:
        """合并所有线程的缓存文件"""
        cache = LocalCache()
        
        # 加载主缓存
        main_a_df = cache.get_csv(f"a_{self.result_file}")
        if main_a_df is None:
            # 明确指定列类型，避免类型不兼容问题
            main_a_df = pl.DataFrame({
                '周期胜率': pl.Series([], dtype=pl.String),
                '平均胜率': pl.Series([], dtype=pl.String),
                '平均收益率': pl.Series([], dtype=pl.String),
                '平均交易次数': pl.Series([], dtype=pl.Float64),
                '最大资金': pl.Series([], dtype=pl.Float64),
                '最小资金': pl.Series([], dtype=pl.Float64),
                '夏普比率': pl.Series([], dtype=pl.Float64),
                '平均资金使用率': pl.Series([], dtype=pl.String),
                '配置': pl.Series([], dtype=pl.String)
            })
        else:
            main_a_df = pl.from_pandas(main_a_df)
        
        main_b_df = cache.get_csv(f"b_{self.result_file}")
        if main_b_df is None:
            main_b_df = pl.DataFrame({col: pl.Series([], dtype=pl.String) for col in RESULT_COLS_B})
        else:
            main_b_df = pl.from_pandas(main_b_df)
        
        # 合并所有线程的缓存
        for thread_id in range(self.thread_count):
            thread_result_file = f"{self.result_file}_thread_{thread_id}"
            
            # 合并a文件
            thread_a_df = cache.get_csv(f"a_{thread_result_file}")
            if thread_a_df is not None and not thread_a_df.empty:
                thread_a_df = pl.from_pandas(thread_a_df)
                main_a_df = pl.concat([main_a_df, thread_a_df], rechunk=True)
                # 删除已合并的线程缓存文件
                cache.delete_file(f"a_{thread_result_file}.csv")
            
            # 合并b文件
            thread_b_df = cache.get_csv(f"b_{thread_result_file}")
            if thread_b_df is not None and not thread_b_df.empty:
                thread_b_df = pl.from_pandas(thread_b_df)
                main_b_df = pl.concat([main_b_df, thread_b_df], rechunk=True)
                # 删除已合并的线程缓存文件
                cache.delete_file(f"b_{thread_result_file}.csv")
        
        # 去重并排序
        if not main_a_df.is_empty():
            main_a_df = main_a_df.unique(subset=['配置'])
            main_a_df = main_a_df.sort(
                by=['平均胜率', '平均收益率'], 
                descending=[True, True]
            )
            # 转换为pandas DataFrame以保存到缓存
            cache.set_csv(f"a_{self.result_file}", main_a_df.to_pandas())
        
        if not main_b_df.is_empty():
            main_b_df = main_b_df.unique(subset=['配置'])
            # 转换为pandas DataFrame以保存到缓存
            cache.set_csv(f"b_{self.result_file}", main_b_df.to_pandas())
        
        print(f"合并完成，主文件 a_{self.result_file}.csv 包含 {len(main_a_df)} 条记录")

    def execute(self) -> list:
        # 先合并所有线程的缓存，确保中断后再运行时能正确加载之前的结果
        self._merge_thread_caches()
        
        # 初始化主缓存
        cache = LocalCache()
        main_a_df = cache.get_csv(f"a_{self.result_file}")
        if main_a_df is None:
            # 明确指定列类型，避免类型不兼容问题
            main_a_df = pl.DataFrame({
                '周期胜率': pl.Series([], dtype=pl.String),
                '平均胜率': pl.Series([], dtype=pl.String),
                '平均收益率': pl.Series([], dtype=pl.String),
                '平均交易次数': pl.Series([], dtype=pl.Float64),
                '最大资金': pl.Series([], dtype=pl.Float64),
                '最小资金': pl.Series([], dtype=pl.Float64),
                '夏普比率': pl.Series([], dtype=pl.Float64),
                '平均资金使用率': pl.Series([], dtype=pl.String),
                '配置': pl.Series([], dtype=pl.String)
            })
        else:
            main_a_df = pl.from_pandas(main_a_df)
        
        main_b_df = cache.get_csv(f"b_{self.result_file}")
        if main_b_df is None:
            main_b_df = pl.DataFrame({col: pl.Series([], dtype=pl.String) for col in RESULT_COLS_B})
        else:
            main_b_df = pl.from_pandas(main_b_df)
        
        # 获取已执行的策略键
        executed_keys = set()
        if '配置' in main_a_df.columns:
            executed_keys.update(main_a_df['配置'].to_list())
        if '配置' in main_b_df.columns:
            executed_keys.update(main_b_df['配置'].to_list())
        
        # 过滤掉已执行的策略（如果不是调试模式）
        remaining_strategies = []
        for s in self.strategies:
            cache_key = "|".join(",".join(map(str, arr)) for arr in [[s.get('base_param_arr')[1]], s.get('buy_param_arr'), s.get('sell_param_arr')])
            if cache_key not in executed_keys or self.chain_debug:
                remaining_strategies.append(s)
        
        total_strategies = len(remaining_strategies)
        print(f"总策略数: {len(self.strategies)}, 已执行: {len(self.strategies) - total_strategies}, 剩余: {total_strategies}")
        print(f"使用进程数: {self.thread_count}")
        
        if total_strategies == 0:
            print("所有策略已执行完毕，无需处理")
            return
        
        # 策略分组
        group_size = total_strategies // self.thread_count
        strategy_groups = []
        for i in range(self.thread_count):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < self.thread_count - 1 else total_strategies
            strategy_groups.append(remaining_strategies[start_idx:end_idx])
            print(f"进程 {i} 处理策略数: {len(remaining_strategies[start_idx:end_idx])}")
        
        # 并行处理
        print(f"创建进程池，最大进程数: {self.thread_count}")
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.thread_count) as executor:
            print("提交任务到进程池...")
            future_to_group = {executor.submit(self._process_strategy_group, group, i): i for i, group in enumerate(strategy_groups)}
            print(f"已提交 {len(future_to_group)} 个任务到进程池")
            
            print("等待进程完成...")
            for future in concurrent.futures.as_completed(future_to_group):
                thread_id = future_to_group[future]
                try:
                    future.result()
                    print(f"进程 {thread_id} 处理完成")
                except Exception as exc:
                    print(f"进程 {thread_id} 处理失败: {exc}")
        
        # 所有策略执行完成后，合并所有线程的缓存
        self._merge_thread_caches()
        
        print("所有策略执行完成")
        return


    def execute_one_strategy(self, strategy, start_date, end_date) -> BacktestResult:
        scalendar = self.calendar
        current_idx = scalendar.start(start_date)
        end_idx = scalendar.start(end_date)
        
        # 日期不符合直接抛异常
        if current_idx == -1 or end_idx == -1:
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")
        
        strategy.bind(self.stock_data, self.calendar)
        strategy.reset()

        while current_idx != -1 and current_idx <= end_idx:
            current_date = scalendar.get_date(current_idx)
            strategy.update_today(current_date)
            strategy.buy()
            strategy.sell()
            strategy.pick()
            strategy.settle_amount()
            current_idx = scalendar.next(current_idx)
        
        result = strategy.calculate_performance(start_date, end_date)
        
        if self.chain_debug:
            print("=" * 50)
            print(f"时间周期: {result.起始日期} 至 {result.结束日期}")
            print(f"资金: {result.初始资金/100:.2f} - > {result.最终资金/100:.2f}")
            print(f"总收益率: {result.总收益率:.2f}%")
            print(f"胜率: {result.胜率:.2f}%")
            print(f"交易次数: {result.交易次数}")
            print(f"最大资金: {result.最大资金/100:.2f}")
            print(f"最小资金: {result.最小资金/100:.2f}")
            print(f"夏普比率: {result.夏普比率:.2f}")
            print(f"平均资金使用率: {result.平均资金使用率:.2f}%")
            print("=" * 50)
        
        return result
