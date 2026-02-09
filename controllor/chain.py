import random

from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        self.win_rate_threshold = param.get("win_rate_threshold", 0.65)  # 胜率阈值，默认65%
        
        self.param = param  # 原始参数
        self.stock_data = sd()  # 股票数据源
        self.calendar = sc()  # 交易日历
        self.result_file = param.get("result_file", None)  # 结果文件

    def _init_cache(self):
        """Initialize cache and return cached dataframes and executed keys."""
        cache = LocalCache()  # 本地缓存

        cached_a_df = cache.get_csv(f"a_{self.result_file}")
        if cached_a_df is None:
            cached_a_df = pd.DataFrame(columns=RESULT_COLS_A)
        
        cached_b_df = cache.get_csv(f"b_{self.result_file}")
        if cached_b_df is None:
            cached_b_df = pd.DataFrame(columns=RESULT_COLS_B)
        
        executed_keys = set(cached_a_df['配置'].tolist()) if '配置' in cached_a_df.columns else set()
        b_keys = set(cached_b_df['配置'].tolist()) if '配置' in cached_b_df.columns else set()
        executed_keys.update(b_keys)
        
        print("已缓存执行策略数:", len(executed_keys))
        
        temp_a_df = pd.DataFrame(columns=RESULT_COLS_A)
        temp_b_df = pd.DataFrame(columns=RESULT_COLS_B)
        
        return cache, cached_a_df, cached_b_df, executed_keys, temp_a_df, temp_b_df
    
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
        
        # 提取日期和资金值
        dates = [dv['date'] for dv in daily_values]
        values = [dv['value'] for dv in daily_values]
        
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

    def _save_cache(self, cache, cached_a_df, cached_b_df, temp_a_df, temp_b_df):
        """Save cache data to files."""
        if cached_a_df.empty:
            cached_a_df = temp_a_df.copy()
        else:
            for col in ['平均交易次数', '平均资金使用率']:
                if col not in cached_a_df.columns:
                    cached_a_df[col] = np.nan
            cached_a_df = pd.concat([cached_a_df, temp_a_df], ignore_index=True)

        # cached_a_df 按平均胜率、平均收益率 排序
        cached_a_df = cached_a_df.sort_values(by=['平均胜率', '平均收益率'], ascending=[False, False])
        cache.set_csv(f"a_{self.result_file}", cached_a_df)
        
        if cached_b_df.empty:
            cached_b_df = temp_b_df.copy()
        else:
            cached_b_df = pd.concat([cached_b_df, temp_b_df], ignore_index=True)
        cache.set_csv(f"b_{self.result_file}", cached_b_df)

    def execute(self) -> list:
        cache, cached_a_df, cached_b_df, executed_keys, temp_a_df, temp_b_df = self._init_cache()
        total_strategies = len(self.strategies)
        print(f"总策略数: {total_strategies}, 已缓存: {len(executed_keys)}")
        
        for idx, params in tqdm(enumerate(self.strategies), desc="执行策略", total=total_strategies):
            cache_key = "|".join(",".join(map(str, arr)) for arr in [[params.get('base_param_arr')[1]],params.get('buy_param_arr'),params.get('sell_param_arr')])
            if cache_key in executed_keys and not self.chain_debug:
                continue
            strategy = UpStrategy(**params)
            results = []
            total_periods = len(self.date_arr)
            max_failures_allowed = int(total_periods * (1 - self.win_rate_threshold))  # 计算允许的最大失败次数
            failure_count = 0
            successful_count = 0  # Track successful results during the loop to avoid duplicate calculation
            all_daily_values = []  # 保存所有日期范围的daily_values
            for s, e in self.date_arr:
                result = self.execute_one_strategy(strategy, s, e)
                # 保存当前日期范围的daily_values
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
                # 达到胜率阈值且完成了所有周期，写入a文件
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
                temp_a_df.loc[len(temp_a_df)] = new_row
            else:
                temp_b_df.loc[len(temp_b_df)] = {"配置": cache_key}
            if self.chain_debug:
                if 'new_row' in locals():
                    print(new_row)
                # 绘制资金变化图表
                self._draw_fund_trend(all_daily_values, f'策略资金变化趋势 - {cache_key}')
            executed_keys.add(cache_key)
            if idx % 100000 == 0:
                self._save_cache(cache, cached_a_df, cached_b_df, temp_a_df, temp_b_df)
                temp_a_df = pd.DataFrame(columns=RESULT_COLS_A)
                temp_b_df = pd.DataFrame(columns=RESULT_COLS_B)
        if not temp_a_df.empty or not temp_b_df.empty:
            self._save_cache(cache, cached_a_df, cached_b_df, temp_a_df, temp_b_df)


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
            print(f"资金: {result.初始资金:.2f} - > {result.最终资金:.2f}")
            print(f"总收益率: {result.总收益率:.2f}%")
            print(f"胜率: {result.胜率:.2f}%")
            print(f"交易次数: {result.交易次数}")
            print(f"最大资金: {result.最大资金:.2f}")
            print(f"最小资金: {result.最小资金:.2f}")
            print(f"夏普比率: {result.夏普比率:.2f}")
            print(f"平均资金使用率: {result.平均资金使用率:.2f}%")
            print("=" * 50)
        
        return result
