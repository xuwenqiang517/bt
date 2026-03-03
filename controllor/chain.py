import logger_config  # 导入日志配置，重定向stdout到log.txt
from typing import List,  Dict, Any

from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from tqdm import tqdm
from local_cache import LocalCache

from dto import *
from dto import ResultSchema
from strategy import Strategy

class Chain:
    def __init__(self, param=None):
        self.strategies = param.get("strategy")  # 策略列表（可能为None，使用生成器模式）
        self.date_arr = param["date_arr"]  # 回测时间周期列表
        self.chain_debug = param.get("chain_debug", False)  # 是否打印报告
        self.win_rate_threshold = param.get("win_rate_threshold", 0.99)  # 胜率阈值，默认65%
        self.processor_count = param.get("processor_count", 1)  # 进程数，默认1
        self.fail_count = param.get("fail_count", 1)  # 允许失败次数，默认1
        self.force_refresh = param.get("force_refresh", False)  # 是否强制刷新数据缓存
        self.sort_periods_by_difficulty = param.get("sort_periods_by_difficulty", True)  # 是否按难度排序周期

        self.param = param  # 原始参数
        self.result_file = param.get("result_file", None)  # 结果文件
        self.run_year = param.get("run_year", True)  # 是否跑年周期，默认True

        # 生成器模式相关参数
        self.use_param_generator = param.get("use_param_generator", False)
        self.param_generator = param.get("param_generator", None)
        self.total_strategy_count = param.get("total_strategy_count", 0)

        # 如果启用，按难度排序周期（熊市优先）
        if self.sort_periods_by_difficulty and self.date_arr:
            self.date_arr = self._sort_periods_by_difficulty()
            # 更新param，确保子进程使用相同的排序
            self.param["date_arr"] = self.date_arr

    def _sort_periods_by_difficulty(self) -> list:
        """按周期难度排序，低收益（熊市）周期优先，实现快速失败优化

        使用指数平均收益作为难度指标，避免预计算所有策略
        """
        stock_data = sd(force_refresh=self.force_refresh)
        calendar = sc()

        period_scores = []
        for start_date, end_date in self.date_arr:
            # 使用指数平均涨跌幅作为市场难度指标
            score = self._calc_period_difficulty(start_date, end_date, stock_data, calendar)
            period_scores.append((start_date, end_date, score))

        # 按难度排序（低收益/负收益周期优先）
        period_scores.sort(key=lambda x: x[2])

        sorted_arr = [(s, e) for s, e, _ in period_scores]

        # 打印排序结果
        print(f"\n周期按难度排序（熊市优先）:")
        for i, (s, e, score) in enumerate(period_scores[:5]):
            print(f"  {i+1}. {s}-{e}: 市场收益={score*100:.2f}%")
        if len(period_scores) > 5:
            print(f"  ... 共{len(period_scores)}个周期")
        print()

        return sorted_arr

    def _calc_period_difficulty(self, start_date: int, end_date: int, stock_data, calendar) -> float:
        """计算周期难度：使用等权平均涨跌幅作为市场基准

        Returns:
            float: 周期平均收益率，越低表示越难（熊市）
        """
        start_idx = calendar.start(start_date)
        end_idx = calendar.start(end_date)

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            return 0.0

        total_return = 0.0
        count = 0

        for idx in range(start_idx, end_idx + 1):
            date = calendar.get_date(idx)
            day_data = stock_data.get_numpy_data_by_date(date)
            if day_data is not None and len(day_data.code) > 0:
                # 使用当日平均涨跌幅（使用属性访问替代dict查找）
                avg_change = np.mean(day_data.change_pct)
                total_return += avg_change
                count += 1

        # 返回平均日收益，负值表示熊市
        return total_return / count if count > 0 else 0.0

    def _draw_trade_details(self, trades_history, daily_values, title, stock_data=None):
        """绘制交易明细图表，在资金曲线上直接标注完整交易信息"""
        if not trades_history or not daily_values:
            if self.chain_debug:
                print("没有交易数据，跳过交易明细图")
            return

        from pathlib import Path
        import os
        import glob

        # 计算保存路径
        data_dir = Path(__file__).resolve().parent.parent / "./data"
        os.makedirs(data_dir, exist_ok=True)

        # 清理旧的"交易明细_"前缀文件
        prefix = "交易明细_"
        for old_file in glob.glob(str(data_dir / f"{prefix}*.png")):
            try:
                os.remove(old_file)
            except OSError:
                pass

        # 生成文件名
        import re
        safe_title = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '_', title)
        safe_title = safe_title[:50]
        file_name = f"{safe_title}.png"
        save_path = data_dir / file_name

        # 准备数据
        dates = [f"{dv.date:08d}" for dv in daily_values]
        values = [dv.value / 100 for dv in daily_values]
        date_to_idx = {dv.date: i for i, dv in enumerate(daily_values)}

        # 创建数值索引用于画图（避免字符串日期导致的分类轴问题）
        x_indices = list(range(len(dates)))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表 - 单张大图显示所有信息
        fig, ax1 = plt.subplots(figsize=(42, 18))

        # ===== 主图：资金曲线 =====
        # 使用数值索引画图，避免字符串日期导致的分类轴问题
        ax1.plot(x_indices, values, label='总资产', linewidth=2, color='navy', alpha=0.8)

        # 收集已卖出的股票代码
        sold_codes = {trade.code for trade in trades_history}
        
        # 获取最后一天未卖出的持仓
        last_day_data = daily_values[-1] if daily_values else None
        unsold_holdings = []
        if last_day_data:
            last_date = last_day_data['date']
            last_holdings = last_day_data.get('holdings', [])
            for h in last_holdings:
                if h['code'] not in sold_codes:
                    unsold_holdings.append({
                        'code': h['code'],
                        'buy_price': h['buy_price'],
                        'current_price': h['close_price'],
                        'profit_rate': h['profit_rate'],
                        'buy_day': h.get('buy_day', last_date),  # 如果没有buy_day，用最后一天
                        'current_day': last_date
                    })
        
        # 标注所有已完成的交易
        for i, trade in enumerate(trades_history):
            sell_date = trade.date
            buy_date = trade.buy_date
            if sell_date not in date_to_idx or buy_date not in date_to_idx:
                continue

            sell_idx = date_to_idx[sell_date]
            buy_idx = date_to_idx[buy_date]
            x_sell = x_indices[sell_idx]
            y_sell = values[sell_idx]
            x_buy = x_indices[buy_idx]
            y_buy = values[buy_idx]

            code = trade.code
            buy_price = trade.buy_price / 100
            sell_price = trade.sell_price / 100
            profit_rate = trade.profit_rate * 100
            profit = trade.profit / 100
            reason = trade.get('reason', '')

            hold_days = sell_idx - buy_idx

            if stock_data:
                stock_name = stock_data.get_stock_name(code)
                name_short = stock_name[:4] if len(stock_name) > 4 else stock_name
            else:
                name_short = str(code)[-6:]

            # 简化卖出原因（区分止盈类型和到期盈亏）
            simple_reason = ''
            if '止损' in reason:
                simple_reason = '止损'
            elif '到期' in reason:
                if '盈利' in reason or profit > 0:
                    simple_reason = '到期盈利'
                else:
                    simple_reason = '到期亏损'
            elif '止盈' in reason or '回落' in reason:
                if '开盘回落' in reason:
                    simple_reason = '开盘回落止盈'
                elif '盘中回落' in reason:
                    simple_reason = '盘中回落止盈'
                else:
                    simple_reason = '回落止盈'

            trade_color = 'red' if profit > 0 else 'green'

            ax1.scatter([x_buy], [y_buy], color='blue', marker='>', s=100, zorder=5)
            ax1.scatter([x_sell], [y_sell], color='gold', marker='s', s=100, zorder=5)

            is_top = i % 2 == 0

            buy_date_str = str(buy_date)[4:]
            sell_date_str = str(sell_date)[4:]

            full_label = (
                f'{code} {name_short}\n'
                f'买:{buy_date_str} {buy_price:.2f}元\n'
                f'卖:{sell_date_str} {sell_price:.2f}元\n'
                f'{profit_rate:+.1f}% | {hold_days}天 | {simple_reason}'
            )

            mid_x = (x_buy + x_sell) / 2
            mid_y = (y_buy + y_sell) / 2

            trade_span = x_sell - x_buy
            base_offset = max(60, min(100, trade_span * 8))

            if is_top:
                y_offset = base_offset + (i % 3) * 35
                va_align = 'bottom'
            else:
                y_offset = -base_offset - (i % 3) * 35
                va_align = 'top'

            box_x = mid_x
            box_y = mid_y + y_offset * (max(values) - min(values)) / 800 if len(values) > 1 else mid_y + y_offset

            if is_top:
                rad_buy = 0.15
                rad_sell = -0.15
            else:
                rad_buy = -0.15
                rad_sell = 0.15

            ax1.annotate('', xy=(x_buy, y_buy), xytext=(box_x, box_y),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, connectionstyle=f'arc3,rad={rad_buy}'))

            ax1.annotate('', xy=(x_sell, y_sell), xytext=(box_x, box_y),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1.2, connectionstyle=f'arc3,rad={rad_sell}'))

            ax1.text(box_x, box_y, full_label, fontsize=7, color='black',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95,
                             edgecolor=trade_color, lw=2))
        
        # 标注未卖出的持仓
        unsold_start_idx = len(trades_history)
        for j, holding in enumerate(unsold_holdings):
            i = unsold_start_idx + j
            code = holding['code']
            buy_price = holding['buy_price'] / 100
            current_price = holding['current_price'] / 100
            profit_rate = holding['profit_rate'] * 100
            buy_day = holding['buy_day']
            current_day = holding['current_day']
            
            if buy_day not in date_to_idx or current_day not in date_to_idx:
                continue
                
            buy_idx = date_to_idx[buy_day]
            current_idx = date_to_idx[current_day]
            x_buy = x_indices[buy_idx]
            y_buy = values[buy_idx]
            x_current = x_indices[current_idx]
            y_current = values[current_idx]
            
            hold_days = current_idx - buy_idx
            
            if stock_data:
                stock_name = stock_data.get_stock_name(code)
                name_short = stock_name[:4] if len(stock_name) > 4 else stock_name
            else:
                name_short = str(code)[-6:]
            
            trade_color = 'red' if profit_rate > 0 else 'green'
            
            ax1.scatter([x_buy], [y_buy], color='blue', marker='>', s=100, zorder=5)
            ax1.scatter([x_current], [y_current], color='purple', marker='o', s=100, zorder=5)
            
            is_top = i % 2 == 0
            
            buy_date_str = str(buy_day)[4:]
            current_date_str = str(current_day)[4:]
            
            full_label = (
                f'{code} {name_short}\n'
                f'买:{buy_date_str} {buy_price:.2f}元\n'
                f'持:{current_date_str} {current_price:.2f}元\n'
                f'{profit_rate:+.1f}% | {hold_days}天 | 持仓中'
            )
            
            mid_x = (x_buy + x_current) / 2
            mid_y = (y_buy + y_current) / 2
            
            trade_span = x_current - x_buy
            base_offset = max(60, min(100, trade_span * 8))
            
            if is_top:
                y_offset = base_offset + (i % 3) * 35
            else:
                y_offset = -base_offset - (i % 3) * 35
            
            box_x = mid_x
            box_y = mid_y + y_offset * (max(values) - min(values)) / 800 if len(values) > 1 else mid_y + y_offset
            
            if is_top:
                rad_buy = 0.15
                rad_current = -0.15
            else:
                rad_buy = -0.15
                rad_current = 0.15
            
            ax1.annotate('', xy=(x_buy, y_buy), xytext=(box_x, box_y),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, connectionstyle=f'arc3,rad={rad_buy}'))
            
            ax1.annotate('', xy=(x_current, y_current), xytext=(box_x, box_y),
                        arrowprops=dict(arrowstyle='->', color='purple', lw=1.2, connectionstyle=f'arc3,rad={rad_current}'))
            
            ax1.text(box_x, box_y, full_label, fontsize=7, color='black',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.95,
                             edgecolor=trade_color, lw=2, linestyle='--'))

        ax1.set_ylabel('资金（元）', fontsize=12, color='navy')
        ax1.set_xlabel('日期', fontsize=12)
        ax1.set_title(f'{title} - 资金曲线与交易明细（共{len(trades_history)}笔交易）', fontsize=16)
        ax1.grid(True, alpha=0.3)
        ax1.ticklabel_format(style='plain', axis='y')

        # 调整日期标签 - 使用数值索引但显示日期字符串
        if len(dates) > 40:
            step = len(dates) // 40
            tick_indices = x_indices[::step]
            tick_labels = dates[::step]
            ax1.set_xticks(tick_indices)
            ax1.set_xticklabels(tick_labels)
        else:
            ax1.set_xticks(x_indices)
            ax1.set_xticklabels(dates)
        plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)

        # 添加图例说明
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='>', color='w', markerfacecolor='blue', markersize=10, label='买入点'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', markersize=10, label='卖出点'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='持仓中'),
            Line2D([0], [0], color='navy', lw=2, label='总资产')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.chain_debug:
            print(f"交易明细图表已保存到: {save_path}")

    @staticmethod
    def _process_strategy_group_worker(strategy_group: List[Dict[str, Any]], thread_id: int, param: dict) -> None:
        """子进程工作函数（静态方法，可序列化）- 用于非生成器模式的多进程"""
        chain = Chain(param=param)
        chain._process_strategy_group(strategy_group, thread_id)

    def _process_strategy_group(self, strategy_group: List[Dict[str, Any]], thread_id: int) -> None:
        """处理一组策略 - 批量累积优化，使用策略对象池复用"""
        cache = LocalCache()
        thread_result_file = f"{self.result_file}_thread_{thread_id}"
        stock_data = sd(force_refresh=self.force_refresh)
        calendar = sc()
        is_debug = self.chain_debug
        cache_filename = f"a_{thread_result_file}"
        cached_a_df = self._load_cache(cache, cache_filename)
        fail_count = self.fail_count
        total_count = len(self.date_arr)

        # 批量累积配置
        BATCH_SIZE = 50  # 每50个策略合并一次
        pending_rows = []
        count = 0

        for params in tqdm(strategy_group, desc=f"进程 {thread_id} 执行策略",
                          total=len(strategy_group), position=thread_id, leave=True, mininterval=1):
            count += 1
            strategy = Strategy(**params)
            results = []
            failure_count = 0
            all_daily_values = []

            for s, e in self.date_arr:
                result = self.execute_one_strategy(strategy, s, e, stock_data, calendar)
                if hasattr(strategy, 'daily_values') and strategy.daily_values:
                    all_daily_values.extend(strategy.daily_values)

                init_amount = params.get('base_param_arr')[0]
                max_drawdown_ok = result.期min >= init_amount * 0.8 if hasattr(result, '期min') else True

                if result.总收益率 <= 0 or not max_drawdown_ok:
                    failure_count += 1
                    continue
                results.append(result)

            if failure_count > fail_count and not is_debug:
                continue

            successful_count = total_count - failure_count
            actual_win_rate = successful_count / total_count if results else 0

            year_result = None
            pick_signal_stats = None
            if self.run_year:
                try:
                    year_result = self.execute_one_strategy(strategy, 20250101, 20260301, stock_data, calendar, calc_sharpe=True)
                    # 年周期回测后，统计选股信号表现
                    if hasattr(strategy, 'pick_signals') and strategy.pick_signals:
                        pick_signal_stats = self._calc_pick_signal_stats(
                            strategy.pick_signals, stock_data, calendar
                        )
                except Exception as e:
                    print(f"年周期执行失败: {e}")

            base_arr = params.get('base_param_arr')
            base_config = [base_arr[1]]
            cache_key = "|".join(",".join(map(str, arr)) for arr in 
                               [base_config, params.get('buy_param_arr'), params.get('pick_param_arr'), params.get('sell_param_arr')])
            new_row = self._create_new_row(actual_win_rate, successful_count, total_count, results, cache_key, year_result, pick_signal_stats)
            
            if is_debug:
                print(f"进程 {thread_id}: {new_row}")
            
            pending_rows.append(new_row)
            
            # 批量合并，减少 Polars concat 次数
            if len(pending_rows) >= BATCH_SIZE:
                cached_a_df = self._batch_merge_rows(cached_a_df, pending_rows)
                pending_rows = []
                self._save_cache(cache, cache_filename, cached_a_df)
        
        # 处理剩余数据
        if pending_rows:
            cached_a_df = self._batch_merge_rows(cached_a_df, pending_rows)
        
        cached_a_df = self._sort_data(cached_a_df)
        self._save_cache(cache, cache_filename, cached_a_df)

    def _batch_merge_rows(self, cached_a_df: pl.DataFrame, rows: list) -> pl.DataFrame:
        """批量合并多行数据到 DataFrame"""
        if not rows:
            return cached_a_df
        
        new_data_df = pl.DataFrame(rows)
        if cached_a_df.is_empty():
            return new_data_df
        
        # 一次性合并，避免多次 concat
        merged = pl.concat([cached_a_df, new_data_df], rechunk=True)
        return self._sort_data(merged)

    def _create_empty_a_df(self) -> pl.DataFrame:
        """创建空的a文件DataFrame - 使用ResultSchema集中定义"""
        return ResultSchema.create_empty_dataframe()
    
    def _load_cache(self, cache: LocalCache, filename: str) -> pl.DataFrame:
        """加载缓存，如果不存在则创建空的DataFrame"""
        df = cache.get_csv_pl(filename)
        return df if df is not None else self._create_empty_a_df()
    
    def _save_cache(self, cache: LocalCache, filename: str, df: pl.DataFrame) -> None:
        """保存DataFrame到缓存"""
        if not df.is_empty():
            cache.set_csv_pl(filename, df)
    
    def _ensure_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """确保DataFrame包含所有必需的列 - 使用ResultSchema集中定义"""
        return ResultSchema.ensure_columns(df)

    def _merge_and_sort_data(self, target_df: pl.DataFrame, new_df: pl.DataFrame) -> pl.DataFrame:
        """合并数据并排序"""
        # 确保所有必要的列存在（兼容旧缓存文件）
        target_df = self._ensure_columns(target_df)
        new_df = self._ensure_columns(new_df)
        
        # 合并数据
        merged_df = pl.concat([target_df, new_df], rechunk=True)
        # 调用 _sort_data 方法进行排序
        return self._sort_data(merged_df)

    def _sort_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """排序DataFrame，按年收益倒排（如果有该列）"""
        if df.is_empty():
            return df
        # 只有在有年收益列时才排序
        if '年收益' not in df.columns:
            return df
        # 将年收益转换为数值类型（处理字符串百分比格式）
        df = df.with_columns(
            pl.col('年收益').str.replace('%', '').cast(pl.Float64).alias('年收益数值')
        )
        # 按年收益数值降序排序
        df = df.sort(by='年收益数值', descending=True)
        # 删除临时列
        df = df.drop('年收益数值')
        return df

    def _calc_pick_signal_stats(self, pick_signals: list, stock_data, calendar) -> dict | None:
        """计算选股信号统计（1/3/5天后盈亏情况）

        计算逻辑：
        - 买入价：选股日的次日开盘价
        - 1日收益：次日收盘 - 次日开盘
        - 3日收益：3个交易日后的收盘 - 次日开盘
        - 5日收益：5个交易日后的收盘 - 次日开盘

        Args:
            pick_signals: 选股信号列表 [{'date': int, 'code': int, 'next_open': int}, ...]
            stock_data: 股票数据对象
            calendar: 交易日历对象

        Returns:
            dict | None: 统计结果字典，如果没有有效数据则返回None
        """
        if not pick_signals:
            return None

        stats = {1: {'profits': []}, 3: {'profits': []}, 5: {'profits': []}}

        for signal in pick_signals:
            date = signal['date']
            code = signal['code']

            # 找到次日（买入日）的索引
            current_idx = calendar.start(date)
            if current_idx == -1:
                continue

            next_idx = current_idx + 1  # 直接+1，下一个交易日索引

            # 获取次日的开盘价（买入价）
            buy_date = calendar.get_date(next_idx)
            # 获取股票数据，idx为-1表示无数据
            buy_day_data, buy_idx = stock_data.get_data_by_date_code(buy_date, code)
            if buy_idx == -1:
                continue
            # 从numpy数组取open价格（使用属性访问替代dict查找）
            buy_price = buy_day_data.open[buy_idx]

            # 获取未来1/3/5个交易日的收盘价
            for days in [1, 3, 5]:
                # 找到N个交易日后的日期（从次日开始算）
                target_idx = next_idx + days - 1  # 直接计算索引，避免循环调用next

                sell_date = calendar.get_date(target_idx)

                # 获取卖出日期的收盘价，idx为-1表示无数据
                sell_day_data, sell_idx = stock_data.get_data_by_date_code(sell_date, code)
                if sell_idx == -1:
                    continue

                # 从numpy数组取close价格（使用属性访问替代dict查找）
                sell_price = sell_day_data.close[sell_idx]
                profit_rate = (sell_price - buy_price) / buy_price if buy_price > 0 else 0
                stats[days]['profits'].append(profit_rate)

        # 检查是否有有效数据（至少有一天有收益数据）
        has_valid_data = any(stats[days]['profits'] for days in [1, 3, 5])
        if not has_valid_data:
            return None

        result = {'选股信号数': len(pick_signals)}

        for days in [1, 3, 5]:
            profits = stats[days]['profits']
            if not profits:
                result[f'{days}日胜率'] = ''
                result[f'{days}日盈亏比'] = ''
                result[f'{days}日平均收益'] = ''
                continue

            win_count = sum(1 for p in profits if p > 0)
            loss_count = len(profits) - win_count
            win_rate = win_count / len(profits) if profits else 0

            avg_profit = sum(profits) / len(profits) if profits else 0

            # 计算盈亏比
            win_profits = [p for p in profits if p > 0]
            loss_profits = [abs(p) for p in profits if p < 0]
            avg_win = sum(win_profits) / len(win_profits) if win_profits else 0
            avg_loss = sum(loss_profits) / len(loss_profits) if loss_profits else 0
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0)

            result[f'{days}日胜率'] = f"{win_rate*100:.1f}%"
            result[f'{days}日盈亏比'] = f"{profit_loss_ratio:.2f}" if profit_loss_ratio != float('inf') else "∞"
            result[f'{days}日平均收益'] = f"{avg_profit*100:.2f}%"

        return result

    def _create_new_row(self, actual_win_rate: float, successful_count: int, total_periods: int,
                        results: list, cache_key: str, year_result: object = None,
                        pick_signal_stats: dict = None) -> dict:
        """创建新的行数据 - 使用ResultSchema集中定义"""
        row = ResultSchema.create_chain_row_from_results(
            actual_win_rate, successful_count, total_periods, results, cache_key, year_result
        )
        # 合并选股信号统计
        if pick_signal_stats:
            row.update(pick_signal_stats)
        return row
    
    def _merge_thread_caches(self) -> None:
        """合并所有进程的缓存文件"""
        cache = LocalCache()
        
        # 加载主缓存
        main_cache_filename = f"a_{self.result_file}"
        main_a_df = self._load_cache(cache, main_cache_filename)
        
        # 合并所有进程的缓存
        for thread_id in range(self.processor_count):
            thread_result_file = f"{self.result_file}_thread_{thread_id}"
            thread_cache_filename = f"a_{thread_result_file}"
            
            # 加载进程缓存
            thread_a_df = cache.get_csv_pl(thread_cache_filename)
            if thread_a_df is not None and not thread_a_df.is_empty():
                # 确保列一致：使用主缓存的列，缺失的列按主缓存类型填空值
                for col in main_a_df.columns:
                    if col not in thread_a_df.columns:
                        # 根据主缓存列的类型选择默认值
                        col_dtype = main_a_df[col].dtype
                        if col_dtype in (pl.Float64, pl.Float32):
                            thread_a_df = thread_a_df.with_columns(pl.lit(0.0).alias(col))
                        elif col_dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8):
                            thread_a_df = thread_a_df.with_columns(pl.lit(0).cast(col_dtype).alias(col))
                        else:
                            thread_a_df = thread_a_df.with_columns(pl.lit("").alias(col))
                # 统一所有数值列为相同类型
                for col in main_a_df.columns:
                    if col in thread_a_df.columns:
                        main_dtype = main_a_df[col].dtype
                        thread_dtype = thread_a_df[col].dtype
                        if main_dtype != thread_dtype:
                            thread_a_df = thread_a_df.with_columns(pl.col(col).cast(main_dtype).alias(col))
                # 按主进程的列顺序重新排列
                thread_a_df = thread_a_df.select(main_a_df.columns)
                main_a_df = pl.concat([main_a_df, thread_a_df], rechunk=True)
                # 删除已合并的进程缓存文件
                cache.delete_file(f"{thread_cache_filename}.csv")
        
        # 去重并排序
        if not main_a_df.is_empty() and not self.chain_debug:
            main_a_df = main_a_df.unique(subset=['配置'])
            # 按年收益排序（处理字符串百分比格式），只有在有该列时才排序
            if '年收益' in main_a_df.columns:
                main_a_df = main_a_df.with_columns(
                    pl.col('年收益').str.replace('%', '').cast(pl.Float64).alias('年收益数值')
                )
                main_a_df = main_a_df.sort(by='年收益数值', descending=True)
                main_a_df = main_a_df.drop('年收益数值')
            # 保存到缓存
            self._save_cache(cache, main_cache_filename, main_a_df)
        
        # print(f"合并完成，主文件 a_{main_cache_filename}.csv 包含 {len(main_a_df)} 条记录")

    def execute(self) -> list:
        # 先合并所有进程的缓存，确保中断后再运行时能正确加载之前的结果
        self._merge_thread_caches()
        

        
        total_strategies = len(self.strategies)
        print(f"总策略数: {len(self.strategies)}, 已执行: {len(self.strategies) - total_strategies}, 剩余: {total_strategies}")
        print(f"使用进程数: {self.processor_count}")
        
        if total_strategies == 0:
            print("所有策略已执行完毕，无需处理")
            return
        
        # 策略分组
        group_size = total_strategies // self.processor_count
        strategy_groups = []
        for i in range(self.processor_count):
            start_idx = i * group_size
            end_idx = (i + 1) * group_size if i < self.processor_count - 1 else total_strategies
            strategy_groups.append(self.strategies[start_idx:end_idx])
            print(f"进程 {i} 处理策略数: {len(self.strategies[start_idx:end_idx])}")
        
        # 处理策略组
        import multiprocessing
        print(f"处理策略，进程数: {self.processor_count}")
        processes = []
        
        if self.processor_count == 1:
            # 单进程时直接在当前进程执行
            print(f"单进程模式，直接在当前进程执行")
            # 直接调用处理方法
            self._process_strategy_group(strategy_groups[0], 0)
        else:
            # 多进程时，主进程处理第一组，其余由子进程处理
            print(f"多进程模式，主进程处理一组，创建 {self.processor_count - 1} 个子进程")

            # 准备子进程参数（使用已排序的date_arr）
            worker_param = self.param.copy()
            worker_param["sort_periods_by_difficulty"] = False  # 子进程不重新排序

            # 创建并启动子进程处理剩余组
            for i in range(1, self.processor_count):
                group = strategy_groups[i]
                process_name = f"ChainProcess-{i}"
                process = multiprocessing.Process(
                    target=self._process_strategy_group_worker,
                    args=(group, i, worker_param),
                    name=process_name
                )
                processes.append(process)
                process.start()
                print(f"启动进程: {process_name}，处理策略数: {len(group)}")
        # 主进程处理第一组
            self._process_strategy_group(strategy_groups[0], 0)
        # 等待所有子进程完成
        for i, process in enumerate[Any](processes):
            process.join()
            print(f"进程 {i+1} ({process.name}) 处理完成")
        
        # 所有策略执行完成后，合并所有进程的缓存
        self._merge_thread_caches()

        return

    def execute_one_strategy(self, strategy, start_date, end_date, stock_data, calendar, calc_sharpe: bool = False) -> BacktestResult:
        """执行单个策略
        Args:
            strategy: 策略实例
            start_date: 起始日期
            end_date: 结束日期
            stock_data: 股票数据
            calendar: 交易日历
            calc_sharpe: 是否计算夏普比率（仅年周期需要）
        """
        scalendar = calendar
        current_idx = scalendar.start(start_date)
        end_idx = scalendar.start(end_date)

        # 日期不符合直接抛异常
        if current_idx == -1 or end_idx == -1:
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")

        strategy.bind(stock_data, calendar)
        strategy.reset()

        while current_idx != -1 and current_idx <= end_idx:
            current_date = scalendar.get_date(current_idx)
            strategy.update_today(current_date, current_idx)  # 传入日期索引，避免重复dict查找
            # 固定先卖后买
            strategy.sell()
            strategy.buy()
            strategy.pick()
            strategy.settle_amount()
            current_idx += 1  # 直接+1，下一个交易日索引

        result = strategy.calculate_performance(start_date, end_date, calc_sharpe)

        if self.chain_debug:
            print("=" * 50)
            print(f"时间周期: {result.起始日期} 至 {result.结束日期}")
            print(f"资金: {result.初始资金/100:.2f} - > {result.最终资金/100:.2f}")
            print(f"总收益率: {result.总收益率*100:.2f}%")
            print(f"胜率: {result.胜率*100:.2f}%")
            print(f"交易次数: {result.交易次数}")
            print(f"最大资金: {result.期max/100:.2f}")
            print(f"最小资金: {result.期min/100:.2f}")
            print(f"夏普比率: {result.夏普比率:.2f}")
            print(f"平均资金使用率: {result.平均资金使用率*100:.2f}%")
            print("=" * 50)
            # 绘制交易明细图表
            self._draw_trade_details(strategy.trades_history, strategy.daily_values, f"交易明细_{start_date}_{end_date}", stock_data)

        return result

    def execute_generator_mode(self) -> list:
        """生成器模式执行 - 动态生成参数，避免内存爆炸"""
        import multiprocessing

        # 先合并所有进程的缓存
        self._merge_thread_caches()

        total_strategies = self.total_strategy_count
        print(f"总策略数: {total_strategies}")
        print(f"使用进程数: {self.processor_count}")

        if total_strategies == 0:
            print("所有策略已执行完毕，无需处理")
            return

        # 计算每个进程的任务范围
        processes = []

        if self.processor_count == 1:
            # 单进程时直接在当前进程执行
            print(f"单进程模式，直接在当前进程执行")
            self._process_strategy_generator(0, 0, total_strategies)
        else:
            # 多进程时，主进程处理第一组，其余由子进程处理
            print(f"多进程模式，主进程处理一组，创建 {self.processor_count - 1} 个子进程")

            # 基于限制后的总策略数计算每个进程的任务量
            base_size = total_strategies // self.processor_count
            remainder = total_strategies % self.processor_count

            # 创建并启动子进程处理剩余组
            for i in range(1, self.processor_count):
                if i < remainder:
                    start_idx = i * (base_size + 1)
                    end_idx = start_idx + base_size + 1
                else:
                    start_idx = remainder * (base_size + 1) + (i - remainder) * base_size
                    end_idx = start_idx + base_size
                process_name = f"ChainProcess-{i}"
                process = multiprocessing.Process(
                    target=self._process_strategy_generator_worker,
                    args=(i, start_idx, end_idx, self.param, self.result_file, total_strategies),
                    name=process_name
                )
                processes.append(process)
                process.start()
                print(f"启动进程: {process_name}，处理策略数: {end_idx - start_idx}")

            # 主进程处理第一组
            if remainder > 0:
                start_idx, end_idx = 0, base_size + 1
            else:
                start_idx, end_idx = 0, base_size
            self._process_strategy_generator(0, start_idx, end_idx)

            # 等待所有子进程完成
            for i, process in enumerate(processes):
                process.join()
                print(f"进程 {i+1} ({process.name}) 处理完成")

        # 所有策略执行完成后，合并所有进程的缓存
        self._merge_thread_caches()

        return

    def _process_strategy_generator(self, thread_id: int, start_idx: int, end_idx: int) -> None:
        """处理指定索引范围的策略（生成器模式），使用策略对象池复用"""
        # 为每个进程创建独立的缓存
        cache = LocalCache()
        thread_result_file = f"{self.result_file}_thread_{thread_id}"

        # 在子进程中初始化 stock_data 和 calendar
        stock_data = sd(force_refresh=self.force_refresh)
        calendar = sc()
        is_debug = self.chain_debug

        # 加载进程本地缓存
        cache_filename = f"a_{thread_result_file}"
        cached_a_df = self._load_cache(cache, cache_filename)
        fail_count = self.fail_count

        # 计算固定值
        total_count = len(self.date_arr)
        count = 0

        # 动态生成参数并处理
        param_count = end_idx - start_idx
        for params in tqdm(self.param_generator.get_slice_params(start_idx, end_idx),
                          desc=f"进程 {thread_id} 执行策略", total=param_count, position=thread_id, leave=True, mininterval=1):
            count += 1
            strategy = Strategy(**params)
            results = []
            failure_count = 0
            all_daily_values = []

            for s, e in self.date_arr:
                result = self.execute_one_strategy(strategy, s, e, stock_data, calendar)
                if hasattr(strategy, 'daily_values') and strategy.daily_values:
                    all_daily_values.extend(strategy.daily_values)

                # 过滤条件：收益率为负 或 最大回撤超过20%（最小资金<初始资金80%）
                init_amount = params.get('base_param_arr')[0]
                max_drawdown_ok = result.期min >= init_amount * 0.8 if hasattr(result, '期min') else True

                if result.总收益率 <= 0 or not max_drawdown_ok:
                    failure_count += 1
                    if failure_count > fail_count and not is_debug:
                        break
                    continue
                results.append(result)

            if failure_count > fail_count and not is_debug:
                continue

            successful_count = total_count - failure_count
            actual_win_rate = successful_count / total_count if results else 0

            # 跑年连续周期 20250101-20260101（根据run_year参数决定是否执行）
            year_result = None
            pick_signal_stats = None
            if self.run_year:
                try:
                    year_result = self.execute_one_strategy(strategy, 20250101, 20260101, stock_data, calendar, calc_sharpe=True)
                    # 年周期回测后，统计选股信号表现
                    if hasattr(strategy, 'pick_signals') and strategy.pick_signals:
                        pick_signal_stats = self._calc_pick_signal_stats(
                            strategy.pick_signals, stock_data, calendar
                        )
                except Exception as e:
                    print(f"年周期执行失败: {e}")

            # 构建配置字符串：只保留可调整参数（持仓数量|买入参数|排序参数|卖出参数）
            base_arr = params.get('base_param_arr')
            # 只保留持仓数量（动态仓位，去掉仓位比例）
            base_config = [base_arr[1]]
            cache_key = "|".join(",".join(map(str, arr)) for arr in [base_config, params.get('buy_param_arr'), params.get('pick_param_arr'), params.get('sell_param_arr')])
            new_row = self._create_new_row(actual_win_rate, successful_count, total_count, results, cache_key, year_result, pick_signal_stats)
            print(new_row)

            if is_debug:
                if 'new_row' in locals():
                    print(f"进程 {thread_id}: {new_row}")

            # 直接将new_row转换为DataFrame并处理
            new_data_df = pl.DataFrame([new_row])
            if cached_a_df.is_empty():
                cached_a_df = new_data_df
            else:
                cached_a_df = self._merge_and_sort_data(cached_a_df, new_data_df)

            if count % 5 == 0:
                cached_a_df = self._sort_data(cached_a_df)
                self._save_cache(cache, cache_filename, cached_a_df)

        cached_a_df = self._sort_data(cached_a_df)
        self._save_cache(cache, cache_filename, cached_a_df)

    @staticmethod
    def _process_strategy_generator_worker(thread_id: int, start_idx: int, end_idx: int, param: dict, result_file: str, total_strategy_count: int = None) -> None:
        """子进程工作函数（静态方法，可序列化）"""
        from param_generator import ParamGenerator

        # 重新创建生成器和Chain实例
        gen = ParamGenerator()
        # 子进程不重新排序周期，使用主进程已排序的date_arr
        param["sort_periods_by_difficulty"] = False
        chain = Chain(param=param)
        chain.param_generator = gen
        chain.result_file = result_file
        # 使用主进程传递的限制后策略数量
        if total_strategy_count is not None:
            chain.total_strategy_count = total_strategy_count

        chain._process_strategy_generator(thread_id, start_idx, end_idx)


