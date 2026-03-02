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
from strategy import Strategy

# 导入日志配置
import logger_config

class Chain:
    def __init__(self, param=None):
        self.strategies = param.get("strategy")  # 策略列表（可能为None，使用生成器模式）
        self.date_arr = param["date_arr"]  # 回测时间周期列表
        self.chain_debug = param.get("chain_debug", False)  # 是否打印报告
        self.win_rate_threshold = param.get("win_rate_threshold", 0.99)  # 胜率阈值，默认65%
        self.processor_count = param.get("processor_count", 1)  # 进程数，默认1
        self.fail_count = param.get("fail_count", 1)  # 允许失败次数，默认1
        self.force_refresh = param.get("force_refresh", False)  # 是否强制刷新数据缓存

        self.param = param  # 原始参数
        self.result_file = param.get("result_file", None)  # 结果文件
        self.run_year = param.get("run_year", True)  # 是否跑年周期，默认True

        # 生成器模式相关参数
        self.use_param_generator = param.get("use_param_generator", False)
        self.param_generator = param.get("param_generator", None)
        self.total_strategy_count = param.get("total_strategy_count", 0)

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
        file_name = f"{safe_title}.png"
        save_path = data_dir / file_name
        
        # 提取日期和资金值（将分转换为元）
        dates = [f"{dv['date']:08d}" for dv in daily_values]
        values = [dv['value'] / 100 for dv in daily_values]
        
        # 调试信息
        # if self.chain_debug:
        #     print(f"绘图数据: 日期数量={len(dates)}, 资金数量={len(values)}")
        #     if dates:
        #         print(f"起始日期: {dates[0]}, 结束日期: {dates[-1]}")
        #     if values:
        #         print(f"起始资金: {values[0]:.2f}, 结束资金: {values[-1]:.2f}")
        #         print(f"最大资金: {max(values):.2f}, 最小资金: {min(values):.2f}")
        
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
        # plt.show()
        # 保存图表
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 关闭图表，释放内存
        
        if self.chain_debug:
            print(f"资金变化图表已保存到: {save_path}")
            print(f"图表保存成功，文件大小: {os.path.getsize(save_path) if os.path.exists(save_path) else 0} bytes")

    def _draw_trade_details(self, trades_history, daily_values, title, stock_data=None):
        """绘制交易明细图表，在资金曲线上直接标注完整交易信息"""
        if not trades_history or not daily_values:
            if self.chain_debug:
                print("没有交易数据，跳过交易明细图")
            return

        from pathlib import Path
        import os

        # 计算保存路径
        data_dir = Path(__file__).resolve().parent.parent / "./data"
        os.makedirs(data_dir, exist_ok=True)

        # 生成文件名
        import re
        safe_title = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '_', title)
        safe_title = safe_title[:50]
        file_name = f"{safe_title}.png"
        save_path = data_dir / file_name

        # 准备数据
        dates = [f"{dv['date']:08d}" for dv in daily_values]
        values = [dv['value'] / 100 for dv in daily_values]
        date_to_idx = {dv['date']: i for i, dv in enumerate(daily_values)}

        # 调试：打印1117-1119期间的数据
        if self.chain_debug:
            print("\n=== 调试：daily_values 数据 ===")
            for dv in daily_values:
                d = dv['date']
                if 20251115 <= d <= 20251122:
                    holdings = dv.get('holdings', [])
                    hold_info = ', '.join([f"{h['code']}({h['profit_rate']:.2%})" for h in holdings]) if holdings else '空仓'
                    print(f"日期 {d}: 总资产 {dv['value']/100:.2f}, 持仓: {hold_info}")

        # 创建数值索引用于画图（避免字符串日期导致的分类轴问题）
        x_indices = list(range(len(dates)))
        idx_to_date = {i: dates[i] for i in range(len(dates))}

        # 收集每只股票的持仓期间盈亏率
        stock_holdings = {}
        for dv in daily_values:
            date = dv['date']
            if date not in date_to_idx:
                continue
            date_idx = date_to_idx[date]
            holdings = dv.get('holdings', [])
            for h in holdings:
                code = h['code']
                profit_rate = h['profit_rate'] * 100
                if code not in stock_holdings:
                    stock_holdings[code] = []
                stock_holdings[code].append((date_idx, profit_rate))

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']
        plt.rcParams['axes.unicode_minus'] = False

        # 创建图表 - 单张大图显示所有信息
        fig, ax1 = plt.subplots(figsize=(42, 18))

        # ===== 主图：资金曲线 =====
        # 使用数值索引画图，避免字符串日期导致的分类轴问题
        ax1.plot(x_indices, values, label='总资产', linewidth=2, color='navy', alpha=0.8)

        # 标注所有交易（不跳过任何交易）
        for i, trade in enumerate(trades_history):
            sell_date = trade['date']
            buy_date = trade['buy_date']
            if sell_date not in date_to_idx or buy_date not in date_to_idx:
                continue

            sell_idx = date_to_idx[sell_date]
            buy_idx = date_to_idx[buy_date]
            x_sell = x_indices[sell_idx]  # 使用数值索引
            y_sell = values[sell_idx]
            x_buy = x_indices[buy_idx]    # 使用数值索引
            y_buy = values[buy_idx]

            code = trade['code']
            buy_price = trade['buy_price'] / 100
            sell_price = trade['sell_price'] / 100
            profit_rate = trade['profit_rate'] * 100
            profit = trade['profit'] / 100
            reason = trade.get('reason', '')

            # 计算持仓天数
            hold_days = sell_idx - buy_idx

            # 获取股票名称
            if stock_data:
                stock_name = stock_data.get_stock_name(code)
                name_short = stock_name[:4] if len(stock_name) > 4 else stock_name
            else:
                name_short = str(code)[-6:]

            # 简化卖出原因（区分止盈类型）
            simple_reason = ''
            if '止损' in reason:
                simple_reason = '止损'
            elif '到期' in reason:
                if '盈利' in reason:
                    simple_reason = '到期盈'
                else:
                    simple_reason = '到期亏'
            elif '止盈' in reason or '回落' in reason:
                # 区分开盘回落和盘中回落
                if '开盘回落' in reason:
                    simple_reason = '开盘回落止盈'
                elif '盘中回落' in reason:
                    simple_reason = '盘中回落止盈'
                else:
                    simple_reason = '回落止盈'

            # 根据盈亏选择颜色（A股习惯：涨红跌绿）
            trade_color = 'red' if profit > 0 else 'green'

            # 标注买入点（蓝色）和卖出点（黄色统一）
            ax1.scatter([x_buy], [y_buy], color='blue', marker='>', s=100, zorder=5)
            ax1.scatter([x_sell], [y_sell], color='gold', marker='s', s=100, zorder=5)

            # 计算标注位置：上下交替
            is_top = i % 2 == 0  # 偶数交易在上方，奇数在下方

            # 构建多行标注框内容
            buy_date_str = str(buy_date)[4:]
            sell_date_str = str(sell_date)[4:]

            # 四行格式：
            # 第一行：股票代码+股票名称
            # 第二行：买入时间+买入价格
            # 第三行：卖出时间+卖出价格
            # 第四行：收益率+持仓天数+卖出原因
            full_label = (
                f'{code} {name_short}\n'
                f'买:{buy_date_str} {buy_price:.2f}元\n'
                f'卖:{sell_date_str} {sell_price:.2f}元\n'
                f'{profit_rate:+.1f}% | {hold_days}天 | {simple_reason}'
            )

            # 计算标注框位置（在买入卖出中间上方或下方）
            mid_x = (x_buy + x_sell) / 2  # 买入卖出日期的中间
            mid_y = (y_buy + y_sell) / 2  # 买入卖出资金的中间

            # 动态计算垂直偏移，避免框重叠
            # 根据交易密度调整间距
            trade_span = x_sell - x_buy  # 交易跨度（天数）
            base_offset = max(60, min(100, trade_span * 8))  # 基础偏移：60-100像素

            if is_top:
                # 标注在上方，逐层递增
                y_offset = base_offset + (i % 3) * 35  # 3层循环
                va_align = 'bottom'
            else:
                # 标注在下方，逐层递增
                y_offset = -base_offset - (i % 3) * 35
                va_align = 'top'

            # 标注框的绝对位置
            box_x = mid_x
            box_y = mid_y + y_offset * (max(values) - min(values)) / 800 if len(values) > 1 else mid_y + y_offset

            # 根据上下位置调整弧线方向，避免交叉
            if is_top:
                rad_buy = 0.15
                rad_sell = -0.15
            else:
                rad_buy = -0.15
                rad_sell = 0.15

            # 先画从标注框到买入点的箭头（蓝色）
            ax1.annotate('', xy=(x_buy, y_buy), xytext=(box_x, box_y),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, connectionstyle=f'arc3,rad={rad_buy}'))

            # 再画从标注框到卖出点的箭头（黄色统一）
            ax1.annotate('', xy=(x_sell, y_sell), xytext=(box_x, box_y),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=1.2, connectionstyle=f'arc3,rad={rad_sell}'))

            # 最后画标注框
            ax1.text(box_x, box_y, full_label, fontsize=7, color='black',
                    ha='center', va='center',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95,
                             edgecolor=trade_color, lw=2))

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
            Line2D([0], [0], marker='s', color='w', markerfacecolor='red', markersize=10, label='卖出点(盈利)'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor='green', markersize=10, label='卖出点(亏损)'),
            Line2D([0], [0], color='navy', lw=2, label='总资产')
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

        if self.chain_debug:
            print(f"交易明细图表已保存到: {save_path}")

    def _process_strategy_group(self, strategy_group: List[Dict[str, Any]], thread_id: int) -> None:
        """处理一组策略 - 批量累积优化"""
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
                max_drawdown_ok = result.最小资金 >= init_amount * 0.8 if hasattr(result, '最小资金') else True

                if result.总收益率 <= 0 or not max_drawdown_ok:
                    failure_count += 1
                    continue
                results.append(result)

            if failure_count > fail_count and not is_debug:
                continue

            successful_count = total_count - failure_count
            actual_win_rate = successful_count / total_count if results else 0

            year_result = None
            if self.run_year:
                try:
                    year_result = self.execute_one_strategy(strategy, 20250101, 20260301, stock_data, calendar)
                except Exception as e:
                    print(f"年周期执行失败: {e}")

            base_arr = params.get('base_param_arr')
            base_config = [base_arr[1]]
            cache_key = "|".join(",".join(map(str, arr)) for arr in 
                               [base_config, params.get('buy_param_arr'), params.get('pick_param_arr'), params.get('sell_param_arr')])
            new_row = self._create_new_row(actual_win_rate, successful_count, total_count, results, cache_key, year_result)
            
            if is_debug:
                print(f"进程 {thread_id}: {new_row}")
                self._draw_fund_trend(all_daily_values, f'资金趋势 - {cache_key}')
            
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
        """创建空的a文件DataFrame - 包含所有列确保类型一致"""
        return pl.DataFrame({
            '周期胜率': pl.Series([], dtype=pl.String),
            '平均胜率': pl.Series([], dtype=pl.String),
            '平均收益率': pl.Series([], dtype=pl.String),
            '平均交易次数': pl.Series([], dtype=pl.Float64),
            '最大资金': pl.Series([], dtype=pl.Float64),
            '最小资金': pl.Series([], dtype=pl.Float64),
            '夏普比率': pl.Series([], dtype=pl.Float64),
            '平均资金使用率': pl.Series([], dtype=pl.String),
            '年周期收益率': pl.Series([], dtype=pl.String),
            '年周期胜率': pl.Series([], dtype=pl.String),
            '年周期夏普': pl.Series([], dtype=pl.Float64),
            '年最大资金': pl.Series([], dtype=pl.Float64),
            '年最小资金': pl.Series([], dtype=pl.Float64),
            '年交易次数': pl.Series([], dtype=pl.Float64),
            '止损次数': pl.Series([], dtype=pl.Float64),
            '到期盈利': pl.Series([], dtype=pl.Float64),
            '到期亏损': pl.Series([], dtype=pl.Float64),
            '回落止盈': pl.Series([], dtype=pl.Float64),
            '配置': pl.Series([], dtype=pl.String)
        })
    
    def _load_cache(self, cache: LocalCache, filename: str) -> pl.DataFrame:
        """加载缓存，如果不存在则创建空的DataFrame"""
        df = cache.get_csv_pl(filename)
        return df if df is not None else self._create_empty_a_df()
    
    def _save_cache(self, cache: LocalCache, filename: str, df: pl.DataFrame) -> None:
        """保存DataFrame到缓存"""
        if not df.is_empty():
            cache.set_csv_pl(filename, df)
    
    def _ensure_columns(self, df: pl.DataFrame) -> pl.DataFrame:
        """确保DataFrame包含所有必需的列，兼容旧缓存文件"""
        required_cols = {
            '平均交易次数': pl.Float64,
            '平均资金使用率': pl.String,
            '年周期收益率': pl.String,
            '年周期胜率': pl.String,
            '年周期夏普': pl.Float64,
            '年最大资金': pl.Float64,
            '年最小资金': pl.Float64,
            '年交易次数': pl.Float64,
            '止损次数': pl.Float64,
            '到期盈利': pl.Float64,
            '到期亏损': pl.Float64,
            '回落止盈': pl.Float64
        }
        for col, dtype in required_cols.items():
            if col not in df.columns:
                if dtype == pl.String:
                    df = df.with_columns(pl.lit("").alias(col))
                else:
                    df = df.with_columns(pl.lit(0.0).alias(col))
        return df

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
        """排序DataFrame，按年周期收益率倒排"""
        if df.is_empty():
            return df
        # 将年周期收益率转换为数值类型（处理字符串百分比格式）
        df = df.with_columns(
            pl.col('年周期收益率').str.replace('%', '').cast(pl.Float64).alias('年周期收益率数值')
        )
        # 按年周期收益率数值降序排序
        df = df.sort(by='年周期收益率数值', descending=True)
        # 删除临时列
        df = df.drop('年周期收益率数值')
        return df
    
    def _create_new_row(self, actual_win_rate: float, successful_count: int, total_periods: int, results: list, cache_key: str, year_result: object = None) -> dict:
        """创建新的行数据"""
        if not results:
            # 当results为空时，返回默认值
            return {
                "周期胜率": f"{int(actual_win_rate * 100)}%({successful_count}/{total_periods})",
                "平均胜率": "0%",
                "平均收益率": "0.00%",
                "平均交易次数": 0.0,
                "最大资金": 0.0,
                "最小资金": 0.0,
                "夏普比率": 0.0,
                "平均资金使用率": "0.00%",
                "年周期收益率": "0.00%",
                "年周期胜率": "0%",
                "年周期夏普": 0.0,
                "年最大资金": 0.0,
                "年最小资金": 0.0,
                "年交易次数": 0.0,
                "止损次数": 0.0,
                "到期盈利": 0.0,
                "到期亏损": 0.0,
                "回落止盈": 0.0,
                "配置": cache_key
            }

        # 当results不为空时，计算各项指标
        base_result = {
            "周期胜率": f"{int(actual_win_rate * 100)}%({successful_count}/{total_periods})",
            "平均胜率": f"{int(np.mean([x.胜率 for x in results]) * 100)}%",
            "平均收益率": f"{float(np.mean([x.总收益率 for x in results])) * 100:.2f}%",
            "平均交易次数": round(float(np.mean([x.交易次数 for x in results])), 1),
            "最大资金": float(max([x.最大资金 for x in results])),
            "最小资金": float(min([x.最小资金 for x in results])),
            "夏普比率": round(float(np.mean([x.夏普比率 for x in results])), 2),
            "平均资金使用率": f"{float(np.mean([x.平均资金使用率 for x in results])) * 100:.2f}%",
            "配置": cache_key
        }

        # 年周期统计（如果有的话）
        if year_result:
            base_result.update({
                "年周期收益率": f"{float(year_result.总收益率) * 100:.2f}%",
                "年周期胜率": f"{int(year_result.胜率 * 100)}%",
                "年周期夏普": round(float(year_result.夏普比率), 2),
                "年最大资金": float(year_result.最大资金),
                "年最小资金": float(year_result.最小资金),
                "年交易次数": float(year_result.交易次数)
            })
        else:
            base_result.update({
                "年周期收益率": "0.00%",
                "年周期胜率": "0%",
                "年周期夏普": 0.0,
                "年最大资金": 0.0,
                "年最小资金": 0.0,
                "年交易次数": 0.0
            })

        # 卖出原因统计（只在年周期回测时有）
        if year_result and hasattr(year_result, '卖出统计'):
            sell_stats = year_result.卖出统计
            base_result.update({
                "止损次数": float(sell_stats.get('止损', 0)),
                "到期盈利": float(sell_stats.get('到期盈利', 0)),
                "到期亏损": float(sell_stats.get('到期亏损', 0)),
                "回落止盈": float(sell_stats.get('回落止盈', 0))
            })
        else:
            base_result.update({
                "止损次数": 0.0,
                "到期盈利": 0.0,
                "到期亏损": 0.0,
                "回落止盈": 0.0
            })

        return base_result
    
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
                # 按主进程的列顺序重新排列（所有进程列结构一致）
                thread_a_df = thread_a_df.select(main_a_df.columns)
                main_a_df = pl.concat([main_a_df, thread_a_df], rechunk=True)
                # 删除已合并的进程缓存文件
                cache.delete_file(f"{thread_cache_filename}.csv")
        
        # 去重并排序
        if not main_a_df.is_empty() and not self.chain_debug:
            main_a_df = main_a_df.unique(subset=['配置'])
            # 按年周期收益率排序（处理字符串百分比格式）
            main_a_df = main_a_df.with_columns(
                pl.col('年周期收益率').str.replace('%', '').cast(pl.Float64).alias('年周期收益率数值')
            )
            main_a_df = main_a_df.sort(by='年周期收益率数值', descending=True)
            main_a_df = main_a_df.drop('年周期收益率数值')
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
            
            
            
            # 创建并启动子进程处理剩余组
            for i in range(1, self.processor_count):
                group = strategy_groups[i]
                process_name = f"ChainProcess-{i}"
                process = multiprocessing.Process(
                    target=self._process_strategy_group,
                    args=(group, i),
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


    def execute_one_strategy(self, strategy, start_date, end_date, stock_data, calendar) -> BacktestResult:
        """执行单个策略"""
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
            strategy.update_today(current_date)
            # 固定先卖后买
            strategy.sell()
            strategy.buy()
            strategy.pick()
            strategy.settle_amount()
            current_idx = scalendar.next(current_idx)
        
        result = strategy.calculate_performance(start_date, end_date)

        if self.chain_debug:
            print("=" * 50)
            print(f"时间周期: {result.起始日期} 至 {result.结束日期}")
            print(f"资金: {result.初始资金/100:.2f} - > {result.最终资金/100:.2f}")
            print(f"总收益率: {result.总收益率*100:.2f}%")
            print(f"胜率: {result.胜率*100:.2f}%")
            print(f"交易次数: {result.交易次数}")
            print(f"最大资金: {result.最大资金/100:.2f}")
            print(f"最小资金: {result.最小资金/100:.2f}")
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
        """处理指定索引范围的策略（生成器模式）"""
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
                max_drawdown_ok = result.最小资金 >= init_amount * 0.8 if hasattr(result, '最小资金') else True

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
            if self.run_year:
                try:
                    year_result = self.execute_one_strategy(strategy, 20250101, 20260101, stock_data, calendar)
                except Exception as e:
                    print(f"年周期执行失败: {e}")

            # 构建配置字符串：只保留可调整参数（持仓数量|买入参数|排序参数|卖出参数）
            base_arr = params.get('base_param_arr')
            # 只保留持仓数量（动态仓位，去掉仓位比例）
            base_config = [base_arr[1]]
            cache_key = "|".join(",".join(map(str, arr)) for arr in [base_config, params.get('buy_param_arr'), params.get('pick_param_arr'), params.get('sell_param_arr')])
            new_row = self._create_new_row(actual_win_rate, successful_count, total_count, results, cache_key, year_result)
            print(new_row)

            if is_debug:
                if 'new_row' in locals():
                    print(f"进程 {thread_id}: {new_row}")
                self._draw_fund_trend(all_daily_values, f'进程 {thread_id} - 策略资金变化趋势 - {cache_key}')

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
        chain = Chain(param=param)
        chain.param_generator = gen
        chain.result_file = result_file
        # 使用主进程传递的限制后策略数量
        if total_strategy_count is not None:
            chain.total_strategy_count = total_strategy_count

        chain._process_strategy_generator(thread_id, start_idx, end_idx)


