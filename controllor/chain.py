from typing import List, Dict, Any
import multiprocessing
from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
import numpy as np
import polars as pl
from tqdm import tqdm
from local_cache import LocalCache
import chart
from dto import *
from strategy import Strategy
import logger_config


def _noop(*args, **kwargs):
    """空操作函数 - debug=False 时作为日志函数"""
    pass


class Chain:
    def __init__(self, param=None):
        self.strategies = param.get("strategy")  # 策略列表（可能为None，使用生成器模式）
        self.date_arr = param["date_arr"]  # 回测时间周期列表
        self.chain_debug = param.get("chain_debug", False)  # 是否打印报告
        self.win_rate_threshold = param.get("win_rate_threshold", 0.99)  # 胜率阈值
        self.processor_count = param.get("processor_count", 1)  # 进程数
        self.fail_count = param.get("fail_count", 1)  # 允许失败次数
        self.force_refresh = param.get("force_refresh", False)  # 是否强制刷新数据缓存
        self.param = param  # 原始参数
        self.result_file = param.get("result_file", None)  # 结果文件
        self.run_year = param.get("run_year", True)  # 是否跑年周期
        self.use_param_generator = param.get("use_param_generator", False)  # 是否使用参数生成器
        self.param_generator = param.get("param_generator", None)  # 参数生成器
        self.total_strategy_count = param.get("total_strategy_count", 0)  # 总策略数量
        # Debug 函数绑定 - 避免在循环中反复判断 if is_debug
        self._debug_log = self._create_debug_logger()

    def _create_debug_logger(self):
        """创建调试日志函数 - 非调试模式返回空操作，调试模式返回实际打印函数"""
        if not self.chain_debug:
            return _noop  # 空操作函数，零开销

        def debug_log(msg, *args):
            print(msg.format(*args) if args else msg)

        return debug_log

    def _debug_log_year_failure(self, error):
        """记录年周期执行失败"""
        if self.chain_debug:
            print(f"年周期执行失败: {error}")

    def _debug_log_performance(self, result):
        """记录绩效详情"""
        if self.chain_debug:
            print("=" * 50)
            print(f"时间周期: {result.起始日期} 至 {result.结束日期}")
            print(f"资金: {result.初始资金/100:.2f} -> {result.最终资金/100:.2f}")
            print(f"总收益率: {result.总收益率*100:.2f}%")
            print(f"胜率: {result.胜率*100:.2f}%")
            print(f"交易次数: {result.交易次数}")
            print(f"最大资金: {result.期max/100:.2f}")
            print(f"最小资金: {result.期min/100:.2f}")
            print(f"夏普比率: {result.夏普比率:.2f}")

    @staticmethod
    def _process_strategy_group_worker(strategy_group: List[Dict[str, Any]], thread_id: int, param: dict) -> None:
        """子进程工作函数（静态方法，可序列化）"""
        chain = Chain(param=param)
        chain._process_strategy_group(strategy_group, thread_id)

    def _process_strategy_group(self, strategy_group: List[Dict[str, Any]], thread_id: int) -> None:
        """处理策略列表（指定参数模式）

        Args:
            strategy_group: 预生成的策略参数列表
            thread_id: 进程ID
        """
        self._process_strategies(
            strategy_iter=strategy_group,
            thread_id=thread_id,
            year_end_date=20260301,
            calc_sharpe=False,
            estimated_total=len(strategy_group)
        )

    def _process_strategy_generator(self, thread_id: int, start_idx: int, end_idx: int, result_file: str = None) -> None:
        """处理指定索引范围的策略（生成器模式）

        Args:
            thread_id: 进程ID
            start_idx: 起始索引
            end_idx: 结束索引
            result_file: 结果文件（保留参数，兼容接口）
        """
        strategy_iter = self.param_generator.get_slice_params(start_idx, end_idx)
        self._process_strategies(
            strategy_iter=strategy_iter,
            thread_id=thread_id,
            year_end_date=20260101,
            calc_sharpe=True,
            estimated_total=end_idx - start_idx
        )

    def _process_strategies(self, strategy_iter, thread_id: int, year_end_date: int, calc_sharpe: bool, estimated_total: int = None) -> None:
        """处理策略列表核心逻辑 - 合并 _process_strategy_group 和 _process_strategy_generator

        Args:
            strategy_iter: 策略迭代器（列表或生成器）
            thread_id: 进程ID
            year_end_date: 年周期结束日期
            calc_sharpe: 是否计算夏普比率
            estimated_total: 预估总数（生成器模式用于显示进度）
        """
        cache = LocalCache()  # 本地缓存
        thread_result_file = f"{self.result_file}_thread_{thread_id}"  # 线程结果文件名
        stock_data = sd(force_refresh=self.force_refresh)  # 股票数据
        calendar = sc()  # 交易日历
        cache_filename = f"a_{thread_result_file}"  # 缓存文件名
        cached_a_df = self._load_cache(cache, cache_filename)  # 缓存DataFrame
        fail_count = self.fail_count  # 允许失败次数
        total_count = len(self.date_arr)  # 总周期数

        param_count = len(strategy_iter) if hasattr(strategy_iter, '__len__') else estimated_total  # 生成器模式用预估总数
        BATCH_SIZE = 500  # 批量合并大小（减少合并次数提升性能）
        pending_rows = []  # 待合并的行

        for idx, params in enumerate(tqdm(strategy_iter, desc=f"进程 {thread_id} 执行策略",
                                         total=param_count, position=thread_id, leave=True, mininterval=5)):
            params['record_pick_signals'] = False  # 周期回测不记录选股信号
            strategy = Strategy(**params)  # 策略实例
            results = []  # 成功的回测结果
            failure_count = 0  # 失败次数
            init_amount = params.get('base_param_arr')[0]  # 初始资金（提取到循环外，避免重复获取）

            for s, e in self.date_arr:
                result = self.execute_one_strategy(strategy, s, e, stock_data, calendar)
                max_drawdown_ok = result.期min >= init_amount * 0.8 if hasattr(result, '期min') else True  # 最大回撤检查

                if result.总收益率 <= 0 or not max_drawdown_ok:
                    failure_count += 1
                    if failure_count > fail_count and not self.chain_debug:
                        break
                    continue
                results.append(result)

            if failure_count > fail_count and not self.chain_debug:
                continue

            successful_count = total_count - failure_count  # 成功次数
            actual_win_rate = successful_count / total_count if results else 0  # 实际胜率
            year_result = None  # 年周期结果
            pick_signal_stats = None  # 选股信号统计

            if self.run_year:
                try:
                    strategy._record_pick_signals = True  # 开启选股信号记录
                    year_result = self.execute_one_strategy(strategy, 20250101, year_end_date, stock_data, calendar, calc_sharpe=calc_sharpe)
                    if hasattr(strategy, 'pick_signals') and strategy.pick_signals:
                        pick_signal_stats = self._calc_pick_signal_stats(strategy.pick_signals, stock_data, calendar)
                except Exception as e:
                    self._debug_log_year_failure(e)

            base_arr = params.get('base_param_arr')  # 基础参数
            base_config = [base_arr[1]]  # 基础配置（持仓数量）
            cache_key = "|".join(",".join(map(str, arr)) for arr in  # 配置key
                               [base_config, params.get('buy_param_arr'), params.get('pick_param_arr'), params.get('sell_param_arr')])
            new_row = self._create_new_row(actual_win_rate, successful_count, total_count, results, cache_key, year_result, pick_signal_stats)

            self._debug_log(f"进程 {thread_id}: {new_row}")

            pending_rows.append(new_row)  # 添加到待合并列表

            if len(pending_rows) >= BATCH_SIZE:  # 达到批量大小则合并
                cached_a_df = self._batch_merge_rows(cached_a_df, pending_rows)
                pending_rows = []
                self._save_cache(cache, cache_filename, cached_a_df)

        if pending_rows:  # 处理剩余数据
            cached_a_df = self._batch_merge_rows(cached_a_df, pending_rows)

        cached_a_df = self._sort_data(cached_a_df)
        self._save_cache(cache, cache_filename, cached_a_df)

    def _batch_merge_rows(self, cached_a_df: pl.DataFrame, rows: list) -> pl.DataFrame:
        """批量合并行到DataFrame

        Args:
            cached_a_df: 缓存的DataFrame
            rows: 待合并的行列表
        """
        if not rows:
            return cached_a_df

        new_data_df = pl.DataFrame(rows)  # 新数据DataFrame
        if cached_a_df.is_empty():
            return new_data_df

        new_data_df = ResultSchema.ensure_columns(new_data_df)  # 确保新数据有所有列
        # 对齐列：只保留缓存中存在的列，缺失的用默认值补充
        for col in cached_a_df.columns:
            if col not in new_data_df.columns:
                dtype = cached_a_df[col].dtype
                if dtype == pl.String:
                    new_data_df = new_data_df.with_columns(pl.lit("").alias(col))
                elif dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8):
                    new_data_df = new_data_df.with_columns(pl.lit(0).cast(dtype).alias(col))
                else:
                    new_data_df = new_data_df.with_columns(pl.lit(0.0).alias(col))
        # 只保留缓存中存在的列
        new_data_df = new_data_df.select(cached_a_df.columns)
        # 确保类型一致
        for col in cached_a_df.columns:
            if cached_a_df[col].dtype != new_data_df[col].dtype:
                new_data_df = new_data_df.with_columns(pl.col(col).cast(cached_a_df[col].dtype).alias(col))
        merged = pl.concat([cached_a_df, new_data_df], rechunk=True)  # 合并
        return self._sort_data(merged)

    def _load_cache(self, cache: LocalCache, filename: str) -> pl.DataFrame:
        """加载缓存DataFrame"""
        df = cache.get_csv_pl(filename)
        return df if df is not None else ResultSchema.create_empty_dataframe()

    def _save_cache(self, cache: LocalCache, filename: str, df: pl.DataFrame) -> None:
        """保存DataFrame到缓存"""
        if not df.is_empty():
            cache.set_csv_pl(filename, df)

    def _merge_and_sort_data(self, target_df: pl.DataFrame, new_df: pl.DataFrame) -> pl.DataFrame:
        """合并并排序数据"""
        target_df = ResultSchema.ensure_columns(target_df)  # 确保目标DataFrame列完整
        new_df = ResultSchema.ensure_columns(new_df)  # 确保新DataFrame列完整
        merged_df = pl.concat([target_df, new_df], rechunk=True)  # 合并
        return self._sort_data(merged_df)

    def _sort_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """按年收益降序排序"""
        if df.is_empty() or '年收益' not in df.columns:
            return df

        df = df.with_columns(
            pl.col('年收益').str.replace('%', '').cast(pl.Float64).alias('年收益数值')
        )
        df = df.sort(by='年收益数值', descending=True)
        return df.drop('年收益数值')

    def _calc_pick_signal_stats(self, pick_signals: list, stock_data, calendar) -> dict | None:
        """计算选股信号统计（1/3/5天后盈亏情况）

        买入价：选股日的次日开盘价
        1日收益：次日收盘 - 次日开盘
        3日收益：3个交易日后的收盘 - 次日开盘
        5日收益：5个交易日后的收盘 - 次日开盘
        """
        if not pick_signals:
            return None

        stats = {1: {'profits': []}, 3: {'profits': []}, 5: {'profits': []}}  # 统计字典

        for signal in pick_signals:
            date = signal['date']  # 信号日期
            code = signal['code']  # 股票代码

            current_idx = calendar.start(date)  # 当前日期索引
            if current_idx == -1:
                continue

            next_idx = current_idx + 1  # 次日索引（买入日）
            if next_idx >= len(calendar.df):
                continue

            buy_date = calendar.get_date(next_idx)  # 买入日期
            buy_data = stock_data.get_full_data_by_date_code(buy_date, code)  # 买入日数据
            if buy_data.close == 0:
                continue
            buy_price = buy_data.open  # 买入价（次日开盘价）

            for days in [1, 3, 5]:
                target_idx = next_idx + days - 1  # 目标日期索引
                if target_idx >= len(calendar.df):
                    continue

                sell_date = calendar.get_date(target_idx)  # 卖出日期
                sell_data = stock_data.get_full_data_by_date_code(sell_date, code)  # 卖出日数据
                if sell_data.close == 0:
                    continue

                sell_price = sell_data.close  # 卖出价（收盘价）
                profit_rate = (sell_price - buy_price) / buy_price if buy_price > 0 else 0  # 收益率
                stats[days]['profits'].append(profit_rate)

        has_valid_data = any(stats[days]['profits'] for days in [1, 3, 5])  # 是否有有效数据
        if not has_valid_data:
            return None

        result = {'选股信号数': len(pick_signals)}  # 结果字典

        for days in [1, 3, 5]:
            profits = stats[days]['profits']
            if not profits:
                result[f'{days}胜'] = ''
                result[f'{days}盈亏比'] = ''
                result[f'{days}收益'] = ''
                continue

            win_count = sum(1 for p in profits if p > 0)  # 盈利次数
            win_rate = win_count / len(profits)  # 胜率
            avg_profit = sum(profits) / len(profits)  # 平均收益

            win_profits = [p for p in profits if p > 0]  # 所有盈利
            loss_profits = [abs(p) for p in profits if p < 0]  # 所有亏损
            avg_win = sum(win_profits) / len(win_profits) if win_profits else 0  # 平均盈利
            avg_loss = sum(loss_profits) / len(loss_profits) if loss_profits else 0  # 平均亏损
            profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else (float('inf') if avg_win > 0 else 0)  # 盈亏比

            result[f'{days}胜'] = f"{win_rate*100:.1f}%"
            result[f'{days}盈亏比'] = f"{profit_loss_ratio:.2f}" if profit_loss_ratio != float('inf') else "∞"
            result[f'{days}收益'] = f"{avg_profit*100:.2f}%"

        return result

    def _create_new_row(self, actual_win_rate: float, successful_count: int, total_periods: int,
                        results: list, cache_key: str, year_result: object = None,
                        pick_signal_stats: dict = None) -> dict:
        """创建新的行数据"""
        row = ResultSchema.create_chain_row_from_results(
            actual_win_rate, successful_count, total_periods, results, cache_key, year_result
        )
        if pick_signal_stats:
            row.update(pick_signal_stats)
        return row

    def _merge_thread_caches(self) -> None:
        """合并所有进程的缓存文件"""
        cache = LocalCache()  # 本地缓存
        main_cache_filename = f"a_{self.result_file}"  # 主缓存文件名
        main_a_df = self._load_cache(cache, main_cache_filename)  # 主缓存DataFrame

        for thread_id in range(self.processor_count):  # 遍历所有进程
            thread_result_file = f"{self.result_file}_thread_{thread_id}"  # 线程结果文件名
            thread_cache_filename = f"a_{thread_result_file}"  # 线程缓存文件名

            thread_a_df = cache.get_csv_pl(thread_cache_filename)  # 线程缓存DataFrame
            if thread_a_df is not None and not thread_a_df.is_empty():
                for col in main_a_df.columns:  # 确保列一致
                    if col not in thread_a_df.columns:
                        col_dtype = main_a_df[col].dtype  # 主缓存列类型
                        if col_dtype in (pl.Float64, pl.Float32):
                            thread_a_df = thread_a_df.with_columns(pl.lit(0.0).alias(col))
                        elif col_dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8):
                            thread_a_df = thread_a_df.with_columns(pl.lit(0).cast(col_dtype).alias(col))
                        else:
                            thread_a_df = thread_a_df.with_columns(pl.lit("").alias(col))

                for col in main_a_df.columns:  # 统一列类型
                    if col in thread_a_df.columns:
                        main_dtype = main_a_df[col].dtype  # 主缓存类型
                        thread_dtype = thread_a_df[col].dtype  # 线程缓存类型
                        if main_dtype != thread_dtype:
                            thread_a_df = thread_a_df.with_columns(pl.col(col).cast(main_dtype).alias(col))

                thread_a_df = thread_a_df.select(main_a_df.columns)  # 重新排列列顺序
                main_a_df = pl.concat([main_a_df, thread_a_df], rechunk=True)  # 合并
                cache.delete_file(f"{thread_cache_filename}.csv")  # 删除已合并的线程缓存

        if not main_a_df.is_empty() and not self.chain_debug:  # 非调试模式下保存
            main_a_df = main_a_df.unique(subset=['配置'])  # 按配置去重
            if '年收益' in main_a_df.columns:  # 按年收益排序
                main_a_df = main_a_df.with_columns(
                    pl.col('年收益').str.replace('%', '').cast(pl.Float64).alias('年收益数值')
                )
                main_a_df = main_a_df.sort(by='年收益数值', descending=True)
                main_a_df = main_a_df.drop('年收益数值')
            self._save_cache(cache, main_cache_filename, main_a_df)

    def execute(self) -> list:
        """执行回测（指定参数模式）- 使用预生成的策略列表"""
        return self._execute_core(
            total_strategies=len(self.strategies),
            worker_func=self._process_strategy_group_worker,
            worker_args_func=lambda i, start_idx, end_idx: (self.strategies[start_idx:end_idx], i, self.param)
        )

    def execute_generator_mode(self) -> None:
        """执行回测（生成器模式）- 使用参数生成器动态生成策略"""
        result_file = self.result_file  # 捕获到闭包中
        self._execute_core(
            total_strategies=self.total_strategy_count,
            worker_func=self._process_strategy_generator_worker,
            worker_args_func=lambda i, start_idx, end_idx: (i, start_idx, end_idx, self.param, result_file)
        )

    def _execute_core(self, total_strategies: int, worker_func, worker_args_func) -> list | None:
        """执行回测核心逻辑 - 所有进程平等干活，主进程纯管理

        Args:
            total_strategies: 总策略数
            worker_func: 子进程工作函数（静态方法）
            worker_args_func: 构建工作函数参数的函数
        """
        self._merge_thread_caches()  # 先合并缓存，确保中断后可恢复

        print(f"总策略数: {total_strategies}")
        print(f"使用进程数: {self.processor_count}")

        if total_strategies == 0:
            print("所有策略已执行完毕，无需处理")
            return

        group_size = total_strategies // self.processor_count  # 每组大小
        processes = []  # 进程列表

        for i in range(self.processor_count):
            start_idx = i * group_size  # 起始索引
            end_idx = (i + 1) * group_size if i < self.processor_count - 1 else total_strategies  # 结束索引
            process_name = f"ChainProcess-{i}"  # 进程名
            args = worker_args_func(i, start_idx, end_idx)
            process = multiprocessing.Process(
                target=worker_func,
                args=args,
                name=process_name
            )
            processes.append(process)
            process.start()
            print(f"启动进程: {process_name}，处理策略索引: {start_idx} - {end_idx}")

        for process in processes:
            process.join()
            print(f"进程 {process.name} 处理完成")

        self._merge_thread_caches()  # 最终合并

    def execute_one_strategy(self, strategy, start_date, end_date, stock_data, calendar, calc_sharpe: bool = False) -> BacktestResult:
        """执行单个策略"""
        scalendar = calendar  # 交易日历
        current_idx = scalendar.start(start_date)  # 起始索引
        end_idx = scalendar.start(end_date)  # 结束索引

        if current_idx == -1 or end_idx == -1:
            raise ValueError(f"Invalid date range: {start_date} to {end_date}")

        strategy.bind(stock_data, calendar)  # 绑定数据
        strategy.reset()  # 重置状态

        while current_idx != -1 and current_idx <= end_idx:
            current_date = scalendar.get_date(current_idx)  # 当前日期
            strategy.update_today(current_date, current_idx)  # 更新今日数据
            strategy.sell()  # 卖出
            strategy.buy()  # 买入
            strategy.pick()  # 选股
            strategy.settle_amount()  # 结算
            current_idx += 1

        result = strategy.calculate_performance(start_date, end_date, calc_sharpe)  # 计算绩效
        self._debug_log_performance(result)  # 调试模式下打印绩效详情
        return result

    @staticmethod
    def _process_strategy_generator_worker(thread_id: int, start_idx: int, end_idx: int, param: dict,
                                          result_file: str, total_strategy_count: int = None) -> None:
        """子进程工作函数（静态方法，可序列化）"""
        from param_generator import ParamGenerator

        gen = ParamGenerator()  # 参数生成器
        chain = Chain(param=param)  # Chain实例
        chain.param_generator = gen
        chain.result_file = result_file
        if total_strategy_count is not None:
            chain.total_strategy_count = total_strategy_count

        chain._process_strategy_generator(thread_id, start_idx, end_idx)