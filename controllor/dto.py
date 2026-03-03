from typing import NamedTuple, Dict, Any, Callable
import polars as pl
import numpy as np


class ResultSchema:
    """回测结果字段模式 - 集中定义所有结果字段，避免分散在多个文件"""

    # 基础字段定义: (名称, 类型, 默认值)
    BASE_FIELDS = [
        ("起始日期", int, 0),
        ("结束日期", int, 0),
        ("初始资金", float, 0.0),
        ("最终资金", float, 0.0),
        ("总收益", float, 0.0),
        ("总收益率", float, 0.0),
        ("胜率", float, 0.0),
        ("交易次数", int, 0),
        ("期max", float, 0.0),
        ("期min", float, 0.0),
        ("平均资金使用率", float, 0.0),
        ("卖出统计", dict, lambda: {'止损': 0, '到期盈利': 0, '到期亏损': 0, '回落止盈': 0}),
    ]

    # Chain层扩展字段
    CHAIN_FIELDS = [
        ("总胜率", str, ""),
        ("期胜率", str, ""),
        ("年胜率", str, "0%"),
        ("期收益", str, ""),
        ("年收益", str, "0.00%"),
        ("年交易数", int, 0),
        ("期max", str, ""),
        ("期min", str, ""),
        ("年max", str, ""),
        ("年min", str, ""),
        ("年夏普", float, 0.0),
        ("配置", str, ""),
        # 选股信号统计字段
        ("选股信号数", int, 0),
        ("1日胜率", str, ""),
        ("3日胜率", str, ""),
        ("5日胜率", str, ""),
        ("1日盈亏比", str, ""),
        ("3日盈亏比", str, ""),
        ("5日盈亏比", str, ""),
        ("1日平均收益", str, ""),
        ("3日平均收益", str, ""),
        ("5日平均收益", str, ""),
        # 卖出策略统计字段
        ("止损次数", int, 0),
        ("到期盈利", int, 0),
        ("到期亏损", int, 0),
        ("回落止盈", int, 0),
    ]

    @classmethod
    def get_all_field_names(cls) -> list:
        """获取所有字段名称"""
        return [f[0] for f in cls.BASE_FIELDS + cls.CHAIN_FIELDS]

    @classmethod
    def get_base_field_names(cls) -> list:
        """获取基础字段名称"""
        return [f[0] for f in cls.BASE_FIELDS]

    @classmethod
    def get_chain_field_names(cls) -> list:
        """获取Chain层字段名称"""
        return [f[0] for f in cls.CHAIN_FIELDS]

    @classmethod
    def create_empty_backtest_result(cls, start_date: int, end_date: int, init_amount: float) -> Dict[str, Any]:
        """创建空的回测结果字典"""
        return {
            "起始日期": start_date,
            "结束日期": end_date,
            "初始资金": init_amount,
            "最终资金": init_amount,
            "总收益": 0.0,
            "总收益率": 0.0,
            "胜率": 0.0,
            "交易次数": 0,
            "期max": init_amount,
            "期min": init_amount,
            "平均资金使用率": 0.0,
            "卖出统计": {'止损': 0, '到期盈利': 0, '到期亏损': 0, '回落止盈': 0},
            "夏普比率": 0.0
        }

    @classmethod
    def create_backtest_result(cls, **kwargs) -> Dict[str, Any]:
        """创建回测结果字典，未提供的字段使用默认值"""
        result = {}
        for name, dtype, default in cls.BASE_FIELDS:
            if name in kwargs:
                result[name] = kwargs[name]
            else:
                result[name] = default() if callable(default) else default
        return result

    @classmethod
    def create_empty_chain_row(cls, cache_key: str = "") -> Dict[str, Any]:
        """创建空的Chain层行数据"""
        result = {}
        for name, dtype, default in cls.CHAIN_FIELDS:
            if name == "配置":
                result[name] = cache_key
            else:
                result[name] = default() if callable(default) else default
        return result

    @classmethod
    def create_chain_row_from_results(cls, actual_win_rate: float, successful_count: int,
                                       total_periods: int, results: list, cache_key: str,
                                       year_result: Any = None) -> Dict[str, Any]:
        """从BacktestResult列表创建Chain层行数据"""
        if not results:
            row = cls.create_empty_chain_row(cache_key)
            row["总胜率"] = f"{successful_count}/{total_periods}"
            return row

        # 资金转换为万为单位
        def to_wan_str(value):
            """将分转换为万，保留整数，带万字"""
            return f"{int(value / 1000000)}万"

        row = {
            "总胜率": f"{successful_count}/{total_periods}",
            "期胜率": f"{int(np.mean([x.胜率 for x in results]) * 100)}%",
            "期收益": f"{float(np.mean([x.总收益率 for x in results])) * 100:.1f}%",
            "期max": to_wan_str(max([x.期max for x in results])),
            "期min": to_wan_str(min([x.期min for x in results])),
            "配置": cache_key,
            # 年周期字段默认值（当年周期未执行时）
            "年收益": "0.0%",
            "年胜率": "0%",
            "年夏普": 0.0,
            "年max": "0万",
            "年min": "0万",
            "年交易数": 0,
            "止损次数": 0,
            "到期盈利": 0,
            "到期亏损": 0,
            "回落止盈": 0
        }

        # 年周期统计
        if year_result:
            row.update({
                "年收益": f"{float(year_result.总收益率) * 100:.1f}%",
                "年胜率": f"{int(year_result.胜率 * 100)}%",
                "年夏普": round(float(year_result.夏普比率), 1),
                "年max": to_wan_str(year_result.期max),
                "年min": to_wan_str(year_result.期min),
                "年交易数": int(year_result.交易次数)
            })
            if hasattr(year_result, '卖出统计'):
                sell_stats = year_result.卖出统计
                row.update({
                    "止损次数": int(sell_stats.get('止损', 0)),
                    "到期盈利": int(sell_stats.get('到期盈利', 0)),
                    "到期亏损": int(sell_stats.get('到期亏损', 0)),
                    "回落止盈": int(sell_stats.get('回落止盈', 0))
                })

        return row

    @classmethod
    def create_empty_dataframe(cls) -> pl.DataFrame:
        """创建空的Chain层DataFrame，包含所有列"""
        data = {}
        for name, dtype, default in cls.CHAIN_FIELDS:
            if dtype == str:
                data[name] = pl.Series([], dtype=pl.String)
            elif dtype == int:
                data[name] = pl.Series([], dtype=pl.Int64)
            else:
                data[name] = pl.Series([], dtype=pl.Float64)
        return pl.DataFrame(data)

    @classmethod
    def ensure_columns(cls, df: pl.DataFrame) -> pl.DataFrame:
        """确保DataFrame包含所有必需的列"""
        for name, dtype, default in cls.CHAIN_FIELDS:
            if name not in df.columns:
                if dtype == str:
                    df = df.with_columns(pl.lit("").alias(name))
                elif dtype == int:
                    df = df.with_columns(pl.lit(0).alias(name))
                else:
                    df = df.with_columns(pl.lit(0.0).alias(name))
        return df


class HoldStock:
    # 仅定义需要的字段，禁用__dict__，大幅节省内存
    __slots__ = [
        "code",          # 股票代码
        "buy_price",     # 买入价格（建议用float，实际价格多为小数）
        "buy_count",     # 买入数量
        "buy_day",       # 买入日期
        "sell_price",    # 卖出价格（初始可设为None）
        "sell_day",      # 卖出日期（初始可设为None）
        "highest_price"  # 持仓期间最高价（初始可设为买入价）
    ]
    
    def __init__(self, code, buy_price, buy_count, buy_day):
        # 初始化必填的买入信息
        self.code = code
        self.buy_price = buy_price
        self.buy_count = buy_count
        self.buy_day = buy_day
        # 初始化可变字段（卖出相关为None，最高价初始为买入价）
        self.sell_price = None
        self.sell_day = None
        self.highest_price = buy_price

    # 可选：添加便捷方法，简化更新操作
    def update_highest_price(self, new_high):
        """更新持仓最高价（仅当新价格更高时）"""
        if new_high > self.highest_price:
            self.highest_price = new_high

    def set_sell_info(self, sell_price, sell_day):
        """设置卖出信息"""
        self.sell_price = sell_price
        self.sell_day = sell_day

SellStrategy=NamedTuple("SellStrategy", [
    ("name", str),
    ("params", object)  # 改为object类型，支持不同类型的参数对象
])

StopLossParams=NamedTuple("StopLossParams", [
    ("rate", float)
])

StopProfitParams=NamedTuple("StopProfitParams", [
    ("rate", float)
])

CumulativeSellParams=NamedTuple("CumulativeSellParams", [
    ("days", int),
    ("min_return", float)
])

TrailingStopProfitParams=NamedTuple("TrailingStopProfitParams", [
    ("rate", float)
])

BacktestResult=NamedTuple("BacktestResult", [
    ("起始日期", int),
    ("结束日期", int),
    ("初始资金", float),
    ("最终资金", float),
    ("总收益", float),
    ("总收益率", float),
    ("胜率", float),
    ("交易次数", int),
    ("期max", float),
    ("期min", float),
    ("平均资金使用率", float),
    ("卖出统计", dict),  # 卖出原因统计：止损、到期盈利、到期亏损、回落止盈
    ("夏普比率", float)  # 仅年周期计算
])

StrategyBacktestResult=NamedTuple("StrategyBacktestResult", [
    ("策略配置", dict),
    ("平均收益率", float),
    ("平均胜率", float),
    ("周期胜率", float),
    ("平均最大回撤", float)
])
