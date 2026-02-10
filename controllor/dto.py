from typing import NamedTuple


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
    ("起始日期", str),
    ("结束日期", str),
    ("初始资金", float),
    ("最终资金", float),
    ("总收益", float),
    ("总收益率", float),
    ("胜率", float),
    ("交易次数", int),
    ("最大资金", float),
    ("最小资金", float),
    ("夏普比率", float),
    ("平均资金使用率", float)
])

StrategyBacktestResult=NamedTuple("StrategyBacktestResult", [
    ("策略配置", dict),
    ("平均收益率", float),
    ("平均胜率", float),
    ("周期胜率", float),
    ("平均最大回撤", float)
])
