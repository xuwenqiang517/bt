"""
统一参数配置中心
所有策略参数在此定义，避免散落在多个文件中
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


def r(start: int, end: int, step: int = 1) -> range:
    """左闭右闭区间，使用更直观"""
    return list(range(start, end + 1, step))


@dataclass(frozen=True)
class ParamRanges:
    """参数范围定义 - 不可变配置

    策略字符串格式: 持仓数量|连涨天数,3日涨幅,5日涨幅,涨幅上限|排序方向|止损率,持仓天数,目标涨幅,回撤率
    示例: 1|2,7,6,3|0|-10,5,12,6
    注意：量比(>1)、涨停条件(0次)已内置，不再作为参数
    """
    # 基础参数
    hold_count_range: List[int] = field(default_factory=lambda: [1])  # 持仓数量
    
    # 买入参数（7个）- 根据回测结果精简
    buy_up_day_min_range: List[int] = field(default_factory=lambda: [-1] + r(1, 4))  # 连涨天数下限: -1, 1-4天
    buy_up_day_max_range: List[int] = field(default_factory=lambda: [-1] + r(4, 7))  # 连涨天数上限: -1, 4-7
    buy_day3_min_range: List[int] = field(default_factory=lambda: [-1] + r(5, 9, 3))  # 3日涨幅下限%: -1, 5-9%
    buy_day3_max_range: List[int] = field(default_factory=lambda: [-1] + r(15, 21, 3))  # 3日涨幅上限%: -1, 15, 18, 21
    buy_day5_min_range: List[int] = field(default_factory=lambda: [-1] + r(9, 18, 3))  # 5日涨幅下限%: -1, 9, 12, 15, 18
    buy_day5_max_range: List[int] = field(default_factory=lambda: [-1] + r(20, 29, 3))  # 5日涨幅上限%: -1, 20, 23, 26, 29
    change_pct_max_range: List[int] = field(default_factory=lambda: [-1] + r(2, 8, 2))  # 当日涨幅上限%: -1, 2, 4, 6
    # 涨停条件已内置固定为0（10天内无涨停），不再作为参数
    # 量比已内置到筛选逻辑中（默认>1），不再作为参数
    # 日内振幅参数已移除（回测证明效果不明显）

    # 选股排序参数（1个）
    sort_desc_range: List[int] = field(default_factory=lambda: [0,1])  # 排序方向, 0=成交量升序(冷门股), 1=成交量降序(热门股)

    # 卖出参数（4个）- 根据回测结果精简
    sell_stop_loss_range: List[int] = field(default_factory=lambda: r(-12, -5,3))  # 止损率%: -12到-5
    sell_hold_days_range: List[int] = field(default_factory=lambda: r(4, 12))  # 持仓天数: 5-9天
    sell_target_return_range: List[int] = field(default_factory=lambda: r(4, 15,2))  # 目标涨幅%: 6-10%
    sell_trailing_range: List[int] = field(default_factory=lambda: r(3, 10,2))  # 回撤止盈率%: 3-5%
    
    # 默认初始资金（分）
    default_init_amount: int = 10000000  # 10万元 = 10000000分
    
    def get_buy_ranges(self) -> List[List[Any]]:
        """获取所有买入参数范围列表（量比、涨停条件已内置，不再作为参数）"""
        return [
            self.buy_up_day_min_range,
            self.buy_up_day_max_range,
            self.buy_day3_min_range,
            self.buy_day3_max_range,
            self.buy_day5_min_range,
            self.buy_day5_max_range,
            self.change_pct_max_range,
        ]
    
    def get_pick_ranges(self) -> List[List[Any]]:
        """获取选股排序参数范围列表"""
        return [self.sort_desc_range]
    
    def get_base_ranges(self) -> List[List[Any]]:
        """获取基础参数范围列表"""
        return [self.hold_count_range]
    
    def get_sell_ranges(self) -> List[List[Any]]:
        """获取卖出参数范围列表"""
        return [
            self.sell_stop_loss_range,
            self.sell_hold_days_range,
            self.sell_target_return_range,
            self.sell_trailing_range,
        ]
    
    def get_total_count(self) -> int:
        """计算总参数组合数"""
        total = 1
        for ranges in (self.get_buy_ranges() + self.get_pick_ranges() + 
                      self.get_base_ranges() + self.get_sell_ranges()):
            total *= len(ranges)
        print(f"total count: {total}")
        return total


# 全局默认配置实例
DEFAULT_PARAM_RANGES = ParamRanges()


def parse_strategy_string(strategy_str: str) -> Dict[str, Any]:
    """
    从字符串解析策略参数
    格式: 持仓数量|连涨天数下限,连涨天数上限,3日涨幅下限,3日涨幅上限,5日涨幅下限,5日涨幅上限,当日涨幅上限|排序方向|止损率,持仓天数,目标涨幅,回撤率
    示例: 1|2,-1,7,-1,14,-1,3|0|-10,5,12,6
    注意：量比、涨停条件已内置，不再作为参数；日内振幅参数已移除
    """
    cleaned = ''.join(strategy_str.split())
    parts = cleaned.split("|")

    base_arr = parts[0]
    buy_arr = parts[1]
    pick_arr = parts[2] if len(parts) > 2 else "0"
    sell_arr = parts[3] if len(parts) > 3 else "-8,2,5,7"

    hold_count = int(base_arr)

    # 解析买入参数（7个参数，量比和涨停条件已内置，振幅参数已移除）
    buy_params_raw = buy_arr.split(",")
    if len(buy_params_raw) != 7:
        raise ValueError(f"买入参数必须是7个，当前提供了 {len(buy_params_raw)} 个: {buy_arr}")

    buy_params = [int(v) for v in buy_params_raw]

    return {
        "base_param_arr": [DEFAULT_PARAM_RANGES.default_init_amount, hold_count],
        "buy_param_arr": buy_params,
        "pick_param_arr": [int(pick_arr)],
        "sell_param_arr": list(map(int, sell_arr.split(","))),
        "debug": 1
    }


def build_strategy_string(base_arr: list, buy_arr: list, pick_arr: list, sell_arr: list) -> str:
    """
    从参数数组构建策略字符串
    注意：buy_arr包含7个参数（量比、涨停条件已内置，振幅参数已移除）
    """
    base_str = str(base_arr[1])  # 只保留持仓数量
    buy_str = ",".join(str(x) for x in buy_arr)  # 7个买入参数
    pick_str = str(pick_arr[0]) if pick_arr else "0"
    sell_str = ",".join(str(x) for x in sell_arr)
    return f"{base_str}|{buy_str}|{pick_str}|{sell_str}"
