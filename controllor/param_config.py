"""
统一参数配置中心
所有策略参数在此定义，避免散落在多个文件中
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass(frozen=True)
class ParamRanges:
    """参数范围定义 - 不可变配置
    
    策略字符串格式: 持仓数量|连涨天数,3日涨幅,5日涨幅,涨幅上限,涨停条件,量比|排序方向|止损率,持仓天数,目标涨幅,回撤率
    示例: 1|2,7,6,3,-1,1|0|-10,5,12,6
    """
    # 基础参数
    hold_count_range: List[int] = field(default_factory=lambda: [1, 2, 3, 4])  # 持仓数量
    
    # 买入参数（6个）- 扩展取值范围增加计算量
    buy_up_day_range: List[int] = field(default_factory=lambda: [-1, 1, 2, 3, 4, 5])  # 连涨天数, -1=不限
    buy_day3_range: List[int] = field(default_factory=lambda: [-1, 3, 5, 7, 9])  # 3日涨幅%, -1=不限
    buy_day5_range: List[int] = field(default_factory=lambda: [-1, 5, 8, 12, 15, 20])  # 5日涨幅%, -1=不限
    change_pct_max_range: List[int] = field(default_factory=lambda: [-1, 3, 5, 7, 9])  # 当日涨幅上限%, -1=不限
    limit_up_count_range: List[int] = field(default_factory=lambda: [-1, 0, 1])  # 涨停条件, -1=不限, 0=0次, 1=1次+
    volume_ratio_range: List[float] = field(default_factory=lambda: [-1, 1, 1.5, 2, 2.5])  # 量比, -1=不限
    
    # 选股排序参数（1个）
    sort_desc_range: List[int] = field(default_factory=lambda: [0, 1])  # 排序方向, 0=成交量升序(冷门股), 1=成交量降序(热门股)
    
    # 卖出参数（4个）- 扩展取值范围
    sell_stop_loss_range: List[int] = field(default_factory=lambda: [-5, -8, -10, -12, -15])  # 止损率%
    sell_hold_days_range: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 10, 15])  # 持仓天数限制
    sell_target_return_range: List[int] = field(default_factory=lambda: [5, 8, 10, 15, 20, 25])  # 目标涨幅%
    sell_trailing_range: List[int] = field(default_factory=lambda: [4, 5, 7, 10, 12])  # 回撤止盈率%
    
    # 默认初始资金（分）
    default_init_amount: int = 10000000  # 10万元 = 10000000分
    
    def get_buy_ranges(self) -> List[List[Any]]:
        """获取所有买入参数范围列表"""
        return [
            self.buy_up_day_range,
            self.buy_day3_range,
            self.buy_day5_range,
            self.change_pct_max_range,
            self.limit_up_count_range,
            self.volume_ratio_range,
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
    格式: 持仓数量|连涨天数,3日涨幅,5日涨幅,涨幅上限,涨停条件,量比|排序方向|止损率,持仓天数,目标涨幅,回撤率
    示例: 1|2,7,6,3,-1,1|0|-10,5,12,6
    """
    cleaned = ''.join(strategy_str.split())
    parts = cleaned.split("|")
    
    base_arr = parts[0]
    buy_arr = parts[1]
    pick_arr = parts[2] if len(parts) > 2 else "0"
    sell_arr = parts[3] if len(parts) > 3 else "-8,2,5,7"
    
    hold_count = int(base_arr)
    
    # 解析买入参数（必须提供完整的6个参数）
    buy_params_raw = buy_arr.split(",")
    if len(buy_params_raw) != 6:
        raise ValueError(f"买入参数必须是6个，当前提供了 {len(buy_params_raw)} 个: {buy_arr}")
    
    buy_params = []
    for i, v in enumerate(buy_params_raw):
        if i == 5:  # 量比可能是小数
            buy_params.append(float(v))
        else:
            buy_params.append(int(v))
    
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
    """
    base_str = str(base_arr[1])  # 只保留持仓数量
    buy_str = ",".join(str(x) for x in buy_arr)
    pick_str = str(pick_arr[0]) if pick_arr else "0"
    sell_str = ",".join(str(x) for x in sell_arr)
    return f"{base_str}|{buy_str}|{pick_str}|{sell_str}"
