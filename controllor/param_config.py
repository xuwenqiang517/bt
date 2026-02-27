"""
统一参数配置中心
所有策略参数在此定义，避免散落在多个文件中
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass(frozen=True)
class ParamRanges:
    """参数范围定义 - 不可变配置

    策略字符串格式: 持仓数量|连涨天数,3日涨幅,5日涨幅,涨幅上限|排序方向|止损率,持仓天数,目标涨幅,回撤率
    示例: 1|2,7,6,3|0|-10,5,12,6
    注意：量比(>1)、涨停条件(0次)已内置，不再作为参数
    """
    # 基础参数
    hold_count_range: List[int] = field(default_factory=lambda: [1])  # 持仓数量
    
    # 买入参数（6个）- 根据回测结果精简
    buy_up_day_range: List[int] = field(default_factory=lambda: [-1,1,2,3])  # 连涨天数: 2,3天最优，5天测试
    buy_day3_range: List[int] = field(default_factory=lambda: [-1,1,2,3,4,5,6,7,8])  # 3日涨幅%: 结果证明此参数无关，固定-1
    buy_day5_range: List[int] = field(default_factory=lambda: [-1,8,10,12,14,16,18,20])  # 5日涨幅%: 12%起步，15%和20%是核心
    change_pct_max_range: List[int] = field(default_factory=lambda: [-1,1,2,3,4,5,6,7,8,9])  # 当日涨幅上限%: 4%最优，6%和8%测试
    # 涨停条件已内置固定为0（10天内无涨停），不再作为参数
    # 量比已内置到筛选逻辑中（默认>1），不再作为参数
    
    # 选股排序参数（1个）
    sort_desc_range: List[int] = field(default_factory=lambda: [0, 1])  # 排序方向, 0=成交量升序(冷门股), 1=成交量降序(热门股)
    
    # 卖出参数（4个）- 根据回测结果精简
    sell_stop_loss_range: List[int] = field(default_factory=lambda: [-8,-10,-12])  # 止损率%: -10最优，-8/-12对比
    sell_hold_days_range: List[int] = field(default_factory=lambda: [2,3,4,5,6,7,8,9])  # 持仓天数: 9天最优，7/12/15对比
    sell_target_return_range: List[int] = field(default_factory=lambda: [3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])  # 目标涨幅%: 15%和18%最优，22%测试
    sell_trailing_range: List[int] = field(default_factory=lambda: [3,4,5,6,7,8,9,10])  # 回撤止盈率%: 10%最优，6%和8%对比
    
    # 默认初始资金（分）
    default_init_amount: int = 10000000  # 10万元 = 10000000分
    
    def get_buy_ranges(self) -> List[List[Any]]:
        """获取所有买入参数范围列表（量比、涨停条件已内置，不再作为参数）"""
        return [
            self.buy_up_day_range,
            self.buy_day3_range,
            self.buy_day5_range,
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
    格式: 持仓数量|连涨天数,3日涨幅,5日涨幅,涨幅上限|排序方向|止损率,持仓天数,目标涨幅,回撤率
    示例: 1|2,7,6,3|0|-10,5,12,6
    注意：量比、涨停条件已内置，不再作为参数
    """
    cleaned = ''.join(strategy_str.split())
    parts = cleaned.split("|")

    base_arr = parts[0]
    buy_arr = parts[1]
    pick_arr = parts[2] if len(parts) > 2 else "0"
    sell_arr = parts[3] if len(parts) > 3 else "-8,2,5,7"

    hold_count = int(base_arr)

    # 解析买入参数（4个参数，量比和涨停条件已内置）
    buy_params_raw = buy_arr.split(",")
    if len(buy_params_raw) != 4:
        raise ValueError(f"买入参数必须是4个，当前提供了 {len(buy_params_raw)} 个: {buy_arr}")

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
    注意：buy_arr包含4个参数（量比、涨停条件已内置，不作为参数）
    """
    base_str = str(base_arr[1])  # 只保留持仓数量
    buy_str = ",".join(str(x) for x in buy_arr)  # 4个买入参数
    pick_str = str(pick_arr[0]) if pick_arr else "0"
    sell_str = ",".join(str(x) for x in sell_arr)
    return f"{base_str}|{buy_str}|{pick_str}|{sell_str}"
