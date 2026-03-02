---
name: "bt-controllor"
description: "Provides project context and coding standards for the bt controllor quant backtesting system. Invoke when user asks about controllor directory, strategy modifications, or any code changes in the backtesting system."
---

# BT Controllor 项目规范

## 项目概述
- **类型**: 量化回测系统
- **核心功能**: 多策略股票回测、参数优化、绩效分析
- **技术栈**: Python + Numba JIT + Polars + Pandas + 多进程

## 目录结构

| 文件 | 职责 |
|------|------|
| bt2.py | 主入口 (bt_all/batch, bt_one/single) |
| chain.py | 策略编排、多进程执行 |
| strategy.py | 核心策略: pick/buy/sell/settle_amount |
| stock_data.py | 股票数据获取、技术指标计算 |
| stock_picker.py | 实时选股 (复用strategy逻辑) |
| stock_calendar.py | 交易日历管理 |
| param_config.py | 参数配置中心 (13个参数) |
| param_generator.py | 参数组合迭代器 |
| dto.py | 数据模型 (HoldStock, BacktestResult) |
| local_cache.py | 本地文件缓存 (Feather/Parquet) |
| logger_config.py | 日志配置 |

## 策略参数格式

```
持仓数量|买入参数7个|排序方向|卖出参数4个
示例: 1|2,-1,7,-1,14,-1,3|0|-10,5,12,6
```

**买入参数(7个)**: 连涨天数下限,上限,3日涨幅下限,上限,5日涨幅下限,上限,当日涨幅上限
**排序方向**: 0=成交量升序(冷门股), 1=成交量降序(热门股)
**卖出参数(4个)**: 止损率%,持仓天数,目标涨幅%,回撤止盈率%

**内置固定条件**: 量比>1, 10天内无涨停

## 编码规范

1. **函数注解**: 所有公开函数必须添加docstring，说明功能、参数、返回值
2. **极简主义**: 只实现必要功能，不添加不必要代码
3. **类型注解**: 鼓励使用Python类型注解
4. **内聚性原则**: 同一概念的数据/逻辑应内聚在同一模块
   - 结果字段定义集中在 dto.py
   - 参数定义集中在 param_config.py
   - 避免同一字段在多个文件硬编码

## 禁止修改的核心逻辑

1. **Numba JIT函数** (strategy.py):
   - `_filter_numba`: 选股筛选
   - `_calc_buy_counts_numba`: 买入股数计算
   - `_check_sell_numba`: 卖出条件检查

2. **缓存机制**:
   - `PickCache` (进程内选股缓存)
   - `LocalCache` (本地文件缓存)

3. **参数解析/构建函数** (param_config.py):
   - `parse_strategy_string()`
   - `build_strategy_string()`

## 新增参数流程

如需新增策略参数，必须同步修改:
1. `param_config.py`: `ParamRanges` 类添加新参数范围
2. `param_config.py`: `parse_strategy_string()` 解析新参数
3. `param_config.py`: `build_strategy_string()` 构建新参数
4. `strategy.py`: `Strategy.__init__()` 读取新参数
5. `strategy.py`: 相关初始化函数使用新参数

## 性能优化要点

1. **Numba JIT**: 核心计算函数使用 `@njit(cache=True)`
2. **两级选股缓存**: 筛选缓存 + 排序缓存
3. **参数顺序优化**: param_generator中买入参数放前面提高缓存命中率
4. **内存优化**: HoldStock使用 `__slots__`, Polars替代部分Pandas

## 重要说明

- **无断点续跑**: 缓存都是运行时缓存，进程中断后无法恢复
- **多进程缓存独立**: 每个进程维护自己的PickCache实例

## 测试规范

### 需要测试的场景（大改动/新功能）

对于以下类型的修改，必须使用专用参数进行测试：

**测试参数**:
```python
param = {
    "strategy": [{
        "base_param_arr": [10000000, 1],
        "buy_param_arr": [2, -1, 7, -1, 14, -1, 3],
        "pick_param_arr": [0],
        "sell_param_arr": [-10, 5, 12, 6],
        "debug": True
    }],
    "date_arr": [(20250101, 20250201)],
    "chain_debug": True,
    "run_year": True,
    "processor_count": 1
}
```

**必须验证的字段**:
- 选股信号数 > 0
- 1日/3日/5日胜率、盈亏比、平均收益有值且格式正确

### 不需要测试的场景（简单改动）

- 注释修改
- 日志输出调整
- 变量重命名（无逻辑变化）
- 代码格式化
