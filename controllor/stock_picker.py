from datetime import date
import polars as pl
import numpy as np

from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
from strategy import Strategy, _filter_numba
from param_config import parse_strategy_string, DEFAULT_PARAM_RANGES


class StockPicker:
    def __init__(self, config_str: str):
        """
        选股器初始化
        config_str: 格式 "持仓数量|买入参数|排序方向|卖出参数"
        例如: "1|2,-1,7,-1,14,-1,3|0|-12,4,2,7"
        买入参数: 连涨天数下限,连涨天数上限,3日涨幅下限,3日涨幅上限,5日涨幅下限,5日涨幅上限,当日涨幅上限 (7个参数)
        排序方向: 0=成交量升序(冷门股), 1=成交量降序(热门股)
        注意: 量比(>1)、涨停条件(0次)已内置，不再作为参数；日内振幅参数已移除
        """
        self.config_str = config_str

        # 使用param_config中的函数解析策略字符串
        parsed = parse_strategy_string(config_str)
        self.base_params = parsed["base_param_arr"]
        self.buy_params = parsed["buy_param_arr"]
        self.pick_params = parsed["pick_param_arr"]
        self.sell_params = parsed["sell_param_arr"]
        self.max_hold = self.base_params[1]

        # 创建Strategy实例，复用其筛选和排序逻辑
        self.strategy = Strategy(
            base_param_arr=self.base_params,
            sell_param_arr=self.sell_params,
            buy_param_arr=self.buy_params,
            pick_param_arr=self.pick_params,
            debug=False
        )

        # 构建筛选条件描述（7个买入参数）
        sort_desc = self.pick_params[0] if len(self.pick_params) > 0 else 0
        self._filter_params = {
            "连涨天数≥": self.buy_params[0] if self.buy_params[0] > 0 else "不限",
            "连涨天数≤": self.buy_params[1] if len(self.buy_params) > 1 and self.buy_params[1] > 0 else "不限",
            "3日涨幅>": f"{self.buy_params[2]}%" if len(self.buy_params) > 2 and self.buy_params[2] > 0 else "不限",
            "3日涨幅<": f"{self.buy_params[3]}%" if len(self.buy_params) > 3 and self.buy_params[3] > 0 else "不限",
            "5日涨幅>": f"{self.buy_params[4]}%" if len(self.buy_params) > 4 and self.buy_params[4] > 0 else "不限",
            "5日涨幅<": f"{self.buy_params[5]}%" if len(self.buy_params) > 5 and self.buy_params[5] > 0 else "不限",
            "当日涨幅<": f"{self.buy_params[6]}%" if len(self.buy_params) > 6 and self.buy_params[6] > 0 else "不限",
            "涨停条件": "10天0涨停(内置)",
            "量比>": "1(内置)",
            "排序": "成交量降序（热门股）" if sort_desc == 1 else "成交量升序（冷门股）"
        }

    def pick(self, target_date: str = None) -> pl.DataFrame:
        """
        选出指定日期符合条件的股票
        target_date: 目标日期，默认为明天（如果当前时间在15点前则是今天）
        """
        self.data = sd()
        self.calendar = sc()
        
        if target_date is None:
            today = date.today().strftime("%Y%m%d")
            target_date = self._get_last_trade_date(today)
        
        print(f"\n{'='*50}")
        print(f"📅 目标日期: {target_date}")
        print(f"⚙️  配置参数: {self.config_str}")
        print(f"📊 筛选条件:")
        for k, v in self._filter_params.items():
            print(f"   • {k}: {v}")
        print(f"{'='*50}\n")

        # 获取当天所有股票的numpy数据
        today_int = int(target_date)
        numpy_data = self.data.get_numpy_data_by_date(today_int)
        if numpy_data is None:
            print(f"❌ 没有找到日期 {target_date} 的股票数据")
            return pl.DataFrame()

        # 将numpy数据转换为polars DataFrame
        today_stock_df = pl.DataFrame({
            'code': numpy_data['code'],
            'open': numpy_data['open'],
            'close': numpy_data['close'],
            'high': numpy_data['high'],
            'low': numpy_data['low'],
            'volume': numpy_data['volume'],
            'amount': numpy_data['amount'],
            'change_pct': numpy_data['change_pct'],
            'change_3d': numpy_data['change_3d'],
            'change_5d': numpy_data['change_5d'],
            'consecutive_up_days': numpy_data['consecutive_up_days'],
            'limit_up_count_10d': numpy_data['limit_up_count_10d'],
            'volume_ratio': numpy_data['volume_ratio'],
            'price_limit_status': numpy_data['price_limit_status']
        })

        print(f"📈 全部股票数量: {len(today_stock_df)}")

        # 使用策略的筛选函数（传入numpy数组）
        buy_params = self.buy_params
        mask = _filter_numba(
            today_stock_df['consecutive_up_days'].to_numpy(),
            today_stock_df['change_3d'].to_numpy(),
            today_stock_df['change_5d'].to_numpy(),
            today_stock_df['change_pct'].to_numpy(),
            today_stock_df['limit_up_count_10d'].to_numpy(),
            today_stock_df['volume_ratio'].to_numpy(),
            buy_params[0],  # buy_up_day_min
            buy_params[1],  # buy_up_day_max
            buy_params[2],  # buy_day3_min
            buy_params[3],  # buy_day3_max
            buy_params[4],  # buy_day5_min
            buy_params[5],  # buy_day5_max
            buy_params[6] if len(buy_params) > 6 else 5  # change_pct_max
        )
        filtered_stocks = today_stock_df.filter(pl.Series(mask))

        if filtered_stocks.is_empty():
            print(f"😢 没有符合筛选条件的股票")
            return pl.DataFrame()

        print(f"🔍 筛选后股票数量: {len(filtered_stocks)}")

        # 限制结果数量
        n = min(self.max_hold, len(filtered_stocks))
        if n > 0:
            # 数据已按成交量升序排列，直接取前n个
            result = filtered_stocks.head(n)
            print(f"✅ 选出 {n} 只股票（按成交量升序，冷门股）:\n")

            # 显示结果
            print(f"{'代码':<10} {'收盘':<10} {'连涨':<8} {'3日涨幅':<12} {'5日涨幅':<12} {'当日涨幅':<10}")
            print("-" * 75)

            for row in result.iter_rows(named=True):
                # 处理可能的None值
                code = row.get('code', '') or ''
                close = row.get('close', 0) or 0
                consecutive_up_days = row.get('consecutive_up_days', 0) or 0
                change_3d = row.get('change_3d', 0) or 0
                change_5d = row.get('change_5d', 0) or 0
                change_pct = row.get('change_pct', 0) or 0
                
                print(f"{code:<10} {close:<10.2f} {consecutive_up_days:<8} "
                      f"{change_3d:<12.2f}% {change_5d:<12.2f}% {change_pct:<10.2f}%")
        else:
            result = pl.DataFrame()
            print(f"😢 没有符合条件的股票")
            return result
        
        return result

    def _get_last_trade_date(self, today: str = None) -> str:
        """获取最后一个交易日
        
        逻辑：
        1. 如果今天是交易日
           - 如果当前时间在15:00以后，使用今天的数据
           - 如果当前时间在15:00之前，使用上一个交易日的数据
        2. 如果今天不是交易日，使用上一个交易日的数据
        """
        from datetime import datetime, time
        
        if today is None:
            today = date.today().strftime("%Y%m%d")
        
        all_dates = self.calendar.df["trade_date"].tolist()
        today_int = int(today)
        
        # 检查今天是否是交易日
        is_today_trading_day = today_int in all_dates
        
        # 获取当前时间
        current_time = datetime.now().time()
        # 检查是否在15:00以后
        is_after_1500 = current_time >= time(15, 0)
        
        if is_today_trading_day and is_after_1500:
            # 今天是交易日且已收盘，使用今天的数据
            return today
        else:
            # 今天不是交易日或未收盘，使用上一个交易日的数据
            past_dates = [d for d in all_dates if d < today_int]
            if past_dates:
                return str(max(past_dates))
            return str(all_dates[0]) if all_dates else today


def main():
    """手动运行选股"""
    import sys
    if len(sys.argv) > 1:
        config = sys.argv[1]
    else:
        # config = "1|-1,-1,2,5,0,-1|0|-12,10,5,5"  # 默认配置
        # config = "1|5,9,12,3,0,1|0|-15,15,5,4" # 85胜率
        # config = "1|3,-1,20,4,0,1|0|-8,9,15,10" #87胜率
        # config = "1|-1,8,18,4,0,1|1|-8,5,11,7" 
        # config = "1|3,7,14,2,-1|0|-10,9,7,3" 
        # config="1|3,-1,7,20,14,25,2|0|-10,5,13,10"
        config="1|-1,-1,8,15,15,20,2|0|-9,6,6,5"
        
    picker = StockPicker(config)
    picker.pick()


if __name__ == "__main__":
    main()
