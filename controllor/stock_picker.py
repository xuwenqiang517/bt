from datetime import date
import polars as pl

from stock_calendar import StockCalendar as sc
from stock_data import StockData as sd
from strategy_impl import UpStrategy


class StockPicker:
    def __init__(self, config_str: str):
        """
        选股器初始化
        config_str: 格式 "最大持仓数|买入参数|卖出参数"
        例如: "3|3,10,15|-15,5,7,6" 表示最大持仓3只，买入参数为[3,10,15]，卖出参数为[-15,5,7,6]
        """
        self.config_str = config_str
        parts = config_str.split("|")
        
        # 解析参数
        self.max_hold = int(parts[0])
        self.buy_params = list(map(int, parts[1].split(",")))
        self.sell_params = list(map(int, parts[2].split(",")))
        
        # 创建基础参数
        self.base_params = [10000000, self.max_hold]  # 初始资金和最大持仓数
        
        # 创建UpStrategy实例，复用其筛选和排序逻辑
        self.strategy = UpStrategy(
            base_param_arr=self.base_params,
            sell_param_arr=self.sell_params,
            buy_param_arr=self.buy_params,
            debug=False
        )
        
        # 构建筛选条件描述
        self._filter_params = {
            "连涨天数≥": self.buy_params[0],
            "3日涨幅>": f"{self.buy_params[1]}%",
            "5日涨幅>": f"{self.buy_params[2]}%",
            "当日涨幅<": "5%"
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

        # 将target_date转换为整数，因为get_data_by_date只接受int类型参数
        today_stock_df = self.data.get_data_by_date(int(target_date))
        if today_stock_df is None or today_stock_df.is_empty():
            print(f"❌ 没有找到日期 {target_date} 的股票数据")
            return pl.DataFrame()

        print(f"📈 全部股票数量: {len(today_stock_df)}")

        # 使用策略的筛选函数
        mask = self.strategy._pick_filter(today_stock_df)
        filtered_stocks = today_stock_df.filter(mask)

        if filtered_stocks.is_empty():
            print(f"😢 没有符合筛选条件的股票")
            return pl.DataFrame()

        print(f"🔍 筛选后股票数量: {len(filtered_stocks)}")

        # 限制结果数量并按成交额排序
        n = min(self.max_hold, len(filtered_stocks))
        if n > 0:
            # 按成交额降序排序并取前n只
            result = filtered_stocks.sort("amount", descending=True).head(n)
            print(f"✅ 选出 {n} 只股票（按成交额排序）:\n")

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
        config = "3|3,10,15|-15,5,7,6"  # 默认配置
    
    picker = StockPicker(config)
    picker.pick()


if __name__ == "__main__":
    main()
