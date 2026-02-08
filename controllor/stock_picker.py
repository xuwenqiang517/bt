from datetime import date
import pandas as pd
import numpy as np
from typing import NamedTuple

from StockCalendar import StockCalendar as sc
from StockData import StockData as sd


class StockPicker:
    def __init__(self, config_str: str):
        """
        选股器初始化
        config_str: 格式 "持仓数|连涨天数|3日涨幅最低|3日涨幅最高|5日涨幅最低|5日涨幅最高"
        例如: "4|2,5,10,8,15" 表示持仓4只，其他参数对应
        """
        self.config_str = config_str
        parts = config_str.split("|")
        self.max_hold = int(parts[0])
        self.buy_params = list(map(int, parts[1].split(",")))
        
        buy_up_day_min = self.buy_params[0]
        buy_day3_min = self.buy_params[1]
        buy_day3_max = self.buy_params[2]
        buy_day5_min = self.buy_params[3]
        buy_day5_max = self.buy_params[4]

        self._filter_params = {
            "连涨天数≥": buy_up_day_min,
            "3日涨幅": f"{buy_day3_min}% ~ {buy_day3_max}%",
            "5日涨幅": f"{buy_day5_min}% ~ {buy_day5_max}%"
        }

        def filter_func(df: pd.DataFrame) -> np.ndarray:
            col_consecutive = df["consecutive_up_days"].values
            col_change3d = df["change_3d"].values
            col_change5d = df["change_5d"].values
            return (
                (col_consecutive >= buy_up_day_min)
                & (col_change3d >= buy_day3_min)
                & (col_change3d <= buy_day3_max)
                & (col_change5d >= buy_day5_min)
                & (col_change5d <= buy_day5_max)
            )
        self._pick_filter = filter_func

        max_hold = self.max_hold
        def sorter_func(df: pd.DataFrame) -> pd.DataFrame:
            n = min(max_hold, len(df))
            if n <= 0:
                return pd.DataFrame()
            vol_rank_values = df["vol_rank"].values
            top_n_indices = np.argpartition(vol_rank_values, n-1)[:n]
            sorted_indices = top_n_indices[np.argsort(vol_rank_values[top_n_indices])]
            return df.iloc[sorted_indices].reset_index(drop=True)
        self._pick_sorter = sorter_func

    def pick(self, target_date: str = None) -> pd.DataFrame:
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

        today_stock_df = self.data.get_data_by_date(target_date)
        if today_stock_df is None or today_stock_df.empty:
            print(f"❌ 没有找到日期 {target_date} 的股票数据")
            return pd.DataFrame()

        print(f"📈 全部股票数量: {len(today_stock_df)}")

        mask = self._pick_filter(today_stock_df)
        filtered_stocks = today_stock_df[mask]

        if filtered_stocks.empty:
            print(f"😢 没有符合筛选条件的股票")
            return pd.DataFrame()

        print(f"🔍 筛选后股票数量: {len(filtered_stocks)}")

        result = self._pick_sorter(filtered_stocks)
        print(f"✅ 选出 {len(result)} 只股票（按vol_rank倒序）:\n")

        print(f"{'代码':<10} {'收盘':<10} {'次日开盘':<10} {'连涨':<8} {'3日涨幅':<12} {'5日涨幅':<12} {'vol_rank':<10}")
        print("-" * 75)

        for idx, row in result.iterrows():
            print(f"{row['code']:<10} {row['close']:<10.2f} {row['next_open']:<10.2f} "
                  f"{row['consecutive_up_days']:<8} {row['change_3d']:<12.2f}% {row['change_5d']:<12.2f}% {row['vol_rank']:<10}")
        
        return result

    def _get_last_trade_date(self, today: str = None) -> str:
        """获取最后一个交易日（今天或之前的最近交易日）"""
        if today is None:
            today = date.today().strftime("%Y%m%d")
        all_dates = self.calendar.df["trade_date"].tolist()
        for d in reversed(all_dates):
            if d <= today:
                return d
        return all_dates[0] if all_dates else today


def main():
    """手动运行选股"""
    import sys
    if len(sys.argv) > 1:
        config = sys.argv[1]
    else:
        config = "4|2,5,10,8,15"
    
    picker = StockPicker(config)
    picker.pick()


if __name__ == "__main__":
    main()
