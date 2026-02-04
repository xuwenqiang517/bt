from datetime import date, datetime
from importlib.machinery import FrozenImporter
import akshare as ak
from LocalCache import LocalCache as lc
import pandas as pd

class StockCalendar:
    def __init__(self):
        today=date.today().strftime("%Y%m%d")
        # self.holidays = set()
        cache_file_name="stock_calendar_"+today
        cache=lc()
        cache.clean(prefix="stock_calendar_",ignore=[cache_file_name])
        self.df = cache.get(cache_file_name)
        self.start_index = 0
        if self.df is None:
            df=ak.tool_trade_date_hist_sina()
            # 1. 统一转换为datetime类型（兼容原始数据为字符串/时间戳的场景）
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            # 2. 批量格式化日期为 %Y%m%d 字符串（向量化操作，效率远高于循环）
            self.df=pd.DataFrame(df["trade_date"].dt.strftime("%Y%m%d").tolist(),columns=["trade_date"])
            cache.set(cache_file_name, self.df)
            print(self.df)
        else:
            print("load from cache")
    
    def show(self):
        print(self.df)

    def start(self,start_date)->str:
        """
        寻找第一个大于等于 start_date 的交易日，并记录索引
        """
        for i, row in self.df.iterrows():
            if row["trade_date"] >= start_date:
                self.start_index = i
                return row["trade_date"]
        return None

    
    def next(self):
        """
        获取下一个交易日，并更新索引
        """
        if self.start_index < len(self.df):
            self.start_index += 1
            trade_date = self.df.iloc[self.start_index]["trade_date"]
            return trade_date
        else:
            return None


        

if __name__ == "__main__":
    sc=StockCalendar()
    # sc.show()
    date=sc.start("20240106")
    print(f"start date: {date}")

    for _ in range(10):
        date=sc.next()
        print(date)
    


