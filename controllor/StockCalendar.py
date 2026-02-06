from datetime import date
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
        # else:
            # print("load from cache")
        
        # 构建日期到索引的映射字典，用于O(1)时间复杂度的gap计算
        self.date_to_index = {}
        for idx, row in self.df.iterrows():
            self.date_to_index[row["trade_date"]] = idx

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

    def gap(self,start:str,end:str)->int:
        """
        计算 start 到 end 之间的交易日数量 用O1的时间复杂度 可以提前在init里面构造最简单高效的数据结构
        """
        if start not in self.date_to_index or end not in self.date_to_index:
            return -1
        
        start_index = self.date_to_index[start]
        end_index = self.date_to_index[end]
        
        if start_index > end_index:
            return -1
        
        return end_index - start_index + 1
        

if __name__ == "__main__":
    sc=StockCalendar()
    date=sc.start("20240106")
    print(f"start date: {date}")

    for _ in range(10):
        date=sc.next()
        print(date)
    


