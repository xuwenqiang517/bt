from datetime import date
import akshare as ak
from local_cache import LocalCache as lc
import pandas as pd

class StockCalendar:
    def __init__(self):
        today=date.today().strftime("%Y%m%d")
        # self.holidays = set()
        cache_file_name="stock_calendar_int_"+today
        cache=lc()
        cache.clean(prefix="stock_calendar_int_",ignore=[cache_file_name])
        self.df = cache.get(cache_file_name)
        self.start_index = 0
        if self.df is None:
            df=ak.tool_trade_date_hist_sina()
            # 1. 统一转换为datetime类型（兼容原始数据为字符串/时间戳的场景）
            df["trade_date"] = pd.to_datetime(df["trade_date"])
            # 2. 批量格式化日期为 %Y%m%d 整数（向量化操作，效率远高于循环）
            self.df=pd.DataFrame(df["trade_date"].dt.strftime("%Y%m%d").astype(int).tolist(),columns=["trade_date"])
            cache.set(cache_file_name, self.df)
            print(self.df)
        # else:
            # print("load from cache")
        
        # 构建日期到索引的映射字典，用于O(1)时间复杂度的gap计算
        self.date_to_index = {}
        for idx, row in self.df.iterrows():
            self.date_to_index[row["trade_date"]] = idx

    def start(self, start_date: int) -> int:
        """
        寻找第一个大于等于 start_date 的交易日索引
        使用 date_to_index 字典实现 O(1) 查找
        """
        if start_date in self.date_to_index:
            idx = self.date_to_index[start_date]
            self.start_index = idx
            return idx

        year = start_date // 10000
        month = (start_date // 100) % 100
        day = start_date % 100

        current = start_date
        while current > 0:
            year = current // 10000
            month = (current // 100) % 100
            day = current % 100
            next_day = day + 1
            if next_day > 31:
                next_day = 1
                month += 1
            if month > 12:
                month = 1
                year += 1
            current = year * 10000 + month * 100 + next_day

            if current in self.date_to_index:
                idx = self.date_to_index[current]
                self.start_index = idx
                return idx

        return -1

    def next(self, current_idx: int = None) -> int:
        """
        获取下一个交易日索引
        """
        if current_idx is None:
            current_idx = self.start_index
        if current_idx < len(self.df) - 1:
            next_idx = current_idx + 1
            self.start_index = next_idx
            return next_idx
        return -1

    def gap(self,start:int,end:int)->int:
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

    def get_date(self, idx: int) -> int:
        """
        根据索引获取日期，使用 df.iloc 实现 O(1) 时间复杂度
        """
        if 0 <= idx < len(self.df):
            return self.df.iloc[idx]["trade_date"]
        return None

    
    def get_last_trade_date(self) -> int:
        """获取最后一个交易日
        如果今天是交易日且已过15:00，返回今天
        否则返回上一个交易日
        """
        from datetime import datetime
        now = datetime.now()
        today_int = int(now.strftime("%Y%m%d"))

        if today_int in self.date_to_index:
            if now.hour >= 15:
                return today_int
            idx = self.date_to_index[today_int]
            if idx > 0:
                return self.df.iloc[idx - 1]["trade_date"]
        else:
            for i in range(len(self.df) - 1, -1, -1):
                date_val = self.df.iloc[i]["trade_date"]
                if date_val < today_int:
                    return date_val
        return today_int

    def get_date_arr(self)->list:
        return [
            [20240701,20240801]
            ,[20240801,20240901]
            ,[20240901,20241001]
            ,[20241001,20241101]
            ,[20241101,20241201]
            ,[20241201,20250101]
            ,[20250101,20250201]
            ,[20250201,20250301]
            ,[20250301,20250401]
            ,[20250401,20250501]
            ,[20250501,20250601]
            ,[20250601,20250701]
            ,[20250701,20250801]
            ,[20250801,20250901]
            ,[20250901,20251001]
            ,[20251001,20251101]
            ,[20251101,20251201]
            ,[20251201,20260101]
            ,[20260101,20260201]
        ]


if __name__ == "__main__":
    sc=StockCalendar()
    index=sc.start(20260101)
    print(index)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    index=sc.next(index)
    date=sc.get_date(index)
    print(date)
    # for _ in range(10):
    #     date=sc.next()
    #     print(date)
    


