from datetime import date
import akshare as ak
from local_cache import LocalCache as lc
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

    def start(self, start_date: str) -> int:
        """
        寻找第一个大于等于 start_date 的交易日索引
        使用 date_to_index 字典实现 O(1) 查找
        """
        if start_date in self.date_to_index:
            idx = self.date_to_index[start_date]
            self.start_index = idx
            return idx

        year = int(start_date[:4])
        month = int(start_date[4:6])
        day = int(start_date[6:8])

        current = start_date
        while len(current) == 8:
            year = int(current[:4])
            month = int(current[4:6])
            day = int(current[6:8])
            next_day = day + 1
            if next_day > 31:
                next_day = 1
                month += 1
            if month > 12:
                month = 1
                year += 1
            current = f"{year:04d}{month:02d}{next_day:02d}"

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

    def get_date(self, idx: int) -> str:
        """
        根据索引获取日期，使用 df.iloc 实现 O(1) 时间复杂度
        """
        if 0 <= idx < len(self.df):
            return self.df.iloc[idx]["trade_date"]
        return None

    def build_day_array(self, start: str, end: str, splits: int) -> list:
        """
        将 start 到 end 之间的交易日分成 splits 个区间
        返回格式: [[起始日1, 结束日1], [起始日2, 结束日2], ...]
        """
        if start not in self.date_to_index or end not in self.date_to_index:
            return []

        start_idx = self.date_to_index[start]
        end_idx = self.date_to_index[end]

        if start_idx > end_idx:
            return []

        all_dates = self.df.iloc[start_idx:end_idx + 1]["trade_date"].tolist()

        if not all_dates:
            return []

        chunk_size = len(all_dates) // splits
        remainder = len(all_dates) % splits

        result = []
        current = 0
        for i in range(splits):
            if i < remainder:
                size = chunk_size + 1
            else:
                size = chunk_size
            if size <= 0:
                break
            chunk = all_dates[current:current + size]
            if chunk:
                result.append([chunk[0], chunk[-1]])
            current += size

        return result

if __name__ == "__main__":
    sc=StockCalendar()
    day_array=sc.build_day_array("20240701","20260206",6)
    print(day_array)
    # for _ in range(10):
    #     date=sc.next()
    #     print(date)
    


