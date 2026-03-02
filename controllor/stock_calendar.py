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

        # 预转换为列表，避免Pandas iloc开销
        self._date_list = self.df["trade_date"].tolist()
        self._total_dates = len(self._date_list)

        # 预计算日期范围，用于快速判断日期是否在范围内
        self._min_date = self._date_list[0] if self._date_list else 0
        self._max_date = self._date_list[-1] if self._date_list else 0

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

    def gap(self, start: int, end: int) -> int:
        """
        计算 start 到 end 之间的交易日数量，使用预计算的日期范围快速判断
        """
        # 快速范围检查（避免两次dict查找）
        if start < self._min_date or start > self._max_date or end < self._min_date or end > self._max_date:
            return -1

        start_index = self.date_to_index.get(start, -1)
        if start_index == -1:
            return -1

        end_index = self.date_to_index.get(end, -1)
        if end_index == -1 or start_index > end_index:
            return -1

        return end_index - start_index + 1

    def get_date(self, idx: int) -> int:
        """
        根据索引获取日期，使用预转换的列表实现 O(1) 时间复杂度
        比 df.iloc 快 5-10 倍
        """
        if 0 <= idx < self._total_dates:
            return self._date_list[idx]
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
                return self._date_list[idx - 1]
        else:
            for i in range(self._total_dates - 1, -1, -1):
                date_val = self._date_list[i]
                if date_val < today_int:
                    return date_val
        return today_int

    def get_date_arr(self, period_days: int = 20) -> list:
        """生成固定交易天数的周期数组

        Args:
            period_days: 每个周期的交易天数，默认20天

        Returns:
            list: [[start_date, end_date], ...]
        """
        start_date = 20240701
        end_date = 20260301

        # 找到起始和结束索引
        start_idx = self.start(start_date)
        end_idx = self.start(end_date)

        if start_idx == -1 or end_idx == -1:
            print(f"警告: 无法找到日期范围 {start_date} - {end_date}")
            return []

        result = []
        current_idx = start_idx

        while current_idx <= end_idx:
            # 计算当前周期的结束索引
            period_end_idx = current_idx + period_days - 1

            # 如果超出范围，停止
            if period_end_idx > end_idx:
                break

            start = self.get_date(current_idx)
            end = self.get_date(period_end_idx)
            result.append([start, end])

            # 移动到下一个周期（不重叠）
            current_idx = period_end_idx + 1

        # 打印周期信息
        print(f"\n周期划分结果 (每{period_days}个交易日):")
        print(f"总周期数: {len(result)}")
        print(f"日期范围: {start_date} - {end_date}")
        if result:
            print(f"第一个周期: {result[0][0]} - {result[0][1]}")
            print(f"最后一个周期: {result[-1][0]} - {result[-1][1]}")
            # 计算丢弃的日期
            last_end_idx = self.start(result[-1][1])
            discarded = end_idx - last_end_idx
            if discarded > 0:
                discarded_start = self.get_date(last_end_idx + 1)
                discarded_end = self.get_date(end_idx)
                print(f"丢弃的日期: {discarded}个交易日 ({discarded_start} - {discarded_end})")
        print()

        return result


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
    


