from calendar import c
from datetime import date,datetime
import time
import akshare as ak
from regex import T
from LocalCache import LocalCache as lc
import pandas as pd
import numpy as np
from tqdm import tqdm
import requests
from typing import NamedTuple

StockDataTuple = NamedTuple("StockDataTuple", [
    ("code", str),
    ("open", float),
    ("high", float),
    ("low", float),
    ("close", float),
    ("change_pct", float)
])


class StockData:
    def __init__(self):
        today=date.today().strftime("%Y%m%d")
        if datetime.now().hour < 15:
            today=(date.today()-pd.Timedelta(days=1)).strftime("%Y%m%d")
        cache=lc()
        print(f"1. 获取股票列表")
        stock_list_df=self.get_stock_list(today,cache)
        print(f"2. 获取股票数据")
        stock_data_df=self.get_stock_data(stock_list_df,today,cache)
        # 一层dict，日期+pd
        print(f"3. 构建数据索引")
        self.date_df_dict=self.convert(stock_data_df)
        # 二层dict，日期+code+list
        print(f"4. 构建数据双重索引")
        start_time=time.time()
        self.data_index=self.build_index(self.date_df_dict)
        print(f"5. 数据索引构建完成 耗时 {time.time()-start_time}ms")
        # print(len(stock_data_df))
        # print(stock_data_df.tail())

    def get_data_by_date(self,today)->pd.DataFrame:
        if today in self.date_df_dict:
            return self.date_df_dict[today]
        else:
            print(f"没有找到日期 {today} 的股票数据")
            return pd.DataFrame()
    def get_data_by_date_code(self,today,code)->pd.DataFrame:
        if today in self.data_index:
            if code in self.data_index[today]:
                return self.data_index[today][code]
            else:
                print(f"没有找到日期 {today} 股票代码 {code} 的股票数据")
                return pd.DataFrame()
        else:
            print(f"没有找到日期 {today} 的股票数据")
            return pd.DataFrame()


    def convert(self,stock_data_df)->dict:
        # 按日期构建索引，key为日期，值为该日期的所有股票数据df，性能要最好
        date_dict = {}
        for trade_date, group in stock_data_df.groupby("date", sort=False):
            date_dict[trade_date] = group.reset_index(drop=True)
        return date_dict
    
    def build_index(self,date_df_dict)->dict:
        """
        优化版：无groupby，单行直接映射为命名元组
        适用于单日数据中 code 唯一的场景
        """
        index_dict = {}
        for date, daily_df in tqdm(date_df_dict.items(), desc="构建索引"):
            # 核心：直接用字典推导式，一行一个股票，无需分组
            code_dict = {
                row.code: StockDataTuple(
                    code=row.code,
                    open=row.open,
                    high=row.high,
                    low=row.low,
                    close=row.close,
                    change_pct=row.change_pct
                )
                # itertuples 比 iterrows 快得多，index=False 跳过索引列
                for row in daily_df.itertuples(index=False, name="StockRow")
            }
            index_dict[date] = code_dict
        return index_dict
    
    def get_stock_list(self,today,cache):
        cache_file_name="stock_list_"+today
        cache.clean(prefix="stock_list_",ignore=[cache_file_name])
        stock_list = cache.get(cache_file_name)

        if stock_list is None:
            # 获取股票列表
            stock_sh = ak.stock_info_sh_name_code(symbol="主板A股")
            stock_sh = stock_sh[["证券代码", "证券简称"]]
            stock_sh.columns = ["代码", "名称"]

            stock_sz = ak.stock_info_sz_name_code(symbol="A股列表")
            stock_sz = stock_sz[["A股代码", "A股简称"]]
            stock_sz.columns = ["代码", "名称"]
            
            stock_list = pd.concat([stock_sh, stock_sz], ignore_index=True)

            stock_list = stock_list[~stock_list["代码"].str.startswith(("688", "300", "301"))]
            stock_list = stock_list[~stock_list["名称"].str.contains("ST")]
            stock_list.rename(columns={"代码": "code", "名称": "name"}, inplace=True)
            cache_file_name="stock_list_"+date.today().strftime("%Y%m%d")
            cache.set(cache_file_name, stock_list)
        return stock_list

    def get_stock_data(self,stock_list,today,cache):
        all_cache_file_name="all_stock_data_"+today
        cache.clean(prefix="all_stock_data_",ignore=[all_cache_file_name])
        stock_data = cache.get(all_cache_file_name)

        if stock_data is None:
            print(f"缓存取股票数据 {all_cache_file_name} 失败，尝试从_tx获取")
            stock_data = pd.DataFrame()
            for index, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Fetching stock data"):
                code = row["code"]
                try:
                    cache_file_name=f"stock_data_{code}_{today}"
                    df=cache.get(cache_file_name)
                    if df is None or df.empty:
                        items = self.get_stock_data_tx(code)
                        if items is not None and len(items)>0:
                            df = pd.DataFrame(items, columns=["date", "open", "close", "high", "low", "volume"])
                            cache.set(cache_file_name, df)
                    if df is not None and not df.empty:
                        df["code"] = code
                        df["date"] = df["date"].str.replace("-", "")
                        df["open"] = df["open"].astype(float)
                        df["close"] = df["close"].astype(float)
                        df["high"] = df["high"].astype(float)
                        df["low"] = df["low"].astype(float)
                        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
                        df["change_pct"] = (df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100
                        df["change_pct"] = df["change_pct"].round(2).fillna(0)
                        df["next_open"] = df["open"].shift(-1)
                        df["next_close"] = df["close"].shift(-1)
                        df["next_high"] = df["high"].shift(-1)
                        df["next_low"] = df["low"].shift(-1)
                        df = df[["code", "date", "open", "close", "high", "low", "volume", "change_pct", "next_open", "next_close"]]
                        df = self.calc_tech(df)
                        stock_data = pd.concat([stock_data, df], ignore_index=True)
                except Exception as e:
                    print(f"Error fetching data for {code}: {e}")
            cache.set(all_cache_file_name,stock_data)
            # cache.clean(prefix="stock_data_")
        return stock_data


    def get_stock_data_tx(self,code):
        market = "sh" if code.startswith("6") else "sz"
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,640,qfq,1"
        headers = {"Referer": "http://finance.qq.com/mp/"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") == 0 and data.get("data") and data["data"].get(f"{market}{code}"):
            items = data["data"][f"{market}{code}"].get("qfqday", [])
            return [row[:6] for row in items]
        return None
        
    
    def calc_tech(self,df: pd.DataFrame) -> pd.DataFrame:
        # ma5 = df["close"].rolling(window=5).mean().round(2)
        # ma10 = df["close"].rolling(window=10).mean().round(2)
        # ma20 = df["close"].rolling(window=20).mean().round(2)
        # vol_ma5 = df["volume"].rolling(window=5).mean().round(2)
        # vol_ma10 = df["volume"].rolling(window=10).mean().round(2)
        # vol_ma20 = df["volume"].rolling(window=20).mean().round(2)
        df["change_3d"] = df["change_pct"].rolling(window=3).sum().round(2)
        df["change_5d"] = df["change_pct"].rolling(window=5).sum().round(2)
        df["change_10d"] = df["change_pct"].rolling(window=10).sum().round(2)
        
        df["consecutive_up_days"] = self.calc_up_days(df["close"].values)
        df["vol_rank"] = df["volume"].rank(ascending=False, method="min").astype(int)
        return df
    
    def calc_up_days(self,close_prices: np.ndarray) -> np.ndarray:
        consecutive_up = []
        count = 0
        for i in range(len(close_prices)):
            if i == 0:
                consecutive_up.append(0)
                count = 0
            else:
                if close_prices[i] > close_prices[i - 1]:
                    count += 1
                else:
                    count = 0
                consecutive_up.append(count)
        return consecutive_up
    
if __name__ == "__main__":
    sd=StockData()
    print("测试按日期获取数据")
    print(sd.get_data_by_date("20240108"))
    print("测试按日期和代码获取数据")
    print(sd.get_data_by_date_code("20240108","600000"))

