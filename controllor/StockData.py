from datetime import date
import akshare as ak
from regex import P
from LocalCache import LocalCache as lc
import pandas as pd
import requests
import numpy as np
from tqdm import tqdm

class StockData:
    def __init__(self):
        today=date.today().strftime("%Y%m%d")
        cache=lc()
        stock_list=self.get_stock_list(today,cache)
        print(stock_list)
        stock_data=self.get_stock_data(stock_list,today,cache)
        print(len(stock_data))
        print(stock_data["600000"])
        
    
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
            
            stock_list = stock_sh._append(stock_sz, ignore_index=True)

            stock_list = stock_list[~stock_list["代码"].str.startswith(("688", "300", "301"))]
            stock_list = stock_list[~stock_list["名称"].str.contains("ST")]
            stock_list.rename(columns={"代码": "code", "名称": "name"}, inplace=True)
            cache_file_name="stock_list_"+date.today().strftime("%Y%m%d")
            cache.set(cache_file_name, stock_list)
        return stock_list

    def get_stock_data(self,stock_list,today,cache):
        cache_file_name="all_stock_data_"+today
        cache.clean(prefix="all_stock_data_",ignore=[cache_file_name])
        stock_data = cache.get(cache_file_name)

        if stock_data is None:
            stock_data = {}
            for index, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Fetching stock data"):
                code = row["code"]
                try:
                    cache_file_name=f"stock_data_{code}_{today}"
                    cache.clean(prefix=f"stock_data_{code}_",ignore=[cache_file_name])
                    df=cache.get(cache_file_name)
                    if df is None or df.empty:
                        df = self.get_stock_data_tx(code)
                        if df is not None and not df.empty:
                            cache.set(cache_file_name, df)
                            stock_data[code] = df
                except Exception as e:
                    print(f"Error fetching data for {code}: {e}")
        return stock_data


    def get_stock_data_tx(self,code) -> pd.DataFrame:
        print(f"获取股票数据 {code}")
        market = "sh" if code.startswith("6") else "sz"
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,640,qfq,1"
        headers = {"Referer": "http://finance.qq.com/mp/"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") == 0 and data.get("data") and data["data"].get(f"{market}{code}"):
            items = data["data"][f"{market}{code}"].get("qfqday", [])
            print(f"获取股票数据 {code} 成功，共 {len(items)} 条数据")
            items = [row[:6] for row in items]
            df = pd.DataFrame(items, columns=["date", "open", "close", "high", "low", "volume"])
            df["code"] = code
            df["date"] = df["date"].str.replace("-", "")
            df["open"] = df["open"].astype(float)
            df["close"] = df["close"].astype(float)
            df["high"] = df["high"].astype(float)
            df["low"] = df["low"].astype(float)
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
            df["change_pct"] = (df["close"] - df["close"].shift(1)) / df["close"].shift(1) * 100
            df["change_pct"] = df["change_pct"].round(2).fillna(0)
            df = df[["code", "date", "open", "close", "high", "low", "volume", "change_pct"]]
            df = self.calc_tech(df)
            return df
        else:
            print(f"获取股票数据 {market}{code} 失败 data:{data}")
            return pd.DataFrame()
        
    
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
    