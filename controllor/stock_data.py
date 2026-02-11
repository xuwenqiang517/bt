from datetime import date,datetime
import akshare as ak
from local_cache import LocalCache as lc
import numpy as np
import polars as pl
from tqdm import tqdm
import requests
# pl 显示所有表头
pl.Config.set_tbl_cols(100)
# pl 显示所有列
pl.Config.set_tbl_rows(10)

class StockData:
    def __init__(self):
        #缓存
        self.use_all_cache=1

        from datetime import timedelta
        today=date.today().strftime("%Y%m%d")
        if datetime.now().hour < 15:
            today=(date.today()-timedelta(days=1)).strftime("%Y%m%d")
        cache=lc()
        print(f"1. 获取股票列表")
        stock_list_df=self.get_stock_list(today,cache)
        print(f"2. 获取股票数据")
        stock_data_df=self.get_stock_data_pl(stock_list_df,today,cache)
        print(f"3. 计算技术指标")
        # 按code分组并应用calc_tech_pl
        grouped = stock_data_df.group_by("code")
        processed_dfs = []
        for code, group in grouped:
            # 确保按日期排序
            sorted_group = group.sort("date")
            tech_df = self.calc_tech_pl(sorted_group)
            processed_dfs.append(tech_df)
        # 合并所有处理后的数据
        processed_stock_data_df = pl.concat(processed_dfs)
        print(f"4. 构建数据索引")
        self.date_df_dict=self.convert(processed_stock_data_df)

    # 选票过滤

    def calc_tech_pl(self,df: pl.DataFrame) -> pl.DataFrame:
        df_tech = df.lazy().with_columns([
                # 计算涨跌幅和下一天开盘价
                ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1) * 100).round(2).fill_null(0).cast(pl.Float32).alias("change_pct"),
                pl.col("open").shift(-1).alias("next_open"),
                # 计算成交金额
                (pl.col("volume") * (pl.col("open") + pl.col("close")) / 2).cast(pl.Int64).alias("amount"),
                # 计算成交量的MA5和MA10
                pl.col("volume").rolling_mean(window_size=5).cast(pl.Int32).alias("ma5_vol"),
                pl.col("volume").rolling_mean(window_size=10).cast(pl.Int32).alias("ma10_vol")
            ]).with_columns([
                # # 计算change_pct是否在3%到5%之间
                # pl.col("change_pct").is_between(3, 5).cast(pl.Int8).alias("change_pct_between_3_5"),
                # # 计算成交量排名
                # pl.col("volume").rank(descending=True, method="min").cast(pl.Int16).alias("volume_rank"),
                # 涨跌幅是float32，直接计算
                pl.col("change_pct").rolling_sum(window_size=3).round(2).cast(pl.Float32).alias("change_3d"),
                pl.col("change_pct").rolling_sum(window_size=5).round(2).cast(pl.Float32).alias("change_5d")
            ]).with_columns([
                pl.Series(
                    name="consecutive_up_days",
                    values=self.calc_up_days(df["close"].to_numpy()),  # 转为numpy数组适配原函数
                    dtype=pl.Int8  # 使用int8存储连续上涨天数
                )
            ]).drop(["ma5_vol", "ma10_vol"]).collect()
            # .with_columns([
            #     # 计算has_limit_up_20d：判断20天内是否有涨停（涨幅>=9.9%）
            #     (pl.col("change_pct").rolling_max(window_size=20) >= 9.9).cast(pl.Int8).alias("has_limit_up_20d"),
            #     # 计算ma_up：当日成交>=1.5 * ma5 & ma5>ma10 为1，否则为0
            #     ((pl.col("volume") >= pl.col("ma5_vol") * 1.5 ) & (pl.col("ma5_vol") > pl.col("ma10_vol"))).cast(pl.Int8).alias("ma_vol_up")
            # ]).with_columns([
            #     # 计算has_limit_up_and_vol_up：同时满足has_limit_up_20d和ma_vol_up两个条件
            #     ((pl.col("has_limit_up_20d") == 1) & (pl.col("ma_vol_up") == 1)).cast(pl.Int8).alias("has_limit_up_and_vol_up"),
            #     # 计算has_limit_up_and_vol_up：同时满足has_limit_up_20d和ma_vol_up两个条件，且change_pct_between_3_5为1
            #     ((pl.col("has_limit_up_20d") == 1) & (pl.col("ma_vol_up") == 1) & (pl.col("change_pct_between_3_5") == 1)).cast(pl.Int8).alias("has_limit_up_and_vol_up_3_5")
            # ]).drop(["ma5_vol", "ma10_vol"]).collect()
        
        return df_tech

    def convert(self,stock_data_df)->dict[int,pl.DataFrame]:
        # 按日期构建索引，key为日期，值为该日期的所有股票数据df，性能要最好
        date_dict = {}
        grouped = stock_data_df.group_by("date")
        for trade_date, group in grouped:
            # 确保trade_date是整数而不是元组
            if isinstance(trade_date, tuple):
                trade_date = trade_date[0]
            if group is None or group.is_empty():
                continue
            # 按成交额从高到低排序，表示当天成交活跃的股票靠前
            group=group.sort("amount" ,descending=True)
            date_dict[trade_date] = group
        return date_dict
    
    def get_data_by_date(self,today: int)->pl.DataFrame:
        if today in self.date_df_dict:
            df = self.date_df_dict[today]
            # 继续以整数形式返回数据
            return df
        else:
            # print(f"没有找到日期 {today} 的股票数据")
            return None
    def get_data_by_date_code(self,today: int,code: int)->pl.DataFrame:
        if today in self.date_df_dict:
            df = self.date_df_dict[today]
            filtered_df = df.filter(pl.col("code") == code)
            return filtered_df
        else:
            # print(f"没有找到日期 {today} 的股票数据")
            return None


    

    
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
            
            # 使用Polars处理
            stock_sh_pl = pl.from_pandas(stock_sh)
            stock_sz_pl = pl.from_pandas(stock_sz)
            
            # 合并数据
            stock_list_pl = pl.concat([stock_sh_pl, stock_sz_pl], rechunk=True)
            
            # 筛选条件
            stock_list_pl = stock_list_pl.filter(
                ~(pl.col("代码").str.starts_with("688") | 
                  pl.col("代码").str.starts_with("300") | 
                  pl.col("代码").str.starts_with("301"))
            )
            stock_list_pl = stock_list_pl.filter(
                ~pl.col("名称").str.contains("ST")
            )
            
            # 重命名列，保持股票代码为字符串类型
            stock_list_pl = stock_list_pl.rename({"代码": "code", "名称": "name"})
            
            # 转换回pandas以保持缓存兼容性
            stock_list = stock_list_pl.to_pandas()
            
            cache_file_name="stock_list_"+date.today().strftime("%Y%m%d")
            cache.set(cache_file_name, stock_list)
        return stock_list


    def get_stock_data_pl(self,stock_list,today,cache):
        all_cache_file_name="all_stock_data_"+today+"_pl"
        cache.clean(prefix="all_stock_data_",ignore=[all_cache_file_name])
        stock_data = None
        if self.use_all_cache:
            stock_data = cache.get_pl(all_cache_file_name)
        if stock_data is None:
            print(f"缓存取股票数据 {all_cache_file_name} 失败，尝试从_tx获取")
            stock_data = pl.DataFrame()
            error_code_arr=['601112', '601399', '601975', '603092', '603175', '603210', '603248', '603262', '603284', '603334', '603352', '603370', '603376', '603402', '603418', '001220', '001239', '001280', '001285', '001325', '001330', '001369', '001390', '001396']
            for index, row in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Fetching stock data"):
                code = row["code"]
                if code in error_code_arr:
                        # print(f"跳过股票 {code}，已在错误列表中")
                        continue
                try:
                    cache_file_name=f"stock_data_{code}_{today}_pl"
                    df=cache.get_pl(cache_file_name)
                    if df is None or df.is_empty():
                        items = self.get_stock_data_tx(code)
                        if items is not None and len(items)>0:
                            df = pl.DataFrame(items, schema=["date", "open", "close", "high", "low", "volume"])
                            cache.set_pl(cache_file_name, df)
                        else:
                            error_code_arr.append(code)
                            print(f"获取股票数据 {code} 失败")
                    if df is not None and not df.is_empty():
                        df = df.with_columns(
                            pl.lit(int(code)).cast(pl.Int32).alias("code"),
                            pl.col("date").str.replace_all("-", "").cast(pl.Int32),
                            # 价格字段转换为分，使用int32
                            (pl.col("open").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("open"),
                            (pl.col("close").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("close"),
                            (pl.col("high").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("high"),
                            (pl.col("low").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("low"),
                            # 成交量使用int32
                            pl.col("volume").cast(pl.Float32).fill_null(0).cast(pl.Int32).alias("volume")
                        )
                        df = df.select(["code", "date", "open", "close", "high", "low", "volume"])
                        stock_data = pl.concat([stock_data, df])
                except Exception as e:
                    print(f"Error fetching data for {code}: {e}")
            cache.set_pl(all_cache_file_name,stock_data)
            # print(f"获取股票数据 {error_code_arr} 失败")

        else:
            print(f"缓存取股票数据 {all_cache_file_name} 成功")
        cache.clean(prefix="stock_data_")
        return stock_data


    def get_stock_data_tx(self,code):
        # code 已经是字符串类型，直接使用
        market = "sh" if code.startswith("6") else "sz"
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={market}{code},day,,,640,qfq,1"
        headers = {"Referer": "http://finance.qq.com/mp/"}
        resp = requests.get(url, headers=headers, timeout=10)
        data = resp.json()
        if data.get("code") == 0 and data.get("data") and data["data"].get(f"{market}{code}"):
            items = data["data"][f"{market}{code}"].get("qfqday", [])
            return [row[:6] for row in items]
        return None
        
    

    def calc_up_days(self, close_prices: np.ndarray) -> list[int]:
        """计算连续上涨天数"""
        consecutive_up = [0] * len(close_prices)
        count = 0
        for i in range(len(close_prices)):
            if i == 0:
                consecutive_up[i] = 0
                count = 0
            else:
                if close_prices[i] >= close_prices[i - 1]:
                    count += 1
                else:
                    count = 0
                consecutive_up[i] = count
        return consecutive_up
    
if __name__ == "__main__":
    sd=StockData()
    print("测试按日期获取数据")
    print(sd.get_data_by_date(20250707))
    print("测试按日期和代码获取数据")
    print(sd.get_data_by_date_code(20250102,721))

