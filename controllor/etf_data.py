"""
ETF 数据获取模块 - 参考 stock_data.py 的缓存机制
使用腾讯接口获取 A 股 ETF 数据
"""
from datetime import date
import time
import requests
from local_cache import LocalCache as lc
import numpy as np
import polars as pl
from tqdm import tqdm
from stock_calendar import StockCalendar as sc


class ETFData:
    """ETF 数据管理类"""

    def __init__(self, force_refresh=False):
        today = sc().get_last_trade_date()
        cache = lc()
        print(f"1. 获取 ETF 列表")
        etf_list_df = self.get_etf_list(str(today), cache)

        # 构建 ETF 代码到名称的缓存字典
        self.code_to_name = {}
        for _, row in etf_list_df.iterrows():
            code_str = str(row['code'])
            name = row['name']
            self.code_to_name[code_str] = name
            if code_str.isdigit():
                self.code_to_name[int(code_str)] = name

        print(f"   已缓存 {len(etf_list_df)} 只 ETF")
        print(f"2. 获取 ETF 数据")
        etf_data_df = self.get_etf_data_pl(etf_list_df, str(today), cache, force_refresh)
        print(f"3. 计算技术指标")

        # 按 code 分组并计算技术指标
        if etf_data_df.is_empty():
            self.etf_data_df = pl.DataFrame()
        else:
            grouped = etf_data_df.group_by("code")
            processed_dfs = []
            for code, group in grouped:
                sorted_group = group.sort("date")
                tech_df = self.calc_tech_pl(sorted_group)
                processed_dfs.append(tech_df)

            self.etf_data_df = pl.concat(processed_dfs) if processed_dfs else pl.DataFrame()
        print(f"4. 数据准备完成，共 {len(self.etf_data_df)} 条记录")

    def calc_tech_pl(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 ETF 技术指标"""
        df_tech = df.lazy().with_columns([
            # 计算涨跌幅
            ((pl.col("close") - pl.col("close").shift(1)) / pl.col("close").shift(1) * 100)
            .round(2).fill_null(0).cast(pl.Float32).alias("change_pct"),
            # 3日涨幅
            ((pl.col("close") - pl.col("close").shift(3)) / pl.col("close").shift(3) * 100)
            .round(2).cast(pl.Float32).alias("change_3d"),
            # 5日涨幅
            ((pl.col("close") - pl.col("close").shift(5)) / pl.col("close").shift(5) * 100)
            .round(2).cast(pl.Float32).alias("change_5d"),
        ]).collect()
        return df_tech

    def get_etf_list(self, today: str, cache) -> pl.DataFrame:
        """获取 ETF 列表"""
        cache_file_name = f"etf_list_{today}"
        cache.clean(prefix="etf_list_", ignore=[cache_file_name])
        etf_list = cache.get(cache_file_name)

        if etf_list is None:
            try:
                # 使用 akshare 获取 ETF 列表
                import akshare as ak
                import pandas as pd

                # 获取 ETF 列表
                all_df = ak.fund_etf_category_sina(symbol="ETF基金")
                # 去掉前缀 sh/sz
                all_df["代码"] = all_df["代码"].str.replace("sh", "").str.replace("sz", "")
                # 过滤 ETF（名称包含 ETF，不包含 LOF）
                all_df = all_df[all_df["名称"].str.contains("ETF") & ~all_df["名称"].str.contains("LOF")]
                # 过滤 A 股 ETF
                all_df = all_df[all_df["代码"].str.startswith(("51", "56", "58", "15", "16"))]
                etf_list = all_df.rename(columns={"代码": "code", "名称": "name"})[["code", "name"]]
                etf_list = etf_list.drop_duplicates(subset=["code"]).reset_index(drop=True)

                cache.set(cache_file_name, etf_list)
                print(f"   从 akshare 获取 ETF 列表: {len(etf_list)} 只")
            except Exception as e:
                print(f"   获取 ETF 列表失败: {e}")
                import pandas as pd
                etf_list = pd.DataFrame({"code": [], "name": []})

        return etf_list

    def get_etf_data_pl(self, etf_list, today: str, cache, force_refresh=False):
        """获取 ETF 历史数据"""
        all_cache_file_name = f"all_etf_data_{today}_pl"
        cache.clean(prefix="all_etf_data_", ignore=[all_cache_file_name])
        etf_data = None

        if not force_refresh:
            etf_data = cache.get_pl(all_cache_file_name)

        if etf_data is None:
            print(f"   缓存取 ETF 数据 {all_cache_file_name} 失败，尝试从腾讯接口获取")
            etf_data = pl.DataFrame()

            for index, row in tqdm(etf_list.iterrows(), total=len(etf_list), desc="Fetching ETF data"):
                code = row["code"]
                try:
                    cache_file_name = f"etf_data_{code}_{today}_pl"
                    df = cache.get_pl(cache_file_name)

                    if df is None or df.is_empty():
                        # 使用 akshare 新浪接口获取 ETF 历史数据
                        df = self.get_etf_data_akshare(code)
                        if df is not None and not df.is_empty():
                            cache.set_pl(cache_file_name, df)
                        else:
                            continue

                    if df is not None and not df.is_empty():
                        df = df.with_columns(
                            pl.lit(int(code)).cast(pl.Int32).alias("code"),
                            pl.col("date").str.replace_all("-", "").cast(pl.Int32),
                            (pl.col("open").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("open"),
                            (pl.col("close").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("close"),
                            (pl.col("high").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("high"),
                            (pl.col("low").cast(pl.Float32) * 100).round(0).cast(pl.Int32).alias("low"),
                            pl.col("volume").cast(pl.Float32).fill_null(0).cast(pl.Int32).alias("volume")
                        )
                        df = df.select(["code", "date", "open", "close", "high", "low", "volume"])
                        etf_data = pl.concat([etf_data, df])

                except Exception as e:
                    print(f"   Error fetching ETF {code}: {e}")
                    continue

            if not etf_data.is_empty():
                cache.set_pl(all_cache_file_name, etf_data)
                print(f"   成功获取 {len(etf_data)} 条 ETF 数据")
        else:
            print(f"   缓存取 ETF 数据 {all_cache_file_name} 成功")

        return etf_data

    def get_etf_data_akshare(self, code: str) -> pl.DataFrame | None:
        """使用 akshare 新浪接口获取 ETF 历史数据"""
        try:
            import akshare as ak
            market = "sh" if code.startswith("5") else "sz"
            df = ak.fund_etf_hist_sina(symbol=f"{market}{code}")
            if df is not None and not df.empty:
                # 转换为 polars DataFrame，确保类型正确
                return pl.DataFrame({
                    "date": df["date"].astype(str).tolist(),
                    "open": df["open"].astype(float).tolist(),
                    "close": df["close"].astype(float).tolist(),
                    "high": df["high"].astype(float).tolist(),
                    "low": df["low"].astype(float).tolist(),
                    "volume": df["volume"].astype(int).tolist()
                })
        except Exception as e:
            pass
        return None

    def get_data_by_date(self, date: int) -> pl.DataFrame:
        """获取指定日期的 ETF 数据"""
        return self.etf_data_df.filter(pl.col("date") == date)

    def get_data_by_date_code(self, date: int, code: int) -> pl.DataFrame:
        """获取指定日期和代码的 ETF 数据"""
        return self.etf_data_df.filter((pl.col("date") == date) & (pl.col("code") == code))

    def get_etf_name(self, code: int | str) -> str:
        """根据 ETF 代码获取名称"""
        name = self.code_to_name.get(code)
        if name:
            return name
        return self.code_to_name.get(str(code), str(code))

    def get_etf_display(self, code: int | str) -> str:
        """获取 ETF 代码+名称的展示字符串"""
        code_str = str(code)
        name = self.code_to_name.get(code)
        if not name:
            name = self.code_to_name.get(code_str, '')
        return f"{code_str} {name}" if name else code_str


if __name__ == "__main__":
    etf = ETFData()
    print("\n测试 ETF 数据获取")
    print("测试按日期获取数据:")
    df = etf.get_data_by_date(20250303)
    print(f"  记录数: {len(df)}")
    if len(df) > 0:
        print(df.head())
    print("\n测试按日期和代码获取数据:")
    df2 = etf.get_data_by_date_code(20250303, 510050)
    print(f"  记录数: {len(df2)}")
    if len(df2) > 0:
        print(df2)
