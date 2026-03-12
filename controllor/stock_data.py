from datetime import date,datetime
import akshare as ak
from local_cache import LocalCache as lc
import numpy as np
import polars as pl
from tqdm import tqdm
import requests
from stock_calendar import StockCalendar as sc
# pl 显示所有表头
pl.Config.set_tbl_cols(100)
# pl 显示所有列
pl.Config.set_tbl_rows(10)


class PickData:
    """选股专用数据结构 - 已过滤（量比>1且10天内无涨停且次日开盘不涨停），字段精简"""
    __slots__ = ['code', 'open', 'close', 'high', 'low', 'volume', 'next_open',
                 'consecutive_up_days', 'change_3d', 'change_5d', 'change_pct']

    def __init__(self, code=None, open_=None, close=None, high=None, low=None, volume=None, next_open=None,
                 consecutive_up_days=None, change_3d=None, change_5d=None, change_pct=None):
        self.code = code
        self.open = open_
        self.close = close
        self.high = high
        self.low = low
        self.volume = volume
        self.next_open = next_open
        self.consecutive_up_days = consecutive_up_days
        self.change_3d = change_3d
        self.change_5d = change_5d
        self.change_pct = change_pct


class FullData:
    """完整数据结构 - 包含所有股票，用于持仓/卖出逻辑，O(1)查询优化
    只保留持仓/卖出需要的字段，选股字段在PickData中
    """
    __slots__ = ['open', 'close', 'high', 'low', 'next_open', 'price_limit_status']

    def __init__(self, open_=0, close=0, high=0, low=0, next_open=0, price_limit_status=0):
        self.open = open_
        self.close = close
        self.high = high
        self.low = low
        self.next_open = next_open
        self.price_limit_status = price_limit_status


# 空PickData对象，用于无数据时返回，避免重复创建
_EMPTY_PICK_DATA = PickData(code=np.array([], dtype=np.int32))
# 空FullData对象
_EMPTY_FULL_DATA = FullData()


class StockData:

    def __init__(self, force_refresh=False):
        today = sc().get_last_trade_date()
        cache=lc()
        print(f"1. 获取股票列表")
        stock_list_df=self.get_stock_list(str(today),cache)
        # 构建股票代码到名称的缓存字典（支持str和int两种key）
        self.code_to_name = {}
        for _, row in stock_list_df.iterrows():
            code_str = str(row['code'])
            name = row['name']
            self.code_to_name[code_str] = name
            # 如果code是纯数字，也缓存int类型的key
            if code_str.isdigit():
                self.code_to_name[int(code_str)] = name
        print(f"   已缓存 {len(stock_list_df)} 只股票名称")
        print(f"2. 获取股票数据")
        stock_data_df=self.get_stock_data_pl(stock_list_df,str(today),cache,force_refresh)
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
        print(f"5. 构建快速查找索引")
        # 统一数据结构：2层 NumPy 结构，支持批量筛选和O(1)精确查询
        # self.date_numpy_dict = {today: {field: np.array, '_code_to_idx': {code: idx}}}

    # 选票过滤

    def calc_tech_pl(self,df: pl.DataFrame) -> pl.DataFrame:
        df_tech = df.lazy().with_columns([
                # 计算涨跌幅和下一天开盘价/收盘价
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
                # 3日涨幅: (今日收盘 - 3天前收盘) / 3天前收盘 * 100
                ((pl.col("close") - pl.col("close").shift(3)) / pl.col("close").shift(3) * 100).round(2).cast(pl.Float32).alias("change_3d"),
                # 5日涨幅: (今日收盘 - 5天前收盘) / 5天前收盘 * 100 (同花顺口径)
                ((pl.col("close") - pl.col("close").shift(5)) / pl.col("close").shift(5) * 100).round(2).cast(pl.Float32).alias("change_5d"),
                # 计算价格限制状态：0=正常，1=涨停，2=跌停
                pl.when(pl.col("open").shift(-1) >= pl.col("close") * 1.095)
                .then(pl.lit(1))
                .when(pl.col("open") <= pl.col("close").shift(1) * 0.905)
                .then(pl.lit(2))
                .otherwise(pl.lit(0))
                .cast(pl.Int8).alias("price_limit_status")
            ]).with_columns([
                pl.Series(
                    name="consecutive_up_days",
                    values=self.calc_up_days(df["close"].to_numpy()),  # 转为numpy数组适配原函数
                    dtype=pl.Int8  # 使用int8存储连续上涨天数
                ),
                # 计算10天内涨停次数（涨幅>=9.9%），短线关注10天即可
                # null值填充为99（表示不满足无涨停条件）
                (pl.col("change_pct") >= 9.9).rolling_sum(window_size=10).cast(pl.Int8).fill_null(99).alias("limit_up_count_10d"),
                # 计算量比：当日成交量 / MA5成交量，用于判断量能变化
                (pl.col("volume") / pl.col("ma5_vol")).round(2).cast(pl.Float32).alias("volume_ratio"),
                # 日内振幅参数已移除（回测证明效果不明显）
            ]).drop(["ma5_vol", "ma10_vol"]).collect()
        
        return df_tech

    def convert(self, stock_data_df) -> dict[int, pl.DataFrame]:
        """构建两层数据结构：
        1. pick_data_dict: 已过滤的选股数据（量比>1且10天内无涨停），按日期分组，用于选股
        2. full_data_dict: O(1)查询结构，key=日期*1000000+代码，用于持仓/卖出逻辑
        """
        self.pick_data_dict: dict[int, PickData] = {}  # 选股专用：已过滤，按日期分组
        self.full_data_dict: dict[int, FullData] = {}  # 完整数据：O(1)查询，key=date*1000000+code
        grouped = stock_data_df.group_by("date")

        # 打印20250701这天的股票数量
        print("\n📊 20250701股票数量统计:")
        print("-" * 40)
        target_date = 20250701
        group = stock_data_df.filter(pl.col("date") == target_date)
        count = len(group)
        print(f"  {target_date}: {count:4d} 只股票")
        print("-" * 40)

        # 打印数据范围信息
        all_dates = sorted([d[0] if isinstance(d, tuple) else d for d, _ in grouped])
        print(f"  数据范围: {all_dates[0]} 至 {all_dates[-1]}")
        print(f"  总交易日: {len(all_dates)} 天\n")

        for trade_date, group in grouped:
            if isinstance(trade_date, tuple):
                trade_date = trade_date[0]
            if group is None or group.is_empty():
                continue
            group = group.sort("volume", descending=False)  # 按成交量升序排列（选冷门股）

            # 提取numpy数组
            codes = group['code'].to_numpy()
            opens = group['open'].to_numpy()
            closes = group['close'].to_numpy()
            highs = group['high'].to_numpy()
            lows = group['low'].to_numpy()
            volumes = group['volume'].to_numpy()
            next_opens = group['next_open'].to_numpy()
            price_limit_status = group['price_limit_status'].to_numpy()
            consecutive_up_days = group['consecutive_up_days'].to_numpy()
            change_3d = group['change_3d'].to_numpy()
            change_5d = group['change_5d'].to_numpy()
            change_pct = group['change_pct'].to_numpy()
            limit_up_count_10d = group['limit_up_count_10d'].to_numpy()
            volume_ratio = group['volume_ratio'].to_numpy()

            # 构建第一层：完整数据O(1)查询结构（只保留持仓/卖出需要的字段）
            base_key = trade_date * 1000000
            for i in range(len(codes)):
                key = base_key + int(codes[i])
                self.full_data_dict[key] = FullData(
                    open_=opens[i],
                    close=closes[i],
                    high=highs[i],
                    low=lows[i],
                    next_open=next_opens[i],
                    price_limit_status=price_limit_status[i]
                )

            # 构建第二层：选股专用数据（前置过滤：量比>1且10天内无涨停且次日开盘不涨停）
            # 量比>1 且 10天内无涨停(limit_up_count_10d == 0) 且 次日开盘不涨停(next_open != close * 1.1)
            # 次日开盘涨停可以通过 next_open 和 close 计算得出，属于T日已知数据
            next_open_limit_up = (next_opens / closes - 1) >= 0.099  # 考虑浮点误差
            mask = (volume_ratio > 1.0) & (limit_up_count_10d == 0) & (~next_open_limit_up)
            filtered_indices = np.where(mask)[0]

            if len(filtered_indices) > 0:
                self.pick_data_dict[trade_date] = PickData(
                    code=codes[filtered_indices],
                    open_=opens[filtered_indices],
                    close=closes[filtered_indices],
                    high=highs[filtered_indices],
                    low=lows[filtered_indices],
                    volume=volumes[filtered_indices],
                    next_open=next_opens[filtered_indices],
                    consecutive_up_days=consecutive_up_days[filtered_indices],
                    change_3d=change_3d[filtered_indices],
                    change_5d=change_5d[filtered_indices],
                    change_pct=change_pct[filtered_indices]
                )
            else:
                # 没有符合条件的股票，创建空对象
                self.pick_data_dict[trade_date] = PickData(code=np.array([], dtype=np.int32))

        return {}

    def get_pick_data_by_date(self, today: int) -> PickData:
        """获取当天已过滤的选股数据（量比>1且10天内无涨停）
        用于选股逻辑，数据已前置过滤，减少计算量
        """
        return self.pick_data_dict.get(today, _EMPTY_PICK_DATA)

    def get_full_data_by_date_code(self, today: int, code: int) -> FullData:
        """精确查询某只股票的完整数据，O(1)复杂度
        用于持仓/卖出逻辑，数据完整，包含所有字段
        key = 日期*1000000 + 股票代码，单次dict查询
        """
        key = today * 1000000 + code
        return self.full_data_dict.get(key, _EMPTY_FULL_DATA)

    def get_numpy_data_by_date(self, today: int) -> PickData:
        """获取当天所有股票的NumPy数据（用于批量筛选）"""
        return self.pick_data_dict.get(today, _EMPTY_PICK_DATA)

    def get_data_by_date_code(self, today: int, code: int) -> FullData:
        """精确查询某只股票的数据，O(1)复杂度"""
        key = today * 1000000 + code
        return self.full_data_dict.get(key, _EMPTY_FULL_DATA)

    def get_stock_name(self, code: int | str) -> str:
        """根据股票代码获取股票名称"""
        # 先尝试直接用code作为key查找（支持int和str）
        name = self.code_to_name.get(code)
        if name:
            return name
        # 如果没找到，转为str再查找
        return self.code_to_name.get(str(code), str(code))

    def get_stock_display(self, code: int | str) -> str:
        """获取股票代码+名称的展示字符串"""
        code_str = str(code)
        # 先尝试直接用code作为key查找
        name = self.code_to_name.get(code)
        if not name:
            # 如果没找到，转为str再查找
            name = self.code_to_name.get(code_str, '')
        return f"{code_str} {name}" if name else code_str

    def get_hs300_codes(self, cache, today):
        """获取沪深300成分股代码列表，带日期缓存"""
        cache_file_name = f"hs300_cons_{today}"
        hs300_df = cache.get(cache_file_name)
        
        if hs300_df is None:
            try:
                # 使用akshare获取沪深300成分股权重数据
                hs300_df = ak.index_stock_cons_weight_csindex(symbol="000300")
                # 只保留成分股代码列
                hs300_df = hs300_df[["成分券代码"]]
                hs300_df.columns = ["code"]
                cache.set(cache_file_name, hs300_df)
            except Exception as e:
                print(f"获取沪深300成分股失败: {e}")
                return set()
        
        return set(hs300_df["code"].astype(str).tolist())

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
            
            # 过滤掉沪深300成分股（只保留中小股）
            print(f"2.1 过滤沪深300成分股")
            hs300_codes = self.get_hs300_codes(cache, today)
            if hs300_codes:
                stock_list_pl = stock_list_pl.filter(
                    ~pl.col("代码").is_in(list(hs300_codes))
                )
                print(f"   排除沪深300成分股 {len(hs300_codes)} 只")
            
            # 重命名列，保持股票代码为字符串类型
            stock_list_pl = stock_list_pl.rename({"代码": "code", "名称": "name"})
            
            # 转换回pandas以保持缓存兼容性
            stock_list = stock_list_pl.to_pandas()
            
            cache_file_name="stock_list_"+date.today().strftime("%Y%m%d")
            cache.set(cache_file_name, stock_list)
        return stock_list


    def get_stock_data_pl(self,stock_list,today,cache,force_refresh=False):
        all_cache_file_name=f"all_stock_data_{today}_pl"
        cache.clean(prefix="all_stock_data_",ignore=[all_cache_file_name])
        stock_data = None
        if not force_refresh:
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
    # print(sd.get_numpy_data_by_date(20250707))
    print("测试按日期和代码获取数据")
    print(sd.get_data_by_date_code(20250102,721))

