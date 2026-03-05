"""
ETF 数据可视化 Web 页面 - 动量分析（最终优化版）
使用 Flask 提供 Web 服务，支持滚动查看
"""
import numpy as np
import polars as pl
from datetime import datetime, timedelta
from flask import Flask, render_template_string
from etf_data import ETFData
import re

app = Flask(__name__)


class ETFWebAnalyzer:
    """ETF Web 分析器 - 最终优化版"""

    def __init__(self):
        self.etf_data = ETFData()
        self.df = self.etf_data.etf_data_df

    def get_recent_data(self, days: int = 30):
        """获取最近N天的数据（至少30天用于回测）"""
        all_dates = self.df.select("date").unique().sort("date")
        recent_dates = all_dates.tail(days)
        date_list = recent_dates["date"].to_list()
        recent_df = self.df.filter(pl.col("date").is_in(date_list))
        return recent_df, date_list

    def extract_industry(self, name: str) -> str:
        """从ETF名称提取行业"""
        # 常见行业关键词映射
        industry_keywords = {
            '电力': ['电力', '电网', '绿电', '绿色电力'],
            '公用事业': ['公用事业'],
            '能源': ['能源', '煤炭', '石油', '油气', '央企能源'],
            '军工': ['军工', '国防', '航空航天', '航天', '通用航空'],
            '有色': ['有色', '稀有金属', '工业有色', '矿业'],
            '农业': ['农业', '农牧', '粮食', '畜牧'],
            '科技': ['科技', '芯片', '半导体', '人工智能', 'AI', '计算机'],
            '医药': ['医药', '医疗', '生物医药', '创新药'],
            '消费': ['消费', '食品饮料', '白酒', '家电'],
            '金融': ['金融', '银行', '证券', '保险'],
            '地产': ['地产', '房地产', '基建', '建筑'],
            '传媒': ['传媒', '游戏', '影视'],
            '通信': ['通信', '5G', '电信'],
            '汽车': ['汽车', '新能源车', '智能汽车'],
            '机械': ['机械', '机床', '工业母机'],
            '材料': ['材料', '化工', '钢铁', '建材'],
            '环保': ['环保', 'ESG'],
            '红利': ['红利', '股息', '现金流'],
            '央企': ['央企', '国企'],
            '一带一路': ['一带一路'],
            '稀土': ['稀土'],
            '资源': ['资源', '大宗商品'],
            '创新': ['创新', '科创'],
        }
        
        for industry, keywords in industry_keywords.items():
            for keyword in keywords:
                if keyword in name:
                    return industry
        
        # 默认返回"其他"
        return '其他'

    def is_up_day(self, close_curr: float, close_prev: float) -> bool:
        """判断是否为上涨日（涨幅>0.1%）"""
        if close_prev <= 0:
            return False
        return (close_curr - close_prev) / close_prev > 0.001

    def validate_data(self, closes: np.ndarray, volumes: np.ndarray, dates: list) -> bool:
        """数据验证：检查异常值"""
        # 检查是否有0值或负数
        if np.any(closes <= 0) or np.any(volumes < 0):
            return False
        
        # 检查是否有NaN或无穷大
        if np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
            return False
        if np.any(np.isnan(volumes)) or np.any(np.isinf(volumes)):
            return False
        
        # 检查价格是否异常（单日涨跌幅超过20%视为异常）
        daily_returns = np.diff(closes) / closes[:-1]
        if np.any(np.abs(daily_returns) > 0.20):
            return False
        
        return True

    def calculate_momentum(self, group_df: pl.DataFrame, all_df: pl.DataFrame = None) -> dict:
        """计算量价动量指标 - 最终优化版"""
        
        sorted_df = group_df.sort("date")
        closes = sorted_df["close"].to_numpy()
        opens = sorted_df["open"].to_numpy()
        highs = sorted_df["high"].to_numpy()
        lows = sorted_df["low"].to_numpy()
        volumes = sorted_df["volume"].to_numpy()
        dates = sorted_df["date"].to_list()
        code = int(sorted_df["code"][0])

        n = len(closes)
        
        # ========== 数据预处理 ==========
        # 条件1: 上市天数>=30天
        if n < 30:
            return None
        
        # 数据验证
        if not self.validate_data(closes, volumes, dates):
            return None
        
        # 条件2: 最新日成交量>=100万
        latest_volume = volumes[-1]
        if latest_volume < 1_000_000:
            return None
        
        # 条件3: 20天涨幅>0（中长期趋势向上）
        total_return_20d = (closes[-1] - closes[-20]) / closes[-20] * 100
        if total_return_20d <= 0:
            return None

        # 计算均线
        ma5 = np.mean(closes[-5:])
        ma10 = np.mean(closes[-10:])
        ma5_vol = np.mean(volumes[-5:])
        ma10_vol = np.mean(volumes[-10:])
        
        # ========== 基础过滤条件 ==========
        # MA5 > MA10 且 MA5_vol > MA10_vol
        if ma5 <= ma10 or ma5_vol <= ma10_vol:
            return None

        # ========== 涨幅计算 ==========
        total_return_10d = (closes[-1] - closes[-10]) / closes[-10] * 100
        total_return_5d = (closes[-1] - closes[-5]) / closes[-5] * 100
        return_1d = (closes[-1] - closes[-2]) / closes[-2] * 100

        # ========== 短期风险过滤 ==========
        # 1天跌幅不超过3%
        if return_1d < -3:
            return None
        
        # 近5均量 / 前20均量
        if len(volumes) >= 25:
            prev_20_vol_avg = np.mean(volumes[-25:-5])
        elif len(volumes) > 5:
            prev_20_vol_avg = np.mean(volumes[:-5])
        else:
            prev_20_vol_avg = ma5_vol
        
        vol_ratio = ma5_vol / prev_20_vol_avg if prev_20_vol_avg > 0 else 1.0
        
        # 限制vol_ratio最大值5
        vol_ratio = min(vol_ratio, 5.0)
        
        # 下跌时禁止放量
        if return_1d < 0 and vol_ratio >= 1.5:
            return None
        
        # 最近3天至少有1天上涨（涨幅>0.1%）
        up_days_3 = sum(1 for i in range(-3, 0) if self.is_up_day(closes[i], closes[i-1]))
        if up_days_3 < 1:
            return None

        # ========== 动量计算 ==========
        price_momentum = total_return_10d

        # 量价健康度
        if price_momentum > 0:
            if return_1d > 0:
                volume_health = vol_ratio
            else:
                volume_health = 1 / (vol_ratio * 2) if vol_ratio > 0 else 0.5
        else:
            volume_health = 1 / vol_ratio if vol_ratio > 0 else 1.0

        # 短期修正系数
        if return_1d < -2:
            short_term_factor = 0.5
        elif return_1d < 0:
            short_term_factor = 0.8
        else:
            short_term_factor = 1.0

        # 趋势得分
        momentum_score = price_momentum * (0.6 + 0.4 * volume_health) * short_term_factor

        # ========== 辅助指标 ==========
        # 量价配合度（最近10天上涨天数占比，涨幅>0.1%）
        up_days_10 = sum(1 for i in range(-10, 0) if self.is_up_day(closes[i], closes[i-1]))
        volume_price_ratio = (up_days_10 / 10) * 100

        # 趋势强度（最近5天上涨天数占比）
        up_days_5_count = sum(1 for i in range(-5, 0) if self.is_up_day(closes[i], closes[i-1]))
        trend_strength = (up_days_5_count / 5) * 100

        # 波动率（日收益率标准差，年化）
        daily_returns = np.diff(closes[-11:]) / closes[-11:-1]
        if len(daily_returns) > 1:
            # 使用标准差直接作为波动率（更合理）
            volatility = np.std(daily_returns) * 100  # 转换为百分比
            # 限制波动率在合理范围（0.1% - 50%）
            volatility = max(min(volatility, 50.0), 0.1)
        else:
            volatility = 0.1

        # 计算OBV
        obv = [0]
        for i in range(1, n):
            if closes[i] > closes[i-1]:
                obv.append(obv[-1] + volumes[i])
            elif closes[i] < closes[i-1]:
                obv.append(obv[-1] - volumes[i])
            else:
                obv.append(obv[-1])

        # 归一化OBV
        obv_norm = np.array(obv)
        if len(obv_norm) > 0 and np.std(obv_norm) > 0:
            obv_norm = (obv_norm - np.mean(obv_norm)) / np.std(obv_norm)

        # 构建K线数据
        kline_data = []
        for i in range(n):
            kline_data.append({
                "x": str(dates[i]),
                "o": float(opens[i]),
                "h": float(highs[i]),
                "l": float(lows[i]),
                "c": float(closes[i])
            })

        # 获取未来3天收益率（回测用）
        future_return_3d = self.get_future_return(code, dates[-1], all_df)

        return {
            "code": code,
            "dates": [str(d) for d in dates],
            "closes": closes.tolist(),
            "opens": opens.tolist(),
            "highs": highs.tolist(),
            "lows": lows.tolist(),
            "volumes": volumes.tolist(),
            "obv": obv_norm.tolist(),
            "kline": kline_data,
            "price_momentum": float(price_momentum),
            "volume_momentum": float(vol_ratio),
            "volume_price_ratio": float(volume_price_ratio),
            "momentum_score": float(momentum_score),
            "volatility": float(volatility),
            "trend_strength": float(trend_strength),
            "latest_close": float(closes[-1]),
            "latest_volume": int(volumes[-1]),
            "total_return_20d": float(total_return_20d),
            "total_return_10d": float(total_return_10d),
            "total_return_5d": float(total_return_5d),
            "return_1d": float(return_1d),
            "future_return_3d": float(future_return_3d) if future_return_3d is not None else None,
            "industry": None,  # 将在analyze_all_etfs中填充
            "industry_weight": 1.0,  # 将在analyze_all_etfs中计算
            "value_score": 0.0  # 将在analyze_all_etfs中计算
        }

    def get_future_return(self, code: int, current_date, all_df: pl.DataFrame = None) -> float:
        """获取未来3天收益率（用于回测）"""
        try:
            if all_df is None:
                all_df = self.df
            
            # 获取该ETF的所有数据
            etf_df = all_df.filter(pl.col("code") == code).sort("date")
            if len(etf_df) == 0:
                return None
            
            dates = etf_df["date"].to_list()
            closes = etf_df["close"].to_numpy()
            
            # 找到当前日期索引
            try:
                current_idx = dates.index(str(current_date))
            except ValueError:
                return None
            
            # 检查是否有未来3天数据
            if current_idx + 3 >= len(closes):
                return None
            
            # 计算未来3天收益率
            future_return = (closes[current_idx + 3] - closes[current_idx]) / closes[current_idx] * 100
            return future_return
            
        except Exception as e:
            return None

    def analyze_all_etfs(self, days: int = 30, top_n: int = 20):
        """分析所有ETF的动量，带过滤和行业分散"""
        recent_df, date_list = self.get_recent_data(days)
        
        # 获取全部数据用于回测
        all_df = self.df

        # 第一步：获取所有ETF代码（过滤前）
        all_codes = set()
        grouped = recent_df.group_by("code")
        for code, _ in grouped:
            if isinstance(code, tuple):
                code = code[0]
            all_codes.add(int(code))
        total_before_filter = len(all_codes)

        # 第二步：收集过滤后的ETF数据
        filtered_etfs = []
        for code in all_codes:
            try:
                group = recent_df.filter(pl.col("code") == code)
                momentum = self.calculate_momentum(group, all_df)
                if momentum:
                    name = self.etf_data.get_etf_name(code)
                    momentum["name"] = name
                    momentum["industry"] = self.extract_industry(name)
                    filtered_etfs.append(momentum)
            except Exception as e:
                # 跳过计算失败的ETF
                continue

        # 第三步：计算行业分布和权重
        industry_counts = {}
        for etf in filtered_etfs:
            industry = etf["industry"]
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        # 计算行业权重并更新value_score
        for etf in filtered_etfs:
            industry = etf["industry"]
            industry_count = industry_counts.get(industry, 1)
            # 行业权重 = 1 / 该行业ETF数量
            industry_weight = 1.0 / industry_count
            etf["industry_weight"] = industry_weight
            # 综合性价比得分 = 趋势得分 * (1 - 波动率/100) * 行业权重
            etf["value_score"] = etf["momentum_score"] * (1 - etf["volatility"] / 100) * industry_weight

        # 第四步：按综合性价比得分排序
        filtered_etfs.sort(key=lambda x: (x["value_score"], x["latest_volume"]), reverse=True)

        # 第五步：每个行业最多保留3只ETF
        industry_limits = {}
        final_etfs = []
        for etf in filtered_etfs:
            industry = etf["industry"]
            if industry_limits.get(industry, 0) < 3:
                final_etfs.append(etf)
                industry_limits[industry] = industry_limits.get(industry, 0) + 1

        # 返回统计信息
        stats = {
            "total_before_filter": total_before_filter,
            "total_after_filter": len(final_etfs),
            "industry_distribution": industry_counts
        }

        return final_etfs[:top_n], final_etfs, date_list, stats


# HTML 模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ETF 动量分析 - 最终优化版</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-chart-financial"></script>
    <script src="https://cdn.jsdelivr.net/npm/luxon"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-luxon"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        h1 {
            text-align: center; color: white; margin-bottom: 30px;
            font-size: 2.5em; text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        .summary {
            background: rgba(255,255,255,0.95); border-radius: 15px;
            padding: 20px; margin-bottom: 30px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .summary h2 {
            color: #333; margin-bottom: 15px; font-size: 1.5em;
            display: flex; align-items: center; flex-wrap: wrap; gap: 10px;
        }
        .formula {
            background: #f8f9fa; padding: 15px; border-radius: 8px;
            margin: 10px 0; font-family: 'Courier New', monospace; color: #555;
            white-space: pre-wrap; line-height: 1.8; font-size: 13px;
        }
        .stats-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px; margin-bottom: 20px;
        }
        .stat-item {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 15px; border-radius: 10px; text-align: center;
        }
        .stat-value { font-size: 2em; font-weight: bold; }
        .stat-label { font-size: 0.9em; opacity: 0.9; margin-top: 5px; }
        .industry-grid {
            display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px; margin-top: 15px;
        }
        .industry-item {
            background: #e9ecef; padding: 8px 12px; border-radius: 20px;
            text-align: center; font-size: 0.85em;
        }
        .industry-item span { font-weight: bold; color: #667eea; }
        .backtest-section {
            background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 10px;
            padding: 15px; margin-top: 20px;
        }
        .backtest-section h3 { color: #856404; margin-bottom: 10px; }
        .backtest-table {
            width: 100%; border-collapse: collapse; font-size: 0.9em;
        }
        .backtest-table th, .backtest-table td {
            padding: 8px; text-align: center; border-bottom: 1px solid #dee2e6;
        }
        .backtest-table th { background: #667eea; color: white; }
        .backtest-table tr:hover { background: #f8f9fa; }
        .positive { color: #dc3545; font-weight: bold; }  /* A股红色表示涨 */
        .negative { color: #28a745; font-weight: bold; }  /* A股绿色表示跌 */
        .etf-card {
            background: white; border-radius: 15px; padding: 25px;
            margin-bottom: 25px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            transition: transform 0.3s ease;
        }
        .etf-card:hover { transform: translateY(-5px); }
        .etf-header {
            display: flex; justify-content: space-between; align-items: center;
            margin-bottom: 20px; flex-wrap: wrap; gap: 10px;
        }
        .etf-title { font-size: 1.8em; font-weight: bold; color: #333; }
        .etf-code { color: #666; font-size: 0.9em; }
        .etf-industry {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 5px 15px; border-radius: 20px;
            font-size: 0.85em;
        }
        .rank-badge {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; width: 50px; height: 50px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center;
            font-size: 1.5em; font-weight: bold;
        }
        .metrics {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 12px; margin-bottom: 20px;
        }
        .metric {
            background: #f8f9fa; padding: 12px; border-radius: 10px; text-align: center;
        }
        .metric.highlight {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white;
        }
        .metric.highlight .metric-label { color: rgba(255,255,255,0.9); }
        .metric.highlight .metric-value { color: white; font-size: 1.6em; }
        .metric-label { color: #666; font-size: 0.8em; margin-bottom: 5px; }
        .metric-value { font-size: 1.3em; font-weight: bold; color: #333; }
        .metric-value.positive { color: #28a745; }
        .metric-value.negative { color: #dc3545; }
        .chart-container {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 20px; margin-top: 20px;
        }
        .chart-box { 
            background: #f8f9fa; 
            padding: 15px; 
            border-radius: 10px; 
            height: 300px;
            position: relative;
        }
        .chart-box canvas {
            max-height: 250px !important;
        }
        .chart-title { font-weight: bold; margin-bottom: 10px; color: #555; }
        .tabs { display: flex; gap: 10px; margin-bottom: 20px; flex-wrap: wrap; }
        .tab-btn {
            padding: 12px 24px; border: none; background: rgba(255,255,255,0.3);
            color: white; border-radius: 25px; cursor: pointer; font-size: 1em;
            transition: all 0.3s ease;
        }
        .tab-btn:hover, .tab-btn.active {
            background: white; color: #667eea;
        }
        .tab-content { display: none; }
        .tab-content.active { display: block; }
        .data-table {
            width: 100%; border-collapse: collapse; background: white;
            border-radius: 10px; overflow: hidden; box-shadow: 0 5px 15px rgba(0,0,0,0.1);
            font-size: 12px;
        }
        .data-table th, .data-table td {
            padding: 8px 6px; text-align: center; border-bottom: 1px solid #eee;
            white-space: nowrap;
        }
        .data-table th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; font-weight: bold;
        }
        .data-table tr:hover { background: #f8f9fa; }
        .copy-btn {
            padding: 8px 20px; background: #667eea; color: white;
            border: none; border-radius: 5px; cursor: pointer; font-size: 14px;
        }
        .copy-btn:hover { background: #5a67d8; }
    </style>
</head>
<body>
    <div class="container">
        <h1>📈 ETF 动量分析 - 最终优化版</h1>
        
        <div class="summary">
            <h2>
                📊 数据统计与计算逻辑
                <button class="copy-btn" onclick="copyLogic()">📋 复制文本</button>
            </h2>
            
            <div class="stats-grid">
                <div class="stat-item">
                    <div class="stat-value">{{ stats.total_before_filter }}</div>
                    <div class="stat-label">过滤前ETF数量</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ stats.total_after_filter }}</div>
                    <div class="stat-label">过滤后ETF数量</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ "%.1f"|format((stats.total_before_filter - stats.total_after_filter) / stats.total_before_filter * 100) }}%</div>
                    <div class="stat-label">过滤比例</div>
                </div>
                <div class="stat-item">
                    <div class="stat-value">{{ stats.industry_distribution|length }}</div>
                    <div class="stat-label">行业数量</div>
                </div>
            </div>
            
            <h3>📊 行业分布</h3>
            <div class="industry-grid">
                {% for industry, count in stats.industry_distribution.items() %}
                <div class="industry-item">{{ industry }}: <span>{{ count }}</span>只</div>
                {% endfor %}
            </div>
            
            <div id="logic-text" class="formula">
========================================
ETF动量筛选系统 - 最终优化计算逻辑
========================================

【数据统计】
过滤前ETF数量: {{ stats.total_before_filter }}
过滤后ETF数量: {{ stats.total_after_filter }}
过滤比例: {{ "%.1f"|format((stats.total_before_filter - stats.total_after_filter) / stats.total_before_filter * 100) }}%

【数据获取】
1. 从akshare获取ETF列表（全部可交易ETF）
2. 使用akshare.fund_etf_hist_sina()获取每只ETF的历史K线数据（至少30天）
3. 数据字段: date(日期), open(开盘), high(最高), low(最低), close(收盘), volume(成交量)
4. 数据预处理:
   - 过滤上市天数&lt;30天的ETF
   - 过滤日成交量（最新日）&lt;100万的ETF
   - 处理异常值：close&lt;=0/volume&lt;0/NaN/Inf的ETF直接过滤
   - 检查单日涨跌幅&gt;20%视为异常数据

【过滤条件】（必须同时满足）
条件1: MA5 &gt; MA10  （5日均线大于10日均线，价格趋势向上）
条件2: MA5_vol &gt; MA10_vol  （5日均量大于10日均量，成交量趋势向上）
条件3: 1天涨幅 &gt; -3%  （单日跌幅不超过3%，避免大跌）
条件4: NOT(1天涨幅 &lt; 0 AND vol_ratio &gt;= 1.5)  （下跌时禁止放量）
条件5: 最近3天上涨天数 &gt;= 1  （短期至少有1天涨，涨幅&gt;0.1%）
条件6: 20天涨幅 &gt; 0  （中长期趋势向上，过滤负涨幅ETF）
条件7: 最新日成交量 &gt; 100万  （过滤低流动性ETF）

【中间变量计算】
# 基础均线/涨幅计算
MA5 = mean(close[-5:])  （最近5日收盘价均值）
MA10 = mean(close[-10:])  （最近10日收盘价均值）
MA5_vol = mean(volume[-5:])  （最近5日成交量均值）
MA10_vol = mean(volume[-10:])  （最近10日成交量均值）
20天涨幅 = (close[-1] - close[-20]) / close[-20] * 100%
10天涨幅 = (close[-1] - close[-10]) / close[-10] * 100%
5天涨幅 = (close[-1] - close[-5]) / close[-5] * 100%
1天涨幅 = (close[-1] - close[-2]) / close[-2] * 100%

# 动量相关计算
价格动量 = 10天涨幅
近5均量 = MA5_vol
前20均量 = mean(volume[-25:-5]) if len(volume)&gt;=25 else mean(volume[:-5]) if len(volume)&gt;5 else MA5_vol
vol_ratio = 近5均量 / 前20均量
vol_ratio = min(vol_ratio, 5)  （限制极端放量，最大值5）

# 量价健康度（分分支）
if 价格动量&gt;0 and 1天涨幅&gt;0:
    量价健康度 = vol_ratio  # 上涨放量，加分
elif 价格动量&gt;0 and 1天涨幅&lt;0:
    量价健康度 = 1/(vol_ratio*2)  # 下跌放量，大幅扣分
else:
    量价健康度 = 1/vol_ratio  # 价格动量为负，扣分

# 短期修正系数
if 1天涨幅 &lt; -2%:
    短期修正系数 = 0.5
elif 1天涨幅 &lt; 0:
    短期修正系数 = 0.8
else:
    短期修正系数 = 1.0

# 核心得分计算
趋势得分 = 价格动量 * (0.6 + 0.4 * 量价健康度) * 短期修正系数

# 辅助指标（修正版）
上涨日定义 = (close[i]-close[i-1])/close[i-1] &gt; 0.1%  # 涨幅&gt;0.1%才算上涨
量价配合度 = sum(上涨日定义 for i in range(-10, -1)) / 10 * 100%  # 最近10天上涨天数占比
趋势强度 = sum(上涨日定义 for i in range(-5, -1)) / 5 * 100%  # 最近5天上涨天数占比

# 波动率（日收益率标准差）
daily_return = (close[-10:] - close[-11:-1]) / close[-11:-1]  # 最近10天日收益率
波动率 = std(daily_return) * 100%  # 标准差直接作为波动率（百分比）
波动率 = max(min(波动率, 50%), 0.1%)  # 限制在0.1%-50%范围内

# 行业权重计算
行业名称 = 从ETF名称/代码提取行业（如电力、公用事业、军工等）
行业权重 = 1 / 该行业在筛选结果中的数量  # 行业越多，权重越高，分散风险

# 综合得分
综合性价比得分 = 趋势得分 * (1 - 波动率/100) * 行业权重

【排序规则】
1. 按"综合性价比得分"从高到低排序
2. 同得分下，按"最新日成交量"从高到低排序（优先高流动性）
3. 每个行业最多保留3只ETF（控制行业集中度）

【回测验证】
对筛选结果前10只ETF，计算未来3天收益率（基于历史数据验证）

【逻辑说明】
1. 新增数据预处理：过滤低流动性/新上市/异常值ETF，提升数据质量
2. 限制vol_ratio最大值5，避免极端放量干扰
3. 波动率改为日收益率标准差，更贴合风险本质
4. 新增行业权重，控制行业集中度风险
5. 上涨日定义为涨幅&gt;0.1%，过滤微涨/平盘
6. 新增20天涨幅&gt;0的过滤条件，保证中长期趋势
7. 排序时优先高流动性ETF，提升实战可交易性
========================================
            </div>
            
            {% if etfs and etfs[0].future_return_3d is not none %}
            <div class="backtest-section">
                <h3>📈 前10只ETF未来3天收益率回测</h3>
                <table class="backtest-table">
                    <thead>
                        <tr>
                            <th>排名</th>
                            <th>代码</th>
                            <th>名称</th>
                            <th>行业</th>
                            <th>综合得分</th>
                            <th>未来3天收益</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for etf in etfs[:10] %}
                        {% if etf.future_return_3d is not none %}
                        <tr>
                            <td>{{ loop.index }}</td>
                            <td>{{ etf.code }}</td>
                            <td>{{ etf.name }}</td>
                            <td>{{ etf.industry }}</td>
                            <td>{{ "%.2f"|format(etf.value_score) }}</td>
                            <td class="{% if etf.future_return_3d >= 0 %}positive{% else %}negative{% endif %}">
                                {{ "%.2f"|format(etf.future_return_3d) }}%
                            </td>
                        </tr>
                        {% endif %}
                        {% endfor %}
                    </tbody>
                </table>
            </div>
            {% endif %}
        </div>

        <div class="tabs">
            <button class="tab-btn active" onclick="switchTab('card')">📊 卡片视图</button>
            <button class="tab-btn" onclick="switchTab('table')">📋 数据表格</button>
        </div>

        <div id="card-view" class="tab-content active">
            {% for etf in etfs %}
            <div class="etf-card">
                <div class="etf-header">
                    <div>
                        <div class="rank-badge">{{ loop.index }}</div>
                    </div>
                    <div style="flex: 1; margin-left: 15px;">
                        <div class="etf-title">{{ etf.name }}</div>
                        <div class="etf-code">{{ etf.code }} | {{ etf.industry }} | 行业权重: {{ "%.2f"|format(etf.industry_weight) }}</div>
                    </div>
                    <div class="etf-industry">性价比 {{ "%.2f"|format(etf.value_score) }}</div>
                </div>
                
                <div class="metrics">
                    <div class="metric">
                        <div class="metric-label">20天涨幅</div>
                        <div class="metric-value {% if etf.total_return_20d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.total_return_20d) }}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">10天涨幅</div>
                        <div class="metric-value {% if etf.total_return_10d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.total_return_10d) }}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">5天涨幅</div>
                        <div class="metric-value {% if etf.total_return_5d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.total_return_5d) }}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">最近1天涨幅</div>
                        <div class="metric-value {% if etf.return_1d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.return_1d) }}%</div>
                    </div>
                    <div class="metric highlight">
                        <div class="metric-label">趋势得分</div>
                        <div class="metric-value">{{ "%.2f"|format(etf.momentum_score) }}</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">价格动量</div>
                        <div class="metric-value">{{ "%.2f"|format(etf.price_momentum) }}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">成交量动量</div>
                        <div class="metric-value">{{ "%.2f"|format(etf.volume_momentum) }}x</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">量价配合度</div>
                        <div class="metric-value">{{ "%.1f"|format(etf.volume_price_ratio) }}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">趋势强度</div>
                        <div class="metric-value">{{ "%.1f"|format(etf.trend_strength) }}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">波动率</div>
                        <div class="metric-value">{{ "%.2f"|format(etf.volatility) }}%</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">最新价</div>
                        <div class="metric-value">{{ "%.2f"|format(etf.latest_close) }}元</div>
                    </div>
                    <div class="metric">
                        <div class="metric-label">成交量</div>
                        <div class="metric-value">{{ "{:,}".format(etf.latest_volume) }}</div>
                    </div>
                    {% if etf.future_return_3d is not none %}
                    <div class="metric" style="background: #fff3cd;">
                        <div class="metric-label">未来3天收益</div>
                        <div class="metric-value {% if etf.future_return_3d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.future_return_3d) }}%</div>
                    </div>
                    {% endif %}
                </div>
                
                <div class="chart-container">
                    <div class="chart-box">
                        <div class="chart-title">K线走势 (最近20天)</div>
                        <canvas id="klineChart{{ loop.index }}" width="400" height="250"></canvas>
                    </div>
                    <div class="chart-box">
                        <div class="chart-title">成交量 & OBV 动量</div>
                        <canvas id="volumeChart{{ loop.index }}" width="400" height="250"></canvas>
                    </div>
                </div>
            </div>
            {% endfor %}
        </div>

        <div id="table-view" class="tab-content">
            <table class="data-table">
                <thead>
                    <tr>
                        <th>排名</th>
                        <th>代码</th>
                        <th>名称</th>
                        <th>行业</th>
                        <th>20天涨幅</th>
                        <th>10天涨幅</th>
                        <th>5天涨幅</th>
                        <th>1天涨幅</th>
                        <th>性价比得分</th>
                        <th>趋势得分</th>
                        <th>价格动量</th>
                        <th>成交量动量</th>
                        <th>量价配合</th>
                        <th>趋势强度</th>
                        <th>波动率</th>
                        <th>行业权重</th>
                        <th>最新价</th>
                        <th>成交量</th>
                        {% if etfs and etfs[0].future_return_3d is not none %}
                        <th>未来3天收益</th>
                        {% endif %}
                    </tr>
                </thead>
                <tbody>
                    {% for etf in all_etfs %}
                    <tr>
                        <td>{{ loop.index }}</td>
                        <td>{{ etf.code }}</td>
                        <td>{{ etf.name }}</td>
                        <td>{{ etf.industry }}</td>
                        <td class="{% if etf.total_return_20d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.total_return_20d) }}%</td>
                        <td class="{% if etf.total_return_10d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.total_return_10d) }}%</td>
                        <td class="{% if etf.total_return_5d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.total_return_5d) }}%</td>
                        <td class="{% if etf.return_1d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.return_1d) }}%</td>
                        <td><strong>{{ "%.2f"|format(etf.value_score) }}</strong></td>
                        <td>{{ "%.2f"|format(etf.momentum_score) }}</td>
                        <td>{{ "%.2f"|format(etf.price_momentum) }}%</td>
                        <td>{{ "%.2f"|format(etf.volume_momentum) }}x</td>
                        <td>{{ "%.1f"|format(etf.volume_price_ratio) }}%</td>
                        <td>{{ "%.1f"|format(etf.trend_strength) }}%</td>
                        <td>{{ "%.2f"|format(etf.volatility) }}%</td>
                        <td>{{ "%.2f"|format(etf.industry_weight) }}</td>
                        <td>{{ "%.2f"|format(etf.latest_close) }}</td>
                        <td>{{ "{:,}".format(etf.latest_volume) }}</td>
                        {% if etf.future_return_3d is not none %}
                        <td class="{% if etf.future_return_3d >= 0 %}positive{% else %}negative{% endif %}">{{ "%.2f"|format(etf.future_return_3d) }}%</td>
                        {% endif %}
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
    </div>

    <script>
        function switchTab(tabName) {
            document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
            document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
            document.getElementById(tabName + '-view').classList.add('active');
            event.target.classList.add('active');
        }

        function copyLogic() {
            const logicText = document.getElementById('logic-text').innerText;
            navigator.clipboard.writeText(logicText).then(function() {
                const btn = document.querySelector('.copy-btn');
                btn.innerText = '✅ 已复制!';
                btn.style.background = '#48bb78';
                setTimeout(function() {
                    btn.innerText = '📋 复制文本';
                    btn.style.background = '#667eea';
                }, 2000);
            }).catch(function(err) {
                alert('复制失败，请手动复制');
            });
        }

        {% for etf in etfs %}
        // K线图 - 使用自定义绘制
        const klineCtx{{ loop.index }} = document.getElementById('klineChart{{ loop.index }}').getContext('2d');
        const klineData{{ loop.index }} = {{ etf.kline|tojson }};
        
        // 计算价格范围
        const prices{{ loop.index }} = klineData{{ loop.index }}.flatMap(d => [d.h, d.l]);
        const minPrice{{ loop.index }} = Math.min(...prices{{ loop.index }});
        const maxPrice{{ loop.index }} = Math.max(...prices{{ loop.index }});
        const priceRange{{ loop.index }} = maxPrice{{ loop.index }} - minPrice{{ loop.index }};
        const padding{{ loop.index }} = priceRange{{ loop.index }} * 0.1;
        
        new Chart(klineCtx{{ loop.index }}, {
            type: 'bar',
            data: {
                labels: klineData{{ loop.index }}.map(d => d.x.substring(5)),
                datasets: [{
                    label: 'K线',
                    data: klineData{{ loop.index }}.map(d => ({
                        o: d.o,
                        h: d.h,
                        l: d.l,
                        c: d.c,
                        up: d.c >= d.o
                    })),
                    backgroundColor: function(context) {
                        const d = context.raw;
                        return d && d.up ? 'rgb(239, 83, 80)' : 'rgb(38, 166, 154)';
                    },
                    barPercentage: 0.7,
                    categoryPercentage: 0.9
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                return '日期: ' + klineData{{ loop.index }}[context[0].dataIndex].x;
                            },
                            label: function(context) {
                                const d = context.raw;
                                return [
                                    `开盘: ${(d.o/100).toFixed(2)}`,
                                    `最高: ${(d.h/100).toFixed(2)}`,
                                    `最低: ${(d.l/100).toFixed(2)}`,
                                    `收盘: ${(d.c/100).toFixed(2)}`
                                ];
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        ticks: {
                            maxTicksLimit: 5,
                            maxRotation: 0,
                            font: { size: 11 }
                        },
                        grid: { display: false }
                    },
                    y: {
                        min: minPrice{{ loop.index }} - padding{{ loop.index }},
                        max: maxPrice{{ loop.index }} + padding{{ loop.index }},
                        ticks: {
                            callback: function(value) {
                                return (value / 100).toFixed(2);
                            },
                            font: { size: 11 }
                        }
                    }
                }
            },
            plugins: [{
                id: 'candlestick{{ loop.index }}',
                beforeDraw: function(chart) {
                    const ctx = chart.ctx;
                    const meta = chart.getDatasetMeta(0);
                    const xScale = chart.scales.x;
                    const yScale = chart.scales.y;
                    
                    ctx.save();
                    klineData{{ loop.index }}.forEach((d, i) => {
                        const x = xScale.getPixelForValue(i);
                        const barWidth = xScale.width / klineData{{ loop.index }}.length * 0.6;
                        const yOpen = yScale.getPixelForValue(d.o);
                        const yClose = yScale.getPixelForValue(d.c);
                        const yHigh = yScale.getPixelForValue(d.h);
                        const yLow = yScale.getPixelForValue(d.l);
                        
                        const isUp = d.c >= d.o;
                        ctx.strokeStyle = isUp ? 'rgb(239, 83, 80)' : 'rgb(38, 166, 154)';
                        ctx.fillStyle = isUp ? 'rgb(239, 83, 80)' : 'rgb(38, 166, 154)';
                        ctx.lineWidth = 1.5;
                        
                        // 画影线
                        ctx.beginPath();
                        ctx.moveTo(x, yHigh);
                        ctx.lineTo(x, yLow);
                        ctx.stroke();
                        
                        // 画实体
                        const bodyTop = Math.min(yOpen, yClose);
                        const bodyHeight = Math.abs(yClose - yOpen);
                        const minBodyHeight = 1;
                        ctx.fillRect(x - barWidth/2, bodyTop, barWidth, Math.max(bodyHeight, minBodyHeight));
                    });
                    ctx.restore();
                }
            }]
        });

        // 成交量图
        const volCtx{{ loop.index }} = document.getElementById('volumeChart{{ loop.index }}').getContext('2d');
        new Chart(volCtx{{ loop.index }}, {
            type: 'bar',
            data: {
                labels: klineData{{ loop.index }}.map(d => d.x.substring(5)),
                datasets: [{
                    label: '成交量',
                    data: {{ etf.volumes|tojson }},
                    backgroundColor: 'rgba(102, 126, 234, 0.6)',
                    yAxisID: 'y'
                }, {
                    label: 'OBV',
                    data: {{ etf.obv|tojson }},
                    type: 'line',
                    borderColor: 'rgb(255, 99, 132)',
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    yAxisID: 'y1',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: true, position: 'top' } },
                scales: {
                    y: {
                        type: 'linear', display: true, position: 'left',
                        ticks: { callback: function(value) { return (value / 10000).toFixed(0) + '万'; } }
                    },
                    y1: { type: 'linear', display: true, position: 'right', grid: { drawOnChartArea: false } }
                }
            }
        });
        {% endfor %}
    </script>
</body>
</html>
"""


# 全局分析器实例
analyzer = None


@app.route('/')
def index():
    """主页"""
    global analyzer
    
    if analyzer is None:
        print("初始化 ETF 分析器...")
        analyzer = ETFWebAnalyzer()
    
    print("分析 ETF 数据...")
    top_etfs, all_etfs, date_list, stats = analyzer.analyze_all_etfs(days=30, top_n=20)
    
    return render_template_string(
        HTML_TEMPLATE,
        etfs=top_etfs,
        all_etfs=all_etfs,
        stats=stats
    )


def main():
    """主函数"""
    print("=" * 60)
    print("启动 ETF 动量分析 Web 服务 - 最终优化版")
    print("=" * 60)
    print("请访问: http://127.0.0.1:8080")
    print("=" * 60)
    
    # 初始化分析器
    global analyzer
    analyzer = ETFWebAnalyzer()
    
    # 启动 Flask 服务
    app.run(host='0.0.0.0', port=8080, debug=False)


if __name__ == '__main__':
    main()
