"""
综合选股器 - 整合ETF和股票选股结果
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import date
import polars as pl
import numpy as np
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# 导入股票选股器
from stock_picker import StockPicker

# 导入ETF相关模块
from etf_data import ETFData

# 邮箱配置
SENDER_EMAIL = "since480@163.com"
SENDER_PASSWORD = "FP3w2trEpAqPN4x8"
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 465
RECEIVER_EMAIL = "598570789@qq.com"


class ETFPicker:
    """ETF选股器 - 基于动量筛选"""
    
    def __init__(self):
        self.etf_data = ETFData()
        self.df = self.etf_data.etf_data_df
        
        # 行业关键词映射
        self.industry_keywords = {
            '公用事业': ['公用事业'],
            '机械': ['机械', '机床', '工业母机'],
            '电力': ['电力', '电网', '绿电', '绿色电力'],
            '稀土': ['稀土'],
            '能源': ['能源', '煤炭', '石油', '油气', '央企能源'],
            '军工': ['军工', '国防', '航空航天', '航天', '通用航空'],
            '农业': ['农业', '农牧', '粮食', '畜牧'],
            '一带一路': ['一带一路'],
            '央企': ['央企', '国企'],
            '资源': ['资源', '大宗商品'],
            '有色': ['有色', '稀有金属', '工业有色', '矿业'],
            '科技': ['科技', '芯片', '半导体', '人工智能', 'AI', '计算机'],
            '医药': ['医药', '医疗', '生物医药', '创新药'],
            '消费': ['消费', '食品饮料', '白酒', '家电'],
            '金融': ['金融', '银行', '证券', '保险'],
            '地产': ['地产', '房地产', '基建', '建筑'],
            '传媒': ['传媒', '游戏', '影视'],
            '通信': ['通信', '5G', '电信'],
            '汽车': ['汽车', '新能源车', '智能汽车'],
            '材料': ['材料', '化工', '钢铁', '建材'],
            '环保': ['环保', 'ESG'],
            '红利': ['红利', '股息', '现金流'],
            '创新': ['创新', '科创'],
        }
    
    def extract_industry(self, name: str) -> str:
        """从ETF名称提取行业"""
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in name:
                    return industry
        return '其他'
    
    def is_up_day(self, close_curr: float, close_prev: float) -> bool:
        """判断是否为上涨日（涨幅>0.1%）"""
        if close_prev <= 0:
            return False
        return (close_curr - close_prev) / close_prev > 0.001
    
    def validate_data(self, closes: np.ndarray, volumes: np.ndarray) -> bool:
        """数据验证：检查异常值"""
        if np.any(closes <= 0) or np.any(volumes < 0):
            return False
        if np.any(np.isnan(closes)) or np.any(np.isinf(closes)):
            return False
        if np.any(np.isnan(volumes)) or np.any(np.isinf(volumes)):
            return False
        daily_returns = np.diff(closes) / closes[:-1]
        if np.any(np.abs(daily_returns) > 0.20):
            return False
        return True
    
    def calculate_momentum(self, group_df: pl.DataFrame) -> dict:
        """计算量价动量指标"""
        try:
            sorted_df = group_df.sort("date")
            closes = sorted_df["close"].to_numpy()
            volumes = sorted_df["volume"].to_numpy()
            code = int(sorted_df["code"][0])
            
            n = len(closes)
            
            # 数据预处理
            if n < 30:
                return None
            if not self.validate_data(closes, volumes):
                return None
            
            latest_volume = volumes[-1]
            if latest_volume < 1_000_000:
                return None
            
            total_return_20d = (closes[-1] - closes[-20]) / closes[-20] * 100
            if total_return_20d <= 0:
                return None
            
            # 计算均线
            ma5 = np.mean(closes[-5:])
            ma10 = np.mean(closes[-10:])
            ma5_vol = np.mean(volumes[-5:])
            ma10_vol = np.mean(volumes[-10:])
            
            # 基础过滤条件
            if ma5 <= ma10 or ma5_vol <= ma10_vol:
                return None
            
            # 涨幅计算
            total_return_10d = (closes[-1] - closes[-10]) / closes[-10] * 100
            total_return_5d = (closes[-1] - closes[-5]) / closes[-5] * 100
            return_1d = (closes[-1] - closes[-2]) / closes[-2] * 100
            
            # 短期风险过滤
            if return_1d < -3:
                return None
            
            # vol_ratio计算
            if len(volumes) >= 25:
                prev_20_vol_avg = np.mean(volumes[-25:-5])
            elif len(volumes) > 5:
                prev_20_vol_avg = np.mean(volumes[:-5])
            else:
                prev_20_vol_avg = ma5_vol
            
            vol_ratio = ma5_vol / prev_20_vol_avg if prev_20_vol_avg > 0 else 1.0
            vol_ratio = min(vol_ratio, 5.0)
            
            # 下跌时禁止放量
            if return_1d < 0 and vol_ratio >= 1.5:
                return None
            
            # 最近3天至少有1天上涨
            up_days_3 = sum(1 for i in range(-3, 0) if self.is_up_day(closes[i], closes[i-1]))
            if up_days_3 < 1:
                return None
            
            # 动量计算
            price_momentum = total_return_10d
            
            if price_momentum > 0:
                if return_1d > 0:
                    volume_health = vol_ratio
                else:
                    volume_health = 1 / (vol_ratio * 2) if vol_ratio > 0 else 0.5
            else:
                volume_health = 1 / vol_ratio if vol_ratio > 0 else 1.0
            
            if return_1d < -2:
                short_term_factor = 0.5
            elif return_1d < 0:
                short_term_factor = 0.8
            else:
                short_term_factor = 1.0
            
            momentum_score = price_momentum * (0.6 + 0.4 * volume_health) * short_term_factor
            
            # 辅助指标
            up_days_10 = sum(1 for i in range(-10, 0) if self.is_up_day(closes[i], closes[i-1]))
            volume_price_ratio = (up_days_10 / 10) * 100
            
            up_days_5_count = sum(1 for i in range(-5, 0) if self.is_up_day(closes[i], closes[i-1]))
            trend_strength = (up_days_5_count / 5) * 100
            
            # 波动率
            daily_returns = np.diff(closes[-11:]) / closes[-11:-1]
            if len(daily_returns) > 1:
                volatility = np.std(daily_returns) * 100
                volatility = max(min(volatility, 50.0), 0.1)
            else:
                volatility = 0.1
            
            name = self.etf_data.get_etf_name(code)
            
            return {
                'code': code,
                'name': name,
                'industry': self.extract_industry(name),
                'price_momentum': float(price_momentum),
                'volume_momentum': float(vol_ratio),
                'volume_price_ratio': float(volume_price_ratio),
                'momentum_score': float(momentum_score),
                'volatility': float(volatility),
                'trend_strength': float(trend_strength),
                'latest_close': float(closes[-1]),
                'latest_volume': int(volumes[-1]),
                'total_return_20d': float(total_return_20d),
                'total_return_10d': float(total_return_10d),
                'total_return_5d': float(total_return_5d),
                'return_1d': float(return_1d),
                'industry_weight': 1.0,
                'value_score': 0.0
            }
            
        except Exception as e:
            return None
    
    def pick(self, top_n: int = 10) -> list:
        """选出TOP N ETF"""
        # 获取所有唯一的ETF代码
        all_codes = self.df.select("code").unique().to_series().to_list()
        
        # 计算动量指标
        filtered_etfs = []
        for code in all_codes:
            group = self.df.filter(pl.col("code") == code)
            momentum = self.calculate_momentum(group)
            if momentum:
                filtered_etfs.append(momentum)
        
        # 计算行业分布和权重
        industry_counts = {}
        for etf in filtered_etfs:
            industry = etf['industry']
            industry_counts[industry] = industry_counts.get(industry, 0) + 1
        
        # 计算行业权重和综合得分
        for etf in filtered_etfs:
            industry = etf['industry']
            industry_count = industry_counts.get(industry, 1)
            industry_weight = 1.0 / industry_count
            etf['industry_weight'] = industry_weight
            etf['value_score'] = etf['momentum_score'] * (1 - etf['volatility'] / 100) * industry_weight
        
        # 按综合性价比得分排序
        filtered_etfs.sort(key=lambda x: (x['value_score'], x['latest_volume']), reverse=True)
        
        # 每个行业最多保留3只ETF
        industry_limits = {}
        final_etfs = []
        for etf in filtered_etfs:
            industry = etf['industry']
            if industry_limits.get(industry, 0) < 3:
                final_etfs.append(etf)
                industry_limits[industry] = industry_limits.get(industry, 0) + 1
        
        return final_etfs[:top_n]


class ComprehensivePicker:
    """综合选股器 - 整合ETF和股票"""
    
    def __init__(self, stock_config: str = "1|1,5,9,15,13,23,2|0|-9,6,12,6"):
        """
        初始化综合选股器
        
        Args:
            stock_config: 股票选股配置字符串
        """
        self.stock_config = stock_config
        self.etf_picker = ETFPicker()
        self.stock_picker = StockPicker(stock_config)
    
    def pick(self, target_date: str = None) -> dict:
        """
        执行综合选股
        
        Returns:
            包含ETF和股票选股结果的字典
        """
        results = {
            'etf': [],
            'stocks': pl.DataFrame(),
            'date': target_date or date.today().strftime("%Y%m%d")
        }
        
        # 1. ETF选股
        print("\n" + "="*80)
        print("📊 ETF动量选股 (TOP 10)")
        print("="*80)
        etf_results = self.etf_picker.pick(top_n=10)
        results['etf'] = etf_results
        
        if etf_results:
            print(f"\n{'排名':<6}{'代码':<10}{'名称':<22}{'行业':<10}{'性价比':<10}{'趋势得分':<10}{'20天涨幅':<10}{'10天涨幅':<10}{'1天涨幅':<10}")
            print("-"*100)
            for i, etf in enumerate(etf_results, 1):
                print(f"{i:<6}{etf['code']:<10}{etf['name'][:20]:<22}{etf['industry']:<10}"
                      f"{etf['value_score']:>8.2f}  {etf['momentum_score']:>8.2f}  "
                      f"{etf['total_return_20d']:>7.2f}% {etf['total_return_10d']:>7.2f}% {etf['return_1d']:>7.2f}%")
        else:
            print("😢 没有符合条件的ETF")
        
        # 2. 股票选股
        print("\n" + "="*80)
        print("📈 股票选股")
        print("="*80)
        print(f"配置: {self.stock_config}")
        
        stock_results = self.stock_picker.pick(target_date)
        results['stocks'] = stock_results
        
        return results
    
    def format_results_html(self, results: dict) -> str:
        """格式化结果为HTML邮件内容 - 适配手机竖屏"""
        html = """
        <html>
        <head>
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                * { margin: 0; padding: 0; box-sizing: border-box; }
                body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; font-size: 13px; color: #333; background: #f5f5f5; }
                .container { max-width: 100%; padding: 8px; }
                h2 { font-size: 15px; color: #333; margin: 12px 0 8px 0; padding: 8px; background: #e8e8e8; border-radius: 4px; }
                table { width: 100%; border-collapse: collapse; background: white; font-size: 12px; }
                th { background: #667eea; color: white; padding: 6px 4px; text-align: center; font-weight: 600; white-space: nowrap; }
                td { padding: 5px 3px; text-align: center; border-bottom: 1px solid #eee; white-space: nowrap; }
                tr:nth-child(even) { background: #fafafa; }
                .positive { color: #dc3545; font-weight: bold; }
                .negative { color: #28a745; font-weight: bold; }
                .code { font-family: 'Courier New', monospace; font-weight: bold; }
                .name { text-align: left; max-width: 80px; overflow: hidden; text-overflow: ellipsis; }
                .score { font-weight: bold; color: #667eea; }
                .summary { background: white; padding: 10px; margin-top: 12px; border-radius: 4px; text-align: center; }
                .summary p { display: inline-block; margin: 0 15px; }
            </style>
        </head>
        <body>
            <div class="container">
        """
        
        # ETF部分 - 对齐时间顺序：20,10,5,1日
        if results['etf']:
            html += """
                <h2>🏆 ETF TOP10</h2>
                <table>
                    <tr>
                        <th>#</th>
                        <th>代码</th>
                        <th>名称</th>
                        <th>得分</th>
                        <th>20日</th>
                        <th>10日</th>
                        <th>5日</th>
                        <th>1日</th>
                    </tr>
            """
            for i, etf in enumerate(results['etf'], 1):
                html += f"""
                    <tr>
                        <td>{i}</td>
                        <td class="code">{etf['code']}</td>
                        <td class="name">{etf['name'][:8]}</td>
                        <td class="score">{etf['value_score']:.1f}</td>
                        <td class="{'positive' if etf['total_return_20d'] >= 0 else 'negative'}">{etf['total_return_20d']:.1f}%</td>
                        <td class="{'positive' if etf['total_return_10d'] >= 0 else 'negative'}">{etf['total_return_10d']:.1f}%</td>
                        <td class="{'positive' if etf['total_return_5d'] >= 0 else 'negative'}">{etf['total_return_5d']:.1f}%</td>
                        <td class="{'positive' if etf['return_1d'] >= 0 else 'negative'}">{etf['return_1d']:.1f}%</td>
                    </tr>
                """
            html += "</table>"
        
        # 股票部分 - 对齐时间顺序：5,3,1日，代码补0，添加名称
        if not results['stocks'].is_empty():
            html += """
                <h2>📈 股票</h2>
                <table>
                    <tr>
                        <th>代码</th>
                        <th>名称</th>
                        <th>连涨</th>
                        <th>5日</th>
                        <th>3日</th>
                        <th>当日</th>
                    </tr>
            """
            for row in results['stocks'].iter_rows(named=True):
                code_str = str(row['code']).zfill(6)
                name = self.stock_picker.data.get_stock_name(row['code'])[:8]
                html += f"""
                    <tr>
                        <td class="code">{code_str}</td>
                        <td class="name">{name}</td>
                        <td>{row['consecutive_up_days']}天</td>
                        <td class="{'positive' if row['change_5d'] >= 0 else 'negative'}">{row['change_5d']:.1f}%</td>
                        <td class="{'positive' if row['change_3d'] >= 0 else 'negative'}">{row['change_3d']:.1f}%</td>
                        <td class="{'positive' if row['change_pct'] >= 0 else 'negative'}">{row['change_pct']:.1f}%</td>
                    </tr>
                """
            html += "</table>"
        
        html += """
            </div>
        </body>
        </html>
        """
        return html
    
    def send_email(self, results: dict):
        """发送选股结果邮件"""
        try:
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"📊 每日选股报告 - {results['date']}"
            msg['From'] = SENDER_EMAIL
            msg['To'] = RECEIVER_EMAIL
            
            # 生成HTML内容
            html_content = self.format_results_html(results)
            msg.attach(MIMEText(html_content, 'html', 'utf-8'))
            
            # 连接SMTP服务器并发送
            with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
                server.login(SENDER_EMAIL, SENDER_PASSWORD)
                server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
            
            print(f"\n📧 邮件已发送至: {RECEIVER_EMAIL}")
            return True
            
        except Exception as e:
            print(f"\n❌ 邮件发送失败: {e}")
            return False
    
    def print_summary(self, results: dict):
        """打印选股结果摘要"""
        print("\n" + "="*80)
        print("📋 综合选股结果摘要")
        print("="*80)
        print(f"日期: {results['date']}")
        print(f"ETF选中: {len(results['etf'])} 只")
        print(f"股票选中: {len(results['stocks'])} 只")
        
        if results['etf']:
            print("\n🏆 ETF TOP 3:")
            for i, etf in enumerate(results['etf'][:3], 1):
                print(f"  {i}. {etf['name']}({etf['code']}) - 性价比:{etf['value_score']:.2f}")
        
        if not results['stocks'].is_empty():
            print("\n🏆 股票 TOP 3:")
            for i, row in enumerate(results['stocks'].head(3).iter_rows(named=True), 1):
                print(f"  {i}. {row['code']} - 连涨:{row['consecutive_up_days']}天 "
                      f"3日:{row['change_3d']:.2f}% 5日:{row['change_5d']:.2f}%")
        
        print("="*80)


def main():
    """主函数"""
    import sys
    from stock_calendar import StockCalendar
    from datetime import datetime

    # 检查今天是否是交易日
    calendar = StockCalendar()
    now = datetime.now()
    today_int = int(now.strftime("%Y%m%d"))

    if today_int not in calendar.date_to_index:
        print(f"📅 今天 ({today_int}) 不是交易日，跳过选股")
        return

    # 默认股票配置
    # stock_config = "1|1,5,9,15,13,23,2|0|-9,6,12,6"
    stock_config = "1|3,7,8,16,15,23,2|0|-9,6,6,3"

    # 如果提供了参数，使用提供的配置
    if len(sys.argv) > 1:
        stock_config = sys.argv[1]

    # 创建综合选股器
    picker = ComprehensivePicker(stock_config=stock_config)

    # 执行选股
    target_date = None
    if len(sys.argv) > 2:
        target_date = sys.argv[2]

    results = picker.pick(target_date)

    # 打印摘要
    picker.print_summary(results)

    # 发送邮件
    picker.send_email(results)


if __name__ == '__main__':
    main()
