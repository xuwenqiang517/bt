"""
股票选股器
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

# 邮箱配置
SENDER_EMAIL = "since480@163.com"
SENDER_PASSWORD = "FP3w2trEpAqPN4x8"
SMTP_SERVER = "smtp.163.com"
SMTP_PORT = 465
RECEIVER_EMAIL = "598570789@qq.com"


class ComprehensivePicker:
    """综合选股器 - 股票"""

    def __init__(self, stock_config: str = "1|1,5,9,15,13,23,2|0|-9,6,12,6"):
        """
        初始化综合选股器

        Args:
            stock_config: 股票选股配置字符串
        """
        self.stock_config = stock_config
        self.stock_picker = StockPicker(stock_config)

    def pick(self, target_date: str = None) -> dict:
        """
        执行选股

        Returns:
            包含股票选股结果的字典
        """
        results = {
            'stocks': pl.DataFrame(),
            'date': target_date or date.today().strftime("%Y%m%d")
        }

        # 股票选股
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
        print("📋 选股结果摘要")
        print("="*80)
        print(f"日期: {results['date']}")
        print(f"股票选中: {len(results['stocks'])} 只")

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
    # stock_config = "1|3,7,8,16,15,23,2|0|-9,6,6,3"
    stock_config = "1|3,-1,10,17,-1,-1,5|0|-10,5,6,2"

    # 如果提供了参数，使用提供的配置
    if len(sys.argv) > 1:
        stock_config = sys.argv[1]

    # 创建选股器
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
