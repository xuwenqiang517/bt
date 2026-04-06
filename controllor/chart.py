"""
图表绘制模块 - 负责回测结果的可视化展示
将绘图逻辑与核心回测逻辑分离，提高代码可维护性
"""
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re
from pathlib import Path
import os


def draw_trade_details(trades_history, daily_values, title, stock_data=None, chain_debug=False):
    """绘制交易明细图表，在资金曲线上直接标注完整交易信息

    Args:
        trades_history: 交易历史记录列表
        daily_values: 每日资金数据列表
        title: 图表标题
        stock_data: 股票数据对象（可选，用于获取股票名称）
        chain_debug: 是否打印调试信息
    """
    if not trades_history or not daily_values:
        if chain_debug:
            print("没有交易数据，跳过交易明细图")
        return

    data_dir = Path(__file__).resolve().parent.parent / "./data"
    os.makedirs(data_dir, exist_ok=True)

    safe_title = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fa5]', '_', title)
    safe_title = safe_title[:50]
    file_name = f"{safe_title}.png"
    save_path = data_dir / file_name

    dates = [f"{dv.date:08d}" for dv in daily_values]
    values = [dv.value / 100 for dv in daily_values]
    date_to_idx = {dv.date: i for i, dv in enumerate(daily_values)}

    x_indices = list(range(len(dates)))
    idx_to_date = {i: dates[i] for i in range(len(dates))}

    stock_holdings = {}
    for dv in daily_values:
        date = dv.date
        if date not in date_to_idx:
            continue
        date_idx = date_to_idx[date]
        holdings = dv.holdings
        for h in holdings:
            code = h.code
            profit_rate = h.profit_rate * 100
            if code not in stock_holdings:
                stock_holdings[code] = []
            stock_holdings[code].append((date_idx, profit_rate))

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'WenQuanYi Micro Hei']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax1 = plt.subplots(figsize=(42, 18))

    ax1.plot(x_indices, values, label='总资产', linewidth=2, color='navy', alpha=0.8)

    sold_codes = {trade.code for trade in trades_history}

    last_day_data = daily_values[-1] if daily_values else None
    unsold_holdings = []
    if last_day_data:
        last_date = last_day_data.date
        last_holdings = last_day_data.holdings
        for h in last_holdings:
            if h.code not in sold_codes:
                unsold_holdings.append({
                    'code': h.code,
                    'buy_price': h.buy_price,
                    'current_price': h.close_price,
                    'profit_rate': h.profit_rate,
                    'buy_day': h.buy_day if h.buy_day else last_date,
                    'current_day': last_date
                })

    for i, trade in enumerate(trades_history):
        sell_date = trade.date
        buy_date = trade.buy_date
        if sell_date not in date_to_idx or buy_date not in date_to_idx:
            continue

        sell_idx = date_to_idx[sell_date]
        buy_idx = date_to_idx[buy_date]
        x_sell = x_indices[sell_idx]
        y_sell = values[sell_idx]
        x_buy = x_indices[buy_idx]
        y_buy = values[buy_idx]

        code = trade.code
        buy_price = trade.buy_price / 100
        sell_price = trade.sell_price / 100
        profit_rate = trade.profit_rate * 100
        profit = trade.profit / 100
        reason = trade.reason

        hold_days = sell_idx - buy_idx

        if stock_data:
            stock_name = stock_data.get_stock_name(code)
            name_short = stock_name[:4] if len(stock_name) > 4 else stock_name
        else:
            name_short = str(code)[-6:]

        simple_reason = ''
        if '止损' in reason:
            simple_reason = '止损'
        elif '到期' in reason:
            if '盈利' in reason or profit > 0:
                simple_reason = '到期盈利'
            else:
                simple_reason = '到期亏损'
        elif '止盈' in reason or '回落' in reason:
            if '开盘回落' in reason:
                simple_reason = '开盘回落止盈'
            elif '盘中回落' in reason:
                simple_reason = '盘中回落止盈'
            else:
                simple_reason = '回落止盈'

        trade_color = 'red' if profit > 0 else 'green'

        ax1.scatter([x_buy], [y_buy], color='blue', marker='>', s=100, zorder=5)
        ax1.scatter([x_sell], [y_sell], color='gold', marker='s', s=100, zorder=5)

        is_top = i % 2 == 0

        buy_date_str = str(buy_date)[4:]
        sell_date_str = str(sell_date)[4:]

        full_label = (
            f'{code} {name_short}\n'
            f'买:{buy_date_str} {buy_price:.2f}元\n'
            f'卖:{sell_date_str} {sell_price:.2f}元\n'
            f'{profit_rate:+.1f}% | {hold_days}天 | {simple_reason}'
        )

        mid_x = (x_buy + x_sell) / 2
        mid_y = (y_buy + y_sell) / 2

        trade_span = x_sell - x_buy
        base_offset = max(60, min(100, trade_span * 8))

        if is_top:
            y_offset = base_offset + (i % 3) * 35
            va_align = 'bottom'
        else:
            y_offset = -base_offset - (i % 3) * 35
            va_align = 'top'

        box_x = mid_x
        box_y = mid_y + y_offset * (max(values) - min(values)) / 800 if len(values) > 1 else mid_y + y_offset

        if is_top:
            rad_buy = 0.15
            rad_sell = -0.15
        else:
            rad_buy = -0.15
            rad_sell = 0.15

        ax1.annotate('', xy=(x_buy, y_buy), xytext=(box_x, box_y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, connectionstyle=f'arc3,rad={rad_buy}'))

        ax1.annotate('', xy=(x_sell, y_sell), xytext=(box_x, box_y),
                    arrowprops=dict(arrowstyle='->', color='orange', lw=1.2, connectionstyle=f'arc3,rad={rad_sell}'))

        ax1.text(box_x, box_y, full_label, fontsize=7, color='black',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.95,
                         edgecolor=trade_color, lw=2))

    unsold_start_idx = len(trades_history)
    for j, holding in enumerate(unsold_holdings):
        i = unsold_start_idx + j
        code = holding['code']
        buy_price = holding['buy_price'] / 100
        current_price = holding['current_price'] / 100
        profit_rate = holding['profit_rate'] * 100
        buy_day = holding['buy_day']
        current_day = holding['current_day']

        if buy_day not in date_to_idx or current_day not in date_to_idx:
            continue

        buy_idx = date_to_idx[buy_day]
        current_idx = date_to_idx[current_day]
        x_buy = x_indices[buy_idx]
        y_buy = values[buy_idx]
        x_current = x_indices[current_idx]
        y_current = values[current_idx]

        hold_days = current_idx - buy_idx

        if stock_data:
            stock_name = stock_data.get_stock_name(code)
            name_short = stock_name[:4] if len(stock_name) > 4 else stock_name
        else:
            name_short = str(code)[-6:]

        trade_color = 'red' if profit_rate > 0 else 'green'

        ax1.scatter([x_buy], [y_buy], color='blue', marker='>', s=100, zorder=5)
        ax1.scatter([x_current], [y_current], color='purple', marker='o', s=100, zorder=5)

        is_top = i % 2 == 0

        buy_date_str = str(buy_day)[4:]
        current_date_str = str(current_day)[4:]

        full_label = (
            f'{code} {name_short}\n'
            f'买:{buy_date_str} {buy_price:.2f}元\n'
            f'持:{current_date_str} {current_price:.2f}元\n'
            f'{profit_rate:+.1f}% | {hold_days}天 | 持仓中'
        )

        mid_x = (x_buy + x_current) / 2
        mid_y = (y_buy + y_current) / 2

        trade_span = x_current - x_buy
        base_offset = max(60, min(100, trade_span * 8))

        if is_top:
            y_offset = base_offset + (i % 3) * 35
        else:
            y_offset = -base_offset - (i % 3) * 35

        box_x = mid_x
        box_y = mid_y + y_offset * (max(values) - min(values)) / 800 if len(values) > 1 else mid_y + y_offset

        if is_top:
            rad_buy = 0.15
            rad_current = -0.15
        else:
            rad_buy = -0.15
            rad_current = 0.15

        ax1.annotate('', xy=(x_buy, y_buy), xytext=(box_x, box_y),
                    arrowprops=dict(arrowstyle='->', color='blue', lw=1.2, connectionstyle=f'arc3,rad={rad_buy}'))

        ax1.annotate('', xy=(x_current, y_current), xytext=(box_x, box_y),
                    arrowprops=dict(arrowstyle='->', color='purple', lw=1.2, connectionstyle=f'arc3,rad={rad_current}'))

        ax1.text(box_x, box_y, full_label, fontsize=7, color='black',
                ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.95,
                         edgecolor=trade_color, lw=2, linestyle='--'))

    ax1.set_ylabel('资金（元）', fontsize=12, color='navy')
    ax1.set_xlabel('日期', fontsize=12)
    ax1.set_title(f'{title} - 资金曲线与交易明细（共{len(trades_history)}笔交易）', fontsize=16)
    ax1.grid(True, alpha=0.3)
    ax1.ticklabel_format(style='plain', axis='y')

    if len(dates) > 40:
        step = len(dates) // 40
        tick_indices = x_indices[::step]
        tick_labels = dates[::step]
        ax1.set_xticks(tick_indices)
        ax1.set_xticklabels(tick_labels)
    else:
        ax1.set_xticks(x_indices)
        ax1.set_xticklabels(dates)
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right', fontsize=9)

    legend_elements = [
        Line2D([0], [0], marker='>', color='w', markerfacecolor='blue', markersize=10, label='买入点'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gold', markersize=10, label='卖出点'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, label='持仓中'),
        Line2D([0], [0], color='navy', lw=2, label='总资产')
    ]
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)

    if chain_debug:
        print(f"交易明细图表已保存到: {save_path}")