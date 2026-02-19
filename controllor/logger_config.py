import sys
import re
from pathlib import Path

# 默认日志文件路径
DEFAULT_LOG_PATH = Path(__file__).resolve().parent.parent / 'data' / 'log.txt'

# ANSI 颜色代码正则表达式
ANSI_ESCAPE_PATTERN = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

class Logger:
    """日志类：同时输出到控制台和文件，文件中去颜色"""
    _instance = None
    
    def __new__(cls, filepath=None):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, filepath=None):
        if self._initialized:
            return
        
        if filepath is None:
            filepath = DEFAULT_LOG_PATH
        
        self.terminal = sys.stdout
        self.log_file = open(filepath, 'w', encoding='utf-8')
        self._initialized = True
    
    def write(self, message):
        # 控制台输出带颜色
        self.terminal.write(message)
        # 文件输出去除颜色代码，并添加视觉标记
        clean_message = ANSI_ESCAPE_PATTERN.sub('', message)
        # 为不同类型的日志添加标记
        marked_message = self._add_log_markers(clean_message)
        self.log_file.write(marked_message)
        self.log_file.flush()
    
    def _add_log_markers(self, message: str) -> str:
        """为日志消息添加视觉标记"""
        # 买入标记
        if '买入' in message and '卖出' not in message:
            message = message.replace('日期 ', '🟢 买入 | 日期 ')
        # 卖出标记 - 统一用蓝色，盈亏部分单独标记
        elif '卖出' in message and '盈亏' in message:
            # 提取盈亏部分
            profit_part = message.split('盈亏')[1].split(',')[0] if '盈亏' in message else ''
            is_profit = '-' not in profit_part
            # 统一前缀
            message = message.replace('日期 ', '🔵 卖出 | 日期 ')
            # 在盈亏后添加颜色标记（亏的用绿色，赚的用红色）
            if is_profit:
                message = message.replace('盈亏 ', '盈亏\U0001F534 ')
            else:
                message = message.replace('盈亏 ', '盈亏\U0001F7E2 ')
        # 卖出标记（无盈亏信息）
        elif '卖出' in message:
            message = message.replace('日期 ', '🔵 卖出 | 日期 ')
        # 持有标记
        elif '持有' in message and '总市值' not in message:
            message = message.replace('日期 ', '📊 持有 | 日期 ')
        # 结算标记
        elif '总市值' in message:
            message = message.replace('日期 ', '💰 结算 | 日期 ')
        # 周期统计
        elif '时间周期:' in message:
            message = '\n' + '='*50 + '\n📈 ' + message
        elif '总收益率:' in message or '胜率:' in message:
            message = '⭐ ' + message

        return message
    
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    
    def close(self):
        if hasattr(self, 'log_file') and self.log_file:
            self.log_file.close()

# 全局日志实例
logger = Logger()

# 重定向 stdout
sys.stdout = logger

def get_logger():
    """获取日志实例"""
    return logger
