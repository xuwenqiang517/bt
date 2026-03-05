"""
ETF动量筛选系统 - 最终优化版
基于量价分析的ETF筛选工具，包含回测验证功能
"""
import numpy as np
import polars as pl
import akshare as ak
import json
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import time

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ETFMomentumScreener:
    """ETF动量筛选器 - 最终优化版"""
    
    def __init__(self, cache_dir: str = "./etf_cache"):
        """
        初始化筛选器
        
        Args:
            cache_dir: 缓存目录路径
        """
        self.cache_dir = cache_dir
        self.etf_list_cache_file = os.path.join(cache_dir, "etf_list.json")
        self.etf_data_cache_file = os.path.join(cache_dir, "etf_data.json")
        
        # 创建缓存目录
        os.makedirs(cache_dir, exist_ok=True)
        
        # ETF列表和数据
        self.etf_list = []
        self.etf_data = {}
        
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
    
    def load_etf_list_from_cache(self) -> bool:
        """从缓存加载ETF列表"""
        try:
            if os.path.exists(self.etf_list_cache_file):
                with open(self.etf_list_cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # 检查缓存是否过期（1天）
                    cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                    if (datetime.now() - cache_time).days < 1:
                        self.etf_list = cached_data.get('data', [])
                        logger.info(f"从缓存加载ETF列表: {len(self.etf_list)}只")
                        return True
        except Exception as e:
            logger.warning(f"加载ETF列表缓存失败: {e}")
        return False
    
    def save_etf_list_to_cache(self):
        """保存ETF列表到缓存"""
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': self.etf_list
            }
            with open(self.etf_list_cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            logger.info(f"ETF列表已缓存: {len(self.etf_list)}只")
        except Exception as e:
            logger.warning(f"保存ETF列表缓存失败: {e}")
    
    def fetch_etf_list(self) -> List[Dict]:
        """
        获取ETF列表（带缓存）
        
        Returns:
            ETF列表，每只ETF包含code和name
        """
        # 先尝试从缓存加载
        if self.load_etf_list_from_cache():
            return self.etf_list
        
        # 从akshare获取
        logger.info("从akshare获取ETF列表...")
        try:
            etf_df = ak.fund_etf_spot_em()
            self.etf_list = []
            for _, row in etf_df.iterrows():
                code = str(row['代码']).strip()
                name = str(row['名称']).strip()
                # 只保留ETF，排除LOF
                if 'ETF' in name and 'LOF' not in name:
                    self.etf_list.append({
                        'code': code,
                        'name': name
                    })
            
            logger.info(f"获取ETF列表成功: {len(self.etf_list)}只")
            self.save_etf_list_to_cache()
            return self.etf_list
            
        except Exception as e:
            logger.error(f"获取ETF列表失败: {e}")
            return []
    
    def load_etf_data_from_cache(self, code: str) -> Optional[Dict]:
        """从缓存加载单只ETF数据"""
        try:
            cache_file = os.path.join(self.cache_dir, f"etf_{code}.json")
            if os.path.exists(cache_file):
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    # 检查缓存是否过期（1天）
                    cache_time = datetime.fromisoformat(cached_data.get('timestamp', '2000-01-01'))
                    if (datetime.now() - cache_time).days < 1:
                        return cached_data.get('data')
        except Exception as e:
            logger.warning(f"加载ETF {code} 缓存失败: {e}")
        return None
    
    def save_etf_data_to_cache(self, code: str, data: Dict):
        """保存单只ETF数据到缓存"""
        try:
            cache_file = os.path.join(self.cache_dir, f"etf_{code}.json")
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'data': data
            }
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存ETF {code} 缓存失败: {e}")
    
    def fetch_etf_data(self, code: str, max_retries: int = 3) -> Optional[Dict]:
        """
        获取单只ETF历史数据（带缓存和重试）
        
        Args:
            code: ETF代码
            max_retries: 最大重试次数
            
        Returns:
            ETF历史数据字典，包含date, open, high, low, close, volume
        """
        # 先尝试从缓存加载
        cached_data = self.load_etf_data_from_cache(code)
        if cached_data:
            return cached_data
        
        # 从akshare获取（带重试）
        for attempt in range(max_retries):
            try:
                market = "sh" if code.startswith("5") else "sz"
                df = ak.fund_etf_hist_sina(symbol=f"{market}{code}")
                
                if df is None or df.empty:
                    logger.warning(f"ETF {code} 数据为空")
                    return None
                
                # 转换为字典格式
                data = {
                    'code': code,
                    'dates': df['date'].astype(str).tolist(),
                    'open': df['open'].astype(float).tolist(),
                    'high': df['high'].astype(float).tolist(),
                    'low': df['low'].astype(float).tolist(),
                    'close': df['close'].astype(float).tolist(),
                    'volume': df['volume'].astype(int).tolist()
                }
                
                # 保存到缓存
                self.save_etf_data_to_cache(code, data)
                return data
                
            except Exception as e:
                logger.warning(f"获取ETF {code} 数据失败(尝试{attempt+1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                continue
        
        logger.error(f"获取ETF {code} 数据最终失败")
        return None
    
    def extract_industry(self, name: str) -> str:
        """
        从ETF名称提取行业
        
        Args:
            name: ETF名称
            
        Returns:
            行业名称
        """
        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword in name:
                    return industry
        return '其他'
    
    def is_up_day(self, close_curr: float, close_prev: float) -> bool:
        """
        判断是否为上涨日（涨幅>0.1%）
        
        Args:
            close_curr: 当日收盘价
            close_prev: 前一日收盘价
            
        Returns:
            是否上涨
        """
        if close_prev <= 0:
            return False
        return (close_curr - close_prev) / close_prev > 0.001
    
    def validate_data(self, data: Dict) -> bool:
        """
        数据验证：检查异常值
        
        Args:
            data: ETF数据字典
            
        Returns:
            数据是否有效
        """
        closes = np.array(data['close'])
        volumes = np.array(data['volume'])
        
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
    
    def calculate_momentum(self, data: Dict, all_data: Dict = None) -> Optional[Dict]:
        """
        计算量价动量指标 - 最终优化版
        
        Args:
            data: ETF历史数据
            all_data: 全部历史数据（用于回测）
            
        Returns:
            动量指标字典，不符合条件返回None
        """
        try:
            closes = np.array(data['close'])
            opens = np.array(data['open'])
            highs = np.array(data['high'])
            lows = np.array(data['low'])
            volumes = np.array(data['volume'])
            dates = data['dates']
            code = data['code']
            
            n = len(closes)
            
            # ========== 数据预处理 ==========
            # 条件1: 上市天数>=30天
            if n < 30:
                return None
            
            # 数据验证
            if not self.validate_data(data):
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
            
            # 波动率（日收益率标准差）
            daily_returns = np.diff(closes[-11:]) / closes[-11:-1]
            if len(daily_returns) > 1:
                volatility = np.std(daily_returns) * 100
                volatility = max(min(volatility, 50.0), 0.1)
            else:
                volatility = 0.1
            
            # ========== 回测：未来3天收益率 ==========
            future_return_3d = None
            if all_data and len(all_data.get('close', [])) > len(closes):
                all_closes = np.array(all_data['close'])
                current_idx = len(closes) - 1
                if current_idx + 3 < len(all_closes):
                    future_return_3d = (all_closes[current_idx + 3] - all_closes[current_idx]) / all_closes[current_idx] * 100
            
            return {
                'code': code,
                'name': data.get('name', ''),
                'industry': self.extract_industry(data.get('name', '')),
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
                'future_return_3d': float(future_return_3d) if future_return_3d is not None else None,
                'industry_weight': 1.0,  # 将在analyze_all_etfs中计算
                'value_score': 0.0  # 将在analyze_all_etfs中计算
            }
            
        except Exception as e:
            logger.warning(f"计算ETF {data.get('code', 'unknown')} 动量失败: {e}")
            return None
    
    def analyze_all_etfs(self, max_etfs: int = None) -> Tuple[List[Dict], Dict]:
        """
        分析所有ETF的动量，带过滤和行业分散
        
        Args:
            max_etfs: 最大分析ETF数量（None表示全部）
            
        Returns:
            (筛选后的ETF列表, 统计信息字典)
        """
        # 获取ETF列表
        etf_list = self.fetch_etf_list()
        if max_etfs:
            etf_list = etf_list[:max_etfs]
        
        total_before_filter = len(etf_list)
        logger.info(f"开始分析 {total_before_filter} 只ETF...")
        
        # 获取所有ETF数据
        all_etf_data = {}
        for etf_info in etf_list:
            code = etf_info['code']
            data = self.fetch_etf_data(code)
            if data:
                data['name'] = etf_info['name']
                all_etf_data[code] = data
        
        logger.info(f"成功获取 {len(all_etf_data)} 只ETF数据")
        
        # 计算动量指标
        filtered_etfs = []
        for code, data in all_etf_data.items():
            momentum = self.calculate_momentum(data, data)
            if momentum:
                filtered_etfs.append(momentum)
        
        logger.info(f"通过过滤的ETF: {len(filtered_etfs)} 只")
        
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
        
        # 计算统计信息
        stats = {
            'total_before_filter': total_before_filter,
            'total_after_filter': len(final_etfs),
            'industry_distribution': industry_counts,
            'avg_volatility': np.mean([etf['volatility'] for etf in final_etfs]) if final_etfs else 0,
            'avg_momentum_score': np.mean([etf['momentum_score'] for etf in final_etfs]) if final_etfs else 0,
            'filter_ratio': (total_before_filter - len(final_etfs)) / total_before_filter * 100 if total_before_filter > 0 else 0
        }
        
        logger.info(f"最终筛选结果: {len(final_etfs)} 只ETF")
        return final_etfs, stats
    
    def run_backtest(self, top_etfs: List[Dict]) -> Dict:
        """
        对前10只ETF进行回测验证
        
        Args:
            top_etfs: 前N只ETF列表
            
        Returns:
            回测结果字典
        """
        backtest_results = []
        valid_returns = []
        
        for i, etf in enumerate(top_etfs[:10]):
            if etf.get('future_return_3d') is not None:
                backtest_results.append({
                    'rank': i + 1,
                    'code': etf['code'],
                    'name': etf['name'],
                    'industry': etf['industry'],
                    'value_score': etf['value_score'],
                    'future_return_3d': etf['future_return_3d']
                })
                valid_returns.append(etf['future_return_3d'])
        
        avg_return = np.mean(valid_returns) if valid_returns else 0
        
        return {
            'results': backtest_results,
            'avg_return_3d': avg_return,
            'valid_count': len(valid_returns)
        }
    
    def print_results(self, etfs: List[Dict], stats: Dict, backtest: Dict):
        """
        打印完整结果
        
        Args:
            etfs: 筛选后的ETF列表
            stats: 统计信息
            backtest: 回测结果
        """
        print("\n" + "="*100)
        print("ETF动量筛选系统 - 最终优化版 - 完整结果")
        print("="*100)
        
        # 1. 统计信息
        print("\n【一、统计信息】")
        print(f"过滤前ETF数量: {stats['total_before_filter']}")
        print(f"过滤后ETF数量: {stats['total_after_filter']}")
        print(f"过滤比例: {stats['filter_ratio']:.1f}%")
        print(f"平均波动率: {stats['avg_volatility']:.2f}%")
        print(f"平均趋势得分: {stats['avg_momentum_score']:.2f}")
        
        # 2. 行业分布
        print("\n【二、行业分布】")
        for industry, count in sorted(stats['industry_distribution'].items(), key=lambda x: x[1], reverse=True):
            print(f"  {industry}: {count}只")
        
        # 3. 完整筛选结果表格
        print("\n【三、完整筛选结果】")
        print("-"*150)
        header = f"{'排名':<6}{'代码':<10}{'名称':<20}{'行业':<10}{'20天涨幅':<10}{'10天涨幅':<10}{'5天涨幅':<10}{'1天涨幅':<10}{'性价比':<10}{'趋势得分':<10}{'价格动量':<10}{'成交量':<10}{'量价配合':<10}{'趋势强度':<10}{'波动率':<10}{'权重':<8}"
        print(header)
        print("-"*150)
        
        for i, etf in enumerate(etfs, 1):
            row = f"{i:<6}{etf['code']:<10}{etf['name'][:18]:<20}{etf['industry']:<10}" \
                  f"{etf['total_return_20d']:>8.2f}% {etf['total_return_10d']:>8.2f}% " \
                  f"{etf['total_return_5d']:>8.2f}% {etf['return_1d']:>8.2f}% " \
                  f"{etf['value_score']:>8.2f}  {etf['momentum_score']:>8.2f}  " \
                  f"{etf['price_momentum']:>8.2f}% {etf['volume_momentum']:>8.2f}x " \
                  f"{etf['volume_price_ratio']:>8.1f}% {etf['trend_strength']:>8.1f}% " \
                  f"{etf['volatility']:>8.2f}% {etf['industry_weight']:>6.2f}"
            print(row)
        
        print("-"*150)
        
        # 4. 前10只ETF回测结果
        if backtest['results']:
            print("\n【四、前10只ETF回测结果（未来3天收益率）】")
            print("-"*80)
            print(f"{'排名':<6}{'代码':<10}{'名称':<20}{'行业':<10}{'性价比':<10}{'未来3天收益':<15}")
            print("-"*80)
            
            for result in backtest['results']:
                return_str = f"{result['future_return_3d']:>10.2f}%"
                print(f"{result['rank']:<6}{result['code']:<10}{result['name'][:18]:<20}" \
                      f"{result['industry']:<10}{result['value_score']:>8.2f}  {return_str}")
            
            print("-"*80)
            print(f"前10只ETF平均未来3天收益率: {backtest['avg_return_3d']:.2f}%")
            print(f"有效回测样本数: {backtest['valid_count']}/10")
        
        print("\n" + "="*100)


def main():
    """主函数"""
    print("="*100)
    print("ETF动量筛选系统 - 最终优化版")
    print("="*100)
    
    # 创建筛选器实例
    screener = ETFMomentumScreener(cache_dir="./etf_cache")
    
    # 运行筛选（可以设置max_etfs限制数量，None表示全部）
    etfs, stats = screener.analyze_all_etfs(max_etfs=None)
    
    # 运行回测
    backtest = screener.run_backtest(etfs)
    
    # 打印结果
    screener.print_results(etfs, stats, backtest)
    
    print("\n筛选完成！")


if __name__ == '__main__':
    main()
