import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
from typing import Optional, Dict, List, Union
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CryptoDataProcessor:
    """
    加密货币数据获取和处理类
    支持多个交易所，包括国内可访问的交易所
    使用ccxt库获取各交易所的市场数据，并提供数据处理功能
    """
    
    # 支持的交易所配置
    SUPPORTED_EXCHANGES = {
        'okx': {
            'name': 'OKX',
            'description': '欧易交易所，国内可访问',
            'class': ccxt.okx
        },
        'huobi': {
            'name': 'Huobi',
            'description': '火币交易所',
            'class': ccxt.huobi
        },
        'gate': {
            'name': 'Gate.io',
            'description': 'Gate.io交易所，国内可访问',
            'class': ccxt.gate
        },
        'kucoin': {
            'name': 'KuCoin',
            'description': '库币交易所，国内可访问',
            'class': ccxt.kucoin
        },
        'bybit': {
            'name': 'Bybit',
            'description': 'Bybit交易所',
            'class': ccxt.bybit
        },
        'bitget': {
            'name': 'Bitget',
            'description': 'Bitget交易所，国内可访问',
            'class': ccxt.bitget
        },
        'binance': {
            'name': 'Binance',
            'description': '币安交易所（需要代理）',
            'class': ccxt.binance
        }
    }
    
    def __init__(self, exchange_id: str = 'okx', api_key: Optional[str] = None, 
                 api_secret: Optional[str] = None, passphrase: Optional[str] = None,
                 sandbox: bool = False, rate_limit: bool = True, proxy: Optional[str] = None):
        """
        初始化加密货币数据处理器
        
        Args:
            exchange_id: 交易所ID ('okx', 'huobi', 'gate', 'kucoin', 'bybit', 'bitget', 'binance')
            api_key: API密钥（可选，用于私有接口）
            api_secret: API密钥（可选，用于私有接口）
            passphrase: 密码短语（某些交易所需要，如OKX）
            sandbox: 是否使用沙盒环境
            rate_limit: 是否启用速率限制
            proxy: 代理设置（可选）
        """
        if exchange_id not in self.SUPPORTED_EXCHANGES:
            raise ValueError(f"不支持的交易所: {exchange_id}. 支持的交易所: {list(self.SUPPORTED_EXCHANGES.keys())}")
        
        self.exchange_id = exchange_id
        self.exchange_info = self.SUPPORTED_EXCHANGES[exchange_id]
        
        # 构建交易所配置
        config = {
            'apiKey': api_key,
            'secret': api_secret,
            'sandbox': sandbox,
            'rateLimit': rate_limit,
            'enableRateLimit': rate_limit,
        }
        
        # 某些交易所需要passphrase
        if passphrase and exchange_id in ['okx', 'kucoin']:
            config['password'] = passphrase
        
        # 代理设置
        if proxy:
            config['proxies'] = {'http': proxy, 'https': proxy}
        
        # 交易所特定配置
        if exchange_id == 'okx':
            config['options'] = {'defaultType': 'spot'}
        elif exchange_id == 'binance':
            config['options'] = {'defaultType': 'spot'}
        
        # 初始化交易所
        exchange_class = self.exchange_info['class']
        self.exchange = exchange_class(config)
        
        # 检查交易所是否支持所需功能
        if not self.exchange.has['fetchOHLCV']:
            raise Exception(f"交易所 {self.exchange_info['name']} 不支持获取OHLCV数据")
        
        logger.info(f"{self.exchange_info['name']} 数据处理器初始化成功")
        logger.info(f"交易所描述: {self.exchange_info['description']}")
    
    @classmethod
    def list_supported_exchanges(cls) -> Dict[str, Dict]:
        """
        列出所有支持的交易所
        
        Returns:
            Dict: 支持的交易所信息
        """
        return cls.SUPPORTED_EXCHANGES
    
    @classmethod
    def get_recommended_exchanges(cls) -> List[str]:
        """
        获取推荐的国内可访问交易所
        
        Returns:
            List[str]: 推荐的交易所ID列表
        """
        return ['okx', 'gate', 'kucoin', 'bitget']
    
    def get_symbols(self) -> List[str]:
        """
        获取所有可用的交易对
        
        Returns:
            List[str]: 交易对列表
        """
        try:
            markets = self.exchange.load_markets()
            symbols = list(markets.keys())
            logger.info(f"获取到 {len(symbols)} 个交易对")
            return symbols
        except Exception as e:
            logger.error(f"获取交易对失败: {e}")
            return []
    
    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """
        获取单个交易对的ticker数据
        
        Args:
            symbol: 交易对符号，如 'BTC/USDT'
            
        Returns:
            Dict: ticker数据
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            logger.info(f"获取 {symbol} ticker数据成功")
            return ticker
        except Exception as e:
            logger.error(f"获取 {symbol} ticker数据失败: {e}")
            return None
    
    def get_all_tickers(self) -> Optional[Dict]:
        """
        获取所有交易对的ticker数据
        
        Returns:
            Dict: 所有ticker数据
        """
        try:
            tickers = self.exchange.fetch_tickers()
            logger.info(f"获取所有ticker数据成功，共 {len(tickers)} 个")
            return tickers
        except Exception as e:
            logger.error(f"获取所有ticker数据失败: {e}")
            return None
    
    def get_ohlcv(self, symbol: str, timeframe: str = '1d', limit: int = 100,
                  since: Optional[int] = None) -> Optional[pd.DataFrame]:
        """
        获取OHLCV数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架 ('1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M')
            limit: 数据条数限制
            since: 开始时间戳（毫秒）
            
        Returns:
            pd.DataFrame: OHLCV数据
        """
        try:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since, limit)
            
            # 转换为DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            
            logger.info(f"获取 {symbol} {timeframe} OHLCV数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} OHLCV数据失败: {e}")
            return None
    
    def get_historical_data(self, symbol: str, timeframe: str = '1d', 
                           days: int = 30) -> Optional[pd.DataFrame]:
        """
        获取历史数据
        
        Args:
            symbol: 交易对符号
            timeframe: 时间框架
            days: 历史天数
            
        Returns:
            pd.DataFrame: 历史数据
        """
        try:
            # 计算开始时间
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            while True:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, current_since, 1000)
                
                if not ohlcv:
                    break
                
                all_data.extend(ohlcv)
                
                # 更新since为最后一条数据的时间戳
                current_since = ohlcv[-1][0] + 1
                
                # 如果获取的数据少于1000条，说明已经到达最新数据
                if len(ohlcv) < 1000:
                    break
                
                # 添加延迟以避免速率限制
                time.sleep(0.1)
            
            # 转换为DataFrame
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df = df.drop_duplicates()  # 去重
            df = df.sort_index()  # 按时间排序
            
            logger.info(f"获取 {symbol} {days}天历史数据成功，共 {len(df)} 条")
            return df
        except Exception as e:
            logger.error(f"获取 {symbol} 历史数据失败: {e}")
            return None
    
    def get_order_book(self, symbol: str, limit: int = 100) -> Optional[Dict]:
        """
        获取订单簿数据
        
        Args:
            symbol: 交易对符号
            limit: 深度限制
            
        Returns:
            Dict: 订单簿数据
        """
        try:
            order_book = self.exchange.fetch_order_book(symbol, limit)
            logger.info(f"获取 {symbol} 订单簿数据成功")
            return order_book
        except Exception as e:
            logger.error(f"获取 {symbol} 订单簿数据失败: {e}")
            return None
    
    def get_trades(self, symbol: str, limit: int = 100) -> Optional[List[Dict]]:
        """
        获取最近交易记录
        
        Args:
            symbol: 交易对符号
            limit: 记录数限制
            
        Returns:
            List[Dict]: 交易记录
        """
        try:
            trades = self.exchange.fetch_trades(symbol, limit=limit)
            logger.info(f"获取 {symbol} 交易记录成功，共 {len(trades)} 条")
            return trades
        except Exception as e:
            logger.error(f"获取 {symbol} 交易记录失败: {e}")
            return None
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标
        
        Args:
            df: OHLCV数据
            
        Returns:
            pd.DataFrame: 包含技术指标的数据
        """
        try:
            # 移动平均线
            df['MA5'] = df['close'].rolling(window=5).mean()
            df['MA10'] = df['close'].rolling(window=10).mean()
            df['MA20'] = df['close'].rolling(window=20).mean()
            df['MA50'] = df['close'].rolling(window=50).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = df['close'].ewm(span=12).mean()
            exp2 = df['close'].ewm(span=26).mean()
            df['MACD'] = exp1 - exp2
            df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
            
            # 布林带
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            bb_std = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
            df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
            
            # 价格变化率
            df['price_change'] = df['close'].pct_change()
            df['price_change_5d'] = df['close'].pct_change(periods=5)
            
            # 成交量移动平均
            df['volume_MA'] = df['volume'].rolling(window=20).mean()
            
            logger.info("技术指标计算完成")
            return df
        except Exception as e:
            logger.error(f"计算技术指标失败: {e}")
            return df
    
    def save_to_csv(self, df: pd.DataFrame, filename: str, data_type: str = 'raw', 
                    custom_path: str = None):
        """
        保存数据到CSV文件
        
        Args:
            df: 数据DataFrame
            filename: 文件名
            data_type: 数据类型 ('raw', 'processed', 'analysis')
            custom_path: 自定义保存路径（可选）
        """
        try:
            import os
            
            if custom_path:
                # 如果指定了自定义路径，使用自定义路径
                base_path = custom_path
            else:
                # 使用项目根目录下的datasets目录
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                base_path = os.path.join(project_root, 'datasets', data_type)
            
            os.makedirs(base_path, exist_ok=True)
            
            filepath = os.path.join(base_path, filename)
            df.to_csv(filepath)
            logger.info(f"数据已保存到 {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"保存数据失败: {e}")
            return None
    
    def get_data_directory(self, data_type: str = 'raw') -> str:
        """
        获取数据目录路径
        
        Args:
            data_type: 数据类型 ('raw', 'processed', 'analysis')
            
        Returns:
            str: 数据目录路径
        """
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, 'datasets', data_type)
    
    def get_market_summary(self, symbols: List[str] = None) -> pd.DataFrame:
        """
        获取市场概览
        
        Args:
            symbols: 交易对列表，如果为None则获取所有主要交易对
            
        Returns:
            pd.DataFrame: 市场概览数据
        """
        if symbols is None:
            # 默认主要交易对
            symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
        
        summary_data = []
        
        for symbol in symbols:
            try:
                ticker = self.get_ticker(symbol)
                if ticker:
                    summary_data.append({
                        'symbol': symbol,
                        'price': ticker['last'],
                        'change_24h': ticker['percentage'],
                        'volume_24h': ticker['quoteVolume'],
                        'high_24h': ticker['high'],
                        'low_24h': ticker['low']
                    })
                time.sleep(0.1)  # 避免速率限制
            except Exception as e:
                logger.error(f"获取 {symbol} 市场数据失败: {e}")
                continue
        
        df = pd.DataFrame(summary_data)
        logger.info(f"市场概览获取完成，共 {len(df)} 个交易对")
        return df


# 使用示例和测试代码
if __name__ == "__main__":
    # 显示支持的交易所
    print("支持的交易所:")
    for exchange_id, info in CryptoDataProcessor.list_supported_exchanges().items():
        print(f"- {exchange_id}: {info['name']} - {info['description']}")
    
    print(f"\n推荐的国内可访问交易所: {CryptoDataProcessor.get_recommended_exchanges()}")
    
    # 初始化数据处理器（使用OKX作为默认交易所）
    try:
        processor = CryptoDataProcessor(exchange_id='okx')
        
        # 获取BTC/USDT的日线数据
        btc_data = processor.get_ohlcv('BTC/USDT', '1d', 100)
        if btc_data is not None:
            print("\nBTC/USDT 最新数据:")
            print(btc_data.tail())
            
            # 计算技术指标
            btc_data_with_indicators = processor.calculate_technical_indicators(btc_data)
            print("\n技术指标:")
            print(btc_data_with_indicators[['close', 'MA20', 'RSI', 'MACD']].tail())
            
            # 保存数据
            processor.save_to_csv(btc_data_with_indicators, 'btc_usdt_daily_okx.csv')
        
        # 获取市场概览
        market_summary = processor.get_market_summary()
        print("\n市场概览:")
        print(market_summary)
        
    except Exception as e:
        print(f"初始化失败: {e}")
        print("请尝试其他交易所或检查网络连接")
