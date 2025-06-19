"""
交易所配置文件
包含各个交易所的详细信息和推荐配置
"""

# 国内可访问的交易所配置
DOMESTIC_EXCHANGES = {
    'okx': {
        'name': 'OKX (欧易)',
        'description': '全球领先的数字资产交易平台，国内可直接访问',
        'website': 'https://www.okx.com',
        'features': ['现货', '期货', '期权', '杠杆'],
        'supported_timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '12h', '1d', '1w', '1M'],
        'rate_limit': 20,  # 每秒请求数
        'recommended': True,
        'connectivity': 'excellent',
        'api_docs': 'https://www.okx.com/docs-v5/zh/'
    },
    'gate': {
        'name': 'Gate.io',
        'description': '老牌数字货币交易平台，国内可访问',
        'website': 'https://www.gate.io',
        'features': ['现货', '期货', '杠杆', '期权'],
        'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '8h', '1d', '7d'],
        'rate_limit': 10,
        'recommended': True,
        'connectivity': 'good',
        'api_docs': 'https://www.gate.io/docs/developers/apiv4/'
    },
    'kucoin': {
        'name': 'KuCoin (库币)',
        'description': '全球性数字货币交易服务平台，国内可访问',
        'website': 'https://www.kucoin.com',
        'features': ['现货', '期货', '杠杆', '期权'],
        'supported_timeframes': ['1min', '3min', '5min', '15min', '30min', '1hour', '2hour', '4hour', '6hour', '8hour', '12hour', '1day', '1week'],
        'rate_limit': 10,
        'recommended': True,
        'connectivity': 'good',
        'api_docs': 'https://docs.kucoin.com/'
    },
    'bitget': {
        'name': 'Bitget',
        'description': '专业的数字资产衍生品交易平台',
        'website': 'https://www.bitget.com',
        'features': ['现货', '期货', '跟单交易'],
        'supported_timeframes': ['1m', '5m', '15m', '30m', '1h', '4h', '12h', '1d', '3d', '1w', '1M'],
        'rate_limit': 20,
        'recommended': True,
        'connectivity': 'good',
        'api_docs': 'https://bitgetlimited.github.io/apidoc/zh/spot/'
    }
}

# 需要代理的交易所
PROXY_REQUIRED_EXCHANGES = {
    'binance': {
        'name': 'Binance (币安)',
        'description': '全球最大的数字货币交易平台（需要代理访问）',
        'website': 'https://www.binance.com',
        'features': ['现货', '期货', '期权', '杠杆', 'NFT'],
        'supported_timeframes': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M'],
        'rate_limit': 10,
        'recommended': False,  # 需要代理
        'connectivity': 'blocked',
        'proxy_required': True,
        'api_docs': 'https://binance-docs.github.io/apidocs/spot/cn/'
    },
    'huobi': {
        'name': 'Huobi (火币)',
        'description': '知名数字资产交易平台',
        'website': 'https://www.huobi.com',
        'features': ['现货', '期货', '杠杆'],
        'supported_timeframes': ['1min', '5min', '15min', '30min', '60min', '4hour', '1day', '1mon', '1week', '1year'],
        'rate_limit': 10,
        'recommended': False,
        'connectivity': 'limited',
        'api_docs': 'https://huobiapi.github.io/docs/spot/v1/cn/'
    },
    'bybit': {
        'name': 'Bybit',
        'description': '专业的加密货币衍生品交易平台',
        'website': 'https://www.bybit.com',
        'features': ['现货', '期货', '期权'],
        'supported_timeframes': ['1', '3', '5', '15', '30', '60', '120', '240', '360', '720', 'D', 'W', 'M'],
        'rate_limit': 10,
        'recommended': False,
        'connectivity': 'limited',
        'api_docs': 'https://bybit-exchange.github.io/docs/zh-cn/v5/intro'
    }
}

# 推荐配置
RECOMMENDED_CONFIG = {
    'primary_exchange': 'okx',      # 主要使用的交易所
    'backup_exchanges': ['gate', 'kucoin', 'bitget'],  # 备用交易所
    'default_timeframe': '1h',      # 默认时间周期
    'default_limit': 1000,          # 默认数据条数
    'rate_limit_buffer': 0.1,       # 速率限制缓冲时间（秒）
    'retry_times': 3,               # 重试次数
    'timeout': 30,                  # 超时时间（秒）
}

# 常用交易对
POPULAR_SYMBOLS = {
    'major': ['BTC/USDT', 'ETH/USDT', 'BNB/USDT'],
    'altcoins': ['ADA/USDT', 'DOT/USDT', 'LINK/USDT', 'UNI/USDT', 'AAVE/USDT'],
    'defi': ['SUSHI/USDT', 'COMP/USDT', 'YFI/USDT', 'CRV/USDT'],
    'layer1': ['SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'FTM/USDT'],
    'meme': ['DOGE/USDT', 'SHIB/USDT']
}

# 时间周期映射（标准化不同交易所的时间周期格式）
TIMEFRAME_MAPPING = {
    'okx': {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1H', '4h': '4H', '1d': '1D', '1w': '1W'
    },
    'gate': {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '7d'
    },
    'kucoin': {
        '1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min',
        '1h': '1hour', '4h': '4hour', '1d': '1day', '1w': '1week'
    },
    'bitget': {
        '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
        '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1w'
    }
}

def get_all_exchanges():
    """获取所有交易所配置"""
    all_exchanges = {}
    all_exchanges.update(DOMESTIC_EXCHANGES)
    all_exchanges.update(PROXY_REQUIRED_EXCHANGES)
    return all_exchanges

def get_recommended_exchanges():
    """获取推荐的交易所列表"""
    recommended = []
    for exchange_id, config in DOMESTIC_EXCHANGES.items():
        if config.get('recommended', False):
            recommended.append(exchange_id)
    return recommended

def get_exchange_timeframe(exchange_id, standard_timeframe):
    """获取交易所特定的时间周期格式"""
    mapping = TIMEFRAME_MAPPING.get(exchange_id, {})
    return mapping.get(standard_timeframe, standard_timeframe)

def print_exchange_info():
    """打印所有交易所信息"""
    print("国内可访问的交易所:")
    print("=" * 50)
    for exchange_id, config in DOMESTIC_EXCHANGES.items():
        print(f"\n{config['name']} ({exchange_id})")
        print(f"描述: {config['description']}")
        print(f"连接性: {config['connectivity']}")
        print(f"推荐: {'是' if config['recommended'] else '否'}")
        print(f"支持功能: {', '.join(config['features'])}")
        print(f"API文档: {config['api_docs']}")
    
    print(f"\n推荐使用顺序: {get_recommended_exchanges()}")

if __name__ == "__main__":
    print_exchange_info()
