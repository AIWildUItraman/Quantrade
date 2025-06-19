# 量化交易系统 (Quantitative Trading System)

🚀 一个支持国内可访问交易所的加密货币量化交易系统，专注于数据获取、技术分析和策略实现。

## 📋 系统概述

本系统旨在为量化交易者提供一个完整的工具链，包括：
- 🔌 多交易所数据获取
- 📊 技术指标计算
- 💾 数据管理和存储
- 📈 交易策略实现

特别优化了国内访问体验，优先支持国内可直接访问的交易所。

## 🔧 主要功能

### 1. 多交易所支持
- ✅ **OKX (欧易)** - 完美支持，推荐首选
- ✅ **Gate.io** - 支持国内访问
- ✅ **KuCoin (库币)** - 支持国内访问
- ✅ **Bitget** - 支持国内访问
- 🔄 **Binance (币安)** - 需要代理访问

### 2. 数据获取与处理
- 实时价格数据
- OHLCV历史数据 (各种时间周期)
- 订单簿数据
- 交易记录
- 技术指标计算 (MA, RSI, MACD, 布林带等)

### 3. 数据存储与管理
- 标准化的数据目录结构
- 数据管理工具
- 自动数据备份和清理

## 🚀 快速开始

### 环境准备
```bash
# 安装依赖
pip install ccxt pandas numpy
```

### 检测可用交易所
```bash
python test_exchanges.py
```

### 获取数据示例
```python
from data.process import CryptoDataProcessor

# 初始化 (使用OKX交易所)
processor = CryptoDataProcessor(exchange_id='okx')

# 获取BTC价格数据
btc_data = processor.get_ohlcv('BTC/USDT', '1h', 100)

# 计算技术指标
data_with_indicators = processor.calculate_technical_indicators(btc_data)

# 保存数据
processor.save_to_csv(data_with_indicators, 'btc_analysis.csv', data_type='processed')
```

## 📁 项目结构
```
quantrade_system/
├── data/                  # 数据处理核心模块
│   └── process.py         # 主数据处理类
├── config/                # 配置文件
│   └── exchanges.py       # 交易所配置
├── datasets/              # 数据存储目录
│   ├── raw/               # 原始数据
│   ├── processed/         # 处理后的数据
│   └── analysis/          # 分析结果
├── strategies/            # 交易策略
├── backtest/              # 回测模块
├── labeling/              # 数据标注
├── models/                # 预测模型
├── tools/                 # 工具集
│   └── data_manager.py    # 数据管理工具
├── examples/              # 使用示例
│   └── okx_example.py     # OKX使用示例
└── test_exchanges.py      # 交易所测试工具
```

## 📊 使用示例

### 获取市场数据
```python
# 初始化数据处理器
processor = CryptoDataProcessor(exchange_id='okx')

# 获取交易对列表
symbols = processor.get_symbols()
print(f"可用交易对: {len(symbols)} 个")

# 获取BTC实时价格
ticker = processor.get_ticker('BTC/USDT')
print(f"BTC价格: ${ticker['last']:.2f}")
```

### 技术分析
```python
# 获取历史数据
data = processor.get_historical_data('BTC/USDT', '1d', 30)

# 计算技术指标
data = processor.calculate_technical_indicators(data)

# 查看最新技术指标
latest = data.iloc[-1]
print(f"RSI: {latest['RSI']:.2f}")
print(f"MACD: {latest['MACD']:.6f}")
```

### 数据管理
```bash
# 列出所有数据文件
python tools/data_manager.py

# 显示文件详情
python tools/data_manager.py --info btc_usdt_1h_raw.csv

# 清理7天前的数据
python tools/data_manager.py --clean 7

# 数据统计
python tools/data_manager.py --stats
```

## 🌟 特色亮点

1. **国内优化**: 优先支持国内可直接访问的交易所
2. **多交易所**: 支持多个交易所，便于数据对比和风险分散
3. **自动测试**: 内置交易所连接测试，确保可用性
4. **数据结构**: 标准化的数据存储结构，方便管理
5. **全功能**: 支持从数据获取到技术分析的完整流程

## 📝 TODO

- [ ] 完善回测系统
- [ ] 添加更多交易策略
- [ ] 实现自动交易功能
- [ ] 添加风险管理模块
- [ ] 优化数据获取性能

## 📜 许可证

MIT License
