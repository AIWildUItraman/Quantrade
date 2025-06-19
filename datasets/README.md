## 数据目录结构说明

### 📁 datasets/
量化交易系统的数据存储目录，按照数据类型和处理阶段进行组织

#### 🔤 目录结构
```
datasets/
├── raw/          # 原始数据
├── processed/    # 处理后的数据
├── analysis/     # 分析结果数据
└── README.md     # 本说明文件
```

#### 📋 各目录说明

##### 📊 raw/ - 原始数据
存储从交易所直接获取的原始数据
- **OHLCV数据**: 开高低收成交量数据
- **Ticker数据**: 实时价格数据
- **订单簿数据**: 买卖盘深度数据
- **交易记录**: 历史交易数据

**命名规范**: `{symbol}_{timeframe}_{exchange}_raw.csv`
- 示例: `btc_usdt_1h_okx_raw.csv`

##### 🔧 processed/ - 处理后的数据
存储经过清洗、标准化处理的数据
- **技术指标**: 添加了MA、RSI、MACD等指标的数据
- **特征工程**: 衍生特征、标签化数据
- **数据清洗**: 去重、异常值处理后的数据

**命名规范**: `{symbol}_{timeframe}_{features}_processed.csv`
- 示例: `btc_usdt_1h_with_indicators_processed.csv`

##### 📈 analysis/ - 分析结果
存储分析报告、策略回测结果等
- **技术分析报告**: 图表、统计分析结果
- **策略回测**: 策略performance数据
- **市场分析**: 相关性分析、趋势分析

**命名规范**: `{analysis_type}_{symbol}_{date}_analysis.csv`
- 示例: `technical_analysis_btc_usdt_20250619_analysis.csv`

#### 🏷️ 文件命名规范

##### 通用格式
```
{coin_pair}_{timeframe}_{exchange}_{data_type}_{date}.csv
```

##### 参数说明
- **coin_pair**: 交易对，用下划线连接 (如: btc_usdt, eth_usdt)
- **timeframe**: 时间周期 (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w)
- **exchange**: 交易所 (okx, binance, gate, etc.)
- **data_type**: 数据类型 (raw, processed, analysis)
- **date**: 日期 (YYYYMMDD) 或 时间范围 (30d, 7d)

##### 示例文件名
```
btc_usdt_1h_okx_raw_20250619.csv           # BTC/USDT 1小时原始数据
eth_usdt_1d_okx_processed_30d.csv          # ETH/USDT 日线处理数据(30天)
market_summary_okx_analysis_20250619.csv   # 市场概览分析
```

#### 📝 使用建议

1. **数据获取**: 优先保存到 `raw/` 目录
2. **数据处理**: 处理后保存到 `processed/` 目录
3. **分析结果**: 分析报告保存到 `analysis/` 目录
4. **定期清理**: 定期清理过期的临时数据文件
5. **备份重要数据**: 对于重要的历史数据建议进行备份

#### 🔧 代码使用示例

```python
from data.process import CryptoDataProcessor

processor = CryptoDataProcessor(exchange_id='okx')

# 获取原始数据并保存到 raw/ 目录
raw_data = processor.get_ohlcv('BTC/USDT', '1h', 100)
processor.save_to_csv(raw_data, 'btc_usdt_1h_okx_raw.csv', data_type='raw')

# 处理数据并保存到 processed/ 目录
processed_data = processor.calculate_technical_indicators(raw_data)
processor.save_to_csv(processed_data, 'btc_usdt_1h_with_indicators.csv', data_type='processed')

# 分析结果保存到 analysis/ 目录
# (分析代码...)
processor.save_to_csv(analysis_result, 'btc_technical_analysis.csv', data_type='analysis')
```

#### 📊 数据版本管理

为了更好地管理数据版本，建议：
1. 在文件名中包含日期或版本号
2. 保留关键时间点的数据快照
3. 使用Git LFS管理大型数据文件（如果使用Git）
4. 定期归档历史数据

---
*最后更新: 2025-06-19*
