"""
OKX交易所数据获取使用示例
基于测试结果，OKX是当前环境下最佳选择
"""

import sys
import os
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.process import CryptoDataProcessor
import pandas as pd
from datetime import datetime, timedelta

def main():
    """主要的数据获取和处理示例"""
    print("🚀 使用OKX交易所获取加密货币数据")
    print("=" * 50)
    
    # 初始化OKX数据处理器
    processor = CryptoDataProcessor(exchange_id='okx')
    
    # 1. 获取主要加密货币的当前价格
    print("\n📊 获取主要加密货币当前价格:")
    major_coins = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'ADA/USDT', 'DOT/USDT']
    
    for symbol in major_coins:
        ticker = processor.get_ticker(symbol)
        if ticker:
            print(f"{symbol:12} ${ticker['last']:>10.2f} ({ticker['percentage']:>+6.2f}%)")
    
    # 2. 获取BTC历史数据并分析
    print(f"\n📈 获取BTC/USDT历史数据分析:")
    
    # 获取过去30天的小时数据
    btc_data = processor.get_historical_data('BTC/USDT', '1h', days=30)
    
    if btc_data is not None:
        print(f"获取到 {len(btc_data)} 条历史数据")
        
        # 计算技术指标
        btc_with_indicators = processor.calculate_technical_indicators(btc_data)
        
        # 显示最新的技术分析
        latest = btc_with_indicators.iloc[-1]
        print(f"\n🔍 BTC技术分析 (最新数据):")
        print(f"当前价格: ${latest['close']:,.2f}")
        print(f"MA5:     ${latest['MA5']:,.2f}")
        print(f"MA20:    ${latest['MA20']:,.2f}")
        print(f"RSI:     {latest['RSI']:.2f}")
        print(f"MACD:    {latest['MACD']:.6f}")
        
        # 判断趋势
        if latest['close'] > latest['MA20']:
            trend = "🔺 上涨趋势"
        else:
            trend = "🔻 下跌趋势"
        print(f"趋势判断: {trend}")
        
        # 判断超买超卖
        if latest['RSI'] > 70:
            rsi_status = "⚠️ 超买"
        elif latest['RSI'] < 30:
            rsi_status = "⚠️ 超卖"
        else:
            rsi_status = "✅ 正常"
        print(f"RSI状态: {rsi_status}")
        
        # 保存数据到processed目录（因为包含技术指标）
        processor.save_to_csv(btc_with_indicators, 'btc_usdt_30d_with_indicators.csv', 
                             data_type='processed')
        print(f"\n💾 数据已保存到: datasets/processed/btc_usdt_30d_with_indicators.csv")
    
    # 3. 获取多个币种的市场概览
    print(f"\n🌍 市场概览:")
    market_summary = processor.get_market_summary()
    if market_summary is not None and not market_summary.empty:
        print(market_summary.to_string(index=False, float_format='%.2f'))
    
    # 4. 获取订单簿数据
    print(f"\n📊 BTC/USDT 订单簿 (前5档):")
    order_book = processor.get_order_book('BTC/USDT', limit=10)
    if order_book:
        print("卖单 (Ask):")
        for i, ask in enumerate(order_book['asks'][:5]):
            print(f"  {ask[0]:>10.2f} - {ask[1]:>8.4f}")
        
        print("买单 (Bid):")
        for i, bid in enumerate(order_book['bids'][:5]):
            print(f"  {bid[0]:>10.2f} - {bid[1]:>8.4f}")
    
    # 5. 批量获取热门币种数据
    print(f"\n🔥 批量获取热门币种数据:")
    hot_coins = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'DOGE/USDT']
    
    for symbol in hot_coins:
        try:
            # 获取最近24小时的数据
            data = processor.get_ohlcv(symbol, '1h', 24)
            if data is not None:
                # 计算24小时统计
                high_24h = data['high'].max()
                low_24h = data['low'].min()
                vol_24h = data['volume'].sum()
                price_change = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]) * 100
                
                print(f"{symbol:12} 24h统计:")
                print(f"  价格变化: {price_change:+6.2f}%")
                print(f"  24h最高: ${high_24h:,.2f}")
                print(f"  24h最低: ${low_24h:,.2f}")
                print(f"  24h成交量: {vol_24h:,.0f}")
                
                # 保存每个币种的原始数据到raw目录
                filename = f"{symbol.replace('/', '_').lower()}_24h_raw.csv"
                processor.save_to_csv(data, filename, data_type='raw')
                print(f"  数据已保存: datasets/raw/{filename}")
                print()
        
        except Exception as e:
            print(f"获取 {symbol} 数据失败: {e}")
    
    print("✅ 数据获取完成!")
    print("\n📁 生成的文件:")
    print("  - datasets/processed/btc_usdt_30d_with_indicators.csv (BTC 30天技术分析)")
    print("  - datasets/raw/btc_usdt_24h_raw.csv (BTC 24小时原始数据)")
    print("  - datasets/raw/eth_usdt_24h_raw.csv (ETH 24小时原始数据)")
    print("  - datasets/raw/sol_usdt_24h_raw.csv (SOL 24小时原始数据)")
    print("  - datasets/raw/doge_usdt_24h_raw.csv (DOGE 24小时原始数据)")
    
    print(f"\n📂 数据目录结构:")
    print("  datasets/")
    print("  ├── raw/          # 原始交易数据")
    print("  ├── processed/    # 带技术指标的处理数据")
    print("  └── analysis/     # 分析结果数据")

def quick_price_check():
    """快速价格检查函数"""
    processor = CryptoDataProcessor(exchange_id='okx')
    
    symbols = ['BTC/USDT', 'ETH/USDT']
    print("🔍 快速价格检查:")
    
    for symbol in symbols:
        ticker = processor.get_ticker(symbol)
        if ticker:
            print(f"{symbol}: ${ticker['last']:,.2f}")

if __name__ == "__main__":
    main()
    
    print(f"\n" + "="*50)
    print("💡 提示: 您也可以直接调用 quick_price_check() 函数进行快速价格查询")
