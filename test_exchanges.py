#!/usr/bin/env python3
"""
快速测试脚本 - 检测可用的交易所
在使用数据处理器之前，先运行此脚本确定哪些交易所可以正常访问
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.process import CryptoDataProcessor
from config.exchanges import get_recommended_exchanges, DOMESTIC_EXCHANGES
import time

def test_basic_connection(exchange_id):
    """测试基础连接"""
    try:
        processor = CryptoDataProcessor(exchange_id=exchange_id)
        # 尝试获取交易对列表（最基础的API调用）
        symbols = processor.get_symbols()
        if symbols and len(symbols) > 0:
            return True, f"成功获取 {len(symbols)} 个交易对"
        else:
            return False, "无法获取交易对列表"
    except Exception as e:
        return False, str(e)

def test_data_fetch(exchange_id):
    """测试数据获取"""
    try:
        processor = CryptoDataProcessor(exchange_id=exchange_id)
        # 尝试获取BTC/USDT的ticker数据
        ticker = processor.get_ticker('BTC/USDT')
        if ticker and 'last' in ticker:
            return True, f"BTC价格: ${ticker['last']:.2f}"
        else:
            return False, "无法获取ticker数据"
    except Exception as e:
        return False, str(e)

def test_ohlcv_fetch(exchange_id):
    """测试OHLCV数据获取"""
    try:
        processor = CryptoDataProcessor(exchange_id=exchange_id)
        # 尝试获取少量OHLCV数据
        ohlcv = processor.get_ohlcv('BTC/USDT', '1h', 5)
        if ohlcv is not None and len(ohlcv) > 0:
            return True, f"获取到 {len(ohlcv)} 条K线数据"
        else:
            return False, "无法获取OHLCV数据"
    except Exception as e:
        return False, str(e)

def comprehensive_test():
    """综合测试所有推荐的交易所"""
    print("🔍 开始测试国内可访问的交易所...")
    print("=" * 60)
    
    recommended_exchanges = get_recommended_exchanges()
    results = {}
    
    for exchange_id in recommended_exchanges:
        print(f"\n📊 测试交易所: {DOMESTIC_EXCHANGES[exchange_id]['name']} ({exchange_id})")
        print("-" * 40)
        
        # 基础连接测试
        print("1. 基础连接测试...", end=" ")
        success1, msg1 = test_basic_connection(exchange_id)
        print("✅ 成功" if success1 else "❌ 失败")
        print(f"   {msg1}")
        
        if not success1:
            results[exchange_id] = {'status': 'failed', 'reason': msg1}
            continue
        
        time.sleep(1)  # 避免频率限制
        
        # 数据获取测试
        print("2. Ticker数据测试...", end=" ")
        success2, msg2 = test_data_fetch(exchange_id)
        print("✅ 成功" if success2 else "❌ 失败")
        print(f"   {msg2}")
        
        time.sleep(1)
        
        # OHLCV数据测试
        print("3. K线数据测试...", end=" ")
        success3, msg3 = test_ohlcv_fetch(exchange_id)
        print("✅ 成功" if success3 else "❌ 失败")
        print(f"   {msg3}")
        
        # 综合评分
        if success1 and success2 and success3:
            results[exchange_id] = {'status': 'excellent', 'score': 100}
            print("🌟 综合评价: 优秀")
        elif success1 and success2:
            results[exchange_id] = {'status': 'good', 'score': 70}
            print("👍 综合评价: 良好")
        elif success1:
            results[exchange_id] = {'status': 'basic', 'score': 40}
            print("⚠️  综合评价: 基础可用")
        else:
            results[exchange_id] = {'status': 'failed', 'score': 0}
            print("❌ 综合评价: 不可用")
        
        time.sleep(2)  # 测试间隔
    
    return results

def print_summary(results):
    """打印测试结果摘要"""
    print("\n" + "=" * 60)
    print("📋 测试结果摘要")
    print("=" * 60)
    
    excellent = [ex for ex, res in results.items() if res['status'] == 'excellent']
    good = [ex for ex, res in results.items() if res['status'] == 'good']
    basic = [ex for ex, res in results.items() if res['status'] == 'basic']
    failed = [ex for ex, res in results.items() if res['status'] == 'failed']
    
    if excellent:
        print(f"🌟 优秀 ({len(excellent)}个): {', '.join(excellent)}")
        print("   推荐优先使用这些交易所")
    
    if good:
        print(f"👍 良好 ({len(good)}个): {', '.join(good)}")
        print("   可以正常使用")
    
    if basic:
        print(f"⚠️  基础 ({len(basic)}个): {', '.join(basic)}")
        print("   连接不稳定，建议作为备选")
    
    if failed:
        print(f"❌ 失败 ({len(failed)}个): {', '.join(failed)}")
        print("   当前网络环境无法访问")
    
    # 给出建议
    print(f"\n💡 使用建议:")
    if excellent:
        print(f"   主要使用: {excellent[0]}")
        if len(excellent) > 1:
            print(f"   备用选择: {', '.join(excellent[1:])}")
    elif good:
        print(f"   主要使用: {good[0]}")
        if len(good) > 1:
            print(f"   备用选择: {', '.join(good[1:])}")
    elif basic:
        print(f"   可尝试使用: {basic[0]}")
    else:
        print("   ⚠️ 当前没有可用的交易所，请检查网络连接或尝试使用代理")

def generate_config_suggestion(results):
    """生成配置建议"""
    excellent = [ex for ex, res in results.items() if res['status'] == 'excellent']
    good = [ex for ex, res in results.items() if res['status'] == 'good']
    
    available = excellent + good
    
    if available:
        print(f"\n⚙️  推荐配置:")
        print(f"# 在您的代码中使用以下配置")
        print(f"processor = CryptoDataProcessor(exchange_id='{available[0]}')")
        
        if len(available) > 1:
            print(f"\n# 备用配置（如果主要交易所出现问题）")
            for backup in available[1:3]:  # 最多显示2个备用选择
                print(f"# processor = CryptoDataProcessor(exchange_id='{backup}')")

def main():
    """主函数"""
    print("🚀 加密货币交易所连接测试工具")
    print("   测试国内可访问的交易所连接情况")
    print("   建议在首次使用数据处理器前运行此测试\n")
    
    try:
        results = comprehensive_test()
        print_summary(results)
        generate_config_suggestion(results)
        
        print(f"\n✨ 测试完成! 现在您可以使用推荐的交易所获取数据了。")
        
    except KeyboardInterrupt:
        print(f"\n⏹️  测试已取消")
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")

if __name__ == "__main__":
    main()
