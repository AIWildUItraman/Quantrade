import requests
import pandas as pd
import time
from datetime import datetime, timezone, timedelta
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def get_session_with_retries():
    """
    创建一个带有重试机制的 requests session
    """
    session = requests.Session()
    
    # 设置重试策略
    retry_strategy = Retry(
        total=3,
        status_forcelist=[429, 500, 502, 503, 504],
        backoff_factor=1
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    # 设置超时和SSL验证
    session.verify = True
    
    return session

def get_all_klines(symbol, interval, start_time, limit=1000):
    """
    分页获取从 start_time 开始的所有 K 线数据，增加错误处理和重试机制
    """
    url = "https://api.binance.com/api/v3/klines"
    klines = []
    session = get_session_with_retries()
    
    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": limit
        }
        
        try:
            # 增加超时设置
            response = session.get(url, params=params, timeout=(10, 30))
            response.raise_for_status()  # 检查HTTP错误
            data = response.json()
            
            # 如果没有数据或返回错误，则退出循环
            if not data or isinstance(data, dict) and data.get("code"):
                print("数据获取失败：", data)
                break
            
            klines.extend(data)
            print(f"已获取 {len(data)} 条数据，总计 {len(klines)} 条")
            
            # 如果返回的数据少于 limit，则表示已经没有更多数据
            if len(data) < limit:
                break
            
            # 更新 start_time 为最后一条数据的开盘时间加1毫秒，避免重复
            start_time = data[-1][0] + 1
            
            # 增加延迟时间，避免请求过快
            time.sleep(1)
            
        except requests.exceptions.SSLError as e:
            print(f"SSL错误: {e}")
            print("等待5秒后重试...")
            time.sleep(5)
            continue
            
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            print("等待3秒后重试...")
            time.sleep(3)
            continue
            
        except Exception as e:
            print(f"未知错误: {e}")
            break
    
    session.close()
    return klines

def process_klines_data(klines):
    """
    处理K线数据，只保留需要的字段，并转换为中国时间
    """
    # Binance 返回的数据格式：
    # [0]开盘时间, [1]开盘价, [2]最高价, [3]最低价, [4]收盘价, [5]成交量, 
    # [6]收盘时间, [7]成交额, [8]成交笔数, [9]主动买入成交量, [10]主动买入成交额, [11]忽略
    
    processed_data = []
    
    for kline in klines:
        processed_data.append({
            'time': kline[0],           # 时间戳（毫秒）
            'open': float(kline[1]),    # 开盘价
            'high': float(kline[2]),    # 最高价
            'low': float(kline[3]),     # 最低价
            'close': float(kline[4]),   # 收盘价
            'volume': float(kline[5]),  # 成交量
            'amount': float(kline[7])   # 成交额（Quote volume）
        })
    
    df = pd.DataFrame(processed_data)
    
    # 将时间戳转换为中国时间（UTC+8）
    df['time'] = pd.to_datetime(df['time'], unit='ms', utc=True)
    # 转换为中国时区
    china_tz = timezone(timedelta(hours=8))
    df['time'] = df['time'].dt.tz_convert(china_tz)
    # 格式化为更易读的格式（可选）
    df['time'] = df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
    
    return df

def main():
    symbol = "NEIROUSDT"
    intervals = ["1h", "2h"]  # 获取1小时和2小时的数据
    
    # 从很早的时间开始获取（0 表示 Unix 纪元时间，API 会自动返回最早可用的数据）
    start_time = 0
    
    for interval in intervals:
        print(f"\n{'='*60}")
        print(f"正在获取 {symbol} {interval} K线数据，请稍候...")
        print(f"{'='*60}\n")
        
        klines = get_all_klines(symbol, interval, start_time)
        
        if not klines:
            print(f"没有获取到 {interval} 数据")
            continue
        
        # 处理数据
        df = process_klines_data(klines)
        
        # 显示数据信息
        print(f"\n数据统计信息:")
        print(f"总记录数: {len(df)}")
        print(f"时间范围: {df['time'].iloc[0]} 至 {df['time'].iloc[-1]}")
        print(f"\n前5条数据预览:")
        print(df.head())
        print(f"\n后5条数据预览:")
        print(df.tail())
        print(f"\n数据统计:")
        print(df.describe())
        
        # 保存为 CSV 文件
        output_file = f"{symbol}_{interval}_data.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 数据已保存到 {output_file}")
        
        
        print(f"\n{'='*60}\n")
        
        # 避免连续请求过快
        if interval != intervals[-1]:
            time.sleep(2)
    
    print("🎉 所有数据下载完成！")

if __name__ == "__main__":
    main()
