import requests
import pandas as pd
import time
from datetime import datetime
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

def main():
    symbol = "NEIROUSDT"  # 注意：确保该交易对在 Binance 上存在
    interval = "1h"
    
    # 从很早的时间开始获取（0 表示 Unix 纪元时间，API 会自动返回最早可用的数据）
    start_time = 0
    
    print(f"正在获取 {symbol} {interval} K线数据，请稍候...")
    klines = get_all_klines(symbol, interval, start_time)
    
    if not klines:
        print("没有获取到数据")
        return
    
    # Binance 返回的数据格式为：
    # [开盘时间, 开盘价, 最高价, 最低价, 收盘价, 成交量, 收盘时间, 成交额, 成交笔数, 主动买入成交量, 主动买入成交额, 忽略]
    columns = ["开盘时间", "开盘价", "最高价", "最低价", "收盘价", "成交量",
               "收盘时间", "成交额", "成交笔数", "主动买入成交量", "主动买入成交额", "忽略"]
    df = pd.DataFrame(klines, columns=columns)
    
    # 将时间戳（毫秒）转换为可读时间
    df["开盘时间"] = pd.to_datetime(df["开盘时间"], unit='ms')
    df["收盘时间"] = pd.to_datetime(df["收盘时间"], unit='ms')
    
    # 保存为 CSV 文件
    output_file = f"{symbol}_{interval}_data.csv"
    df.to_csv(output_file, index=False)
    print(f"数据已保存到 {output_file}，共 {len(df)} 条记录")

if __name__ == "__main__":
    main()
