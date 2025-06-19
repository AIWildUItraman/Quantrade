#!/usr/bin/env python3
"""
数据管理工具
用于管理datasets目录中的数据文件
"""

import os
import sys
import pandas as pd
from datetime import datetime, timedelta
import glob

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class DataManager:
    """数据管理器"""
    
    def __init__(self):
        self.project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.datasets_dir = os.path.join(self.project_root, 'datasets')
        
        # 确保目录存在
        for subdir in ['raw', 'processed', 'analysis']:
            os.makedirs(os.path.join(self.datasets_dir, subdir), exist_ok=True)
    
    def list_data_files(self, data_type: str = None):
        """
        列出数据文件
        
        Args:
            data_type: 数据类型 ('raw', 'processed', 'analysis') 或 None (全部)
        """
        print("📂 数据文件列表")
        print("=" * 60)
        
        if data_type:
            directories = [data_type]
        else:
            directories = ['raw', 'processed', 'analysis']
        
        total_files = 0
        total_size = 0
        
        for dir_name in directories:
            dir_path = os.path.join(self.datasets_dir, dir_name)
            if not os.path.exists(dir_path):
                continue
                
            files = glob.glob(os.path.join(dir_path, '*.csv'))
            
            if files:
                print(f"\n📁 {dir_name.upper()}/ ({len(files)} 个文件)")
                print("-" * 40)
                
                for file_path in sorted(files):
                    filename = os.path.basename(file_path)
                    file_size = os.path.getsize(file_path)
                    mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    
                    # 格式化文件大小
                    if file_size < 1024:
                        size_str = f"{file_size}B"
                    elif file_size < 1024 * 1024:
                        size_str = f"{file_size/1024:.1f}KB"
                    else:
                        size_str = f"{file_size/(1024*1024):.1f}MB"
                    
                    print(f"  {filename:<40} {size_str:>8} {mod_time.strftime('%Y-%m-%d %H:%M')}")
                    
                    total_files += 1
                    total_size += file_size
            else:
                print(f"\n📁 {dir_name.upper()}/ (空)")
        
        # 总计
        if total_size < 1024 * 1024:
            total_size_str = f"{total_size/1024:.1f}KB"
        else:
            total_size_str = f"{total_size/(1024*1024):.1f}MB"
        
        print(f"\n📊 总计: {total_files} 个文件, {total_size_str}")
    
    def get_file_info(self, filename: str):
        """获取文件详细信息"""
        file_found = False
        
        for dir_name in ['raw', 'processed', 'analysis']:
            file_path = os.path.join(self.datasets_dir, dir_name, filename)
            if os.path.exists(file_path):
                file_found = True
                print(f"📄 文件信息: {filename}")
                print("=" * 50)
                print(f"位置: datasets/{dir_name}/{filename}")
                print(f"大小: {os.path.getsize(file_path):,} 字节")
                print(f"修改时间: {datetime.fromtimestamp(os.path.getmtime(file_path))}")
                
                # 尝试读取CSV并显示基本信息
                try:
                    df = pd.read_csv(file_path)
                    print(f"数据行数: {len(df):,}")
                    print(f"数据列数: {len(df.columns)}")
                    print(f"列名: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}")
                    
                    if 'datetime' in df.columns:
                        df['datetime'] = pd.to_datetime(df['datetime'])
                        print(f"时间范围: {df['datetime'].min()} 到 {df['datetime'].max()}")
                    
                    print(f"\n前5行数据:")
                    print(df.head().to_string())
                    
                except Exception as e:
                    print(f"读取文件出错: {e}")
                
                break
        
        if not file_found:
            print(f"❌ 文件 {filename} 未找到")
    
    def clean_old_files(self, days: int = 7):
        """清理旧文件"""
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_files = []
        
        for dir_name in ['raw', 'processed', 'analysis']:
            dir_path = os.path.join(self.datasets_dir, dir_name)
            if not os.path.exists(dir_path):
                continue
            
            files = glob.glob(os.path.join(dir_path, '*.csv'))
            
            for file_path in files:
                mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                if mod_time < cutoff_date:
                    filename = os.path.basename(file_path)
                    os.remove(file_path)
                    deleted_files.append(f"{dir_name}/{filename}")
        
        if deleted_files:
            print(f"🗑️  已删除 {len(deleted_files)} 个超过 {days} 天的文件:")
            for file in deleted_files:
                print(f"  - {file}")
        else:
            print(f"✅ 没有超过 {days} 天的文件需要清理")
    
    def backup_data(self, backup_dir: str = None):
        """备份数据"""
        if not backup_dir:
            backup_dir = os.path.join(self.project_root, 'backup', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        os.makedirs(backup_dir, exist_ok=True)
        
        import shutil
        
        try:
            shutil.copytree(self.datasets_dir, os.path.join(backup_dir, 'datasets'))
            print(f"✅ 数据已备份到: {backup_dir}")
        except Exception as e:
            print(f"❌ 备份失败: {e}")
    
    def show_stats(self):
        """显示数据统计"""
        print("📊 数据统计")
        print("=" * 40)
        
        stats = {}
        
        for dir_name in ['raw', 'processed', 'analysis']:
            dir_path = os.path.join(self.datasets_dir, dir_name)
            if os.path.exists(dir_path):
                files = glob.glob(os.path.join(dir_path, '*.csv'))
                total_size = sum(os.path.getsize(f) for f in files)
                stats[dir_name] = {
                    'files': len(files),
                    'size': total_size
                }
        
        for dir_name, stat in stats.items():
            size_mb = stat['size'] / (1024 * 1024)
            print(f"{dir_name.upper():>10}: {stat['files']:>3} 文件, {size_mb:>6.1f}MB")
        
        total_files = sum(s['files'] for s in stats.values())
        total_size = sum(s['size'] for s in stats.values()) / (1024 * 1024)
        print(f"{'总计':>10}: {total_files:>3} 文件, {total_size:>6.1f}MB")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据管理工具')
    parser.add_argument('--list', '-l', nargs='?', const='all', 
                       choices=['raw', 'processed', 'analysis', 'all'], 
                       help='列出数据文件 (默认: all)')
    parser.add_argument('--info', '-i', type=str, help='显示文件详细信息')
    parser.add_argument('--clean', '-c', nargs='?', const=7, type=int, 
                       help='清理N天前的文件 (默认7天)')
    parser.add_argument('--backup', '-b', nargs='?', const=None, type=str, 
                       help='备份数据到指定目录')
    parser.add_argument('--stats', '-s', action='store_true', help='显示数据统计')
    
    args = parser.parse_args()
    
    dm = DataManager()
    
    if args.list is not None:
        list_type = None if args.list == 'all' else args.list
        dm.list_data_files(list_type)
    elif args.info:
        dm.get_file_info(args.info)
    elif args.clean is not None:
        dm.clean_old_files(args.clean)
    elif args.backup is not None:
        dm.backup_data(args.backup)
    elif args.stats:
        dm.show_stats()
    else:
        # 默认显示所有文件
        dm.list_data_files()

if __name__ == "__main__":
    main()
