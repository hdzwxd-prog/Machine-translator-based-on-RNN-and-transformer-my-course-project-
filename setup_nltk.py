#!/usr/bin/env python3
"""
NLTK数据下载脚本
在训练前运行此脚本下载必需的NLTK数据，避免训练时卡住
"""
import nltk
import sys

def download_nltk_data():
    """下载NLTK必需的数据"""
    print("=" * 80)
    print("下载NLTK数据")
    print("=" * 80)
    
    # 新版本NLTK (3.8+)使用punkt_tab，旧版本使用punkt
    # 同时下载两者以确保兼容性
    required_data = ['punkt_tab', 'punkt']
    
    for data_name in required_data:
        print(f"\n下载 {data_name}...")
        try:
            nltk.download(data_name, quiet=False)
            print(f"✓ {data_name} 下载完成")
        except Exception as e:
            print(f"✗ {data_name} 下载失败: {e}")
            print(f"  请检查网络连接，或手动下载：")
            print(f"  python -c \"import nltk; nltk.download('{data_name}')\"")
            return False
    
    print("\n" + "=" * 80)
    print("所有NLTK数据下载完成！")
    print("=" * 80)
    return True

if __name__ == "__main__":
    success = download_nltk_data()
    sys.exit(0 if success else 1)

