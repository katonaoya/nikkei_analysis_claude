#!/usr/bin/env python3
"""
Test J-Quants API connection and basic data retrieval
Phase 8: Model and Data Accuracy Improvement - Step 1
"""

import pandas as pd
from pathlib import Path
from datetime import datetime, date, timedelta

# Add project root to path
import sys
sys.path.append(str(Path(__file__).parent))

from src.data.jquants_client import JQuantsClient, create_jquants_client
from src.data.stock_data_fetcher import StockDataFetcher, create_stock_data_fetcher
from utils.config import get_config
from utils.logger import get_logger

def test_jquants_api_connection():
    """Test J-Quants API connection and authentication"""
    print("Phase 8: モデル・データ精度向上")
    print("=" * 50)
    print("Step 1: J-Quants API接続テスト")
    print()
    
    logger = get_logger("jquants_test")
    
    # Check configuration
    print("1. 設定確認...")
    config = get_config()
    
    # Check required environment variables
    required_vars = [
        'JQUANTS_MAIL_ADDRESS',
        'JQUANTS_PASSWORD',
        'JQUANTS_REFRESH_TOKEN'
    ]
    
    missing_vars = []
    for var in required_vars:
        value = config.get(f'api.jquants.{var.lower().replace("jquants_", "")}')
        if not value or value.startswith('your_'):
            missing_vars.append(var)
            print(f"  ❌ {var}: 設定されていません")
        else:
            print(f"  ✅ {var}: 設定済み")
    
    if missing_vars:
        print(f"\n⚠️  必要な環境変数が設定されていません: {', '.join(missing_vars)}")
        print("以下の手順で設定してください:")
        print("1. .envファイルを編集")
        print("2. J-Quants API の認証情報を入力")
        print("3. このテストを再実行")
        return False
    
    print("\n2. J-Quants APIクライアント初期化...")
    try:
        client = create_jquants_client()
        print("  ✅ クライアント初期化成功")
    except Exception as e:
        print(f"  ❌ クライアント初期化失敗: {str(e)}")
        return False
    
    print("\n3. API認証テスト...")
    try:
        # Test authentication
        client.ensure_authenticated()
        if client.id_token:
            print("  ✅ API認証成功")
        else:
            print("  ❌ API認証失敗 - トークンが取得できません")
            return False
    except Exception as e:
        print(f"  ❌ API認証エラー: {str(e)}")
        return False
    
    print("\n4. 銘柄リスト取得テスト...")
    try:
        # Test fetching symbol list (NIKKEI 225)
        nikkei225_df = client.get_nikkei225_components()
        if not nikkei225_df.empty:
            symbols = nikkei225_df['Code'].tolist() if 'Code' in nikkei225_df.columns else nikkei225_df.iloc[:, 0].tolist()
            print(f"  ✅ 日経225構成銘柄取得成功: {len(symbols)}銘柄")
            print(f"  サンプル銘柄: {symbols[:5]}")
        else:
            print("  ⚠️  銘柄リスト取得成功だが、データが空です")
            return False
    except Exception as e:
        print(f"  ❌ 銘柄リスト取得エラー: {str(e)}")
        return False
    
    print("\n5. 株価データ取得テスト（少量データ）...")
    try:
        # Test fetching stock data for a small date range and few symbols
        test_symbols = symbols[:3]  # Test with first 3 symbols
        test_start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
        test_end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"  テスト対象: {test_symbols}")
        print(f"  期間: {test_start_date} ～ {test_end_date}")
        
        stock_data = client.get_stock_prices(
            symbols=test_symbols,
            start_date=test_start_date,
            end_date=test_end_date
        )
        
        if stock_data and not stock_data.empty:
            print(f"  ✅ 株価データ取得成功: {len(stock_data)}レコード")
            print(f"  データ列: {list(stock_data.columns)}")
            print(f"  データ期間: {stock_data['Date'].min()} ～ {stock_data['Date'].max()}")
            print(f"  銘柄数: {stock_data['Code'].nunique()}")
        else:
            print("  ⚠️  株価データ取得成功だが、データが空です")
            return False
            
    except Exception as e:
        print(f"  ❌ 株価データ取得エラー: {str(e)}")
        return False
    
    print("\n" + "=" * 50)
    print("✅ J-Quants API接続テスト完了")
    print("次のステップ: 実際の日本株データ取得実行")
    
    return True

def test_stock_data_fetcher():
    """Test StockDataFetcher with small dataset"""
    print("\n6. StockDataFetcher統合テスト...")
    
    try:
        fetcher = create_stock_data_fetcher()
        
        # Test with small date range
        test_start_date = date(2024, 8, 1)
        test_end_date = date(2024, 8, 7)
        
        print(f"  テスト期間: {test_start_date} ～ {test_end_date}")
        
        # Fetch small amount of data
        data = fetcher.fetch_stock_data(
            start_date=test_start_date,
            end_date=test_end_date,
            limit_symbols=5  # Limit to 5 symbols for testing
        )
        
        if data and not data.empty:
            print(f"  ✅ StockDataFetcher取得成功: {len(data)}レコード")
            print(f"  銘柄数: {data['Code'].nunique()}")
            print(f"  期間: {data['Date'].min()} ～ {data['Date'].max()}")
            
            # Show sample data
            print("\n  サンプルデータ:")
            print(data.head(3).to_string())
        else:
            print("  ⚠️  StockDataFetcher取得成功だが、データが空です")
            return False
            
    except Exception as e:
        print(f"  ❌ StockDataFetcher エラー: {str(e)}")
        return False
    
    return True

def main():
    """Main test function"""
    success = True
    
    # Test API connection
    if test_jquants_api_connection():
        # Test data fetcher if API connection works
        if not test_stock_data_fetcher():
            success = False
    else:
        success = False
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 すべてのテストが成功しました！")
        print("Phase 8 の次のステップに進むことができます。")
    else:
        print("❌ テストで問題が発生しました。")
        print("上記のエラーメッセージを確認して修正してください。")
    
    return success

if __name__ == "__main__":
    main()