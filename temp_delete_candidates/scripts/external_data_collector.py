#!/usr/bin/env python3
"""
外部データ収集システム - Yahoo Finance API
無料でマクロ経済データを取得
"""

import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class ExternalDataCollector:
    """外部データ収集システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.external_dir = self.data_dir / "external"
        self.external_dir.mkdir(parents=True, exist_ok=True)
        
        # Yahoo Finance ティッカー設定
        self.tickers = {
            "us_10y": "^TNX",        # 米国10年国債利回り
            "jp_10y": "^TNX-JP",     # 日本10年国債利回り（試行用）
            "sp500": "^GSPC",        # S&P500指数
            "usd_jpy": "JPY=X",      # ドル円
            "nikkei": "^N225",       # 日経平均（参考用）
            "vix": "^VIX"           # VIX恐怖指数
        }
        
        # 代替ティッカー（日本国債用）
        self.jp_bond_alternatives = ["^TNX-JP", "JP10Y.JP", "JP10Y:U.S."]
        
    def test_data_availability(self):
        """データ取得可能性テスト"""
        logger.info("🔍 外部データ取得可能性テスト...")
        
        results = {}
        for name, ticker in self.tickers.items():
            try:
                logger.info(f"  {name} ({ticker}) テスト中...")
                data = yf.download(ticker, period="5d", interval="1d", progress=False)
                
                if not data.empty and len(data) > 0:
                    latest_date = data.index[-1].strftime('%Y-%m-%d')
                    latest_value = data['Close'].iloc[-1]
                    results[name] = {
                        'status': 'OK',
                        'ticker': ticker,
                        'latest_date': latest_date,
                        'latest_value': latest_value,
                        'data_points': len(data)
                    }
                    logger.info(f"    ✅ OK: {latest_date} = {latest_value:.2f}")
                else:
                    results[name] = {'status': 'EMPTY', 'ticker': ticker}
                    logger.warning(f"    ⚠️ データなし")
                    
            except Exception as e:
                results[name] = {'status': 'ERROR', 'ticker': ticker, 'error': str(e)}
                logger.error(f"    ❌ エラー: {e}")
        
        # 日本国債の代替ティッカーテスト
        if results.get('jp_10y', {}).get('status') != 'OK':
            logger.info("  日本国債代替ティッカーテスト...")
            for alt_ticker in self.jp_bond_alternatives:
                try:
                    logger.info(f"    {alt_ticker} テスト中...")
                    data = yf.download(alt_ticker, period="5d", interval="1d", progress=False)
                    if not data.empty and len(data) > 0:
                        latest_date = data.index[-1].strftime('%Y-%m-%d')
                        latest_value = data['Close'].iloc[-1]
                        results['jp_10y'] = {
                            'status': 'OK',
                            'ticker': alt_ticker,
                            'latest_date': latest_date,
                            'latest_value': latest_value,
                            'data_points': len(data)
                        }
                        self.tickers['jp_10y'] = alt_ticker
                        logger.info(f"      ✅ 代替成功: {alt_ticker}")
                        break
                except:
                    continue
        
        return results
    
    def collect_historical_data(self, start_date="2016-01-01"):
        """過去データの一括収集"""
        logger.info(f"📊 過去データ収集開始 (from {start_date})...")
        
        all_external_data = {}
        successful_tickers = {}
        
        for name, ticker in self.tickers.items():
            try:
                logger.info(f"  {name} ({ticker}) 収集中...")
                
                # 長期データ取得
                data = yf.download(ticker, start=start_date, end=None, interval="1d", progress=False)
                
                if not data.empty and len(data) > 0:
                    # 基本的な前処理
                    processed_data = data.copy()
                    
                    # インデックスをDateカラムに変換
                    processed_data.reset_index(inplace=True)
                    processed_data['Date'] = pd.to_datetime(processed_data['Date'])
                    
                    # カラム名を統一
                    if 'Close' in processed_data.columns:
                        processed_data[f'{name}_value'] = processed_data['Close']
                        processed_data[f'{name}_change'] = processed_data['Close'].pct_change()
                        processed_data[f'{name}_volatility'] = processed_data['Close'].rolling(20).std()
                    
                    # 必要カラムのみ保持
                    keep_cols = ['Date', f'{name}_value', f'{name}_change', f'{name}_volatility']
                    processed_data = processed_data[keep_cols].copy()
                    
                    all_external_data[name] = processed_data
                    successful_tickers[name] = ticker
                    
                    logger.info(f"    ✅ {name}: {len(processed_data):,}件 ({processed_data['Date'].min().date()} - {processed_data['Date'].max().date()})")
                else:
                    logger.warning(f"    ⚠️ {name}: データなし")
                    
            except Exception as e:
                logger.error(f"    ❌ {name}: エラー - {e}")
        
        return all_external_data, successful_tickers
    
    def merge_external_data(self, external_data_dict):
        """外部データの統合"""
        logger.info("🔗 外部データ統合...")
        
        if not external_data_dict:
            logger.error("❌ 統合するデータがありません")
            return None
        
        # 最初のデータをベースに順次結合
        merged_data = None
        for name, data in external_data_dict.items():
            if merged_data is None:
                merged_data = data.copy()
                logger.info(f"  ベース: {name} ({len(data):,}件)")
            else:
                merged_data = merged_data.merge(data, on='Date', how='outer')
                logger.info(f"  結合: {name} → 合計 {len(merged_data):,}件")
        
        # 日付でソート
        merged_data = merged_data.sort_values('Date').reset_index(drop=True)
        
        # 欠損値の前埋め（外部データは週末等で欠損が多いため）
        numeric_cols = merged_data.select_dtypes(include=[np.number]).columns
        merged_data[numeric_cols] = merged_data[numeric_cols].fillna(method='ffill')
        
        logger.info(f"✅ 外部データ統合完了: {len(merged_data):,}件")
        logger.info(f"期間: {merged_data['Date'].min().date()} - {merged_data['Date'].max().date()}")
        
        return merged_data
    
    def save_external_data(self, merged_data, filename="external_macro_data.parquet"):
        """外部データの保存"""
        if merged_data is None:
            return
            
        filepath = self.external_dir / filename
        merged_data.to_parquet(filepath, index=False)
        logger.info(f"💾 外部データ保存: {filepath}")
        logger.info(f"  ファイルサイズ: {filepath.stat().st_size / 1024:.1f} KB")
        
        # サンプル表示
        logger.info(f"\n📋 外部データサンプル:")
        logger.info(f"カラム数: {len(merged_data.columns)}")
        logger.info(f"カラム: {list(merged_data.columns)}")
        logger.info(f"\n最新5日分:")
        print(merged_data.tail().to_string(index=False))
    
    def integrate_with_existing_data(self):
        """既存のJ-Quantsデータと統合"""
        logger.info("🔄 既存データとの統合...")
        
        # 既存データ読み込み
        processed_dir = self.data_dir / "processed"
        existing_files = list(processed_dir.glob("*.parquet"))
        
        if not existing_files:
            logger.error("❌ 既存の処理済みデータが見つかりません")
            return None
            
        existing_data = pd.read_parquet(existing_files[0])
        existing_data['Date'] = pd.to_datetime(existing_data['Date'])
        logger.info(f"既存データ: {len(existing_data):,}件")
        
        # 外部データ読み込み
        external_file = self.external_dir / "external_macro_data.parquet"
        if not external_file.exists():
            logger.error("❌ 外部データファイルが見つかりません")
            return None
            
        external_data = pd.read_parquet(external_file)
        external_data['Date'] = pd.to_datetime(external_data['Date'])
        logger.info(f"外部データ: {len(external_data):,}件")
        
        # 日付ベースで統合
        integrated_data = existing_data.merge(external_data, on='Date', how='left')
        
        # 外部データの前埋め（株式取引日と外部データの取得日のズレを調整）
        external_cols = [col for col in integrated_data.columns if any(prefix in col for prefix in ['us_10y', 'jp_10y', 'sp500', 'usd_jpy', 'nikkei', 'vix'])]
        integrated_data[external_cols] = integrated_data[external_cols].fillna(method='ffill')
        
        logger.info(f"✅ データ統合完了: {len(integrated_data):,}件")
        logger.info(f"新規カラム数: {len(external_cols)}")
        logger.info(f"総カラム数: {len(integrated_data.columns)}")
        
        # 統合データ保存
        integrated_file = processed_dir / "integrated_with_external.parquet"
        integrated_data.to_parquet(integrated_file, index=False)
        logger.info(f"💾 統合データ保存: {integrated_file}")
        
        return integrated_data

def main():
    """メイン実行"""
    logger.info("🚀 外部データ収集システム開始")
    logger.info("🎯 目標: Yahoo Finance APIでマクロ経済データを無料取得")
    
    collector = ExternalDataCollector()
    
    try:
        # 1. データ取得可能性テスト
        test_results = collector.test_data_availability()
        
        # 2. 成功したティッカーで過去データ収集
        external_data_dict, successful_tickers = collector.collect_historical_data()
        
        if not external_data_dict:
            logger.error("❌ データ収集に失敗しました")
            return
        
        # 3. 外部データ統合
        merged_external = collector.merge_external_data(external_data_dict)
        
        # 4. 外部データ保存
        collector.save_external_data(merged_external)
        
        # 5. 既存データと統合
        integrated_data = collector.integrate_with_existing_data()
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎉 外部データ収集完了")
        logger.info("="*80)
        
        logger.info(f"✅ 取得成功ティッカー: {len(successful_tickers)}")
        for name, ticker in successful_tickers.items():
            logger.info(f"  {name}: {ticker}")
        
        if integrated_data is not None:
            logger.info(f"\n📊 最終統合データ:")
            logger.info(f"  レコード数: {len(integrated_data):,}件")
            logger.info(f"  カラム数: {len(integrated_data.columns)}")
            logger.info(f"  期間: {integrated_data['Date'].min().date()} - {integrated_data['Date'].max().date()}")
            
            # 外部データカラムの統計
            external_cols = [col for col in integrated_data.columns if any(prefix in col for prefix in ['us_10y', 'jp_10y', 'sp500', 'usd_jpy', 'nikkei', 'vix'])]
            logger.info(f"  外部データカラム: {len(external_cols)}個")
            
            # 欠損値チェック
            missing_stats = integrated_data[external_cols].isnull().sum()
            if missing_stats.sum() > 0:
                logger.info(f"  欠損値あり:")
                for col, missing_count in missing_stats[missing_stats > 0].items():
                    logger.info(f"    {col}: {missing_count:,}件 ({missing_count/len(integrated_data)*100:.1f}%)")
            else:
                logger.info(f"  ✅ 欠損値なし")
        
        logger.info(f"\n🚀 次のステップ: 新しい外部特徴量で精度評価を実行")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()