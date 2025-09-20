#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Yahoo Finance 10年分拡張データ取得システム
外部指標データ（USD/JPY, VIX, TOPIX, 日経225等）を10年間分取得
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
import time

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YahooFinanceExtendedFetcher:
    """Yahoo Finance 10年分拡張データ取得システム"""
    
    def __init__(self, start_date: str = None, end_date: str = None):
        """初期化"""
        # デフォルト期間設定（10年間）
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
        
        self.start_date = start_date
        self.end_date = end_date
        
        # 外部指標シンボル定義
        self.symbols = {
            'usdjpy': 'USDJPY=X',      # USD/JPY
            'vix': '^VIX',             # VIX恐怖指数
            'nikkei225': '^N225',      # 日経225指数
            'topix': '^TOPX',          # TOPIX指数
            'sp500': '^GSPC',          # S&P500
            'nasdaq': '^IXIC',         # NASDAQ
            'dxy': 'DX-Y.NYB',         # ドルインデックス
            'gold': 'GC=F',            # 金先物
            'crude_oil': 'CL=F',       # 原油先物
            'jgb_10y': '^TNX'          # 10年債利回り（代替）
        }
        
        # 保存ディレクトリ
        self.output_dir = Path("data/external_extended")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"データ取得期間: {self.start_date} 〜 {self.end_date}")
        logger.info(f"対象指標数: {len(self.symbols)}個")
    
    def fetch_symbol_data(self, symbol: str, name: str) -> pd.DataFrame:
        """個別シンボルのデータ取得"""
        logger.info(f"取得開始: {name} ({symbol})")
        
        try:
            # Yahoo Financeからデータ取得
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=self.start_date, end=self.end_date, auto_adjust=True)
            
            if data.empty:
                logger.warning(f"{name}: データが見つかりません")
                return pd.DataFrame()
            
            # インデックスをDateカラムに変換
            data = data.reset_index()
            data['Symbol'] = symbol
            data['Name'] = name
            
            # 基本的な技術指標計算
            data['Daily_Return'] = data['Close'].pct_change()
            data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # 移動平均
            data['MA_5'] = data['Close'].rolling(5).mean()
            data['MA_20'] = data['Close'].rolling(20).mean()
            data['MA_60'] = data['Close'].rolling(60).mean()
            
            # ボラティリティ
            data['Volatility_5'] = data['Daily_Return'].rolling(5).std()
            data['Volatility_20'] = data['Daily_Return'].rolling(20).std()
            
            # 移動平均からの乖離率
            data['MA20_Deviation'] = (data['Close'] - data['MA_20']) / data['MA_20']
            
            # VIX特有の指標
            if 'VIX' in symbol:
                data['VIX_Spike'] = (data['Close'] > data['MA_20'] * 1.5).astype(int)
                data['VIX_High'] = (data['Close'] > 30).astype(int)
            
            # USD/JPY特有の指標
            if 'USDJPY' in symbol:
                data['USDJPY_Trend'] = np.where(data['Close'] > data['MA_20'], 1, 
                                               np.where(data['Close'] < data['MA_20'], -1, 0))
            
            logger.info(f"{name}: {len(data)}件取得完了")
            return data
            
        except Exception as e:
            logger.error(f"{name} ({symbol}) 取得エラー: {e}")
            return pd.DataFrame()
    
    def fetch_all_external_data(self) -> dict:
        """全外部指標データ取得"""
        logger.info("🚀 Yahoo Finance 10年分外部指標データ取得開始!")
        
        all_data = {}
        
        for name, symbol in self.symbols.items():
            # レート制限対策
            time.sleep(1)
            
            data = self.fetch_symbol_data(symbol, name)
            if not data.empty:
                all_data[name] = data
                
                # 個別ファイル保存
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_file = self.output_dir / f"{name}_10years_{timestamp}.parquet"
                data.to_parquet(output_file, index=False)
                logger.info(f"保存完了: {output_file}")
        
        return all_data
    
    def create_integrated_dataset(self, all_data: dict) -> pd.DataFrame:
        """統合データセット作成"""
        logger.info("統合データセット作成開始...")
        
        # 基準日付を日経225から取得
        if 'nikkei225' in all_data and not all_data['nikkei225'].empty:
            base_dates = all_data['nikkei225'][['Date']].copy()
            base_dates = base_dates.sort_values('Date')
        else:
            logger.error("日経225データがないため統合できません")
            return pd.DataFrame()
        
        logger.info(f"基準日数: {len(base_dates)}日間")
        
        # 各指標データをマージ
        integrated_df = base_dates.copy()
        
        for name, data in all_data.items():
            if data.empty:
                continue
            
            # 必要なカラムを選択してリネーム
            merge_cols = ['Date', 'Close', 'Daily_Return', 'Volatility_20', 'MA_20', 'MA20_Deviation']
            
            # 特殊指標の追加
            if 'VIX' in name:
                merge_cols.extend(['VIX_Spike', 'VIX_High'])
            elif 'usdjpy' in name:
                merge_cols.extend(['USDJPY_Trend'])
            
            # 存在するカラムのみ選択
            available_cols = [col for col in merge_cols if col in data.columns]
            merge_data = data[available_cols].copy()
            
            # カラム名にプレフィックス追加（Dateは除く）
            rename_dict = {col: f"{name}_{col}" for col in available_cols if col != 'Date'}
            merge_data = merge_data.rename(columns=rename_dict)
            
            # マージ
            integrated_df = pd.merge(integrated_df, merge_data, on='Date', how='left')
            logger.info(f"{name}: マージ完了 ({len(merge_data)}日間)")
        
        # 前方補完で欠損値処理
        integrated_df = integrated_df.fillna(method='ffill')
        integrated_df = integrated_df.fillna(method='bfill')
        
        logger.info(f"統合データセット作成完了: {len(integrated_df)}件, {len(integrated_df.columns)}カラム")
        
        return integrated_df
    
    def save_integrated_data(self, integrated_df: pd.DataFrame):
        """統合データ保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Parquet形式で保存
        parquet_file = self.output_dir / f"external_integrated_10years_{timestamp}.parquet"
        integrated_df.to_parquet(parquet_file, index=False)
        
        # CSV形式でも保存（確認用）
        csv_file = self.output_dir / f"external_integrated_10years_{timestamp}.csv"
        integrated_df.to_csv(csv_file, index=False)
        
        logger.info(f"統合データ保存完了:")
        logger.info(f"  Parquet: {parquet_file}")
        logger.info(f"  CSV: {csv_file}")
        
        # データ統計情報
        logger.info(f"統合データ統計:")
        logger.info(f"  期間: {integrated_df['Date'].min()} 〜 {integrated_df['Date'].max()}")
        logger.info(f"  レコード数: {len(integrated_df):,}件")
        logger.info(f"  カラム数: {len(integrated_df.columns)}個")
        logger.info(f"  欠損値: {integrated_df.isnull().sum().sum()}個")
        
        return parquet_file, csv_file
    
    def run_extended_fetch(self):
        """拡張データ取得実行"""
        logger.info("📊 Yahoo Finance 10年分拡張データ取得実行開始!")
        
        try:
            # 全データ取得
            all_data = self.fetch_all_external_data()
            
            if not all_data:
                logger.error("データ取得に失敗しました")
                return None
            
            # 統合データセット作成
            integrated_df = self.create_integrated_dataset(all_data)
            
            if integrated_df.empty:
                logger.error("統合データセット作成に失敗しました")
                return None
            
            # 保存
            parquet_file, csv_file = self.save_integrated_data(integrated_df)
            
            # 成功統計
            success_count = len([k for k, v in all_data.items() if not v.empty])
            total_count = len(self.symbols)
            
            logger.info(f"🎉 Yahoo Finance 10年分拡張データ取得完了!")
            logger.info(f"成功率: {success_count}/{total_count} ({success_count/total_count*100:.1f}%)")
            logger.info(f"統合データ: {len(integrated_df):,}件")
            logger.info(f"保存先: {parquet_file}")
            
            return {
                'all_data': all_data,
                'integrated_data': integrated_df,
                'files': {'parquet': parquet_file, 'csv': csv_file},
                'success_rate': success_count / total_count,
                'summary': {
                    'total_records': len(integrated_df),
                    'date_range': f"{integrated_df['Date'].min()} - {integrated_df['Date'].max()}",
                    'columns': len(integrated_df.columns),
                    'symbols_success': success_count,
                    'symbols_total': total_count
                }
            }
            
        except Exception as e:
            logger.error(f"拡張データ取得エラー: {e}")
            return None

def main():
    """メイン実行"""
    # 10年間のデータを取得
    fetcher = YahooFinanceExtendedFetcher()
    
    results = fetcher.run_extended_fetch()
    
    if results:
        print(f"\n✅ Yahoo Finance 10年分データ取得完了!")
        print(f"📊 取得統計:")
        print(f"  - 成功率: {results['success_rate']:.1%}")
        print(f"  - 総レコード数: {results['summary']['total_records']:,}件")
        print(f"  - データ期間: {results['summary']['date_range']}")
        print(f"  - カラム数: {results['summary']['columns']}個")
        print(f"  - 成功シンボル: {results['summary']['symbols_success']}/{results['summary']['symbols_total']}")
        print(f"\n📁 保存ファイル:")
        print(f"  - Parquet: {results['files']['parquet']}")
        print(f"  - CSV: {results['files']['csv']}")
    else:
        print("\n❌ データ取得に失敗しました")

if __name__ == "__main__":
    main()