#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ統合デバッグスクリプト
外部指標データと株価データの統合問題を調査
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def debug_data_integration():
    """データ統合問題のデバッグ"""
    
    # データファイル
    stock_file = "data/processed/nikkei225_complete_225stocks_20250909_230649.parquet"
    external_file = "data/external_extended/external_integrated_10years_20250909_231815.parquet"
    
    logger.info("データ読み込み開始...")
    
    # 株価データ読み込み
    stock_df = pd.read_parquet(stock_file)
    logger.info(f"株価データ: {len(stock_df):,}件, {stock_df['Code'].nunique()}銘柄")
    logger.info(f"株価データ期間: {stock_df['Date'].min()} 〜 {stock_df['Date'].max()}")
    logger.info(f"株価データカラム: {list(stock_df.columns)}")
    
    # 外部指標データ読み込み
    external_df = pd.read_parquet(external_file)
    logger.info(f"外部指標データ: {len(external_df):,}件")
    logger.info(f"外部指標データ期間: {external_df['Date'].min()} 〜 {external_df['Date'].max()}")
    logger.info(f"外部指標データカラム数: {len(external_df.columns)}")
    
    # 日付型統一
    stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
    external_df['Date'] = pd.to_datetime(external_df['Date']).dt.tz_localize(None)
    
    # 日付重複確認
    stock_dates = set(stock_df['Date'].dt.date)
    external_dates = set(external_df['Date'].dt.date)
    
    logger.info(f"株価データ日付数: {len(stock_dates)}")
    logger.info(f"外部指標データ日付数: {len(external_dates)}")
    logger.info(f"重複日付数: {len(stock_dates & external_dates)}")
    
    # サンプルデータでテスト統合
    sample_stock = stock_df.head(1000).copy()
    
    # 統合テスト
    logger.info("統合テスト開始...")
    merged_df = pd.merge(sample_stock, external_df, on='Date', how='left')
    logger.info(f"統合結果: {len(merged_df):,}件")
    
    # 欠損値確認
    external_cols = [col for col in external_df.columns if col != 'Date']
    for col in external_cols:
        if col in merged_df.columns:
            null_count = merged_df[col].isnull().sum()
            logger.info(f"{col}の欠損値: {null_count}件 ({null_count/len(merged_df)*100:.1f}%)")
    
    # 基本的な特徴量計算テスト
    logger.info("基本特徴量計算テスト...")
    merged_df['Returns'] = merged_df['Close'].pct_change(fill_method=None)
    
    # 目的変数作成テスト
    logger.info("目的変数作成テスト...")
    merged_df['Target'] = 0
    
    # 銘柄別に処理
    for code in merged_df['Code'].unique()[:3]:  # 3銘柄のみテスト
        mask = merged_df['Code'] == code
        code_data = merged_df[mask].copy()
        next_high = code_data['High'].shift(-1)
        prev_close = code_data['Close'].shift(1)
        merged_df.loc[mask, 'Target'] = (next_high / prev_close > 1.01).astype(int)
    
    # 欠損値処理テスト
    logger.info("欠損値処理テスト...")
    logger.info(f"処理前: {len(merged_df)}件")
    
    # 無限値処理
    merged_df = merged_df.replace([np.inf, -np.inf], np.nan)
    
    # 必須カラムの欠損値除去
    merged_df = merged_df.dropna(subset=['Close', 'Date', 'Code'])
    logger.info(f"必須カラム欠損値除去後: {len(merged_df)}件")
    
    # Target欠損値除去
    merged_df = merged_df.dropna(subset=['Target'])
    logger.info(f"Target欠損値除去後: {len(merged_df)}件")
    
    # 前方補完
    merged_df = merged_df.fillna(method='ffill').fillna(method='bfill')
    logger.info(f"前方補完後: {len(merged_df)}件")
    
    # 最終クリーンアップ
    merged_df = merged_df.dropna()
    logger.info(f"最終クリーンアップ後: {len(merged_df)}件")
    
    if len(merged_df) > 0:
        logger.info(f"Target正例率: {merged_df['Target'].mean():.3f}")
        logger.info("✅ データ統合テスト成功")
        
        # サンプル保存
        sample_file = "debug_sample_integrated.parquet"
        merged_df.to_parquet(sample_file, index=False)
        logger.info(f"サンプル統合データ保存: {sample_file}")
        
    else:
        logger.error("❌ データ統合後に0件になりました")
    
    return merged_df

if __name__ == "__main__":
    debug_data_integration()