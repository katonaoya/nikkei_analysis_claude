#!/usr/bin/env python3
"""
外部データ統合修正
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

def fix_external_data():
    """外部データの修正と統合"""
    logger.info("🔧 外部データ修正と統合")
    
    # 外部データファイル読み込み
    external_file = Path("data/external/external_macro_data.parquet")
    if not external_file.exists():
        logger.error("❌ 外部データファイルが見つかりません")
        return None
    
    # マルチレベルカラムを修正
    external_data = pd.read_parquet(external_file)
    
    # カラム名を修正（タプルから文字列に）
    if isinstance(external_data.columns[0], tuple):
        external_data.columns = [col[0] for col in external_data.columns]
    
    logger.info(f"外部データ: {len(external_data):,}件")
    logger.info(f"カラム: {list(external_data.columns)}")
    
    # 日付カラムの処理
    external_data['Date'] = pd.to_datetime(external_data['Date'])
    
    # 既存データ読み込み
    processed_dir = Path("data/processed")
    existing_files = list(processed_dir.glob("*.parquet"))
    
    if not existing_files:
        logger.error("❌ 既存の処理済みデータが見つかりません")
        return None
        
    existing_data = pd.read_parquet(existing_files[0])
    existing_data['Date'] = pd.to_datetime(existing_data['Date'])
    logger.info(f"既存データ: {len(existing_data):,}件")
    
    # 日付ベースで統合
    logger.info("🔗 データ統合中...")
    integrated_data = existing_data.merge(external_data, on='Date', how='left')
    
    # 外部データの前埋め（平日の株式データに週末の外部データを適用）
    external_cols = [col for col in external_data.columns if col != 'Date']
    logger.info(f"外部データカラム: {len(external_cols)}個")
    
    # 前埋め処理
    integrated_data[external_cols] = integrated_data[external_cols].fillna(method='ffill')
    
    # 欠損値統計
    missing_stats = integrated_data[external_cols].isnull().sum()
    logger.info(f"前埋め後の欠損値:")
    for col, missing_count in missing_stats[missing_stats > 0].items():
        logger.info(f"  {col}: {missing_count:,}件 ({missing_count/len(integrated_data)*100:.1f}%)")
    
    # 残りの欠損値は0で埋める
    integrated_data[external_cols] = integrated_data[external_cols].fillna(0)
    
    logger.info(f"✅ データ統合完了: {len(integrated_data):,}件")
    logger.info(f"総カラム数: {len(integrated_data.columns)}")
    
    # 統合データ保存
    integrated_file = processed_dir / "integrated_with_external.parquet"
    integrated_data.to_parquet(integrated_file, index=False)
    logger.info(f"💾 統合データ保存: {integrated_file}")
    
    # サンプル表示
    logger.info(f"\n📋 統合データサンプル（外部データ部分）:")
    external_sample = integrated_data[['Date'] + external_cols].tail()
    print(external_sample.to_string(index=False))
    
    return integrated_data

def validate_integration():
    """統合データの検証"""
    logger.info("✅ 統合データ検証")
    
    integrated_file = Path("data/processed/integrated_with_external.parquet")
    if not integrated_file.exists():
        logger.error("❌ 統合データファイルが見つかりません")
        return
    
    data = pd.read_parquet(integrated_file)
    logger.info(f"統合データ: {len(data):,}件, {len(data.columns)}カラム")
    
    # 外部データカラムの確認
    external_pattern_cols = [col for col in data.columns if any(pattern in col for pattern in ['us_10y', 'sp500', 'usd_jpy', 'nikkei', 'vix'])]
    logger.info(f"外部データカラム: {len(external_pattern_cols)}個")
    
    for col in external_pattern_cols:
        non_null_count = data[col].notna().sum()
        logger.info(f"  {col}: {non_null_count:,}/{len(data):,}件 ({non_null_count/len(data)*100:.1f}%)")
    
    # 統計情報
    if external_pattern_cols:
        logger.info(f"\n📊 外部データ統計:")
        stats = data[external_pattern_cols].describe()
        print(stats.round(4).to_string())
    
    # Binary_Directionの存在確認
    if 'Binary_Direction' in data.columns:
        valid_targets = data['Binary_Direction'].notna().sum()
        logger.info(f"\n🎯 予測対象: {valid_targets:,}件")
        logger.info(f"✅ 次のステップ: 新特徴量での精度評価が可能")
    else:
        logger.warning("⚠️ Binary_Directionが見つかりません")

def main():
    """メイン実行"""
    logger.info("🚀 外部データ統合修正システム")
    
    try:
        # 1. 外部データ修正と統合
        integrated_data = fix_external_data()
        
        if integrated_data is not None:
            # 2. 統合データ検証
            validate_integration()
            
            logger.info("\n" + "="*80)
            logger.info("✅ 外部データ統合修正完了")
            logger.info("="*80)
            logger.info("🚀 次は新特徴量での精度評価を実行してください")
        else:
            logger.error("❌ 統合に失敗しました")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()