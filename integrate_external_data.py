#!/usr/bin/env python
"""
外部データを統合データファイルに組み込むスクリプト
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import logging

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def integrate_external_data():
    """外部データを統合ファイルに組み込む"""
    
    # 1. 既存の統合ファイルを読み込み
    integrated_path = Path("data/processed/integrated_with_external.parquet")
    if not integrated_path.exists():
        logger.error("統合ファイルが見つかりません")
        return False
    
    logger.info(f"既存の統合ファイルを読み込み: {integrated_path}")
    integrated_df = pd.read_parquet(integrated_path)
    integrated_df['Date'] = pd.to_datetime(integrated_df['Date'])
    logger.info(f"統合ファイルのデータ数: {len(integrated_df)}")
    
    # 2. 外部データを読み込み
    external_path = Path("data/external/external_macro_data.parquet")
    if not external_path.exists():
        logger.error("外部データファイルが見つかりません")
        return False
    
    logger.info(f"外部データを読み込み: {external_path}")
    external_df = pd.read_parquet(external_path)
    external_df['Date'] = pd.to_datetime(external_df['Date'])
    logger.info(f"外部データのデータ数: {len(external_df)}")
    logger.info(f"外部データの期間: {external_df['Date'].min()} ～ {external_df['Date'].max()}")
    
    # 3. 必要な列を選択（既存の列名に合わせる）
    external_cols_mapping = {
        'sp500_change': 'sp500_change',
        'vix_change': 'vix_change', 
        'nikkei_change': 'nikkei_change',
        'us_10y_change': 'us_10y_change',
        'usd_jpy_change': 'usd_jpy_change'
    }
    
    # 外部データから必要な列を抽出
    external_features = external_df[['Date'] + list(external_cols_mapping.keys())].copy()
    
    # 4. 既存データの外部データ列を更新（Dateでマージ）
    logger.info("外部データを統合中...")
    
    # 既存の外部データ列を削除（既にある場合）
    for col in external_cols_mapping.values():
        if col in integrated_df.columns:
            integrated_df = integrated_df.drop(columns=[col])
    
    # Dateで左結合（統合ファイルの全データを保持）
    updated_df = integrated_df.merge(external_features, on='Date', how='left')
    
    # 欠損値の処理（前日の値で埋める）
    for col in external_cols_mapping.values():
        if col in updated_df.columns:
            # 各銘柄ごとに前方補完
            updated_df[col] = updated_df.groupby('Code')[col].fillna(method='ffill')
            # それでも欠損している場合は0で埋める
            updated_df[col] = updated_df[col].fillna(0)
    
    # 5. データの検証
    logger.info("データ検証中...")
    for col in external_cols_mapping.values():
        non_zero = (updated_df[col] != 0).sum()
        total = len(updated_df)
        coverage = (non_zero / total) * 100
        logger.info(f"  {col}: {non_zero:,}/{total:,} ({coverage:.1f}%)")
    
    # 最新日付のデータを確認
    latest_date = updated_df['Date'].max()
    latest_data = updated_df[updated_df['Date'] == latest_date]
    logger.info(f"\n最新日付 ({latest_date}) のデータ:")
    for col in external_cols_mapping.values():
        if col in latest_data.columns:
            non_zero = (latest_data[col] != 0).sum()
            logger.info(f"  {col}: {non_zero}/{len(latest_data)} 件が非ゼロ")
    
    # 6. バックアップを作成
    backup_path = integrated_path.parent / f"integrated_backup_before_external_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    logger.info(f"バックアップを作成: {backup_path}")
    integrated_df.to_parquet(backup_path, compression='snappy', index=False)
    
    # 7. 更新したデータを保存
    logger.info(f"更新データを保存: {integrated_path}")
    updated_df.to_parquet(integrated_path, compression='snappy', index=False)
    
    logger.info(f"✅ 外部データ統合完了:")
    logger.info(f"   - 総レコード数: {len(updated_df)}")
    logger.info(f"   - 日付範囲: {updated_df['Date'].min()} ～ {updated_df['Date'].max()}")
    
    # 特定日付のデータサンプルを表示
    sample_dates = ['2025-08-29', '2025-09-01', '2025-09-02']
    for date_str in sample_dates:
        sample_date = pd.to_datetime(date_str)
        sample_data = updated_df[updated_df['Date'] == sample_date]
        if len(sample_data) > 0:
            logger.info(f"\n{date_str}のデータサンプル:")
            first_row = sample_data.iloc[0]
            for col in external_cols_mapping.values():
                if col in first_row:
                    logger.info(f"  {col}: {first_row[col]:.4f}")
    
    return True

if __name__ == "__main__":
    success = integrate_external_data()
    if success:
        print("\n✅ 外部データの統合が完了しました")
    else:
        print("\n❌ 外部データの統合に失敗しました")