#!/usr/bin/env python
"""
最新の収集データを統合ファイルにマージするスクリプト
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

def merge_latest_data():
    """最新データを統合ファイルにマージ"""
    
    # 1. 既存の統合ファイルを読み込み
    integrated_path = Path("data/processed/integrated_with_external.parquet")
    if not integrated_path.exists():
        logger.error("統合ファイルが見つかりません")
        return False
    
    logger.info(f"既存の統合ファイルを読み込み: {integrated_path}")
    integrated_df = pd.read_parquet(integrated_path)
    integrated_df['Date'] = pd.to_datetime(integrated_df['Date'])
    old_max_date = integrated_df['Date'].max()
    logger.info(f"既存データの最新日付: {old_max_date}")
    
    # 2. rawディレクトリから最新のデータファイルを探す
    raw_dir = Path("data/raw")
    latest_files = []
    
    # 2025年8月30日以降のファイルを探す
    for file_path in raw_dir.glob("nikkei225_historical_*.parquet"):
        if "2025-08-30" in str(file_path) or "2025-09" in str(file_path):
            latest_files.append(file_path)
    
    if not latest_files:
        logger.warning("新しいデータファイルが見つかりません")
        return False
    
    # 3. 全ての新しいデータを読み込んで結合
    new_data_frames = []
    for file_path in latest_files:
        logger.info(f"読み込み: {file_path.name}")
        df = pd.read_parquet(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        # 既存データより新しい日付のみ
        new_dates = df[df['Date'] > old_max_date]
        if len(new_dates) > 0:
            logger.info(f"  → {new_dates['Date'].nunique()}日分の新データ")
            new_data_frames.append(new_dates)
    
    if not new_data_frames:
        logger.info("新しい日付のデータはありません")
        return True
    
    # データを結合
    new_data = pd.concat(new_data_frames, ignore_index=True)
    new_data = new_data.drop_duplicates(subset=['Date', 'Code']).sort_values(['Date', 'Code'])
    
    logger.info(f"新データ期間: {new_data['Date'].min()} ～ {new_data['Date'].max()}")
    logger.info(f"新データ件数: {len(new_data)}件")
    
    # 4. 特徴量を計算
    logger.info("特徴量を計算中...")
    
    # 各銘柄ごとに特徴量を計算
    processed_data = []
    for code in new_data['Code'].unique():
        # 既存データと新データを結合して計算（移動平均等のため）
        existing_code_data = integrated_df[integrated_df['Code'] == code].tail(100)  # 直近100日分
        new_code_data = new_data[new_data['Code'] == code]
        
        # 一時的に結合
        temp_df = pd.concat([existing_code_data, new_code_data], ignore_index=True)
        temp_df = temp_df.sort_values('Date')
        
        # 特徴量計算
        # Binary_Direction (翌日の価格変化)
        temp_df['Binary_Direction'] = (temp_df['Close'].shift(-1) > temp_df['Close']).astype(int)
        
        # 移動平均
        temp_df['MA5'] = temp_df['Close'].rolling(window=5, min_periods=1).mean()
        temp_df['MA20'] = temp_df['Close'].rolling(window=20, min_periods=1).mean()
        temp_df['MA60'] = temp_df['Close'].rolling(window=60, min_periods=1).mean()
        
        # ボラティリティ
        temp_df['Volatility_20'] = temp_df['Close'].pct_change().rolling(window=20, min_periods=1).std()
        
        # 価格と移動平均の比率
        temp_df['Price_vs_MA20'] = temp_df['Close'] / temp_df['MA20'] - 1
        
        # RSI
        delta = temp_df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        temp_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 新しい日付のデータのみ抽出
        new_rows = temp_df[temp_df['Date'] > old_max_date]
        if len(new_rows) > 0:
            processed_data.append(new_rows)
    
    if not processed_data:
        logger.warning("処理済みデータがありません")
        return False
    
    # 処理済みデータを結合
    processed_df = pd.concat(processed_data, ignore_index=True)
    
    # 市場全体の特徴量を追加
    for date in processed_df['Date'].unique():
        date_data = processed_df[processed_df['Date'] == date]
        
        # Market Breadth
        up_count = (date_data['Close'] > date_data['Open']).sum()
        total_count = len(date_data)
        market_breadth = up_count / total_count if total_count > 0 else 0.5
        
        # Market Return
        if len(date_data) > 0:
            market_return = ((date_data['Close'] / date_data['Open']) - 1).mean()
        else:
            market_return = 0
        
        processed_df.loc[processed_df['Date'] == date, 'Market_Breadth'] = market_breadth
        processed_df.loc[processed_df['Date'] == date, 'Market_Return'] = market_return
    
    # 外部データ列（デフォルト値）
    external_cols = ['sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change']
    for col in external_cols:
        if col in integrated_df.columns and col not in processed_df.columns:
            processed_df[col] = 0
    
    # 5. 既存データの列に合わせる
    for col in integrated_df.columns:
        if col not in processed_df.columns:
            if col in ['Binary_Direction', 'Market_Breadth', 'Market_Return']:
                continue  # 既に計算済み
            processed_df[col] = np.nan
    
    # 列の順序を合わせる
    processed_df = processed_df[integrated_df.columns]
    
    # 6. 既存データと結合
    updated_df = pd.concat([integrated_df, processed_df], ignore_index=True)
    updated_df = updated_df.drop_duplicates(subset=['Date', 'Code'], keep='last')
    updated_df = updated_df.sort_values(['Date', 'Code']).reset_index(drop=True)
    
    # 7. バックアップを作成
    backup_path = integrated_path.parent / f"integrated_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    logger.info(f"バックアップを作成: {backup_path}")
    integrated_df.to_parquet(backup_path, compression='snappy', index=False)
    
    # 8. 更新したデータを保存
    logger.info(f"更新データを保存: {integrated_path}")
    updated_df.to_parquet(integrated_path, compression='snappy', index=False)
    
    logger.info(f"✅ データ更新完了:")
    logger.info(f"   - 旧データ最終日: {old_max_date}")
    logger.info(f"   - 新データ最終日: {updated_df['Date'].max()}")
    logger.info(f"   - 総レコード数: {len(integrated_df)} → {len(updated_df)}")
    logger.info(f"   - 追加レコード数: {len(updated_df) - len(integrated_df)}")
    
    # 最新の日付を表示
    logger.info("\n最新5日分のデータ:")
    latest_dates = updated_df['Date'].value_counts().sort_index().tail(5)
    for date, count in latest_dates.items():
        logger.info(f"   {date.strftime('%Y-%m-%d')}: {count}件")
    
    return True

if __name__ == "__main__":
    success = merge_latest_data()
    if success:
        print("\n✅ 統合データファイルの更新が完了しました")
    else:
        print("\n❌ 統合データファイルの更新に失敗しました")