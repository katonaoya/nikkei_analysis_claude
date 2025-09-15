#!/usr/bin/env python
"""
統合データファイルを最新データで更新するスクリプト
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

def update_integrated_data():
    """統合データファイルを更新"""
    
    # 1. 既存の統合ファイルを読み込み
    integrated_path = Path("data/processed/integrated_with_external.parquet")
    if integrated_path.exists():
        logger.info(f"既存の統合ファイルを読み込み: {integrated_path}")
        integrated_df = pd.read_parquet(integrated_path)
        integrated_df['Date'] = pd.to_datetime(integrated_df['Date'])
        old_max_date = integrated_df['Date'].max()
        logger.info(f"既存データの最新日付: {old_max_date}")
    else:
        logger.error("統合ファイルが見つかりません")
        return False
    
    # 2. 2024年のrawデータから最新データを取得
    raw_2024_path = Path("data/raw/nikkei225_historical_2024-01-01_2024-12-31_20250830_195947.parquet")
    if not raw_2024_path.exists():
        logger.error(f"2024年データファイルが見つかりません: {raw_2024_path}")
        return False
    
    logger.info(f"2024年データを読み込み: {raw_2024_path}")
    raw_2024_df = pd.read_parquet(raw_2024_path)
    raw_2024_df['Date'] = pd.to_datetime(raw_2024_df['Date'])
    
    # 3. 統合ファイルに含まれていない新しいデータを抽出
    new_data = raw_2024_df[raw_2024_df['Date'] > old_max_date].copy()
    
    if len(new_data) == 0:
        logger.info("新しいデータはありません")
        return True
    
    logger.info(f"新しいデータを発見: {new_data['Date'].min()} ～ {new_data['Date'].max()} ({len(new_data)}件)")
    
    # 4. 特徴量を計算（既存の統合ファイルと同じ形式に）
    # 必要な列をチェック
    required_cols = integrated_df.columns.tolist()
    logger.info(f"必要な列数: {len(required_cols)}")
    
    # 基本的な特徴量を追加
    for code in new_data['Code'].unique():
        code_data = new_data[new_data['Code'] == code].sort_values('Date')
        
        # Binary_Direction (翌日の価格変化)
        code_data['Binary_Direction'] = (code_data['Close'].shift(-1) > code_data['Close']).astype(int)
        
        # 移動平均
        code_data['MA5'] = code_data['Close'].rolling(window=5, min_periods=1).mean()
        code_data['MA20'] = code_data['Close'].rolling(window=20, min_periods=1).mean()
        code_data['MA60'] = code_data['Close'].rolling(window=60, min_periods=1).mean()
        
        # ボラティリティ
        code_data['Volatility_20'] = code_data['Close'].pct_change().rolling(window=20, min_periods=1).std()
        
        # 価格と移動平均の比率
        code_data['Price_vs_MA20'] = code_data['Close'] / code_data['MA20'] - 1
        
        # RSI（簡易版）
        delta = code_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
        rs = gain / loss.replace(0, 1e-10)
        code_data['RSI'] = 100 - (100 / (1 + rs))
        
        new_data.loc[new_data['Code'] == code, code_data.columns] = code_data
    
    # 市場全体の特徴量を追加
    for date in new_data['Date'].unique():
        date_data = new_data[new_data['Date'] == date]
        
        # Market Breadth (上昇銘柄の割合)
        up_count = (date_data['Close'] > date_data['Open']).sum()
        total_count = len(date_data)
        market_breadth = up_count / total_count if total_count > 0 else 0.5
        
        # Market Return (市場全体のリターン)
        market_return = date_data['Close'].pct_change().mean()
        
        new_data.loc[new_data['Date'] == date, 'Market_Breadth'] = market_breadth
        new_data.loc[new_data['Date'] == date, 'Market_Return'] = market_return
    
    # 外部データのプレースホルダー（統合ファイルに存在する場合）
    external_cols = ['sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change']
    for col in external_cols:
        if col in integrated_df.columns and col not in new_data.columns:
            new_data[col] = 0  # デフォルト値
    
    # 5. 既存データと新しいデータを結合
    # 既存データの列に合わせる
    for col in integrated_df.columns:
        if col not in new_data.columns:
            new_data[col] = np.nan
    
    # 列の順序を合わせる
    new_data = new_data[integrated_df.columns]
    
    # 結合
    updated_df = pd.concat([integrated_df, new_data], ignore_index=True)
    updated_df = updated_df.sort_values(['Date', 'Code']).reset_index(drop=True)
    
    # 6. バックアップを作成
    backup_path = integrated_path.parent / f"integrated_with_external_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    logger.info(f"バックアップを作成: {backup_path}")
    integrated_df.to_parquet(backup_path, compression='snappy', index=False)
    
    # 7. 更新したデータを保存
    logger.info(f"更新データを保存: {integrated_path}")
    updated_df.to_parquet(integrated_path, compression='snappy', index=False)
    
    logger.info(f"✅ データ更新完了:")
    logger.info(f"   - 旧データ: {integrated_df['Date'].min()} ～ {integrated_df['Date'].max()} ({len(integrated_df)}件)")
    logger.info(f"   - 新データ: {updated_df['Date'].min()} ～ {updated_df['Date'].max()} ({len(updated_df)}件)")
    logger.info(f"   - 追加件数: {len(updated_df) - len(integrated_df)}件")
    
    return True

if __name__ == "__main__":
    success = update_integrated_data()
    if success:
        print("✅ 統合データファイルの更新が完了しました")
    else:
        print("❌ 統合データファイルの更新に失敗しました")