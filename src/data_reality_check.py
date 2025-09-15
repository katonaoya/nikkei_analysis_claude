"""
データの実在性・妥当性検証スクリプト
100%精度の結果が本当に実データに基づいているかを確認
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_real_data():
    """実データの詳細分析"""
    logger.info("=== 実データ検証開始 ===")
    
    # データ読み込み
    data_file = Path("data/nikkei225_full_data/nikkei225_full_10years_175stocks_20250831_020101.pkl")
    if not data_file.exists():
        logger.error(f"データファイルが見つかりません: {data_file}")
        return
    
    df = pd.read_pickle(data_file)
    
    # 基本情報
    logger.info(f"データサイズ: {len(df):,}レコード")
    logger.info(f"列数: {len(df.columns)}")
    logger.info(f"銘柄数: {df['Code'].nunique()}")
    logger.info(f"期間: {df['Date'].min()} ～ {df['Date'].max()}")
    
    print(f"\n=== データ基本情報 ===")
    print(f"レコード数: {len(df):,}")
    print(f"銘柄数: {df['Code'].nunique()}")
    print(f"期間: {df['Date'].min()} ～ {df['Date'].max()}")
    
    # データサンプル表示
    print(f"\n=== データサンプル ===")
    print(df.head(10))
    
    # 列情報
    print(f"\n=== 全カラム一覧 ===")
    for i, col in enumerate(df.columns):
        print(f"{i+1:2d}. {col}")
    
    # 価格データの統計
    print(f"\n=== 価格統計 ===")
    if 'Close' in df.columns:
        print(f"終値統計:")
        print(df['Close'].describe())
        
        # 実際のリターン計算
        df['close_price'] = pd.to_numeric(df['Close'], errors='coerce')
        df['daily_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None)
        df['next_day_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None).shift(-1)
        
        print(f"\n=== リターン分析 ===")
        print(f"日次リターン統計:")
        print(df['daily_return'].describe())
        
        # 2%以上上昇の実際の頻度
        target_2pct = (df['next_day_return'] >= 0.02)
        print(f"\n2%以上上昇の頻度: {target_2pct.mean():.1%} ({target_2pct.sum():,}/{len(df):,})")
        
        # 1%以上上昇の実際の頻度  
        target_1pct = (df['next_day_return'] >= 0.01)
        print(f"1%以上上昇の頻度: {target_1pct.mean():.1%} ({target_1pct.sum():,}/{len(df):,})")
        
        # 0.5%以上上昇の実際の頻度
        target_05pct = (df['next_day_return'] >= 0.005)
        print(f"0.5%以上上昇の頻度: {target_05pct.mean():.1%} ({target_05pct.sum():,}/{len(df):,})")
        
    # データの実在性確認
    print(f"\n=== データ実在性確認 ===")
    
    # 欠損値チェック
    missing_data = df.isnull().sum()
    print(f"欠損値がある列: {len(missing_data[missing_data > 0])}")
    if len(missing_data[missing_data > 0]) > 0:
        print("欠損値詳細:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"  {col}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # 重複データチェック
    duplicates = df.duplicated(subset=['Code', 'Date']).sum()
    print(f"重複レコード: {duplicates:,}")
    
    # 価格の妥当性チェック（異常値）
    if 'Close' in df.columns:
        close_prices = pd.to_numeric(df['Close'], errors='coerce')
        print(f"\n価格範囲: {close_prices.min():.0f} ～ {close_prices.max():,.0f} 円")
        
        # 異常な価格変動チェック
        daily_changes = df.groupby('Code')['close_price'].pct_change(fill_method=None).abs()
        extreme_changes = daily_changes > 0.5  # 50%以上の変動
        print(f"極端な日次変動(>50%): {extreme_changes.sum():,} ({extreme_changes.mean():.3%})")
        
    # ボリュームの妥当性
    if 'Volume' in df.columns:
        volumes = pd.to_numeric(df['Volume'], errors='coerce')
        print(f"出来高範囲: {volumes.min():,.0f} ～ {volumes.max():,.0f}")
        zero_volume = (volumes == 0).sum()
        print(f"出来高0の日: {zero_volume:,} ({zero_volume/len(df)*100:.1f}%)")
    
    # 銘柄別データ数確認
    print(f"\n=== 銘柄別データ分析 ===")
    stock_counts = df['Code'].value_counts()
    print(f"銘柄別レコード数統計:")
    print(f"  平均: {stock_counts.mean():.0f}")
    print(f"  最小: {stock_counts.min()}")  
    print(f"  最大: {stock_counts.max()}")
    print(f"  標準偏差: {stock_counts.std():.0f}")
    
    # データ品質スコア
    quality_score = 100
    if duplicates > 0:
        quality_score -= 10
    if missing_data.sum() > len(df) * 0.01:  # 1%以上欠損
        quality_score -= 20
    if extreme_changes.mean() > 0.01:  # 1%以上が極端変動
        quality_score -= 15
        
    print(f"\n=== データ品質評価 ===")
    print(f"品質スコア: {quality_score}/100")
    
    return df


if __name__ == "__main__":
    df = analyze_real_data()