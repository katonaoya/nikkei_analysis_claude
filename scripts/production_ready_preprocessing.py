#!/usr/bin/env python3
"""
本番運用レベルのデータ前処理
欠損値処理、データリークage完全排除、時系列整合性確保
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime, timedelta
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class ProductionDataPreprocessor:
    """本番運用レベルのデータ前処理"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_clean_raw_data(self, file_pattern: str) -> pd.DataFrame:
        """生データの読み込みとクリーニング"""
        logger.info("📊 Loading and cleaning raw data...")
        
        data_files = list(self.raw_dir.glob(f"{file_pattern}.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No data files found matching pattern: {file_pattern}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        # 全ファイルを結合
        dfs = []
        for file_path in sorted(data_files):
            logger.info(f"Loading: {file_path.name}")
            df_file = pd.read_parquet(file_path)
            
            # 重複列の問題をここで修正
            if df_file.columns.duplicated().any():
                logger.warning(f"File {file_path.name} has duplicate columns, fixing...")
                df_file = df_file.loc[:, ~df_file.columns.duplicated()]
            
            df_file = self._standardize_columns(df_file)
            dfs.append(df_file)
        
        # データ結合
        df = pd.concat(dfs, ignore_index=True)
        
        # 結合後の重複列チェック（最終確認）
        if df.columns.duplicated().any():
            logger.warning("Duplicate columns found after concat, removing...")
            df = df.loc[:, ~df.columns.duplicated()]
        
        logger.info(f"Final columns: {list(df.columns)}")
        
        # 重複除去（より厳格）
        initial_count = len(df)
        df = df.drop_duplicates(subset=['Date', 'Code'], keep='last')
        duplicate_removed = initial_count - len(df)
        if duplicate_removed > 0:
            logger.info(f"Removed {duplicate_removed} duplicate records")
        
        # 日付順ソート
        df = df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records for {df['Code'].nunique()} stocks")
        logger.info(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """カラム名の標準化"""
        df = df.copy()
        
        # 重複列の削除
        df = df.loc[:, ~df.columns.duplicated()]
        
        # 日付変換
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
        
        # カラム名の統一
        rename_map = {
            'close': 'Close', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'volume': 'Volume', 'code': 'Code'
        }
        
        df = df.rename(columns=rename_map)
        return df
    
    def strict_missing_value_handling(self, df: pd.DataFrame) -> pd.DataFrame:
        """厳格な欠損値処理"""
        logger.info("🔧 Applying strict missing value handling...")
        
        # 基本的な価格データが欠損している行を除外
        critical_cols = ['Date', 'Code', 'Close']
        for col in critical_cols:
            if col in df.columns:
                missing_count = df[col].isnull().sum()
                if missing_count > 0:
                    logger.warning(f"Removing {missing_count} records with missing {col}")
                    df = df[df[col].notna()]
        
        # 価格データの妥当性チェック（より厳格）
        if 'Close' in df.columns:
            # 負の価格や0の除外
            invalid_price_count = (df['Close'] <= 0).sum()
            if invalid_price_count > 0:
                logger.warning(f"Removing {invalid_price_count} records with invalid prices")
                df = df[df['Close'] > 0]
            
            # 極端に高い価格（10万円超）の除外 - より厳格
            extreme_price_count = (df['Close'] > 100000).sum()
            if extreme_price_count > 0:
                logger.warning(f"Removing {extreme_price_count} records with extreme prices (>100k)")
                df = df[df['Close'] <= 100000]
            
            # 極端に安い価格（10円未満）の除外
            low_price_count = (df['Close'] < 10).sum()
            if low_price_count > 0:
                logger.warning(f"Removing {low_price_count} records with extremely low prices (<10)")
                df = df[df['Close'] >= 10]
        
        # 銘柄ごとの連続性チェックと日付ギャップ問題の修正
        df = df.sort_values(['Code', 'Date'])
        
        clean_stocks = []
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code].copy().sort_values('Date')
            
            # 5日以上の日付ギャップがある場合、そのギャップを削除
            stock_data['Date_Diff'] = stock_data['Date'].diff().dt.days
            large_gaps = stock_data['Date_Diff'] > 5
            
            if large_gaps.any():
                logger.warning(f"Stock {code}: Removing {large_gaps.sum()} records with large date gaps")
                stock_data = stock_data[~large_gaps]
            
            # 最低30日のデータが必要
            if len(stock_data) >= 30:
                stock_data = stock_data.drop('Date_Diff', axis=1)
                clean_stocks.append(stock_data)
        
        if not clean_stocks:
            raise ValueError("No stocks with sufficient data after cleaning")
        
        df = pd.concat(clean_stocks, ignore_index=True)
        
        # 数値列の前方補完（最大5日まで）
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in ['Date', 'Code']]
        
        for code in df['Code'].unique():
            mask = df['Code'] == code
            stock_data = df[mask].copy()
            
            # 前方補完（最大5営業日）
            for col in numeric_cols:
                if col in stock_data.columns:
                    stock_data[col] = stock_data[col].fillna(method='ffill', limit=5)
            
            df.loc[mask] = stock_data
        
        # 残った欠損値は除外（保守的アプローチ）
        initial_count = len(df)
        df = df.dropna(subset=numeric_cols[:10])  # 主要な数値列のみチェック
        removed_count = initial_count - len(df)
        
        if removed_count > 0:
            logger.info(f"Removed {removed_count} records with remaining missing values")
        
        # 最終的な欠損値レポート
        missing_summary = df.isnull().sum()
        total_missing = missing_summary.sum()
        missing_rate = total_missing / (len(df) * len(df.columns))
        
        logger.info(f"Final missing values: {total_missing} ({missing_rate:.2%})")
        
        return df
    
    def generate_leak_free_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """リークageを完全排除した特徴量生成"""
        logger.info("🔒 Generating leak-free features...")
        
        results = []
        
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code].copy().sort_values('Date')
            
            if len(stock_data) < 30:  # 最低30日のデータが必要
                continue
            
            # 基本データ
            dates = stock_data['Date'].values
            close = stock_data['Close'].values
            high = stock_data['High'].values if 'High' in stock_data.columns else close
            low = stock_data['Low'].values if 'Low' in stock_data.columns else close
            volume = stock_data['Volume'].values if 'Volume' in stock_data.columns else np.ones(len(stock_data))
            
            # データフレームとして構築
            stock_df = pd.DataFrame({
                'Date': dates,
                'Code': code,  # 文字列として直接指定
                'Close': close,
                'High': high,
                'Low': low,
                'Volume': volume
            })
            
            # 厳格なリターン計算（リークage防止）
            returns = np.concatenate([[0], np.diff(np.log(close))])
            stock_df['Returns'] = returns
            
            # 移動平均（厳格な時系列処理）
            for period in [5, 10, 20]:
                ma = pd.Series(close).rolling(period, min_periods=period).mean().values
                stock_df[f'MA_{period}'] = ma
                
                # 価格と移動平均の乖離（前日時点の情報のみ使用）
                price_vs_ma = np.roll(close / ma - 1, 1)  # 1日ラグを適用
                price_vs_ma[0] = 0  # 最初の値は0
                stock_df[f'Price_vs_MA{period}'] = price_vs_ma
            
            # RSI（厳格な実装）
            rsi = self._calculate_strict_rsi(close, 14)
            stock_df['RSI_14'] = rsi
            
            # ボラティリティ（過去20日のリターンから計算）
            vol = pd.Series(returns).rolling(20, min_periods=20).std().values
            stock_df['Volatility_20'] = vol
            
            # ボリューム指標
            vol_ma = pd.Series(volume).rolling(10, min_periods=10).mean().values
            stock_df['Volume_MA_10'] = vol_ma
            vol_ratio = volume / vol_ma
            vol_ratio[vol_ma == 0] = 1  # ゼロ除算対策
            stock_df['Volume_Ratio'] = vol_ratio
            
            # 価格位置（過去20日のレンジ内での位置）
            high_20 = pd.Series(high).rolling(20, min_periods=20).max().values
            low_20 = pd.Series(low).rolling(20, min_periods=20).min().values
            price_position = (close - low_20) / (high_20 - low_20)
            price_position[high_20 == low_20] = 0.5  # レンジがない場合は中央値
            stock_df['Price_Position'] = price_position
            
            # ラグ特徴量（明示的な過去データ）
            for lag in [1, 2, 5]:
                stock_df[f'Return_Lag_{lag}'] = np.roll(returns, lag)
                stock_df[f'Close_Lag_{lag}'] = np.roll(close, lag)
            
            # ターゲット変数（厳格な未来データ）
            # 次日のリターン（現在のCloseから次日のCloseへの変化）
            next_close = np.roll(close, -1)
            next_return = next_close / close - 1
            next_return[-1] = 0  # 最後の日は予測不可能なので0
            
            stock_df['Next_Day_Return'] = next_return
            stock_df['Binary_Direction'] = (next_return > 0).astype(int)
            
            # 無限大・NaNの処理
            stock_df = stock_df.replace([np.inf, -np.inf], np.nan)
            
            # 最初の数日と最後の日は予測に使用しない（不完全なデータのため）
            stock_df = stock_df.iloc[20:-1]  # 最初の20日と最後の1日を除外
            
            if len(stock_df) > 0:
                results.append(stock_df)
        
        if not results:
            raise ValueError("No valid data after processing")
        
        final_df = pd.concat(results, ignore_index=True)
        
        # 最終的な欠損値処理
        final_df = final_df.fillna(0)
        
        logger.info(f"Generated features for {len(final_df)} records")
        logger.info(f"Feature count: {len(final_df.columns)}")
        
        return final_df
    
    def _calculate_strict_rsi(self, prices: np.array, period: int = 14) -> np.array:
        """厳格なRSI計算（リークage防止）"""
        if len(prices) < period + 1:
            return np.full(len(prices), 50.0)
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 移動平均の計算
        avg_gains = pd.Series(gains).rolling(period, min_periods=period).mean()
        avg_losses = pd.Series(losses).rolling(period, min_periods=period).mean()
        
        # RSIの計算
        rs = avg_gains / avg_losses
        rs = rs.fillna(0)  # 0除算対策
        
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50)  # NaNは50で埋める
        
        # 最初の値は50で埋める
        result = np.full(len(prices), 50.0)
        result[1:] = rsi.values
        
        return result
    
    def add_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市場全体の特徴量追加（リークage防止）"""
        logger.info("📈 Adding market-wide features...")
        
        # 各日の市場全体指標を計算
        market_features = []
        
        for date in sorted(df['Date'].unique()):
            day_data = df[df['Date'] == date]
            
            if len(day_data) > 1:
                # 前日比リターンの計算（その日のリターンのみ使用）
                returns = day_data['Returns'].values
                market_return = np.nanmean(returns)
                market_vol = np.nanstd(returns)
                market_breadth = np.nanmean(returns > 0)
                
                # 前日の情報を使用（リークage防止）
                market_features.append({
                    'Date': date,
                    'Market_Return': market_return if not np.isnan(market_return) else 0,
                    'Market_Volatility': market_vol if not np.isnan(market_vol) else 0,
                    'Market_Breadth': market_breadth if not np.isnan(market_breadth) else 0.5
                })
            else:
                market_features.append({
                    'Date': date,
                    'Market_Return': 0,
                    'Market_Volatility': 0,
                    'Market_Breadth': 0.5
                })
        
        market_df = pd.DataFrame(market_features)
        
        # 1日ラグを適用（リークage防止）
        market_df['Market_Return'] = market_df['Market_Return'].shift(1).fillna(0)
        market_df['Market_Volatility'] = market_df['Market_Volatility'].shift(1).fillna(0)
        market_df['Market_Breadth'] = market_df['Market_Breadth'].shift(1).fillna(0.5)
        
        # メインデータフレームとマージ
        df = df.merge(market_df, on='Date', how='left')
        
        # 相対リターンの計算
        df['Relative_Return'] = df['Returns'] - df['Market_Return']
        
        return df
    
    def final_quality_check(self, df: pd.DataFrame) -> dict:
        """最終品質チェック"""
        logger.info("✅ Performing final quality check...")
        
        issues = []
        stats = {}
        
        # 列の存在確認
        if 'Code' not in df.columns:
            raise ValueError("Required column 'Code' not found in dataframe")
        if 'Date' not in df.columns:
            raise ValueError("Required column 'Date' not found in dataframe")
        
        # 基本統計
        stats['total_records'] = len(df)
        stats['unique_stocks'] = df['Code'].nunique()
        stats['date_range'] = {
            'start': df['Date'].min(),
            'end': df['Date'].max(),
            'span_days': (df['Date'].max() - df['Date'].min()).days
        }
        
        # 欠損値チェック
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        missing_rate = total_missing / (len(df) * len(df.columns))
        
        stats['missing_values'] = {
            'total': int(total_missing),
            'rate': float(missing_rate)
        }
        
        if missing_rate > 0.05:  # 5%超の欠損値
            issues.append(f"High missing value rate: {missing_rate:.2%}")
        
        # ターゲット変数の分布
        if 'Binary_Direction' in df.columns:
            target_dist = df['Binary_Direction'].value_counts()
            stats['target_distribution'] = target_dist.to_dict()
            
            minority_ratio = target_dist.min() / target_dist.sum()
            if minority_ratio < 0.3:
                issues.append(f"Severe class imbalance: {minority_ratio:.1%}")
        
        # 無限大・異常値チェック
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col in df.columns]
        
        for col in numeric_cols:
            if col not in ['Date', 'Code']:
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    issues.append(f"Infinite values in {col}: {inf_count}")
                
                # 極端な外れ値
                if col == 'Returns':
                    extreme_returns = (np.abs(df[col]) > 0.5).sum()  # 50%超の日次変動
                    if extreme_returns > 0:
                        issues.append(f"Extreme returns in {col}: {extreme_returns}")
        
        return {
            'status': 'PASS' if not issues else 'WARNING',
            'issues': issues,
            'stats': stats
        }
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> Path:
        """処理済みデータの保存"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"production_features_{timestamp}.parquet"
        
        output_path = self.processed_dir / filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Saved production-ready data to: {output_path}")
        
        return output_path

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Production-ready data preprocessing")
    parser.add_argument("--file-pattern", type=str, default="nikkei225_historical_20*", help="Input file pattern")
    parser.add_argument("--output-filename", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    try:
        preprocessor = ProductionDataPreprocessor()
        
        print("🚀 PRODUCTION-READY DATA PREPROCESSING")
        print("="*60)
        
        # 1. 生データ読み込み・クリーニング
        print("\n📊 Loading and cleaning raw data...")
        df = preprocessor.load_and_clean_raw_data(args.file_pattern)
        
        # 2. 厳格な欠損値処理
        print("\n🔧 Applying strict missing value handling...")
        df = preprocessor.strict_missing_value_handling(df)
        
        # 3. リークfreeな特徴量生成
        print("\n🔒 Generating leak-free features...")
        df = preprocessor.generate_leak_free_features(df)
        
        # 4. 市場全体特徴量追加
        print("\n📈 Adding market-wide features...")
        df = preprocessor.add_market_features(df)
        
        # 5. 最終品質チェック
        print("\n✅ Performing final quality check...")
        quality_report = preprocessor.final_quality_check(df)
        
        # 6. 保存
        print("\n💾 Saving production-ready data...")
        output_path = preprocessor.save_processed_data(df, args.output_filename)
        
        # 結果レポート
        print("\n" + "="*60)
        print("📋 PREPROCESSING RESULTS")
        print("="*60)
        
        stats = quality_report['stats']
        print(f"📄 Total records: {stats['total_records']:,}")
        print(f"📈 Unique stocks: {stats['unique_stocks']}")
        print(f"📅 Date range: {stats['date_range']['start']} to {stats['date_range']['end']}")
        print(f"🔧 Features: {len(df.columns)}")
        print(f"💾 Output file: {output_path.name}")
        
        missing_info = stats['missing_values']
        print(f"❌ Missing values: {missing_info['total']} ({missing_info['rate']:.2%})")
        
        if quality_report['issues']:
            print(f"\n⚠️ Issues found:")
            for issue in quality_report['issues']:
                print(f"  - {issue}")
        else:
            print(f"\n✅ No critical issues found")
        
        print(f"\nQuality Status: {'✅ PASS' if quality_report['status'] == 'PASS' else '⚠️ WARNING'}")
        print("\n✅ Production-ready preprocessing completed!")
        
    except Exception as e:
        logger.error(f"Preprocessing failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())