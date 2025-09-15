#!/usr/bin/env python3
"""
高精度を目指した拡張特徴量生成スクリプト
70%台の精度を目標とした高度な特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from loguru import logger
import talib
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureGenerator:
    """高精度を目指した拡張特徴量生成器"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.processed_dir.mkdir(parents=True, exist_ok=True)
    
    def load_stock_data(self, file_pattern: str = "*nikkei225_historical*") -> pd.DataFrame:
        """全データファイルの読み込み"""
        data_files = list(self.raw_dir.glob(f"{file_pattern}.parquet"))
        
        if not data_files:
            raise FileNotFoundError(f"No data files found matching pattern: {file_pattern}")
        
        logger.info(f"Found {len(data_files)} data files")
        
        # 全ファイルを結合
        dfs = []
        for file_path in sorted(data_files):
            logger.info(f"Loading data from: {file_path.name}")
            df_file = pd.read_parquet(file_path)
            dfs.append(df_file)
        
        df = pd.concat(dfs, ignore_index=True)
        df = self._standardize_columns(df)
        df = df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records for {df['Code'].nunique()} stocks")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """カラム名の標準化"""
        df = df.copy()
        
        # 日付変換
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
        
        # カラム名の統一
        rename_map = {
            'close': 'Close',
            'open': 'Open', 
            'high': 'High',
            'low': 'Low',
            'volume': 'Volume',
            'code': 'Code'
        }
        
        df = df.rename(columns=rename_map)
        return df
    
    def generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高精度を目指した拡張特徴量生成"""
        logger.info("Starting enhanced feature generation...")
        
        results = []
        
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code].copy().sort_values('Date')
            
            if len(stock_data) < 100:  # 最低100日のデータが必要
                continue
            
            # 基本特徴量
            features_dict = self._generate_basic_features(stock_data)
            
            # 高度なテクニカル指標
            features_dict.update(self._generate_advanced_technical_features(stock_data))
            
            # マクロ経済特徴量
            features_dict.update(self._generate_macro_features(stock_data, df))
            
            # 相対パフォーマンス特徴量
            features_dict.update(self._generate_relative_features(stock_data, df))
            
            # 時系列ラグ特徴量
            features_dict.update(self._generate_lag_features(stock_data))
            
            # ターゲット変数
            features_dict.update(self._generate_targets(stock_data))
            
            # データフレームに変換
            stock_features = pd.DataFrame(features_dict)
            stock_features['Code'] = code
            
            results.append(stock_features)
        
        # 全銘柄を結合
        final_df = pd.concat(results, ignore_index=True)
        
        logger.info("Enhanced feature generation completed")
        return final_df
    
    def _generate_basic_features(self, stock_data: pd.DataFrame) -> dict:
        """基本特徴量"""
        close = stock_data['Close'].values
        high = stock_data['High'].values
        low = stock_data['Low'].values
        volume = stock_data['Volume'].values if 'Volume' in stock_data.columns else np.ones(len(close))
        
        return {
            'Date': stock_data['Date'],
            'Close': close,
            'High': high,
            'Low': low,
            'Volume': volume,
            'Returns': np.concatenate([[0], np.diff(np.log(close))]),
        }
    
    def _generate_advanced_technical_features(self, stock_data: pd.DataFrame) -> dict:
        """高度なテクニカル指標"""
        close = stock_data['Close'].values
        high = stock_data['High'].values
        low = stock_data['Low'].values
        volume = stock_data['Volume'].values if 'Volume' in stock_data.columns else np.ones(len(close))
        
        features = {}
        
        # 移動平均（複数期間）
        for period in [5, 10, 20, 50, 100]:
            ma = pd.Series(close).rolling(period).mean()
            features[f'MA_{period}'] = ma.values
            features[f'Price_vs_MA{period}'] = close / ma.values - 1
        
        # ボリンジャーバンド
        ma20 = pd.Series(close).rolling(20).mean()
        std20 = pd.Series(close).rolling(20).std()
        features['BB_Upper'] = (ma20 + 2 * std20).values
        features['BB_Lower'] = (ma20 - 2 * std20).values
        features['BB_Position'] = (close - (ma20 - 2 * std20).values) / (4 * std20.values)
        
        # RSI（複数期間）
        for period in [14, 21, 30]:
            try:
                rsi = talib.RSI(close.astype(np.float64), timeperiod=period)
                features[f'RSI_{period}'] = rsi
            except:
                features[f'RSI_{period}'] = np.full(len(close), 50.0)
        
        # MACD
        try:
            macd, macdsignal, macdhist = talib.MACD(close.astype(np.float64))
            features['MACD'] = macd
            features['MACD_Signal'] = macdsignal
            features['MACD_Hist'] = macdhist
        except:
            features['MACD'] = np.zeros(len(close))
            features['MACD_Signal'] = np.zeros(len(close))
            features['MACD_Hist'] = np.zeros(len(close))
        
        # ストキャスティクス
        try:
            slowk, slowd = talib.STOCH(high.astype(np.float64), low.astype(np.float64), close.astype(np.float64))
            features['Stoch_K'] = slowk
            features['Stoch_D'] = slowd
        except:
            features['Stoch_K'] = np.full(len(close), 50.0)
            features['Stoch_D'] = np.full(len(close), 50.0)
        
        # ATR（Average True Range）
        try:
            atr = talib.ATR(high.astype(np.float64), low.astype(np.float64), close.astype(np.float64))
            features['ATR'] = atr
        except:
            features['ATR'] = np.zeros(len(close))
        
        # ボラティリティ（複数期間）
        for period in [10, 20, 50]:
            vol = pd.Series(close).pct_change().rolling(period).std()
            features[f'Volatility_{period}'] = vol.values
        
        # ボリューム指標
        vol_ma = pd.Series(volume).rolling(20).mean()
        features['Volume_MA_20'] = vol_ma.values
        features['Volume_Ratio'] = volume / vol_ma.values
        
        # 価格レンジ
        features['Daily_Range'] = (high - low) / close
        features['Close_Position'] = (close - low) / (high - low)
        
        return features
    
    def _generate_macro_features(self, stock_data: pd.DataFrame, all_data: pd.DataFrame) -> dict:
        """マクロ経済特徴量"""
        features = {}
        
        # 日次の市場全体指標
        dates = stock_data['Date']
        
        market_features = []
        for date in dates:
            day_data = all_data[all_data['Date'] == date]
            if len(day_data) > 0:
                market_return = day_data['Close'].pct_change().mean()
                market_vol = day_data['Close'].pct_change().std()
                market_breadth = (day_data['Close'].pct_change() > 0).mean()
                
                market_features.append({
                    'Market_Return': market_return if not pd.isna(market_return) else 0,
                    'Market_Volatility': market_vol if not pd.isna(market_vol) else 0,
                    'Market_Breadth': market_breadth if not pd.isna(market_breadth) else 0.5
                })
            else:
                market_features.append({
                    'Market_Return': 0,
                    'Market_Volatility': 0,
                    'Market_Breadth': 0.5
                })
        
        for key in ['Market_Return', 'Market_Volatility', 'Market_Breadth']:
            features[key] = [f[key] for f in market_features]
        
        return features
    
    def _generate_relative_features(self, stock_data: pd.DataFrame, all_data: pd.DataFrame) -> dict:
        """相対パフォーマンス特徴量"""
        features = {}
        
        close = stock_data['Close'].values
        returns = np.concatenate([[0], np.diff(np.log(close))])
        
        dates = stock_data['Date']
        relative_returns = []
        relative_rank = []
        
        for i, date in enumerate(dates):
            day_data = all_data[all_data['Date'] == date]
            if len(day_data) > 1:
                day_returns = day_data['Close'].pct_change()
                market_return = day_returns.mean()
                
                rel_return = returns[i] - market_return if not pd.isna(market_return) else 0
                
                # 相対ランキング（パーセンタイル）
                stock_return = returns[i]
                rank = (day_returns < stock_return).mean() if not pd.isna(stock_return) else 0.5
                
                relative_returns.append(rel_return)
                relative_rank.append(rank)
            else:
                relative_returns.append(0)
                relative_rank.append(0.5)
        
        features['Relative_Return'] = relative_returns
        features['Market_Rank'] = relative_rank
        
        return features
    
    def _generate_lag_features(self, stock_data: pd.DataFrame) -> dict:
        """ラグ特徴量（過去の情報）"""
        features = {}
        
        close = stock_data['Close'].values
        returns = np.concatenate([[0], np.diff(np.log(close))])
        
        # 過去の収益率
        for lag in [1, 2, 3, 5, 10]:
            lag_returns = np.concatenate([np.zeros(lag), returns[:-lag]])
            features[f'Return_Lag_{lag}'] = lag_returns
        
        # 過去の移動平均からの乖離
        ma20 = pd.Series(close).rolling(20).mean().values
        price_vs_ma = (close / ma20 - 1)
        for lag in [1, 2, 5]:
            lag_feature = np.concatenate([np.zeros(lag), price_vs_ma[:-lag]])
            features[f'MA_Deviation_Lag_{lag}'] = lag_feature
        
        # 過去のボラティリティ
        vol = pd.Series(returns).rolling(20).std().values
        for lag in [1, 5]:
            lag_vol = np.concatenate([np.zeros(lag), vol[:-lag]])
            features[f'Volatility_Lag_{lag}'] = lag_vol
        
        return features
    
    def _generate_targets(self, stock_data: pd.DataFrame) -> dict:
        """ターゲット変数"""
        close = stock_data['Close'].values
        
        # 次日リターン
        next_day_return = np.concatenate([np.diff(np.log(close)), [0]])
        
        # バイナリ方向
        binary_direction = (next_day_return > 0).astype(int)
        
        # 強い上昇/下降
        strong_threshold = 0.02  # 2%
        strong_up = (next_day_return > strong_threshold).astype(int)
        strong_down = (next_day_return < -strong_threshold).astype(int)
        
        return {
            'Next_Day_Return': next_day_return,
            'Binary_Direction': binary_direction,
            'Strong_Up': strong_up,
            'Strong_Down': strong_down,
            'Return_Direction': np.where(next_day_return > 0, 1, 0)
        }
    
    def save_features(self, df: pd.DataFrame, output_filename: str = None) -> Path:
        """特徴量の保存"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"enhanced_features_{timestamp}.parquet"
        
        output_path = self.processed_dir / output_filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Enhanced features saved to: {output_path}")
        
        return output_path
    
    def create_summary(self, df: pd.DataFrame) -> dict:
        """特徴量サマリー作成"""
        return {
            'total_records': len(df),
            'unique_stocks': df['Code'].nunique(),
            'feature_count': len(df.columns),
            'date_range': {
                'start': df['Date'].min(),
                'end': df['Date'].max()
            },
            'missing_values': df.isnull().sum().sum(),
            'features': list(df.columns)
        }

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Enhanced feature generation for high accuracy")
    parser.add_argument(
        "--file-pattern",
        type=str,
        default="nikkei225_historical_20*",
        help="Pattern to match input files"
    )
    parser.add_argument(
        "--output-filename", 
        type=str,
        help="Output filename"
    )
    
    args = parser.parse_args()
    
    try:
        generator = EnhancedFeatureGenerator()
        
        # データ読み込み
        print("📊 Loading stock data...")
        df = generator.load_stock_data(args.file_pattern)
        
        # 拡張特徴量生成
        print("🔧 Generating enhanced features...")
        features_df = generator.generate_enhanced_features(df)
        
        # 保存
        print("💾 Saving enhanced features...")
        output_path = generator.save_features(features_df, args.output_filename)
        
        # サマリー
        summary = generator.create_summary(features_df)
        
        # 結果表示
        print("\n" + "="*60)
        print("🚀 ENHANCED FEATURE GENERATION RESULTS")
        print("="*60)
        print(f"📄 Total records: {summary['total_records']:,}")
        print(f"📈 Unique stocks: {summary['unique_stocks']}")
        print(f"🔧 Features generated: {summary['feature_count']}")
        print(f"📅 Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
        print(f"💾 Output file: {output_path.name}")
        
        if summary['missing_values'] > 0:
            print(f"⚠️  Missing values: {summary['missing_values']}")
        else:
            print("✅ No missing values")
        
        print(f"\n📋 Enhanced features count: {len([f for f in summary['features'] if f not in ['Date', 'Code', 'Close', 'High', 'Low', 'Volume']])}")
        
        # サンプルデータ表示
        sample_cols = ['Date', 'Code', 'Close', 'MA_20', 'RSI_14', 'MACD', 'Next_Day_Return', 'Binary_Direction']
        available_cols = [col for col in sample_cols if col in features_df.columns]
        print(f"\n🎯 Sample data preview:")
        print(features_df[available_cols].head(10).to_string(index=False))
        
        print("\n✅ Enhanced feature generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Enhanced feature generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())