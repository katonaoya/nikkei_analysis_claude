#!/usr/bin/env python3
"""
高精度を目指した拡張特徴量生成スクリプト（簡略版）
TALibなしで実装した高度な特徴量エンジニアリング
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class SimpleEnhancedFeatureGenerator:
    """TALibなしの高精度特徴量生成器"""
    
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
        
        # 各データフレームのカラム確認と統一
        unified_dfs = []
        for df_file in dfs:
            # カラム名の統一
            df_file = self._standardize_columns(df_file)
            unified_dfs.append(df_file)
        
        df = pd.concat(unified_dfs, ignore_index=True)
        
        # 重複除去
        df = df.drop_duplicates(subset=['Date', 'Code'], keep='last')
        df = df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        logger.info(f"Loaded {len(df)} records for {df['Code'].nunique()} stocks")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """カラム名の標準化"""
        df = df.copy()
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
            df = df.drop('date', axis=1)
        
        rename_map = {
            'close': 'Close', 'open': 'Open', 'high': 'High',
            'low': 'Low', 'volume': 'Volume', 'code': 'Code'
        }
        
        df = df.rename(columns=rename_map)
        return df
    
    def _calculate_rsi(self, prices: np.array, period: int = 14) -> np.array:
        """RSI計算（TALibなし）"""
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = pd.Series(gains).rolling(period).mean()
        avg_loss = pd.Series(losses).rolling(period).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return np.concatenate([[50], rsi.values])  # 最初の値は50で埋める
    
    def _calculate_stochastic(self, high: np.array, low: np.array, close: np.array, period: int = 14) -> tuple:
        """ストキャスティクス計算"""
        high_series = pd.Series(high)
        low_series = pd.Series(low)
        close_series = pd.Series(close)
        
        lowest_low = low_series.rolling(period).min()
        highest_high = high_series.rolling(period).max()
        
        k_percent = 100 * (close_series - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(3).mean()
        
        return k_percent.values, d_percent.values
    
    def _calculate_macd(self, prices: np.array, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """MACD計算"""
        prices_series = pd.Series(prices)
        
        ema_fast = prices_series.ewm(span=fast).mean()
        ema_slow = prices_series.ewm(span=slow).mean()
        
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return macd_line.values, signal_line.values, histogram.values
    
    def _calculate_bollinger_bands(self, prices: np.array, period: int = 20, std_dev: float = 2) -> tuple:
        """ボリンジャーバンド計算"""
        prices_series = pd.Series(prices)
        
        sma = prices_series.rolling(period).mean()
        std = prices_series.rolling(period).std()
        
        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)
        
        return upper_band.values, sma.values, lower_band.values
    
    def generate_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高精度特徴量生成"""
        logger.info("Starting enhanced feature generation...")
        
        results = []
        
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code].copy().sort_values('Date')
            
            if len(stock_data) < 100:
                continue
            
            close = stock_data['Close'].values
            high = stock_data['High'].values if 'High' in stock_data.columns else close
            low = stock_data['Low'].values if 'Low' in stock_data.columns else close
            volume = stock_data['Volume'].values if 'Volume' in stock_data.columns else np.ones(len(close))
            
            # 基本特徴量
            features = {
                'Date': stock_data['Date'],
                'Code': code,
                'Close': close,
                'High': high,
                'Low': low,
                'Volume': volume
            }
            
            # リターン
            returns = np.concatenate([[0], np.diff(np.log(close))])
            features['Returns'] = returns
            
            # 移動平均（多期間）
            for period in [5, 10, 20, 50, 100]:
                ma = pd.Series(close).rolling(period).mean().values
                features[f'MA_{period}'] = ma
                features[f'Price_vs_MA{period}'] = (close / ma) - 1
                features[f'MA{period}_Slope'] = np.gradient(ma)
            
            # ボリンジャーバンド
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close)
            features['BB_Upper'] = bb_upper
            features['BB_Middle'] = bb_middle
            features['BB_Lower'] = bb_lower
            features['BB_Width'] = (bb_upper - bb_lower) / bb_middle
            features['BB_Position'] = (close - bb_lower) / (bb_upper - bb_lower)
            
            # RSI（複数期間）
            for period in [14, 21, 30]:
                rsi = self._calculate_rsi(close, period)
                features[f'RSI_{period}'] = rsi
                features[f'RSI_{period}_Oversold'] = (rsi < 30).astype(int)
                features[f'RSI_{period}_Overbought'] = (rsi > 70).astype(int)
            
            # MACD
            macd, signal, histogram = self._calculate_macd(close)
            features['MACD'] = macd
            features['MACD_Signal'] = signal
            features['MACD_Histogram'] = histogram
            features['MACD_Bullish'] = (macd > signal).astype(int)
            
            # ストキャスティクス
            stoch_k, stoch_d = self._calculate_stochastic(high, low, close)
            features['Stoch_K'] = stoch_k
            features['Stoch_D'] = stoch_d
            features['Stoch_Oversold'] = ((stoch_k < 20) & (stoch_d < 20)).astype(int)
            features['Stoch_Overbought'] = ((stoch_k > 80) & (stoch_d > 80)).astype(int)
            
            # ボラティリティ（複数期間）
            for period in [10, 20, 30, 50]:
                vol = pd.Series(returns).rolling(period).std().values
                features[f'Volatility_{period}'] = vol
            
            # 価格レンジ
            features['Daily_Range'] = (high - low) / close
            features['True_Range'] = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            features['ATR_14'] = pd.Series(features['True_Range']).rolling(14).mean().values
            
            # ボリューム指標
            for period in [10, 20, 50]:
                vol_ma = pd.Series(volume).rolling(period).mean().values
                features[f'Volume_MA_{period}'] = vol_ma
                features[f'Volume_Ratio_{period}'] = volume / vol_ma
            
            # 価格の位置
            for period in [20, 50]:
                high_period = pd.Series(high).rolling(period).max().values
                low_period = pd.Series(low).rolling(period).min().values
                features[f'Price_Position_{period}'] = (close - low_period) / (high_period - low_period)
            
            # ラグ特徴量
            for lag in [1, 2, 3, 5, 10]:
                features[f'Return_Lag_{lag}'] = np.roll(returns, lag)
                features[f'Close_Lag_{lag}'] = np.roll(close, lag)
                features[f'Volume_Lag_{lag}'] = np.roll(volume, lag)
            
            # 移動平均の傾き
            for period in [5, 10, 20]:
                ma = pd.Series(close).rolling(period).mean().values
                slope = np.gradient(ma)
                features[f'MA{period}_Slope'] = slope
                features[f'MA{period}_Rising'] = (slope > 0).astype(int)
            
            # モメンタム
            for period in [5, 10, 20]:
                momentum = close / np.roll(close, period) - 1
                features[f'Momentum_{period}'] = momentum
            
            # 相対強度
            features['High_Low_Ratio'] = high / low
            features['Close_Position'] = (close - low) / (high - low)
            
            # 複合指標
            features['Bull_Bear_Power'] = (high - pd.Series(close).rolling(13).mean()) - (pd.Series(close).rolling(13).mean() - low)
            
            # マクロ特徴量（簡略版）
            dates = stock_data['Date']
            market_features = []
            for date in dates:
                day_data = df[df['Date'] == date]
                if len(day_data) > 1:
                    market_return = day_data['Close'].pct_change().mean()
                    market_vol = day_data['Close'].pct_change().std() 
                    market_breadth = (day_data['Close'].pct_change() > 0).mean()
                else:
                    market_return = market_vol = market_breadth = 0
                    
                market_features.append([
                    market_return if not pd.isna(market_return) else 0,
                    market_vol if not pd.isna(market_vol) else 0,
                    market_breadth if not pd.isna(market_breadth) else 0.5
                ])
            
            market_features = np.array(market_features)
            features['Market_Return'] = market_features[:, 0]
            features['Market_Volatility'] = market_features[:, 1]
            features['Market_Breadth'] = market_features[:, 2]
            features['Relative_Return'] = returns - market_features[:, 0]
            
            # ターゲット変数
            next_return = np.roll(returns, -1)
            features['Next_Day_Return'] = next_return
            features['Binary_Direction'] = (next_return > 0).astype(int)
            features['Strong_Up'] = (next_return > 0.02).astype(int)
            features['Strong_Down'] = (next_return < -0.02).astype(int)
            
            # データフレーム作成
            stock_df = pd.DataFrame({k: v for k, v in features.items() if len(v) == len(close)})
            results.append(stock_df)
        
        final_df = pd.concat(results, ignore_index=True)
        
        # 無限大・NaNの処理
        final_df = final_df.replace([np.inf, -np.inf], np.nan)
        
        # 前方補完してから0埋め
        numeric_cols = final_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col not in ['Date', 'Code']:
                final_df[col] = final_df.groupby('Code')[col].fillna(method='ffill')
        final_df = final_df.fillna(0)
        
        logger.info("Enhanced feature generation completed")
        return final_df
    
    def save_features(self, df: pd.DataFrame, output_filename: str = None) -> Path:
        """特徴量の保存"""
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"enhanced_features_{timestamp}.parquet"
        
        output_path = self.processed_dir / output_filename
        df.to_parquet(output_path, index=False)
        logger.info(f"Enhanced features saved to: {output_path}")
        
        return output_path

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Simple enhanced feature generation")
    parser.add_argument("--file-pattern", type=str, default="nikkei225_historical_20*", help="File pattern")
    parser.add_argument("--output-filename", type=str, help="Output filename")
    
    args = parser.parse_args()
    
    try:
        generator = SimpleEnhancedFeatureGenerator()
        
        print("📊 Loading stock data...")
        df = generator.load_stock_data(args.file_pattern)
        
        print("🔧 Generating enhanced features...")
        features_df = generator.generate_enhanced_features(df)
        
        print("💾 Saving enhanced features...")
        output_path = generator.save_features(features_df, args.output_filename)
        
        print("\n" + "="*60)
        print("🚀 ENHANCED FEATURE GENERATION RESULTS")
        print("="*60)
        print(f"📄 Total records: {len(features_df):,}")
        print(f"📈 Unique stocks: {features_df['Code'].nunique()}")
        print(f"🔧 Features generated: {len(features_df.columns)}")
        print(f"📅 Date range: {features_df['Date'].min()} to {features_df['Date'].max()}")
        print(f"💾 Output file: {output_path.name}")
        
        missing_count = features_df.isnull().sum().sum()
        if missing_count > 0:
            print(f"⚠️  Missing values: {missing_count}")
        else:
            print("✅ No missing values")
        
        print("\n✅ Enhanced feature generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Enhanced feature generation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())