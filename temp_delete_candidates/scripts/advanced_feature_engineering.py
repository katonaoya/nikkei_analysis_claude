#!/usr/bin/env python3
"""
高度な特徴量エンジニアリング - 精度向上のための追加特徴量
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """高度な特徴量生成"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
    
    def load_base_features(self, filename: str) -> pd.DataFrame:
        """ベース特徴量読み込み"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded base features: {df.shape}")
        return df
    
    def add_advanced_technical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度なテクニカル指標"""
        logger.info("🔧 Adding advanced technical features...")
        
        results = []
        
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code].copy().sort_values('Date')
            
            if len(stock_data) < 50:
                continue
            
            close = stock_data['Close'].values
            high = stock_data['High'].values if 'High' in stock_data.columns else close
            low = stock_data['Low'].values if 'Low' in stock_data.columns else close
            volume = stock_data['Volume'].values if 'Volume' in stock_data.columns else np.ones(len(close))
            
            # 既存データをコピー
            enhanced_data = stock_data.copy()
            
            # 1. ボリンジャーバンド
            ma_20 = pd.Series(close).rolling(20, min_periods=20).mean().values
            std_20 = pd.Series(close).rolling(20, min_periods=20).std().values
            upper_bb = ma_20 + 2 * std_20
            lower_bb = ma_20 - 2 * std_20
            bb_position = (close - lower_bb) / (upper_bb - lower_bb)
            bb_position = np.clip(bb_position, 0, 1)
            enhanced_data['BB_Position'] = bb_position
            
            # 2. MACD
            ema_12 = pd.Series(close).ewm(span=12).mean().values
            ema_26 = pd.Series(close).ewm(span=26).mean().values
            macd = ema_12 - ema_26
            macd_signal = pd.Series(macd).ewm(span=9).mean().values
            enhanced_data['MACD'] = macd
            enhanced_data['MACD_Signal'] = macd_signal
            enhanced_data['MACD_Hist'] = macd - macd_signal
            
            # 3. ストキャスティクス
            lowest_14 = pd.Series(low).rolling(14, min_periods=14).min().values
            highest_14 = pd.Series(high).rolling(14, min_periods=14).max().values
            k_percent = (close - lowest_14) / (highest_14 - lowest_14) * 100
            k_percent = np.nan_to_num(k_percent, nan=50)
            d_percent = pd.Series(k_percent).rolling(3, min_periods=3).mean().values
            enhanced_data['Stoch_K'] = k_percent
            enhanced_data['Stoch_D'] = d_percent
            
            # 4. Williams %R
            williams_r = (highest_14 - close) / (highest_14 - lowest_14) * -100
            williams_r = np.nan_to_num(williams_r, nan=-50)
            enhanced_data['Williams_R'] = williams_r
            
            # 5. CCI (Commodity Channel Index)
            typical_price = (high + low + close) / 3
            sma_tp = pd.Series(typical_price).rolling(20, min_periods=20).mean().values
            mad = pd.Series(typical_price).rolling(20, min_periods=20).apply(
                lambda x: np.mean(np.abs(x - x.mean())), raw=True
            ).values
            cci = (typical_price - sma_tp) / (0.015 * mad)
            cci = np.nan_to_num(cci, nan=0)
            enhanced_data['CCI'] = cci
            
            results.append(enhanced_data)
        
        final_df = pd.concat(results, ignore_index=True)
        logger.info(f"Added technical features: {final_df.shape}")
        return final_df
    
    def add_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """モメンタム特徴量"""
        logger.info("📈 Adding momentum features...")
        
        results = []
        
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code].copy().sort_values('Date')
            
            if len(stock_data) < 30:
                continue
            
            close = stock_data['Close'].values
            enhanced_data = stock_data.copy()
            
            # 1. ROC (Rate of Change)
            for period in [5, 10, 20]:
                roc = (close / np.roll(close, period) - 1) * 100
                roc[:period] = 0
                enhanced_data[f'ROC_{period}'] = roc
            
            # 2. モメンタム
            for period in [5, 10, 20]:
                momentum = close - np.roll(close, period)
                momentum[:period] = 0
                enhanced_data[f'Momentum_{period}'] = momentum
            
            # 3. 価格加速度（2階微分的概念）
            returns = np.diff(np.log(close))
            returns = np.concatenate([[0], returns])
            acceleration = np.diff(returns)
            acceleration = np.concatenate([[0], acceleration])
            enhanced_data['Price_Acceleration'] = acceleration
            
            # 4. トレンド強度
            for period in [10, 20]:
                trend_strength = pd.Series(close).rolling(period, min_periods=period).apply(
                    lambda x: np.corrcoef(np.arange(len(x)), x)[0, 1] if len(x) >= 2 else 0,
                    raw=True
                ).values
                trend_strength = np.nan_to_num(trend_strength, nan=0)
                enhanced_data[f'Trend_Strength_{period}'] = trend_strength
            
            results.append(enhanced_data)
        
        final_df = pd.concat(results, ignore_index=True)
        logger.info(f"Added momentum features: {final_df.shape}")
        return final_df
    
    def add_market_microstructure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """マーケットマイクロストラクチャー特徴量"""
        logger.info("🏛️ Adding market microstructure features...")
        
        results = []
        
        for code in df['Code'].unique():
            stock_data = df[df['Code'] == code].copy().sort_values('Date')
            
            if len(stock_data) < 30:
                continue
            
            enhanced_data = stock_data.copy()
            close = stock_data['Close'].values
            high = stock_data['High'].values if 'High' in stock_data.columns else close
            low = stock_data['Low'].values if 'Low' in stock_data.columns else close
            volume = stock_data['Volume'].values if 'Volume' in stock_data.columns else np.ones(len(close))
            
            # 1. ボリューム重み付け平均価格との乖離
            vwap_5 = pd.Series((high + low + close) / 3 * volume).rolling(5).sum() / pd.Series(volume).rolling(5).sum()
            enhanced_data['Price_vs_VWAP'] = close / vwap_5.values - 1
            
            # 2. イントラデイリターン変動
            if 'Open' in stock_data.columns:
                intraday_return = (close - stock_data['Open'].values) / stock_data['Open'].values
                intraday_vol = pd.Series(intraday_return).rolling(10, min_periods=10).std().values
                enhanced_data['Intraday_Volatility'] = intraday_vol
            
            # 3. ボリュームショック
            vol_ma = pd.Series(volume).rolling(20, min_periods=20).mean().values
            vol_std = pd.Series(volume).rolling(20, min_periods=20).std().values
            vol_shock = (volume - vol_ma) / (vol_std + 1e-8)
            enhanced_data['Volume_Shock'] = vol_shock
            
            # 4. 価格ギャップ
            gap = np.concatenate([[0], np.diff(close) / close[:-1]])
            gap_ma = pd.Series(np.abs(gap)).rolling(10, min_periods=10).mean().values
            enhanced_data['Gap_Magnitude'] = np.abs(gap) / (gap_ma + 1e-8)
            
            results.append(enhanced_data)
        
        final_df = pd.concat(results, ignore_index=True)
        logger.info(f"Added microstructure features: {final_df.shape}")
        return final_df
    
    def add_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """市場レジーム特徴量"""
        logger.info("🌊 Adding market regime features...")
        
        # 市場全体のボラティリティレジーム
        market_returns = df.groupby('Date')['Returns'].mean()
        market_vol_short = market_returns.rolling(10).std()
        market_vol_long = market_returns.rolling(60).std()
        vol_regime = (market_vol_short / market_vol_long).fillna(1)
        
        # 日付をキーとした辞書作成
        vol_regime_dict = vol_regime.to_dict()
        
        df['Market_Vol_Regime'] = df['Date'].map(vol_regime_dict).fillna(1)
        
        # セクター相対パフォーマンス（簡易版）
        sector_returns = df.groupby('Date')['Returns'].mean()
        df['Relative_to_Market'] = df.groupby('Date')['Returns'].transform(lambda x: x - x.mean())
        
        logger.info(f"Added regime features: {df.shape}")
        return df

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Advanced feature engineering")
    parser.add_argument("--input-file", required=True, help="Input features file")
    parser.add_argument("--output-file", required=True, help="Output features file")
    
    args = parser.parse_args()
    
    try:
        engineer = AdvancedFeatureEngineer()
        
        print("📊 Loading base features...")
        df = engineer.load_base_features(args.input_file)
        
        print("🔧 Adding advanced technical features...")
        df = engineer.add_advanced_technical_features(df)
        
        print("📈 Adding momentum features...")
        df = engineer.add_momentum_features(df)
        
        print("🏛️ Adding microstructure features...")
        df = engineer.add_market_microstructure_features(df)
        
        print("🌊 Adding regime features...")
        df = engineer.add_regime_features(df)
        
        # 保存
        output_path = engineer.processed_dir / args.output_file
        df.to_parquet(output_path, index=False)
        
        print(f"\n✅ Enhanced features saved: {df.shape}")
        print(f"📄 Feature count: {len(df.columns)}")
        print(f"💾 Output: {output_path}")
        
        # 新しい特徴量リスト
        new_features = [col for col in df.columns if col not in [
            'Date', 'Code', 'Close', 'High', 'Low', 'Volume', 'Returns',
            'MA_5', 'MA_10', 'MA_20', 'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA20',
            'RSI_14', 'Volatility_20', 'Volume_MA_10', 'Volume_Ratio', 'Price_Position',
            'Return_Lag_1', 'Return_Lag_2', 'Return_Lag_5', 'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_5',
            'Next_Day_Return', 'Binary_Direction', 'Market_Return', 'Market_Volatility', 
            'Market_Breadth', 'Relative_Return'
        ]]
        
        print(f"\n🆕 New features added ({len(new_features)}):")
        for feature in new_features:
            print(f"   - {feature}")
        
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())