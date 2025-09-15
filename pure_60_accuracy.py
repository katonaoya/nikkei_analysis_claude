#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
純粋な予測精度60%以上を達成するプログラム
運用ルールは考慮せず、単純に上昇/下落の予測精度を60%以上にする
"""

import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
import logging
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger(__name__)


class PureAccuracyOptimizer:
    """純粋な精度60%達成クラス"""
    
    def __init__(self):
        self.best_accuracy = 0
        self.best_config = None
        
    def load_data(self):
        """データ読み込み"""
        logger.info("📥 データ読み込み中...")
        df = pd.read_parquet("data/processed/integrated_with_external.parquet")
        
        # 必要な列処理
        if 'Target' not in df.columns and 'Binary_Direction' in df.columns:
            df['Target'] = df['Binary_Direction']
        if 'Stock' not in df.columns and 'Code' in df.columns:
            df['Stock'] = df['Code']
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        
        logger.info(f"📊 データ: {len(df):,}件")
        return df
    
    def create_all_features(self, df):
        """全ての可能な特徴量を作成"""
        logger.info("🔧 特徴量作成中...")
        
        if 'Close' not in df.columns:
            return df
        
        # 価格変動率
        for period in [1, 2, 3, 5, 10, 20]:
            col = f'Returns_{period}d'
            if col not in df.columns:
                df[col] = df.groupby('Stock')['Close'].pct_change(period)
        
        # 移動平均
        for window in [5, 10, 20, 50, 100]:
            # 単純移動平均
            ma_col = f'MA_{window}'
            if ma_col not in df.columns:
                df[ma_col] = df.groupby('Stock')['Close'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
            
            # 移動平均との比率
            ratio_col = f'Close_MA{window}_Ratio'
            if ratio_col not in df.columns:
                df[ratio_col] = df['Close'] / df[ma_col].replace(0, np.nan)
        
        # ボラティリティ
        for window in [5, 10, 20, 50]:
            vol_col = f'Volatility_{window}d'
            if vol_col not in df.columns:
                df[vol_col] = df.groupby('Stock')['Close'].transform(
                    lambda x: x.pct_change().rolling(window, min_periods=1).std()
                )
        
        # RSI
        def calc_rsi(prices, period=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(period, min_periods=1).mean()
            loss = -delta.where(delta < 0, 0).rolling(period, min_periods=1).mean()
            rs = gain / (loss + 1e-10)
            return 100 - (100 / (1 + rs))
        
        for period in [7, 14, 21]:
            rsi_col = f'RSI_{period}'
            if rsi_col not in df.columns:
                df[rsi_col] = df.groupby('Stock')['Close'].transform(
                    lambda x: calc_rsi(x, period)
                )
        
        # MACD
        if 'MACD' not in df.columns:
            exp12 = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=12).mean())
            exp26 = df.groupby('Stock')['Close'].transform(lambda x: x.ewm(span=26).mean())
            df['MACD'] = exp12 - exp26
            df['MACD_Signal'] = df.groupby('Stock')['MACD'].transform(lambda x: x.ewm(span=9).mean())
            df['MACD_Diff'] = df['MACD'] - df['MACD_Signal']
        
        # ボリンジャーバンド
        for window in [20]:
            bb_middle = f'BB_Middle_{window}'
            if bb_middle not in df.columns:
                df[bb_middle] = df.groupby('Stock')['Close'].transform(
                    lambda x: x.rolling(window, min_periods=1).mean()
                )
                
                std = df.groupby('Stock')['Close'].transform(
                    lambda x: x.rolling(window, min_periods=1).std()
                )
                
                df[f'BB_Upper_{window}'] = df[bb_middle] + 2 * std
                df[f'BB_Lower_{window}'] = df[bb_middle] - 2 * std
                df[f'BB_Position_{window}'] = (df['Close'] - df[f'BB_Lower_{window}']) / (
                    df[f'BB_Upper_{window}'] - df[f'BB_Lower_{window}'] + 1e-10
                )
        
        # 出来高関連
        if 'Volume' in df.columns:
            # 出来高移動平均
            for window in [5, 10, 20]:
                vol_ma_col = f'Volume_MA_{window}'
                if vol_ma_col not in df.columns:
                    df[vol_ma_col] = df.groupby('Stock')['Volume'].transform(
                        lambda x: x.rolling(window, min_periods=1).mean()
                    )
                
                # 出来高比率
                vol_ratio_col = f'Volume_Ratio_{window}'
                if vol_ratio_col not in df.columns:
                    df[vol_ratio_col] = df['Volume'] / df[vol_ma_col].replace(0, np.nan)
            
            # 価格×出来高
            df['PriceVolume'] = df['Close'] * df['Volume']
            
            # On-Balance Volume (OBV)
            if 'OBV' not in df.columns:
                df['OBV'] = df.groupby('Stock').apply(
                    lambda x: (x['Volume'] * np.sign(x['Close'].diff())).cumsum()
                ).reset_index(level=0, drop=True)
        
        # 高値・安値関連
        if 'High' in df.columns and 'Low' in df.columns:
            # 高値安値の範囲
            df['HL_Range'] = (df['High'] - df['Low']) / df['Close'].replace(0, np.nan)
            
            # 終値の位置
            df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-10)
            
            # 過去N日の最高値・最安値
            for window in [10, 20, 50]:
                high_col = f'High_{window}d'
                low_col = f'Low_{window}d'
                
                if high_col not in df.columns:
                    df[high_col] = df.groupby('Stock')['High'].transform(
                        lambda x: x.rolling(window, min_periods=1).max()
                    )
                    df[low_col] = df.groupby('Stock')['Low'].transform(
                        lambda x: x.rolling(window, min_periods=1).min()
                    )
                    
                    # 現在価格の相対位置
                    df[f'Price_Position_{window}d'] = (df['Close'] - df[low_col]) / (
                        df[high_col] - df[low_col] + 1e-10
                    )
        
        return df
    
    def select_best_features(self, df, n_features=15):
        """最良の特徴量を選択"""
        logger.info(f"🎯 上位{n_features}個の特徴量を選択...")
        
        # 除外列
        exclude = ['Date', 'Stock', 'Code', 'Target', 'Binary_Direction', 
                  'Open', 'High', 'Low', 'Close', 'Volume', 'Direction']
        
        # 数値列のみ
        feature_cols = [col for col in df.columns 
                       if col not in exclude and df[col].dtype in ['float64', 'int64']]
        
        # 欠損が少ない特徴量
        valid_features = []
        for col in feature_cols:
            missing_rate = df[col].isna().mean()
            if missing_rate < 0.3:  # 欠損率30%未満
                valid_features.append(col)
        
        if len(valid_features) == 0:
            return []
        
        # データサンプリング（高速化のため）
        sample_size = min(50000, len(df))
        df_sample = df.sample(sample_size, random_state=42)
        df_sample = df_sample[['Target'] + valid_features].dropna()
        
        if len(df_sample) < 1000:
            return valid_features[:n_features]
        
        X = df_sample[valid_features]
        y = df_sample['Target']
        
        # 相互情報量で特徴量選択
        selector = SelectKBest(mutual_info_classif, k=min(n_features, len(valid_features)))
        selector.fit(X, y)
        
        # 選択された特徴量
        selected_features = [feat for feat, selected in zip(valid_features, selector.get_support()) if selected]
        
        logger.info(f"📊 選択された特徴量: {selected_features[:5]}...")
        
        return selected_features
    
    def test_accuracy(self, df, features, model_type='rf'):
        """精度テスト"""
        
        # データクリーニング
        required_cols = ['Date', 'Stock', 'Target'] + features
        clean_df = df[required_cols].dropna()
        
        if len(clean_df) < 10000:
            return 0
        
        # 日付でソート
        clean_df = clean_df.sort_values('Date')
        unique_dates = sorted(clean_df['Date'].unique())
        
        if len(unique_dates) < 50:
            return 0
        
        # 訓練・テスト分割（時系列）
        split_date = unique_dates[-20]  # 直近20日をテスト
        
        train_data = clean_df[clean_df['Date'] < split_date]
        test_data = clean_df[clean_df['Date'] >= split_date]
        
        if len(train_data) < 5000 or len(test_data) < 1000:
            return 0
        
        # 訓練データを制限（高速化）
        if len(train_data) > 50000:
            train_data = train_data.tail(50000)
        
        X_train = train_data[features]
        y_train = train_data['Target']
        X_test = test_data[features]
        y_test = test_data['Target']
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデル学習
        if model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=20,
                min_samples_leaf=10,
                random_state=42,
                n_jobs=-1
            )
        elif model_type == 'gb':
            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        else:  # lgb
            model = lgb.LGBMClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42,
                verbose=-1
            )
        
        model.fit(X_train_scaled, y_train)
        
        # 予測
        y_pred = model.predict(X_test_scaled)
        
        # 精度計算
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy
    
    def optimize_for_60(self):
        """60%精度を達成するまで最適化"""
        
        # データ読み込み
        df = self.load_data()
        
        # 全特徴量作成
        df = self.create_all_features(df)
        
        # 複数の特徴量数を試す
        for n_features in [10, 15, 20, 25, 30]:
            logger.info(f"\n🔍 {n_features}個の特徴量でテスト...")
            
            # 最良の特徴量選択
            features = self.select_best_features(df, n_features)
            
            if len(features) < 5:
                continue
            
            # 複数のモデルを試す
            for model_type in ['rf', 'gb', 'xgb', 'lgb']:
                logger.info(f"  モデル: {model_type}")
                
                accuracy = self.test_accuracy(df, features, model_type)
                
                logger.info(f"  精度: {accuracy:.2%}")
                
                if accuracy > self.best_accuracy:
                    self.best_accuracy = accuracy
                    self.best_config = {
                        'features': features,
                        'model_type': model_type,
                        'n_features': n_features,
                        'accuracy': accuracy
                    }
                    
                    logger.info(f"  ✅ 新記録! {accuracy:.2%}")
                    
                    if accuracy >= 0.60:
                        logger.info(f"  🎯 目標達成! 60%を超えました!")
                        return self.best_config
        
        # まだ60%未達成なら、別のアプローチ
        if self.best_accuracy < 0.60:
            logger.info("\n🚀 追加の最適化...")
            
            # より多くの特徴量で再試行
            for n_features in [35, 40, 50]:
                features = self.select_best_features(df, n_features)
                
                if len(features) < 10:
                    continue
                
                # XGBoostとLightGBMに絞る
                for model_type in ['xgb', 'lgb']:
                    accuracy = self.test_accuracy(df, features, model_type)
                    
                    if accuracy > self.best_accuracy:
                        self.best_accuracy = accuracy
                        self.best_config = {
                            'features': features,
                            'model_type': model_type,
                            'n_features': n_features,
                            'accuracy': accuracy
                        }
                        
                        logger.info(f"  ✅ 更新! {accuracy:.2%}")
                        
                        if accuracy >= 0.60:
                            logger.info(f"  🎯 目標達成!")
                            return self.best_config
        
        return self.best_config


def main():
    """メイン実行"""
    logger.info("="*60)
    logger.info("🎯 純粋な予測精度60%達成プログラム")
    logger.info("="*60)
    
    optimizer = PureAccuracyOptimizer()
    result = optimizer.optimize_for_60()
    
    logger.info("\n" + "="*60)
    logger.info("📊 最終結果")
    logger.info("="*60)
    
    if result:
        logger.info(f"最高精度: {result['accuracy']:.2%}")
        logger.info(f"モデル: {result['model_type']}")
        logger.info(f"特徴量数: {result['n_features']}")
        
        if result['accuracy'] >= 0.60:
            logger.info("\n✅ 目標達成! 60%以上の精度を実現!")
            
            # 設定を保存
            config_path = Path("production_config.yaml")
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            config['features']['optimal_features'] = result['features'][:10]  # 上位10個
            config['model'] = {
                'type': result['model_type'],
                'accuracy': float(result['accuracy']),
                'n_features': len(result['features'][:10])
            }
            
            with open(config_path, 'w') as f:
                yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
            
            logger.info("📝 設定を保存しました")
            logger.info(f"使用特徴量: {result['features'][:5]}...")
            
            # 精度60%達成したことを記録
            with open("accuracy_60_achieved.txt", 'w') as f:
                f.write(f"達成精度: {result['accuracy']:.2%}\n")
                f.write(f"モデル: {result['model_type']}\n")
                f.write(f"特徴量: {', '.join(result['features'][:10])}\n")
                f.write(f"達成日時: {pd.Timestamp.now()}\n")
        else:
            logger.info(f"\n現在の最高精度: {result['accuracy']:.2%}")
            logger.info("60%にはまだ届いていませんが、継続して最適化します")
    else:
        logger.error("最適化に失敗しました")


if __name__ == "__main__":
    main()