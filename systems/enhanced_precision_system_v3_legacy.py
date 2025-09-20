#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Precision System V3 Legacy (性能比較用)
メモリ最適化前の性能を確認するための検証版
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime, timedelta
import logging
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegacyPerformanceTest:
    """旧版性能テスト（小データセットでの比較検証）"""
    
    def __init__(self):
        self.output_dir = Path("models/legacy_test")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_and_prepare_test_data(self) -> pd.DataFrame:
        """テスト用データ読み込み（最小限）"""
        logger.info("🔍 テスト用データ準備中...")
        
        # 最新のデータファイルを使用
        stock_file = "data/processed/nikkei225_complete_225stocks_20250915_200849.parquet"
        external_file = "data/processed/enhanced_integrated_data.parquet"
        
        # データ読み込み
        stock_df = pd.read_parquet(stock_file)
        external_df = pd.read_parquet(external_file)
        
        # 小規模テスト用に制限（メモリ使用量を意図的に制御）
        stock_df = stock_df.head(50000)  # 5万件
        external_df = external_df.head(10000)  # 1万件
        
        logger.info(f"テスト用株価データ: {len(stock_df):,}件")
        logger.info(f"テスト用外部データ: {len(external_df):,}件")
        
        return stock_df, external_df
    
    def create_legacy_features(self, stock_df: pd.DataFrame, external_df: pd.DataFrame) -> pd.DataFrame:
        """旧版の特徴量作成（全特徴量、全期間）"""
        logger.info("🔧 Legacy特徴量作成開始...")
        
        # 全期間のデータを使用（メモリ許可範囲内で）
        enhanced_df = stock_df.copy()
        
        # 全銘柄に対してフル特徴量作成
        for code in enhanced_df['Code'].unique():
            mask = enhanced_df['Code'] == code
            code_data = enhanced_df[mask].copy().sort_values('Date')
            
            if len(code_data) < 20:
                continue
                
            # 全特徴量（旧版仕様）
            code_data['Returns'] = code_data['Close'].pct_change()
            code_data['Log_Returns'] = np.log(code_data['Close'] / code_data['Close'].shift(1))
            code_data['High_Low_Ratio'] = code_data['High'] / code_data['Low']
            
            # 移動平均（全期間）
            for window in [5, 10, 20, 50]:
                code_data[f'MA_{window}'] = code_data['Close'].rolling(window).mean()
                code_data[f'MA_{window}_ratio'] = code_data['Close'] / code_data[f'MA_{window}']
            
            # ボラティリティ（全期間）
            for window in [5, 10, 20]:
                code_data[f'Volatility_{window}'] = code_data['Returns'].rolling(window).std()
            
            # RSI（複数期間）
            for window in [14, 21, 30]:
                delta = code_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / loss
                code_data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # ボリンジャーバンド（複数期間）
            for window in [10, 20]:
                rolling_mean = code_data['Close'].rolling(window).mean()
                rolling_std = code_data['Close'].rolling(window).std()
                code_data[f'BB_{window}_upper'] = rolling_mean + (rolling_std * 2)
                code_data[f'BB_{window}_lower'] = rolling_mean - (rolling_std * 2)
                code_data[f'BB_{window}_ratio'] = (code_data['Close'] - code_data[f'BB_{window}_lower']) / (code_data[f'BB_{window}_upper'] - code_data[f'BB_{window}_lower'])
            
            # MACD
            exp1 = code_data['Close'].ewm(span=12).mean()
            exp2 = code_data['Close'].ewm(span=26).mean()
            code_data['MACD'] = exp1 - exp2
            code_data['MACD_signal'] = code_data['MACD'].ewm(span=9).mean()
            code_data['MACD_histogram'] = code_data['MACD'] - code_data['MACD_signal']
            
            # ボリューム特徴量（複数期間）
            for window in [10, 20, 50]:
                code_data[f'Volume_MA_{window}'] = code_data['Volume'].rolling(window).mean()
                code_data[f'Volume_ratio_{window}'] = code_data['Volume'] / code_data[f'Volume_MA_{window}']
            
            enhanced_df.loc[mask] = code_data
        
        # ターゲット変数作成
        enhanced_df['Target'] = 0
        for code in enhanced_df['Code'].unique():
            mask = enhanced_df['Code'] == code
            code_data = enhanced_df[mask].copy()
            next_high = code_data['High'].shift(-1)
            prev_close = code_data['Close'].shift(1)
            enhanced_df.loc[mask, 'Target'] = (next_high / prev_close > 1.01).astype(int)
        
        # 欠損値処理
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.dropna(subset=['Close', 'Date', 'Code', 'Target'])
        
        logger.info(f"Legacy特徴量作成完了: {len(enhanced_df):,}件, {len(enhanced_df.columns)}カラム")
        return enhanced_df
    
    def test_legacy_performance(self, df: pd.DataFrame) -> dict:
        """旧版性能テスト"""
        logger.info("🎯 Legacy性能テスト開始...")
        
        # 特徴量準備（全特徴量使用）
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Code', 'Target'] and 
                       df[col].dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(method='ffill').fillna(0)
        y = df['Target']
        
        # 時系列分割
        split_idx = int(len(df) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # 特徴量選択（旧版は多くの特徴量使用）
        selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # モデル学習（旧版パラメータ）
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=300,  # 多い
            max_depth=8,       # 深い
            learning_rate=0.03, # 低い
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # 予測・評価
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'feature_count': len(feature_cols),
            'selected_features': X_train_selected.shape[1],
            'train_size': len(X_train),
            'test_size': len(X_test),
            'data_size': len(df)
        }
        
        logger.info(f"🎯 Legacy性能結果:")
        logger.info(f"  精度: {accuracy:.4f}")
        logger.info(f"  適合率: {precision:.4f}")
        logger.info(f"  再現率: {recall:.4f}")
        logger.info(f"  F1スコア: {f1:.4f}")
        logger.info(f"  全特徴量数: {len(feature_cols)}")
        logger.info(f"  選択特徴量数: {X_train_selected.shape[1]}")
        
        return result

# テスト実行
if __name__ == "__main__":
    logger.info("🔬 Legacy vs 最適化版 性能比較テスト開始")
    
    tester = LegacyPerformanceTest()
    
    # データ準備
    stock_df, external_df = tester.load_and_prepare_test_data()
    
    # Legacy特徴量作成
    legacy_df = tester.create_legacy_features(stock_df, external_df)
    
    # Legacy性能テスト
    legacy_result = tester.test_legacy_performance(legacy_df)
    
    logger.info("🎉 Legacy性能テスト完了")
    logger.info(f"比較用データセット: {legacy_result['data_size']:,}件で検証")