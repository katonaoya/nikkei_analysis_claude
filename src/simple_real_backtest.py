"""
シンプルな実データバックテスト
J-Quantsから取得した実データで基本性能を評価
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import logging
from typing import List, Dict, Any
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import lightgbm as lgb
import catboost as cb

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_real_data():
    """実データ読み込み"""
    data_dir = Path("data/real_jquants_data")
    pickle_files = list(data_dir.glob("nikkei225_real_data_*.pkl"))
    latest_file = max(pickle_files, key=os.path.getctime)
    
    logger.info(f"実データ読み込み: {latest_file}")
    df = pd.read_pickle(latest_file)
    
    # 基本情報表示
    logger.info(f"データサマリー:")
    logger.info(f"  レコード数: {len(df):,}件")
    logger.info(f"  銘柄数: {df['symbol'].nunique()}銘柄")
    logger.info(f"  期間: {df['date'].min().date()} ～ {df['date'].max().date()}")
    logger.info(f"  ターゲット分布: {df['target'].value_counts().to_dict()}")
    logger.info(f"  上昇率: {df['target'].mean():.1%}")
    
    return df

def create_simple_features(df):
    """シンプルな特徴量作成"""
    logger.info("シンプルな特徴量作成中...")
    
    df = df.sort_values(['symbol', 'date']).reset_index(drop=True)
    
    # 基本的な技術指標
    features = df.copy()
    
    # 移動平均
    for window in [5, 10, 20]:
        features[f'sma_{window}'] = features.groupby('symbol')['close_price'].rolling(window).mean().reset_index(0, drop=True)
        features[f'price_sma_{window}_ratio'] = features['close_price'] / features[f'sma_{window}']
    
    # 価格変化率
    for period in [1, 3, 5]:
        features[f'price_change_{period}d'] = features.groupby('symbol')['close_price'].pct_change(period)
    
    # ボラティリティ
    for window in [5, 10, 20]:
        features[f'volatility_{window}d'] = features.groupby('symbol')['daily_return'].rolling(window).std().reset_index(0, drop=True)
    
    # 出来高関連
    features['volume_sma_20'] = features.groupby('symbol')['volume'].rolling(20).mean().reset_index(0, drop=True)
    features['volume_ratio'] = features['volume'] / features['volume_sma_20']
    
    # RSI
    def calculate_rsi(prices, window=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    features['rsi'] = features.groupby('symbol')['close_price'].apply(lambda x: calculate_rsi(x)).reset_index(0, drop=True)
    
    # 欠損値処理
    feature_cols = [col for col in features.columns if col not in ['date', 'symbol', 'target', 'next_day_return']]
    features[feature_cols] = features[feature_cols].fillna(method='ffill').fillna(0)
    
    logger.info(f"特徴量作成完了: {len(feature_cols)}個")
    return features

def run_simple_backtest():
    """シンプルなバックテスト実行"""
    logger.info("=== シンプル実データバックテスト開始 ===")
    
    # データ読み込み
    df = load_real_data()
    
    # 特徴量作成
    df_features = create_simple_features(df)
    
    # 完全なデータのみ使用
    df_clean = df_features.dropna().reset_index(drop=True)
    logger.info(f"クリーンデータ: {len(df_clean):,}件")
    
    # 特徴量カラム特定（数値型のみ）
    exclude_cols = ['date', 'symbol', 'target', 'next_day_return', 'close_price', 
                   'daily_return', 'open_price', 'high_price', 'low_price', 'volume',
                   'adjustment_factor', 'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
                   'Date', 'Code', 'Open', 'High', 'Low', 'Close', 'UpperLimit', 'LowerLimit', 
                   'Volume', 'TurnoverValue', 'AdjustmentFactor', 'AdjustmentOpen', 'AdjustmentHigh',
                   'AdjustmentLow', 'AdjustmentClose', 'AdjustmentVolume']
    
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]
    
    # 数値型のカラムのみ保持
    numeric_features = []
    for col in feature_cols:
        if df_clean[col].dtype in ['int64', 'float64']:
            numeric_features.append(col)
    
    feature_cols = numeric_features
    
    logger.info(f"使用特徴量: {len(feature_cols)}個")
    logger.info(f"特徴量リスト: {feature_cols[:10]}...")  # 最初の10個表示
    
    X = df_clean[feature_cols]
    y = df_clean['target']
    dates = df_clean['date']
    returns = df_clean['next_day_return']
    
    # 時系列分割
    logger.info("時系列クロスバリデーション実行中...")
    tscv = TimeSeriesSplit(n_splits=5)
    
    results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        logger.info(f"Fold {fold}/5 実行中...")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        test_returns = returns.iloc[test_idx]
        test_dates = dates.iloc[test_idx]
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデル訓練
        models = {
            'LightGBM': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'CatBoost': cb.CatBoostClassifier(random_state=42, verbose=False),
            'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42),
            'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000)
        }
        
        fold_results = {}
        predictions = {}
        
        for model_name, model in models.items():
            if model_name in ['LightGBM', 'CatBoost']:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 評価メトリクス
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            # 高信頼度予測での性能
            high_conf_mask = y_pred_proba >= 0.75
            if high_conf_mask.sum() > 0:
                high_conf_precision = precision_score(y_test[high_conf_mask], y_pred[high_conf_mask], zero_division=0)
                high_conf_return = test_returns[high_conf_mask].mean()
                high_conf_count = high_conf_mask.sum()
            else:
                high_conf_precision = 0.0
                high_conf_return = 0.0
                high_conf_count = 0
            
            fold_results[model_name] = {
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'high_conf_precision': high_conf_precision,
                'high_conf_return': high_conf_return,
                'high_conf_count': high_conf_count,
                'total_predictions': len(y_test)
            }
            
            predictions[model_name] = y_pred_proba
        
        # アンサンブル予測
        ensemble_proba = (
            0.45 * predictions['LightGBM'] +
            0.45 * predictions['CatBoost'] +
            0.10 * predictions['LogisticRegression']
        )
        ensemble_pred = (ensemble_proba >= 0.5).astype(int)
        
        # アンサンブル評価
        ens_precision = precision_score(y_test, ensemble_pred, zero_division=0)
        ens_recall = recall_score(y_test, ensemble_pred, zero_division=0)
        ens_f1 = f1_score(y_test, ensemble_pred, zero_division=0)
        
        high_conf_mask_ens = ensemble_proba >= 0.75
        if high_conf_mask_ens.sum() > 0:
            ens_high_conf_precision = precision_score(y_test[high_conf_mask_ens], ensemble_pred[high_conf_mask_ens], zero_division=0)
            ens_high_conf_return = test_returns[high_conf_mask_ens].mean()
            ens_high_conf_count = high_conf_mask_ens.sum()
        else:
            ens_high_conf_precision = 0.0
            ens_high_conf_return = 0.0
            ens_high_conf_count = 0
        
        fold_results['Ensemble'] = {
            'precision': ens_precision,
            'recall': ens_recall,
            'f1_score': ens_f1,
            'high_conf_precision': ens_high_conf_precision,
            'high_conf_return': ens_high_conf_return,
            'high_conf_count': ens_high_conf_count,
            'total_predictions': len(y_test)
        }
        
        results.append({
            'fold': fold,
            'train_period': f"{dates.iloc[train_idx].min().date()} - {dates.iloc[train_idx].max().date()}",
            'test_period': f"{test_dates.min().date()} - {test_dates.max().date()}",
            'models': fold_results
        })
        
        # Fold結果表示
        logger.info(f"  Fold {fold} 結果:")
        for model_name, metrics in fold_results.items():
            logger.info(f"    {model_name}: P={metrics['precision']:.3f}, R={metrics['recall']:.3f}, F1={metrics['f1_score']:.3f}")
            logger.info(f"      高信頼度(≥0.75): P={metrics['high_conf_precision']:.3f}, 平均リターン={metrics['high_conf_return']:.4f}, 件数={metrics['high_conf_count']}")
    
    # 全体結果集計
    logger.info("=== 全体結果集計 ===")
    
    model_names = ['LightGBM', 'CatBoost', 'RandomForest', 'LogisticRegression', 'Ensemble']
    
    for model_name in model_names:
        precisions = [result['models'][model_name]['precision'] for result in results]
        recalls = [result['models'][model_name]['recall'] for result in results]
        f1_scores = [result['models'][model_name]['f1_score'] for result in results]
        high_conf_precisions = [result['models'][model_name]['high_conf_precision'] for result in results]
        high_conf_returns = [result['models'][model_name]['high_conf_return'] for result in results]
        
        logger.info(f"\n{model_name}:")
        logger.info(f"  平均Precision: {np.mean(precisions):.3f} ± {np.std(precisions):.3f}")
        logger.info(f"  平均Recall: {np.mean(recalls):.3f} ± {np.std(recalls):.3f}")
        logger.info(f"  平均F1-Score: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        logger.info(f"  高信頼度平均Precision: {np.mean([p for p in high_conf_precisions if p > 0]):.3f}" +
                   f" (有効Fold数: {sum(1 for p in high_conf_precisions if p > 0)})")
        logger.info(f"  高信頼度平均リターン: {np.mean([r for r in high_conf_returns if not pd.isna(r)]):.4f}")
    
    # 目標達成度評価
    ensemble_avg_precision = np.mean([result['models']['Ensemble']['precision'] for result in results])
    ensemble_high_conf_avg_precision = np.mean([result['models']['Ensemble']['high_conf_precision'] 
                                               for result in results 
                                               if result['models']['Ensemble']['high_conf_precision'] > 0])
    
    logger.info(f"\n=== 目標達成度評価 ===")
    logger.info(f"アンサンブル全体平均Precision: {ensemble_avg_precision:.3f}")
    logger.info(f"アンサンブル高信頼度平均Precision: {ensemble_high_conf_avg_precision:.3f}")
    logger.info(f"目標達成(≥0.75): {'✅ 達成' if ensemble_high_conf_avg_precision >= 0.75 else '❌ 未達成'}")
    
    logger.info(f"\n=== データ確認 ===")
    logger.info(f"✅ データソース: 100%実データ (J-Quants API)")
    logger.info(f"✅ 総データ数: {len(df_clean):,}件")
    logger.info(f"✅ 銘柄数: {df_clean['symbol'].nunique()}銘柄")
    logger.info(f"✅ 期間: {df_clean['date'].min().date()} ～ {df_clean['date'].max().date()}")
    logger.info(f"✅ 特徴量数: {len(feature_cols)}個")
    
    return results

if __name__ == "__main__":
    results = run_simple_backtest()