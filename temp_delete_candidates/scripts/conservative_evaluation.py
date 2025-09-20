#!/usr/bin/env python3
"""
保守的な評価スクリプト - 現実的な性能検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

class ConservativeEvaluator:
    """保守的な評価システム"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
    
    def load_data(self, filename: str) -> pd.DataFrame:
        """データ読み込み"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded data: {df.shape}")
        return df
    
    def prepare_conservative_features(self, df: pd.DataFrame) -> tuple:
        """保守的な特徴量選択"""
        
        # 基本特徴量のみ使用（ラグ特徴量は除外）
        basic_features = [
            'MA_5', 'MA_10', 'MA_20',
            'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA20',
            'RSI_14', 'Volatility_20', 'Volume_Ratio', 'Price_Position'
        ]
        
        # 利用可能な特徴量のみ選択
        available_features = [col for col in basic_features if col in df.columns]
        
        # ターゲットと日付の確認
        if 'Binary_Direction' not in df.columns:
            raise ValueError("Target column 'Binary_Direction' not found")
        if 'Date' not in df.columns:
            raise ValueError("Date column not found")
        
        # データクリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[available_features].fillna(0)
        y = clean_df['Binary_Direction']
        dates = clean_df['Date']
        
        logger.info(f"Conservative features: {len(available_features)}")
        logger.info(f"Clean samples: {len(X)}")
        logger.info(f"Date range: {dates.min()} to {dates.max()}")
        logger.info(f"Target balance: {y.value_counts().to_dict()}")
        
        return X, y, dates, available_features
    
    def time_series_validation(self, X, y, dates, n_splits=3):
        """厳格な時系列検証"""
        logger.info("🔍 Running conservative time series validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # シンプルなモデル
        models = {
            'Logistic_Regression': LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            'Random_Forest_Simple': RandomForestClassifier(
                n_estimators=100,  # 少ない木の数
                max_depth=5,       # 浅い深度
                min_samples_split=100,  # 大きな分割サンプル数
                min_samples_leaf=50,    # 大きな葉サンプル数
                max_features='sqrt',
                random_state=42,
                class_weight='balanced'
            )
        }
        
        results = {}
        
        for model_name, model in models.items():
            logger.info(f"Testing {model_name}...")
            
            fold_results = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 訓練
                model.fit(X_train, y_train)
                
                # 予測
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # 期間情報
                train_dates = dates.iloc[train_idx]
                test_dates = dates.iloc[test_idx]
                
                fold_result = {
                    'fold': fold + 1,
                    'accuracy': accuracy,
                    'train_start': train_dates.min(),
                    'train_end': train_dates.max(),
                    'test_start': test_dates.min(),
                    'test_end': test_dates.max(),
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx)
                }
                
                fold_results.append(fold_result)
                logger.info(f"  Fold {fold+1}: {accuracy:.1%}")
            
            # 統計
            accuracies = [r['accuracy'] for r in fold_results]
            results[model_name] = {
                'fold_results': fold_results,
                'avg_accuracy': np.mean(accuracies),
                'std_accuracy': np.std(accuracies),
                'min_accuracy': min(accuracies),
                'max_accuracy': max(accuracies)
            }
        
        return results
    
    def realistic_performance_simulation(self, X, y, dates):
        """現実的な性能シミュレーション"""
        logger.info("💰 Running realistic performance simulation...")
        
        # 最後の1年をテストに使用
        test_date = dates.max() - pd.DateOffset(days=365)
        train_mask = dates <= test_date
        test_mask = dates > test_date
        
        if test_mask.sum() == 0:
            logger.warning("No test data available for realistic simulation")
            return None
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y[train_mask]
        y_test = y[test_mask]
        
        # シンプルなロジスティック回帰
        model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
        model.fit(X_train, y_train)
        
        # 予測確率
        pred_proba = model.predict_proba(X_test)[:, 1]
        
        # 閾値別の性能
        thresholds = [0.5, 0.55, 0.6, 0.65, 0.7]
        
        results = []
        for threshold in thresholds:
            predictions = (pred_proba >= threshold).astype(int)
            accuracy = accuracy_score(y_test, predictions)
            
            # 実際のトレード数
            trade_count = predictions.sum()
            
            if trade_count > 0:
                # トレードした銘柄のリターン
                actual_returns = y_test[predictions == 1]
                win_rate = actual_returns.mean()
                
                results.append({
                    'threshold': threshold,
                    'accuracy': accuracy,
                    'trades': trade_count,
                    'win_rate': win_rate,
                    'trade_ratio': trade_count / len(y_test)
                })
        
        return results

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Conservative evaluation")
    parser.add_argument("--features-file", required=True, help="Features file name")
    
    args = parser.parse_args()
    
    try:
        evaluator = ConservativeEvaluator()
        
        print("📊 Loading data...")
        df = evaluator.load_data(args.features_file)
        
        print("🔧 Preparing conservative features...")
        X, y, dates, features = evaluator.prepare_conservative_features(df)
        
        print("\n📈 Running time series validation...")
        ts_results = evaluator.time_series_validation(X, y, dates, n_splits=3)
        
        print("\n💰 Running realistic simulation...")
        realistic_results = evaluator.realistic_performance_simulation(X, y, dates)
        
        # レポート
        print("\n" + "="*60)
        print("📋 CONSERVATIVE EVALUATION RESULTS")
        print("="*60)
        
        for model_name, result in ts_results.items():
            print(f"\n🤖 {model_name}:")
            print(f"   Average Accuracy: {result['avg_accuracy']:.1%} ± {result['std_accuracy']:.1%}")
            print(f"   Range: {result['min_accuracy']:.1%} - {result['max_accuracy']:.1%}")
        
        if realistic_results:
            print(f"\n💡 Realistic Performance (Last Year):")
            for result in realistic_results:
                print(f"   Threshold {result['threshold']:.2f}: "
                      f"Win Rate {result['win_rate']:.1%}, "
                      f"Trades {result['trades']:,} ({result['trade_ratio']:.1%})")
        
        # 判定
        avg_perf = np.mean([r['avg_accuracy'] for r in ts_results.values()])
        if avg_perf <= 0.55:
            print(f"\n✅ REALISTIC: Average performance {avg_perf:.1%} is within expected range")
        else:
            print(f"\n⚠️ SUSPICIOUS: Average performance {avg_perf:.1%} may indicate remaining issues")
        
        print("\n✅ Conservative evaluation completed!")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())