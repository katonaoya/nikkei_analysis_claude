#!/usr/bin/env python3
"""
高速アンサンブルテスト - 効率的な精度向上検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class QuickEnsembleTester:
    """高速アンサンブルテスト"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
    
    def load_and_prepare(self, filename: str, sample_ratio: float = 0.2) -> tuple:
        """データ読み込みとサンプリング"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded features: {df.shape}")
        
        # 高速化のためサンプリング（時系列を保持）
        if sample_ratio < 1.0:
            unique_dates = sorted(df['Date'].unique())
            sample_dates = unique_dates[::int(1/sample_ratio)]
            df = df[df['Date'].isin(sample_dates)]
            logger.info(f"Sampled to: {df.shape}")
        
        # 特徴量準備
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Volume',
            'Next_Day_Return', 'Binary_Direction'
        }
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[feature_cols].fillna(0)
        y = clean_df['Binary_Direction']
        dates = clean_df['Date']
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        return X, y, dates, feature_cols
    
    def quick_model_comparison(self, X, y, dates):
        """高速モデル比較"""
        logger.info("🚀 Quick model comparison...")
        
        # 高速モデル群
        models = {
            'Logistic_Enhanced': LogisticRegression(
                C=0.1, class_weight='balanced', random_state=42, max_iter=500
            ),
            'RandomForest_Fast': RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=30,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'ExtraTrees_Fast': ExtraTreesClassifier(
                n_estimators=100, max_depth=8, min_samples_split=40,
                class_weight='balanced', random_state=43, n_jobs=-1
            )
        }
        
        tscv = TimeSeriesSplit(n_splits=2)  # 高速化のため2分割
        results = {}
        
        # データ標準化
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        for model_name, model in models.items():
            logger.info(f"Testing {model_name}...")
            
            fold_accuracies = []
            fold_aucs = []
            predictions_all = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train = X_scaled.iloc[train_idx] if 'Logistic' in model_name else X.iloc[train_idx]
                X_test = X_scaled.iloc[test_idx] if 'Logistic' in model_name else X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 訓練
                model.fit(X_train, y_train)
                
                # 予測
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                
                # 確率予測
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                    fold_aucs.append(auc)
                    predictions_all.extend(y_proba)
                else:
                    predictions_all.extend(y_pred)
            
            results[model_name] = {
                'avg_accuracy': np.mean(fold_accuracies),
                'std_accuracy': np.std(fold_accuracies),
                'avg_auc': np.mean(fold_aucs) if fold_aucs else None,
                'predictions': predictions_all
            }
            
            auc_text = f", AUC: {np.mean(fold_aucs):.3f}" if fold_aucs else ""
            logger.info(f"  {model_name}: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}{auc_text}")
        
        return results
    
    def ensemble_combination(self, individual_results, y):
        """アンサンブル組み合わせテスト"""
        logger.info("🎯 Testing ensemble combinations...")
        
        # 予測値を配列化（長さを揃える）
        model_names = list(individual_results.keys())
        
        # 最小長を取得
        min_length = min(len(individual_results[name]['predictions']) for name in model_names)
        logger.info(f"Aligning predictions to length: {min_length}")
        
        # yも同じ長さに調整
        y_aligned = y.iloc[:min_length] if len(y) > min_length else y
        
        predictions_matrix = np.array([
            individual_results[name]['predictions'][:min_length] 
            for name in model_names
        ])
        
        # 様々なアンサンブル戦略
        ensemble_strategies = {
            'Simple_Average': np.mean(predictions_matrix, axis=0),
        }
        
        # 性能重み付け
        accuracies = [individual_results[name]['avg_accuracy'] for name in model_names]
        weights = np.array(accuracies) / np.sum(accuracies)
        ensemble_strategies['Weighted_Performance'] = np.average(predictions_matrix, axis=0, weights=weights)
        
        # 上位2モデルの平均
        top_2_indices = np.argsort(accuracies)[-2:]
        ensemble_strategies['Best_Two_Average'] = np.mean(predictions_matrix[top_2_indices], axis=0)
        
        # 各戦略の評価
        ensemble_results = {}
        
        for strategy_name, ensemble_pred in ensemble_strategies.items():
            if ensemble_pred is not None:
                # 二値化
                binary_pred = (ensemble_pred >= 0.5).astype(int)
                accuracy = accuracy_score(y_aligned, binary_pred)
                
                ensemble_results[strategy_name] = {
                    'accuracy': accuracy,
                    'predictions': ensemble_pred
                }
                
                logger.info(f"  {strategy_name}: {accuracy:.3f}")
        
        return ensemble_results

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Quick ensemble test")
    parser.add_argument("--features-file", required=True, help="Features file")
    parser.add_argument("--sample-ratio", type=float, default=0.3, help="Sampling ratio for speed")
    
    args = parser.parse_args()
    
    try:
        tester = QuickEnsembleTester()
        
        print("📊 Loading and preparing data...")
        X, y, dates, feature_cols = tester.load_and_prepare(args.features_file, args.sample_ratio)
        
        print("🤖 Running quick model comparison...")
        individual_results = tester.quick_model_comparison(X, y, dates)
        
        print("🎯 Testing ensemble combinations...")
        ensemble_results = tester.ensemble_combination(individual_results, y)
        
        # 結果レポート
        print("\n" + "="*60)
        print("📋 QUICK ENSEMBLE TEST RESULTS")
        print("="*60)
        
        print("\n🤖 Individual Models:")
        for name, result in individual_results.items():
            auc_text = f", AUC: {result['avg_auc']:.3f}" if result['avg_auc'] else ""
            print(f"   {name:25s}: {result['avg_accuracy']:.3f} ± {result['std_accuracy']:.3f}{auc_text}")
        
        print("\n🎯 Ensemble Strategies:")
        for name, result in ensemble_results.items():
            print(f"   {name:25s}: {result['accuracy']:.3f}")
        
        # 最良結果
        all_results = {**{k: v['avg_accuracy'] for k, v in individual_results.items()},
                      **{k: v['accuracy'] for k, v in ensemble_results.items()}}
        
        best_model = max(all_results, key=all_results.get)
        best_accuracy = all_results[best_model]
        
        print(f"\n🏆 Best Performance:")
        print(f"   Model: {best_model}")
        print(f"   Accuracy: {best_accuracy:.3f} ({best_accuracy:.1%})")
        
        # 目標達成判定
        if best_accuracy >= 0.53:
            print("\n🎉 SUCCESS: Target accuracy 53%+ achieved!")
        elif best_accuracy >= 0.52:
            print("\n👍 GOOD: Significant improvement, close to target!")
        elif best_accuracy >= 0.515:
            print("\n📈 PROGRESS: Noticeable improvement achieved")
        else:
            print("\n💡 MODERATE: Some improvement, need additional techniques")
        
        baseline = 0.505  # 元のベースライン
        improvement = best_accuracy - baseline
        print(f"\n📈 Improvement over baseline: +{improvement:.3f} (+{improvement*100:.1f}%)")
        
        print("\n✅ Quick ensemble test completed!")
        
        return 0 if best_accuracy >= 0.52 else 1
        
    except Exception as e:
        logger.error(f"Ensemble test failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())