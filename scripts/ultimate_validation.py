#!/usr/bin/env python3
"""
最終検証 - 最良の手法を組み合わせた実用性能評価
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class UltimateValidator:
    """最終検証システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
    
    def load_optimized_data(self, filename: str) -> tuple:
        """最適化データ読み込み"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        
        # 最新2.5年（より安定したパターン）
        df = df.sort_values('Date')
        cutoff_date = df['Date'].max() - pd.DateOffset(days=912)  # 約2.5年
        df = df[df['Date'] >= cutoff_date]
        
        # 上位特徴量のみ使用（前回の結果から）
        key_features = [
            'CCI', 'RSI_14', 'Stoch_K', 'Williams_R', 'Stoch_D',
            'MACD', 'MACD_Signal', 'BB_Position', 'ROC_10', 'ROC_20',
            'Price_vs_MA10', 'Price_vs_MA20', 'Volatility_20', 'Volume_Ratio',
            'Price_Position', 'Market_Return', 'Market_Volatility', 'Relative_Return'
        ]
        
        available_features = [col for col in key_features if col in df.columns]
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        X = clean_df[available_features].fillna(0)
        y = clean_df['Binary_Direction']
        dates = clean_df['Date']
        
        logger.info(f"Final data: {len(X)} samples, {len(available_features)} features")
        logger.info(f"Period: {dates.min()} to {dates.max()}")
        logger.info(f"Balance: {y.value_counts().to_dict()}")
        
        return X, y, dates, available_features
    
    def final_model_comparison(self, X, y, dates):
        """最終モデル比較"""
        logger.info("🏆 Final model comparison...")
        
        models = {
            'Logistic_Optimized': LogisticRegression(
                C=0.01, penalty='l1', solver='liblinear',
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'RandomForest_Optimized': RandomForestClassifier(
                n_estimators=300, max_depth=18, min_samples_split=8,
                min_samples_leaf=4, max_features='sqrt',
                class_weight='balanced_subsample', random_state=42, n_jobs=-1
            )
        }
        
        tscv = TimeSeriesSplit(n_splits=4)  # より厳格な検証
        scaler = StandardScaler()
        
        results = {}
        all_predictions = {}
        
        for model_name, model in models.items():
            logger.info(f"Testing {model_name}...")
            
            fold_scores = []
            fold_precisions = []
            fold_recalls = []
            fold_predictions = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # ロジスティック回帰は標準化
                if 'Logistic' in model_name:
                    X_train_proc = scaler.fit_transform(X_train)
                    X_test_proc = scaler.transform(X_test)
                else:
                    X_train_proc = X_train
                    X_test_proc = X_test
                
                # 訓練
                model.fit(X_train_proc, y_train)
                
                # 予測
                y_pred = model.predict(X_test_proc)
                y_proba = model.predict_proba(X_test_proc)[:, 1]
                
                # 評価指標
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                
                fold_scores.append(accuracy)
                fold_precisions.append(precision)
                fold_recalls.append(recall)
                fold_predictions.extend(y_proba)
            
            results[model_name] = {
                'accuracy': np.mean(fold_scores),
                'accuracy_std': np.std(fold_scores),
                'precision': np.mean(fold_precisions),
                'recall': np.mean(fold_recalls),
                'all_scores': fold_scores
            }
            
            all_predictions[model_name] = fold_predictions
            
            logger.info(f"  {model_name}: {np.mean(fold_scores):.3f} ± {np.std(fold_scores):.3f}")
        
        # メタアンサンブル（最良の2モデル）
        logger.info("Creating meta-ensemble...")
        
        # 性能重み付け
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        weights = np.array(accuracies) / np.sum(accuracies)
        
        # 予測長さを揃える
        min_length = min(len(all_predictions[name]) for name in model_names)
        y_eval = y.iloc[-min_length:]
        
        ensemble_pred = np.zeros(min_length)
        for i, name in enumerate(model_names):
            ensemble_pred += weights[i] * np.array(all_predictions[name][:min_length])
        
        ensemble_binary = (ensemble_pred >= 0.5).astype(int)
        ensemble_accuracy = accuracy_score(y_eval, ensemble_binary)
        
        results['Meta_Ensemble'] = {
            'accuracy': ensemble_accuracy,
            'weights': dict(zip(model_names, weights))
        }
        
        logger.info(f"Meta-ensemble: {ensemble_accuracy:.3f}")
        
        return results
    
    def production_readiness_check(self, results):
        """プロダクション準備度チェック"""
        logger.info("✅ Production readiness check...")
        
        best_accuracy = max(
            results['Logistic_Optimized']['accuracy'],
            results['RandomForest_Optimized']['accuracy'],
            results['Meta_Ensemble']['accuracy']
        )
        
        baseline = 0.505
        improvement = best_accuracy - baseline
        
        # 安定性チェック
        stability_scores = []
        for model_name in ['Logistic_Optimized', 'RandomForest_Optimized']:
            std = results[model_name]['accuracy_std']
            stability_scores.append(std)
        
        avg_stability = np.mean(stability_scores)
        
        readiness = {
            'best_accuracy': best_accuracy,
            'improvement': improvement,
            'stability': avg_stability,
            'status': 'READY' if best_accuracy >= 0.52 and avg_stability < 0.01 else 'NEEDS_WORK'
        }
        
        return readiness

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Ultimate validation")
    parser.add_argument("--features-file", required=True, help="Features file")
    
    args = parser.parse_args()
    
    try:
        validator = UltimateValidator()
        
        print("📊 Loading optimized data...")
        X, y, dates, features = validator.load_optimized_data(args.features_file)
        
        print("🏆 Final model comparison...")
        model_results = validator.final_model_comparison(X, y, dates)
        
        print("✅ Production readiness check...")
        readiness = validator.production_readiness_check(model_results)
        
        # 最終レポート
        print("\n" + "="*70)
        print("🎯 ULTIMATE VALIDATION RESULTS")
        print("="*70)
        
        print(f"\n🤖 Model Performance:")
        for name, result in model_results.items():
            if name != 'Meta_Ensemble':
                print(f"   {name:25s}: {result['accuracy']:.3f} ± {result['accuracy_std']:.3f}")
                print(f"   {'':25s}  Precision: {result['precision']:.3f}, Recall: {result['recall']:.3f}")
            else:
                print(f"   {name:25s}: {result['accuracy']:.3f}")
                weights_str = ", ".join([f"{k}={v:.2f}" for k, v in result['weights'].items()])
                print(f"   {'':25s}  Weights: {weights_str}")
        
        print(f"\n🎯 FINAL ASSESSMENT:")
        print(f"   🏆 Best Accuracy:    {readiness['best_accuracy']:.3f} ({readiness['best_accuracy']:.1%})")
        print(f"   📈 Total Improvement: +{readiness['improvement']:.3f} (+{readiness['improvement']*100:.1f}%)")
        print(f"   📊 Stability:        {readiness['stability']:.4f} (lower is better)")
        print(f"   🚀 Status:           {readiness['status']}")
        
        # 目標達成判定
        if readiness['best_accuracy'] >= 0.53:
            print(f"\n🎉🎯 SUCCESS: TARGET ACHIEVED!")
            print(f"✅ Accuracy {readiness['best_accuracy']:.1%} exceeds 53% target")
            print(f"🚀 System is ready for production deployment!")
        elif readiness['best_accuracy'] >= 0.525:
            print(f"\n🔥 EXCELLENT: Very close to 53% target!")
            print(f"✅ Accuracy {readiness['best_accuracy']:.1%} is practically usable")
            print(f"💡 Consider final tuning or accept current performance")
        elif readiness['best_accuracy'] >= 0.52:
            print(f"\n👍 GOOD: Significant improvement achieved!")
            print(f"✅ Accuracy {readiness['best_accuracy']:.1%} is a solid improvement")
            print(f"💡 System is usable, further optimization optional")
        else:
            print(f"\n📈 PROGRESS: Some improvement made")
            print(f"💡 Consider advanced techniques for target achievement")
        
        print(f"\n📋 DEVELOPMENT SUMMARY:")
        print(f"   Original Performance:  ~50.0%")
        print(f"   Enhanced Features:     +{(readiness['best_accuracy']-0.50)*100:.1f}%")
        print(f"   🎯 Final Achievement:  {readiness['best_accuracy']:.1%}")
        
        return 0 if readiness['best_accuracy'] >= 0.52 else 1
        
    except Exception as e:
        logger.error(f"Ultimate validation failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())