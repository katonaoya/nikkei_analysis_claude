#!/usr/bin/env python3
"""
深層学習による精度向上 - 軽量なニューラルネットワーク
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

class DeepLearningBooster:
    """深層学習精度向上"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
    
    def load_recent_data(self, filename: str) -> tuple:
        """最新データの読み込み（高速化）"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        
        # 最新3年のデータのみ使用（より関連性が高い）
        df = df.sort_values('Date')
        cutoff_date = df['Date'].max() - pd.DateOffset(years=3)
        df = df[df['Date'] >= cutoff_date]
        
        logger.info(f"Using recent 3 years: {df.shape}")
        
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Volume',
            'Next_Day_Return', 'Binary_Direction'
        }
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        clean_df = df[df['Binary_Direction'].notna()].copy()
        
        X = clean_df[feature_cols].fillna(0)
        y = clean_df['Binary_Direction']
        dates = clean_df['Date']
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        logger.info(f"Date range: {dates.min()} to {dates.max()}")
        logger.info(f"Balance: {y.value_counts().to_dict()}")
        
        return X, y, dates, feature_cols
    
    def create_neural_networks(self):
        """多様なニューラルネットワーク"""
        
        networks = {
            'MLP_Deep_Narrow': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25, 10),
                activation='relu',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=300,
                random_state=42
            ),
            'MLP_Wide_Shallow': MLPClassifier(
                hidden_layer_sizes=(200, 100),
                activation='tanh',
                alpha=0.01,
                learning_rate='adaptive',
                max_iter=300,
                random_state=43
            ),
            'MLP_Moderate': MLPClassifier(
                hidden_layer_sizes=(150, 75, 25),
                activation='relu',
                alpha=0.005,
                learning_rate='adaptive',
                max_iter=300,
                random_state=44
            ),
            'MLP_Regularized': MLPClassifier(
                hidden_layer_sizes=(80, 40, 20),
                activation='logistic',
                alpha=0.1,
                learning_rate='constant',
                learning_rate_init=0.001,
                max_iter=300,
                random_state=45
            )
        }
        
        return networks
    
    def neural_ensemble_evaluation(self, X, y, dates):
        """ニューラルアンサンブル評価"""
        logger.info("🧠 Neural ensemble evaluation...")
        
        networks = self.create_neural_networks()
        scaler = StandardScaler()
        
        tscv = TimeSeriesSplit(n_splits=2)  # 高速化
        
        results = {}
        all_predictions = {}
        
        for network_name, network in networks.items():
            logger.info(f"Training {network_name}...")
            
            fold_scores = []
            fold_predictions = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                # データ標準化
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # 訓練
                network.fit(X_train_scaled, y_train)
                
                # 予測
                y_pred = network.predict(X_test_scaled)
                accuracy = accuracy_score(y_test, y_pred)
                fold_scores.append(accuracy)
                
                # 確率予測
                y_proba = network.predict_proba(X_test_scaled)[:, 1]
                fold_predictions.extend(y_proba)
            
            avg_score = np.mean(fold_scores)
            results[network_name] = {
                'avg_accuracy': avg_score,
                'std_accuracy': np.std(fold_scores),
                'predictions': fold_predictions
            }
            
            all_predictions[network_name] = fold_predictions
            logger.info(f"  {network_name}: {avg_score:.3f} ± {np.std(fold_scores):.3f}")
        
        # ニューラルアンサンブル
        logger.info("Creating neural ensemble...")
        
        # 性能重み付きアンサンブル
        scores = [results[name]['avg_accuracy'] for name in networks.keys()]
        weights = np.array(scores) / np.sum(scores)
        
        # 予測値の長さを揃える
        min_length = min(len(predictions) for predictions in all_predictions.values())
        aligned_predictions = {
            name: predictions[:min_length] 
            for name, predictions in all_predictions.items()
        }
        
        predictions_matrix = np.array(list(aligned_predictions.values()))
        ensemble_proba = np.average(predictions_matrix, axis=0, weights=weights)
        
        # アンサンブル評価（最新データ）
        y_recent = y.iloc[-min_length:]
        ensemble_binary = (ensemble_proba >= 0.5).astype(int)
        ensemble_accuracy = accuracy_score(y_recent, ensemble_binary)
        
        logger.info(f"Neural ensemble accuracy: {ensemble_accuracy:.3f}")
        
        return {
            'individual_results': results,
            'ensemble_accuracy': ensemble_accuracy,
            'best_individual': max(scores),
            'ensemble_predictions': ensemble_proba
        }

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Deep learning boost")
    parser.add_argument("--features-file", required=True, help="Features file")
    
    args = parser.parse_args()
    
    try:
        booster = DeepLearningBooster()
        
        print("📊 Loading recent data...")
        X, y, dates, feature_cols = booster.load_recent_data(args.features_file)
        
        print("🧠 Neural ensemble evaluation...")
        neural_results = booster.neural_ensemble_evaluation(X, y, dates)
        
        # 最終レポート
        print("\n" + "="*60)
        print("🧠 DEEP LEARNING BOOST RESULTS")
        print("="*60)
        
        print("\n🤖 Neural Networks:")
        for name, result in neural_results['individual_results'].items():
            print(f"   {name:20s}: {result['avg_accuracy']:.3f} ± {result['std_accuracy']:.3f}")
        
        ensemble_acc = neural_results['ensemble_accuracy']
        best_individual = neural_results['best_individual']
        
        print(f"\n🎯 Neural Ensemble: {ensemble_acc:.3f}")
        print(f"🏆 Best Individual: {best_individual:.3f}")
        
        final_best = max(ensemble_acc, best_individual)
        baseline = 0.505
        total_improvement = final_best - baseline
        
        print(f"\n📈 PERFORMANCE SUMMARY:")
        print(f"   Baseline:        50.5%")
        print(f"   🎯 Final Best:   {final_best:.1%}")
        print(f"   📈 Improvement:  +{total_improvement:.3f} (+{total_improvement*100:.1f}%)")
        
        if final_best >= 0.53:
            print("\n🎉🎯 TARGET ACHIEVED: 53%+ accuracy!")
            print("🚀 Ready for production deployment!")
        elif final_best >= 0.525:
            print("\n🔥 EXCELLENT: Very close to target!")
            print("💡 Consider sector-specific optimization")
        elif final_best >= 0.52:
            print("\n👍 GOOD: Significant improvement!")
            print("💡 Try advanced ensemble or market regime analysis")
        else:
            print("\n📈 PROGRESS: Further optimization needed")
        
        return 0 if final_best >= 0.52 else 1
        
    except Exception as e:
        logger.error(f"Deep learning boost failed: {e}")
        return 1

if __name__ == "__main__":
    exit(main())