#!/usr/bin/env python3
"""
アンサンブル最適化 - 複数モデルによる精度向上
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class EnsembleOptimizer:
    """アンサンブル最適化"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
    
    def load_features(self, filename: str) -> pd.DataFrame:
        """特徴量読み込み"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded features: {df.shape}")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> tuple:
        """データ準備"""
        
        # 除外列
        exclude_cols = {
            'Date', 'Code', 'Close', 'High', 'Low', 'Volume',
            'Next_Day_Return', 'Binary_Direction'
        }
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # クリーニング
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[feature_cols].fillna(0)
        y = clean_df['Binary_Direction']
        dates = clean_df['Date']
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        logger.info(f"Target balance: {y.value_counts().to_dict()}")
        
        return X, y, dates, feature_cols
    
    def create_optimized_models(self):
        """最適化モデル群"""
        
        models = {
            # 1. 線形モデル
            'Logistic_L1': LogisticRegression(
                penalty='l1', solver='liblinear', C=0.1,
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'Logistic_L2': LogisticRegression(
                penalty='l2', C=1.0,
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            
            # 2. 木ベースモデル（多様性重視）
            'RandomForest_Conservative': RandomForestClassifier(
                n_estimators=200, max_depth=8, min_samples_split=50,
                min_samples_leaf=20, max_features='sqrt',
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'RandomForest_Aggressive': RandomForestClassifier(
                n_estimators=300, max_depth=15, min_samples_split=20,
                min_samples_leaf=10, max_features='log2',
                class_weight='balanced', random_state=43, n_jobs=-1
            ),
            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=200, max_depth=10, min_samples_split=30,
                min_samples_leaf=15, max_features='sqrt',
                class_weight='balanced', random_state=44, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingClassifier(
                n_estimators=100, learning_rate=0.1, max_depth=6,
                min_samples_split=50, min_samples_leaf=20,
                random_state=45
            ),
            
            # 3. ニューラルネットワーク
            'MLP_Small': MLPClassifier(
                hidden_layer_sizes=(50, 25), activation='relu', 
                alpha=0.01, learning_rate='adaptive', max_iter=500,
                random_state=46
            ),
            'MLP_Medium': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25), activation='tanh',
                alpha=0.001, learning_rate='adaptive', max_iter=500,
                random_state=47
            )
        }
        
        return models
    
    def evaluate_individual_models(self, X, y, dates, models, n_splits=3):
        """個別モデル評価"""
        logger.info("🔍 Evaluating individual models...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
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
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train = X_scaled.iloc[train_idx] if 'MLP' in model_name or 'Logistic' in model_name else X.iloc[train_idx]
                X_test = X_scaled.iloc[test_idx] if 'MLP' in model_name or 'Logistic' in model_name else X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 訓練
                model.fit(X_train, y_train)
                
                # 予測
                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
                
                # AUC（可能な場合）
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    auc = roc_auc_score(y_test, y_proba)
                    fold_aucs.append(auc)
            
            results[model_name] = {
                'accuracies': fold_accuracies,
                'avg_accuracy': np.mean(fold_accuracies),
                'std_accuracy': np.std(fold_accuracies),
                'avg_auc': np.mean(fold_aucs) if fold_aucs else None
            }
            
            logger.info(f"  {model_name}: {np.mean(fold_accuracies):.3f} ± {np.std(fold_accuracies):.3f}")
        
        return results
    
    def create_ensemble_predictions(self, X, y, dates, models, n_splits=3):
        """アンサンブル予測"""
        logger.info("🎯 Creating ensemble predictions...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        ensemble_results = []
        
        # データ標準化
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Ensemble fold {fold + 1}...")
            
            # 各モデルの予測を収集
            model_predictions = {}
            model_probabilities = {}
            
            for model_name, model in models.items():
                X_train = X_scaled.iloc[train_idx] if 'MLP' in model_name or 'Logistic' in model_name else X.iloc[train_idx]
                X_test = X_scaled.iloc[test_idx] if 'MLP' in model_name or 'Logistic' in model_name else X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                
                # 訓練と予測
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                model_predictions[model_name] = y_pred
                
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_test)[:, 1]
                    model_probabilities[model_name] = y_proba
            
            # アンサンブル戦略
            y_test = y.iloc[test_idx]
            
            # 1. 単純多数決
            predictions_array = np.array(list(model_predictions.values()))
            majority_vote = np.round(np.mean(predictions_array, axis=0)).astype(int)
            majority_accuracy = accuracy_score(y_test, majority_vote)
            
            # 2. 重み付き平均（確率ベース）
            if model_probabilities:
                prob_array = np.array(list(model_probabilities.values()))
                weighted_prob = np.mean(prob_array, axis=0)
                weighted_pred = (weighted_prob >= 0.5).astype(int)
                weighted_accuracy = accuracy_score(y_test, weighted_pred)
            else:
                weighted_accuracy = majority_accuracy
            
            # 3. 上位モデルのみ使用
            top_models = ['RandomForest_Aggressive', 'GradientBoosting', 'ExtraTrees']
            top_predictions = [model_predictions[name] for name in top_models if name in model_predictions]
            if top_predictions:
                top_pred_array = np.array(top_predictions)
                top_vote = np.round(np.mean(top_pred_array, axis=0)).astype(int)
                top_accuracy = accuracy_score(y_test, top_vote)
            else:
                top_accuracy = majority_accuracy
            
            ensemble_results.append({
                'fold': fold + 1,
                'majority_accuracy': majority_accuracy,
                'weighted_accuracy': weighted_accuracy,
                'top_models_accuracy': top_accuracy,
                'test_samples': len(y_test)
            })
        
        return ensemble_results

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Ensemble optimization")
    parser.add_argument("--features-file", required=True, help="Features file")
    
    args = parser.parse_args()
    
    try:
        optimizer = EnsembleOptimizer()
        
        print("📊 Loading features...")
        df = optimizer.load_features(args.features_file)
        
        print("🔧 Preparing data...")
        X, y, dates, feature_cols = optimizer.prepare_data(df)
        
        print("🤖 Creating optimized models...")
        models = optimizer.create_optimized_models()
        
        print("🔍 Evaluating individual models...")
        individual_results = optimizer.evaluate_individual_models(X, y, dates, models)
        
        print("🎯 Creating ensemble predictions...")
        ensemble_results = optimizer.create_ensemble_predictions(X, y, dates, models)
        
        # レポート
        print("\n" + "="*60)
        print("📋 ENSEMBLE OPTIMIZATION RESULTS")
        print("="*60)
        
        print("\n🤖 Individual Model Performance:")
        for model_name, result in individual_results.items():
            auc_text = f", AUC: {result['avg_auc']:.3f}" if result['avg_auc'] else ""
            print(f"   {model_name:25s}: {result['avg_accuracy']:.3f} ± {result['std_accuracy']:.3f}{auc_text}")
        
        print("\n🎯 Ensemble Performance:")
        majority_accs = [r['majority_accuracy'] for r in ensemble_results]
        weighted_accs = [r['weighted_accuracy'] for r in ensemble_results]
        top_accs = [r['top_models_accuracy'] for r in ensemble_results]
        
        print(f"   Majority Vote:    {np.mean(majority_accs):.3f} ± {np.std(majority_accs):.3f}")
        print(f"   Weighted Average: {np.mean(weighted_accs):.3f} ± {np.std(weighted_accs):.3f}")
        print(f"   Top Models Only:  {np.mean(top_accs):.3f} ± {np.std(top_accs):.3f}")
        
        best_ensemble = max(np.mean(majority_accs), np.mean(weighted_accs), np.mean(top_accs))
        best_individual = max([r['avg_accuracy'] for r in individual_results.values()])
        
        improvement = best_ensemble - best_individual
        print(f"\n📈 Best Ensemble vs Best Individual:")
        print(f"   Improvement: +{improvement:.3f} ({improvement*100:.1f}%)")
        
        if best_ensemble > 0.52:
            print("\n🎉 SUCCESS: Achieved >52% accuracy target!")
        elif best_ensemble > 0.515:
            print("\n👍 GOOD: Significant improvement achieved")
        else:
            print("\n💡 MODERATE: Some improvement, try additional techniques")
        
        print("\n✅ Ensemble optimization completed!")
        
    except Exception as e:
        logger.error(f"Ensemble optimization failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())