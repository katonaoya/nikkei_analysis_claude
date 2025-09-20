#!/usr/bin/env python3
"""
アグレッシブ最適化 - 目標53%達成のための強力な手法
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import warnings
warnings.filterwarnings('ignore')

class AggressiveOptimizer:
    """アグレッシブ最適化"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
    
    def load_data(self, filename: str) -> tuple:
        """データ読み込みと準備"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded: {df.shape}")
        
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
    
    def feature_selection_optimization(self, X, y, dates):
        """特徴量選択最適化"""
        logger.info("🎯 Optimizing feature selection...")
        
        tscv = TimeSeriesSplit(n_splits=2)
        
        # 様々な特徴量選択手法
        selectors = {
            'SelectKBest_20': SelectKBest(f_classif, k=20),
            'SelectKBest_30': SelectKBest(f_classif, k=30),
            'RFE_RandomForest': RFE(
                RandomForestClassifier(n_estimators=50, random_state=42),
                n_features_to_select=25
            )
        }
        
        results = {}
        
        for selector_name, selector in selectors.items():
            logger.info(f"Testing {selector_name}...")
            
            fold_accuracies = []
            
            for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
                X_train = X.iloc[train_idx]
                X_test = X.iloc[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 特徴量選択
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # モデル訓練
                model = LogisticRegression(C=0.1, class_weight='balanced', random_state=42, max_iter=500)
                model.fit(X_train_selected, y_train)
                
                # 予測
                y_pred = model.predict(X_test_selected)
                accuracy = accuracy_score(y_test, y_pred)
                fold_accuracies.append(accuracy)
            
            avg_accuracy = np.mean(fold_accuracies)
            results[selector_name] = {
                'avg_accuracy': avg_accuracy,
                'selector': selector,
                'selected_features': None
            }
            
            logger.info(f"  {selector_name}: {avg_accuracy:.3f}")
        
        # 最良の選択手法を特定
        best_selector_name = max(results, key=lambda x: results[x]['avg_accuracy'])
        logger.info(f"Best feature selector: {best_selector_name}")
        
        return results[best_selector_name]
    
    def hyperparameter_optimization(self, X, y, dates, selected_features=None):
        """ハイパーパラメータ最適化"""
        logger.info("⚡ Hyperparameter optimization...")
        
        if selected_features is not None:
            X = X[selected_features]
        
        # 候補パラメータ（現実的な範囲）
        rf_configs = [
            {'n_estimators': 200, 'max_depth': 12, 'min_samples_split': 20, 'max_features': 'sqrt'},
            {'n_estimators': 300, 'max_depth': 15, 'min_samples_split': 15, 'max_features': 'log2'},
            {'n_estimators': 500, 'max_depth': 20, 'min_samples_split': 10, 'max_features': 'sqrt'},
            {'n_estimators': 400, 'max_depth': 25, 'min_samples_split': 5, 'max_features': 'log2'}
        ]
        
        gb_configs = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            {'n_estimators': 150, 'learning_rate': 0.05, 'max_depth': 8},
            {'n_estimators': 200, 'learning_rate': 0.02, 'max_depth': 10}
        ]
        
        tscv = TimeSeriesSplit(n_splits=2)
        best_results = {}
        
        # RandomForest最適化
        logger.info("Optimizing RandomForest...")
        best_rf_score = 0
        best_rf_config = None
        
        for config in rf_configs:
            model = RandomForestClassifier(
                class_weight='balanced', random_state=42, n_jobs=-1, **config
            )
            
            fold_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                fold_scores.append(score)
            
            avg_score = np.mean(fold_scores)
            if avg_score > best_rf_score:
                best_rf_score = avg_score
                best_rf_config = config
        
        best_results['RandomForest'] = {'score': best_rf_score, 'config': best_rf_config}
        logger.info(f"Best RF: {best_rf_score:.3f} with {best_rf_config}")
        
        # GradientBoosting最適化
        logger.info("Optimizing GradientBoosting...")
        best_gb_score = 0
        best_gb_config = None
        
        for config in gb_configs:
            model = GradientBoostingClassifier(random_state=42, **config)
            
            fold_scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                score = accuracy_score(y_test, model.predict(X_test))
                fold_scores.append(score)
            
            avg_score = np.mean(fold_scores)
            if avg_score > best_gb_score:
                best_gb_score = avg_score
                best_gb_config = config
        
        best_results['GradientBoosting'] = {'score': best_gb_score, 'config': best_gb_config}
        logger.info(f"Best GB: {best_gb_score:.3f} with {best_gb_config}")
        
        return best_results
    
    def final_ensemble_test(self, X, y, dates, best_configs):
        """最終アンサンブルテスト"""
        logger.info("🏆 Final ensemble test...")
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 最適化されたモデル
        best_rf = RandomForestClassifier(
            class_weight='balanced', random_state=42, n_jobs=-1,
            **best_configs['RandomForest']['config']
        )
        
        best_gb = GradientBoostingClassifier(
            random_state=42, **best_configs['GradientBoosting']['config']
        )
        
        # ロジスティック回帰（特徴量選択付き）
        scaler = StandardScaler()
        
        ensemble_scores = []
        individual_scores = {'RF': [], 'GB': [], 'LR': []}
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Final test fold {fold + 1}...")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # RF予測
            best_rf.fit(X_train, y_train)
            rf_pred = best_rf.predict_proba(X_test)[:, 1]
            rf_binary = (rf_pred >= 0.5).astype(int)
            rf_score = accuracy_score(y_test, rf_binary)
            individual_scores['RF'].append(rf_score)
            
            # GB予測
            best_gb.fit(X_train, y_train)
            gb_pred = best_gb.predict_proba(X_test)[:, 1]
            gb_binary = (gb_pred >= 0.5).astype(int)
            gb_score = accuracy_score(y_test, gb_binary)
            individual_scores['GB'].append(gb_score)
            
            # LR予測（標準化）
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            lr_model = LogisticRegression(C=0.1, class_weight='balanced', random_state=42, max_iter=500)
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict_proba(X_test_scaled)[:, 1]
            lr_binary = (lr_pred >= 0.5).astype(int)
            lr_score = accuracy_score(y_test, lr_binary)
            individual_scores['LR'].append(lr_score)
            
            # アンサンブル予測（重み付き平均）
            weights = np.array([rf_score, gb_score, lr_score])
            weights = weights / weights.sum()
            
            ensemble_pred = weights[0] * rf_pred + weights[1] * gb_pred + weights[2] * lr_pred
            ensemble_binary = (ensemble_pred >= 0.5).astype(int)
            ensemble_score = accuracy_score(y_test, ensemble_binary)
            ensemble_scores.append(ensemble_score)
            
            logger.info(f"  Fold {fold+1}: RF={rf_score:.3f}, GB={gb_score:.3f}, LR={lr_score:.3f}, Ensemble={ensemble_score:.3f}")
        
        return {
            'ensemble_scores': ensemble_scores,
            'individual_scores': individual_scores,
            'avg_ensemble': np.mean(ensemble_scores),
            'avg_rf': np.mean(individual_scores['RF']),
            'avg_gb': np.mean(individual_scores['GB']),
            'avg_lr': np.mean(individual_scores['LR'])
        }

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Aggressive optimization")
    parser.add_argument("--features-file", required=True, help="Features file")
    
    args = parser.parse_args()
    
    try:
        optimizer = AggressiveOptimizer()
        
        print("📊 Loading data...")
        X, y, dates, feature_cols = optimizer.load_data(args.features_file)
        
        print("🎯 Feature selection optimization...")
        best_selector = optimizer.feature_selection_optimization(X, y, dates)
        
        print("⚡ Hyperparameter optimization...")
        best_configs = optimizer.hyperparameter_optimization(X, y, dates)
        
        print("🏆 Final ensemble test...")
        final_results = optimizer.final_ensemble_test(X, y, dates, best_configs)
        
        # 結果レポート
        print("\n" + "="*60)
        print("📋 AGGRESSIVE OPTIMIZATION RESULTS")
        print("="*60)
        
        print(f"\n🎯 Feature Selection:")
        print(f"   Best selector accuracy: {best_selector['avg_accuracy']:.3f}")
        
        print(f"\n⚡ Hyperparameter Optimization:")
        for model_name, result in best_configs.items():
            print(f"   {model_name}: {result['score']:.3f}")
        
        print(f"\n🏆 Final Ensemble Results:")
        print(f"   RandomForest:     {final_results['avg_rf']:.3f}")
        print(f"   GradientBoosting: {final_results['avg_gb']:.3f}")
        print(f"   Logistic:         {final_results['avg_lr']:.3f}")
        print(f"   🎯 Ensemble:      {final_results['avg_ensemble']:.3f}")
        
        # 目標達成判定
        best_score = final_results['avg_ensemble']
        baseline = 0.505
        improvement = best_score - baseline
        
        print(f"\n📈 Performance Summary:")
        print(f"   Baseline:    50.5%")
        print(f"   Enhanced:    {best_score:.1%}")
        print(f"   Improvement: +{improvement:.3f} (+{improvement*100:.1f}%)")
        
        if best_score >= 0.53:
            print("\n🎉 SUCCESS: Target 53%+ achieved!")
            status = "SUCCESS"
        elif best_score >= 0.525:
            print("\n🔥 EXCELLENT: Very close to target!")
            status = "EXCELLENT"
        elif best_score >= 0.52:
            print("\n👍 GOOD: Significant improvement!")
            status = "GOOD"
        elif best_score >= 0.515:
            print("\n📈 PROGRESS: Notable improvement")
            status = "PROGRESS"
        else:
            print("\n💡 MODERATE: Some improvement")
            status = "MODERATE"
        
        print(f"\n🎯 Next steps recommendation:")
        if best_score < 0.53:
            print("   Consider deep learning or sector-specific models")
        else:
            print("   Ready for production deployment!")
        
        return {
            'final_accuracy': best_score,
            'improvement': improvement,
            'status': status
        }
        
    except Exception as e:
        logger.error(f"Aggressive optimization failed: {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result and result['final_accuracy'] >= 0.53:
        exit(0)  # 成功
    else:
        exit(1)  # さらなる最適化が必要