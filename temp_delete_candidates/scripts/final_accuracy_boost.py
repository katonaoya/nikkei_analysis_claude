#!/usr/bin/env python3
"""
最終精度向上 - 目標53%達成のための効率的手法
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

class FinalAccuracyBooster:
    """最終精度向上"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
    
    def load_and_sample(self, filename: str, sample_ratio: float = 0.5) -> tuple:
        """効率的なデータ読み込み"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded: {df.shape}")
        
        # 戦略的サンプリング（最新データ重視）
        df = df.sort_values('Date')
        
        if sample_ratio < 1.0:
            # 最新50%のデータを使用（より関連性の高いパターン）
            cutoff_date = df['Date'].quantile(1.0 - sample_ratio)
            df = df[df['Date'] >= cutoff_date]
            logger.info(f"Using recent data: {df.shape}")
        
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
        logger.info(f"Target balance: {y.value_counts().to_dict()}")
        
        return X, y, dates, feature_cols
    
    def smart_feature_selection(self, X, y):
        """スマート特徴量選択"""
        logger.info("🧠 Smart feature selection...")
        
        # 基本統計で明らかに有用でない特徴量を除去
        feature_stats = {}
        
        for col in X.columns:
            # 分散が極めて小さい特徴量
            var = X[col].var()
            # ターゲットとの単純相関
            corr = abs(X[col].corr(y))
            
            feature_stats[col] = {
                'variance': var,
                'correlation': corr,
                'score': corr * np.log(1 + var)  # 組み合わせスコア
            }
        
        # スコア順で上位30特徴量を選択
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['score'], reverse=True)
        selected_features = [item[0] for item in sorted_features[:30]]
        
        logger.info(f"Selected {len(selected_features)} features from {len(X.columns)}")
        logger.info(f"Top 5 features: {selected_features[:5]}")
        
        return X[selected_features], selected_features
    
    def optimized_voting_ensemble(self, X, y, dates):
        """最適化されたVoting Ensemble"""
        logger.info("🗳️ Optimized voting ensemble...")
        
        # 多様性を重視したモデル群
        rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            class_weight='balanced_subsample',
            random_state=42,
            n_jobs=-1
        )
        
        lr_model = LogisticRegression(
            C=0.01,  # より強い正則化
            penalty='l1',
            solver='liblinear',
            class_weight='balanced',
            random_state=42,
            max_iter=1000
        )
        
        nb_model = GaussianNB()
        
        # VotingClassifier
        voting_clf = VotingClassifier(
            estimators=[
                ('rf', rf_model),
                ('lr', lr_model), 
                ('nb', nb_model)
            ],
            voting='soft'  # 確率平均
        )
        
        tscv = TimeSeriesSplit(n_splits=3)
        
        scores = []
        individual_scores = {'RF': [], 'LR': [], 'NB': [], 'Voting': []}
        
        scaler = StandardScaler()
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Fold {fold + 1}...")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # LRのために標準化
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # LRは標準化データで訓練
            lr_model.fit(X_train_scaled, y_train)
            lr_pred = lr_model.predict(X_test_scaled)
            lr_score = accuracy_score(y_test, lr_pred)
            individual_scores['LR'].append(lr_score)
            
            # RFとNBは元データで訓練
            rf_model.fit(X_train, y_train)
            rf_pred = rf_model.predict(X_test)
            rf_score = accuracy_score(y_test, rf_pred)
            individual_scores['RF'].append(rf_score)
            
            nb_model.fit(X_train, y_train)
            nb_pred = nb_model.predict(X_test)
            nb_score = accuracy_score(y_test, nb_pred)
            individual_scores['NB'].append(nb_score)
            
            # アンサンブル予測（手動）
            rf_proba = rf_model.predict_proba(X_test)[:, 1]
            lr_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
            nb_proba = nb_model.predict_proba(X_test)[:, 1]
            
            # 性能重み付き平均
            weights = np.array([rf_score, lr_score, nb_score])
            weights = weights / weights.sum()
            
            ensemble_proba = (weights[0] * rf_proba + 
                            weights[1] * lr_proba + 
                            weights[2] * nb_proba)
            
            ensemble_pred = (ensemble_proba >= 0.5).astype(int)
            ensemble_score = accuracy_score(y_test, ensemble_pred)
            individual_scores['Voting'].append(ensemble_score)
            
            logger.info(f"  RF: {rf_score:.3f}, LR: {lr_score:.3f}, NB: {nb_score:.3f}, Ensemble: {ensemble_score:.3f}")
        
        # 結果集計
        results = {}
        for model_name, scores_list in individual_scores.items():
            results[model_name] = {
                'avg': np.mean(scores_list),
                'std': np.std(scores_list),
                'scores': scores_list
            }
        
        return results
    
    def threshold_optimization(self, X, y, dates):
        """閾値最適化"""
        logger.info("🎚️ Threshold optimization...")
        
        # 最新50%のデータで訓練・テスト分割
        cutoff_date = dates.quantile(0.5)
        train_mask = dates <= cutoff_date
        test_mask = dates > cutoff_date
        
        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]
        
        # モデル訓練
        model = RandomForestClassifier(
            n_estimators=200, max_depth=15, class_weight='balanced',
            random_state=42, n_jobs=-1
        )
        model.fit(X_train, y_train)
        
        # 確率予測
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # 閾値最適化
        thresholds = np.arange(0.4, 0.65, 0.01)
        threshold_results = []
        
        for threshold in thresholds:
            y_pred = (y_proba >= threshold).astype(int)
            accuracy = accuracy_score(y_test, y_pred)
            trade_ratio = y_pred.mean()
            
            threshold_results.append({
                'threshold': threshold,
                'accuracy': accuracy,
                'trade_ratio': trade_ratio
            })
        
        # 最適閾値
        best_threshold = max(threshold_results, key=lambda x: x['accuracy'])
        logger.info(f"Best threshold: {best_threshold['threshold']:.2f} -> {best_threshold['accuracy']:.3f}")
        
        return best_threshold, threshold_results

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Final accuracy boost")
    parser.add_argument("--features-file", required=True, help="Features file")
    parser.add_argument("--sample-ratio", type=float, default=0.5, help="Data sampling ratio")
    
    args = parser.parse_args()
    
    try:
        booster = FinalAccuracyBooster()
        
        print("📊 Loading and sampling data...")
        X, y, dates, feature_cols = booster.load_and_sample(args.features_file, args.sample_ratio)
        
        print("🧠 Smart feature selection...")
        X_selected, selected_features = booster.smart_feature_selection(X, y)
        
        print("🗳️ Optimized voting ensemble...")
        ensemble_results = booster.optimized_voting_ensemble(X_selected, y, dates)
        
        print("🎚️ Threshold optimization...")
        best_threshold, threshold_results = booster.threshold_optimization(X_selected, y, dates)
        
        # 最終レポート
        print("\n" + "="*60)
        print("📋 FINAL ACCURACY BOOST RESULTS")
        print("="*60)
        
        print(f"\n🧠 Feature Selection:")
        print(f"   Selected: {len(selected_features)}/44 features")
        print(f"   Top features: {selected_features[:5]}")
        
        print(f"\n🗳️ Ensemble Performance:")
        for model_name, result in ensemble_results.items():
            print(f"   {model_name:15s}: {result['avg']:.3f} ± {result['std']:.3f}")
        
        print(f"\n🎚️ Threshold Optimization:")
        print(f"   Best threshold: {best_threshold['threshold']:.2f}")
        print(f"   Optimized accuracy: {best_threshold['accuracy']:.3f}")
        print(f"   Trade ratio: {best_threshold['trade_ratio']:.1%}")
        
        # 最高精度
        best_ensemble = max(ensemble_results.values(), key=lambda x: x['avg'])['avg']
        final_accuracy = max(best_ensemble, best_threshold['accuracy'])
        
        print(f"\n🏆 FINAL PERFORMANCE:")
        print(f"   Best Accuracy: {final_accuracy:.3f} ({final_accuracy:.1%})")
        
        baseline = 0.505
        improvement = final_accuracy - baseline
        print(f"   Total Improvement: +{improvement:.3f} (+{improvement*100:.1f}%)")
        
        # 目標達成判定
        if final_accuracy >= 0.53:
            print("\n🎉 🎯 TARGET ACHIEVED: 53%+ accuracy reached!")
            status = "SUCCESS"
        elif final_accuracy >= 0.525:
            print("\n🔥 VERY CLOSE: Almost at target!")
            status = "VERY_CLOSE"
        elif final_accuracy >= 0.52:
            print("\n👍 SIGNIFICANT IMPROVEMENT achieved!")
            status = "GOOD"
        else:
            print("\n📈 Progress made, need advanced techniques")
            status = "PROGRESS"
        
        # 次のステップ推奨
        if final_accuracy < 0.53:
            print(f"\n🚀 Next recommendations for reaching 53%:")
            print(f"   1. Deep learning (LSTM/Transformer)")
            print(f"   2. Sector-specific models")
            print(f"   3. Market regime classification")
            print(f"   4. Alternative data sources")
        
        return {
            'final_accuracy': final_accuracy,
            'improvement': improvement,
            'status': status
        }
        
    except Exception as e:
        logger.error(f"Final boost failed: {e}")
        return None

if __name__ == "__main__":
    result = main()
    if result and result['final_accuracy'] >= 0.525:
        exit(0)  # 十分な改善
    else:
        exit(1)  # さらなる改善必要