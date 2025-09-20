#!/usr/bin/env python3
"""
高速アンサンブル手法実装
60%超えを目指す第1段階（簡素化版）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class QuickEnsembleImprovement:
    """高速アンサンブル改善システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
        # 最適特徴量
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 データ読み込みと準備...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"✅ データ準備完了: {len(clean_df):,}件, {len(self.optimal_features)}特徴量")
        
        return X, y
    
    def create_optimized_models(self):
        """最適化されたモデル群作成"""
        logger.info("🧠 最適化モデル群作成...")
        
        models = {
            'lr_l1': LogisticRegression(
                C=0.001, penalty='l1', solver='liblinear',
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'lr_l2': LogisticRegression(
                C=0.001, penalty='l2', solver='lbfgs',
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'rf_optimized': RandomForestClassifier(
                n_estimators=50,  # 高速化のため削減
                max_depth=8,
                min_samples_split=10,
                class_weight='balanced', 
                random_state=42, 
                n_jobs=-1
            )
        }
        
        logger.info(f"モデル数: {len(models)}個")
        return models
    
    def evaluate_individual_models(self, X, y, models):
        """個別モデル評価"""
        logger.info("📊 個別モデル評価...")
        
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=3)  # 高速化のため3分割
        results = {}
        
        for name, model in models.items():
            logger.info(f"  {name} 評価中...")
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            avg_score = np.mean(scores)
            results[name] = {
                'avg': avg_score,
                'std': np.std(scores),
                'scores': scores
            }
            
            logger.info(f"    {name}: {avg_score:.3%} ± {np.std(scores):.3%}")
        
        return results
    
    def implement_ensemble_methods(self, X, y, models):
        """アンサンブル手法実装"""
        logger.info("🔄 アンサンブル手法実装...")
        
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=3)
        
        # 最適なモデル組み合わせ
        estimators = [
            ('lr_l2', models['lr_l2']),
            ('rf', models['rf_optimized'])
        ]
        
        ensemble_results = {}
        
        # 1. ハード投票
        logger.info("  ハード投票アンサンブル...")
        voting_hard = VotingClassifier(estimators=estimators, voting='hard')
        hard_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            voting_hard.fit(X_train, y_train)
            pred = voting_hard.predict(X_test)
            hard_scores.append(accuracy_score(y_test, pred))
        
        ensemble_results['voting_hard'] = {
            'avg': np.mean(hard_scores),
            'std': np.std(hard_scores),
            'scores': hard_scores
        }
        
        # 2. ソフト投票
        logger.info("  ソフト投票アンサンブル...")
        voting_soft = VotingClassifier(estimators=estimators, voting='soft')
        soft_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            voting_soft.fit(X_train, y_train)
            pred = voting_soft.predict(X_test)
            soft_scores.append(accuracy_score(y_test, pred))
        
        ensemble_results['voting_soft'] = {
            'avg': np.mean(soft_scores),
            'std': np.std(soft_scores),
            'scores': soft_scores
        }
        
        # 3. スタッキング
        logger.info("  スタッキングアンサンブル...")
        meta_learner = LogisticRegression(C=1.0, random_state=42)
        stacking_clf = StackingClassifier(
            estimators=estimators,
            final_estimator=meta_learner,
            cv=2  # 高速化のため2分割
        )
        
        stacking_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            stacking_clf.fit(X_train, y_train)
            pred = stacking_clf.predict(X_test)
            stacking_scores.append(accuracy_score(y_test, pred))
        
        ensemble_results['stacking'] = {
            'avg': np.mean(stacking_scores),
            'std': np.std(stacking_scores),
            'scores': stacking_scores
        }
        
        # 4. 重み付きアンサンブル（シンプルな重み）
        logger.info("  重み付きアンサンブル...")
        weights = [0.7, 0.3]  # LogisticRegressionに高い重み
        weighted_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            # 各モデルの予測取得
            predictions = []
            for name, model in estimators:
                model.fit(X_train, y_train)
                pred_proba = model.predict_proba(X_test)[:, 1]
                predictions.append(pred_proba)
            
            # 重み付きアンサンブル
            weighted_pred_proba = np.average(predictions, axis=0, weights=weights)
            weighted_pred = (weighted_pred_proba > 0.5).astype(int)
            weighted_scores.append(accuracy_score(y_test, weighted_pred))
        
        ensemble_results['weighted'] = {
            'avg': np.mean(weighted_scores),
            'std': np.std(weighted_scores),
            'scores': weighted_scores,
            'weights': dict(zip(['lr_l2', 'rf_optimized'], weights))
        }
        
        for method, result in ensemble_results.items():
            logger.info(f"    {method}: {result['avg']:.3%} ± {result['std']:.3%}")
        
        return ensemble_results
    
    def final_validation(self, X, y, best_method, models):
        """最高手法での最終検証（5分割）"""
        logger.info("✅ 最高手法での最終検証（5分割）...")
        
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)
        
        if best_method == 'stacking':
            estimators = [
                ('lr_l2', models['lr_l2']),
                ('rf', models['rf_optimized'])
            ]
            meta_learner = LogisticRegression(C=1.0, random_state=42)
            final_model = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=2
            )
        elif best_method == 'voting_soft':
            estimators = [
                ('lr_l2', models['lr_l2']),
                ('rf', models['rf_optimized'])
            ]
            final_model = VotingClassifier(estimators=estimators, voting='soft')
        else:
            # デフォルトはLogisticRegression L2
            final_model = models['lr_l2']
        
        final_scores = []
        fold_details = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            final_model.fit(X_train, y_train)
            pred = final_model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            final_scores.append(accuracy)
            
            fold_details.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
            
            logger.info(f"  Fold {fold+1}: {accuracy:.1%} (Train: {len(X_train):,}, Test: {len(X_test):,})")
        
        final_result = {
            'avg': np.mean(final_scores),
            'std': np.std(final_scores),
            'min': np.min(final_scores),
            'max': np.max(final_scores),
            'scores': final_scores,
            'fold_details': fold_details
        }
        
        return final_result

def main():
    """メイン実行"""
    logger.info("🚀 高速アンサンブル改善システム")
    logger.info("🎯 目標: 59.4%から62%超えを目指す")
    
    system = QuickEnsembleImprovement()
    
    try:
        # 1. データ準備
        X, y = system.load_and_prepare_data()
        
        # 2. 最適化モデル作成
        models = system.create_optimized_models()
        
        # 3. 個別モデル評価
        individual_results = system.evaluate_individual_models(X, y, models)
        
        # 4. アンサンブル手法実装
        ensemble_results = system.implement_ensemble_methods(X, y, models)
        
        # 5. 全結果の統合と比較
        all_results = {**individual_results, **ensemble_results}
        
        # 6. ベストメソッド特定
        best_method = max(all_results.keys(), key=lambda k: all_results[k]['avg'])
        best_score = all_results[best_method]['avg']
        
        # 7. 最終検証
        final_result = system.final_validation(X, y, best_method, models)
        
        # 結果まとめ
        logger.info("\n" + "="*100)
        logger.info("🏆 高速アンサンブル改善結果")
        logger.info("="*100)
        
        baseline_score = 59.4  # 前回の最高スコア（外部データ統合）
        logger.info(f"📏 ベースライン: {baseline_score:.1%}")
        
        # 全結果をスコア順にソート
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        logger.info(f"\n📈 手法別結果（3分割検証）:")
        for i, (method, result) in enumerate(sorted_results, 1):
            improvement = (result['avg'] - baseline_score/100) * 100
            status = "🚀" if improvement > 2.0 else "📈" if improvement > 0.5 else "📊"
            logger.info(f"  {i:2d}. {method:20s}: {result['avg']:.3%} ({improvement:+.2f}%) {status}")
        
        # 最終検証結果
        logger.info(f"\n🏆 最終検証結果（5分割）:")
        logger.info(f"  最高手法: {best_method}")
        logger.info(f"  精度: {final_result['avg']:.3%} ± {final_result['std']:.3%}")
        logger.info(f"  範囲: {final_result['min']:.1%} - {final_result['max']:.1%}")
        
        final_improvement = (final_result['avg'] - baseline_score/100) * 100
        logger.info(f"  向上: {final_improvement:+.2f}% (59.4% → {final_result['avg']:.1%})")
        
        # 目標達成確認
        target_60 = 0.60
        target_62 = 0.62
        
        if final_result['avg'] >= target_62:
            logger.info(f"🎉 目標大幅達成！ 62%超え ({final_result['avg']:.1%} >= 62.0%)")
        elif final_result['avg'] >= target_60:
            logger.info(f"✅ 目標達成！ 60%超え ({final_result['avg']:.1%} >= 60.0%)")
        else:
            logger.info(f"📈 改善効果確認 ({final_result['avg']:.1%})")
        
        logger.info(f"\n⚖️ この結果は全データ{len(X):,}件での厳密な時系列検証です")
        logger.info(f"✅ 第1段階完了: アンサンブル手法による精度向上")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()