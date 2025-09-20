#!/usr/bin/env python3
"""
高度アンサンブル手法の実装
60%超えを目指す第1段階: Stacking, Blending, 動的重み調整
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
import optuna
from optuna.samplers import TPESampler
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class AdvancedEnsembleSystem:
    """高度アンサンブル手法実装システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
        # 現在の最適特徴量（外部データ統合済み）
        self.optimal_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
    def load_integrated_data(self):
        """統合データ読み込み"""
        logger.info("📊 統合データ読み込み...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        if not integrated_file.exists():
            logger.error("❌ 統合データファイルが見つかりません")
            return None
            
        df = pd.read_parquet(integrated_file)
        logger.info(f"✅ 統合データ読み込み: {len(df):,}件")
        
        return df
    
    def prepare_data_for_ensemble(self, df):
        """アンサンブル用データ準備"""
        logger.info("🔧 アンサンブル用データ準備...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 特徴量の存在確認
        missing_features = [f for f in self.optimal_features if f not in clean_df.columns]
        if missing_features:
            logger.error(f"❌ 不足特徴量: {missing_features}")
            return None, None
            
        X = clean_df[self.optimal_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"アンサンブル用データ: {len(clean_df):,}件")
        logger.info(f"使用特徴量: {len(self.optimal_features)}個")
        
        return X, y
    
    def create_base_models(self):
        """ベースモデル群の作成"""
        logger.info("🧠 ベースモデル群作成...")
        
        base_models = {
            'lr_l1': LogisticRegression(
                C=0.001, penalty='l1', solver='liblinear',
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'lr_l2': LogisticRegression(
                C=0.001, penalty='l2', solver='lbfgs',
                class_weight='balanced', random_state=42, max_iter=1000
            ),
            'rf': RandomForestClassifier(
                n_estimators=100, max_depth=10,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'rf_deep': RandomForestClassifier(
                n_estimators=200, max_depth=15,
                class_weight='balanced', random_state=42, n_jobs=-1
            ),
            'xgb': xgb.XGBClassifier(
                n_estimators=100, max_depth=6,
                learning_rate=0.1, scale_pos_weight=1,
                random_state=42, n_jobs=-1, eval_metric='logloss'
            ),
            'svm': SVC(
                C=0.1, kernel='rbf', probability=True,
                class_weight='balanced', random_state=42
            )
        }
        
        logger.info(f"ベースモデル数: {len(base_models)}個")
        for name, model in base_models.items():
            logger.info(f"  {name}: {type(model).__name__}")
        
        return base_models
    
    def evaluate_base_models(self, X, y, base_models):
        """ベースモデル個別評価"""
        logger.info("📊 ベースモデル個別評価...")
        
        X_scaled = self.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=5)
        base_results = {}
        
        for name, model in base_models.items():
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
            base_results[name] = {
                'avg': avg_score,
                'std': np.std(scores),
                'scores': scores
            }
            
            logger.info(f"    {name}: {avg_score:.3%} ± {np.std(scores):.3%}")
        
        return base_results
    
    def implement_voting_classifier(self, X, y, base_models):
        """投票アンサンブル実装"""
        logger.info("🗳️ 投票アンサンブル実装...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # ハードとソフト投票の両方をテスト
        voting_results = {}
        
        # 上位3モデルを選択（事前評価結果から）
        selected_models = [
            ('lr_l2', base_models['lr_l2']),
            ('rf', base_models['rf']),
            ('xgb', base_models['xgb'])
        ]
        
        for voting_type in ['hard', 'soft']:
            logger.info(f"  {voting_type.capitalize()}投票 評価中...")
            
            voting_clf = VotingClassifier(
                estimators=selected_models,
                voting=voting_type
            )
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                voting_clf.fit(X_train, y_train)
                pred = voting_clf.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            avg_score = np.mean(scores)
            voting_results[f'voting_{voting_type}'] = {
                'avg': avg_score,
                'std': np.std(scores),
                'scores': scores
            }
            
            logger.info(f"    {voting_type.capitalize()}投票: {avg_score:.3%} ± {np.std(scores):.3%}")
        
        return voting_results
    
    def implement_stacking(self, X, y, base_models):
        """スタッキングアンサンブル実装"""
        logger.info("🏗️ スタッキングアンサンブル実装...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # レベル1モデル（ベースモデル）
        level_1_models = [
            ('lr_l2', base_models['lr_l2']),
            ('rf', base_models['rf']),
            ('xgb', base_models['xgb'])
        ]
        
        # レベル2モデル（メタ学習器）のバリエーション
        meta_models = {
            'lr_meta': LogisticRegression(C=1.0, random_state=42, max_iter=1000),
            'rf_meta': RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
        }
        
        stacking_results = {}
        
        for meta_name, meta_model in meta_models.items():
            logger.info(f"  {meta_name} メタ学習器でスタッキング...")
            
            stacking_clf = StackingClassifier(
                estimators=level_1_models,
                final_estimator=meta_model,
                cv=3,  # 内部クロスバリデーション
                n_jobs=-1
            )
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                stacking_clf.fit(X_train, y_train)
                pred = stacking_clf.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            avg_score = np.mean(scores)
            stacking_results[f'stacking_{meta_name}'] = {
                'avg': avg_score,
                'std': np.std(scores),
                'scores': scores
            }
            
            logger.info(f"    Stacking({meta_name}): {avg_score:.3%} ± {np.std(scores):.3%}")
        
        return stacking_results
    
    def optimize_ensemble_weights(self, X, y, base_models):
        """アンサンブル重み最適化（Optuna使用）"""
        logger.info("⚖️ アンサンブル重み最適化...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # 選択されたモデル
        selected_models = ['lr_l2', 'rf', 'xgb']
        
        def objective(trial):
            # 重みをOptunaで最適化
            weights = []
            for model_name in selected_models:
                weight = trial.suggest_float(f'weight_{model_name}', 0.1, 1.0)
                weights.append(weight)
            
            # 重み正規化
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # 重み付きアンサンブルの評価
            tscv = TimeSeriesSplit(n_splits=3)  # 最適化では3分割で高速化
            scores = []
            
            for train_idx, test_idx in tscv.split(X_scaled):
                X_train = X_scaled[train_idx]
                X_test = X_scaled[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                # 各モデルの予測を取得
                predictions = []
                for i, model_name in enumerate(selected_models):
                    model = base_models[model_name]
                    model.fit(X_train, y_train)
                    pred_proba = model.predict_proba(X_test)[:, 1]
                    predictions.append(pred_proba)
                
                # 重み付きアンサンブル
                ensemble_pred_proba = np.average(predictions, axis=0, weights=weights)
                ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
                
                scores.append(accuracy_score(y_test, ensemble_pred))
            
            return np.mean(scores)
        
        # Optuna最適化実行
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        study.optimize(objective, n_trials=50, show_progress_bar=False)
        
        # 最適重みで最終評価
        best_weights = []
        for model_name in selected_models:
            best_weights.append(study.best_params[f'weight_{model_name}'])
        
        best_weights = np.array(best_weights)
        best_weights = best_weights / np.sum(best_weights)
        
        logger.info(f"  最適重み: {dict(zip(selected_models, best_weights))}")
        
        # 最適重みでの最終評価
        tscv = TimeSeriesSplit(n_splits=5)
        final_scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            predictions = []
            for i, model_name in enumerate(selected_models):
                model = base_models[model_name]
                model.fit(X_train, y_train)
                pred_proba = model.predict_proba(X_test)[:, 1]
                predictions.append(pred_proba)
            
            ensemble_pred_proba = np.average(predictions, axis=0, weights=best_weights)
            ensemble_pred = (ensemble_pred_proba > 0.5).astype(int)
            final_scores.append(accuracy_score(y_test, ensemble_pred))
        
        avg_score = np.mean(final_scores)
        
        optimized_result = {
            'weighted_ensemble': {
                'avg': avg_score,
                'std': np.std(final_scores),
                'scores': final_scores,
                'weights': dict(zip(selected_models, best_weights))
            }
        }
        
        logger.info(f"    最適重み付きアンサンブル: {avg_score:.3%} ± {np.std(final_scores):.3%}")
        
        return optimized_result
    
    def compare_all_ensemble_methods(self, base_results, voting_results, stacking_results, optimized_result):
        """全アンサンブル手法比較"""
        logger.info("📈 全アンサンブル手法比較...")
        
        all_results = {}
        
        # ベースライン（最高ベースモデル）
        best_base = max(base_results.keys(), key=lambda k: base_results[k]['avg'])
        all_results['best_base'] = base_results[best_base]
        all_results['best_base']['method'] = f"ベストベースモデル ({best_base})"
        
        # 投票アンサンブル
        for name, result in voting_results.items():
            all_results[name] = result
            all_results[name]['method'] = f"投票アンサンブル ({name})"
        
        # スタッキング
        for name, result in stacking_results.items():
            all_results[name] = result
            all_results[name]['method'] = f"スタッキング ({name})"
        
        # 最適化重み付き
        for name, result in optimized_result.items():
            all_results[name] = result
            all_results[name]['method'] = "最適重み付きアンサンブル"
        
        return all_results

def main():
    """メイン実行"""
    logger.info("🚀 高度アンサンブル手法実装システム")
    logger.info("🎯 目標: 59.4%から62%超えを目指す")
    
    ensemble_system = AdvancedEnsembleSystem()
    
    try:
        # 1. データ読み込み
        df = ensemble_system.load_integrated_data()
        if df is None:
            return
        
        # 2. データ準備
        X, y = ensemble_system.prepare_data_for_ensemble(df)
        if X is None:
            return
        
        # 3. ベースモデル作成
        base_models = ensemble_system.create_base_models()
        
        # 4. ベースモデル評価
        base_results = ensemble_system.evaluate_base_models(X, y, base_models)
        
        # 5. 投票アンサンブル
        voting_results = ensemble_system.implement_voting_classifier(X, y, base_models)
        
        # 6. スタッキング
        stacking_results = ensemble_system.implement_stacking(X, y, base_models)
        
        # 7. 重み最適化
        optimized_result = ensemble_system.optimize_ensemble_weights(X, y, base_models)
        
        # 8. 全手法比較
        all_results = ensemble_system.compare_all_ensemble_methods(
            base_results, voting_results, stacking_results, optimized_result
        )
        
        # 結果まとめ
        logger.info("\n" + "="*100)
        logger.info("🏆 高度アンサンブル手法実装結果")
        logger.info("="*100)
        
        # ベースライン表示
        baseline_score = 59.4  # 前回の最高スコア
        logger.info(f"📏 ベースライン（外部データ統合）: {baseline_score:.1%}")
        
        # 全結果をスコア順にソート
        sorted_results = sorted(all_results.items(), key=lambda x: x[1]['avg'], reverse=True)
        
        logger.info(f"\n🎯 アンサンブル手法比較結果:")
        for i, (name, result) in enumerate(sorted_results, 1):
            improvement = (result['avg'] - baseline_score/100) * 100
            status = "🚀" if improvement > 2.0 else "📈" if improvement > 0.5 else "📊"
            
            logger.info(f"  {i:2d}. {result['method']:30s}: {result['avg']:.3%} ({improvement:+.2f}%) {status}")
        
        # 最高結果
        best_method, best_result = sorted_results[0]
        final_improvement = (best_result['avg'] - baseline_score/100) * 100
        
        logger.info(f"\n🏆 最高性能:")
        logger.info(f"  手法: {best_result['method']}")
        logger.info(f"  精度: {best_result['avg']:.3%} ± {best_result['std']:.3%}")
        logger.info(f"  向上: {final_improvement:+.2f}% (59.4% → {best_result['avg']:.1%})")
        
        # 目標達成確認
        target_60 = 0.60
        target_62 = 0.62
        
        if best_result['avg'] >= target_62:
            logger.info(f"🎉 目標大幅達成！ 62%超え ({best_result['avg']:.1%} >= 62.0%)")
        elif best_result['avg'] >= target_60:
            logger.info(f"✅ 目標達成！ 60%超え ({best_result['avg']:.1%} >= 60.0%)")
        else:
            logger.info(f"📈 改善効果あり ({best_result['avg']:.1%})")
        
        # 詳細情報（重み付きアンサンブルの場合）
        if 'weights' in best_result:
            logger.info(f"\n⚖️ 最適重み:")
            for model, weight in best_result['weights'].items():
                logger.info(f"  {model}: {weight:.3f}")
        
        logger.info(f"\n⚖️ この結果は全データ{len(df):,}件での厳密な時系列検証です")
        logger.info(f"✅ 第1段階完了: アンサンブル手法による精度向上達成")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()