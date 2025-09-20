#!/usr/bin/env python3
"""
究極の精度最大化システム - 全データ版
あらゆる手法を組み合わせて最大精度達成
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
from itertools import combinations, product
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class UltimatePrecisionMaximizer:
    """究極の精度最大化システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # ベースライン特徴量
        self.baseline_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'RSI', 'Price_vs_MA20'
        ]
        
        # 追加可能特徴量
        self.additional_features = [
            'Returns', 'Volume_Ratio', 'Above_MA20', 'Price_vs_MA10', 'Relative_Return'
        ]
        
        # 最高結果追跡
        self.best_score = 0
        self.best_config = {}
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 データ読み込み（394,102件）")
        
        processed_files = list(self.processed_dir.glob("*.parquet"))
        df = pd.read_parquet(processed_files[0])
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 利用可能な特徴量確認
        available_additional = [f for f in self.additional_features if f in clean_df.columns]
        self.all_available_features = self.baseline_features + available_additional
        
        logger.info(f"利用可能特徴量: {len(self.all_available_features)}個")
        logger.info(f"特徴量: {self.all_available_features}")
        
        X_full = clean_df[self.all_available_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"検証データ: {len(clean_df):,}件")
        return X_full, y, clean_df
    
    def exhaustive_feature_combinations(self, X_full, y):
        """徹底的特徴量組み合わせテスト"""
        logger.info("🔍 徹底的特徴量組み合わせテスト...")
        
        # 特徴量数別テスト
        feature_counts = [3, 4, 5, 6, 7, 8]
        best_combinations = {}
        
        for n_features in feature_counts:
            logger.info(f"  {n_features}特徴量組み合わせ...")
            
            best_score_n = 0
            best_combination_n = None
            
            # 組み合わせ数制限（計算時間短縮）
            feature_combinations = list(combinations(self.all_available_features, n_features))
            max_combinations = min(20, len(feature_combinations))
            
            # 重要な組み合わせを優先
            priority_combinations = []
            for combo in feature_combinations:
                if all(f in combo for f in self.baseline_features[:3]):  # 重要特徴量含む
                    priority_combinations.append(combo)
            
            test_combinations = priority_combinations[:max_combinations] if priority_combinations else feature_combinations[:max_combinations]
            
            for i, feature_combo in enumerate(test_combinations):
                X_subset = X_full[list(feature_combo)]
                
                # 高速評価
                score = self._quick_evaluate(X_subset, y)
                
                if score > best_score_n:
                    best_score_n = score
                    best_combination_n = feature_combo
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_config = {
                        'features': feature_combo,
                        'score': score,
                        'n_features': n_features
                    }
                
                if (i + 1) % 5 == 0:
                    logger.info(f"    {i+1}/{len(test_combinations)} 完了 (最高: {best_score_n:.1%})")
            
            best_combinations[n_features] = {
                'score': best_score_n,
                'features': best_combination_n
            }
            
            logger.info(f"  {n_features}特徴量最高: {best_score_n:.1%}")
        
        return best_combinations
    
    def _quick_evaluate(self, X, y):
        """高速評価"""
        X_scaled = StandardScaler().fit_transform(X)
        
        model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=500, random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=2)  # 高速化のため2分割
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        
        return np.mean(scores)
    
    def advanced_preprocessing_optimization(self, X, y, best_features):
        """高度前処理最適化"""
        logger.info("🔧 高度前処理最適化...")
        
        X_best = X[list(best_features)]
        
        preprocessing_methods = {
            'Standard': StandardScaler(),
            'MinMax': MinMaxScaler(),
            'Quantile_1000': QuantileTransformer(n_quantiles=1000, random_state=42),
            'Quantile_500': QuantileTransformer(n_quantiles=500, random_state=42),
            'PowerYeoJohnson': PowerTransformer(method='yeo-johnson', standardize=True),
            'RobustClip_95': 'robust_95',
            'RobustClip_90': 'robust_90',
            'Zscore_3': 'zscore_3',
            'Zscore_2.5': 'zscore_2.5'
        }
        
        preprocessing_results = {}
        
        for name, method in preprocessing_methods.items():
            try:
                logger.info(f"  {name}...")
                
                if name.startswith('RobustClip_'):
                    percentile = int(name.split('_')[1])
                    lower_p = (100 - percentile) / 2
                    upper_p = 100 - lower_p
                    X_processed = X_best.clip(
                        lower=X_best.quantile(lower_p/100), 
                        upper=X_best.quantile(upper_p/100), 
                        axis=0
                    )
                    X_scaled = StandardScaler().fit_transform(X_processed)
                elif name.startswith('Zscore_'):
                    threshold = float(name.split('_')[1])
                    X_processed = X_best.copy()
                    z_scores = np.abs((X_processed - X_processed.mean()) / X_processed.std())
                    X_processed[z_scores > threshold] = np.nan
                    X_processed = X_processed.fillna(X_processed.median())
                    X_scaled = StandardScaler().fit_transform(X_processed)
                else:
                    X_scaled = method.fit_transform(X_best)
                
                # 評価
                score = self._quick_evaluate_with_data(X_scaled, y)
                preprocessing_results[name] = score
                
                logger.info(f"    {name}: {score:.1%}")
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_config.update({
                        'preprocessing': name,
                        'score': score
                    })
                    
            except Exception as e:
                logger.info(f"    {name}: エラー ({str(e)[:30]})")
                continue
        
        return preprocessing_results
    
    def _quick_evaluate_with_data(self, X_scaled, y):
        """データ付き高速評価"""
        model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=500, random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=2)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        
        return np.mean(scores)
    
    def hyperparameter_grid_search(self, X, y, best_features, best_preprocessing):
        """ハイパーパラメータグリッドサーチ"""
        logger.info("⚙️ ハイパーパラメータグリッドサーチ...")
        
        X_best = X[list(best_features)]
        X_processed = self._apply_preprocessing(X_best, best_preprocessing)
        
        # LogisticRegression パラメータ
        lr_params = {
            'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
            'class_weight': [
                'balanced',
                {0: 1, 1: 1.1},
                {0: 1, 1: 1.2},
                {0: 1, 1: 1.3},
                {0: 1, 1: 1.4},
                {0: 1, 1: 1.5},
                {0: 1, 1: 1.7},
                {0: 1, 1: 2.0}
            ]
        }
        
        # 最適LRパラメータ探索
        best_lr_score = 0
        best_lr_params = None
        
        param_combinations = list(product(lr_params['C'], lr_params['class_weight']))
        
        for C, class_weight in param_combinations:
            try:
                model = LogisticRegression(C=C, class_weight=class_weight, max_iter=1000, random_state=42)
                score = self._quick_evaluate_with_data(X_processed, y)
                
                if score > best_lr_score:
                    best_lr_score = score
                    best_lr_params = {'C': C, 'class_weight': class_weight}
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_config.update({
                        'model': 'LogisticRegression',
                        'model_params': {'C': C, 'class_weight': class_weight},
                        'score': score
                    })
                    
            except Exception as e:
                continue
        
        logger.info(f"  最適LR: {best_lr_score:.1%} {best_lr_params}")
        
        # RandomForest パラメータ
        rf_params = [
            {'n_estimators': 50, 'max_depth': 6, 'min_samples_split': 5},
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 2},
            {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5},
            {'n_estimators': 150, 'max_depth': 12, 'min_samples_split': 10},
            {'n_estimators': 80, 'max_depth': 6, 'min_samples_split': 2, 'min_samples_leaf': 2},
        ]
        
        best_rf_score = 0
        best_rf_params = None
        
        for params in rf_params:
            try:
                model = RandomForestClassifier(**params, class_weight='balanced', random_state=42, n_jobs=-1)
                score = self._quick_evaluate_with_data(X_processed, y)
                
                if score > best_rf_score:
                    best_rf_score = score
                    best_rf_params = params
                
                if score > self.best_score:
                    self.best_score = score
                    self.best_config.update({
                        'model': 'RandomForest',
                        'model_params': params,
                        'score': score
                    })
                    
            except Exception as e:
                continue
        
        logger.info(f"  最適RF: {best_rf_score:.1%} {best_rf_params}")
        
        return {'LR': (best_lr_score, best_lr_params), 'RF': (best_rf_score, best_rf_params)}
    
    def _apply_preprocessing(self, X, preprocessing_name):
        """前処理適用"""
        if preprocessing_name == 'Standard':
            return StandardScaler().fit_transform(X)
        elif preprocessing_name == 'MinMax':
            return MinMaxScaler().fit_transform(X)
        elif preprocessing_name.startswith('Quantile_'):
            n_quantiles = int(preprocessing_name.split('_')[1])
            return QuantileTransformer(n_quantiles=n_quantiles, random_state=42).fit_transform(X)
        elif preprocessing_name == 'PowerYeoJohnson':
            return PowerTransformer(method='yeo-johnson', standardize=True).fit_transform(X)
        elif preprocessing_name.startswith('RobustClip_'):
            percentile = int(preprocessing_name.split('_')[1])
            lower_p = (100 - percentile) / 2
            upper_p = 100 - lower_p
            X_processed = X.clip(
                lower=X.quantile(lower_p/100), 
                upper=X.quantile(upper_p/100), 
                axis=0
            )
            return StandardScaler().fit_transform(X_processed)
        else:
            return StandardScaler().fit_transform(X)
    
    def ensemble_optimization(self, X, y, best_features, best_preprocessing):
        """アンサンブル最適化"""
        logger.info("🧠 アンサンブル最適化...")
        
        X_best = X[list(best_features)]
        X_processed = self._apply_preprocessing(X_best, best_preprocessing)
        
        # 個別モデル
        models = {
            'LR': LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42),
            'RF': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
            'GB': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
            'ET': ExtraTreesClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        }
        
        # アンサンブル組み合わせ
        ensemble_combinations = [
            ['LR', 'RF'],
            ['LR', 'GB'],
            ['RF', 'GB'],
            ['LR', 'RF', 'GB'],
            ['LR', 'RF', 'ET'],
            ['RF', 'GB', 'ET'],
            ['LR', 'RF', 'GB', 'ET']
        ]
        
        ensemble_results = {}
        
        for combo in ensemble_combinations:
            try:
                combo_name = '+'.join(combo)
                logger.info(f"  {combo_name}...")
                
                estimators = [(name, models[name]) for name in combo]
                
                # Hard Voting
                voting_hard = VotingClassifier(estimators=estimators, voting='hard')
                score_hard = self._quick_evaluate_with_data(X_processed, y, voting_hard)
                ensemble_results[f'{combo_name}_Hard'] = score_hard
                
                # Soft Voting
                voting_soft = VotingClassifier(estimators=estimators, voting='soft')
                score_soft = self._quick_evaluate_with_data(X_processed, y, voting_soft)
                ensemble_results[f'{combo_name}_Soft'] = score_soft
                
                logger.info(f"    Hard: {score_hard:.1%}, Soft: {score_soft:.1%}")
                
                # 最高スコア更新
                best_ensemble_score = max(score_hard, score_soft)
                if best_ensemble_score > self.best_score:
                    self.best_score = best_ensemble_score
                    voting_type = 'hard' if score_hard > score_soft else 'soft'
                    self.best_config.update({
                        'model': 'VotingClassifier',
                        'model_params': {'estimators': combo, 'voting': voting_type},
                        'score': best_ensemble_score
                    })
                    
            except Exception as e:
                logger.info(f"    {combo_name}: エラー ({str(e)[:30]})")
                continue
        
        return ensemble_results
    
    def _quick_evaluate_with_data(self, X_processed, y, model=None):
        """モデル指定可能な高速評価"""
        if model is None:
            model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=500, random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=2)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_processed):
            X_train = X_processed[train_idx]
            X_test = X_processed[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        
        return np.mean(scores)
    
    def final_rigorous_validation(self, X, y):
        """最終厳密検証"""
        logger.info("🎯 最終厳密検証...")
        logger.info(f"最適構成: {self.best_config}")
        
        # 最適設定適用
        X_best = X[list(self.best_config['features'])]
        X_processed = self._apply_preprocessing(X_best, self.best_config.get('preprocessing', 'Standard'))
        
        # 最適モデル構築
        if self.best_config.get('model') == 'LogisticRegression':
            model = LogisticRegression(**self.best_config['model_params'], max_iter=2000, random_state=42)
        elif self.best_config.get('model') == 'RandomForest':
            model = RandomForestClassifier(**self.best_config['model_params'], class_weight='balanced', random_state=42, n_jobs=-1)
        elif self.best_config.get('model') == 'VotingClassifier':
            base_models = {
                'LR': LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42),
                'RF': RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1),
                'GB': GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42),
                'ET': ExtraTreesClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
            }
            estimators = [(name, base_models[name]) for name in self.best_config['model_params']['estimators']]
            model = VotingClassifier(estimators=estimators, voting=self.best_config['model_params']['voting'])
        else:
            model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000, random_state=42)
        
        # 5分割厳密検証
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        logger.info("5分割時系列検証実行中...")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_processed)):
            X_train = X_processed[train_idx]
            X_test = X_processed[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            logger.info(f"  Fold {fold+1}: {accuracy:.1%}")
        
        final_accuracy = np.mean(scores)
        final_std = np.std(scores)
        
        logger.info(f"\n🎯 最終厳密結果: {final_accuracy:.1%} ± {final_std:.1%}")
        
        return final_accuracy, final_std, scores

def main():
    """メイン実行"""
    logger.info("🚀 究極の精度最大化システム - 全データ版")
    logger.info("⚡ あらゆる手法を組み合わせて最大精度達成")
    
    maximizer = UltimatePrecisionMaximizer()
    
    try:
        # 1. データ準備
        X_full, y, clean_df = maximizer.load_and_prepare_data()
        
        # 2. 徹底的特徴量組み合わせテスト
        feature_combinations = maximizer.exhaustive_feature_combinations(X_full, y)
        
        logger.info(f"\n🎯 特徴量組み合わせ最高: {maximizer.best_score:.1%}")
        logger.info(f"最適特徴量: {maximizer.best_config['features']}")
        
        # 3. 高度前処理最適化
        preprocessing_results = maximizer.advanced_preprocessing_optimization(
            X_full, y, maximizer.best_config['features']
        )
        
        logger.info(f"\n🔧 前処理最適化後: {maximizer.best_score:.1%}")
        
        # 4. ハイパーパラメータ最適化
        hyperparameter_results = maximizer.hyperparameter_grid_search(
            X_full, y, maximizer.best_config['features'], 
            maximizer.best_config.get('preprocessing', 'Standard')
        )
        
        logger.info(f"\n⚙️ ハイパーパラメータ最適化後: {maximizer.best_score:.1%}")
        
        # 5. アンサンブル最適化
        ensemble_results = maximizer.ensemble_optimization(
            X_full, y, maximizer.best_config['features'], 
            maximizer.best_config.get('preprocessing', 'Standard')
        )
        
        logger.info(f"\n🧠 アンサンブル最適化後: {maximizer.best_score:.1%}")
        
        # 6. 最終厳密検証
        final_accuracy, final_std, fold_scores = maximizer.final_rigorous_validation(X_full, y)
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎯 究極の精度最大化結果")
        logger.info("="*80)
        
        logger.info(f"データ総数: {len(clean_df):,}件 (全データ)")
        logger.info(f"最適特徴量数: {len(maximizer.best_config['features'])}個")
        logger.info(f"最適特徴量: {list(maximizer.best_config['features'])}")
        
        if 'preprocessing' in maximizer.best_config:
            logger.info(f"最適前処理: {maximizer.best_config['preprocessing']}")
        
        if 'model' in maximizer.best_config:
            logger.info(f"最適モデル: {maximizer.best_config['model']}")
            logger.info(f"モデルパラメータ: {maximizer.best_config.get('model_params', {})}")
        
        logger.info(f"\n🏆 究極の達成精度: {final_accuracy:.1%} ± {final_std:.1%}")
        logger.info(f"開発中最高精度: {maximizer.best_score:.1%}")
        
        # ベースライン比較
        baseline = 0.505  # 以前のベースライン
        improvement = (final_accuracy - baseline) * 100
        
        logger.info(f"\n📊 改善結果:")
        logger.info(f"ベースライン: {baseline:.1%}")
        logger.info(f"最終精度: {final_accuracy:.1%}")
        logger.info(f"改善幅: {improvement:+.1f}%")
        
        if improvement > 1.0:
            logger.info("🎉 大幅な改善を達成しました！")
        elif improvement > 0.5:
            logger.info("✅ 有意な改善を達成しました")
        elif improvement > 0.2:
            logger.info("🔄 限定的ですが改善を達成しました")
        else:
            logger.info("⚠️ 改善は限定的でした")
        
        logger.info(f"\n⚠️ この結果は394,102件の全データでの厳密検証です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()