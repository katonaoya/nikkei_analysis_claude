#!/usr/bin/env python3
"""
系統的最適化分析 - データ追加以外の改善手法
パラメータ調整・ロジック変更・前処理改善等の包括的検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, QuantileTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class SystematicOptimizer:
    """系統的最適化システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # ベースライン特徴量（現在最高の組み合わせ）
        self.baseline_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'RSI', 'Price_vs_MA20'
        ]
        
    def load_full_data(self):
        """全データ読み込み"""
        logger.info("📊 全データ読み込み（394,102件）")
        
        processed_files = list(self.processed_dir.glob("*.parquet"))
        if not processed_files:
            logger.error("❌ 処理済みデータが見つかりません")
            return None
            
        df = pd.read_parquet(processed_files[0])
        logger.info(f"✅ 全データ読み込み完了: {len(df):,}件")
        
        return df
    
    def prepare_data(self, df):
        """データ準備"""
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[self.baseline_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"検証データ: {len(clean_df):,}件")
        logger.info(f"特徴量: {self.baseline_features}")
        
        return X, y, clean_df
    
    def baseline_evaluation(self, X, y):
        """ベースライン評価"""
        logger.info("📊 ベースライン評価...")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42)
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        for train_idx, test_idx in tscv.split(X_scaled):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            scores.append(accuracy_score(y_test, pred))
        
        baseline_score = np.mean(scores)
        logger.info(f"ベースライン精度: {baseline_score:.1%}")
        
        return baseline_score
    
    def preprocessing_optimization(self, X, y):
        """前処理最適化"""
        logger.info("🔧 前処理最適化...")
        
        # 各種スケーラーのテスト
        scalers = {
            'StandardScaler': StandardScaler(),
            'MinMaxScaler': MinMaxScaler(),
            'RobustScaler': RobustScaler(),
            'QuantileTransformer': QuantileTransformer(n_quantiles=1000, random_state=42),
            'None': None
        }
        
        # 外れ値処理パターン
        outlier_methods = {
            'None': lambda x: x,
            'Clip_99': lambda x: x.clip(lower=x.quantile(0.01), upper=x.quantile(0.99), axis=0),
            'Clip_95': lambda x: x.clip(lower=x.quantile(0.025), upper=x.quantile(0.975), axis=0),
            'Winsorize': lambda x: x.clip(lower=x.quantile(0.05), upper=x.quantile(0.95), axis=0)
        }
        
        preprocessing_results = {}
        
        for outlier_name, outlier_func in outlier_methods.items():
            for scaler_name, scaler in scalers.items():
                logger.info(f"  {outlier_name} + {scaler_name}...")
                
                # 外れ値処理
                X_processed = outlier_func(X.copy())
                
                # スケーリング
                if scaler is not None:
                    X_scaled = scaler.fit_transform(X_processed)
                else:
                    X_scaled = X_processed.values
                
                # 評価
                model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42)
                tscv = TimeSeriesSplit(n_splits=3)
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
                combo_name = f"{outlier_name}+{scaler_name}"
                preprocessing_results[combo_name] = avg_score
                
                logger.info(f"    {combo_name}: {avg_score:.1%}")
        
        # 最高前処理
        best_preprocessing = max(preprocessing_results.keys(), key=lambda k: preprocessing_results[k])
        best_preprocessing_score = preprocessing_results[best_preprocessing]
        
        logger.info(f"\n🏆 最高前処理: {best_preprocessing} ({best_preprocessing_score:.1%})")
        
        return preprocessing_results, best_preprocessing
    
    def hyperparameter_optimization(self, X, y, best_preprocessing):
        """ハイパーパラメータ最適化"""
        logger.info("⚙️ ハイパーパラメータ最適化...")
        
        # 最適前処理を適用
        outlier_method, scaler_method = best_preprocessing.split('+')
        
        # 外れ値処理
        if outlier_method == 'Clip_99':
            X_processed = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=0)
        elif outlier_method == 'Clip_95':
            X_processed = X.clip(lower=X.quantile(0.025), upper=X.quantile(0.975), axis=0)
        elif outlier_method == 'Winsorize':
            X_processed = X.clip(lower=X.quantile(0.05), upper=X.quantile(0.95), axis=0)
        else:
            X_processed = X.copy()
        
        # スケーリング
        if scaler_method == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_method == 'RobustScaler':
            scaler = RobustScaler()
        elif scaler_method == 'QuantileTransformer':
            scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
        else:
            scaler = None
        
        if scaler is not None:
            X_scaled = scaler.fit_transform(X_processed)
        else:
            X_scaled = X_processed.values
        
        # モデル別ハイパーパラメータ最適化
        hyperparameter_results = {}
        
        # 1. LogisticRegression
        logger.info("  LogisticRegression最適化...")
        lr_param_grid = {
            'C': [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0],
            'class_weight': ['balanced', {0: 1, 1: 1.1}, {0: 1, 1: 1.2}, {0: 1, 1: 1.3}, {0: 1, 1: 1.5}],
            'solver': ['liblinear', 'lbfgs'],
            'penalty': ['l1', 'l2']
        }
        
        # solverとpenaltyの互換性チェック
        compatible_params = []
        for params in self._generate_param_combinations(lr_param_grid):
            if params['solver'] == 'liblinear' or params['penalty'] == 'l2':
                compatible_params.append(params)
        
        best_lr_score = 0
        best_lr_params = None
        
        # 効率的なグリッドサーチ（ランダムサンプリング）
        import random
        random.seed(42)
        sampled_params = random.sample(compatible_params, min(20, len(compatible_params)))
        
        for params in sampled_params:
            try:
                model = LogisticRegression(**params, max_iter=2000, random_state=42)
                tscv = TimeSeriesSplit(n_splits=3)
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
                
                if avg_score > best_lr_score:
                    best_lr_score = avg_score
                    best_lr_params = params
                    
            except Exception as e:
                continue
        
        hyperparameter_results['LogisticRegression'] = {
            'score': best_lr_score,
            'params': best_lr_params
        }
        
        logger.info(f"    最適LR: {best_lr_score:.1%} {best_lr_params}")
        
        # 2. RandomForest
        logger.info("  RandomForest最適化...")
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 8, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'class_weight': ['balanced', 'balanced_subsample']
        }
        
        rf_combinations = list(self._generate_param_combinations(rf_param_grid))
        sampled_rf_params = random.sample(rf_combinations, min(15, len(rf_combinations)))
        
        best_rf_score = 0
        best_rf_params = None
        
        for params in sampled_rf_params:
            try:
                model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
                tscv = TimeSeriesSplit(n_splits=3)
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
                
                if avg_score > best_rf_score:
                    best_rf_score = avg_score
                    best_rf_params = params
                    
            except Exception as e:
                continue
        
        hyperparameter_results['RandomForest'] = {
            'score': best_rf_score,
            'params': best_rf_params
        }
        
        logger.info(f"    最適RF: {best_rf_score:.1%} {best_rf_params}")
        
        return hyperparameter_results
    
    def _generate_param_combinations(self, param_grid):
        """パラメータ組み合わせ生成"""
        import itertools
        keys = param_grid.keys()
        values = param_grid.values()
        combinations = list(itertools.product(*values))
        return [dict(zip(keys, combo)) for combo in combinations]
    
    def advanced_modeling_techniques(self, X, y, best_preprocessing):
        """高度なモデリング手法"""
        logger.info("🧠 高度なモデリング手法...")
        
        # データ前処理
        outlier_method, scaler_method = best_preprocessing.split('+')
        X_processed = self._apply_preprocessing(X, outlier_method, scaler_method)
        
        advanced_results = {}
        
        # 1. アンサンブル手法
        logger.info("  アンサンブル手法...")
        
        base_models = [
            ('lr', LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)),
            ('gb', GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42))
        ]
        
        voting_clf = VotingClassifier(estimators=base_models, voting='soft')
        
        # アンサンブル評価
        tscv = TimeSeriesSplit(n_splits=3)
        ensemble_scores = []
        
        for train_idx, test_idx in tscv.split(X_processed):
            X_train = X_processed[train_idx]
            X_test = X_processed[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            voting_clf.fit(X_train, y_train)
            pred = voting_clf.predict(X_test)
            ensemble_scores.append(accuracy_score(y_test, pred))
        
        advanced_results['VotingEnsemble'] = np.mean(ensemble_scores)
        logger.info(f"    VotingEnsemble: {np.mean(ensemble_scores):.1%}")
        
        # 2. Neural Network
        logger.info("  Neural Network...")
        try:
            mlp = MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            
            nn_scores = []
            for train_idx, test_idx in tscv.split(X_processed):
                X_train = X_processed[train_idx]
                X_test = X_processed[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                mlp.fit(X_train, y_train)
                pred = mlp.predict(X_test)
                nn_scores.append(accuracy_score(y_test, pred))
            
            advanced_results['NeuralNetwork'] = np.mean(nn_scores)
            logger.info(f"    NeuralNetwork: {np.mean(nn_scores):.1%}")
            
        except Exception as e:
            logger.info(f"    NeuralNetwork: スキップ ({str(e)[:50]})")
        
        # 3. Gradient Boosting詳細調整
        logger.info("  Gradient Boosting...")
        gb_params = [
            {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1},
            {'n_estimators': 150, 'max_depth': 6, 'learning_rate': 0.05},
            {'n_estimators': 200, 'max_depth': 8, 'learning_rate': 0.03}
        ]
        
        best_gb_score = 0
        for params in gb_params:
            gb = GradientBoostingClassifier(**params, random_state=42)
            gb_scores = []
            
            for train_idx, test_idx in tscv.split(X_processed):
                X_train = X_processed[train_idx]
                X_test = X_processed[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                gb.fit(X_train, y_train)
                pred = gb.predict(X_test)
                gb_scores.append(accuracy_score(y_test, pred))
            
            avg_gb_score = np.mean(gb_scores)
            if avg_gb_score > best_gb_score:
                best_gb_score = avg_gb_score
        
        advanced_results['GradientBoosting'] = best_gb_score
        logger.info(f"    GradientBoosting: {best_gb_score:.1%}")
        
        return advanced_results
    
    def _apply_preprocessing(self, X, outlier_method, scaler_method):
        """前処理適用"""
        # 外れ値処理
        if outlier_method == 'Clip_99':
            X_processed = X.clip(lower=X.quantile(0.01), upper=X.quantile(0.99), axis=0)
        elif outlier_method == 'Clip_95':
            X_processed = X.clip(lower=X.quantile(0.025), upper=X.quantile(0.975), axis=0)
        elif outlier_method == 'Winsorize':
            X_processed = X.clip(lower=X.quantile(0.05), upper=X.quantile(0.95), axis=0)
        else:
            X_processed = X.copy()
        
        # スケーリング
        if scaler_method == 'StandardScaler':
            scaler = StandardScaler()
        elif scaler_method == 'MinMaxScaler':
            scaler = MinMaxScaler()
        elif scaler_method == 'RobustScaler':
            scaler = RobustScaler()
        elif scaler_method == 'QuantileTransformer':
            scaler = QuantileTransformer(n_quantiles=1000, random_state=42)
        else:
            scaler = None
        
        if scaler is not None:
            return scaler.fit_transform(X_processed)
        else:
            return X_processed.values
    
    def class_imbalance_techniques(self, X, y, best_preprocessing):
        """クラス不均衡対応技術"""
        logger.info("⚖️ クラス不均衡対応技術...")
        
        X_processed = self._apply_preprocessing(X, *best_preprocessing.split('+'))
        
        # クラス分布確認
        class_counts = y.value_counts()
        logger.info(f"クラス分布: {class_counts.to_dict()}")
        
        imbalance_results = {}
        
        # 1. 各種class_weight設定
        class_weights = [
            'balanced',
            {0: 1, 1: 1.1},
            {0: 1, 1: 1.2},
            {0: 1, 1: 1.3},
            {0: 1, 1: 1.5},
            {0: 1, 1: 2.0}
        ]
        
        for weight in class_weights:
            model = LogisticRegression(C=0.01, class_weight=weight, max_iter=1000, random_state=42)
            
            tscv = TimeSeriesSplit(n_splits=3)
            scores = []
            
            for train_idx, test_idx in tscv.split(X_processed):
                X_train = X_processed[train_idx]
                X_test = X_processed[test_idx]
                y_train = y.iloc[train_idx]
                y_test = y.iloc[test_idx]
                
                model.fit(X_train, y_train)
                pred = model.predict(X_test)
                scores.append(accuracy_score(y_test, pred))
            
            avg_score = np.mean(scores)
            weight_str = str(weight) if isinstance(weight, dict) else weight
            imbalance_results[f'weight_{weight_str}'] = avg_score
            
            logger.info(f"    class_weight={weight_str}: {avg_score:.1%}")
        
        return imbalance_results
    
    def final_rigorous_validation(self, X, y, best_config):
        """最終厳密検証"""
        logger.info("🎯 最終厳密検証...")
        
        # 最適設定適用
        X_processed = self._apply_preprocessing(X, *best_config['preprocessing'].split('+'))
        
        # 最適モデル構築
        if best_config['model_type'] == 'LogisticRegression':
            model = LogisticRegression(**best_config['params'], max_iter=2000, random_state=42)
        elif best_config['model_type'] == 'RandomForest':
            model = RandomForestClassifier(**best_config['params'], random_state=42, n_jobs=-1)
        else:
            model = LogisticRegression(C=0.01, class_weight='balanced', max_iter=2000, random_state=42)
        
        # 5分割厳密検証
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        precision_scores = []
        recall_scores = []
        f1_scores = []
        
        logger.info("5分割時系列検証実行中...")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_processed)):
            X_train = X_processed[train_idx]
            X_test = X_processed[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            
            scores.append(accuracy_score(y_test, pred))
            precision_scores.append(precision_score(y_test, pred))
            recall_scores.append(recall_score(y_test, pred))
            f1_scores.append(f1_score(y_test, pred))
            
            logger.info(f"  Fold {fold+1}: Acc={scores[-1]:.1%}, Prec={precision_scores[-1]:.1%}, Rec={recall_scores[-1]:.1%}")
        
        final_metrics = {
            'accuracy': {'mean': np.mean(scores), 'std': np.std(scores)},
            'precision': {'mean': np.mean(precision_scores), 'std': np.std(precision_scores)},
            'recall': {'mean': np.mean(recall_scores), 'std': np.std(recall_scores)},
            'f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores)}
        }
        
        logger.info(f"\n🎯 最終結果:")
        logger.info(f"精度: {final_metrics['accuracy']['mean']:.1%} ± {final_metrics['accuracy']['std']:.1%}")
        logger.info(f"適合率: {final_metrics['precision']['mean']:.1%} ± {final_metrics['precision']['std']:.1%}")
        logger.info(f"再現率: {final_metrics['recall']['mean']:.1%} ± {final_metrics['recall']['std']:.1%}")
        logger.info(f"F1スコア: {final_metrics['f1']['mean']:.1%} ± {final_metrics['f1']['std']:.1%}")
        
        return final_metrics

def main():
    """メイン実行"""
    logger.info("🚀 系統的最適化分析 - データ追加以外の改善手法")
    logger.info("🎯 目標: パラメータ・ロジック・前処理による精度向上")
    
    optimizer = SystematicOptimizer()
    
    try:
        # 1. データ準備
        df = optimizer.load_full_data()
        if df is None:
            return
        
        X, y, clean_df = optimizer.prepare_data(df)
        
        # 2. ベースライン評価
        baseline_score = optimizer.baseline_evaluation(X, y)
        
        # 3. 前処理最適化
        preprocessing_results, best_preprocessing = optimizer.preprocessing_optimization(X, y)
        
        # 4. ハイパーパラメータ最適化
        hyperparameter_results = optimizer.hyperparameter_optimization(X, y, best_preprocessing)
        
        # 5. 高度なモデリング手法
        advanced_results = optimizer.advanced_modeling_techniques(X, y, best_preprocessing)
        
        # 6. クラス不均衡対応
        imbalance_results = optimizer.class_imbalance_techniques(X, y, best_preprocessing)
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎯 系統的最適化分析結果")
        logger.info("="*80)
        
        logger.info(f"ベースライン精度: {baseline_score:.1%}")
        
        # 前処理結果
        logger.info(f"\n🔧 最適前処理: {best_preprocessing} ({preprocessing_results[best_preprocessing]:.1%})")
        logger.info(f"前処理による改善: {(preprocessing_results[best_preprocessing] - baseline_score)*100:+.1f}%")
        
        # ハイパーパラメータ結果
        logger.info(f"\n⚙️ ハイパーパラメータ最適化結果:")
        best_hp_score = 0
        best_hp_model = None
        for model_name, result in hyperparameter_results.items():
            logger.info(f"  {model_name}: {result['score']:.1%}")
            if result['score'] > best_hp_score:
                best_hp_score = result['score']
                best_hp_model = model_name
        logger.info(f"ハイパーパラメータによる改善: {(best_hp_score - baseline_score)*100:+.1f}%")
        
        # 高度手法結果
        logger.info(f"\n🧠 高度なモデリング手法結果:")
        best_advanced_score = 0
        for method, score in advanced_results.items():
            logger.info(f"  {method}: {score:.1%}")
            best_advanced_score = max(best_advanced_score, score)
        logger.info(f"高度手法による改善: {(best_advanced_score - baseline_score)*100:+.1f}%")
        
        # クラス不均衡結果
        logger.info(f"\n⚖️ クラス不均衡対応結果:")
        best_imbalance_score = max(imbalance_results.values())
        best_imbalance_method = max(imbalance_results.keys(), key=lambda k: imbalance_results[k])
        logger.info(f"  最高: {best_imbalance_method}: {best_imbalance_score:.1%}")
        logger.info(f"クラス不均衡対応による改善: {(best_imbalance_score - baseline_score)*100:+.1f}%")
        
        # 全体の最高精度
        all_scores = [
            baseline_score,
            preprocessing_results[best_preprocessing],
            best_hp_score,
            best_advanced_score,
            best_imbalance_score
        ]
        
        max_achieved = max(all_scores)
        improvement = (max_achieved - baseline_score) * 100
        
        logger.info(f"\n🏆 最高達成精度: {max_achieved:.1%}")
        logger.info(f"総改善幅: {improvement:+.1f}%")
        
        # 最適構成特定
        best_config = {
            'preprocessing': best_preprocessing,
            'model_type': best_hp_model,
            'params': hyperparameter_results[best_hp_model]['params'] if best_hp_model else {},
            'score': max_achieved
        }
        
        # 最終検証
        final_metrics = optimizer.final_rigorous_validation(X, y, best_config)
        
        logger.info(f"\n📊 改善可能性評価:")
        if improvement > 0.5:
            logger.info("✅ 有意な改善が可能です")
        elif improvement > 0.2:
            logger.info("🔄 限定的な改善が可能です")
        else:
            logger.info("⚠️ 大幅な改善は困難です")
        
        logger.info(f"\n⚠️ この結果は394,102件の全データでの検証です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()