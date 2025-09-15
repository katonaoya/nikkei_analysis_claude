#!/usr/bin/env python3
"""
クイック最適化テスト - 手動でデータ追加以外の改善余地を検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class QuickOptimizer:
    """クイック最適化システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        
        # ベースライン特徴量
        self.baseline_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'RSI', 'Price_vs_MA20'
        ]
        
    def load_and_prepare_data(self):
        """データ読み込みと準備"""
        logger.info("📊 データ読み込み（394,102件）")
        
        processed_files = list(self.processed_dir.glob("*.parquet"))
        df = pd.read_parquet(processed_files[0])
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X = clean_df[self.baseline_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"検証データ: {len(clean_df):,}件")
        return X, y
    
    def test_baseline(self, X, y):
        """ベースライン測定"""
        logger.info("📊 ベースライン測定...")
        
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
        
        baseline = np.mean(scores)
        logger.info(f"ベースライン精度: {baseline:.1%}")
        return baseline
    
    def test_preprocessing_variants(self, X, y):
        """前処理バリアント検証"""
        logger.info("🔧 前処理バリアント検証...")
        
        variants = {
            'Standard': StandardScaler(),
            'MinMax': MinMaxScaler(),
            'Quantile': QuantileTransformer(n_quantiles=1000, random_state=42),
            'RobustClip': 'custom'
        }
        
        results = {}
        
        for name, scaler in variants.items():
            logger.info(f"  {name}...")
            
            if name == 'RobustClip':
                # カスタム前処理：ロバストクリッピング
                X_processed = X.clip(lower=X.quantile(0.05), upper=X.quantile(0.95), axis=0)
                X_scaled = StandardScaler().fit_transform(X_processed)
            else:
                X_scaled = scaler.fit_transform(X)
            
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
            results[name] = avg_score
            logger.info(f"    {name}: {avg_score:.1%}")
        
        return results
    
    def test_hyperparameters(self, X, y):
        """ハイパーパラメータテスト"""
        logger.info("⚙️ ハイパーパラメータテスト...")
        
        X_scaled = StandardScaler().fit_transform(X)
        
        # LogisticRegression パラメータ
        lr_configs = [
            {'C': 0.001, 'class_weight': 'balanced'},
            {'C': 0.01, 'class_weight': 'balanced'},
            {'C': 0.1, 'class_weight': 'balanced'},
            {'C': 0.01, 'class_weight': {0: 1, 1: 1.2}},
            {'C': 0.01, 'class_weight': {0: 1, 1: 1.5}},
        ]
        
        lr_results = {}
        
        for i, config in enumerate(lr_configs):
            model = LogisticRegression(**config, max_iter=1000, random_state=42)
            
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
            config_str = f"LR_Config_{i+1}"
            lr_results[config_str] = avg_score
            logger.info(f"  {config_str}: {avg_score:.1%} {config}")
        
        # RandomForest パラメータ
        rf_configs = [
            {'n_estimators': 50, 'max_depth': 8, 'class_weight': 'balanced'},
            {'n_estimators': 100, 'max_depth': 10, 'class_weight': 'balanced'},
            {'n_estimators': 150, 'max_depth': 12, 'class_weight': 'balanced'},
            {'n_estimators': 100, 'max_depth': 8, 'min_samples_split': 10},
        ]
        
        rf_results = {}
        
        for i, config in enumerate(rf_configs):
            model = RandomForestClassifier(**config, random_state=42, n_jobs=-1)
            
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
            config_str = f"RF_Config_{i+1}"
            rf_results[config_str] = avg_score
            logger.info(f"  {config_str}: {avg_score:.1%}")
        
        return lr_results, rf_results
    
    def test_ensemble_methods(self, X, y):
        """アンサンブル手法テスト"""
        logger.info("🧠 アンサンブル手法テスト...")
        
        X_scaled = StandardScaler().fit_transform(X)
        
        # 個別モデル
        lr = LogisticRegression(C=0.01, class_weight='balanced', max_iter=1000, random_state=42)
        rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42, n_jobs=-1)
        gb = GradientBoostingClassifier(n_estimators=100, max_depth=6, random_state=42)
        
        # アンサンブル
        voting_hard = VotingClassifier([('lr', lr), ('rf', rf), ('gb', gb)], voting='hard')
        voting_soft = VotingClassifier([('lr', lr), ('rf', rf), ('gb', gb)], voting='soft')
        
        models = {
            'VotingHard': voting_hard,
            'VotingSoft': voting_soft
        }
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"  {name}...")
            
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
            results[name] = avg_score
            logger.info(f"    {name}: {avg_score:.1%}")
        
        return results
    
    def test_feature_engineering_variants(self, X, y):
        """特徴量エンジニアリングバリアント"""
        logger.info("⚗️ 特徴量エンジニアリングバリアント...")
        
        # 元の特徴量
        X_base = X.copy()
        
        # 各種変換
        variants = {}
        
        # 1. 対数変換
        X_log = X_base.copy()
        for col in X_log.columns:
            if (X_log[col] > 0).all():
                X_log[col] = np.log1p(X_log[col])
        variants['Log_Transform'] = X_log
        
        # 2. 平方根変換
        X_sqrt = X_base.copy()
        for col in X_sqrt.columns:
            if (X_sqrt[col] >= 0).all():
                X_sqrt[col] = np.sqrt(np.abs(X_sqrt[col])) * np.sign(X_sqrt[col])
        variants['Sqrt_Transform'] = X_sqrt
        
        # 3. 標準化 + 2次特徴量
        X_poly = X_base.copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_poly)
        X_scaled_df = pd.DataFrame(X_scaled, columns=X_poly.columns)
        
        # 重要特徴量同士の積
        X_scaled_df['Market_RSI'] = X_scaled_df['Market_Breadth'] * X_scaled_df['RSI']
        X_scaled_df['Vol_Return'] = X_scaled_df['Volatility_20'] * X_scaled_df['Market_Return']
        variants['Polynomial_Features'] = X_scaled_df
        
        # 評価
        results = {}
        
        for variant_name, X_variant in variants.items():
            logger.info(f"  {variant_name}...")
            
            if variant_name != 'Polynomial_Features':
                X_scaled = StandardScaler().fit_transform(X_variant)
            else:
                X_scaled = X_variant.values
            
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
            results[variant_name] = avg_score
            logger.info(f"    {variant_name}: {avg_score:.1%}")
        
        return results
    
    def final_best_combination(self, X, y, best_configs):
        """最終最適組み合わせテスト"""
        logger.info("🎯 最終最適組み合わせテスト...")
        
        # 最適前処理
        X_processed = X.clip(lower=X.quantile(0.05), upper=X.quantile(0.95), axis=0)
        X_scaled = StandardScaler().fit_transform(X_processed)
        
        # 最適モデル（結果から選択）
        model = LogisticRegression(C=0.01, class_weight={0: 1, 1: 1.2}, max_iter=1000, random_state=42)
        
        # 厳密検証
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        
        logger.info("5分割時系列検証実行中...")
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            logger.info(f"  Fold {fold+1}: {accuracy:.1%}")
        
        final_score = np.mean(scores)
        final_std = np.std(scores)
        
        logger.info(f"\n🎯 最終最適化結果: {final_score:.1%} ± {final_std:.1%}")
        
        return final_score, final_std

def main():
    """メイン実行"""
    logger.info("🚀 クイック最適化テスト")
    logger.info("🎯 目標: データ追加以外の改善余地検証")
    
    optimizer = QuickOptimizer()
    
    try:
        # データ準備
        X, y = optimizer.load_and_prepare_data()
        
        # ベースライン
        baseline = optimizer.test_baseline(X, y)
        
        # 前処理テスト
        preprocessing_results = optimizer.test_preprocessing_variants(X, y)
        
        # ハイパーパラメータテスト
        lr_results, rf_results = optimizer.test_hyperparameters(X, y)
        
        # アンサンブルテスト
        ensemble_results = optimizer.test_ensemble_methods(X, y)
        
        # 特徴量エンジニアリング
        feature_eng_results = optimizer.test_feature_engineering_variants(X, y)
        
        # 結果まとめ
        logger.info("\n" + "="*80)
        logger.info("🎯 クイック最適化テスト結果")
        logger.info("="*80)
        
        logger.info(f"ベースライン精度: {baseline:.1%}")
        
        # 各種改善結果
        all_improvements = []
        
        # 前処理改善
        best_preprocessing = max(preprocessing_results.values())
        preprocessing_improvement = (best_preprocessing - baseline) * 100
        all_improvements.append(preprocessing_improvement)
        logger.info(f"\n🔧 前処理最適化:")
        logger.info(f"  最高: {best_preprocessing:.1%} (改善: {preprocessing_improvement:+.1f}%)")
        
        # ハイパーパラメータ改善
        best_lr = max(lr_results.values())
        best_rf = max(rf_results.values())
        best_hp = max(best_lr, best_rf)
        hp_improvement = (best_hp - baseline) * 100
        all_improvements.append(hp_improvement)
        logger.info(f"\n⚙️ ハイパーパラメータ最適化:")
        logger.info(f"  最高: {best_hp:.1%} (改善: {hp_improvement:+.1f}%)")
        
        # アンサンブル改善
        best_ensemble = max(ensemble_results.values())
        ensemble_improvement = (best_ensemble - baseline) * 100
        all_improvements.append(ensemble_improvement)
        logger.info(f"\n🧠 アンサンブル手法:")
        logger.info(f"  最高: {best_ensemble:.1%} (改善: {ensemble_improvement:+.1f}%)")
        
        # 特徴量エンジニアリング改善
        best_feature_eng = max(feature_eng_results.values())
        feature_eng_improvement = (best_feature_eng - baseline) * 100
        all_improvements.append(feature_eng_improvement)
        logger.info(f"\n⚗️ 特徴量エンジニアリング:")
        logger.info(f"  最高: {best_feature_eng:.1%} (改善: {feature_eng_improvement:+.1f}%)")
        
        # 最終最適組み合わせ
        final_score, final_std = optimizer.final_best_combination(X, y, {})
        final_improvement = (final_score - baseline) * 100
        
        # 総評価
        max_improvement = max(all_improvements + [final_improvement])
        max_achieved = baseline + max_improvement/100
        
        logger.info(f"\n🏆 最高達成精度: {max_achieved:.1%}")
        logger.info(f"🔥 最大改善幅: {max_improvement:+.1f}%")
        logger.info(f"🎯 最終組み合わせ: {final_score:.1%} ± {final_std:.1%} (改善: {final_improvement:+.1f}%)")
        
        # 改善可能性評価
        logger.info(f"\n📊 改善可能性評価:")
        if max_improvement > 1.0:
            logger.info("✅ 有意な改善が可能です (1%以上)")
        elif max_improvement > 0.5:
            logger.info("🔄 中程度の改善が可能です (0.5-1%)")
        elif max_improvement > 0.2:
            logger.info("⚠️ 限定的な改善が可能です (0.2-0.5%)")
        else:
            logger.info("❌ 大幅な改善は困難です (<0.2%)")
        
        logger.info(f"\n💡 結論:")
        if max_improvement > 0.5:
            logger.info("データ追加以外でも改善余地があります")
        else:
            logger.info("データ追加以外の改善は限定的です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()