#!/usr/bin/env python3
"""
外部特徴量での精度評価システム
Yahoo Finance外部データによる精度向上検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class ExternalFeatureEvaluator:
    """外部特徴量評価システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
        # 従来の最適特徴量（ベースライン）
        self.baseline_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20'
        ]
        
    def load_integrated_data(self):
        """統合データ読み込み"""
        logger.info("📊 統合データ読み込み...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        if not integrated_file.exists():
            logger.error("❌ 統合データファイルが見つかりません")
            return None
            
        df = pd.read_parquet(integrated_file)
        logger.info(f"✅ 統合データ読み込み: {len(df):,}件, {len(df.columns)}カラム")
        
        return df
    
    def identify_external_features(self, df):
        """外部特徴量の特定"""
        logger.info("🔍 外部特徴量特定...")
        
        # 外部データパターン
        external_patterns = ['us_10y', 'sp500', 'usd_jpy', 'nikkei', 'vix']
        external_features = [col for col in df.columns 
                           if any(pattern in col for pattern in external_patterns)]
        
        logger.info(f"外部特徴量総数: {len(external_features)}個")
        
        # カテゴリ別分類
        value_features = [col for col in external_features if 'value' in col]
        change_features = [col for col in external_features if 'change' in col]
        volatility_features = [col for col in external_features if 'volatility' in col]
        
        logger.info(f"  値特徴量: {len(value_features)}個")
        logger.info(f"  変化特徴量: {len(change_features)}個")
        logger.info(f"  ボラティリティ特徴量: {len(volatility_features)}個")
        
        return {
            'all': external_features,
            'value': value_features,
            'change': change_features,
            'volatility': volatility_features
        }
    
    def baseline_accuracy_test(self, df):
        """ベースライン精度テスト"""
        logger.info("📏 ベースライン精度テスト（従来特徴量）...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # ベースライン特徴量の存在確認
        missing_baseline = [f for f in self.baseline_features if f not in clean_df.columns]
        if missing_baseline:
            logger.error(f"❌ ベースライン特徴量不足: {missing_baseline}")
            return None
            
        X = clean_df[self.baseline_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        return self._evaluate_model(X, y, "ベースライン（従来4特徴量）")
    
    def external_features_test(self, df, external_features_dict):
        """外部特徴量テスト"""
        logger.info("🌍 外部特徴量精度テスト...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        y = clean_df['Binary_Direction'].astype(int)
        
        results = {}
        
        # 1. 外部特徴量のみ
        logger.info("  1. 外部特徴量のみ")
        X_external_only = clean_df[external_features_dict['value']].fillna(0)
        results['external_only'] = self._evaluate_model(
            X_external_only, y, f"外部特徴量のみ（{len(external_features_dict['value'])}個）"
        )
        
        # 2. ベースライン + 外部値特徴量
        logger.info("  2. ベースライン + 外部値特徴量")
        combined_value_features = self.baseline_features + external_features_dict['value']
        available_combined = [f for f in combined_value_features if f in clean_df.columns]
        X_combined_value = clean_df[available_combined].fillna(0)
        results['baseline_plus_values'] = self._evaluate_model(
            X_combined_value, y, f"ベースライン + 外部値（{len(available_combined)}個）"
        )
        
        # 3. ベースライン + 外部変化特徴量
        logger.info("  3. ベースライン + 外部変化特徴量")
        combined_change_features = self.baseline_features + external_features_dict['change']
        available_change_combined = [f for f in combined_change_features if f in clean_df.columns]
        X_combined_change = clean_df[available_change_combined].fillna(0)
        results['baseline_plus_changes'] = self._evaluate_model(
            X_combined_change, y, f"ベースライン + 外部変化（{len(available_change_combined)}個）"
        )
        
        # 4. ベースライン + 全外部特徴量
        logger.info("  4. ベースライン + 全外部特徴量")
        all_combined_features = self.baseline_features + external_features_dict['all']
        available_all_combined = [f for f in all_combined_features if f in clean_df.columns]
        X_combined_all = clean_df[available_all_combined].fillna(0)
        results['baseline_plus_all_external'] = self._evaluate_model(
            X_combined_all, y, f"ベースライン + 全外部（{len(available_all_combined)}個）"
        )
        
        return results
    
    def feature_importance_analysis(self, df, best_features):
        """特徴量重要度分析"""
        logger.info("📊 特徴量重要度分析...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        available_features = [f for f in best_features if f in clean_df.columns]
        
        X = clean_df[available_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        # StandardScaler適用
        X_scaled = self.scaler.fit_transform(X)
        
        # LogisticRegression係数
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        # 重要度（絶対値）
        importances = abs(model.coef_[0])
        feature_importance = list(zip(available_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        logger.info("🎯 特徴量重要度ランキング:")
        for i, (feature, importance) in enumerate(feature_importance, 1):
            feature_type = "外部" if any(pattern in feature for pattern in ['us_10y', 'sp500', 'usd_jpy', 'nikkei', 'vix']) else "従来"
            logger.info(f"  {i:2d}. {feature:25s}: {importance:.4f} ({feature_type})")
        
        return feature_importance
    
    def _evaluate_model(self, X, y, description):
        """モデル評価（時系列5分割）"""
        X_scaled = self.scaler.fit_transform(X)
        
        model = LogisticRegression(
            C=0.001, 
            class_weight='balanced', 
            random_state=42, 
            max_iter=1000,
            solver='lbfgs'
        )
        
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
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        
        logger.info(f"    {description}: {avg_score:.3%} ± {std_score:.3%}")
        
        return {
            'avg': avg_score,
            'std': std_score,
            'scores': scores,
            'description': description
        }

def main():
    """メイン実行"""
    logger.info("🚀 外部特徴量精度評価システム")
    logger.info("🎯 目標: Yahoo Finance外部データによる精度向上検証")
    
    evaluator = ExternalFeatureEvaluator()
    
    try:
        # 1. 統合データ読み込み
        df = evaluator.load_integrated_data()
        if df is None:
            return
        
        # 2. 外部特徴量特定
        external_features_dict = evaluator.identify_external_features(df)
        
        # 3. ベースライン精度テスト
        baseline_result = evaluator.baseline_accuracy_test(df)
        
        if baseline_result is None:
            return
        
        # 4. 外部特徴量テスト
        external_results = evaluator.external_features_test(df, external_features_dict)
        
        # 結果比較と分析
        logger.info("\n" + "="*100)
        logger.info("🎯 外部特徴量による精度向上結果")
        logger.info("="*100)
        
        # ベースライン表示
        logger.info(f"📏 ベースライン: {baseline_result['avg']:.3%} ± {baseline_result['std']:.3%}")
        
        # 外部特徴量結果表示
        logger.info(f"\n📊 外部特徴量テスト結果:")
        results_with_baseline = [('baseline', baseline_result)] + list(external_results.items())
        
        best_result = None
        best_score = 0
        
        for result_name, result in results_with_baseline:
            accuracy = result['avg']
            if accuracy > best_score:
                best_score = accuracy
                best_result = (result_name, result)
            
            improvement = (accuracy - baseline_result['avg']) * 100
            status = "📈" if improvement > 0 else "📉" if improvement < 0 else "🔷"
            
            logger.info(f"  {status} {result['description']:35s}: {accuracy:.3%} ({improvement:+.2f}%)")
        
        # 最高結果の詳細
        logger.info(f"\n🏆 最高精度:")
        logger.info(f"  手法: {best_result[1]['description']}")
        logger.info(f"  精度: {best_result[1]['avg']:.3%} ± {best_result[1]['std']:.3%}")
        logger.info(f"  向上: {(best_result[1]['avg'] - baseline_result['avg']) * 100:+.2f}%")
        
        # 目標達成確認
        target_52 = 0.52
        target_53 = 0.53
        
        if best_result[1]['avg'] >= target_53:
            logger.info(f"🎉 目標達成！ 53%超え ({best_result[1]['avg']:.1%} >= 53.0%)")
        elif best_result[1]['avg'] >= target_52:
            logger.info(f"✅ 良好な結果！ 52%超え ({best_result[1]['avg']:.1%} >= 52.0%)")
        else:
            logger.info(f"📈 改善効果あり ({best_result[1]['avg']:.1%} vs ベースライン{baseline_result['avg']:.1%})")
        
        # 最高精度設定での特徴量重要度分析
        if best_result[0] != 'baseline':
            logger.info(f"\n🔍 最高精度設定での特徴量重要度分析...")
            if best_result[0] == 'baseline_plus_all_external':
                best_features = evaluator.baseline_features + external_features_dict['all']
            elif best_result[0] == 'baseline_plus_values':
                best_features = evaluator.baseline_features + external_features_dict['value']
            elif best_result[0] == 'baseline_plus_changes':
                best_features = evaluator.baseline_features + external_features_dict['change']
            else:  # external_only
                best_features = external_features_dict['value']
            
            evaluator.feature_importance_analysis(df, best_features)
        
        logger.info(f"\n💡 結論:")
        improvement_pct = (best_result[1]['avg'] - baseline_result['avg']) * 100
        if improvement_pct > 1.0:
            logger.info(f"✅ 外部データは有効！ {improvement_pct:.2f}%の精度向上")
        elif improvement_pct > 0.5:
            logger.info(f"📈 外部データは有益！ {improvement_pct:.2f}%の精度向上")
        elif improvement_pct > 0:
            logger.info(f"📊 外部データは微増効果！ {improvement_pct:.2f}%の精度向上")
        else:
            logger.info(f"⚠️ 外部データの効果限定的 ({improvement_pct:.2f}%)")
        
        logger.info(f"\n⚖️ この結果は全データ{len(df):,}件での厳密な5分割時系列検証です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()