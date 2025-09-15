#!/usr/bin/env python3
"""
厳密な10年間データでの精度検証
2014-2024の完全10年間での検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from datetime import datetime, date
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class Strict10YearValidator:
    """厳密な10年間検証システム"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
        # ベースライン特徴量
        self.baseline_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20'
        ]
        
        # 最適外部特徴量（前回の結果から）
        self.optimal_external_features = [
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
    
    def load_and_filter_10_years(self):
        """10年間データの厳密抽出"""
        logger.info("📊 10年間データの厳密抽出...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        if not integrated_file.exists():
            logger.error("❌ 統合データファイルが見つかりません")
            return None
            
        df = pd.read_parquet(integrated_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 現在の利用可能な完全年度を確認
        available_years = sorted(df['Date'].dt.year.unique())
        logger.info(f"利用可能年度: {available_years}")
        
        # 最新の完全10年間を決定
        # 2025年は8月までなので、2024年を最終年として2015-2024の10年間を使用
        end_year = 2024
        start_year = end_year - 9  # 10年間なので9年差
        
        logger.info(f"検証期間設定: {start_year}年～{end_year}年 (完全10年間)")
        
        # 期間フィルタリング
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        filtered_df = df[
            (df['Date'] >= start_date) & 
            (df['Date'] <= end_date)
        ].copy()
        
        logger.info(f"✅ 10年間データ抽出完了:")
        logger.info(f"  期間: {filtered_df['Date'].min().date()} - {filtered_df['Date'].max().date()}")
        logger.info(f"  データ数: {len(filtered_df):,}件")
        
        # 年別分布確認
        yearly_dist = filtered_df.groupby(filtered_df['Date'].dt.year).size()
        logger.info(f"  年別分布:")
        for year, count in yearly_dist.items():
            logger.info(f"    {year}年: {count:,}件")
        
        return filtered_df
    
    def validate_data_quality(self, df):
        """データ品質検証"""
        logger.info("🔍 10年間データ品質検証...")
        
        # 予測対象の確認
        valid_target = df['Binary_Direction'].notna().sum()
        logger.info(f"予測対象数: {valid_target:,}件")
        
        # 外部データの充実度確認
        external_cols = [col for col in df.columns if any(pattern in col for pattern in ['us_10y', 'sp500', 'usd_jpy', 'nikkei', 'vix'])]
        logger.info(f"外部特徴量数: {len(external_cols)}個")
        
        # 最適外部特徴量の存在確認
        missing_optimal = [f for f in self.optimal_external_features if f not in df.columns]
        if missing_optimal:
            logger.error(f"❌ 最適外部特徴量不足: {missing_optimal}")
            return False
        
        # 外部データの欠損状況
        logger.info("外部データ欠損状況:")
        for col in self.optimal_external_features:
            missing_rate = df[col].isnull().mean()
            logger.info(f"  {col}: {missing_rate*100:.1f}%欠損")
        
        return True
    
    def baseline_10_year_test(self, df):
        """ベースライン10年間テスト"""
        logger.info("📏 ベースライン10年間テスト...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # ベースライン特徴量の存在確認
        missing_baseline = [f for f in self.baseline_features if f not in clean_df.columns]
        if missing_baseline:
            logger.error(f"❌ ベースライン特徴量不足: {missing_baseline}")
            return None
            
        X = clean_df[self.baseline_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        logger.info(f"ベースライン検証データ: {len(clean_df):,}件")
        
        return self._time_series_evaluation(X, y, "ベースライン（従来4特徴量・10年間）")
    
    def external_10_year_test(self, df):
        """外部特徴量10年間テスト"""
        logger.info("🌍 外部特徴量10年間テスト...")
        
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        y = clean_df['Binary_Direction'].astype(int)
        
        # 最適組み合わせ（ベースライン + 外部変化特徴量）
        optimal_features = self.baseline_features + self.optimal_external_features
        available_features = [f for f in optimal_features if f in clean_df.columns]
        
        X = clean_df[available_features].fillna(0)
        
        logger.info(f"外部特徴量検証データ: {len(clean_df):,}件")
        logger.info(f"使用特徴量: {len(available_features)}個")
        
        return self._time_series_evaluation(X, y, f"最適外部特徴量（{len(available_features)}個・10年間）")
    
    def _time_series_evaluation(self, X, y, description):
        """時系列評価（10分割でより厳密に）"""
        X_scaled = self.scaler.fit_transform(X)
        
        model = LogisticRegression(
            C=0.001, 
            class_weight='balanced', 
            random_state=42, 
            max_iter=1000,
            solver='lbfgs'
        )
        
        # 10分割でより厳密な評価
        tscv = TimeSeriesSplit(n_splits=10)
        scores = []
        fold_details = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            fold_details.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
        
        avg_score = np.mean(scores)
        std_score = np.std(scores)
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        logger.info(f"  {description}:")
        logger.info(f"    平均精度: {avg_score:.3%} ± {std_score:.3%}")
        logger.info(f"    範囲: {min_score:.1%} - {max_score:.1%}")
        logger.info(f"    10分割詳細:")
        for detail in fold_details:
            logger.info(f"      Fold{detail['fold']:2d}: {detail['accuracy']:.1%} (Train:{detail['train_size']:,}, Test:{detail['test_size']:,})")
        
        return {
            'avg': avg_score,
            'std': std_score,
            'min': min_score,
            'max': max_score,
            'scores': scores,
            'description': description,
            'fold_details': fold_details
        }
    
    def compare_results(self, baseline_result, external_result):
        """結果比較と統計的有意性検定"""
        logger.info("📊 10年間検証結果比較...")
        
        baseline_score = baseline_result['avg']
        external_score = external_result['avg']
        improvement = (external_score - baseline_score) * 100
        
        # 統計的有意性の簡易チェック（信頼区間比較）
        baseline_ci = baseline_result['std'] * 1.96  # 95%信頼区間
        external_ci = external_result['std'] * 1.96
        
        logger.info(f"ベースライン: {baseline_score:.3%} (95%CI: ±{baseline_ci:.3%})")
        logger.info(f"外部特徴量: {external_score:.3%} (95%CI: ±{external_ci:.3%})")
        logger.info(f"改善効果: {improvement:+.2f}%")
        
        # 統計的有意性の判定
        if improvement > baseline_ci + external_ci:
            significance = "統計的に有意な改善"
        elif improvement > 0:
            significance = "改善傾向（有意性要検証）"
        else:
            significance = "改善効果なし"
        
        logger.info(f"統計的評価: {significance}")
        
        return {
            'improvement_pct': improvement,
            'significance': significance,
            'baseline_score': baseline_score,
            'external_score': external_score
        }

def main():
    """メイン実行"""
    logger.info("🚀 厳密な10年間データ検証システム")
    logger.info("🎯 目標: 10年分の実データでの正確な精度評価")
    
    validator = Strict10YearValidator()
    
    try:
        # 1. 10年間データの厳密抽出
        df_10years = validator.load_and_filter_10_years()
        if df_10years is None:
            return
        
        # 2. データ品質検証
        if not validator.validate_data_quality(df_10years):
            return
        
        # 3. ベースライン10年間テスト
        baseline_result = validator.baseline_10_year_test(df_10years)
        if baseline_result is None:
            return
        
        # 4. 外部特徴量10年間テスト
        external_result = validator.external_10_year_test(df_10years)
        
        # 5. 結果比較
        comparison = validator.compare_results(baseline_result, external_result)
        
        # 最終レポート
        logger.info("\n" + "="*100)
        logger.info("🎯 厳密な10年間検証結果")
        logger.info("="*100)
        
        logger.info(f"📊 検証期間: 2015-2024年 (完全10年間)")
        logger.info(f"📈 検証データ数: {len(df_10years):,}件")
        logger.info(f"🔬 評価方法: 10分割時系列クロスバリデーション")
        
        logger.info(f"\n📏 ベースライン結果:")
        logger.info(f"  精度: {baseline_result['avg']:.3%} ± {baseline_result['std']:.3%}")
        logger.info(f"  範囲: {baseline_result['min']:.1%} - {baseline_result['max']:.1%}")
        
        logger.info(f"\n🌍 外部特徴量結果:")
        logger.info(f"  精度: {external_result['avg']:.3%} ± {external_result['std']:.3%}")
        logger.info(f"  範囲: {external_result['min']:.1%} - {external_result['max']:.1%}")
        
        logger.info(f"\n🎯 改善効果:")
        logger.info(f"  精度向上: {comparison['improvement_pct']:+.2f}%")
        logger.info(f"  統計的評価: {comparison['significance']}")
        
        # 結論
        if comparison['external_score'] > 0.55:
            logger.info(f"\n🎉 優秀な結果！10年間検証で55%超えを達成")
        elif comparison['external_score'] > 0.52:
            logger.info(f"\n✅ 良好な結果！10年間検証で52%超えを達成")
        elif comparison['improvement_pct'] > 1.0:
            logger.info(f"\n📈 有意な改善効果を確認")
        else:
            logger.info(f"\n📊 改善効果は限定的")
        
        logger.info(f"\n⚖️ この結果は完全な10年分の実データでの厳密検証です")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()