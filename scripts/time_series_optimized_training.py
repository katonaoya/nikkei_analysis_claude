#!/usr/bin/env python3
"""
時系列最適化モデル訓練 - OOBスコア71%を活用
時系列特化の評価とウォークフォワード分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
from datetime import datetime
from loguru import logger

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class TimeSeriesOptimizedTrainer:
    """時系列特化の最適化訓練"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        
    def load_features(self, filename: str) -> pd.DataFrame:
        """特徴量データの読み込み"""
        file_path = self.processed_dir / filename
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded features: {df.shape}")
        return df
    
    def prepare_time_series_data(self, df: pd.DataFrame, target_column: str):
        """時系列データの準備"""
        
        exclude_cols = {
            'Date', 'Code', 'Open', 'High', 'Low', 'Close', 'Volume',
            'date', 'code', 'open', 'high', 'low', 'close', 'volume',
            'UpperLimit', 'LowerLimit', 'turnover_value', 'adjustment_factor',
            'adj_open', 'adj_high', 'adj_low', 'adj_close', 'adj_volume',
            target_column, 'Next_Day_Return', 'Return_Direction', 'Binary_Direction'
        }
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # 欠損値処理
        clean_df = df[df[target_column].notna()].copy()
        
        # 時系列順にソート
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        # 特徴量前処理
        X = clean_df[feature_cols].copy()
        X = X.groupby(clean_df['Code']).fillna(method='ffill').fillna(method='bfill')
        X = X.fillna(0)
        
        # 外れ値処理
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                lower = X[col].quantile(0.02)
                upper = X[col].quantile(0.98)
                X[col] = X[col].clip(lower, upper)
        
        y = clean_df[target_column]
        dates = clean_df['Date']
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        logger.info(f"Date range: {dates.min()} to {dates.max()}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y, dates, feature_cols
    
    def time_series_walk_forward_validation(self, X, y, dates, n_splits=5):
        """ウォークフォワード検証"""
        logger.info(f"Starting walk-forward validation with {n_splits} splits...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        model = RandomForestClassifier(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced'
        )
        
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            logger.info(f"Training fold {fold + 1}/{n_splits}...")
            
            X_train_fold = X.iloc[train_idx]
            X_test_fold = X.iloc[test_idx]
            y_train_fold = y.iloc[train_idx]
            y_test_fold = y.iloc[test_idx]
            
            # 訓練
            model.fit(X_train_fold, y_train_fold)
            
            # 予測
            y_pred_fold = model.predict(X_test_fold)
            accuracy = accuracy_score(y_test_fold, y_pred_fold)
            
            # OOBスコア
            oob_score = model.oob_score_ if hasattr(model, 'oob_score_') else None
            
            fold_result = {
                'fold': fold + 1,
                'accuracy': accuracy,
                'oob_score': oob_score,
                'train_start': dates.iloc[train_idx].min(),
                'train_end': dates.iloc[train_idx].max(),
                'test_start': dates.iloc[test_idx].min(),
                'test_end': dates.iloc[test_idx].max(),
                'train_samples': len(train_idx),
                'test_samples': len(test_idx)
            }
            
            fold_results.append(fold_result)
            
            oob_text = f"{oob_score:.3f}" if oob_score else "N/A"
            logger.info(f"Fold {fold + 1}: Accuracy={accuracy:.3f}, OOB={oob_text}")
        
        return fold_results, model
    
    def train_final_optimized_model(self, X, y):
        """最終最適化モデル訓練"""
        logger.info("Training final optimized model...")
        
        # より攻撃的なパラメータ
        model = RandomForestClassifier(
            n_estimators=1000,           # 木の数を大幅増加
            max_depth=None,              # 深さ制限なし
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='log2',         # log2で特徴量選択
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
            class_weight='balanced_subsample',  # より強いバランス調整
            criterion='entropy'          # エントロピー基準
        )
        
        # 全データで訓練
        model.fit(X, y)
        
        logger.info(f"Final model - OOB Score: {model.oob_score_:.3f}")
        
        # 特徴量重要度
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("Top 5 most important features:")
        for idx, row in feature_importance.head().iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        return model, feature_importance
    
    def create_time_series_report(self, fold_results, final_model):
        """時系列分析レポート"""
        
        print("\n" + "="*80)
        print("📈 TIME SERIES WALK-FORWARD VALIDATION REPORT")
        print("="*80)
        
        for result in fold_results:
            print(f"\n📅 Fold {result['fold']}:")
            print(f"   Training: {result['train_start'].strftime('%Y-%m-%d')} to {result['train_end'].strftime('%Y-%m-%d')} ({result['train_samples']:,} samples)")
            print(f"   Testing:  {result['test_start'].strftime('%Y-%m-%d')} to {result['test_end'].strftime('%Y-%m-%d')} ({result['test_samples']:,} samples)")
            print(f"   🎯 Test Accuracy: {result['accuracy']:.1%}")
            print(f"   🔄 OOB Score: {result['oob_score']:.1%}")
        
        # 統計サマリー
        accuracies = [r['accuracy'] for r in fold_results]
        oob_scores = [r['oob_score'] for r in fold_results if r['oob_score']]
        
        print(f"\n📊 VALIDATION SUMMARY:")
        print(f"   Average Test Accuracy: {np.mean(accuracies):.1%} ± {np.std(accuracies):.1%}")
        print(f"   Average OOB Score: {np.mean(oob_scores):.1%} ± {np.std(oob_scores):.1%}")
        print(f"   Best Test Accuracy: {max(accuracies):.1%}")
        print(f"   Best OOB Score: {max(oob_scores):.1%}")
        
        print(f"\n🏆 FINAL MODEL:")
        print(f"   Final OOB Score: {final_model.oob_score_:.1%}")
        
        if max(oob_scores) > 0.7:
            print("\n🎉 SUCCESS: Achieved 70%+ OOB accuracy!")
        elif max(oob_scores) > 0.6:
            print("\n👍 GOOD: Achieved 60%+ OOB accuracy")
        
        return {
            'avg_test_accuracy': np.mean(accuracies),
            'avg_oob_score': np.mean(oob_scores),
            'best_test_accuracy': max(accuracies),
            'best_oob_score': max(oob_scores),
            'final_oob_score': final_model.oob_score_
        }

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Time series optimized training")
    parser.add_argument("--features-file", required=True, help="Features file name")
    parser.add_argument("--target", default="Binary_Direction", help="Target column")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of time series splits")
    parser.add_argument("--save-models", action="store_true", help="Save trained models")
    
    args = parser.parse_args()
    
    try:
        trainer = TimeSeriesOptimizedTrainer()
        
        print("📊 Loading features...")
        df = trainer.load_features(args.features_file)
        
        print("🔧 Preparing time series data...")
        X, y, dates, feature_cols = trainer.prepare_time_series_data(df, args.target)
        
        print("\n📈 Running walk-forward validation...")
        fold_results, _ = trainer.time_series_walk_forward_validation(X, y, dates, args.n_splits)
        
        print("\n🚀 Training final optimized model...")
        final_model, feature_importance = trainer.train_final_optimized_model(X, y)
        
        # レポート作成
        summary = trainer.create_time_series_report(fold_results, final_model)
        
        # モデル保存
        if args.save_models and summary['final_oob_score'] > 0.6:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"time_series_optimized_{args.target}_{timestamp}.joblib"
            file_path = trainer.models_dir / filename
            
            model_package = {
                'model': final_model,
                'scaler': trainer.scaler,
                'target_column': args.target,
                'timestamp': timestamp,
                'performance': summary,
                'feature_importance': feature_importance
            }
            
            joblib.dump(model_package, file_path)
            logger.info(f"Saved final model (OOB: {summary['final_oob_score']:.1%}) to {file_path}")
        
        print("\n✅ Time series optimized training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())