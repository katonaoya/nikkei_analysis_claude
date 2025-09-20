#!/usr/bin/env python3
"""
最適化された高性能モデル訓練スクリプト
70%台の精度を目指した高度なパラメータ調整
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
from datetime import datetime
from loguru import logger

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

class OptimizedModelTrainer:
    """高精度を目指した最適化モデル訓練"""
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.scaler = StandardScaler()
        
    def load_features(self, filename: str) -> pd.DataFrame:
        """特徴量データの読み込み"""
        file_path = self.processed_dir / filename
        
        if not file_path.exists():
            raise FileNotFoundError(f"Features file not found: {file_path}")
        
        df = pd.read_parquet(file_path)
        logger.info(f"Loaded features: {df.shape}")
        return df
    
    def prepare_data(self, df: pd.DataFrame, target_column: str, test_size: float = 0.2):
        """データ準備（最適化版）"""
        
        # 特徴量選択（メタデータ除外）
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
        
        # より高度な特徴量処理
        X = clean_df[feature_cols].copy()
        
        # 前方補完 + 後方補完
        X = X.groupby(clean_df['Code']).apply(lambda group: group.fillna(method='ffill').fillna(method='bfill')).reset_index(drop=True)
        X = X.fillna(0)  # それでも残る場合は0で埋める
        
        y = clean_df[target_column]
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # 時系列分割（より現実的）
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def create_optimized_models(self):
        """最適化されたモデルセット"""
        models = {
            'random_forest_optimized': RandomForestClassifier(
                n_estimators=500,          # 大幅増加
                max_depth=25,              # 深い木
                min_samples_split=2,       # より柔軟な分割
                min_samples_leaf=1,        # より細かい葉
                max_features='sqrt',       # 特徴量選択の最適化
                bootstrap=True,
                oob_score=True,            # Out-of-bag評価
                n_jobs=-1,                 # 並列処理
                random_state=42,
                class_weight='balanced'    # クラス不均衡対策
            ),
            
            'extra_trees': ExtraTreesClassifier(
                n_estimators=300,
                max_depth=30,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            ),
            
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features='sqrt',
                random_state=42
            ),
            
            'logistic_optimized': LogisticRegression(
                C=1.0,
                solver='liblinear',
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            ),
            
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                alpha=0.001,
                learning_rate='adaptive',
                max_iter=500,
                random_state=42
            )
        }
        
        return models
    
    def train_optimized_models(self, X_train, X_test, y_train, y_test):
        """最適化モデルの訓練"""
        models = self.create_optimized_models()
        results = {}
        
        # 特徴量正規化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 特徴量選択（重要な特徴量のみ）
        selector = SelectKBest(score_func=f_classif, k=min(20, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        logger.info(f"Selected {X_train_selected.shape[1]} features from {X_train.shape[1]}")
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # スケーリングが必要なモデル
                if name in ['logistic_optimized', 'neural_network']:
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)
                    y_pred_proba = model.predict_proba(X_test_selected)
                else:
                    # 木ベースのモデル（元の特徴量）
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                
                # 評価
                accuracy = accuracy_score(y_test, y_pred)
                
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba
                }
                
                logger.info(f"{name} - Accuracy: {accuracy:.3f}")
                
                # Random Forestの場合、特徴量重要度も出力
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    if len(importances) == len(X_train.columns):
                        feature_importance = pd.DataFrame({
                            'feature': X_train.columns,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        logger.info(f"Top 5 features for {name}:")
                        for idx, row in feature_importance.head().iterrows():
                            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
                # Out-of-bag score
                if hasattr(model, 'oob_score_'):
                    logger.info(f"{name} - OOB Score: {model.oob_score_:.3f}")
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        return results, selector
    
    def save_model(self, model_name: str, model_data: dict, target_column: str, selector=None) -> Path:
        """最適化モデルの保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"optimized_{model_name}_{target_column}_{timestamp}.joblib"
        file_path = self.models_dir / filename
        
        model_package = {
            'model': model_data['model'],
            'scaler': self.scaler,
            'selector': selector,
            'target_column': target_column,
            'timestamp': timestamp,
            'performance': {k: v for k, v in model_data.items() if k not in ['model', 'predictions', 'probabilities']}
        }
        
        joblib.dump(model_package, file_path)
        logger.info(f"Saved optimized {model_name} to {file_path}")
        
        return file_path
    
    def create_performance_report(self, results: dict, y_test):
        """詳細な性能レポート"""
        
        print("\n" + "="*70)
        print("🚀 OPTIMIZED MODEL PERFORMANCE REPORT")
        print("="*70)
        
        # 精度順にソート
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for model_name, data in sorted_results:
            print(f"\n🤖 {model_name.upper().replace('_', ' ')}:")
            print(f"   🎯 Accuracy: {data['accuracy']:.1%}")
            
            # 詳細な分類レポート
            y_pred = data['predictions']
            print(f"\n   📊 Detailed Classification Report:")
            report = classification_report(y_test, y_pred, output_dict=True)
            print(f"   Precision (Up): {report['1']['precision']:.3f}")
            print(f"   Recall (Up): {report['1']['recall']:.3f}")
            print(f"   F1-Score (Up): {report['1']['f1-score']:.3f}")
            
            # 混同行列
            cm = confusion_matrix(y_test, y_pred)
            print(f"   📈 Confusion Matrix:")
            print(f"   True Down/Predicted Down: {cm[0,0]}")
            print(f"   True Down/Predicted Up: {cm[0,1]}")
            print(f"   True Up/Predicted Down: {cm[1,0]}")
            print(f"   True Up/Predicted Up: {cm[1,1]}")
            
        # 最高性能のモデル
        best_model_name = sorted_results[0][0]
        best_accuracy = sorted_results[0][1]['accuracy']
        print(f"\n🏆 BEST MODEL: {best_model_name.upper()} with {best_accuracy:.1%} accuracy")

def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(description="Optimized model training for high accuracy")
    parser.add_argument("--features-file", required=True, help="Features file name")
    parser.add_argument("--target", default="Binary_Direction", help="Target column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test data proportion")
    parser.add_argument("--save-models", action="store_true", help="Save trained models")
    
    args = parser.parse_args()
    
    try:
        trainer = OptimizedModelTrainer()
        
        # データ読み込み
        print("📊 Loading features...")
        df = trainer.load_features(args.features_file)
        
        # データ準備
        print("🔧 Preparing training data...")
        X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(
            df, args.target, args.test_size
        )
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Features: {len(feature_cols)}")
        
        # 最適化モデル訓練
        print("\n🤖 Training optimized models...")
        results, selector = trainer.train_optimized_models(X_train, X_test, y_train, y_test)
        
        # 性能レポート
        trainer.create_performance_report(results, y_test)
        
        # モデル保存
        if args.save_models:
            print(f"\n💾 Saving optimized models...")
            for model_name, model_data in results.items():
                if model_data['accuracy'] > 0.6:  # 60%以上のみ保存
                    trainer.save_model(model_name, model_data, args.target, selector)
        
        print("\n✅ Optimized model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())