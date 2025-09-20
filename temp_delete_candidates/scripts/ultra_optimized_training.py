#!/usr/bin/env python3
"""
超最適化モデル訓練 - 70%台の精度を目指す
XGBoost、LightGBM、アンサンブルを使用
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import argparse
from datetime import datetime
from loguru import logger

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

class UltraOptimizedTrainer:
    """超高精度を目指すモデル訓練"""
    
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
        """データ準備（超最適化版）"""
        
        # 特徴量選択
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
        
        # 特徴量前処理
        X = clean_df[feature_cols].copy()
        
        # 高度な前処理
        # 1. 前方補完 + 後方補完
        X = X.groupby(clean_df['Code']).fillna(method='ffill').fillna(method='bfill')
        X = X.fillna(0)
        
        # 2. 外れ値処理（5-95パーセンタイルでクリップ）
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                lower = X[col].quantile(0.05)
                upper = X[col].quantile(0.95)
                X[col] = X[col].clip(lower, upper)
        
        y = clean_df[target_column]
        
        logger.info(f"Features: {len(feature_cols)}, Samples: {len(X)}")
        logger.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        # 時系列分割（最新80%を訓練、最後20%をテスト）
        split_point = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def create_ultra_models(self):
        """超最適化モデルセット"""
        models = {
            'random_forest_ultra': RandomForestClassifier(
                n_estimators=1000,         # 大幅増加
                max_depth=30,              # より深く
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.8,          # より多くの特徴量を使用
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced',
                criterion='gini'           # エントロピーも試す
            ),
            
            'extra_trees_ultra': ExtraTreesClassifier(
                n_estimators=800,
                max_depth=35,
                min_samples_split=2,
                min_samples_leaf=1,
                max_features=0.9,
                bootstrap=True,
                oob_score=True,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'
            ),
            
            'lightgbm': lgb.LGBMClassifier(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=15,
                num_leaves=31,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                bagging_freq=5,
                min_child_samples=20,
                random_state=42,
                class_weight='balanced',
                n_jobs=-1,
                verbosity=-1
            ),
            
            'logistic_ultra': LogisticRegression(
                C=0.1,                     # より強い正則化
                solver='liblinear',
                max_iter=2000,
                random_state=42,
                class_weight='balanced',
                penalty='l1'               # L1正則化
            )
        }
        
        return models
    
    def train_ultra_models(self, X_train, X_test, y_train, y_test):
        """超最適化モデル訓練"""
        models = self.create_ultra_models()
        results = {}
        
        # 特徴量スケーリング
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 特徴量選択（より厳格）
        selector = SelectKBest(score_func=f_classif, k=min(15, X_train.shape[1]))
        X_train_selected = selector.fit_transform(X_train_scaled, y_train)
        X_test_selected = selector.transform(X_test_scaled)
        
        logger.info(f"Selected {X_train_selected.shape[1]} features from {X_train.shape[1]}")
        
        trained_models = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            try:
                # モデル別の特徴量選択
                if name in ['logistic_ultra']:
                    # 線形モデルはスケール済み+選択済み特徴量
                    model.fit(X_train_selected, y_train)
                    y_pred = model.predict(X_test_selected)
                    y_pred_proba = model.predict_proba(X_test_selected)
                elif name == 'lightgbm':
                    # LightGBMは元の特徴量
                    model.fit(X_train, y_train, eval_set=(X_test, y_test), callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)
                else:
                    # 木ベースのモデル
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
                
                trained_models[name] = model
                
                logger.info(f"{name} - Accuracy: {accuracy:.3f}")
                
                # 特徴量重要度
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                    if len(importances) == len(X_train.columns):
                        feature_importance = pd.DataFrame({
                            'feature': X_train.columns,
                            'importance': importances
                        }).sort_values('importance', ascending=False)
                        
                        logger.info(f"Top 3 features for {name}:")
                        for idx, row in feature_importance.head(3).iterrows():
                            logger.info(f"  {row['feature']}: {row['importance']:.4f}")
                
                # OOB Score
                if hasattr(model, 'oob_score_'):
                    logger.info(f"{name} - OOB Score: {model.oob_score_:.3f}")
                    
                # クロスバリデーション（時間がある場合のみ）
                if len(X_train) < 50000:  # データが小さい場合のみ
                    cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                    logger.info(f"{name} - CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                    
            except Exception as e:
                logger.error(f"Error training {name}: {e}")
                continue
        
        # アンサンブルモデル
        if len(trained_models) >= 2:
            logger.info("Creating ensemble model...")
            
            # 上位3モデルでアンサンブル
            sorted_models = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
            top_models = [(name, data['model']) for name, data in sorted_models[:3]]
            
            ensemble = VotingClassifier(
                estimators=top_models,
                voting='soft'  # 確率ベースの投票
            )
            
            try:
                # アンサンブル用のデータ準備（最も良いモデルの前処理に合わせる）
                best_model_name = sorted_models[0][0]
                if best_model_name in ['logistic_ultra']:
                    ensemble_X_train = X_train_selected
                    ensemble_X_test = X_test_selected
                else:
                    ensemble_X_train = X_train
                    ensemble_X_test = X_test
                
                ensemble.fit(ensemble_X_train, y_train)
                ensemble_pred = ensemble.predict(ensemble_X_test)
                ensemble_proba = ensemble.predict_proba(ensemble_X_test)
                ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
                
                results['ensemble'] = {
                    'model': ensemble,
                    'accuracy': ensemble_accuracy,
                    'predictions': ensemble_pred,
                    'probabilities': ensemble_proba
                }
                
                logger.info(f"ensemble - Accuracy: {ensemble_accuracy:.3f}")
                
            except Exception as e:
                logger.error(f"Error creating ensemble: {e}")
        
        return results, selector
    
    def create_ultra_report(self, results: dict, y_test):
        """超詳細性能レポート"""
        
        print("\n" + "="*80)
        print("🚀 ULTRA-OPTIMIZED MODEL PERFORMANCE REPORT")
        print("="*80)
        
        # 精度順にソート
        sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
        
        for i, (model_name, data) in enumerate(sorted_results):
            rank_emoji = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1:2d}."
            
            print(f"\n{rank_emoji} {model_name.upper().replace('_', ' ')}:")
            print(f"   🎯 Accuracy: {data['accuracy']:.1%}")
            
            # 分類レポートの要約
            y_pred = data['predictions']
            report = classification_report(y_test, y_pred, output_dict=True)
            
            print(f"   📊 Precision: {report['weighted avg']['precision']:.3f}")
            print(f"   📊 Recall: {report['weighted avg']['recall']:.3f}")
            print(f"   📊 F1-Score: {report['weighted avg']['f1-score']:.3f}")
            
            # 上昇予測の精度
            if '1' in report:
                print(f"   📈 Up Prediction Precision: {report['1']['precision']:.3f}")
                print(f"   📈 Up Prediction Recall: {report['1']['recall']:.3f}")
        
        # 最高成績
        best_model_name = sorted_results[0][0]
        best_accuracy = sorted_results[0][1]['accuracy']
        print(f"\n🏆 BEST MODEL: {best_model_name.upper()} with {best_accuracy:.1%} accuracy")
        
        if best_accuracy > 0.7:
            print("🎉 TARGET ACHIEVED: 70%+ accuracy!")
        elif best_accuracy > 0.6:
            print("👍 GOOD PERFORMANCE: 60%+ accuracy")
        else:
            print("📈 ROOM FOR IMPROVEMENT")

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description="Ultra-optimized training for maximum accuracy")
    parser.add_argument("--features-file", required=True, help="Features file name")
    parser.add_argument("--target", default="Binary_Direction", help="Target column")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test data proportion")
    parser.add_argument("--save-models", action="store_true", help="Save trained models")
    
    args = parser.parse_args()
    
    try:
        trainer = UltraOptimizedTrainer()
        
        print("📊 Loading features...")
        df = trainer.load_features(args.features_file)
        
        print("🔧 Preparing training data...")
        X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(
            df, args.target, args.test_size
        )
        print(f"   Training samples: {len(X_train):,}")
        print(f"   Test samples: {len(X_test):,}")
        print(f"   Features: {len(feature_cols)}")
        
        print("\n🚀 Training ultra-optimized models...")
        results, selector = trainer.train_ultra_models(X_train, X_test, y_train, y_test)
        
        trainer.create_ultra_report(results, y_test)
        
        if args.save_models:
            print(f"\n💾 Saving high-performance models...")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            for model_name, model_data in results.items():
                if model_data['accuracy'] > 0.55:  # 55%以上のみ保存
                    filename = f"ultra_{model_name}_{args.target}_{timestamp}.joblib"
                    file_path = trainer.models_dir / filename
                    
                    model_package = {
                        'model': model_data['model'],
                        'scaler': trainer.scaler,
                        'selector': selector,
                        'target_column': args.target,
                        'timestamp': timestamp,
                        'accuracy': model_data['accuracy']
                    }
                    
                    joblib.dump(model_package, file_path)
                    logger.info(f"Saved {model_name} ({model_data['accuracy']:.1%}) to {file_path}")
        
        print("\n✅ Ultra-optimized training completed!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())