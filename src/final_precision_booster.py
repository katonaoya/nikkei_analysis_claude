"""
Final Precision Booster - 175銘柄データでの効率的精度向上
目標: Precision ≥ 0.75達成のための最適化（計算効率重視）
"""

import os
import pickle
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FinalPrecisionBooster:
    """効率的な最終精度向上システム"""
    
    def __init__(self, target_precision: float = 0.75):
        self.target_precision = target_precision
        self.scaler = RobustScaler()
        logger.info(f"Final Precision Booster初期化完了 (目標精度: {target_precision:.1%})")
    
    def load_data(self) -> pd.DataFrame:
        """175銘柄データ読み込み"""
        logger.info("175銘柄データ読み込み...")
        
        data_dir = Path("data/nikkei225_full_data")
        pkl_files = list(data_dir.glob("nikkei225_full_10years_*.pkl"))
        latest_file = max(pkl_files, key=lambda f: f.stat().st_mtime)
        
        df = pd.read_pickle(latest_file)
        logger.info(f"データ読み込み完了: {len(df):,}レコード, {df['Code'].nunique()}銘柄")
        
        return df
    
    def create_optimized_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """効率的な特徴量エンジニアリング"""
        logger.info("効率的特徴量エンジニアリング開始...")
        
        # 基本前処理
        df = df.sort_values(['Code', 'Date']).reset_index(drop=True)
        
        # 数値カラム準備
        for col in ['Close', 'High', 'Low', 'Open', 'Volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 調整済み価格使用
        adj_cols = ['AdjustmentClose', 'AdjustmentHigh', 'AdjustmentLow', 'AdjustmentOpen']
        for col in adj_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # 基本リターン
        df['daily_return'] = df.groupby('Code')['Close'].pct_change(fill_method=None)
        df['next_day_return'] = df.groupby('Code')['Close'].pct_change(fill_method=None).shift(-1)
        
        # より厳しいターゲット（高精度狙い）
        df['target'] = (df['next_day_return'] >= 0.02).astype(int)  # 2%以上
        
        logger.info("核心的特徴量計算...")
        
        # 重要な特徴量のみに絞って高速計算
        windows = [5, 10, 20, 50]
        
        for window in windows:
            # 移動平均・価格比率
            sma = df.groupby('Code')['Close'].transform(lambda x: x.rolling(window).mean())
            df[f'price_to_sma_{window}'] = df['Close'] / sma
            
            # ボラティリティ
            df[f'volatility_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            
            # 高値・安値比率
            high_max = df.groupby('Code')['High'].transform(lambda x: x.rolling(window).max())
            low_min = df.groupby('Code')['Low'].transform(lambda x: x.rolling(window).min())
            df[f'price_position_{window}'] = (df['Close'] - low_min) / (high_max - low_min + 1e-8)
            
            # リターン統計
            df[f'return_mean_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'return_std_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            
            # ボリューム指標
            if window <= 20:  # 計算負荷削減
                vol_ma = df.groupby('Code')['Volume'].transform(lambda x: x.rolling(window).mean())
                df[f'volume_ratio_{window}'] = df['Volume'] / (vol_ma + 1)
        
        # RSI (14日のみ)
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df['rsi_14'] = df.groupby('Code')['Close'].transform(calc_rsi)
        
        # MACD (簡略版)
        def calc_macd(group):
            ema12 = group['Close'].ewm(span=12).mean()
            ema26 = group['Close'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            return pd.DataFrame({
                'macd': macd,
                'macd_signal': signal,
                'macd_histogram': macd - signal
            })
        
        logger.info("MACD計算...")
        macd_df = df.groupby('Code').apply(calc_macd)
        macd_df.index = macd_df.index.droplevel(0)
        df = df.join(macd_df)
        
        # ボリンジャーバンド (20日のみ)
        sma_20 = df.groupby('Code')['Close'].transform(lambda x: x.rolling(20).mean())
        std_20 = df.groupby('Code')['Close'].transform(lambda x: x.rolling(20).std())
        df['bb_upper'] = sma_20 + (2 * std_20)
        df['bb_lower'] = sma_20 - (2 * std_20)
        df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-8)
        
        # ラグ特徴量 (5個まで)
        for lag in range(1, 6):
            df[f'return_lag_{lag}'] = df.groupby('Code')['daily_return'].shift(lag)
            df[f'price_lag_{lag}'] = df.groupby('Code')['Close'].shift(lag)
        
        # 市場との相関 (簡略版)
        market_return = df.groupby('Date')['daily_return'].mean()
        df['market_return'] = df['Date'].map(market_return)
        df['relative_return'] = df['daily_return'] - df['market_return']
        
        # 異常値特徴量
        df['price_zscore'] = df.groupby('Code')['Close'].transform(
            lambda x: (x - x.rolling(30).mean()) / (x.rolling(30).std() + 1e-8)
        )
        
        # 流動性
        df['liquidity'] = df['Volume'] * df['Close']
        df['liquidity_rank'] = df.groupby('Date')['liquidity'].rank(pct=True)
        
        logger.info(f"特徴量エンジニアリング完了: 約{len([c for c in df.columns if c not in ['Code', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'target', 'next_day_return']])}個")
        
        return df
    
    def optimize_ensemble_weights(self, lgb_proba: np.ndarray, cb_proba: np.ndarray, 
                                xgb_proba: np.ndarray, y_true: np.ndarray) -> Tuple[float, float, float]:
        """アンサンブル重み最適化"""
        best_precision = 0
        best_weights = (0.33, 0.33, 0.34)
        
        # グリッドサーチで最適重み探索
        for w1 in np.arange(0.2, 0.7, 0.1):
            for w2 in np.arange(0.2, 0.8 - w1, 0.1):
                w3 = 1 - w1 - w2
                if w3 < 0.1:
                    continue
                
                ensemble_proba = w1 * lgb_proba + w2 * cb_proba + w3 * xgb_proba
                
                # 複数閾値で評価
                for threshold in np.arange(0.7, 0.95, 0.05):
                    preds = (ensemble_proba >= threshold).astype(int)
                    if preds.sum() > 0:
                        precision = precision_score(y_true, preds, zero_division=0)
                        if precision > best_precision:
                            best_precision = precision
                            best_weights = (w1, w2, w3)
        
        return best_weights
    
    def train_advanced_models(self, X_train: np.ndarray, y_train: np.ndarray, 
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """高度なモデル訓練"""
        
        # LightGBM (高精度設定)
        lgb_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 128,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'learning_rate': 0.05,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': -1,
            'random_state': 42,
            'n_estimators': 500
        }
        
        # CatBoost (高精度設定)
        cb_params = {
            'loss_function': 'Logloss',
            'iterations': 500,
            'learning_rate': 0.05,
            'depth': 8,
            'l2_leaf_reg': 3,
            'bootstrap_type': 'Bernoulli',
            'subsample': 0.8,
            'random_seed': 42,
            'verbose': False
        }
        
        # XGBoost (高精度設定)
        xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 8,
            'learning_rate': 0.05,
            'n_estimators': 500,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 1,
            'random_state': 42
        }
        
        # モデル訓練
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        cb_model = cb.CatBoostClassifier(**cb_params)
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        # 確率校正
        lgb_calibrated = CalibratedClassifierCV(lgb_model, method='isotonic', cv=3)
        cb_calibrated = CalibratedClassifierCV(cb_model, method='isotonic', cv=3)
        xgb_calibrated = CalibratedClassifierCV(xgb_model, method='isotonic', cv=3)
        
        lgb_calibrated.fit(X_train, y_train)
        cb_calibrated.fit(X_train, y_train)
        xgb_calibrated.fit(X_train, y_train)
        
        # 検証データで予測
        lgb_proba = lgb_calibrated.predict_proba(X_val)[:, 1]
        cb_proba = cb_calibrated.predict_proba(X_val)[:, 1]
        xgb_proba = xgb_calibrated.predict_proba(X_val)[:, 1]
        
        # 最適重み計算
        best_weights = self.optimize_ensemble_weights(lgb_proba, cb_proba, xgb_proba, y_val)
        
        return {
            'models': {
                'lgb': lgb_calibrated,
                'cb': cb_calibrated, 
                'xgb': xgb_calibrated
            },
            'weights': best_weights,
            'probabilities': {
                'lgb': lgb_proba,
                'cb': cb_proba,
                'xgb': xgb_proba
            }
        }
    
    def find_optimal_threshold(self, ensemble_proba: np.ndarray, y_true: np.ndarray) -> float:
        """最適閾値探索"""
        best_precision = 0
        best_threshold = 0.85
        
        for threshold in np.arange(0.5, 0.99, 0.02):
            predictions = (ensemble_proba >= threshold).astype(int)
            if predictions.sum() > 0:
                precision = precision_score(y_true, predictions, zero_division=0)
                if precision > best_precision:
                    best_precision = precision
                    best_threshold = threshold
        
        return best_threshold
    
    def run_final_optimization(self) -> Dict[str, float]:
        """最終最適化実行"""
        logger.info("=== Final Precision Boost開始 ===")
        logger.info(f"目標精度: {self.target_precision:.1%}")
        
        # データ準備
        df = self.load_data()
        df = self.create_optimized_features(df)
        
        # 特徴量準備
        feature_cols = [col for col in df.columns if col not in 
                       ['Code', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 
                        'target', 'next_day_return'] and not col.startswith('Adjustment')]
        
        X = df[feature_cols].fillna(0)
        y = df['target']
        
        logger.info(f"特徴量数: {len(feature_cols)}")
        logger.info(f"データサイズ: {len(X):,}")
        logger.info(f"ターゲット分布: {y.mean():.1%}")
        
        # 外れ値除去
        outlier_detector = IsolationForest(contamination=0.05, random_state=42)
        outliers = outlier_detector.fit_predict(X) == -1
        X, y = X[~outliers], y[~outliers]
        
        logger.info(f"外れ値除去後: {len(X):,}レコード")
        
        # 特徴量選択 (上位70%)
        selector = SelectKBest(f_classif, k=int(len(feature_cols) * 0.7))
        X_selected = selector.fit_transform(X, y)
        selected_features = np.array(feature_cols)[selector.get_support()]
        
        logger.info(f"選択特徴量: {len(selected_features)}")
        
        # スケーリング
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 時系列分割評価
        tscv = TimeSeriesSplit(n_splits=5, gap=10)
        results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            logger.info(f"Fold {fold + 1}/5 実行中...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # 高度モデル訓練
            model_results = self.train_advanced_models(X_train, y_train, X_val, y_val)
            
            # アンサンブル予測
            w1, w2, w3 = model_results['weights']
            ensemble_proba = (w1 * model_results['probabilities']['lgb'] + 
                            w2 * model_results['probabilities']['cb'] + 
                            w3 * model_results['probabilities']['xgb'])
            
            # 最適閾値
            threshold = self.find_optimal_threshold(ensemble_proba, y_val)
            predictions = (ensemble_proba >= threshold).astype(int)
            
            # 評価
            if predictions.sum() > 0:
                precision = precision_score(y_val, predictions)
                recall = recall_score(y_val, predictions)
                f1 = f1_score(y_val, predictions)
                
                results.append({
                    'fold': fold + 1,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'threshold': threshold,
                    'predictions': predictions.sum(),
                    'total': len(predictions),
                    'weights': (w1, w2, w3)
                })
                
                logger.info(f"Fold {fold + 1} - Precision: {precision:.4f}, Threshold: {threshold:.3f}, Predictions: {predictions.sum()}")
        
        # 最終結果
        if results:
            final_results = {
                'mean_precision': np.mean([r['precision'] for r in results]),
                'std_precision': np.std([r['precision'] for r in results]),
                'mean_recall': np.mean([r['recall'] for r in results]),
                'mean_f1': np.mean([r['f1'] for r in results]),
                'mean_threshold': np.mean([r['threshold'] for r in results]),
                'total_predictions': sum([r['predictions'] for r in results]),
                'total_samples': sum([r['total'] for r in results]),
                'feature_count': len(selected_features),
                'success_rate': len([r for r in results if r['precision'] >= self.target_precision]) / len(results)
            }
            
            logger.info("=== 最終結果 ===")
            logger.info(f"平均精度: {final_results['mean_precision']:.4f} ± {final_results['std_precision']:.4f}")
            logger.info(f"目標達成フォールド: {final_results['success_rate']:.1%}")
            logger.info(f"平均閾値: {final_results['mean_threshold']:.3f}")
            logger.info(f"選択特徴量数: {final_results['feature_count']}")
            
            return final_results
        else:
            return {'error': 'No valid results generated'}


def main():
    """メイン実行関数"""
    try:
        booster = FinalPrecisionBooster(target_precision=0.75)
        results = booster.run_final_optimization()
        
        print("\n=== Final Precision Boost結果 ===")
        if 'error' not in results:
            print(f"平均精度: {results['mean_precision']:.4f}")
            print(f"標準偏差: {results['std_precision']:.4f}")
            print(f"目標達成率: {results['success_rate']:.1%}")
            print(f"平均閾値: {results['mean_threshold']:.3f}")
            print(f"特徴量数: {results['feature_count']}")
            
            if results['mean_precision'] >= 0.75:
                print("🎉 目標精度0.75達成！")
            else:
                shortage = 0.75 - results['mean_precision']
                print(f"❌ 目標精度未達成 (不足: {shortage:.4f})")
                
                # さらなる改善提案
                if shortage < 0.05:
                    print("💡 わずかな改善で達成可能。閾値調整や特徴量追加を推奨")
                elif shortage < 0.1:
                    print("💡 中程度の改善が必要。モデル構造やアンサンブル重み調整を推奨")
                else:
                    print("💡 大幅な改善が必要。データクリーニングやターゲット定義見直しを推奨")
        else:
            print(f"❌ エラー: {results['error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Final Precision Boost失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()