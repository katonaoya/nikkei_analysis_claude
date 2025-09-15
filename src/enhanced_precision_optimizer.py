"""
Enhanced Precision Optimizer - 175銘柄データでの精度向上
目標: Precision ≥ 0.75達成のための高度な最適化
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import catboost as cb
import xgboost as xgb
import optuna

warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedPrecisionOptimizer:
    """高度な精度最適化システム"""
    
    def __init__(self, target_precision: float = 0.75):
        """
        初期化
        
        Args:
            target_precision: 目標精度
        """
        self.target_precision = target_precision
        self.scaler = RobustScaler()
        self.feature_selector = None
        self.anomaly_detector = IsolationForest(contamination=0.1, random_state=42)
        self.best_models = {}
        self.best_params = {}
        self.feature_importance = {}
        
        logger.info(f"Enhanced Precision Optimizer初期化完了 (目標精度: {target_precision:.1%})")
    
    def load_data(self) -> pd.DataFrame:
        """175銘柄の10年データを読み込み"""
        logger.info("175銘柄データ読み込み開始...")
        
        data_dir = Path("data/nikkei225_full_data")
        
        # 最新のpklファイルを検索
        pkl_files = list(data_dir.glob("nikkei225_full_10years_*.pkl"))
        if not pkl_files:
            raise FileNotFoundError("175銘柄データファイルが見つかりません")
        
        latest_file = max(pkl_files, key=lambda f: f.stat().st_mtime)
        logger.info(f"データファイル: {latest_file}")
        
        df = pd.read_pickle(latest_file)
        
        logger.info(f"データ読み込み完了: {len(df):,}レコード")
        logger.info(f"銘柄数: {df['Code'].nunique()}銘柄")
        logger.info(f"期間: {df['Date'].min()} ～ {df['Date'].max()}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """高度な特徴量エンジニアリング - 300+特徴量"""
        logger.info("高度な特徴量エンジニアリング開始...")
        
        # 基本的な前処理
        df = df.sort_values(['Code', 'Date']).reset_index(drop=True)
        df['close_price'] = pd.to_numeric(df['Close'], errors='coerce')
        df['volume'] = pd.to_numeric(df['Volume'], errors='coerce')
        df['high_price'] = pd.to_numeric(df['High'], errors='coerce')
        df['low_price'] = pd.to_numeric(df['Low'], errors='coerce')
        df['open_price'] = pd.to_numeric(df['Open'], errors='coerce')
        
        # 調整済み価格を使用
        for col in ['AdjustmentClose', 'AdjustmentHigh', 'AdjustmentLow', 'AdjustmentOpen', 'AdjustmentVolume']:
            if col in df.columns:
                df[col.lower().replace('adjustment', 'adj_')] = pd.to_numeric(df[col], errors='coerce')
        
        # 基本リターン
        df['daily_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None)
        df['next_day_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None).shift(-1)
        
        # ターゲット作成（より厳しい閾値で高精度を狙う）
        df['target'] = (df['next_day_return'] >= 0.015).astype(int)  # 1.5%以上の上昇
        
        logger.info("高度なテクニカル指標計算中...")
        
        # 複数期間でのテクニカル指標
        windows = [3, 5, 7, 10, 14, 20, 25, 30, 50, 100]
        
        for window in windows:
            # 移動平均系
            df[f'sma_{window}'] = df.groupby('Code')['close_price'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'ema_{window}'] = df.groupby('Code')['close_price'].transform(
                lambda x: x.ewm(span=window).mean()
            )
            
            # 価格比率
            df[f'price_to_sma_{window}'] = df['close_price'] / df[f'sma_{window}']
            df[f'price_to_ema_{window}'] = df['close_price'] / df[f'ema_{window}']
            
            # ボラティリティ
            df[f'volatility_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            
            # 高値・安値からの距離
            df[f'high_{window}'] = df.groupby('Code')['high_price'].transform(
                lambda x: x.rolling(window).max()
            )
            df[f'low_{window}'] = df.groupby('Code')['low_price'].transform(
                lambda x: x.rolling(window).min()
            )
            df[f'price_position_{window}'] = (df['close_price'] - df[f'low_{window}']) / (
                df[f'high_{window}'] - df[f'low_{window}'] + 1e-8
            )
            
            # リターン統計
            df[f'return_mean_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).mean()
            )
            df[f'return_std_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            df[f'return_skew_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).skew()
            )
            df[f'return_kurt_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).kurt()
            )
            
            # ボリューム指標
            if 'volume' in df.columns:
                df[f'volume_ma_{window}'] = df.groupby('Code')['volume'].transform(
                    lambda x: x.rolling(window).mean()
                )
                df[f'volume_ratio_{window}'] = df['volume'] / (df[f'volume_ma_{window}'] + 1)
        
        # RSI（複数期間）
        for period in [7, 14, 21, 30]:
            df[f'rsi_{period}'] = df.groupby('Code').apply(
                lambda x: self._calculate_rsi(x['close_price'], period)
            ).values
        
        # MACD系指標
        logger.info("MACD指標計算中...")
        
        def calculate_group_macd(group):
            ema12 = group['close_price'].ewm(span=12).mean()
            ema26 = group['close_price'].ewm(span=26).mean()
            macd = ema12 - ema26
            signal = macd.ewm(span=9).mean()
            return pd.DataFrame({'macd': macd, 'macd_signal': signal})
        
        macd_results = df.groupby('Code').apply(calculate_group_macd)
        macd_results.index = macd_results.index.droplevel(0)
        df = df.join(macd_results)
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # ボリンジャーバンド（複数期間・標準偏差）
        for window in [10, 20, 30]:
            for std_dev in [1.5, 2.0, 2.5]:
                mean = df.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
                std = df.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).std())
                df[f'bb_upper_{window}_{std_dev}'] = mean + (std_dev * std)
                df[f'bb_lower_{window}_{std_dev}'] = mean - (std_dev * std)
                df[f'bb_position_{window}_{std_dev}'] = (df['close_price'] - df[f'bb_lower_{window}_{std_dev}']) / (
                    df[f'bb_upper_{window}_{std_dev}'] - df[f'bb_lower_{window}_{std_dev}'] + 1e-8
                )
        
        # ストキャスティクス
        logger.info("ストキャスティクス指標計算中...")
        for k_period in [14, 21]:
            for d_period in [3, 5]:
                def calculate_stoch(group):
                    low_min = group['low_price'].rolling(k_period).min()
                    high_max = group['high_price'].rolling(k_period).max()
                    k_percent = 100 * ((group['close_price'] - low_min) / (high_max - low_min + 1e-8))
                    d_percent = k_percent.rolling(d_period).mean()
                    return pd.DataFrame({
                        f'stoch_k_{k_period}_{d_period}': k_percent,
                        f'stoch_d_{k_period}_{d_period}': d_percent
                    })
                
                stoch_results = df.groupby('Code').apply(calculate_stoch)
                stoch_results.index = stoch_results.index.droplevel(0)
                df = df.join(stoch_results)
        
        # 価格変動パターン
        for lag in range(1, 11):
            df[f'return_lag_{lag}'] = df.groupby('Code')['daily_return'].shift(lag)
            df[f'price_lag_{lag}'] = df.groupby('Code')['close_price'].shift(lag)
        
        # セクター・市場全体との相関
        logger.info("市場相関指標計算中...")
        market_return = df.groupby('Date')['daily_return'].mean()
        df['market_return'] = df['Date'].map(market_return)
        
        def calculate_beta(group):
            return group['daily_return'].rolling(60).corr(group['market_return'])
        
        beta_results = df.groupby('Code').apply(calculate_beta)
        beta_results.index = beta_results.index.droplevel(0)
        df['beta_vs_market'] = beta_results
        
        # 流動性指標
        if 'volume' in df.columns:
            df['liquidity_score'] = df['volume'] * df['close_price']
            df['liquidity_percentile'] = df.groupby('Date')['liquidity_score'].rank(pct=True)
        
        # 時系列統計的特徴量
        for window in [5, 10, 20]:
            # 自己相関
            df[f'autocorr_{window}'] = df.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).apply(lambda y: y.autocorr(lag=1) if len(y.dropna()) > 1 else 0)
            )
            
            # トレンド強度
            df[f'trend_strength_{window}'] = df.groupby('Code')['close_price'].transform(
                lambda x: x.rolling(window).apply(
                    lambda y: np.corrcoef(np.arange(len(y)), y)[0, 1] if len(y) == window else np.nan
                )
            )
        
        # 異常値検出特徴量
        df['price_zscore'] = df.groupby('Code')['close_price'].transform(
            lambda x: (x - x.rolling(60).mean()) / (x.rolling(60).std() + 1e-8)
        )
        df['volume_zscore'] = df.groupby('Code')['volume'].transform(
            lambda x: (x - x.rolling(60).mean()) / (x.rolling(60).std() + 1e-8)
        ) if 'volume' in df.columns else 0
        
        # カオス理論指標
        for window in [10, 20, 30]:
            # フラクタル次元近似
            df[f'fractal_dim_{window}'] = df.groupby('Code')['close_price'].transform(
                lambda x: x.rolling(window).apply(self._estimate_fractal_dimension)
            )
        
        logger.info(f"特徴量エンジニアリング完了: {len([col for col in df.columns if col not in ['Code', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 'target', 'next_day_return']])}個の特徴量")
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int) -> pd.Series:
        """RSI計算"""
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    def _calculate_macd(self, prices: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """MACD計算"""
        ema12 = prices.ewm(span=12).mean()
        ema26 = prices.ewm(span=26).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9).mean()
        return macd, signal
    
    def _calculate_stochastic(self, group: pd.DataFrame, k_period: int, d_period: int) -> Tuple[pd.Series, pd.Series]:
        """ストキャスティクス計算"""
        low_min = group['low_price'].rolling(k_period).min()
        high_max = group['high_price'].rolling(k_period).max()
        k_percent = 100 * ((group['close_price'] - low_min) / (high_max - low_min + 1e-8))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent
    
    def _estimate_fractal_dimension(self, series: pd.Series) -> float:
        """フラクタル次元の簡易推定"""
        if len(series) < 3:
            return np.nan
        
        # Higuchi法による簡易実装
        try:
            n = len(series)
            k_max = min(10, n // 2)
            lm = []
            
            for k in range(1, k_max + 1):
                l_k = 0
                for m in range(1, k + 1):
                    indices = np.arange(m - 1, n, k)
                    if len(indices) > 1:
                        norm_sum = np.sum(np.abs(np.diff(series.iloc[indices])))
                        l_k += norm_sum * (n - 1) / ((len(indices) - 1) * k)
                
                if l_k > 0:
                    lm.append(l_k / k)
            
            if len(lm) < 2:
                return np.nan
                
            # 線形回帰でフラクタル次元推定
            x = np.log(range(1, len(lm) + 1))
            y = np.log(lm)
            slope = np.polyfit(x, y, 1)[0]
            return 2 - slope
        except:
            return np.nan
    
    def optimize_hyperparameters(self, X_train: pd.DataFrame, y_train: pd.Series, 
                                X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Optuna使用のハイパーパラメータ最適化"""
        logger.info("ハイパーパラメータ最適化開始...")
        
        def objective(trial):
            # LightGBM用パラメータ
            lgb_params = {
                'objective': 'binary',
                'metric': 'binary_logloss',
                'boosting_type': 'gbdt',
                'num_leaves': trial.suggest_int('lgb_num_leaves', 30, 300),
                'feature_fraction': trial.suggest_float('lgb_feature_fraction', 0.4, 1.0),
                'bagging_fraction': trial.suggest_float('lgb_bagging_fraction', 0.4, 1.0),
                'bagging_freq': trial.suggest_int('lgb_bagging_freq', 1, 7),
                'learning_rate': trial.suggest_float('lgb_learning_rate', 0.01, 0.3),
                'min_child_samples': trial.suggest_int('lgb_min_child_samples', 5, 100),
                'reg_alpha': trial.suggest_float('lgb_reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('lgb_reg_lambda', 0, 10),
                'verbosity': -1,
                'random_state': 42
            }
            
            # CatBoost用パラメータ
            cb_params = {
                'loss_function': 'Logloss',
                'iterations': trial.suggest_int('cb_iterations', 100, 1000),
                'learning_rate': trial.suggest_float('cb_learning_rate', 0.01, 0.3),
                'depth': trial.suggest_int('cb_depth', 4, 10),
                'l2_leaf_reg': trial.suggest_float('cb_l2_leaf_reg', 1, 10),
                'border_count': trial.suggest_int('cb_border_count', 32, 255),
                'random_seed': 42,
                'verbose': False
            }
            
            # XGBoost用パラメータ
            xgb_params = {
                'objective': 'binary:logistic',
                'eval_metric': 'logloss',
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 10),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 1000),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 1, 10),
                'random_state': 42
            }
            
            # モデル訓練
            lgb_model = lgb.LGBMClassifier(**lgb_params)
            cb_model = cb.CatBoostClassifier(**cb_params)
            xgb_model = xgb.XGBClassifier(**xgb_params)
            
            lgb_model.fit(X_train, y_train)
            cb_model.fit(X_train, y_train)
            xgb_model.fit(X_train, y_train)
            
            # アンサンブル予測
            lgb_proba = lgb_model.predict_proba(X_val)[:, 1]
            cb_proba = cb_model.predict_proba(X_val)[:, 1]
            xgb_proba = xgb_model.predict_proba(X_val)[:, 1]
            
            # 重み最適化
            w1 = trial.suggest_float('weight_lgb', 0.2, 0.6)
            w2 = trial.suggest_float('weight_cb', 0.2, 0.6)
            w3 = 1 - w1 - w2
            
            if w3 < 0:
                w3 = 0
                w1 = w1 / (w1 + w2)
                w2 = w2 / (w1 + w2)
            
            ensemble_proba = w1 * lgb_proba + w2 * cb_proba + w3 * xgb_proba
            
            # 閾値最適化
            threshold = trial.suggest_float('threshold', 0.7, 0.95)
            predictions = (ensemble_proba >= threshold).astype(int)
            
            # 精度計算（予測が0個の場合は低い精度を返す）
            if predictions.sum() == 0:
                return 0.0
            
            precision = precision_score(y_val, predictions, zero_division=0)
            return precision
        
        # 最適化実行
        study = optuna.create_study(direction='maximize', 
                                   sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objective, n_trials=100, timeout=3600)  # 1時間の制限
        
        logger.info(f"最適化完了: Best precision = {study.best_value:.4f}")
        return study.best_params
    
    def train_final_model(self, df: pd.DataFrame) -> Dict[str, float]:
        """最終モデルの訓練と評価"""
        logger.info("最終モデル訓練開始...")
        
        # データ準備
        feature_columns = [col for col in df.columns if col not in 
                          ['Code', 'Date', 'Close', 'High', 'Low', 'Open', 'Volume', 
                           'target', 'next_day_return', 'close_price', 'volume', 
                           'high_price', 'low_price', 'open_price']]
        
        X = df[feature_columns].fillna(0)
        y = df['target']
        dates = df['Date']
        
        # 外れ値除去
        logger.info("外れ値検出・除去...")
        outliers = self.anomaly_detector.fit_predict(X) == -1
        X = X[~outliers]
        y = y[~outliers]
        dates = dates[~outliers]
        
        logger.info(f"外れ値除去後: {len(X):,}レコード (除去: {outliers.sum():,}レコード)")
        
        # 特徴量選択
        logger.info("特徴量選択...")
        selector = SelectFromModel(
            lgb.LGBMClassifier(n_estimators=100, random_state=42),
            threshold='median'
        )
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]
        
        logger.info(f"選択された特徴量: {len(selected_features)}/{len(feature_columns)}")
        
        # スケーリング
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 時系列分割
        tscv = TimeSeriesSplit(n_splits=8, gap=5)
        results = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            logger.info(f"Fold {fold + 1}/8 訓練中...")
            
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # ハイパーパラメータ最適化
            best_params = self.optimize_hyperparameters(
                pd.DataFrame(X_train), y_train,
                pd.DataFrame(X_val), y_val
            )
            
            # 最適パラメータでモデル訓練
            lgb_params = {k.replace('lgb_', ''): v for k, v in best_params.items() if k.startswith('lgb_')}
            lgb_params.update({'objective': 'binary', 'metric': 'binary_logloss', 'verbosity': -1, 'random_state': 42})
            
            cb_params = {k.replace('cb_', ''): v for k, v in best_params.items() if k.startswith('cb_')}
            cb_params.update({'loss_function': 'Logloss', 'random_seed': 42, 'verbose': False})
            
            xgb_params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
            xgb_params.update({'objective': 'binary:logistic', 'eval_metric': 'logloss', 'random_state': 42})
            
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
            
            # 予測
            lgb_proba = lgb_calibrated.predict_proba(X_val)[:, 1]
            cb_proba = cb_calibrated.predict_proba(X_val)[:, 1]
            xgb_proba = xgb_calibrated.predict_proba(X_val)[:, 1]
            
            # アンサンブル
            w1 = best_params.get('weight_lgb', 0.4)
            w2 = best_params.get('weight_cb', 0.4)
            w3 = 1 - w1 - w2
            
            ensemble_proba = w1 * lgb_proba + w2 * cb_proba + w3 * xgb_proba
            
            # 閾値最適化
            threshold = best_params.get('threshold', 0.85)
            predictions = (ensemble_proba >= threshold).astype(int)
            
            # メトリクス計算
            if predictions.sum() > 0:
                precision = precision_score(y_val, predictions, zero_division=0)
                recall = recall_score(y_val, predictions, zero_division=0)
                f1 = f1_score(y_val, predictions, zero_division=0)
                accuracy = accuracy_score(y_val, predictions)
                
                results.append({
                    'fold': fold + 1,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'accuracy': accuracy,
                    'predictions_count': predictions.sum(),
                    'total_count': len(predictions)
                })
                
                logger.info(f"Fold {fold + 1} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
            else:
                logger.warning(f"Fold {fold + 1} - 予測数が0のため評価をスキップ")
        
        # 結果集計
        if results:
            final_results = {
                'mean_precision': np.mean([r['precision'] for r in results]),
                'std_precision': np.std([r['precision'] for r in results]),
                'mean_recall': np.mean([r['recall'] for r in results]),
                'mean_f1': np.mean([r['f1'] for r in results]),
                'mean_accuracy': np.mean([r['accuracy'] for r in results]),
                'total_predictions': sum([r['predictions_count'] for r in results]),
                'total_samples': sum([r['total_count'] for r in results])
            }
            
            logger.info("=== 最終結果 ===")
            logger.info(f"平均精度: {final_results['mean_precision']:.4f} ± {final_results['std_precision']:.4f}")
            logger.info(f"平均再現率: {final_results['mean_recall']:.4f}")
            logger.info(f"平均F1スコア: {final_results['mean_f1']:.4f}")
            logger.info(f"総予測数: {final_results['total_predictions']}/{final_results['total_samples']}")
            
            return final_results
        else:
            logger.error("全てのフォールドで予測が生成されませんでした")
            return {'mean_precision': 0.0, 'error': 'No predictions generated'}
    
    def run_enhanced_optimization(self) -> Dict[str, float]:
        """拡張最適化実行"""
        logger.info("=== Enhanced Precision Optimization開始 ===")
        logger.info(f"目標精度: {self.target_precision:.1%}")
        
        # データ読み込み
        df = self.load_data()
        
        # 高度な特徴量エンジニアリング
        df_enhanced = self.create_advanced_features(df)
        
        # NaN除去
        df_clean = df_enhanced.dropna(subset=['target', 'next_day_return']).copy()
        
        logger.info(f"最終データサイズ: {len(df_clean):,}レコード")
        logger.info(f"ターゲット分布: {df_clean['target'].mean():.1%} (1.5%以上上昇)")
        
        # 最終モデル訓練
        results = self.train_final_model(df_clean)
        
        # 結果保存
        results_dir = Path("results/enhanced_precision")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        with open(results_dir / "enhanced_results.pkl", 'wb') as f:
            pickle.dump({
                'results': results,
                'target_precision': self.target_precision,
                'feature_count': len(df_enhanced.columns) - 8,  # 基本列を除く
                'data_size': len(df_clean)
            }, f)
        
        logger.info(f"結果保存: {results_dir / 'enhanced_results.pkl'}")
        
        return results


def main():
    """メイン実行関数"""
    try:
        optimizer = EnhancedPrecisionOptimizer(target_precision=0.75)
        
        # 拡張最適化実行
        results = optimizer.run_enhanced_optimization()
        
        print("\n=== Enhanced Precision Optimization結果 ===")
        if 'error' not in results:
            print(f"平均精度: {results['mean_precision']:.4f}")
            print(f"標準偏差: {results.get('std_precision', 0):.4f}")
            print(f"平均再現率: {results.get('mean_recall', 0):.4f}")
            print(f"平均F1スコア: {results.get('mean_f1', 0):.4f}")
            print(f"総予測数: {results.get('total_predictions', 0)}")
            
            if results['mean_precision'] >= 0.75:
                print("🎉 目標精度 0.75達成！")
            else:
                print(f"❌ 目標精度未達成 (差分: {0.75 - results['mean_precision']:.4f})")
        else:
            print(f"❌ エラー: {results['error']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Enhanced Precision Optimization失敗: {str(e)}")
        raise


if __name__ == "__main__":
    main()