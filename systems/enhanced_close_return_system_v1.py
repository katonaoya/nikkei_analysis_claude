#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Close-to-Close Precision System V1
外部データ統合 + 厳密なバックテストによる改善版

改善点：
1. 外部経済指標データの統合（USD/JPY, VIX, 日経225指数等）
2. ウォークフォワード最適化による厳密なバックテスト
3. 複雑性を抑えたシンプルな実装
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import joblib
from datetime import datetime, timedelta
import logging
import argparse
import json
import warnings
import os
from pathlib import Path
from typing import Optional, Tuple

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class CloseReturnPrecisionSystemV1:
    """外部データ統合 + 厳密バックテスト版"""
    
    def __init__(
        self,
        stock_file: str = None,
        external_file: str = None,
        target_return: float = 0.01,
        imbalance_boost: float = 1.0,
        imbalance_strategy: str = "scale_pos",
        focal_gamma: float = 2.0,
        positive_oversample_ratio: float = 1.0,
        calibration_method: str = "platt",
        calibration_window_days: int = 45,
        calibration_min_samples: int = 4500,
        calibration_min_positives: int = 500,
        calibration_min_negatives: int = 500,
    ):
        """初期化"""
        # デフォルトファイルパス（動的に最新ファイルを取得）
        if stock_file is None:
            stock_file = self._find_latest_stock_file()
        if external_file is None:
            external_file = self._find_latest_external_file()
            
        self.stock_file = stock_file
        self.external_file = external_file
        self.target_return = target_return
        self.imbalance_boost = imbalance_boost
        self.imbalance_strategy = imbalance_strategy
        self.focal_gamma = focal_gamma
        self.positive_oversample_ratio = max(positive_oversample_ratio, 1.0)
        self._rng = np.random.default_rng(42)

        calibration_method_normalized = (calibration_method or "none").lower()
        valid_calibration_methods = {"platt", "isotonic", "none"}
        if calibration_method_normalized not in valid_calibration_methods:
            logger.warning(
                "未知のキャリブレーション手法 '%s' が指定されたため 'platt' を使用します",
                calibration_method,
            )
            calibration_method_normalized = "platt"
        self.calibration_method = calibration_method_normalized
        self.calibration_window_days = max(int(calibration_window_days or 0), 0)
        self.calibration_min_samples = max(int(calibration_min_samples or 0), 0)
        self.calibration_min_positives = max(int(calibration_min_positives or 0), 0)
        self.calibration_min_negatives = max(int(calibration_min_negatives or 0), 0)
        
        # 保存ディレクトリ
        self.output_dir = Path("models/enhanced_close_v1")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_params = self._load_model_params()

        logger.info("🎯 Close-to-Close Precision System V1 初期化完了")
        logger.info(f"株価データ: {self.stock_file}")
        logger.info(f"外部データ: {self.external_file}")
        logger.info(f"クラス重みブースト係数: {self.imbalance_boost:.3f}")
        logger.info(f"不均衡戦略: {self.imbalance_strategy}")
        if self.imbalance_strategy == "focal":
            logger.info(f"Focal Gamma: {self.focal_gamma:.2f}")
        if self.positive_oversample_ratio > 1.0:
            logger.info(f"Positive oversample ratio: {self.positive_oversample_ratio:.2f}")
        if self.calibration_method != "none":
            logger.info(
                "キャリブレーション設定: method=%s, window_days=%d, min_samples=%d, min_pos=%d, min_neg=%d",
                self.calibration_method,
                self.calibration_window_days,
                self.calibration_min_samples,
                self.calibration_min_positives,
                self.calibration_min_negatives,
            )

    def _load_model_params(self) -> dict:
        params_path = Path('config/close_model_params.json')
        if params_path.exists():
            with params_path.open() as f:
                return json.load(f)
        return {
            'n_estimators': 300,
            'learning_rate': 0.03,
            'max_depth': 8,
            'num_leaves': 63,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'min_child_samples': 20
        }

    def _find_latest_stock_file(self) -> str:
        """最新の株価データファイルを取得"""
        import glob
        
        # 複数のパターンを試す
        patterns = [
            "data/processed/nikkei225_complete_*.parquet",
            "data/real_jquants_data/nikkei225_real_data_*.pkl",
            "data/processed/nikkei225_*.parquet"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    file_time = os.path.getmtime(file)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file
                except:
                    continue
        
        if latest_file is None:
            # フォールバック: 固定ファイル名
            latest_file = "data/processed/nikkei225_complete_225stocks_20250909_230649.parquet"
            logger.warning(f"最新株価ファイルが見つからないため、固定ファイルを使用: {latest_file}")
        else:
            logger.info(f"最新株価ファイル取得: {latest_file}")
        
        return latest_file
    
    def _find_latest_external_file(self) -> str:
        """最新の外部データファイルを取得"""
        import glob
        
        # 複数のパターンを試す
        patterns = [
            "data/external_extended/external_integrated_*.parquet",
            "data/processed/enhanced_integrated_data.parquet",
            "data/processed/external_*.parquet"
        ]
        
        latest_file = None
        latest_time = 0
        
        for pattern in patterns:
            files = glob.glob(pattern)
            for file in files:
                try:
                    file_time = os.path.getmtime(file)
                    if file_time > latest_time:
                        latest_time = file_time
                        latest_file = file
                except:
                    continue
        
        if latest_file is None:
            # フォールバック: 固定ファイル名
            latest_file = "data/external_extended/external_integrated_10years_20250909_231815.parquet"
            logger.warning(f"最新外部データファイルが見つからないため、固定ファイルを使用: {latest_file}")
        else:
            logger.info(f"最新外部データファイル取得: {latest_file}")
        
        return latest_file
    
    def load_and_integrate_data(self) -> pd.DataFrame:
        """データ読み込みと統合"""
        logger.info("📊 データ読み込み開始...")
        
        # 株価データ読み込み
        stock_df = pd.read_parquet(self.stock_file)
        logger.info(f"株価データ: {len(stock_df):,}件, {stock_df['Code'].nunique()}銘柄")

        # 調整後終値・高値・安値・出来高を優先利用
        if 'AdjustmentClose' in stock_df.columns:
            stock_df['Close_raw'] = stock_df['Close']
            stock_df['High_raw'] = stock_df['High']
            stock_df['Low_raw'] = stock_df['Low']
            stock_df['Volume_raw'] = stock_df['Volume']
            stock_df['Close'] = stock_df['AdjustmentClose']
            stock_df['High'] = stock_df.get('AdjustmentHigh', stock_df['High'])
            stock_df['Low'] = stock_df.get('AdjustmentLow', stock_df['Low'])
            stock_df['Volume'] = stock_df.get('AdjustmentVolume', stock_df['Volume'])
            logger.info('Adjusted price columns applied')
            stock_df['TargetReturn'] = getattr(self, 'target_return', 0.01)

        # 外部データ読み込み（ファイルが存在する場合のみ）
        external_df = None
        if os.path.exists(self.external_file):
            try:
                external_df = pd.read_parquet(self.external_file)
                logger.info(f"外部データ: {len(external_df):,}件, {len(external_df.columns)}カラム")
            except Exception as e:
                logger.warning(f"外部データ読み込みエラー: {e}")
                external_df = None
        else:
            logger.warning(f"外部データファイルが見つかりません: {self.external_file}")
        
        # 日付型統一
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
        
        # 外部データ統合（サイズに関わらず重要指標のみを取り込む）
        if external_df is not None:
            try:
                external_df['Date'] = pd.to_datetime(external_df['Date']).dt.tz_localize(None)
                
                # 重要指標キーワードにマッチする列を優先的に選択
                keyword_candidates = ['usdjpy', 'vix', 'nikkei', 'sp500', 'topix', 'dow', 'nasdaq', 'commodity', 'yield']
                selected_cols = [
                    col for col in external_df.columns
                    if any(key in col.lower() for key in keyword_candidates)
                ]

                # フォールバック: 数値列のうち最多非欠損トップ N を採用
                if not selected_cols:
                    numeric_cols = external_df.select_dtypes(include=[np.number]).columns.tolist()
                    non_null_counts = external_df[numeric_cols].count().sort_values(ascending=False)
                    selected_cols = non_null_counts.head(25).index.tolist()

                # 統合する列数を上限化して学習負荷を抑える
                max_external_features = 30
                if len(selected_cols) > max_external_features:
                    selected_cols = selected_cols[:max_external_features]

                external_selected = external_df[['Date'] + selected_cols].copy()

                # 重複日付が存在する場合は最終行を採用
                if external_selected['Date'].duplicated().any():
                    external_selected = external_selected.sort_values('Date')
                    external_selected = external_selected.groupby('Date').tail(1)

                stock_df = pd.merge(stock_df, external_selected, on='Date', how='left')
                logger.info(f"外部データ統合完了: {len(selected_cols)}指標")

            except Exception as e:
                logger.warning(f"外部データ統合エラー: {e}")
        else:
            logger.info("外部データ統合をスキップ（外部データ未検出）")
        
        logger.info(f"統合後データ: {len(stock_df):,}件, {len(stock_df.columns)}カラム")
        return stock_df
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """拡張特徴量作成（メモリ最適化版）"""
        logger.info("🔥 拡張特徴量エンジニアリング開始...")
        
        # 全期間のデータを使用（既存構成を踏襲）
        df_recent = df.copy()

        logger.info(f"全期間データ使用: {len(df_recent):,}件（約10年間）")

        # 処理用のDataFrameを準備（銘柄単位で処理してリークを防止）
        enhanced_dfs = []
        unique_codes = df_recent['Code'].unique()
        logger.info(f"全銘柄一括処理: {len(unique_codes)}銘柄")

        for code in unique_codes:
            code_data = df_recent[df_recent['Code'] == code].copy().sort_values('Date')
            
            if len(code_data) < 50:  # データが少ない銘柄はスキップ
                continue
           
            # 基本特徴量（必要最小限）
            code_data['Returns'] = code_data['Close'].pct_change()
            code_data['High_Low_Ratio'] = code_data['High'] / code_data['Low']
                
            # 移動平均（重要な期間のみ）
            for window in [5, 20]:
                code_data[f'MA_{window}'] = code_data['Close'].rolling(window).mean()
                code_data[f'MA_{window}_ratio'] = code_data['Close'] / code_data[f'MA_{window}']
                
            # ボラティリティ（1つのwindowのみ）
            code_data['Volatility_20'] = code_data['Returns'].rolling(20).std()
                
            # RSI（1つのwindowのみ）
            window = 14
            delta = code_data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / loss
            code_data['RSI_14'] = 100 - (100 / (1 + rs))
                
            # MACD（シンプル版）
            exp1 = code_data['Close'].ewm(span=12).mean()
            exp2 = code_data['Close'].ewm(span=26).mean()
            code_data['MACD'] = exp1 - exp2
                
            # ボリューム特徴量
            code_data['Volume_MA_20'] = code_data['Volume'].rolling(20).mean()
            code_data['Volume_ratio'] = code_data['Volume'] / code_data['Volume_MA_20']
            code_data['Volume_zscore_20'] = (code_data['Volume'] - code_data['Volume_MA_20']) / code_data['Volume'].rolling(20).std()

            # 終値ベースの追加特徴量
            code_data['Close_gap_prev'] = code_data['Open'] / code_data['Close'].shift(1) - 1
            code_data['Returns_3d'] = code_data['Close'].pct_change(periods=3)
            code_data['Volatility_5'] = code_data['Returns'].rolling(5).std()
            if 'nikkei_change' in code_data.columns:
                code_data['Relative_to_index'] = code_data['Returns'] - code_data['nikkei_change']
            else:
                code_data['Relative_to_index'] = code_data['Returns']

            # 外部データ特徴量は最小限に
            for col in code_data.columns:
                if any(key in col.lower() for key in ['usdjpy', 'vix']):
                    if code_data[col].notna().sum() > 50:
                        code_data[f'{col}_change'] = code_data[col].pct_change()

            enhanced_dfs.append(code_data)

        if not enhanced_dfs:
            raise ValueError("特徴量を生成できる銘柄がありません")

        enhanced_df = pd.concat(enhanced_dfs, ignore_index=True)
        
        # 目的変数作成
        logger.info("目的変数作成...")
        enhanced_df['Target'] = 0
        threshold = 1.0 + float(self.target_return)
        for code in enhanced_df['Code'].unique():
            mask = enhanced_df['Code'] == code
            code_data = enhanced_df[mask].copy().sort_values('Date')
            next_close = code_data['Close'].shift(-1)
            current_close = code_data['Close']
            target = (next_close / current_close >= threshold).astype(int)
            enhanced_df.loc[code_data.index, 'Target'] = target
        
        # 無限値・欠損値処理
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.dropna(subset=['Close', 'Date', 'Code', 'Target'])
        
        logger.info(f"🔥 特徴量作成完了: {len(enhanced_df):,}件")
        logger.info(f"特徴量数: {len(enhanced_df.columns)}カラム")
        logger.info(f"正例率: {enhanced_df['Target'].mean():.3f}")
        
        return enhanced_df

    def _compute_scale_pos_weight(self, y: pd.Series) -> float:
        """LightGBM用のscale_pos_weightを計算（ブースト係数に対応）"""
        # yは0/1のSeriesを想定
        if not isinstance(y, pd.Series):
            y = pd.Series(y)
        pos_count = float(y.sum())
        total = len(y)
        neg_count = total - pos_count
        if pos_count <= 0 or neg_count <= 0:
            base_weight = 1.0
        else:
            base_weight = neg_count / pos_count

        boost = max(float(self.imbalance_boost), 1e-3)
        adjusted = max(base_weight * boost, 1e-3)
        logger.debug(
            "scale_pos_weight計算: 正例=%d, 負例=%d, base=%.4f, boost=%.3f, adjusted=%.4f",
            int(pos_count),
            int(neg_count),
            base_weight,
            boost,
            adjusted
        )
        return adjusted

    def _compute_sample_weights(self, y: pd.Series) -> Optional[np.ndarray]:
        """不均衡戦略に基づきサンプル重みを計算"""
        strategy = (self.imbalance_strategy or "").lower()
        if strategy in ("", "none", "scale_pos"):
            return None

        if not isinstance(y, pd.Series):
            y = pd.Series(y)

        positives = float(y.sum())
        total = len(y)
        negatives = total - positives
        epsilon = 1e-9

        if strategy == "balanced":
            if positives == 0 or negatives == 0:
                return np.ones(total)
            pos_weight = total / (2 * positives)
            neg_weight = total / (2 * negatives)
            return np.where(y == 1, pos_weight, neg_weight)

        if strategy == "focal":
            gamma = max(self.focal_gamma, 0.0)
            if positives == 0 or negatives == 0:
                return np.ones(total)
            pos_rate = positives / total
            neg_rate = max(1.0 - pos_rate, epsilon)
            pos_weight = (neg_rate) ** gamma
            neg_weight = (pos_rate + epsilon) ** gamma
            # 正例を強調するためブーストも掛け合わせ
            pos_weight *= self.imbalance_boost
            return np.where(y == 1, pos_weight, neg_weight)

        return None

    def _apply_positive_oversample(self, X: pd.DataFrame, y: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """正例を単純リサンプリングで水増し"""
        ratio = float(self.positive_oversample_ratio)
        if ratio <= 1.0 or y.sum() == 0:
            return X, y

        positives = y[y == 1]
        current_pos = len(positives)
        target_pos = int(min(len(y) * ratio, current_pos * ratio))
        additional = target_pos - current_pos
        if additional <= 0:
            return X, y

        pos_indices = positives.index.to_numpy()
        sampled_idx = self._rng.choice(pos_indices, size=additional, replace=True)
        X_extra = X.loc[sampled_idx].copy()
        y_extra = y.loc[sampled_idx].copy()

        X_aug = pd.concat([X, X_extra], axis=0, ignore_index=True)
        y_aug = pd.concat([y, y_extra], axis=0, ignore_index=True)
        return X_aug, y_aug

    def _split_calibration_holdout(
        self,
        df_sorted: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """キャリブレーション用のホールドアウトを日付ベースで分離"""
        if self.calibration_method == "none":
            return df_sorted, None

        unique_dates = np.array(sorted(pd.Series(df_sorted['Date']).unique()))
        total_dates = len(unique_dates)
        if total_dates == 0:
            return df_sorted, None

        desired_days = self.calibration_window_days or max(int(total_dates * 0.15), 1)
        calibration_days = min(desired_days, total_dates)

        min_train_dates = max(int(total_dates * 0.5), 60)
        while calibration_days > 0 and total_dates - calibration_days < min_train_dates:
            calibration_days -= 1

        if calibration_days <= 0:
            logger.info("キャリブレーション用の期間が確保できなかったためスキップします")
            return df_sorted, None

        calibration_dates = unique_dates[-calibration_days:]
        mask = df_sorted['Date'].isin(calibration_dates)
        calibration_df = df_sorted[mask].copy()
        train_df = df_sorted[~mask].copy()

        if calibration_df.empty:
            logger.info("キャリブレーション用データが抽出できなかったためスキップします")
            return df_sorted, None

        total_samples = len(calibration_df)
        positive_count = int(calibration_df['Target'].sum())
        negative_count = int(total_samples - positive_count)

        if self.calibration_min_samples and total_samples < self.calibration_min_samples:
            logger.warning(
                "キャリブレーションサンプルが不足 (%d件 < 最低%d件) のためスキップします",
                total_samples,
                self.calibration_min_samples,
            )
            return df_sorted, None

        if self.calibration_min_positives and positive_count < self.calibration_min_positives:
            logger.warning(
                "キャリブレーション用正例が不足 (%d件 < 最低%d件) のためスキップします",
                positive_count,
                self.calibration_min_positives,
            )
            return df_sorted, None

        if self.calibration_min_negatives and negative_count < self.calibration_min_negatives:
            logger.warning(
                "キャリブレーション用負例が不足 (%d件 < 最低%d件) のためスキップします",
                negative_count,
                self.calibration_min_negatives,
            )
            return df_sorted, None

        logger.info(
            "📏 キャリブレーションホールドアウト確保: %s〜%s / %d件 (正例率 %.3f)",
            calibration_df['Date'].min().date(),
            calibration_df['Date'].max().date(),
            total_samples,
            calibration_df['Target'].mean(),
        )

        return train_df, calibration_df

    @staticmethod
    def _build_calibration_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10):
        y_true_arr = np.asarray(y_true)
        y_prob_arr = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(y_prob_arr, bins, right=False) - 1

        total = len(y_true_arr)
        bin_data = []
        ece = 0.0
        mce = 0.0

        for idx in range(n_bins):
            mask = bin_indices == idx
            count = int(mask.sum())
            lower = float(bins[idx])
            upper = float(bins[idx + 1])
            if count == 0:
                bin_data.append(
                    {
                        'lower': lower,
                        'upper': upper,
                        'count': 0,
                        'avg_pred': None,
                        'actual_rate': None,
                    }
                )
                continue

            avg_pred = float(y_prob_arr[mask].mean())
            actual_rate = float(y_true_arr[mask].mean())
            diff = abs(actual_rate - avg_pred)
            weight = count / total if total else 0.0
            ece += weight * diff
            mce = max(mce, diff)
            bin_data.append(
                {
                    'lower': lower,
                    'upper': upper,
                    'count': count,
                    'avg_pred': avg_pred,
                    'actual_rate': actual_rate,
                }
            )

        return bin_data, float(ece), float(mce)

    def _compute_calibration_metrics(
        self,
        y_true: np.ndarray,
        raw_proba: np.ndarray,
        calibrated_proba: np.ndarray,
    ) -> dict:
        metrics = {}
        try:
            metrics['brier_raw'] = float(brier_score_loss(y_true, raw_proba))
            metrics['brier_calibrated'] = float(brier_score_loss(y_true, calibrated_proba))
        except ValueError:
            metrics['brier_raw'] = float('nan')
            metrics['brier_calibrated'] = float('nan')

        bins_raw, ece_raw, mce_raw = self._build_calibration_bins(y_true, raw_proba)
        bins_cal, ece_cal, mce_cal = self._build_calibration_bins(y_true, calibrated_proba)

        metrics['ece_raw'] = ece_raw
        metrics['ece_calibrated'] = ece_cal
        metrics['mce_raw'] = mce_raw
        metrics['mce_calibrated'] = mce_cal
        metrics['bins_raw'] = bins_raw
        metrics['bins_calibrated'] = bins_cal
        metrics['sample_size'] = int(len(y_true))
        metrics['positive_rate'] = float(np.mean(y_true)) if len(y_true) else float('nan')
        return metrics

    def _fit_calibrator(
        self,
        model: lgb.LGBMClassifier,
        selector: Optional[SelectKBest],
        scaler: Optional[RobustScaler],
        feature_cols: list,
        calibration_df: Optional[pd.DataFrame],
    ) -> dict:
        if calibration_df is None or self.calibration_method == "none":
            return {'method': 'none', 'model': None, 'reason': 'not_available'}

        X_cal = calibration_df[feature_cols].copy()
        y_cal = calibration_df['Target'].astype(int)

        if selector is not None:
            X_cal = selector.transform(X_cal)
        if scaler is not None:
            X_cal = scaler.transform(X_cal)

        raw_proba = model.predict_proba(X_cal)[:, 1]
        raw_proba = np.clip(raw_proba, 1e-6, 1 - 1e-6)

        method = self.calibration_method
        calibrator_model = None
        calibrated_proba = raw_proba

        try:
            if method == 'platt':
                calibrator_model = LogisticRegression(solver='lbfgs', max_iter=1000)
                calibrator_model.fit(raw_proba.reshape(-1, 1), y_cal)
                calibrated_proba = calibrator_model.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
            elif method == 'isotonic':
                calibrator_model = IsotonicRegression(out_of_bounds='clip')
                calibrator_model.fit(raw_proba, y_cal)
                calibrated_proba = calibrator_model.predict(raw_proba)
            else:
                return {'method': 'none', 'model': None, 'reason': 'unsupported_method'}
        except Exception as exc:
            logger.warning("キャリブレーション学習に失敗したためスキップします: %s", exc)
            return {'method': 'none', 'model': None, 'reason': 'fit_failed', 'error': str(exc)}

        metrics = self._compute_calibration_metrics(y_cal.to_numpy(), raw_proba, calibrated_proba)

        if metrics.get('brier_raw') is not None and metrics.get('brier_calibrated') is not None:
            logger.info(
                "🎯 キャリブレーション指標: Brier %.4f → %.4f / ECE %.4f → %.4f",
                metrics['brier_raw'],
                metrics['brier_calibrated'],
                metrics['ece_raw'],
                metrics['ece_calibrated'],
            )

        calibration_info = {
            'method': method,
            'model': calibrator_model,
            'metrics': metrics,
            'holdout': {
                'start_date': str(calibration_df['Date'].min().date()),
                'end_date': str(calibration_df['Date'].max().date()),
                'sample_size': metrics.get('sample_size', len(calibration_df)),
                'positive_rate': metrics.get('positive_rate'),
            },
            'reason': None,
        }

        if method == 'platt' and calibrator_model is not None:
            calibration_info['coef'] = float(calibrator_model.coef_[0][0])
            calibration_info['intercept'] = float(calibrator_model.intercept_[0])

        return calibration_info

    @staticmethod
    def apply_calibration(probabilities, calibration_info: Optional[dict]):
        """保存されたキャリブレーション情報を用いて確率を調整"""
        if not calibration_info:
            return probabilities

        probs_array = np.asarray(probabilities, dtype=float)
        was_scalar = probs_array.shape == ()
        probs_flat = probs_array.reshape(-1)

        method = (calibration_info.get('method') or calibration_info.get('type') or 'platt').lower()

        try:
            if method == 'platt':
                model = calibration_info.get('model')
                if model is not None:
                    calibrated = model.predict_proba(probs_flat.reshape(-1, 1))[:, 1]
                elif 'coef' in calibration_info and 'intercept' in calibration_info:
                    linear = calibration_info['coef'] * probs_flat + calibration_info['intercept']
                    calibrated = 1.0 / (1.0 + np.exp(-linear))
                else:
                    return probabilities
            elif method == 'isotonic':
                model = calibration_info.get('model')
                if model is None:
                    return probabilities
                calibrated = model.predict(probs_flat)
            elif method == 'none':
                return probabilities
            else:
                return probabilities
        except Exception:
            return probabilities

        calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
        if was_scalar:
            return float(calibrated[0])
        return calibrated.reshape(probs_array.shape)
    
    def walk_forward_optimization(self, df: pd.DataFrame, initial_train_size: int = 252*2) -> list:
        """ウォークフォワード最適化（メモリ最適化版）"""
        logger.info("📈 ウォークフォワード最適化開始...")
        
        # 全量データで検証（再現性と陽性サンプルを最大限活用）
        df_sampled = df.copy()
        logger.info(f"ウォークフォワード入力データ: {len(df_sampled):,}件（全量使用）")

        # ソートして銘柄毎に欠損を前方補完
        df_sorted = df_sampled.sort_values(['Code', 'Date']).reset_index(drop=True)
        feature_cols = [
            col for col in df_sorted.columns
            if col not in ['Date', 'Code', 'Target'] and str(df_sorted[col].dtype) in ['int64', 'float64', 'int32', 'float32']
        ]
        df_sorted[feature_cols] = (
            df_sorted.groupby('Code')[feature_cols]
            .apply(lambda g: g.fillna(method='ffill'))
            .reset_index(level=0, drop=True)
        )
        df_sorted[feature_cols] = df_sorted[feature_cols].fillna(0)

        unique_dates = sorted(df_sorted['Date'].unique())
        
        results = []
        step_size = 21  # 約1ヶ月リバランスで期間解像度を向上
        
        # 特徴量カラム選択（重要な特徴量のみ）
        # feature_cols は前段で定義済み

        # 特徴量数制限
        if len(feature_cols) > 30:
            # 欠損値が少ない特徴量を優先選択
            non_null_counts = df_sorted[feature_cols].count()
            top_features = non_null_counts.nlargest(30).index.tolist()
            feature_cols = top_features
        
        logger.info(f"使用特徴量数: {len(feature_cols)}")
        logger.info(f"評価ステップ幅: {step_size}営業日ごと")
        
        # 初期サイズを小さく設定
        initial_train_size = min(initial_train_size, len(unique_dates) // 3)
        
        for i in range(initial_train_size, len(unique_dates) - step_size, step_size):
            try:
                # 期間設定
                train_end_idx = i
                test_start_idx = i
                test_end_idx = min(i + step_size, len(unique_dates))
                
                train_dates = unique_dates[:train_end_idx]
                test_dates = unique_dates[test_start_idx:test_end_idx]
                
                # データ分割
                train_df = df_sorted[df_sorted['Date'].isin(train_dates)]
                test_df = df_sorted[df_sorted['Date'].isin(test_dates)]
                
                if len(train_df) == 0 or len(test_df) == 0:
                    continue
                
                # 特徴量・目的変数分離
                X_train = train_df[feature_cols].copy()
                y_train = train_df['Target'].copy()
                X_test = test_df[feature_cols].copy()
                y_test = test_df['Target'].copy()

                # オーバーサンプリング
                X_train, y_train = self._apply_positive_oversample(X_train, y_train)
                sample_weight = self._compute_sample_weights(y_train)

                # 特徴量選択（既存構成を踏襲）
                selector = SelectKBest(score_func=f_classif, k=min(30, len(feature_cols)))
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # スケーリング
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_selected)
                X_test_scaled = scaler.transform(X_test_selected)
                
                # モデル学習（パラメータ軽量化）
                scale_pos_weight = self._compute_scale_pos_weight(y_train)
                if self.imbalance_strategy and self.imbalance_strategy.lower() not in ("", "none", "scale_pos"):
                    scale_pos_weight = 1.0

                params = self.model_params.copy()
                params.update({
                    'objective': 'binary',
                    'random_state': 42,
                    'verbose': -1,
                    'scale_pos_weight': scale_pos_weight
                })
                model = lgb.LGBMClassifier(**params)
                
                model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
                
                # 予測
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # 評価
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                result = {
                    'period': f"{train_dates[-1].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}",
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'positive_rate': y_test.mean()
                }
                
                results.append(result)
                logger.info(f"期間 {result['period']}: 精度={accuracy:.4f}, 適合率={precision:.4f}")
                
            except Exception as e:
                logger.warning(f"期間 {i} でエラー: {e}")
                continue
        
        return results
    
    def train_final_model(self, df: pd.DataFrame) -> dict:
        """最終モデル学習（メモリ最適化版）"""
        logger.info("🤖 最終モデル学習開始...")
        
        # 全量データを使用（サンプリングを廃止）
        df_sampled = df.copy()
        logger.info(f"最終学習データ件数: {len(df_sampled):,}件（全量使用）")

        # 特徴量準備
        df_sorted = df_sampled.sort_values(['Code', 'Date']).reset_index(drop=True)
        feature_cols = [
            col for col in df_sorted.columns
            if col not in ['Date', 'Code', 'Target'] and str(df_sorted[col].dtype) in ['int64', 'float64', 'int32', 'float32']
        ]

        # 特徴量数制限
        if len(feature_cols) > 25:
            non_null_counts = df_sorted[feature_cols].count()
            feature_cols = non_null_counts.nlargest(25).index.tolist()

        df_sorted[feature_cols] = (
            df_sorted.groupby('Code')[feature_cols]
            .apply(lambda g: g.fillna(method='ffill'))
            .reset_index(level=0, drop=True)
        )
        df_sorted[feature_cols] = df_sorted[feature_cols].fillna(0)

        train_df, calibration_df = self._split_calibration_holdout(df_sorted)
        training_df = train_df if calibration_df is not None else df_sorted

        X = training_df[feature_cols].copy()
        y = training_df['Target'].copy()
        X, y = self._apply_positive_oversample(X, y)

        dataset_size = len(X)
        if dataset_size < 10:
            raise ValueError("学習に十分なサンプル数がありません")

        split_idx = int(dataset_size * 0.8)
        if split_idx <= 0 or split_idx >= dataset_size:
            split_idx = dataset_size // 2

        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        # 特徴量選択（既存構成を踏襲）
        selector = SelectKBest(score_func=f_classif, k=min(30, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # モデル学習（既存構成を踏襲）
        scale_pos_weight = self._compute_scale_pos_weight(y_train)
        if self.imbalance_strategy and self.imbalance_strategy.lower() not in ("", "none", "scale_pos"):
            scale_pos_weight = 1.0
        sample_weight = self._compute_sample_weights(y_train)

        params = self.model_params.copy()
        params.update({
            'objective': 'binary',
            'random_state': 42,
            'verbose': -1,
            'scale_pos_weight': scale_pos_weight
        })
        model = lgb.LGBMClassifier(**params)
        
        logger.info(f"クラス重み (scale_pos_weight): {scale_pos_weight:.2f}")

        model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
        
        # 予測・評価
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

        calibration_info = self._fit_calibrator(model, selector, scaler, feature_cols, calibration_df)
        if calibration_info.get('method') == 'none' and calibration_info.get('reason'):
            logger.info("キャリブレーションをスキップしました: %s", calibration_info.get('reason'))
        calibrated_proba = self.apply_calibration(y_pred_proba, calibration_info)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"🎯 最終モデル性能:")
        logger.info(f"  精度: {accuracy:.4f}")
        logger.info(f"  適合率: {precision:.4f}")
        logger.info(f"  再現率: {recall:.4f}")
        logger.info(f"  F1スコア: {f1:.4f}")
        
        # モデル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_data = {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'target_return': self.target_return,
            'model_params': self.model_params,
            'imbalance_boost': self.imbalance_boost,
            'imbalance_strategy': self.imbalance_strategy,
            'focal_gamma': self.focal_gamma,
            'positive_oversample_ratio': self.positive_oversample_ratio,
            'calibration_method': self.calibration_method,
            'calibration_window_days': self.calibration_window_days,
            'calibration_min_samples': self.calibration_min_samples,
            'calibration_min_positives': self.calibration_min_positives,
            'calibration_min_negatives': self.calibration_min_negatives,
            'calibration': calibration_info
        }
        
        model_file = self.output_dir / f"close_model_v1_{accuracy:.4f}acc_{timestamp}.joblib"
        joblib.dump(model_data, model_file)
        logger.info(f"🎯 モデル保存完了: {model_file}")
        
        return model_data
    
    def run_enhanced_system(self):
        """拡張システム実行"""
        logger.info("🚀 Close-to-Close Precision System V1 実行開始!")
        
        try:
            # データ統合
            df = self.load_and_integrate_data()
            
            # 特徴量作成
            enhanced_df = self.create_enhanced_features(df)
            
            # ウォークフォワード最適化
            wfo_results = self.walk_forward_optimization(enhanced_df)
            
            # 最終モデル学習
            final_model = self.train_final_model(enhanced_df)
            
            # 結果統計
            if wfo_results:
                wfo_accuracies = [r['accuracy'] for r in wfo_results]
                wfo_mean_acc = np.mean(wfo_accuracies)
                wfo_std_acc = np.std(wfo_accuracies)
                
                logger.info(f"\n📊 ウォークフォワード最適化結果:")
                logger.info(f"  期間数: {len(wfo_results)}")
                logger.info(f"  平均精度: {wfo_mean_acc:.4f} ± {wfo_std_acc:.4f}")
                logger.info(f"  最高精度: {max(wfo_accuracies):.4f}")
                logger.info(f"  最低精度: {min(wfo_accuracies):.4f}")
            
            # 結果保存
            results = {
                'final_model_accuracy': final_model['accuracy'],
                'wfo_mean_accuracy': wfo_mean_acc if wfo_results else 0,
                'wfo_std_accuracy': wfo_std_acc if wfo_results else 0,
                'wfo_results': wfo_results,
                'data_size': len(enhanced_df),
                'feature_count': len(final_model['feature_cols']),
                'target_return': self.target_return,
                'model_params': self.model_params,
                'external_data_integrated': os.path.exists(self.external_file),
                'imbalance_strategy': self.imbalance_strategy,
                'positive_oversample_ratio': self.positive_oversample_ratio,
                'calibration_method': self.calibration_method,
                'calibration_window_days': self.calibration_window_days,
                'calibration_min_samples': self.calibration_min_samples,
                'calibration_min_positives': self.calibration_min_positives,
                'calibration_min_negatives': self.calibration_min_negatives,
                'calibration_metrics': final_model.get('calibration', {}).get('metrics') if isinstance(final_model.get('calibration'), dict) else None,
            }
            
            results_file = self.output_dir / f"close_results_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(results, results_file)
            
            logger.info(f"🎉 Close-to-Close Precision System V1 完了!")
            logger.info(f"最終精度: {final_model['accuracy']:.4f}")
            logger.info(f"データ統合: {'✅' if results['external_data_integrated'] else '❌'}")
            logger.info(f"結果保存: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"システム実行エラー: {e}")
            return None

def main():
    """メイン実行"""
    parser = argparse.ArgumentParser(description='Close-to-Close Precision System V1')
    parser.add_argument('--target-return', type=float, default=0.01, help='終値ベースの判定閾値 (例: 0.01=+1%)')
    parser.add_argument('--imbalance-boost', type=float, default=1.0, help='クラス不均衡対策として scale_pos_weight に掛ける倍率')
    parser.add_argument('--imbalance-strategy', type=str, default='scale_pos', choices=['scale_pos', 'balanced', 'focal', 'none'], help='追加のサンプル重み戦略')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='focal戦略用ガンマ値 (imbalance-strategy=focal のみ使用)')
    parser.add_argument('--positive-oversample-ratio', type=float, default=1.0, help='正例の単純オーバーサンプリング倍率 (>1で増幅)')
    parser.add_argument('--calibration-method', type=str, default='platt', choices=['platt', 'isotonic', 'none'], help='予測確率キャリブレーション手法')
    parser.add_argument('--calibration-window-days', type=int, default=45, help='キャリブレーション用に確保する営業日数（終端側）')
    parser.add_argument('--calibration-min-samples', type=int, default=4500, help='キャリブレーションを実施するために必要な最小サンプル数')
    parser.add_argument('--calibration-min-positives', type=int, default=500, help='キャリブレーションに必要な最小正例件数')
    parser.add_argument('--calibration-min-negatives', type=int, default=500, help='キャリブレーションに必要な最小負例件数')
    args = parser.parse_args()

    system = CloseReturnPrecisionSystemV1(
        target_return=args.target_return,
        imbalance_boost=args.imbalance_boost,
        imbalance_strategy=args.imbalance_strategy,
        focal_gamma=args.focal_gamma,
        positive_oversample_ratio=args.positive_oversample_ratio,
        calibration_method=args.calibration_method,
        calibration_window_days=args.calibration_window_days,
        calibration_min_samples=args.calibration_min_samples,
        calibration_min_positives=args.calibration_min_positives,
        calibration_min_negatives=args.calibration_min_negatives,
    )
    results = system.run_enhanced_system()
    
    if results:
        print(f"\n✅ Close-to-Close Precision System V1 実行完了!")
        print(f"📊 最終精度: {results['final_model_accuracy']:.4f}")
        if results['wfo_mean_accuracy'] > 0:
            print(f"📈 ウォークフォワード平均精度: {results['wfo_mean_accuracy']:.4f}")
        print(f"📁 データ統合: {'成功' if results['external_data_integrated'] else '外部データなし'}")
        print(f"📊 データ量: {results['data_size']:,}件")
        print(f"🔧 特徴量数: {results['feature_count']}個")
    else:
        print("\n❌ システム実行に失敗しました")

if __name__ == "__main__":
    main()
