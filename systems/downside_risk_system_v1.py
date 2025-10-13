#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Downside / Risk モデル (Codex-B 担当)

- 終値データから短期リスク指標を生成し、下落確率を推定
- 出力は `production_dir` 直下に `downside_predictions.parquet` / `risk_predictions.parquet`
- 学習は軽量なロジスティック回帰（不足時はヒューリスティック）
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "return_1d",
    "return_3d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_30d",
    "market_return_1d",
    "relative_return_1d",
    "volatility_5",
    "volatility_10",
    "volatility_20",
    "drawdown_5",
    "drawdown_10",
    "atr_5",
    "atr_10",
    "gap_return",
    "gap_direction",
    "gap_is_down",
    "volume_ratio_5",
    "volume_ratio_10",
    "volume_zscore_5",
    "sector_momentum_5",
    "sector_momentum_20",
    "sector_momentum_10",
    "sector_momentum_60",
    "relative_return_5d",
    "market_return_5d",
    "market_volatility_ratio",
]


@dataclass
class DownsideModelBundle:
    feature_columns: List[str]
    scaler: Optional[StandardScaler]
    model: Optional[LogisticRegression]
    calibrator: Optional[IsotonicRegression]
    metadata: Dict[str, object]


class DownsideRiskSystemV1:
    """終値ベースのダウンサイド確率・リスク推定システム"""

    def __init__(
        self,
        stock_file: Optional[str] = None,
        down_threshold: float = -0.01,
        down_thresholds: Optional[Iterable[float]] = None,
        horizon_days: int = 1,
        models_dir: Optional[str] = None,
        production_dir: Optional[str] = None,
        hard_negative_source: Optional[str] = None,
    ) -> None:
        self.stock_file = Path(stock_file) if stock_file else None
        if down_thresholds is not None:
            thresholds = [float(th) for th in down_thresholds]
        else:
            thresholds = [float(down_threshold)]
        if not thresholds:
            raise ValueError("down_thresholds must contain at least one value")
        self.down_thresholds = thresholds
        self.down_threshold = self.down_thresholds[0]
        self.horizon_days = max(int(horizon_days), 1)
        self.models_dir = Path(models_dir or "models/downside_risk_v1")
        self.production_dir = Path(production_dir or "production_data")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.production_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.models_dir / "downside_risk_model_v1.joblib"
        self.hard_negative_source = Path(hard_negative_source) if hard_negative_source else None
        self.hard_negative_keys = self._load_hard_negative_keys()
        if self.stock_file is None:
            self.stock_file = self._find_latest_stock_file()
        logger.info("📉 DownsideRiskSystemV1 初期化: stock=%s, horizon=%d", self.stock_file, self.horizon_days)

    # ------------------------------------------------------------------
    # 公開 API
    # ------------------------------------------------------------------
    def run(self, predict_date: Optional[str] = None, retrain: bool = False) -> None:
        """モデル学習＋予測を実行"""
        df = self._load_stock_data()
        features = self._prepare_features(df)

        bundle = None
        if retrain or not self.model_path.exists():
            logger.info("🛠 モデル再学習を実行します (retrain=%s)", retrain)
            bundle = self._train_model(features)
            self._save_model(bundle)
        else:
            bundle = self._load_model()
            if bundle is None:
                logger.warning("既存モデルを読込できなかったため再学習を行います")
                bundle = self._train_model(features)
                self._save_model(bundle)

        predict_day = self._resolve_predict_date(features, predict_date)
        logger.info("🗓️ 予測対象日: %s", predict_day.date())

        pred_rows = features[features['Date'] == predict_day].copy()
        if pred_rows.empty:
            logger.warning("対象日の特徴量が見つからなかったため処理を中断します")
            return

        prob_down = self._predict_probabilities(pred_rows, bundle)
        risk_scores = self._compute_risk_scores(pred_rows)

        predictions = pd.DataFrame(
            {
                'analysis_date': pred_rows['Date'].dt.normalize(),
                'code': pred_rows['Code'].astype(str).str.zfill(4),
                'prob_down': prob_down,
                'risk_score': risk_scores,
                'future_return': pred_rows['future_return'],
            }
        )

        for threshold in self.down_thresholds:
            column_name = self._down_target_column_name(threshold)
            predictions[column_name] = (pred_rows['future_return'] <= threshold).astype(int)

        extra_labels = ['down_target_1pct_2d', 'drawdown_3pct_3d', 'no_rebound_2d']
        for label in extra_labels:
            if label in pred_rows.columns:
                predictions[label] = pred_rows[label].astype(int)
        if 'future_return_2d' in pred_rows.columns:
            predictions['future_return_2d'] = pred_rows['future_return_2d']

        self._persist_outputs(predictions)
        logger.info(
            "✅ Downside/Risk 出力完了: %d銘柄 (avg prob_down=%.3f)",
            len(predictions),
            float(np.nanmean(prob_down)) if len(prob_down) else 0.0,
        )

    # ------------------------------------------------------------------
    # 内部処理
    # ------------------------------------------------------------------
    def _find_latest_stock_file(self) -> Path:
        patterns = [
            "data/processed/nikkei225_complete_*.parquet",
            "data/processed/nikkei225_*stocks_*.parquet",
            "data/processed/nikkei225_*.parquet",
        ]
        latest_path: Optional[Path] = None
        latest_mtime = -1.0
        for pattern in patterns:
            for path_str in Path('.').glob(pattern):
                mtime = path_str.stat().st_mtime
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest_path = path_str
        if latest_path is None:
            raise FileNotFoundError("利用可能な株価データが見つかりませんでした。--stock-file を指定してください")
        return latest_path.resolve()

    def _load_stock_data(self) -> pd.DataFrame:
        if not self.stock_file.exists():
            raise FileNotFoundError(f"株価データが見つかりません: {self.stock_file}")

        suffix = self.stock_file.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(self.stock_file)
        elif suffix in {".csv", ".tsv"}:
            sep = '\t' if suffix == ".tsv" else ','
            df = pd.read_csv(self.stock_file, sep=sep)
        elif suffix in {".pkl", ".pickle"}:
            df = pd.read_pickle(self.stock_file)
        else:
            raise ValueError(f"未対応のファイル形式です: {suffix}")

        if 'Code' not in df.columns or 'Date' not in df.columns or 'Close' not in df.columns:
            raise KeyError("株価データには Code / Date / Close 列が必要です")

        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
        df = df.sort_values(['Code', 'Date']).reset_index(drop=True)
        return df

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        grouped = df.groupby('Code', group_keys=False)

        df['return_1d'] = grouped['Close'].pct_change()
        df['return_3d'] = grouped['Close'].pct_change(periods=3)
        df['return_5d'] = grouped['Close'].pct_change(periods=5)
        df['return_10d'] = grouped['Close'].pct_change(periods=10)
        df['return_20d'] = grouped['Close'].pct_change(periods=20)
        df['return_30d'] = grouped['Close'].pct_change(periods=30)

        market_return_5d = df.groupby('Date')['return_5d'].transform('mean')
        df['market_return_5d'] = market_return_5d
        df['relative_return_5d'] = df['return_5d'] - df['market_return_5d']

        rolling_returns = grouped['return_1d'].rolling(window=5, min_periods=3)
        df['volatility_5'] = rolling_returns.std().reset_index(level=0, drop=True)
        rolling_returns_10 = grouped['return_1d'].rolling(window=10, min_periods=5)
        df['volatility_10'] = rolling_returns_10.std().reset_index(level=0, drop=True)
        rolling_returns_20 = grouped['return_1d'].rolling(window=20, min_periods=10)
        df['volatility_20'] = rolling_returns_20.std().reset_index(level=0, drop=True)

        rolling_max = grouped['Close'].rolling(window=5, min_periods=3).max().reset_index(level=0, drop=True)
        df['drawdown_5'] = df['Close'] / rolling_max - 1.0
        rolling_max_10 = grouped['Close'].rolling(window=10, min_periods=5).max().reset_index(level=0, drop=True)
        df['drawdown_10'] = df['Close'] / rolling_max_10 - 1.0

        prev_close = grouped['Close'].shift(1)
        if {'Open', 'High', 'Low'}.issubset(df.columns):
            true_range = pd.DataFrame(
                {
                    'hl': df['High'] - df['Low'],
                    'hc': (df['High'] - prev_close).abs(),
                    'lc': (df['Low'] - prev_close).abs(),
                }
            ).max(axis=1)
            df['atr_5'] = (
                grouped.apply(lambda g: true_range.loc[g.index].rolling(window=5, min_periods=3).mean())
                .reset_index(level=0, drop=True)
            )
            df['atr_10'] = (
                grouped.apply(lambda g: true_range.loc[g.index].rolling(window=10, min_periods=5).mean())
                .reset_index(level=0, drop=True)
            )
        else:
            df['atr_5'] = pd.NA
            df['atr_10'] = pd.NA

        if 'Open' in df.columns:
            df['gap_return'] = (df['Open'] - prev_close) / prev_close
            gap_values = df['gap_return'].astype(float)
            df['gap_direction'] = np.sign(gap_values)
            df.loc[gap_values.isna(), 'gap_direction'] = np.nan
            df['gap_is_down'] = np.nan
            valid_mask = ~gap_values.isna()
            df.loc[valid_mask, 'gap_is_down'] = (gap_values.loc[valid_mask] < 0).astype(float)
        else:
            df['gap_return'] = pd.NA
            df['gap_direction'] = pd.NA
            df['gap_is_down'] = pd.NA

        if 'Volume' in df.columns:
            rolling_mean = grouped['Volume'].rolling(window=5, min_periods=3).mean().reset_index(level=0, drop=True)
            rolling_std = grouped['Volume'].rolling(window=5, min_periods=3).std().reset_index(level=0, drop=True)
            df['volume_ratio_5'] = df['Volume'] / rolling_mean
            df['volume_ratio_5'] = df['volume_ratio_5'].replace([pd.NA, pd.NaT, float('inf'), float('-inf')], pd.NA)
            df['volume_zscore_5'] = (df['Volume'] - rolling_mean) / rolling_std
            rolling_mean_10 = grouped['Volume'].rolling(window=10, min_periods=5).mean().reset_index(level=0, drop=True)
            df['volume_ratio_10'] = df['Volume'] / rolling_mean_10
            df['volume_ratio_10'] = df['volume_ratio_10'].replace([pd.NA, pd.NaT, float('inf'), float('-inf')], pd.NA)
        else:
            df['volume_ratio_5'] = pd.NA
            df['volume_zscore_5'] = pd.NA
            df['volume_ratio_10'] = pd.NA

        market_return = df.groupby('Date')['return_1d'].transform('mean')
        df['market_return_1d'] = market_return
        df['relative_return_1d'] = df['return_1d'] - df['market_return_1d']
        market_volatility = df.groupby('Date')['volatility_5'].transform('mean')
        df['market_volatility_ratio'] = df['volatility_5'] / market_volatility.replace({0.0: np.nan})

        future_close = grouped['Close'].shift(-self.horizon_days)
        df['future_return'] = future_close / df['Close'] - 1.0
        df['target'] = (df['future_return'] <= self.down_threshold).astype(int)

        # Additional horizons for extended labels
        future_close_2 = grouped['Close'].shift(-2)
        df['future_return_2d'] = future_close_2 / df['Close'] - 1.0

        future_close_3 = grouped['Close'].shift(-3)

        def _forward_max(series: pd.Series, steps: int) -> pd.Series:
            values = series.to_numpy()
            n = len(values)
            result = np.full(n, np.nan)
            for i in range(n):
                start = i + 1
                end = min(n, i + steps + 1)
                if start < n:
                    window = values[start:end]
                    if window.size:
                        result[i] = np.nanmax(window)
            return pd.Series(result, index=series.index)

        future_max_3 = grouped['Close'].transform(lambda s: _forward_max(s, 3))
        future_max_2 = grouped['Close'].transform(lambda s: _forward_max(s, 2))

        df['down_target_1pct_2d'] = (df['future_return_2d'] <= -0.01).astype(int)
        df['drawdown_3pct_3d'] = (
            future_max_3.notna()
            & ((future_max_3 / df['Close']) <= 0.97)
        ).astype(int)
        df['no_rebound_2d'] = (
            future_max_2.notna()
            & ((future_max_2 / df['Close']) <= 1.002)
        ).astype(int)

        if 'Sector' not in df.columns:
            df['Sector'] = 'Unknown'
        sector_returns = df.groupby(['Sector', 'Date'])['return_1d'].mean().rename('sector_return')
        df = df.join(sector_returns, on=['Sector', 'Date'])
        sector_group = df.groupby('Code', group_keys=False)
        df['sector_momentum_5'] = sector_group['sector_return'].rolling(window=5, min_periods=3).mean().reset_index(level=0, drop=True)
        df['sector_momentum_20'] = sector_group['sector_return'].rolling(window=20, min_periods=10).mean().reset_index(level=0, drop=True)
        df['sector_momentum_10'] = sector_group['sector_return'].rolling(window=10, min_periods=5).mean().reset_index(level=0, drop=True)
        df['sector_momentum_60'] = sector_group['sector_return'].rolling(window=60, min_periods=20).mean().reset_index(level=0, drop=True)

        df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].astype(float)
        return df

    def _load_hard_negative_keys(self) -> Optional[set]:
        source_paths: List[Path] = []
        if self.hard_negative_source is not None:
            source_paths.append(self.hard_negative_source)
        default_candidates = Path('production_data/multi_model_candidates.parquet')
        if default_candidates.exists():
            source_paths.append(default_candidates)
        for path in source_paths:
            if path.exists():
                try:
                    df = pd.read_parquet(path)
                except Exception as exc:
                    logger.warning("Hard negative sourceの読み込みに失敗しました: %s", exc)
                    continue
                required_cols = {'analysis_date', 'code', 'prediction_probability', 'future_return'}
                if not required_cols.issubset(df.columns):
                    continue
                df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
                df['code'] = df['code'].astype(str).str.zfill(4)
                mask = (df['prediction_probability'] >= 0.55) & (df['future_return'] <= -0.01)
                if mask.any():
                    keys = set(zip(df.loc[mask, 'analysis_date'], df.loc[mask, 'code']))
                    logger.info("🔍 Hard negative candidates 読込: %d 件 (source=%s)", len(keys), path)
                    return keys
        return None

    def _apply_sampling(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty or 'Date' not in df or 'target' not in df:
            return df
        work_df = df.copy()
        work_df['Date'] = pd.to_datetime(work_df['Date'])
        work_df['Code'] = work_df['Code'].astype(str).str.zfill(4)

        # Temporal SMOTE (up to 2x positives in last 6 months)
        positives = work_df[work_df['target'] == 1]
        if not positives.empty:
            cutoff = positives['Date'].max() - pd.Timedelta(days=180)
            recent_pos = positives[positives['Date'] >= cutoff]
            synthetic_rows: List[pd.Series] = []
            max_new = max(0, len(recent_pos))
            rng = np.random.default_rng(42)
            for _, row in recent_pos.iterrows():
                if len(synthetic_rows) >= max_new:
                    break
                neighbors = recent_pos[(recent_pos['Code'] == row['Code']) & (recent_pos['Date'] != row['Date'])]
                neighbors = neighbors[neighbors['Date'].sub(row['Date']).abs() <= pd.Timedelta(days=3)]
                if neighbors.empty:
                    neighbors = recent_pos[recent_pos['Code'] != row['Code']]
                    neighbors = neighbors[neighbors['Date'].sub(row['Date']).abs() <= pd.Timedelta(days=3)]
                if neighbors.empty:
                    continue
                partner = neighbors.sample(1, random_state=42).iloc[0]
                alpha = float(rng.uniform(0.2, 0.8))
                synthetic = row.copy()
                synthetic[FEATURE_COLUMNS] = alpha * row[FEATURE_COLUMNS].to_numpy() + (1 - alpha) * partner[FEATURE_COLUMNS].to_numpy()
                synthetic['Date'] = row['Date']
                synthetic['Code'] = row['Code']
                synthetic_rows.append(synthetic)
            if synthetic_rows:
                work_df = pd.concat([work_df, pd.DataFrame(synthetic_rows)], ignore_index=True)
                logger.info("🔁 Temporal SMOTE により %d 件の陽性サンプルを生成", len(synthetic_rows))

        # Hard negative mining (reinforce difficult positives)
        if self.hard_negative_keys:
            mask = (
                (work_df['target'] == 1)
                & work_df.apply(lambda r: (r['Date'].normalize(), r['Code']) in self.hard_negative_keys, axis=1)
            )
            hard_samples = work_df[mask]
            if not hard_samples.empty:
                work_df = pd.concat([work_df, hard_samples], ignore_index=True)
                logger.info("🧲 Hard negative サンプルを複製: %d 件", len(hard_samples))

        return work_df

    def _train_model(self, features: pd.DataFrame) -> DownsideModelBundle:
        train_df = features.dropna(subset=FEATURE_COLUMNS + ['target']).copy()
        train_df = self._apply_sampling(train_df)
        if train_df.empty or train_df['target'].nunique() < 2:
            logger.warning("学習に十分なデータが無いためヒューリスティックにフォールバックします")
            return DownsideModelBundle(
                feature_columns=FEATURE_COLUMNS,
                scaler=None,
                model=None,
                calibrator=None,
                metadata={'strategy': 'heuristic'},
            )

        train_core_df, calib_df = self._split_train_calibration(train_df)
        if train_core_df.empty or train_core_df['target'].nunique() < 2:
            logger.warning("キャリブレーション分割後に学習データが不足したため全データで再学習します")
            train_core_df = train_df
            calib_df = train_df.iloc[0:0]

        X_train = train_core_df[FEATURE_COLUMNS].to_numpy()
        y_train = train_core_df['target'].astype(int).to_numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        model = LogisticRegression(max_iter=300, class_weight='balanced', solver='lbfgs')
        model.fit(X_scaled, y_train)

        calibrator: Optional[IsotonicRegression] = None
        calibration_stats: Dict[str, object]

        if not calib_df.empty and calib_df['target'].nunique() > 1:
            X_cal = scaler.transform(calib_df[FEATURE_COLUMNS].to_numpy())
            y_cal = calib_df['target'].astype(int).to_numpy()
            raw_proba = model.predict_proba(X_cal)[:, 1]

            raw_metrics = self._calculate_calibration_stats(y_cal, raw_proba)
            if len(np.unique(raw_proba)) >= 2:
                calibrator = IsotonicRegression(out_of_bounds='clip')
                calibrator.fit(raw_proba, y_cal)
                calibrated_proba = calibrator.transform(raw_proba)
                calibrated_metrics = self._calculate_calibration_stats(y_cal, calibrated_proba)
                calibration_stats = {
                    'holdout_size': float(len(y_cal)),
                    'positive_ratio_holdout': float(y_cal.mean()),
                    'raw': raw_metrics,
                    'calibrated': calibrated_metrics,
                }
                logger.info(
                    "🎯 キャリブレーション: Brier %.4f → %.4f / ECE %.4f → %.4f",
                    raw_metrics['brier_score'],
                    calibrated_metrics['brier_score'],
                    raw_metrics['expected_calibration_error'],
                    calibrated_metrics['expected_calibration_error'],
                )
            else:
                calibration_stats = {
                    'holdout_size': float(len(y_cal)),
                    'positive_ratio_holdout': float(y_cal.mean()),
                    'raw': raw_metrics,
                    'calibrated': None,
                    'note': 'calibration_skipped_constant_predictions',
                }
                logger.warning("キャリブレーション対象の予測値が定数のため補正をスキップしました")
        else:
            calibration_stats = {
                'holdout_size': float(len(calib_df)),
                'positive_ratio_holdout': float(calib_df['target'].mean()) if len(calib_df) else 0.0,
                'raw': None,
                'calibrated': None,
                'note': 'calibration_data_insufficient',
            }
            logger.warning("キャリブレーション用データが不足したため補正は実施されませんでした")

        metadata = {
            'positive_ratio': float(train_df['target'].astype(int).mean()),
            'sample_count': float(len(train_df)),
            'down_threshold': self.down_threshold,
            'down_thresholds': self.down_thresholds,
            'horizon_days': float(self.horizon_days),
            'calibration_stats': calibration_stats,
        }

        bundle = DownsideModelBundle(
            feature_columns=FEATURE_COLUMNS,
            scaler=scaler,
            model=model,
            calibrator=calibrator,
            metadata=metadata,
        )
        self._log_evaluation_metrics(features, bundle)
        return bundle

    def _split_train_calibration(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        ordered = df.sort_values('Date').copy()
        unique_dates = ordered['Date'].dropna().unique()

        if len(unique_dates) <= 1:
            return ordered, ordered.iloc[0:0]

        cal_days = max(5, int(len(unique_dates) * 0.2))
        cal_days = min(cal_days, len(unique_dates) - 1)
        cal_dates = set(unique_dates[-cal_days:])

        calib_df = ordered[ordered['Date'].isin(cal_dates)].copy()
        train_df = ordered[~ordered['Date'].isin(cal_dates)].copy()

        if train_df.empty:
            train_df = ordered.iloc[:-1].copy()
            calib_df = ordered.iloc[-1:].copy()

        return train_df, calib_df

    def _calculate_calibration_stats(
        self,
        y_true: np.ndarray,
        proba: np.ndarray,
        n_bins: int = 10,
    ) -> Dict[str, float]:
        y_true_arr = np.asarray(y_true, dtype=float)
        proba_arr = np.asarray(proba, dtype=float)

        brier = brier_score_loss(y_true_arr, proba_arr)

        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        ece = 0.0
        mce = 0.0

        for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
            if np.isclose(upper, 1.0):
                mask = (proba_arr >= lower) & (proba_arr <= upper)
            else:
                mask = (proba_arr >= lower) & (proba_arr < upper)

            if not np.any(mask):
                continue

            prop = float(mask.mean())
            acc = float(y_true_arr[mask].mean())
            conf = float(proba_arr[mask].mean())
            diff = abs(conf - acc)
            ece += diff * prop
            mce = max(mce, diff)

        return {
            'brier_score': float(brier),
            'expected_calibration_error': float(ece),
            'maximum_calibration_error': float(mce),
        }

    @staticmethod
    def _precision_top_fraction(y_true: np.ndarray, proba: np.ndarray, fraction: float = 0.2) -> float:
        if len(proba) == 0:
            return float('nan')
        k = max(1, int(len(proba) * fraction))
        if k <= 0:
            return float('nan')
        idx = np.argsort(proba)[-k:]
        return float(np.mean(y_true[idx]))

    def _perform_time_series_cv(self, features: pd.DataFrame) -> Optional[Dict[str, float]]:
        df = features.dropna(subset=FEATURE_COLUMNS + ['target']).copy()
        if df.empty:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        cutoff = df['Date'].max() - pd.Timedelta(days=120)
        df = df[df['Date'] >= cutoff]
        if df['Date'].nunique() < 30 or df['target'].nunique() < 2:
            return None

        X = df[FEATURE_COLUMNS].to_numpy()
        y = df['target'].astype(int).to_numpy()
        tscv = TimeSeriesSplit(n_splits=5)
        stats: List[Dict[str, float]] = []

        for train_idx, test_idx in tscv.split(X):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue
            scaler = StandardScaler().fit(X[train_idx])
            model = LogisticRegression(max_iter=300, class_weight='balanced', solver='lbfgs')
            model.fit(scaler.transform(X[train_idx]), y[train_idx])
            proba = model.predict_proba(scaler.transform(X[test_idx]))[:, 1]
            fold_stats = self._calculate_calibration_stats(y[test_idx], proba)
            fold_stats['precision_top20'] = self._precision_top_fraction(y[test_idx], proba, fraction=0.2)
            stats.append(fold_stats)

        if not stats:
            return None
        aggregate = {
            key: float(np.nanmean([fold[key] for fold in stats]))
            for key in stats[0]
        }
        aggregate['fold_count'] = float(len(stats))
        return aggregate

    def _evaluate_bundle(self, features: pd.DataFrame, bundle: DownsideModelBundle, *, window_days: int) -> Optional[Dict[str, float]]:
        df = features.dropna(subset=FEATURE_COLUMNS + ['target']).copy()
        if df.empty:
            return None
        df['Date'] = pd.to_datetime(df['Date'])
        cutoff = df['Date'].max() - pd.Timedelta(days=window_days)
        df = df[df['Date'] >= cutoff]
        if df.empty:
            return None
        proba = self._predict_probabilities(df, bundle)
        y_true = df['target'].astype(int).to_numpy()
        stats = self._calculate_calibration_stats(y_true, proba)
        stats['precision_top20'] = self._precision_top_fraction(y_true, proba, fraction=0.2)
        stats['sample_size'] = float(len(df))
        stats['positive_ratio'] = float(y_true.mean())
        return stats

    def _log_evaluation_metrics(self, features: pd.DataFrame, bundle: DownsideModelBundle) -> None:
        overall = self._evaluate_bundle(features, bundle, window_days=365)
        recent = self._evaluate_bundle(features, bundle, window_days=90)
        cv_stats = self._perform_time_series_cv(features)

        def _fmt(stats: Optional[Dict[str, float]]) -> str:
            if not stats:
                return "N/A"
            return (
                f"Brier={stats['brier_score']:.4f}, "
                f"ECE={stats['expected_calibration_error']:.4f}, "
                f"Prec@20={stats['precision_top20']:.2%}, "
                f"n={int(stats.get('sample_size', 0))}"
            )

        logger.info("📊 Downside評価: overall=%s | recent90=%s", _fmt(overall), _fmt(recent))
        if overall and overall['brier_score'] > 0.20:
            logger.warning("Brierが目標を上回っています (%.4f > 0.20)", overall['brier_score'])
        if overall and overall['expected_calibration_error'] > 0.10:
            logger.warning("ECEが目標を上回っています (%.4f > 0.10)", overall['expected_calibration_error'])
        if cv_stats:
            logger.info(
                "🧪 CV評価: folds=%d Brier=%.4f ECE=%.4f Prec@20=%.2f%%",
                int(cv_stats.get('fold_count', 0)),
                cv_stats['brier_score'],
                cv_stats['expected_calibration_error'],
                cv_stats['precision_top20'] * 100.0,
            )

    def _predict_probabilities(self, df: pd.DataFrame, bundle: DownsideModelBundle) -> np.ndarray:
        X = df[FEATURE_COLUMNS].fillna(0.0).to_numpy()
        if bundle.model is None or bundle.scaler is None:
            risk = self._compute_risk_scores(df)
            return np.clip(0.4 + 0.6 * risk, 0.0, 1.0)
        X_scaled = bundle.scaler.transform(X)
        proba = bundle.model.predict_proba(X_scaled)[:, 1]
        if bundle.calibrator is not None:
            proba = bundle.calibrator.transform(proba)
        return np.clip(proba, 0.0, 1.0)

    def _compute_risk_scores(self, df: pd.DataFrame) -> np.ndarray:
        vol_component = df['volatility_5'].abs().fillna(0.0)
        draw_component = df['drawdown_5'].abs().fillna(0.0)
        atr_component = df.get('atr_5', pd.Series(0.0, index=df.index)).fillna(0.0)
        gap_component = df.get('gap_return', pd.Series(0.0, index=df.index)).abs().fillna(0.0)
        combined = (
            0.5 * vol_component
            + 0.2 * draw_component
            + 0.2 * atr_component
            + 0.1 * gap_component
        )
        base = combined.fillna(0.0)
        if base.empty:
            return np.zeros(len(df))
        scale = base.quantile(0.9)
        scale = float(scale) if scale and np.isfinite(scale) else 1e-6
        risk = np.clip(base / (scale + 1e-6), 0.0, 1.0)
        return risk.to_numpy()

    def _resolve_predict_date(self, features: pd.DataFrame, predict_date: Optional[str]) -> pd.Timestamp:
        unique_dates = sorted(features['Date'].dropna().unique())
        if not unique_dates:
            raise ValueError("特徴量に有効な日付が存在しません")
        if predict_date is None:
            return pd.to_datetime(unique_dates[-1])
        predict_ts = pd.to_datetime(predict_date)
        if predict_ts not in unique_dates:
            logger.warning("指定日のデータが無いため最も近い過去日に置き換えます")
            prev_dates = [d for d in unique_dates if d <= predict_ts]
            if prev_dates:
                predict_ts = prev_dates[-1]
            else:
                predict_ts = unique_dates[-1]
        return pd.to_datetime(predict_ts)

    def _save_model(self, bundle: DownsideModelBundle) -> None:
        payload = {
            'feature_columns': bundle.feature_columns,
            'scaler': bundle.scaler,
            'model': bundle.model,
            'calibrator': bundle.calibrator,
            'metadata': bundle.metadata,
        }
        joblib.dump(payload, self.model_path)
        meta_path = self.model_path.with_suffix('.json')
        meta_path.write_text(json.dumps(bundle.metadata, indent=2, ensure_ascii=False))
        logger.info("💾 モデル保存: %s", self.model_path)

    def _load_model(self) -> Optional[DownsideModelBundle]:
        if not self.model_path.exists():
            return None
        payload = joblib.load(self.model_path)
        feature_columns = payload.get('feature_columns', FEATURE_COLUMNS)
        return DownsideModelBundle(
            feature_columns=feature_columns,
            scaler=payload.get('scaler'),
            model=payload.get('model'),
            calibrator=payload.get('calibrator'),
            metadata=payload.get('metadata', {}),
        )

    def _down_target_column_name(self, threshold: float) -> str:
        abs_pct = abs(threshold) * 100
        if abs_pct.is_integer():
            suffix = f"{int(abs_pct)}pct"
        else:
            suffix = f"{abs_pct:.1f}pct".replace('.', '_')
        return f'down_target_{suffix}'

    def _persist_outputs(self, predictions: pd.DataFrame) -> None:
        downside_path = self.production_dir / 'downside_predictions.parquet'
        risk_path = self.production_dir / 'risk_predictions.parquet'

        target_cols = [col for col in predictions.columns if col.startswith('down_target_')]
        downside_cols = ['analysis_date', 'code', 'prob_down', 'future_return'] + target_cols
        for extra in ['drawdown_3pct_3d', 'no_rebound_2d', 'future_return_2d']:
            if extra in predictions.columns and extra not in downside_cols:
                downside_cols.append(extra)
        risk_cols = ['analysis_date', 'code', 'risk_score']

        predictions[downside_cols].to_parquet(downside_path, index=False)
        predictions[risk_cols].to_parquet(risk_path, index=False)

        logger.info("📄 出力: %s / %s", downside_path.name, risk_path.name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downside & Risk 推定")
    parser.add_argument('--stock-file', type=str, help='終値データファイル (parquet/csv/pkl)。未指定時は最新ファイルを探索')
    parser.add_argument('--predict-date', type=str, help='予測対象日 (YYYY-MM-DD)')
    parser.add_argument('--down-threshold', type=float, default=-0.01, help='下落ラベル判定の閾値')
    parser.add_argument('--down-thresholds', type=str, help='カンマ区切りで複数の下落閾値を指定 (-0.01,-0.015 など)')
    parser.add_argument('--horizon-days', type=int, default=1, help='未来リターンの判定期間（日数）')
    parser.add_argument('--models-dir', type=str, help='モデル保存ディレクトリ')
    parser.add_argument('--production-dir', type=str, help='推論結果保存ディレクトリ')
    parser.add_argument('--retrain', action='store_true', help='既存モデルを無視して再学習')
    parser.add_argument('--hard-negative-source', type=str, help='ハードネガティブ候補を取得する候補データ（parquet）')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    threshold_list = None
    if args.down_thresholds:
        parsed_thresholds = [
            float(chunk.strip())
            for chunk in args.down_thresholds.split(',')
            if chunk.strip()
        ]
        base_threshold = float(args.down_threshold)
        ordered: List[float] = [base_threshold]
        for value in parsed_thresholds:
            if value not in ordered:
                ordered.append(value)
        threshold_list = ordered
    system = DownsideRiskSystemV1(
        stock_file=args.stock_file,
        down_threshold=args.down_threshold,
        down_thresholds=threshold_list,
        horizon_days=args.horizon_days,
        models_dir=args.models_dir,
        production_dir=args.production_dir,
        hard_negative_source=args.hard_negative_source,
    )
    system.run(predict_date=args.predict_date, retrain=args.retrain)


if __name__ == '__main__':
    main()
