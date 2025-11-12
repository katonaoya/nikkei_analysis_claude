#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close price ensemble system for priority-2 modeling refresh."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb

try:  # CatBoost is optional during development / CI
    from catboost import CatBoostClassifier

    CATBOOST_AVAILABLE = True
except Exception:  # pragma: no cover - handled at runtime
    CATBOOST_AVAILABLE = False

sys.path.append(str(Path(__file__).resolve().parent.parent))

from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1

logger = logging.getLogger(__name__)


@dataclass
class BaseModelSpec:
    name: str
    builder: Callable[[], object]
    needs_scale_pos_weight: bool = False


class ClosePrecisionEnsembleTrainer:
    """Stacked ensemble trainer that reuses the Phase-1 feature pipeline."""

    def __init__(
        self,
        target_return: float = 0.01,
        imbalance_boost: float = 1.0,
        imbalance_strategy: str = "scale_pos",
        n_splits: int = 4,
        test_size: int = 42,
        evaluation_window: int = 60,
        base_model_names: Optional[Iterable[str]] = None,
        max_training_rows: Optional[int] = None,
        output_dir: Optional[Union[str, Path]] = None,
        analysis_dir: Optional[Union[str, Path]] = None,
        top_n: int = 3,
    ) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )

        self.core_system = CloseReturnPrecisionSystemV1(
            target_return=target_return,
            imbalance_boost=imbalance_boost,
            imbalance_strategy=imbalance_strategy,
            calibration_method="none",
        )
        self.n_splits = n_splits
        self.test_size = test_size
        self.evaluation_window = evaluation_window
        self.max_training_rows = max_training_rows
        self.random_state = 42
        self.top_n = max(int(top_n), 1)

        requested_models = tuple(base_model_names) if base_model_names else ("lightgbm", "catboost", "logreg")
        self.base_model_specs = self._build_base_model_specs(requested_models)
        if not self.base_model_specs:
            raise ValueError("少なくとも1つのベースモデルが必要です。")

        self.meta_model: Optional[LogisticRegression] = None
        self.meta_config: Dict[str, Union[str, float, Dict[int, float]]] = {}
        self.calibrator = None

        self.output_dir = Path(output_dir) if output_dir else Path("models/ensemble_close_v2")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir = Path(analysis_dir) if analysis_dir else Path("analysis")
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "🧭 Ensemble trainer initialised with %d base models: %s",
            len(self.base_model_specs),
            ", ".join(self.base_model_specs.keys()),
        )

    # ------------------------------------------------------------------
    # Model builders
    # ------------------------------------------------------------------
    def _build_base_model_specs(self, names: Sequence[str]) -> Dict[str, BaseModelSpec]:
        specs: Dict[str, BaseModelSpec] = {}

        if "lightgbm" in names:
            params = self.core_system.model_params.copy()
            params.update(
                {
                    "objective": "binary",
                    "random_state": self.random_state,
                    "n_jobs": -1,
                    "verbose": -1,
                }
            )

            def build_lgbm() -> lgb.LGBMClassifier:
                return lgb.LGBMClassifier(**params)

            specs["lightgbm"] = BaseModelSpec(
                name="lightgbm",
                builder=build_lgbm,
                needs_scale_pos_weight=True,
            )

        if "catboost" in names and CATBOOST_AVAILABLE:
            def build_catboost() -> CatBoostClassifier:
                return CatBoostClassifier(
                    iterations=300,
                    depth=6,
                    learning_rate=0.03,
                    loss_function="Logloss",
                    eval_metric="Logloss",
                    random_seed=self.random_state,
                    auto_class_weights="Balanced",
                    verbose=False,
                )

            specs["catboost"] = BaseModelSpec(name="catboost", builder=build_catboost)

        if "logreg" in names:
            def build_logreg() -> Pipeline:
                return Pipeline(
                    steps=[
                        ("scaler", StandardScaler(with_mean=False)),
                        (
                            "clf",
                            LogisticRegression(
                                max_iter=400,
                                class_weight="balanced",
                                solver="saga",
                                penalty="l2",
                                random_state=self.random_state,
                                n_jobs=-1,
                            ),
                        ),
                    ]
                )

            specs["logreg"] = BaseModelSpec(
                name="logreg",
                builder=build_logreg,
            )

        missing = set(names) - set(specs.keys())
        if missing:
            logger.warning("⚠️  利用不可のベースモデルをスキップ: %s", ", ".join(sorted(missing)))

        return specs

    # ------------------------------------------------------------------
    # Data preparation helpers
    # ------------------------------------------------------------------
    def _load_feature_dataset(self) -> Tuple[pd.DataFrame, List[str]]:
        logger.info("📥 Loading base dataset + engineered features...")
        base_df = self.core_system.load_and_integrate_data()
        feature_df = self.core_system.create_enhanced_features(base_df)

        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna(subset=["Target", "Date", "Code"])
        feature_df = feature_df.sort_values(["Date", "Code"]).reset_index(drop=True)

        numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
        numeric_cols = [col for col in numeric_cols if col not in {"Target"}]

        if not numeric_cols:
            raise ValueError("学習用の数値特徴量が見つかりません。")

        feature_df = self._forward_fill_by_code(feature_df, numeric_cols)

        if self.max_training_rows:
            feature_df = feature_df.tail(self.max_training_rows).reset_index(drop=True)
            logger.info(
                "📏 Downsampled dataset to the most recent %d rows for experimentation.",
                self.max_training_rows,
            )

        feature_df[numeric_cols] = feature_df[numeric_cols].fillna(0.0).astype(np.float32)
        feature_df["Target"] = feature_df["Target"].astype(int)

        logger.info(
            "📊 Feature matrix ready: %d rows, %d numeric features, positive ratio %.3f",
            len(feature_df),
            len(numeric_cols),
            feature_df["Target"].mean(),
        )

        return feature_df, numeric_cols

    @staticmethod
    def _forward_fill_by_code(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        filled = df.copy()
        filled[numeric_cols] = (
            filled.groupby("Code", observed=True)[numeric_cols]
            .apply(lambda group: group.fillna(method="ffill"))
            .reset_index(level=0, drop=True)
        )
        return filled

    # ------------------------------------------------------------------
    # Training + evaluation core
    # ------------------------------------------------------------------
    def _build_meta_model(self, C: float, class_weight: Optional[Union[str, Dict[int, float]]]) -> LogisticRegression:
        return LogisticRegression(
            max_iter=1000,
            class_weight=class_weight,
            solver="lbfgs",
            random_state=self.random_state,
            C=C,
        )

    def _select_meta_model(
        self,
        stacked_features: np.ndarray,
        y_train: np.ndarray,
        train_valid_df: pd.DataFrame,
    ) -> Tuple[LogisticRegression, Dict[str, Union[str, float, Dict[int, float]]]]:
        candidate_class_weights: List[Union[str, Dict[int, float]]] = [
            "balanced",
            {0: 1.0, 1: 0.75},
            {0: 1.0, 1: 0.55},
            {0: 1.0, 1: 0.45},
            {0: 1.0, 1: 0.35},
            {0: 1.0, 1: 0.25},
        ]
        candidate_C = [0.2, 0.4, 0.7, 1.0]

        best_model: Optional[LogisticRegression] = None
        best_config: Dict[str, Union[str, float, Dict[int, float]]] = {}
        best_precision = -np.inf

        for class_weight in candidate_class_weights:
            for C in candidate_C:
                model = self._build_meta_model(C=C, class_weight=class_weight)
                model.fit(stacked_features, y_train)
                prob = model.predict_proba(stacked_features)[:, 1]
                topn_metrics = self._topn_metrics(
                    train_valid_df,
                    prob,
                    prefix=f"candidate_top{self.top_n}",
                )
                precision = topn_metrics[f"candidate_top{self.top_n}_precision"]
                if np.isnan(precision):
                    continue
                if precision > best_precision:
                    best_precision = precision
                    best_model = model
                    best_config = {
                        "class_weight": class_weight,
                        "C": C,
                        "topn_precision": precision,
                    }

        if best_model is None:
            logger.warning("メタモデル候補から最適解を見つけられなかったため、デフォルト設定を使用します")
            best_model = self._build_meta_model(C=1.0, class_weight="balanced")
            best_model.fit(stacked_features, y_train)
            best_config = {"class_weight": "balanced", "C": 1.0, "topn_precision": float("nan")}

        logger.info(
            "🥇 Meta model選定: class_weight=%s, C=%.2f, Top-%d precision=%.3f",
            best_config.get("class_weight"),
            best_config.get("C"),
            self.top_n,
            best_config.get("topn_precision"),
        )

        return best_model, best_config

    def train_and_evaluate(self) -> Dict[str, Dict[str, float]]:
        dataset, feature_cols = self._load_feature_dataset()
        unique_dates = np.array(sorted(dataset["Date"].unique()))

        if len(unique_dates) < (self.evaluation_window + self.test_size + self.n_splits):
            raise ValueError("データ期間が不足しているため、時系列検証を構成できません。")

        eval_start = unique_dates[-self.evaluation_window]
        train_df = dataset[dataset["Date"] < eval_start].reset_index(drop=True)
        holdout_df = dataset[dataset["Date"] >= eval_start].reset_index(drop=True)

        if train_df.empty or holdout_df.empty:
            raise ValueError("訓練またはホールドアウトのデータが不足しています。")

        logger.info(
            "🧪 Training span: %s → %s (%d rows)",
            train_df["Date"].min().date(),
            train_df["Date"].max().date(),
            len(train_df),
        )
        logger.info(
            "🧪 Holdout span: %s → %s (%d rows)",
            holdout_df["Date"].min().date(),
            holdout_df["Date"].max().date(),
            len(holdout_df),
        )

        oof_predictions, fold_metrics = self._run_time_series_oof(train_df, feature_cols)

        valid_mask = np.ones(len(train_df), dtype=bool)
        for preds in oof_predictions.values():
            valid_mask &= ~np.isnan(preds)

        if valid_mask.sum() < len(train_df):
            dropped = len(train_df) - int(valid_mask.sum())
            logger.info("🧹 Dropped %d early samples without validation coverage", dropped)

        stacked_features = np.column_stack(
            [oof_predictions[name][valid_mask] for name in self.base_model_specs]
        )
        y_train = train_df.loc[valid_mask, "Target"].values
        train_valid_df = train_df.loc[valid_mask].copy().reset_index(drop=True)

        self.meta_model, self.meta_config = self._select_meta_model(
            stacked_features,
            y_train,
            train_valid_df,
        )

        oof_meta_prob = self.meta_model.predict_proba(stacked_features)[:, 1]
        self.calibrator = self._fit_probability_calibrator(oof_meta_prob, train_valid_df)
        if self.calibrator is not None:
            oof_meta_prob = self.calibrator.predict(oof_meta_prob)

        cv_metrics = self._summarise_metrics(
            y_true=y_train,
            probas=oof_meta_prob,
            label_prefix="ensemble_oof",
        )

        base_cv_metrics = {
            f"{name}_oof": self._summarise_metrics(
                y_train,
                oof_predictions[name][valid_mask],
                label_prefix=f"{name}_oof",
            )
            for name in self.base_model_specs
        }

        base_topn_metrics = {
            f"{name}_top{self.top_n}": self._topn_metrics(
                train_valid_df,
                oof_predictions[name][valid_mask],
                prefix=name,
            )
            for name in self.base_model_specs
        }
        ensemble_topn_metrics = self._topn_metrics(
            train_valid_df,
            oof_meta_prob,
            prefix="ensemble_oof",
        )

        for name in self.base_model_specs:
            base_cv_metrics[f"{name}_oof"].update(base_topn_metrics[f"{name}_top{self.top_n}"])
        cv_metrics.update(ensemble_topn_metrics)

        fitted_base_models = self._fit_base_models(train_df, feature_cols)

        holdout_metrics, holdout_predictions = self._evaluate_holdout(
            holdout_df,
            feature_cols,
            fitted_base_models,
        )

        final_base_models = self._fit_base_models(dataset, feature_cols)

        dataset_X = dataset[feature_cols].values.astype(np.float32)
        stacked_full = np.column_stack([
            self._predict_proba(final_base_models[name], dataset_X)
            for name in self.base_model_specs
        ])
        y_full = dataset["Target"].values

        meta_class_weight = self.meta_config.get("class_weight", "balanced")
        self.meta_model = self._build_meta_model(
            C=float(self.meta_config.get("C", 1.0)),
            class_weight=meta_class_weight,
        )
        self.meta_model.fit(stacked_full, y_full)

        train_mask_full = dataset["Date"] < eval_start
        if train_mask_full.any():
            calibration_probs = self.meta_model.predict_proba(stacked_full[train_mask_full.values])[:, 1]
            calibration_df = dataset.loc[train_mask_full, ["Date", "Target"]].copy().reset_index(drop=True)
            self.calibrator = self._fit_probability_calibrator(calibration_probs, calibration_df)

        artifact_path = self._persist_artifacts(
            feature_cols=feature_cols,
            base_models=final_base_models,
            cv_metrics=cv_metrics,
            base_cv_metrics=base_cv_metrics,
            holdout_metrics=holdout_metrics,
        )

        self._persist_metrics_report(
            cv_metrics=cv_metrics,
            base_cv_metrics=base_cv_metrics,
            holdout_metrics=holdout_metrics,
            fold_metrics=fold_metrics,
            holdout_predictions=holdout_predictions,
            eval_start=eval_start,
        )

        logger.info("💾 Ensemble artifact saved to %s", artifact_path)
        return {
            "ensemble_cv": cv_metrics,
            "ensemble_holdout": holdout_metrics["ensemble"],
        }

    # ------------------------------------------------------------------
    # Cross-validation helpers
    # ------------------------------------------------------------------
    def _run_time_series_oof(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Tuple[Dict[str, np.ndarray], List[Dict[str, float]]]:
        unique_train_dates = np.array(sorted(train_df["Date"].unique()))
        splitter = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)

        oof_predictions = {
            name: np.full(len(train_df), np.nan, dtype=np.float32)
            for name in self.base_model_specs
        }
        fold_metrics: List[Dict[str, float]] = []

        for fold_id, (date_idx_train, date_idx_val) in enumerate(splitter.split(unique_train_dates), start=1):
            train_dates = unique_train_dates[date_idx_train]
            val_dates = unique_train_dates[date_idx_val]
            train_mask = train_df["Date"].isin(train_dates)
            val_mask = train_df["Date"].isin(val_dates)

            X_train = train_df.loc[train_mask, feature_cols].values
            y_train = train_df.loc[train_mask, "Target"].values
            X_val = train_df.loc[val_mask, feature_cols].values
            y_val = train_df.loc[val_mask, "Target"].values

            X_train = X_train.astype(np.float32)
            X_val = X_val.astype(np.float32)

            fold_info = {
                "fold": fold_id,
                "train_rows": int(train_mask.sum()),
                "val_rows": int(val_mask.sum()),
                "val_start": train_df.loc[val_mask, "Date"].min().strftime("%Y-%m-%d"),
                "val_end": train_df.loc[val_mask, "Date"].max().strftime("%Y-%m-%d"),
            }

            for name, spec in self.base_model_specs.items():
                model = spec.builder()
                if spec.needs_scale_pos_weight:
                    spw = self.core_system._compute_scale_pos_weight(pd.Series(y_train))
                    model.set_params(scale_pos_weight=spw)

                model.fit(X_train, y_train)
                prob = self._predict_proba(model, X_val)
                val_indices = np.where(val_mask.values)[0]
                oof_predictions[name][val_indices] = prob

                metrics = self._summarise_metrics(y_val, prob, threshold=0.5)
                fold_info[f"{name}_precision"] = metrics["precision"]
                fold_info[f"{name}_recall"] = metrics["recall"]

            fold_metrics.append(fold_info)
            logger.info(
                "🌀 Fold %d finished: train %d rows, val %d rows (%s → %s)",
                fold_id,
                fold_info["train_rows"],
                fold_info["val_rows"],
                fold_info["val_start"],
                fold_info["val_end"],
            )

        return oof_predictions, fold_metrics

    def _fit_base_models(
        self,
        df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, object]:
        X = df[feature_cols].values.astype(np.float32)
        y = df["Target"].values

        models: Dict[str, object] = {}
        for name, spec in self.base_model_specs.items():
            model = spec.builder()
            if spec.needs_scale_pos_weight:
                spw = self.core_system._compute_scale_pos_weight(pd.Series(y))
                model.set_params(scale_pos_weight=spw)

            model.fit(X, y)
            models[name] = model
            logger.info("✅ Base model '%s' fitted on %d rows", name, len(df))
        return models

    def _evaluate_holdout(
        self,
        holdout_df: pd.DataFrame,
        feature_cols: List[str],
        base_models: Dict[str, object],
    ) -> Tuple[Dict[str, Dict[str, float]], pd.DataFrame]:
        X_holdout = holdout_df[feature_cols].values.astype(np.float32)
        y_holdout = holdout_df["Target"].values

        base_predictions: Dict[str, np.ndarray] = {}
        metrics: Dict[str, Dict[str, float]] = {}

        for name, model in base_models.items():
            prob = self._predict_proba(model, X_holdout)
            base_predictions[name] = prob
            metrics[name] = self._summarise_metrics(y_holdout, prob, label_prefix=name)

        stacked_holdout = np.column_stack([base_predictions[name] for name in self.base_model_specs])
        ensemble_prob = self.meta_model.predict_proba(stacked_holdout)[:, 1]
        if self.calibrator is not None:
            ensemble_prob = self.calibrator.predict(ensemble_prob)
        metrics["ensemble"] = self._summarise_metrics(y_holdout, ensemble_prob, label_prefix="ensemble")

        holdout_pred_df = holdout_df[["Date", "Code", "Target"]].copy()
        for name, prob in base_predictions.items():
            holdout_pred_df[f"prob_{name}"] = prob
        holdout_pred_df["prob_ensemble"] = ensemble_prob

        for name, prob in base_predictions.items():
            topn_stats = self._topn_metrics(
                holdout_pred_df.assign(current_prob=prob),
                prob,
                prefix=f"{name}_top{self.top_n}",
            )
            metrics[name].update(topn_stats)

        ensemble_topn = self._topn_metrics(
            holdout_pred_df,
            ensemble_prob,
            prefix=f"ensemble_top{self.top_n}",
        )
        metrics["ensemble"].update(ensemble_topn)

        ensemble_metrics = metrics["ensemble"]
        logger.info(
            "🧮 Holdout precision %.3f / recall %.3f / accuracy %.3f",
            ensemble_metrics.get("ensemble_precision", float("nan")),
            ensemble_metrics.get("ensemble_recall", float("nan")),
            ensemble_metrics.get("ensemble_accuracy", float("nan")),
        )
        logger.info(
            "🎯 Holdout Top-%d precision %.3f (平均選定銘柄 %.2f件/日)",
            self.top_n,
            ensemble_metrics.get(f"ensemble_top{self.top_n}_precision", float("nan")),
            ensemble_metrics.get(f"ensemble_top{self.top_n}_avg_selected", float("nan")),
        )

        return metrics, holdout_pred_df

    # ------------------------------------------------------------------
    # Metrics + persistence helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _predict_proba(model: object, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            if isinstance(prob, tuple):  # CatBoost returns (proba, indexes)
                prob = prob[0]
            return prob[:, 1]
        if hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            return 1 / (1 + np.exp(-decision))
        preds = model.predict(X)
        return preds.astype(float)

    @staticmethod
    def _summarise_metrics(
        y_true: np.ndarray,
        probas: np.ndarray,
        threshold: float = 0.5,
        label_prefix: str = "",
    ) -> Dict[str, float]:
        preds = (probas >= threshold).astype(int)
        metrics = {
            "precision": precision_score(y_true, preds, zero_division=0),
            "recall": recall_score(y_true, preds, zero_division=0),
            "accuracy": accuracy_score(y_true, preds),
            "f1": f1_score(y_true, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_true, probas) if len(np.unique(y_true)) > 1 else float("nan"),
        }
        if label_prefix:
            metrics = {f"{label_prefix}_{k}": v for k, v in metrics.items()}
        return metrics

    def _topn_metrics(
        self,
        df: pd.DataFrame,
        probas: np.ndarray,
        prefix: str,
    ) -> Dict[str, float]:
        working = df.copy().reset_index(drop=True)
        working["__score__"] = probas

        hits = 0
        selected = 0
        days = 0

        for _, group in working.groupby("Date"):
            top_group = group.nlargest(self.top_n, "__score__")
            hits += float(top_group["Target"].sum())
            selected += len(top_group)
            days += 1

        precision = hits / selected if selected else float("nan")
        avg_selected = selected / days if days else 0.0

        return {
            f"{prefix}_precision": precision,
            f"{prefix}_hits": hits,
            f"{prefix}_selected": selected,
            f"{prefix}_days": days,
            f"{prefix}_avg_selected": avg_selected,
        }

    def _fit_probability_calibrator(
        self,
        probabilities: np.ndarray,
        df: pd.DataFrame,
        min_days: int = 63,
    ) -> Optional[IsotonicRegression]:
        if len(df) != len(probabilities):
            raise ValueError("確率とデータ行数が一致しません")

        calib_df = df.copy().reset_index(drop=True)
        calib_df["__prob__"] = probabilities

        last_date = calib_df["Date"].max()
        if pd.isna(last_date):
            return None

        cutoff = last_date - pd.Timedelta(days=min_days)
        window_df = calib_df[calib_df["Date"] >= cutoff]
        if len(window_df) < 500:
            window_df = calib_df

        prob = window_df["__prob__"].values
        target = window_df["Target"].values

        if len(np.unique(target)) < 2:
            logger.warning("キャリブレーションに必要な正例/負例が不足しているためスキップします")
            return None

        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(prob, target)
        logger.info("🧪 確率キャリブレーションを実施: 使用データ %d件", len(window_df))
        return calibrator

    def _persist_artifacts(
        self,
        feature_cols: List[str],
        base_models: Dict[str, object],
        cv_metrics: Dict[str, float],
        base_cv_metrics: Dict[str, Dict[str, float]],
        holdout_metrics: Dict[str, Dict[str, float]],
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        artifact = {
            "feature_cols": feature_cols,
            "base_models": base_models,
            "meta_model": self.meta_model,
            "meta_config": self.meta_config,
            "calibrator": self.calibrator,
            "cv_metrics": cv_metrics,
            "base_cv_metrics": base_cv_metrics,
            "holdout_metrics": holdout_metrics,
            "target_return": self.core_system.target_return,
            "n_splits": self.n_splits,
            "test_size": self.test_size,
            "evaluation_window": self.evaluation_window,
            "top_n": self.top_n,
            "timestamp": timestamp,
        }

        artifact_path = self.output_dir / f"close_precision_ensemble_{timestamp}.joblib"
        joblib.dump(artifact, artifact_path)

        latest_path = self.output_dir / "latest_ensemble_model.joblib"
        joblib.dump(artifact, latest_path)
        return artifact_path

    def _persist_metrics_report(
        self,
        cv_metrics: Dict[str, float],
        base_cv_metrics: Dict[str, Dict[str, float]],
        holdout_metrics: Dict[str, Dict[str, float]],
        fold_metrics: List[Dict[str, float]],
        holdout_predictions: pd.DataFrame,
        eval_start: np.datetime64,
    ) -> None:
        summary = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_start_date": str(eval_start)[:10],
            "top_n": self.top_n,
            "meta_config": self.meta_config,
            "ensemble_cv_metrics": cv_metrics,
            "base_cv_metrics": base_cv_metrics,
            "holdout_metrics": holdout_metrics,
            "fold_details": fold_metrics,
        }

        metrics_path = self.analysis_dir / "ensemble_precision_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as fp:
            json.dump(summary, fp, ensure_ascii=False, indent=2)

        holdout_path = self.analysis_dir / "ensemble_holdout_predictions.csv"
        holdout_predictions.to_csv(holdout_path, index=False)

    # ------------------------------------------------------------------
    # CLI entrypoint
    # ------------------------------------------------------------------
    @classmethod
    def from_args(cls) -> "ClosePrecisionEnsembleTrainer":
        parser = argparse.ArgumentParser(description="Priority-2 ensemble trainer")
        parser.add_argument("--target-return", type=float, default=0.01)
        parser.add_argument("--imbalance-boost", type=float, default=1.0)
        parser.add_argument("--imbalance-strategy", type=str, default="scale_pos")
        parser.add_argument("--n-splits", type=int, default=4)
        parser.add_argument("--test-size", type=int, default=42)
        parser.add_argument("--evaluation-window", type=int, default=60)
        parser.add_argument(
            "--base-models",
            type=str,
            default="lightgbm,catboost,logreg",
            help="Comma separated list of base models (lightgbm, catboost, logreg).",
        )
        parser.add_argument(
            "--max-training-rows",
            type=int,
            default=None,
            help="Optional cap on recent rows for faster experimentation.",
        )
        parser.add_argument(
            "--output-dir",
            type=str,
            default=None,
            help="Override ensemble model output directory.",
        )
        parser.add_argument(
            "--analysis-dir",
            type=str,
            default=None,
            help="Override metrics export directory.",
        )
        parser.add_argument(
            "--top-n",
            type=int,
            default=3,
            help="Number of daily recommendations to optimise precision for.",
        )

        args = parser.parse_args()
        model_names = tuple(
            name.strip() for name in args.base_models.split(",") if name.strip()
        )
        return cls(
            target_return=args.target_return,
            imbalance_boost=args.imbalance_boost,
            imbalance_strategy=args.imbalance_strategy,
            n_splits=args.n_splits,
            test_size=args.test_size,
            evaluation_window=args.evaluation_window,
            base_model_names=model_names,
            max_training_rows=args.max_training_rows,
            output_dir=args.output_dir,
            analysis_dir=args.analysis_dir,
            top_n=args.top_n,
        )


def main() -> None:
    trainer = ClosePrecisionEnsembleTrainer.from_args()
    trainer.train_and_evaluate()


if __name__ == "__main__":
    main()
