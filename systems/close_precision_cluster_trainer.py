#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Train cluster-specific meta models on top of the existing ensemble."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1

ARTIFACT_DIR = ROOT / "models" / "ensemble_close_v2"
LATEST_ARTIFACT = ARTIFACT_DIR / "latest_ensemble_model.joblib"
OUTPUT_DIR = ARTIFACT_DIR
REPORT_PATH = ROOT / "analysis" / "cluster_meta_metrics.json"


@dataclass
class ClusterConfig:
    code_to_cluster: Dict[str, str]
    cluster_meta: Dict[str, Dict[str, object]]
    cluster_metrics: Dict[str, Dict[str, float]]
    holdout_precision: float
    holdout_top3_precision: float


class ClusterMetaTrainer:
    def __init__(self, evaluation_window: int = 63, top_n: int = 3, min_samples: int = 500) -> None:
        self.evaluation_window = evaluation_window
        self.top_n = top_n
        self.min_samples = min_samples

        self.system = CloseReturnPrecisionSystemV1(calibration_method="none")
        self.artifact = joblib.load(LATEST_ARTIFACT)
        self.base_models = self.artifact["base_models"]
        self.meta_model = self.artifact["meta_model"]
        self.calibrator = self.artifact.get("calibrator")
        self.feature_cols: List[str] = self.artifact["feature_cols"]

    # ------------------------------------------------------------------
    def build_dataset(self) -> pd.DataFrame:
        base_df = self.system.load_and_integrate_data()
        features = self.system.create_enhanced_features(base_df)
        features = features.sort_values(["Date", "Code"]).reset_index(drop=True)
        features["Date"] = pd.to_datetime(features["Date"])
        features["Code"] = features["Code"].astype(str)

        numeric_cols = features.select_dtypes(include=[np.number]).columns
        features[numeric_cols] = (
            features.groupby("Code")[numeric_cols]
            .apply(lambda g: g.fillna(method="ffill"))
            .reset_index(level=0, drop=True)
        )
        features[numeric_cols] = features[numeric_cols].fillna(0.0)

        # derive liquidity/volatility clusters
        agg = (
            features.groupby("Code")
            .agg(
                avg_turnover=("TurnoverValue", "mean"),
                avg_volume=("Volume", "mean"),
                avg_volatility=("Volatility_20", "mean") if "Volatility_20" in features.columns else ("Returns", "std"),
            )
            .reset_index()
        )

        def assign_bucket(series: pd.Series, labels: Tuple[str, str, str]) -> pd.Series:
            q1, q2 = series.quantile([0.33, 0.66]).tolist()
            return pd.cut(series, bins=[-np.inf, q1, q2, np.inf], labels=labels).astype(str)

        agg["LiquidityCluster"] = assign_bucket(agg["avg_turnover"], ("Low", "Mid", "High"))
        agg["VolatilityCluster"] = assign_bucket(agg["avg_volatility"], ("Calm", "Neutral", "Active"))
        agg["ClusterLabel"] = agg["LiquidityCluster"] + "_" + agg["VolatilityCluster"]

        merged = features.merge(agg[["Code", "LiquidityCluster", "VolatilityCluster", "ClusterLabel"]], on="Code", how="left")
        merged["ClusterLabel"].fillna("Unknown", inplace=True)
        return merged

    # ------------------------------------------------------------------
    def compute_base_outputs(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].values.astype(np.float32)
        base_outputs = []
        for name in sorted(self.base_models.keys()):
            model = self.base_models[name]
            prob = self._predict_proba(model, X)
            base_outputs.append(prob)
        stacked = np.column_stack(base_outputs)
        return stacked

    @staticmethod
    def _predict_proba(model, X: np.ndarray) -> np.ndarray:
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(X)
            if isinstance(prob, tuple):
                prob = prob[0]
            return prob[:, 1]
        if hasattr(model, "decision_function"):
            decision = model.decision_function(X)
            return 1.0 / (1.0 + np.exp(-decision))
        preds = model.predict(X)
        return preds.astype(float)

    # ------------------------------------------------------------------
    def train(self) -> ClusterConfig:
        dataset = self.build_dataset()
        stacked_features = self.compute_base_outputs(dataset)

        unique_dates = np.array(sorted(dataset["Date"].unique()))
        if len(unique_dates) <= self.evaluation_window:
            raise ValueError("データ期間が短すぎます")
        eval_start = unique_dates[-self.evaluation_window]

        train_mask = dataset["Date"] < eval_start
        holdout_mask = dataset["Date"] >= eval_start

        train_df = dataset.loc[train_mask].reset_index(drop=True)
        holdout_df = dataset.loc[holdout_mask].reset_index(drop=True)
        train_features = stacked_features[train_mask.values]
        holdout_features = stacked_features[holdout_mask.values]

        code_to_cluster = (
            train_df.groupby("Code")["ClusterLabel"]
            .agg(lambda x: x.mode().iat[0] if not x.mode().empty else "Unknown")
            .to_dict()
        )

        cluster_meta: Dict[str, Dict[str, object]] = {}
        cluster_metrics: Dict[str, Dict[str, float]] = {}

        clusters = train_df["ClusterLabel"].unique()
        for cluster in clusters:
            cluster_mask = train_df["ClusterLabel"] == cluster
            if cluster_mask.sum() < self.min_samples:
                continue

            model, calib, metrics = self._train_cluster_model(
                train_df.loc[cluster_mask],
                train_features[cluster_mask.values],
                eval_start,
                cluster,
            )
            cluster_meta[cluster] = {"model": model, "calibrator": calib}
            cluster_metrics[cluster] = metrics

        # evaluate on holdout using cluster models
        holdout_probs = self._predict_with_clusters(
            codes=holdout_df["Code"].to_numpy(),
            clusters=holdout_df["ClusterLabel"].to_numpy(),
            features=holdout_features,
            global_model=self.meta_model,
            global_calibrator=self.calibrator,
            cluster_meta=cluster_meta,
        )

        holdout_top3_precision = self._topn_precision(
            dates=holdout_df["Date"].to_numpy(),
            codes=holdout_df["Code"].to_numpy(),
            probs=holdout_probs,
            targets=holdout_df["Target"].to_numpy(),
            top_n=self.top_n,
        )

        global_probs = self.meta_model.predict_proba(holdout_features)[:, 1]
        if self.calibrator is not None:
            global_probs = self.calibrator.predict(global_probs)
        global_top3_precision = self._topn_precision(
            dates=holdout_df["Date"].to_numpy(),
            codes=holdout_df["Code"].to_numpy(),
            probs=global_probs,
            targets=holdout_df["Target"].to_numpy(),
            top_n=self.top_n,
        )

        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "evaluation_start": str(eval_start.date()),
            "top_n": self.top_n,
            "baseline_top3_precision": global_top3_precision,
            "cluster_top3_precision": holdout_top3_precision,
            "cluster_metrics": cluster_metrics,
        }
        REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2))

        # update artifact and persist
        updated = dict(self.artifact)
        updated["cluster_meta_models"] = cluster_meta
        updated["code_cluster_map"] = code_to_cluster
        updated["cluster_metrics"] = cluster_metrics
        updated["cluster_holdout_top3_precision"] = holdout_top3_precision
        updated["baseline_holdout_top3_precision"] = global_top3_precision

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUTPUT_DIR / f"close_precision_ensemble_clustered_{timestamp}.joblib"
        joblib.dump(updated, out_path)
        joblib.dump(updated, LATEST_ARTIFACT)

        return ClusterConfig(
            code_to_cluster=code_to_cluster,
            cluster_meta=cluster_meta,
            cluster_metrics=cluster_metrics,
            holdout_precision=global_top3_precision,
            holdout_top3_precision=holdout_top3_precision,
        )

    # ------------------------------------------------------------------
    def _train_cluster_model(
        self,
        cluster_df: pd.DataFrame,
        cluster_features: np.ndarray,
        eval_start: np.datetime64,
        cluster_name: str,
    ) -> Tuple[LogisticRegression, Optional[IsotonicRegression], Dict[str, float]]:
        # use internal split for validation
        dates = np.array(sorted(cluster_df["Date"].unique()))
        if len(dates) <= self.evaluation_window:
            split_idx = len(cluster_df) // 5
            train_X = cluster_features[:-split_idx]
            train_y = cluster_df["Target"].values[:-split_idx]
            val_X = cluster_features[-split_idx:]
            val_y = cluster_df["Target"].values[-split_idx:]
        else:
            tscv = TimeSeriesSplit(n_splits=3, test_size=21)
            best_model = None
            best_precision = -np.inf
            best_config = {}
            for class_weight in self._class_weight_grid():
                for C in [0.2, 0.4, 0.7, 1.0]:
                    fold_precisions = []
                    for train_idx, val_idx in tscv.split(cluster_features):
                        model = LogisticRegression(
                            max_iter=1000,
                            class_weight=class_weight,
                            solver="lbfgs",
                            C=C,
                            random_state=42,
                        )
                        model.fit(cluster_features[train_idx], cluster_df["Target"].values[train_idx])
                        val_prob = model.predict_proba(cluster_features[val_idx])[:, 1]
                        fold_precisions.append(
                            self._topn_precision(
                                dates=cluster_df["Date"].values[val_idx],
                                codes=cluster_df["Code"].values[val_idx],
                                probs=val_prob,
                                targets=cluster_df["Target"].values[val_idx],
                                top_n=self.top_n,
                            )
                        )
                    mean_precision = float(np.nanmean(fold_precisions))
                    if mean_precision > best_precision:
                        best_precision = mean_precision
                        best_model = LogisticRegression(
                            max_iter=1000,
                            class_weight=class_weight,
                            solver="lbfgs",
                            C=C,
                            random_state=42,
                        )
                        best_model.fit(cluster_features, cluster_df["Target"].values)
                        best_config = {"C": C, "class_weight": class_weight, "cv_top3_precision": mean_precision}
            if best_model is None:
                best_model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
                best_model.fit(cluster_features, cluster_df["Target"].values)
                best_config = {"C": 1.0, "class_weight": "balanced", "cv_top3_precision": float("nan")}
            calibrator = self._fit_calibrator(best_model, cluster_features, cluster_df["Target"].values)
            metrics = {
                "cv_top3_precision": best_config["cv_top3_precision"],
                "C": best_config["C"],
                "class_weight": str(best_config["class_weight"]),
                "samples": int(len(cluster_df)),
            }
            return best_model, calibrator, metrics

        # fallback simple split
        model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42)
        model.fit(train_X, train_y)
        calibrator = self._fit_calibrator(model, train_X, train_y)
        val_prob = model.predict_proba(val_X)[:, 1]
        val_precision = self._topn_precision(
            dates=cluster_df["Date"].values[-len(val_y):],
            codes=cluster_df["Code"].values[-len(val_y):],
            probs=val_prob,
            targets=val_y,
            top_n=self.top_n,
        )
        metrics = {
            "holdout_top3_precision": float(val_precision),
            "samples": int(len(cluster_df)),
        }
        return model, calibrator, metrics

    @staticmethod
    def _class_weight_grid() -> List[Optional[dict]]:
        return [
            "balanced",
            {0: 1.0, 1: 0.75},
            {0: 1.0, 1: 0.55},
            {0: 1.0, 1: 0.45},
            {0: 1.0, 1: 0.35},
            {0: 1.0, 1: 0.25},
        ]

    @staticmethod
    def _fit_calibrator(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Optional[IsotonicRegression]:
        prob = model.predict_proba(X)[:, 1]
        if len(np.unique(y)) < 2:
            return None
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(prob, y)
        return calibrator

    def _predict_with_clusters(
        self,
        codes: np.ndarray,
        clusters: np.ndarray,
        features: np.ndarray,
        global_model,
        global_calibrator,
        cluster_meta: Dict[str, Dict[str, object]],
    ) -> np.ndarray:
        global_prob = global_model.predict_proba(features)[:, 1]
        if global_calibrator is not None:
            global_prob = global_calibrator.predict(global_prob)
        preds = global_prob.copy()

        for idx, (code, cluster) in enumerate(zip(codes, clusters)):
            info = cluster_meta.get(cluster)
            if not info:
                continue
            model: LogisticRegression = info["model"]
            calibrator: Optional[IsotonicRegression] = info.get("calibrator")
            prob = model.predict_proba(features[idx : idx + 1])[:, 1]
            if calibrator is not None:
                prob = calibrator.predict(prob)
            preds[idx] = prob[0]
        return preds

    def _topn_precision(
        self,
        dates: np.ndarray,
        codes: np.ndarray,
        probs: np.ndarray,
        targets: np.ndarray,
        top_n: int,
    ) -> float:
        df = pd.DataFrame({"Date": dates, "Code": codes, "Probability": probs, "Target": targets})
        hits = 0
        selected = 0
        for _, group in df.groupby("Date"):
            ranked = group.nlargest(top_n, "Probability")
            hits += int(ranked["Target"].sum())
            selected += len(ranked)
        if selected == 0:
            return float("nan")
        return hits / selected


def main() -> None:
    trainer = ClusterMetaTrainer()
    config = trainer.train()
    print(json.dumps(
        {
            "baseline_top3_precision": config.holdout_precision,
            "cluster_top3_precision": config.holdout_top3_precision,
            "trained_clusters": list(config.cluster_meta.keys()),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
