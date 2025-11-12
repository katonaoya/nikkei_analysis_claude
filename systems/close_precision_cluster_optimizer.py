#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluate multiple cluster groupings and pick the best-performing meta ensemble."""

from __future__ import annotations

import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from systems.close_precision_cluster_trainer import ClusterMetaTrainer, ClusterConfig

ANALYSIS_DIR = ROOT / "analysis"
OPTIMIZER_REPORT = ANALYSIS_DIR / "cluster_optimizer_results.json"


def quantile_labels(series: pd.Series, bins: int, prefix: str) -> pd.Series:
    if bins <= 1:
        return pd.Series([f"{prefix}_All"] * len(series), index=series.index)
    quantiles = np.linspace(0, 1, bins + 1)
    # avoid duplicate edges by adding tiny noise
    series_no_nan = series.fillna(series.median())
    edges = series_no_nan.quantile(quantiles).to_numpy()
    edges = np.unique(edges)
    if len(edges) <= 2:
        edges = np.linspace(series_no_nan.min(), series_no_nan.max(), bins + 1)
    labels = [f"{prefix}_{i+1}" for i in range(len(edges) - 1)]
    return pd.cut(series_no_nan, bins=np.unique(edges), labels=labels, include_lowest=True).astype(str)


def make_liquidity_cluster(agg: pd.DataFrame, bins: int) -> pd.Series:
    return quantile_labels(np.log1p(agg["avg_turnover"]), bins, "Liq")


def make_volatility_cluster(agg: pd.DataFrame, bins: int) -> pd.Series:
    return quantile_labels(np.log1p(agg["avg_volatility"].clip(lower=0)), bins, "Vol")


def make_momentum_cluster(agg: pd.DataFrame, bins: int) -> pd.Series:
    return quantile_labels(agg["avg_return"], bins, "Mom")


def make_positive_ratio_cluster(agg: pd.DataFrame, bins: int) -> pd.Series:
    return quantile_labels(agg["positive_ratio"], bins, "Pos")


def make_kmeans_cluster(agg: pd.DataFrame, n_clusters: int) -> pd.Series:
    cols = ["log_turnover", "log_volume", "log_volatility", "avg_return", "positive_ratio"]
    data = agg[cols].fillna(0.0).to_numpy()
    scaler = StandardScaler()
    scaled = scaler.fit_transform(data)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    return pd.Series([f"KM{n_clusters}_{label}" for label in labels], index=agg.index)


def combine_labels(*series_list: pd.Series) -> pd.Series:
    result = series_list[0].astype(str)
    for series in series_list[1:]:
        result = result + "_" + series.astype(str)
    return result


def build_scenarios(agg: pd.DataFrame) -> Dict[str, pd.Series]:
    scenarios: Dict[str, pd.Series] = {}

    # Liquidity only
    scenarios["liquidity_q3"] = make_liquidity_cluster(agg, 3)
    scenarios["liquidity_q4"] = make_liquidity_cluster(agg, 4)

    # Volatility only
    scenarios["volatility_q3"] = make_volatility_cluster(agg, 3)
    scenarios["volatility_q4"] = make_volatility_cluster(agg, 4)

    # Positive ratio (ヒット率の高さ）
    scenarios["positive_q3"] = make_positive_ratio_cluster(agg, 3)

    # Momentum only
    scenarios["momentum_q3"] = make_momentum_cluster(agg, 3)

    # Combined grids
    scenarios["liquidity3_vol3"] = combine_labels(
        make_liquidity_cluster(agg, 3),
        make_volatility_cluster(agg, 3),
    )
    scenarios["liquidity4_vol3"] = combine_labels(
        make_liquidity_cluster(agg, 4),
        make_volatility_cluster(agg, 3),
    )
    scenarios["liquidity3_vol4"] = combine_labels(
        make_liquidity_cluster(agg, 3),
        make_volatility_cluster(agg, 4),
    )
    scenarios["liq3_vol3_mom3"] = combine_labels(
        make_liquidity_cluster(agg, 3),
        make_volatility_cluster(agg, 3),
        make_momentum_cluster(agg, 3),
    )

    # KMeans
    for k in [4, 5, 6, 7, 8]:
        scenarios[f"kmeans_{k}"] = make_kmeans_cluster(agg, k)

    return scenarios


def evaluate_scenarios(
    trainer: ClusterMetaTrainer,
    dataset: pd.DataFrame,
    agg: pd.DataFrame,
    scenarios: Dict[str, pd.Series],
) -> Dict[str, Dict[str, float]]:
    stacked = trainer.compute_base_outputs(dataset)

    unique_dates = np.array(sorted(dataset["Date"].unique()))
    eval_start = unique_dates[-trainer.evaluation_window]
    mask_train = dataset["Date"] < eval_start
    mask_holdout = dataset["Date"] >= eval_start

    train_df = dataset.loc[mask_train].reset_index(drop=True)
    holdout_df = dataset.loc[mask_holdout].reset_index(drop=True)
    train_features = stacked[mask_train.values]
    holdout_features = stacked[mask_holdout.values]

    results: Dict[str, Dict[str, float]] = {}
    best_name = None
    best_precision = -math.inf
    best_config: Optional[ClusterConfig] = None
    best_labels_series: Optional[pd.Series] = None

    for name, labels in scenarios.items():
        # Align labels with codes
        code_labels = labels.reindex(agg.index).fillna("Unknown")
        code_labels_map = code_labels.to_dict()

        train_df["ClusterLabel"] = train_df["Code"].map(code_labels_map).fillna("Unknown")
        holdout_df["ClusterLabel"] = holdout_df["Code"].map(code_labels_map).fillna("Unknown")

        config = train_with_labels(
            trainer,
            train_df,
            holdout_df,
            train_features,
            holdout_features,
            scenario_name=name,
            persist=False,
        )

        results[name] = {
            "top3_precision": config.holdout_top3_precision,
            "baseline_precision": config.holdout_precision,
            "clusters": len(config.cluster_meta),
            "trained_codes": len(config.code_to_cluster),
        }

        if config.holdout_top3_precision > best_precision:
            best_precision = config.holdout_top3_precision
            best_name = name
            best_config = config
            best_labels_series = labels

    if best_name and best_config:
        # Persist best scenario using a fresh trainer/dataset to avoid state carryover
        persist_trainer = ClusterMetaTrainer(
            evaluation_window=trainer.evaluation_window,
            top_n=trainer.top_n,
            min_samples=trainer.min_samples,
        )
        persist_dataset = persist_trainer.build_dataset()
        persist_stacked = persist_trainer.compute_base_outputs(persist_dataset)

        persist_agg = (
            persist_dataset.groupby("Code")
            .agg(
                avg_turnover=("TurnoverValue", "mean"),
                avg_volume=("Volume", "mean"),
                avg_volatility=("Volatility_20", "mean") if "Volatility_20" in persist_dataset.columns else ("Returns", "std"),
                avg_return=("Returns", "mean"),
                positive_ratio=("Target", "mean"),
            )
            .reset_index()
        )
        persist_agg["log_turnover"] = np.log1p(persist_agg["avg_turnover"]).fillna(0.0)
        persist_agg["log_volume"] = np.log1p(persist_agg["avg_volume"]).fillna(0.0)
        persist_agg["log_volatility"] = np.log1p(persist_agg["avg_volatility"].clip(lower=0)).fillna(0.0)
        persist_agg.set_index("Code", inplace=True)

        code_labels_map = best_labels_series.reindex(persist_agg.index).fillna("Unknown").to_dict()

        unique_dates = np.array(sorted(persist_dataset["Date"].unique()))
        eval_start = unique_dates[-persist_trainer.evaluation_window]
        mask_train = persist_dataset["Date"] < eval_start
        mask_holdout = persist_dataset["Date"] >= eval_start

        p_train_df = persist_dataset.loc[mask_train].reset_index(drop=True)
        p_holdout_df = persist_dataset.loc[mask_holdout].reset_index(drop=True)
        p_train_features = persist_stacked[mask_train.values]
        p_holdout_features = persist_stacked[mask_holdout.values]

        p_train_df["ClusterLabel"] = p_train_df["Code"].map(code_labels_map).fillna("Unknown")
        p_holdout_df["ClusterLabel"] = p_holdout_df["Code"].map(code_labels_map).fillna("Unknown")

        train_with_labels(
            persist_trainer,
            p_train_df,
            p_holdout_df,
            p_train_features,
            p_holdout_features,
            scenario_name=best_name,
            persist=True,
        )
        results["best_scenario"] = {
            "name": best_name,
            "top3_precision": best_config.holdout_top3_precision,
            "baseline_precision": best_config.holdout_precision,
        }
    else:
        results["best_scenario"] = {"name": None}

    OPTIMIZER_REPORT.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    return results


def train_with_labels(
    trainer: ClusterMetaTrainer,
    train_df: pd.DataFrame,
    holdout_df: pd.DataFrame,
    train_features: np.ndarray,
    holdout_features: np.ndarray,
    scenario_name: str,
    persist: bool,
) -> ClusterConfig:
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
        if cluster_mask.sum() < trainer.min_samples:
            continue
        model, calib, metrics = trainer._train_cluster_model(
            train_df.loc[cluster_mask],
            train_features[cluster_mask.values],
            np.datetime64(holdout_df["Date"].min()),
            cluster,
        )
        cluster_meta[cluster] = {"model": model, "calibrator": calib}
        cluster_metrics[cluster] = metrics

    holdout_probs = trainer._predict_with_clusters(
        codes=holdout_df["Code"].to_numpy(),
        clusters=holdout_df["ClusterLabel"].to_numpy(),
        features=holdout_features,
        global_model=trainer.meta_model,
        global_calibrator=trainer.calibrator,
        cluster_meta=cluster_meta,
    )
    holdout_top3_precision = trainer._topn_precision(
        dates=holdout_df["Date"].to_numpy(),
        codes=holdout_df["Code"].to_numpy(),
        probs=holdout_probs,
        targets=holdout_df["Target"].to_numpy(),
        top_n=trainer.top_n,
    )

    global_probs = trainer.meta_model.predict_proba(holdout_features)[:, 1]
    if trainer.calibrator is not None:
        global_probs = trainer.calibrator.predict(global_probs)
    global_top3_precision = trainer._topn_precision(
        dates=holdout_df["Date"].to_numpy(),
        codes=holdout_df["Code"].to_numpy(),
        probs=global_probs,
        targets=holdout_df["Target"].to_numpy(),
        top_n=trainer.top_n,
    )

    config = ClusterConfig(
        code_to_cluster=code_to_cluster,
        cluster_meta=cluster_meta,
        cluster_metrics=cluster_metrics,
        holdout_precision=global_top3_precision,
        holdout_top3_precision=holdout_top3_precision,
    )

    if persist:
        artifact = dict(trainer.artifact)
        artifact["cluster_meta_models"] = cluster_meta
        artifact["code_cluster_map"] = code_to_cluster
        artifact["cluster_metrics"] = cluster_metrics
        artifact["cluster_holdout_top3_precision"] = holdout_top3_precision
        artifact["baseline_holdout_top3_precision"] = global_top3_precision
        artifact["cluster_scenario"] = scenario_name

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = ROOT / "models" / "ensemble_close_v2" / f"close_precision_ensemble_{scenario_name}_{timestamp}.joblib"
        joblib.dump(artifact, out_path)
        joblib.dump(artifact, ROOT / "models" / "ensemble_close_v2" / "latest_ensemble_model.joblib")

        report = {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "scenario": scenario_name,
            "top_n": trainer.top_n,
            "baseline_top3_precision": global_top3_precision,
            "cluster_top3_precision": holdout_top3_precision,
            "cluster_metrics": cluster_metrics,
        }
        (ROOT / "analysis" / "cluster_meta_metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))

    return config


def main() -> None:
    trainer = ClusterMetaTrainer()
    dataset = trainer.build_dataset()

    agg = (
        dataset.groupby("Code")
        .agg(
            avg_turnover=("TurnoverValue", "mean"),
            avg_volume=("Volume", "mean"),
            avg_volatility=("Volatility_20", "mean") if "Volatility_20" in dataset.columns else ("Returns", "std"),
            avg_return=("Returns", "mean"),
            positive_ratio=("Target", "mean"),
        )
        .reset_index()
    )
    agg["log_turnover"] = np.log1p(agg["avg_turnover"]).fillna(0.0)
    agg["log_volume"] = np.log1p(agg["avg_volume"]).fillna(0.0)
    agg["log_volatility"] = np.log1p(agg["avg_volatility"].clip(lower=0)).fillna(0.0)
    agg.set_index("Code", inplace=True)

    scenarios = build_scenarios(agg)
    results = evaluate_scenarios(trainer, dataset, agg, scenarios)
    print(json.dumps(results.get("best_scenario", {}), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
