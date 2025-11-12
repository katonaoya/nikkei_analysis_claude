#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sector-wise diagnostics for multi-model planning.

This script loads the latest ensemble artifact, merges sector metadata,
computes baseline Top-3 precision per sector cluster, and writes
summary tables for further modelling.
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1

SECTOR_MAP_PATH = ROOT / "docment" / "ユーザー情報" / "nikkei225_matched_companies_20250909_230026.csv"
ENSEMBLE_ARTIFACT = ROOT / "models" / "ensemble_close_v2" / "latest_ensemble_model.joblib"
OUTPUT_DIR = ROOT / "analysis"
SUMMARY_CSV = OUTPUT_DIR / "sector_top3_precision_summary.csv"
DAILY_CSV = OUTPUT_DIR / "sector_top3_daily_breakdown.csv"
CONFIG_JSON = OUTPUT_DIR / "sector_grouping_plan.json"


@dataclass
class SectorSummary:
    group_label: str
    observations: int
    positive_ratio: float
    top3_hits: int
    top3_selected: int

    @property
    def top3_precision(self) -> float:
        if self.top3_selected == 0:
            return float("nan")
        return self.top3_hits / self.top3_selected


def load_dataset() -> pd.DataFrame:
    system = CloseReturnPrecisionSystemV1(calibration_method="none")
    base_df = system.load_and_integrate_data()
    features = system.create_enhanced_features(base_df)
    features = features.sort_values(["Date", "Code"]).reset_index(drop=True)
    features["Code"] = features["Code"].astype(str)
    # merge sector metadata
    if SECTOR_MAP_PATH.exists():
        sector_df = pd.read_csv(SECTOR_MAP_PATH, encoding="utf-8-sig")
        sector_df["Code"] = sector_df["target_code"].astype(str)
        sector_df = sector_df[["Code", "target_name", "sector"]]
        sector_df = sector_df.rename(
            columns={
                "target_name": "CompanyNameJP",
                "sector": "SectorName",
            }
        )
        features = features.merge(sector_df, on="Code", how="left")
    else:
        features["SectorName"] = "Unknown"
    features["SectorName"].fillna("Unknown", inplace=True)

    # 流動性・ボラティリティのクラスタリング指標を作る
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
        return pd.cut(
            series,
            bins=[-np.inf, q1, q2, np.inf],
            labels=labels,
        ).astype(str)

    agg["LiquidityCluster"] = assign_bucket(agg["avg_turnover"], ("Low", "Mid", "High"))
    agg["VolatilityCluster"] = assign_bucket(agg["avg_volatility"], ("Calm", "Neutral", "Active"))

    features = features.merge(agg[["Code", "LiquidityCluster", "VolatilityCluster"]], on="Code", how="left")
    features["LiquidityCluster"].fillna("Unknown", inplace=True)
    features["VolatilityCluster"].fillna("Unknown", inplace=True)

    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].fillna(0.0)
    return features


def load_ensemble_artifact():
    if not ENSEMBLE_ARTIFACT.exists():
        raise FileNotFoundError(f"Ensemble artifact not found: {ENSEMBLE_ARTIFACT}")
    artifact = joblib.load(ENSEMBLE_ARTIFACT)
    base_models = artifact["base_models"]
    meta_model = artifact["meta_model"]
    feature_cols = artifact["feature_cols"]
    calibrator = artifact.get("calibrator")
    top_n = artifact.get("top_n", 3)
    return artifact, base_models, meta_model, calibrator, feature_cols, top_n


def compute_probabilities(df: pd.DataFrame, feature_cols: List[str], base_models: Dict[str, object], meta_model, calibrator) -> np.ndarray:
    X = df[feature_cols].values.astype(np.float32)
    base_outputs = []
    for name in sorted(base_models.keys()):
        model = base_models[name]
        prob = predict_proba(model, X)
        base_outputs.append(prob)
    stacked = np.column_stack(base_outputs)
    ensemble_prob = meta_model.predict_proba(stacked)[:, 1]
    if calibrator is not None:
        ensemble_prob = calibrator.predict(ensemble_prob)
    return ensemble_prob


def predict_proba(model, X: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(X)
        if isinstance(prob, tuple):
            prob = prob[0]
        return prob[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X)
        return 1 / (1 + np.exp(-decision))
    preds = model.predict(X)
    return preds.astype(float)


def evaluate_top3(df: pd.DataFrame, probabilities: np.ndarray, top_n: int, group_col: str) -> Tuple[List[SectorSummary], pd.DataFrame]:
    work = df.copy().reset_index(drop=True)
    work["__prob__"] = probabilities

    counters: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    total_counts = work.groupby(group_col)["Target"].agg(["count", "mean"]).reset_index()

    daily_rows = []

    for date, day_df in work.groupby("Date"):
        ranked = day_df.nlargest(top_n, "__prob__")
        for _, row in ranked.iterrows():
            label = (row.get(group_col) or "Unknown")
            counters[label]["selected"] += 1
            if row.get("Target") == 1:
                counters[label]["hits"] += 1
            daily_rows.append(
                {
                    "Date": date,
                    "Code": row["Code"],
                    group_col: label,
                    "Probability": row["__prob__"],
                    "Target": row["Target"],
                }
            )

    summaries: List[SectorSummary] = []
    for _, record in total_counts.iterrows():
        label = record[group_col] or "Unknown"
        selected = counters[label]["selected"]
        hits = counters[label]["hits"]
        summaries.append(
            SectorSummary(
                group_label=str(label),
                observations=int(record["count"]),
                positive_ratio=float(record["mean"]),
                top3_hits=hits,
                top3_selected=selected,
            )
        )

    return summaries, pd.DataFrame(daily_rows)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset()
    artifact, base_models, meta_model, calibrator, feature_cols, top_n = load_ensemble_artifact()

    # 対象期間: 最新63営業日（約3ヶ月）
    last_date = dataset["Date"].max()
    cutoff = last_date - pd.Timedelta(days=126)
    recent_df = dataset[dataset["Date"] >= cutoff].copy()

    probabilities = compute_probabilities(recent_df, feature_cols, base_models, meta_model, calibrator)
    grouping_columns = {
        "SectorName": "sector",
        "LiquidityCluster": "liquidity",
        "VolatilityCluster": "volatility",
    }

    summary_frames = []
    daily_frames = []

    for col, label in grouping_columns.items():
        summaries, daily_df = evaluate_top3(recent_df, probabilities, top_n=top_n, group_col=col)
        if not summaries:
            continue
        summary_df = pd.DataFrame(
            {
                "Grouping": label,
                "GroupLabel": [s.group_label for s in summaries],
                "Observations": [s.observations for s in summaries],
                "PositiveRatio": [s.positive_ratio for s in summaries],
                "Top3Hits": [s.top3_hits for s in summaries],
                "Top3Selected": [s.top3_selected for s in summaries],
                "Top3Precision": [s.top3_precision for s in summaries],
            }
        )
        summary_frames.append(summary_df)
        daily_df = daily_df.rename(columns={col: "GroupLabel"})
        daily_df["Grouping"] = label
        daily_frames.append(daily_df)

    if summary_frames:
        summary_table = pd.concat(summary_frames, ignore_index=True)
        summary_table = summary_table.sort_values(["Grouping", "Top3Precision"], ascending=[True, False])
        summary_table.to_csv(SUMMARY_CSV, index=False)

    if daily_frames:
        daily_table = pd.concat(daily_frames, ignore_index=True)
        daily_table.to_csv(DAILY_CSV, index=False)

    config = {
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "top_n": int(top_n),
        "universe_days": int(recent_df["Date"].nunique()),
        "sector_candidates": summary_df.head(10).to_dict(orient="records"),
        "artifact_path": str(ENSEMBLE_ARTIFACT.relative_to(ROOT)),
    }
    CONFIG_JSON.write_text(json.dumps(config, ensure_ascii=False, indent=2))

    print(f"Sector summary saved to: {SUMMARY_CSV}")
    print(f"Daily breakdown saved to: {DAILY_CSV}")
    print(f"Plan metadata saved to: {CONFIG_JSON}")


if __name__ == "__main__":
    main()
