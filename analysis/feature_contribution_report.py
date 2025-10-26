#!/usr/bin/env python3
"""Generate feature contribution and importance report for the close-return model."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("feature_contribution")


def load_model(model_path: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_dataset(dataset_path: Path, feature_cols: list[str], sample_size: int | None = None, random_state: int = 42) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    df = pd.read_parquet(dataset_path)
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Required features missing from dataset: {missing}")
    filtered = df[feature_cols].dropna()
    if sample_size and len(filtered) > sample_size:
        filtered = filtered.sample(sample_size, random_state=random_state)
    LOGGER.info("Loaded dataset with %d rows after dropping NaNs", len(filtered))
    return filtered


def compute_lightgbm_importance(model, feature_names: list[str]) -> pd.DataFrame:
    booster = model.booster_
    gain = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    return pd.DataFrame(
        {
            "feature": feature_names,
            "gain_importance": gain,
            "split_importance": split,
        }
    )


def compute_shap_contrib(model, scaler, selector, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    X_values = X.values
    X_scaled = scaler.transform(X_values)
    X_selected = selector.transform(X_scaled)
    contributions = model.predict(X_selected, pred_contrib=True)
    shap_values = contributions[:, :-1]  # last column is base value
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs})
    return df


def build_report(model_bundle, dataset_path: Path, output_dir: Path, sample_size: int | None = 50000) -> Path:
    feature_cols = model_bundle["feature_cols"]
    dataset = load_dataset(dataset_path, feature_cols, sample_size=sample_size)

    scaler = model_bundle.get("scaler")
    selector = model_bundle.get("selector")
    lgbm_model = model_bundle["model"]

    feature_names = feature_cols
    importance_df = compute_lightgbm_importance(lgbm_model, feature_names)
    shap_df = compute_shap_contrib(lgbm_model, scaler, selector, dataset, feature_names)
    report = importance_df.merge(shap_df, on="feature", how="left")
    report.sort_values("mean_abs_shap", ascending=False, inplace=True)
    shap_threshold = report["mean_abs_shap"].median() * 0.1 if not report.empty else 0.0
    report["low_importance_flag"] = report["mean_abs_shap"] <= shap_threshold

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "feature_contribution_report.csv"
    report.to_csv(output_path, index=False)
    LOGGER.info("Saved feature contribution report to %s", output_path)
    return output_path


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Generate feature contribution report")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/enhanced_close_v1/latest_calibrated_model.joblib"),
        help="Path to the trained model bundle",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("data/processed/enhanced_integrated_data.parquet"),
        help="Feature dataset used for evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("analysis"),
        help="Directory to store the report",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=50000,
        help="Number of rows to sample for SHAP calculation",
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    model_bundle = load_model(args.model_path)
    build_report(model_bundle, args.dataset_path, args.output_dir, sample_size=args.sample_size)


if __name__ == "__main__":  # pragma: no cover
    main()
