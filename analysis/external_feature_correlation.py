#!/usr/bin/env python3
"""Compute correlations between external features and target returns."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ENHANCED_PATH = Path("data/processed/enhanced_integrated_data.parquet")
OUTPUT_PATH = Path("analysis/external_feature_correlations.csv")

EXTERNAL_PREFIXES = [
    "margin_",
    "hf_",
    "news_",
]


def main() -> None:
    df = pd.read_parquet(ENHANCED_PATH)
    df = df.copy()
    df = df.sort_values(["Stock", "Date"])
    df["Target"] = df.groupby("Stock")["close"].shift(-1) / df["close"] - 1
    df = df.dropna(subset=["Target"])

    cols = [col for col in df.columns if any(col.startswith(prefix) for prefix in EXTERNAL_PREFIXES)]
    corr_records = []
    for col in cols:
        series = df[col]
        if series.dtype.kind not in "fci" or series.isna().all():
            continue
        corr = series.corr(df["Target"])
        corr_records.append({"feature": col, "corr_vs_target": corr})

    corr_df = pd.DataFrame(corr_records).sort_values("corr_vs_target", key=lambda s: s.abs(), ascending=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(OUTPUT_PATH, index=False)
    print("Saved:", OUTPUT_PATH)


if __name__ == "__main__":
    main()
