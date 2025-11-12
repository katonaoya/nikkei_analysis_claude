#!/usr/bin/env python3
"""Generate coverage/quality diagnostics for external datasets."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MARGIN_DIR = Path("data/external/margin_balances")
HIGHFREQ_DIR = Path("data/external/highfreq_market")
DATASET_PATH = Path("data/processed/enhanced_integrated_data.parquet")
CODES_PATH = Path("data/nikkei225_codes.csv")
OUTPUT_DIR = Path("analysis")


@dataclass
class ReportPaths:
    margin_summary: Path
    highfreq_summary: Path
    report_json: Path


def load_latest(path: Path, pattern: str) -> Optional[Path]:
    if not path.exists():
        return None
    files = sorted(path.glob(pattern))
    return files[-1] if files else None


def summarize_margin() -> pd.DataFrame:
    latest = load_latest(MARGIN_DIR, "margin_features_*.parquet")
    if latest is None:
        raise FileNotFoundError("No margin feature files found")

    margin_df = pd.read_parquet(latest)
    margin_df["Code"] = margin_df["Code"].astype(str)
    margin_df["Date"] = pd.to_datetime(margin_df["Date"])

    codes_df = pd.read_csv(CODES_PATH, dtype={"code": str})
    codes_df["code"] = codes_df["code"].str.zfill(4) + "0"

    coverage = (
        margin_df.groupby("Code")["Date"]
        .agg(["min", "max", "count"])
        .rename(columns={"count": "days_available"})
        .reset_index()
    )
    coverage = coverage.merge(codes_df.rename(columns={"code": "Code", "name": "Name"}), on="Code", how="right")
    coverage["missing"] = coverage["days_available"].isna()
    coverage["days_available"] = coverage["days_available"].fillna(0).astype(int)
    return coverage


def summarize_highfreq(enhanced_df: pd.DataFrame) -> pd.DataFrame:
    latest = load_latest(HIGHFREQ_DIR, "highfreq_features_*.parquet")
    if latest is None:
        raise FileNotFoundError("No high frequency feature files found")

    hf_df = pd.read_parquet(latest)
    hf_df["Date"] = pd.to_datetime(hf_df["Date"]).dt.tz_localize(None)
    # Compare with actual returns from enhanced dataset
    enhanced_df = enhanced_df.copy()
    enhanced_df["Date"] = pd.to_datetime(enhanced_df["Date"]).dt.tz_localize(None)
    nikkei_returns = (
        enhanced_df.groupby("Date")["Returns"].mean().rename("mean_returns").reset_index()
    )
    merged = hf_df.merge(nikkei_returns, on="Date", how="left")
    stats = {
        "rows": len(hf_df),
        "start_date": hf_df["Date"].min().isoformat() if not hf_df.empty else None,
        "end_date": hf_df["Date"].max().isoformat() if not hf_df.empty else None,
    }
    if "hf_nikkei225_close_return" in merged.columns:
        stats["correlation_vs_returns"] = (
            merged[["hf_nikkei225_close_return", "mean_returns"]]
            .dropna()
            .corr()
            .iloc[0, 1]
        )
    stats_df = pd.DataFrame([stats])
    stats_df["source_file"] = str(latest)
    return stats_df


def build_report() -> ReportPaths:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    coverage_df = summarize_margin()
    margin_path = OUTPUT_DIR / "margin_data_coverage.csv"
    coverage_df.to_csv(margin_path, index=False)

    enhanced_df = pd.read_parquet(DATASET_PATH)
    hf_summary_df = summarize_highfreq(enhanced_df)
    hf_path = OUTPUT_DIR / "highfreq_data_summary.csv"
    hf_summary_df.to_csv(hf_path, index=False)

    report = {
        "margin": {
            "total_codes": int(coverage_df.shape[0]),
            "missing_codes": int((coverage_df["days_available"] == 0).sum()),
            "min_days_available": int(coverage_df["days_available"].min()),
            "max_days_available": int(coverage_df["days_available"].max()),
        },
        "highfreq": hf_summary_df.to_dict(orient="records"),
    }
    report_path = OUTPUT_DIR / "external_data_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return ReportPaths(margin_path, hf_path, report_path)


if __name__ == "__main__":
    paths = build_report()
    print("Margin coverage:", paths.margin_summary)
    print("High frequency summary:", paths.highfreq_summary)
    print("Report JSON:", paths.report_json)
