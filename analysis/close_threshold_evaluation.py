#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""終値ベース閾値評価ユーティリティ"""

import argparse
import re
from bisect import bisect_right
from pathlib import Path
from typing import List

import pandas as pd

REPORT_DIRS = [Path("production_reports/2025-09"), Path("production_reports/2025-10")]
REPORT_PATTERN_NEW = re.compile(r"\d+位: (?P<name>.+?) \((?P<code>\d+)\)")
REPORT_PATTERN_DATE = re.compile(r"基準日付: (\d{4}-\d{2}-\d{2})")
REPORT_PATTERN_ALT_DATE = re.compile(r"対象日: (\d{4}-\d{2}-\d{2})")


def find_latest_price_file() -> Path:
    files = sorted(Path("data/processed").glob("nikkei225_complete_225stocks_*.parquet"))
    if not files:
        raise FileNotFoundError("nikkei225_complete_225stocks_*.parquet が見つかりません")
    return files[-1]


def parse_reports() -> pd.DataFrame:
    records = []
    for report_dir in REPORT_DIRS:
        if not report_dir.exists():
            continue
        for report in sorted(report_dir.glob("*.md")):
            text = report.read_text(encoding="utf-8")
            m = REPORT_PATTERN_DATE.search(text) or REPORT_PATTERN_ALT_DATE.search(text)
            if not m:
                continue
            base_date = pd.to_datetime(m.group(1))
            for match in REPORT_PATTERN_NEW.finditer(text):
                records.append({
                    "report": report.name,
                    "report_date": base_date,
                    "code_raw": match.group("code").strip(),
                    "company": match.group("name").strip(),
                })
    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("推奨銘柄が見つかりませんでした")
    return df


def load_price_data() -> pd.DataFrame:
    path = find_latest_price_file()
    df = pd.read_parquet(path, columns=["Date", "Code", "Close"])
    df["Date"] = pd.to_datetime(df["Date"])
    return df.sort_values(["Code", "Date"])


def evaluate_thresholds(thresholds: List[float], transaction_cost: float = 0.0) -> pd.DataFrame:
    report_df = parse_reports()
    price_df = load_price_data()

    code_groups = price_df.groupby("Code")
    code_dates = {code: grp["Date"].tolist() for code, grp in code_groups}

    rows = []
    for rec in report_df.itertuples():
        code_int = int(float(rec.code_raw))
        code = code_int * 10 if code_int < 10000 else code_int
        entry = {"report_date": rec.report_date, "company": rec.company, "code": code}
        if code not in code_dates:
            entry["status"] = "code_missing"
            rows.append(entry)
            continue
        grp = code_groups.get_group(code)
        base_rows = grp[grp["Date"] == rec.report_date]
        if base_rows.empty:
            entry["status"] = "base_missing"
            rows.append(entry)
            continue
        idx = bisect_right(code_dates[code], rec.report_date.to_pydatetime())
        if idx >= len(code_dates[code]):
            entry["status"] = "next_missing"
            rows.append(entry)
            continue
        base_close = base_rows["Close"].iloc[0]
        next_close = grp.iloc[idx]["Close"]
        ret = next_close / base_close - 1
        entry.update({
            "status": "OK",
            "return_next": ret,
            "net_return": ret - transaction_cost,
        })
        rows.append(entry)

    results = pd.DataFrame(rows)
    valid = results[results["status"] == "OK"].copy()

    summaries = []
    for th in thresholds:
        hit = (valid["return_next"] >= th).astype(int)
        df_hit = valid.assign(hit=hit)
        summary = df_hit.groupby("report_date").agg(
            recommendations=("hit", "size"),
            hits=("hit", "sum"),
            hit_rate=("hit", "mean"),
            avg_return=("return_next", "mean"),
            avg_net_return=("net_return", "mean"),
        ).reset_index()
        summaries.append({
            "threshold": th,
            "overall_hit_rate": hit.mean(),
            "overall_avg_return": valid["return_next"].mean(),
            "overall_avg_net_return": valid["net_return"].mean(),
            "daily_table": summary,
        })

    return pd.DataFrame(summaries)


def main():
    parser = argparse.ArgumentParser(description="終値閾値評価")
    parser.add_argument("--thresholds", type=str, default="0.006,0.008,0.01,0.012", help="カンマ区切り閾値")
    parser.add_argument("--transaction-cost", type=float, default=0.0, help="1取引あたりのコスト比率")
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x]
    df = evaluate_thresholds(thresholds, transaction_cost=args.transaction_cost)
    for row in df.itertuples():
        print("---")
        print(f"閾値 +{row.threshold*100:.1f}%")
        print(f"  全体ヒット率: {row.overall_hit_rate*100:.2f}%")
        print(f"  平均翌日リターン: {row.overall_avg_return*100:.2f}%")
        print(f"  平均翌日ネットリターン: {row.overall_avg_net_return*100:.2f}%")

if __name__ == "__main__":
    main()
