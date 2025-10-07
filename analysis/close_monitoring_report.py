#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推奨銘柄のモニタリングレポートを生成する"""

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.close_threshold_evaluation import evaluate_thresholds

OUTPUT_DIR = Path("reports/monitoring")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    thresholds = [0.006, 0.008, 0.010]
    summary_df = evaluate_thresholds(thresholds)

    records = []
    for row in summary_df.itertuples():
        records.append({
            "threshold": row.threshold,
            "overall_hit_rate": row.overall_hit_rate,
            "overall_avg_return": row.overall_avg_return
        })
        daily_table = row.daily_table
        daily_table["threshold"] = row.threshold
        daily_table.to_csv(OUTPUT_DIR / f"threshold_{row.threshold:.3f}_daily.csv", index=False)

    summary_csv = OUTPUT_DIR / "threshold_summary.csv"
    pd.DataFrame(records).to_csv(summary_csv, index=False)

    meta = {
        "generated_at": datetime.now().isoformat(),
        "thresholds": thresholds,
        "source_reports": [str(p) for p in Path('production_reports').rglob('*.md')]
    }
    with (OUTPUT_DIR / "metadata.json").open('w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f"Monitoring summary saved to {summary_csv}")


if __name__ == "__main__":
    main()
