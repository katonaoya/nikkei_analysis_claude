#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""下落モデルのキャリブレーション指標 (Brier/ECE) を集計するスクリプト."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def load_predictions(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"予測ファイルが見つかりません: {path}")
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
        elif suffix in {".csv", ".tsv"}:
            sep = '\t' if suffix == ".tsv" else ','
            df = pd.read_csv(path, sep=sep)
        elif suffix in {".json"}:
            df = pd.read_json(path)
        elif suffix in {".pkl", ".pickle"}:
            df = pd.read_pickle(path)
        else:
            raise ValueError(f"未対応のファイル形式です: {path}")
        frames.append(df)

    if not frames:
        raise ValueError("予測ファイルが読み込めませんでした")

    df = pd.concat(frames, ignore_index=True)
    if 'prob_down' not in df.columns:
        raise KeyError("予測データに prob_down 列が存在しません")
    return df


def compute_calibration_metrics(prob: pd.Series, label: pd.Series, bins: int) -> Dict[str, float]:
    if len(prob) != len(label):
        raise ValueError("prob と label の長さが一致しません")

    mask = label.notna() & prob.notna()
    prob = prob[mask].astype(float)
    label = label[mask].astype(float)
    if prob.empty:
        return {
            'count': 0,
            'positive_rate': np.nan,
            'mean_prob': np.nan,
            'brier_score': np.nan,
            'ece': np.nan,
            'max_gap': np.nan,
        }

    brier = float(np.mean((prob - label) ** 2))

    # Expected Calibration Error
    bins = max(int(bins), 1)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.clip(np.digitize(prob, bin_edges, right=True) - 1, 0, bins - 1)
    calib_df = pd.DataFrame({
        'prob': prob,
        'label': label,
        'bin': bin_ids,
    })
    grouped = calib_df.groupby('bin')
    total_count = len(calib_df)
    ece = 0.0
    max_gap = 0.0
    for _, grp in grouped:
        bin_prob = float(grp['prob'].mean())
        bin_acc = float(grp['label'].mean())
        gap = abs(bin_prob - bin_acc)
        weight = len(grp) / total_count
        ece += gap * weight
        max_gap = max(max_gap, gap)

    return {
        'count': float(total_count),
        'positive_rate': float(label.mean()) if len(label) else np.nan,
        'mean_prob': float(prob.mean()) if len(prob) else np.nan,
        'brier_score': brier,
        'ece': float(ece),
        'max_gap': float(max_gap),
    }


def build_report(df: pd.DataFrame, *, label_prefix: str, bins: int) -> pd.DataFrame:
    labels = [col for col in df.columns if col.startswith(label_prefix)]
    if not labels and label_prefix == 'down_target_' and 'down_target' in df.columns:
        labels = ['down_target']
    if not labels:
        raise ValueError(f"'{label_prefix}' で始まるラベル列が見つかりません")

    records: List[Dict[str, float]] = []
    for col in labels:
        metrics = compute_calibration_metrics(df['prob_down'], df[col], bins)
        row = {'label_column': col}
        row.update(metrics)
        records.append(row)

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="下落モデルのキャリブレーション指標を算出")
    parser.add_argument('--predictions', type=str, nargs='+', default=['production_data/downside_predictions.parquet'], help='予測ファイル (複数指定可)')
    parser.add_argument('--label-prefix', type=str, default='down_target_', help='評価対象ラベル列の接頭辞')
    parser.add_argument('--bins', type=int, default=10, help='ECE 計算に用いるビン数')
    parser.add_argument('--export-csv', type=str, help='結果を保存する CSV パス')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(p) for p in args.predictions]
    df = load_predictions(paths)
    report_df = build_report(df, label_prefix=args.label_prefix, bins=args.bins)

    with pd.option_context('display.float_format', lambda v: f"{v:.4f}"):
        print(report_df)

    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        report_df.to_csv(export_path, index=False)
        print(f"📄 指標を出力しました: {export_path}")


if __name__ == '__main__':
    main()
