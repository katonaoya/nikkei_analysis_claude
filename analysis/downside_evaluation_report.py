#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Aggregate evaluation for downside risk predictions."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

TARGET_LABELS = [
    'down_target_1pct',
    'down_target_1pct_2d',
    'down_target_1_5pct',
    'down_target_2pct',
    'drawdown_3pct_3d',
    'no_rebound_2d',
]


def load_predictions(paths: Iterable[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        frames.append(df)
    if not frames:
        raise FileNotFoundError('指定パスから読み込める予測ファイルがありません')
    df = pd.concat(frames, ignore_index=True)
    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
    df = df.sort_values('analysis_date')
    return df


def compute_metrics(df: pd.DataFrame, label: str, bins: int) -> Dict[str, float]:
    if label not in df.columns:
        raise KeyError(f'ラベル列 {label} が見つかりません')
    mask = df['prob_down'].notna() & df[label].notna() & df['future_return'].notna()
    if not mask.any():
        return {'count': 0}
    prob = df.loc[mask, 'prob_down'].astype(float)
    target = df.loc[mask, label].astype(float)
    future = df.loc[mask, 'future_return'].astype(float)
    brier = float(np.mean((prob - target) ** 2))
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for lower, upper in zip(bin_edges[:-1], bin_edges[1:]):
        in_bin = (prob >= lower) & (prob < upper if upper < 1.0 else prob <= upper)
        if not in_bin.any():
            continue
        acc = target[in_bin].mean()
        conf = prob[in_bin].mean()
        ece += abs(acc - conf) * in_bin.mean()
    return {
        'count': int(mask.sum()),
        'positive_rate': float(target.mean()),
        'mean_prob': float(prob.mean()),
        'brier_score': brier,
        'ece': float(ece),
        'avg_future_return': float(future[target == 1].mean()) if (target == 1).any() else float('nan'),
    }


def render_markdown(metrics: Dict[str, Dict[str, float]], start: pd.Timestamp, end: pd.Timestamp, output: Path) -> None:
    lines = [
        '# Downside Evaluation Summary',
        '',
        f'- 期間: {start.date()} 〜 {end.date()}',
        '',
        '| ラベル | 件数 | 陽性率 | Brier | ECE | 備考 |',
        '| --- | --- | --- | --- | --- | --- |',
    ]
    for label, stats in metrics.items():
        if stats.get('count', 0) == 0:
            line = f'| {label} | 0 | - | - | - | データ不足 |'
        else:
            note = '✅ Pass' if stats['brier_score'] <= 0.20 and stats['ece'] <= 0.10 else '⚠️ 未達'
            line = (
                f"| {label} | {stats['count']} | {stats['positive_rate']:.2%} | "
                f"{stats['brier_score']:.4f} | {stats['ece']:.4f} | {note} |"
            )
        lines.append(line)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description='Downside 予測の集計レポートを生成')
    parser.add_argument('--pred-dir', type=str, default='production_data', help='予測ファイルが格納されたディレクトリ')
    parser.add_argument('--lookback-days', type=int, default=90, help='直近何日分を集計するか')
    parser.add_argument('--bins', type=int, default=10, help='ECE計算時のビン数')
    parser.add_argument('--output-markdown', type=str, default='analysis/downside_evaluation_summary.md', help='Markdown出力先')
    parser.add_argument('--output-csv', type=str, default='analysis/downside_evaluation_metrics.csv', help='メトリクスCSV出力先')
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    prediction_files = sorted(pred_dir.glob('downside_predictions_*.parquet'))
    latest = pred_dir / 'downside_predictions.parquet'
    if latest.exists():
        prediction_files.append(latest)

    df = load_predictions(prediction_files)
    if args.lookback_days > 0:
        cutoff = df['analysis_date'].max() - pd.Timedelta(days=args.lookback_days)
        df = df[df['analysis_date'] >= cutoff]
    if df.empty:
        raise ValueError('指定期間内のデータが存在しません')

    metrics_records = {}
    for label in TARGET_LABELS:
        if label not in df.columns:
            continue
        metrics_records[label] = compute_metrics(df, label, bins=args.bins)

    csv_rows = []
    for label, stats in metrics_records.items():
        row = {'label': label}
        row.update(stats)
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(args.output_csv, index=False)
    render_markdown(metrics_records, df['analysis_date'].min(), df['analysis_date'].max(), Path(args.output_markdown))
    print(f"📄 Markdown: {args.output_markdown}")
    print(f"📊 CSV: {args.output_csv}")


if __name__ == '__main__':
    main()
