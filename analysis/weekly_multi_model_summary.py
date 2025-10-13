#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a weekly-style summary for multi-model metrics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd


@dataclass
class MetricsSummary:
    start_date: pd.Timestamp
    end_date: pd.Timestamp
    days: int
    precision_mean: float
    precision_median: float
    avg_return_mean: float
    coverage_mean: float
    fallback_ratio_mean: float
    fallback_full_days: int
    fallback_fraction: float
    selected_total: int
    fallback_total: int


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"メトリクスファイルが見つかりません: {path}")
    df = pd.read_csv(path, parse_dates=['analysis_date'])
    if 'precision' not in df or 'fallback_ratio' not in df:
        raise KeyError('precision / fallback_ratio 列が必要です')
    df = df.sort_values('analysis_date')
    return df


def select_recent_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    if df.empty:
        return df
    unique_dates = df['analysis_date'].dropna().unique()
    if days <= 0 or len(unique_dates) <= days:
        return df
    keep_dates = set(unique_dates[-days:])
    return df[df['analysis_date'].isin(keep_dates)].copy()


def compute_summary(df: pd.DataFrame) -> MetricsSummary:
    if df.empty:
        raise ValueError('集計対象のデータが空です')

    precision = df['precision'].dropna()
    avg_return = df['avg_return'].dropna()
    fallback_ratio = df['fallback_ratio']

    fallback_total = int(df['fallback_count'].sum()) if 'fallback_count' in df else int((df['fallback_ratio'] > 0).sum())
    selected_total = int(df['selected_count'].sum()) if 'selected_count' in df else len(df)
    fallback_full_days = int((df['fallback_ratio'] == 1.0).sum())

    summary = MetricsSummary(
        start_date=df['analysis_date'].min(),
        end_date=df['analysis_date'].max(),
        days=df['analysis_date'].nunique(),
        precision_mean=float(precision.mean()) if not precision.empty else float('nan'),
        precision_median=float(precision.median()) if not precision.empty else float('nan'),
        avg_return_mean=float(avg_return.mean()) if not avg_return.empty else float('nan'),
        coverage_mean=float(df['coverage'].mean()) if 'coverage' in df else float('nan'),
        fallback_ratio_mean=float(fallback_ratio.mean()),
        fallback_full_days=fallback_full_days,
        fallback_fraction=fallback_total / selected_total if selected_total else float('nan'),
        selected_total=selected_total,
        fallback_total=fallback_total,
    )
    return summary


def render_markdown(summary: MetricsSummary, *, title: str, window: int) -> str:
    lines = [
        f"# {title}",
        "",
        f"- 期間: {summary.start_date.date()} 〜 {summary.end_date.date()} ({summary.days} 営業日, 集計窓 {window} 日)",
        f"- Precision: 平均 {summary.precision_mean:.2%} / 中央値 {summary.precision_median:.2%}",
        f"- 平均リターン: {summary.avg_return_mean:.2%}",
        f"- フォールバック比率: 平均 {summary.fallback_ratio_mean:.2%} / 完全フォールバック日 {summary.fallback_full_days}",
        f"- フォールバック採用件数: {summary.fallback_total} / 総採用 {summary.selected_total} ({summary.fallback_fraction:.2%})",
    ]
    if not pd.isna(summary.coverage_mean):
        lines.insert(4, f"- Coverage: 平均 {summary.coverage_mean:.2%}")
    return "\n".join(lines) + "\n"


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='マルチモデル指標の週次サマリを生成')
    parser.add_argument('--metrics', type=str, default='production_data/multi_model_metrics.csv', help='メトリクスCSVパス')
    parser.add_argument('--days', type=int, default=14, help='最新何営業日を集計するか')
    parser.add_argument('--title', type=str, default='マルチモデル指標サマリ', help='レポートタイトル')
    parser.add_argument('--output', type=str, default=None, help='出力Markdownパス (未指定なら標準出力のみ)')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    df = load_metrics(Path(args.metrics))
    df_recent = select_recent_days(df, args.days)
    summary = compute_summary(df_recent)
    markdown = render_markdown(summary, title=args.title, window=args.days)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(markdown, encoding='utf-8')
        print(f"📄 週次サマリを出力しました: {output_path}")

    print(markdown, end='')


if __name__ == '__main__':
    main()

