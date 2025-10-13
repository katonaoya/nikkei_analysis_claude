#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plot multi-model precision vs fallback ratio trends."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def load_metrics(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"メトリクスファイルが見つかりません: {path}")
    df = pd.read_csv(path, parse_dates=['analysis_date'])
    if 'precision' not in df or 'fallback_ratio' not in df:
        raise KeyError('precision / fallback_ratio 列が必要です')
    return df


def filter_date_range(df: pd.DataFrame, start: Optional[str], end: Optional[str]) -> pd.DataFrame:
    filtered = df.copy()
    if start:
        filtered = filtered[filtered['analysis_date'] >= pd.to_datetime(start)]
    if end:
        filtered = filtered[filtered['analysis_date'] <= pd.to_datetime(end)]
    return filtered


def compute_moving_average(df: pd.DataFrame, window: int) -> pd.DataFrame:
    if window <= 0:
        raise ValueError('window は 1 以上で指定してください')
    result = df.sort_values('analysis_date').copy()
    result['precision_ma'] = result['precision'].rolling(window=window, min_periods=1).mean()
    result['fallback_ratio_ma'] = result['fallback_ratio'].rolling(window=window, min_periods=1).mean()
    return result


def plot_metrics(df: pd.DataFrame, *, output: Path, title: str, dpi: int) -> None:
    if df.empty:
        raise ValueError('プロット対象のデータが空です')

    fig, ax1 = plt.subplots(figsize=(10, 5), dpi=dpi)
    ax2 = ax1.twinx()

    ax1.plot(df['analysis_date'], df['precision_ma'], color='#0055A4', label='Precision (MA)')
    ax1.scatter(df['analysis_date'], df['precision'], color='#5DA5DA', alpha=0.4, s=18, label='Precision (daily)')

    ax2.plot(df['analysis_date'], df['fallback_ratio_ma'], color='#FF7F0E', label='Fallback Ratio (MA)')
    ax2.scatter(df['analysis_date'], df['fallback_ratio'], color='#FFBB78', alpha=0.4, s=18, label='Fallback Ratio (daily)')

    ax1.set_ylabel('Precision')
    ax1.set_ylim(0, 1)
    ax2.set_ylabel('Fallback Ratio')
    ax2.set_ylim(0, 1)
    ax1.set_xlabel('Analysis Date')
    ax1.set_title(title)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    ax1.grid(True, linestyle='--', alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='フォールバック比率とPrecisionの推移を可視化')
    parser.add_argument('--input', type=str, default='production_data/multi_model_metrics.csv', help='メトリクスCSVのパス')
    parser.add_argument('--output', type=str, default='analysis/figures/fallback_precision_latest.png', help='出力画像パス')
    parser.add_argument('--window', type=int, default=7, help='移動平均の窓幅 (営業日単位)')
    parser.add_argument('--start-date', type=str, default=None, help='対象開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='対象終了日 (YYYY-MM-DD)')
    parser.add_argument('--title', type=str, default='Fallback Ratio vs Precision', help='グラフタイトル')
    parser.add_argument('--dpi', type=int, default=150, help='画像解像度 DPI')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)
    metrics_path = Path(args.input)
    output_path = Path(args.output)

    df = load_metrics(metrics_path)
    df = filter_date_range(df, args.start_date, args.end_date)
    df = compute_moving_average(df, args.window)
    plot_metrics(df, output=output_path, title=args.title, dpi=args.dpi)
    print(f"📈 プロットを保存しました: {output_path}")


if __name__ == '__main__':
    main(sys.argv[1:])
