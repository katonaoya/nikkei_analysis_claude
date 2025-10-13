#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""マルチモデル推奨の Precision / Return / Coverage を日次で記録する."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.daily_stock_recommendation_multi import (
    prepare_candidate_scores,
    select_top_candidates,
)


def load_config(path: Path) -> Dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    return json.loads(path.read_text())


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"候補データが見つかりません: {path}")
    df = pd.read_parquet(path)
    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
    df['code'] = df['code'].astype(str).str.zfill(4)
    return df


def compute_daily_metrics(
    selected_df: pd.DataFrame,
    *,
    target_return: float,
) -> pd.DataFrame:
    if selected_df.empty:
        return pd.DataFrame(
            [
                {
                    'analysis_date': pd.NaT,
                    'selected_count': 0,
                    'valid_count': 0,
                    'hit_count': 0,
                    'precision': np.nan,
                    'avg_return': np.nan,
                    'coverage': 0.0,
                    'fallback_count': 0,
                    'passed_all_count': 0,
                    'fallback_ratio': 0.0,
                    'passed_all_ratio': 0.0,
                }
            ]
        )

    selected_df = selected_df.copy()
    selected_df['analysis_date'] = pd.to_datetime(selected_df['analysis_date']).dt.normalize()

    records: List[Dict[str, object]] = []
    for analysis_date, group in selected_df.groupby('analysis_date', sort=True):
        valid = group.dropna(subset=['future_return'])
        valid_count = len(valid)
        hits = int((valid['future_return'] >= target_return).sum()) if valid_count else 0
        precision = float(hits / valid_count) if valid_count else np.nan
        avg_return = float(valid['future_return'].mean()) if valid_count else np.nan
        coverage = 1.0 if len(group) > 0 else 0.0
        fallback_count = int((~group['passed_all_filters'].astype(bool)).sum())
        passed_all_count = int(group['passed_all_filters'].astype(bool).sum())
        fallback_ratio = float(fallback_count / len(group)) if len(group) else 0.0
        passed_all_ratio = float(passed_all_count / len(group)) if len(group) else 0.0
        records.append(
            {
                'analysis_date': analysis_date,
                'selected_count': int(len(group)),
                'valid_count': int(valid_count),
                'hit_count': hits,
                'precision': precision,
                'avg_return': avg_return,
                'coverage': coverage,
                'fallback_count': fallback_count,
                'passed_all_count': passed_all_count,
                'fallback_ratio': fallback_ratio,
                'passed_all_ratio': passed_all_ratio,
            }
        )
    return pd.DataFrame(records)


def append_metrics(log_path: Path, daily_df: pd.DataFrame) -> pd.DataFrame:
    daily_df = daily_df.dropna(subset=['analysis_date'])
    if daily_df.empty:
        return pd.DataFrame(
            columns=[
                'analysis_date',
                'selected_count',
                'valid_count',
                'hit_count',
                'precision',
                'avg_return',
                'coverage',
                'fallback_count',
                'passed_all_count',
                'fallback_ratio',
                'passed_all_ratio',
            ]
        )

    if log_path.exists():
        existing = pd.read_csv(log_path, parse_dates=['analysis_date'])
        combined = pd.concat([existing, daily_df], ignore_index=True)
    else:
        combined = daily_df.copy()

    combined['analysis_date'] = pd.to_datetime(combined['analysis_date']).dt.normalize()
    combined = combined.drop_duplicates('analysis_date', keep='last')
    combined = combined.sort_values('analysis_date')
    combined.to_csv(log_path, index=False)
    return combined


def rotate_archives(
    combined_df: pd.DataFrame,
    *,
    log_path: Path,
    archive_dir: Path,
    keep_months: int,
) -> pd.DataFrame:
    if combined_df is None or combined_df.empty:
        return combined_df

    archive_dir.mkdir(parents=True, exist_ok=True)

    df = combined_df.copy()
    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
    df['month'] = df['analysis_date'].dt.to_period('M')

    unique_months = sorted(df['month'].dropna().unique())
    if not unique_months:
        return combined_df

    if keep_months is None:
        keep_months = 0
    if keep_months < 0:
        keep_months = 0

    months_to_keep = set(unique_months[-keep_months:]) if keep_months else set()

    for month in unique_months:
        if month in months_to_keep:
            continue

        month_df = df[df['month'] == month].drop(columns='month', errors='ignore')
        if month_df.empty:
            continue

        archive_path = archive_dir / f"multi_model_metrics_{month.strftime('%Y%m')}.csv"
        if archive_path.exists():
            archive_existing = pd.read_csv(archive_path, parse_dates=['analysis_date'])
            month_df = pd.concat([archive_existing, month_df], ignore_index=True)
            month_df['analysis_date'] = pd.to_datetime(month_df['analysis_date']).dt.normalize()
            month_df = month_df.sort_values('analysis_date').drop_duplicates('analysis_date', keep='last')

        month_df.to_csv(archive_path, index=False)

    trimmed = df[df['month'].isin(months_to_keep)].drop(columns='month', errors='ignore')
    trimmed = trimmed.sort_values('analysis_date')
    trimmed.to_csv(log_path, index=False)
    return trimmed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="マルチモデル推奨の指標を日次ログへ記録")
    parser.add_argument('--input', type=str, default='production_data/multi_model_candidates.parquet', help='候補データセット (Parquet)')
    parser.add_argument('--config', type=str, default='config/multi_model_recommendation.json', help='設定ファイル')
    parser.add_argument('--output', type=str, default='production_data/multi_model_metrics.csv', help='出力 CSV パス')
    parser.add_argument('--days', type=int, default=30, help='最新何営業日分を出力に反映するか')
    parser.add_argument('--allow-fallback', action='store_true', help='フォールバックを許容する (閾値未達も採用)')
    parser.add_argument('--fallback-max', type=int, default=None, help='1日あたりのフォールバック最大数')
    parser.add_argument('--fallback-min-passed', type=int, default=None, help='フォールバック適用を検討する最低合格件数')
    parser.add_argument('--fallback-min-passed-ratio', type=float, default=None, help='フォールバック採用を検討する合格比率')
    parser.add_argument('--fallback-min-composite', type=float, default=None, help='フォールバック採用を許可する最低統合スコア')
    parser.add_argument('--fallback-min-up-prob', type=float, default=None, help='フォールバック採用を許可する最低上昇確率')
    parser.add_argument('--fallback-risk-margin', type=float, default=None, help='リスク閾値に対するフォールバック許容マージン')
    parser.add_argument('--fallback-block-ratio', type=float, default=None, help='合格銘柄が一定比率に達した場合にフォールバックを停止する比率')
    parser.add_argument('--fallback-max-per-sector', type=int, default=None, help='フォールバック採用をセクターごとに制限する上限')
    parser.add_argument('--target-return', type=float, default=0.01, help='Precision 計算時のヒット閾値')
    parser.add_argument('--archive-dir', type=str, default='production_data/multi_model_metrics_archive', help='月次アーカイブ出力先ディレクトリ')
    parser.add_argument('--keep-months', type=int, default=1, help='最新何ヶ月分のデータを本体CSVに保持するか (0で全アーカイブ)')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = load_candidates(Path(args.input))
    config = load_config(Path(args.config))

    thresholds = config.get('thresholds', {})
    weights = config.get('weights', {})
    fallback_cfg = config.get('fallback', {})
    upside_cfg = config.get('upside', {})

    thresholds = {
        'up': float(thresholds.get('up', 0.44)),
        'down': float(thresholds.get('down', 0.30)),
        'risk': float(thresholds.get('risk', 0.60)),
    }
    weights = {
        'up': float(weights.get('up', 1.0)),
        'down': float(weights.get('down', 0.5)),
        'risk': float(weights.get('risk', 0.3)),
    }

    top_n = int(config.get('top_n', 5))
    max_per_sector = int(upside_cfg.get('max_per_sector', 3))

    if args.allow_fallback:
        require_passed_all = False
    else:
        require_passed_all = bool(fallback_cfg.get('require_passed_all', True))

    fallback_max = args.fallback_max
    if fallback_max is None:
        fallback_max_cfg = fallback_cfg.get('max_fallback')
        if fallback_max_cfg is not None:
            try:
                fallback_max = int(fallback_max_cfg)
            except (TypeError, ValueError):
                fallback_max = None

    fallback_min_passed = args.fallback_min_passed
    if fallback_min_passed is None:
        fallback_min_passed_cfg = fallback_cfg.get('min_passed_all')
        if fallback_min_passed_cfg is not None:
            try:
                fallback_min_passed = int(fallback_min_passed_cfg)
            except (TypeError, ValueError):
                fallback_min_passed = None

    fallback_min_ratio = args.fallback_min_passed_ratio
    if fallback_min_ratio is None:
        fallback_min_ratio_cfg = fallback_cfg.get('min_passed_ratio')
        if fallback_min_ratio_cfg is not None:
            try:
                fallback_min_ratio = float(fallback_min_ratio_cfg)
            except (TypeError, ValueError):
                fallback_min_ratio = None

    fallback_max_per_sector = fallback_cfg.get('max_per_sector')
    if args.fallback_max_per_sector is not None:
        fallback_max_per_sector = args.fallback_max_per_sector
    if fallback_max_per_sector is not None:
        try:
            fallback_max_per_sector = int(fallback_max_per_sector)
        except (TypeError, ValueError):
            fallback_max_per_sector = None

    fallback_min_composite = args.fallback_min_composite
    if fallback_min_composite is None:
        fallback_min_comp_cfg = fallback_cfg.get('min_composite')
        if fallback_min_comp_cfg is not None:
            try:
                fallback_min_composite = float(fallback_min_comp_cfg)
            except (TypeError, ValueError):
                fallback_min_composite = None

    fallback_min_up_prob = args.fallback_min_up_prob
    if fallback_min_up_prob is None:
        fallback_min_up_cfg = fallback_cfg.get('min_up_probability')
        if fallback_min_up_cfg is not None:
            try:
                fallback_min_up_prob = float(fallback_min_up_cfg)
            except (TypeError, ValueError):
                fallback_min_up_prob = None

    fallback_risk_margin = args.fallback_risk_margin
    if fallback_risk_margin is None:
        fallback_risk_margin_cfg = fallback_cfg.get('risk_margin')
        if fallback_risk_margin_cfg is not None:
            try:
                fallback_risk_margin = float(fallback_risk_margin_cfg)
            except (TypeError, ValueError):
                fallback_risk_margin = None

    fallback_block_ratio = args.fallback_block_ratio
    if fallback_block_ratio is None:
        fallback_block_ratio_cfg = fallback_cfg.get('block_ratio')
        if fallback_block_ratio_cfg is not None:
            try:
                fallback_block_ratio = float(fallback_block_ratio_cfg)
            except (TypeError, ValueError):
                fallback_block_ratio = None

    scored = prepare_candidate_scores(candidates, thresholds, weights)
    selections = select_top_candidates(
        scored,
        top_n=top_n,
        max_per_sector=max_per_sector,
        require_passed_all=require_passed_all,
        fallback_max_fallback=fallback_max,
        fallback_min_passed_all=fallback_min_passed,
        fallback_min_passed_ratio=fallback_min_ratio,
        fallback_max_per_sector=fallback_max_per_sector,
        fallback_min_composite=fallback_min_composite,
        fallback_min_up_prob=fallback_min_up_prob,
        fallback_risk_margin=fallback_risk_margin,
        fallback_block_ratio=fallback_block_ratio,
        risk_threshold=thresholds.get('risk'),
    )

    selected_df = pd.DataFrame(selections)
    daily_df = compute_daily_metrics(selected_df, target_return=float(args.target_return))
    daily_df = daily_df.dropna(subset=['analysis_date'])

    if args.days and args.days > 0 and not daily_df.empty:
        unique_dates = sorted(daily_df['analysis_date'].unique())
        if len(unique_dates) > args.days:
            keep = set(unique_dates[-args.days:])
            daily_df = daily_df[daily_df['analysis_date'].isin(keep)]

    combined_df = append_metrics(Path(args.output), daily_df)

    if args.keep_months is not None:
        combined_df = rotate_archives(
            combined_df,
            log_path=Path(args.output),
            archive_dir=Path(args.archive_dir),
            keep_months=int(args.keep_months),
        )

    if not daily_df.empty:
        display_df = daily_df.sort_values('analysis_date')
        with pd.option_context('display.float_format', lambda v: f"{v:.4f}"):
            print(display_df.to_string(index=False))


if __name__ == '__main__':
    main()
