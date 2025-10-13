#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Precision/coverage report for multi-model candidate datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.daily_stock_recommendation_multi import prepare_candidate_scores, select_top_candidates


DEFAULT_CONFIG = Path("config/multi_model_recommendation.json")


def load_config(path: Path) -> Dict[str, Dict[str, float]]:
    if not path.exists():
        raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
    with path.open() as f:
        return json.load(f)


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"候補データが見つかりません: {path}")
    return pd.read_parquet(path)


def compute_metrics(
    candidates: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
    top_n: int,
    max_per_sector: int,
    require_passed_all: bool,
    fallback_max_fallback: Optional[int],
    fallback_min_passed_all: Optional[int],
    fallback_min_passed_ratio: Optional[float],
    fallback_max_per_sector: Optional[int],
    fallback_min_composite: Optional[float],
    fallback_min_up_prob: Optional[float],
    fallback_risk_margin: Optional[float],
    fallback_block_ratio: Optional[float],
) -> pd.DataFrame:
    scored = prepare_candidate_scores(candidates, thresholds, weights)
    selected = select_top_candidates(
        scored,
        top_n=top_n,
        max_per_sector=max_per_sector,
        require_passed_all=require_passed_all,
        fallback_max_fallback=fallback_max_fallback,
        fallback_min_passed_all=fallback_min_passed_all,
        fallback_min_passed_ratio=fallback_min_passed_ratio,
        fallback_max_per_sector=fallback_max_per_sector,
        fallback_min_composite=fallback_min_composite,
        fallback_min_up_prob=fallback_min_up_prob,
        fallback_risk_margin=fallback_risk_margin,
        fallback_block_ratio=fallback_block_ratio,
        risk_threshold=thresholds.get('risk'),
    )

    selected_df = pd.DataFrame(selected)
    if selected_df.empty:
        return pd.DataFrame()

    selected_df['analysis_date'] = pd.to_datetime(selected_df['analysis_date']).dt.normalize()
    selected_df['future_return'] = selected_df['future_return'].astype(float)
    return selected_df


def summarise(selected: pd.DataFrame, total_days: int) -> Dict[str, float]:
    if selected.empty:
        return {
            'selected_days': 0,
            'selected_total': 0,
            'valid_total': 0,
            'precision': float('nan'),
            'avg_return': float('nan'),
            'coverage_rate': 0.0,
        }

    valid = selected.dropna(subset=['future_return'])
    hits = (valid['future_return'] >= 0.01).sum()
    precision = hits / len(valid) if len(valid) else float('nan')
    avg_return = valid['future_return'].mean() if len(valid) else float('nan')
    selected_days = selected['analysis_date'].nunique()
    coverage = selected_days / total_days if total_days else 0.0

    return {
        'selected_days': selected_days,
        'selected_total': len(selected),
        'valid_total': len(valid),
        'precision': precision,
        'avg_return': avg_return,
        'coverage_rate': coverage,
    }


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="マルチモデル候補データの精度サマリを出力")
    parser.add_argument('--input', type=str, default='production_data/multi_model_candidates.parquet', help='候補データのパス')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG), help='設定ファイルのパス')
    parser.add_argument('--top-n', type=int, default=None, help='選抜件数 (未指定時は設定ファイルを使用)')
    parser.add_argument('--max-per-sector', type=int, default=None, help='セクター上限 (未指定時は設定ファイルを使用)')
    parser.add_argument('--allow-fallback', action='store_true', help='閾値未達候補も補完するか')
    parser.add_argument('--fallback-max', type=int, default=None, help='1日あたり許容するフォールバック枠数')
    parser.add_argument('--fallback-min-passed', type=int, default=None, help='フォールバック適用を検討する最低合格件数')
    parser.add_argument('--fallback-min-passed-ratio', type=float, default=None, help='フォールバックを検討する合格比率の上限 (例: 0.4)')
    parser.add_argument('--fallback-max-per-sector', type=int, default=None, help='フォールバック採用をセクターごとに制限する上限')
    parser.add_argument('--fallback-min-composite', type=float, default=None, help='フォールバック採用を許可する最低統合スコア')
    parser.add_argument('--fallback-min-up-prob', type=float, default=None, help='フォールバック採用を許可する最低上昇確率')
    parser.add_argument('--fallback-risk-margin', type=float, default=None, help='リスク閾値に対するフォールバック許容マージン')
    parser.add_argument('--fallback-block-ratio', type=float, default=None, help='合格銘柄が一定比率に達した場合にフォールバックを禁止する比率')
    parser.add_argument('--threshold-up', type=float, help='上昇閾値を上書き')
    parser.add_argument('--threshold-down', type=float, help='下落閾値を上書き')
    parser.add_argument('--risk-threshold', type=float, help='リスク閾値を上書き')
    parser.add_argument('--export-csv', type=str, help='選抜結果を出力するCSVパス')
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> None:
    args = parse_args(argv)

    config = load_config(Path(args.config))
    thresholds = dict(config.get('thresholds', {}))
    weights = dict(config.get('weights', {}))
    if not thresholds or not weights:
        raise ValueError('設定ファイルに thresholds / weights が含まれていません')

    if args.threshold_up is not None:
        thresholds['up'] = float(args.threshold_up)
    if args.threshold_down is not None:
        thresholds['down'] = float(args.threshold_down)
    if args.risk_threshold is not None:
        thresholds['risk'] = float(args.risk_threshold)

    top_n = args.top_n if args.top_n is not None else int(config.get('top_n', 5))
    max_per_sector = args.max_per_sector if args.max_per_sector is not None else int(config.get('upside', {}).get('max_per_sector', 3))
    require_passed_all = not args.allow_fallback and bool(config.get('fallback', {}).get('require_passed_all', True))
    fallback_cfg = config.get('fallback', {})
    fallback_max = args.fallback_max if args.fallback_max is not None else fallback_cfg.get('max_fallback')
    if fallback_max is not None:
        try:
            fallback_max = int(fallback_max)
        except (TypeError, ValueError):
            fallback_max = None
    fallback_min_passed = args.fallback_min_passed if args.fallback_min_passed is not None else fallback_cfg.get('min_passed_all')
    if fallback_min_passed is not None:
        try:
            fallback_min_passed = int(fallback_min_passed)
        except (TypeError, ValueError):
            fallback_min_passed = None
    fallback_min_ratio = args.fallback_min_passed_ratio if args.fallback_min_passed_ratio is not None else fallback_cfg.get('min_passed_ratio')
    if fallback_min_ratio is not None:
        try:
            fallback_min_ratio = float(fallback_min_ratio)
        except (TypeError, ValueError):
            fallback_min_ratio = None
    fallback_max_per_sector = args.fallback_max_per_sector
    if fallback_max_per_sector is None:
        fallback_max_per_sector = fallback_cfg.get('max_per_sector')
    if fallback_max_per_sector is not None:
        try:
            fallback_max_per_sector = int(fallback_max_per_sector)
        except (TypeError, ValueError):
            fallback_max_per_sector = None

    fallback_min_composite = args.fallback_min_composite if args.fallback_min_composite is not None else fallback_cfg.get('min_composite')
    if fallback_min_composite is not None:
        try:
            fallback_min_composite = float(fallback_min_composite)
        except (TypeError, ValueError):
            fallback_min_composite = None

    fallback_min_up_prob = args.fallback_min_up_prob if args.fallback_min_up_prob is not None else fallback_cfg.get('min_up_probability')
    if fallback_min_up_prob is not None:
        try:
            fallback_min_up_prob = float(fallback_min_up_prob)
        except (TypeError, ValueError):
            fallback_min_up_prob = None

    fallback_risk_margin = args.fallback_risk_margin if args.fallback_risk_margin is not None else fallback_cfg.get('risk_margin')
    if fallback_risk_margin is not None:
        try:
            fallback_risk_margin = float(fallback_risk_margin)
        except (TypeError, ValueError):
            fallback_risk_margin = None

    fallback_block_ratio = args.fallback_block_ratio if args.fallback_block_ratio is not None else fallback_cfg.get('block_ratio')
    if fallback_block_ratio is not None:
        try:
            fallback_block_ratio = float(fallback_block_ratio)
        except (TypeError, ValueError):
            fallback_block_ratio = None

    candidates = load_candidates(Path(args.input))
    print(f"📥 候補データ読み込み: {len(candidates)} 件 / {candidates['analysis_date'].nunique()} 営業日")

    selected = compute_metrics(
        candidates,
        thresholds=thresholds,
        weights=weights,
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
    )

    summary = summarise(selected, total_days=candidates['analysis_date'].nunique())
    print("\n📊 指標サマリ")
    print(f"  採用営業日数: {summary['selected_days']}")
    print(f"  採用銘柄数: {summary['selected_total']}")
    print(f"  Precision: {summary['precision']:.2%}" if pd.notna(summary['precision']) else "  Precision: N/A")
    print(f"  平均リターン: {summary['avg_return']:.2%}" if pd.notna(summary['avg_return']) else "  平均リターン: N/A")
    print(f"  Coverage: {summary['coverage_rate']:.2%}")

    if args.export_csv and not selected.empty:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        selected.to_csv(export_path, index=False)
        print(f"📄 選抜データを出力しました: {export_path}")


if __name__ == '__main__':
    main()
