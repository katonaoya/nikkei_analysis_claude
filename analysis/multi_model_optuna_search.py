#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Optuna-based multi-objective tuning for multi-model recommendation thresholds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import optuna
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.daily_stock_recommendation_multi import prepare_candidate_scores, select_top_candidates

DEFAULT_CONFIG = Path("config/multi_model_recommendation.json")


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"候補データが見つかりません: {path}")
    df = pd.read_parquet(path)
    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
    df['code'] = df['code'].astype(str).str.zfill(4)
    return df


def evaluate_combination(
    candidates: pd.DataFrame,
    thresholds: Dict[str, float],
    fallback_params: Dict[str, float],
    *,
    top_n: int,
    max_per_sector: int,
    require_passed_all: bool,
    target_return: float,
    transaction_cost: float,
) -> Dict[str, float]:
    scored = prepare_candidate_scores(candidates, thresholds, {'up':1.0,'down':0.5,'risk':0.4})
    selected = select_top_candidates(
        scored,
        top_n=top_n,
        max_per_sector=max_per_sector,
        require_passed_all=require_passed_all,
        fallback_max_fallback=1,
        fallback_min_passed_all=2,
        fallback_min_passed_ratio=fallback_params['min_ratio'],
        fallback_max_per_sector=1,
        fallback_min_composite=fallback_params['min_composite'],
        fallback_min_up_prob=fallback_params['min_up_prob'],
        fallback_risk_margin=fallback_params['risk_margin'],
        fallback_block_ratio=fallback_params['block_ratio'],
        risk_threshold=thresholds['risk'],
    )
    if not selected:
        return {
            'precision': 0.0,
            'average_return': 0.0,
            'coverage': 0.0,
            'fallback_ratio': 1.0,
            'selected_total': 0,
        }
    df_sel = pd.DataFrame(selected)
    df_sel['analysis_date'] = pd.to_datetime(df_sel['analysis_date']).dt.normalize()
    valid = df_sel.dropna(subset=['future_return'])
    precision = valid['future_return'].ge(target_return).mean() if not valid.empty else 0.0
    average_return = (valid['future_return'] - transaction_cost).mean() if not valid.empty else -transaction_cost
    coverage = df_sel['analysis_date'].nunique() / candidates['analysis_date'].nunique()
    fallback_ratio = (~df_sel['passed_all_filters']).mean()
    return {
        'precision': float(precision),
        'average_return': float(average_return),
        'coverage': float(coverage),
        'fallback_ratio': float(fallback_ratio),
        'selected_total': int(len(df_sel)),
    }


def parse_weights(value: Optional[str]) -> Dict[str, float]:
    default = {'precision': 0.5, 'coverage': 0.3, 'fallback': 0.2}
    if not value:
        return default
    weights: Dict[str, float] = {}
    for chunk in value.split(','):
        name, weight_str = chunk.split(':', 1)
        weights[name.strip()] = float(weight_str)
    total = sum(abs(v) for v in weights.values())
    if total == 0:
        raise ValueError('weights sum to zero')
    return {k: v / total for k, v in weights.items()}


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Optuna でマルチモデル推奨の閾値を探索')
    parser.add_argument('--input', type=str, default='production_data/multi_model_candidates.parquet', help='候補データ (parquet)')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG), help='既存設定ファイル')
    parser.add_argument('--trials', type=int, default=50, help='Optuna トライアル回数')
    parser.add_argument('--weights', type=str, default=None, help="評価重み (例: 'precision:0.6,coverage:0.3,fallback:0.1')")
    parser.add_argument('--penalty', type=float, default=0.2, help='制約違反時に減点する値')
    parser.add_argument('--target-precision', type=float, default=0.45, help='Precision の目標値')
    parser.add_argument('--target-coverage', type=float, default=0.60, help='Coverage の目標値')
    parser.add_argument('--max-fallback', type=float, default=0.40, help='フォールバック比率の上限')
    parser.add_argument('--output', type=str, default='analysis/multi_model_optuna_trials.csv', help='結果CSV')
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_argument_parser()
    args = parser.parse_args(argv)

    candidates = load_candidates(Path(args.input))
    config = json.loads(Path(args.config).read_text()) if Path(args.config).exists() else {}
    fallback_cfg = config.get('fallback', {})
    upside_cfg = config.get('upside', {})

    weights_cfg = parse_weights(args.weights)
    target_return = float(config.get('upside', {}).get('target_return', 0.01))
    transaction_cost = float(config.get('upside', {}).get('transaction_cost', 0.0))

    top_n = int(config.get('top_n', 5))
    max_per_sector = int(upside_cfg.get('max_per_sector', 2))
    require_passed_all = bool(fallback_cfg.get('require_passed_all', True))

    trials_records = []

    def objective(trial: optuna.Trial) -> float:
        thresholds = {
            'up': trial.suggest_float('threshold_up', 0.18, 0.22),
            'down': trial.suggest_float('threshold_down', 0.48, 0.52),
            'risk': trial.suggest_float('threshold_risk', 0.38, 0.45),
        }
        fallback_params = {
            'min_ratio': trial.suggest_float('fallback_min_passed_ratio', 0.40, 0.65),
            'min_composite': trial.suggest_float('fallback_min_composite', -0.05, 0.0),
            'min_up_prob': trial.suggest_float('fallback_min_up_prob', 0.14, 0.20),
            'risk_margin': trial.suggest_float('fallback_risk_margin', 0.02, 0.06),
            'block_ratio': trial.suggest_float('fallback_block_ratio', 0.15, 0.35),
        }

        metrics = evaluate_combination(
            candidates,
            thresholds,
            fallback_params,
            top_n=top_n,
            max_per_sector=max_per_sector,
            require_passed_all=require_passed_all,
            target_return=target_return,
            transaction_cost=transaction_cost,
        )

        penalty = 0.0
        if metrics['precision'] < args.target_precision:
            penalty += args.penalty
        if metrics['coverage'] < args.target_coverage:
            penalty += args.penalty
        if metrics['fallback_ratio'] > args.max_fallback:
            penalty += args.penalty

        score = (
            weights_cfg.get('precision', 0.0) * metrics['precision']
            + weights_cfg.get('coverage', 0.0) * metrics['coverage']
            - weights_cfg.get('fallback', 0.0) * metrics['fallback_ratio']
        ) - penalty

        trial.set_user_attr('metrics', metrics)
        trial.set_user_attr('thresholds', thresholds)
        trial.set_user_attr('fallback_params', fallback_params)

        return -score  # minimize negative score

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=args.trials)

    for trial in study.trials:
        row = {
            'trial': trial.number,
            'score': -trial.value,
        }
        row.update(trial.user_attrs.get('thresholds', {}))
        row.update({
            'fallback_min_passed_ratio': trial.user_attrs.get('fallback_params', {}).get('min_ratio'),
            'fallback_min_composite': trial.user_attrs.get('fallback_params', {}).get('min_composite'),
            'fallback_min_up_prob': trial.user_attrs.get('fallback_params', {}).get('min_up_prob'),
            'fallback_risk_margin': trial.user_attrs.get('fallback_params', {}).get('risk_margin'),
            'fallback_block_ratio': trial.user_attrs.get('fallback_params', {}).get('block_ratio'),
        })
        row.update(trial.user_attrs.get('metrics', {}))
        trials_records.append(row)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trials_records).to_csv(output_path, index=False)
    best = study.best_trial
    print(f"✅ Best score: {-best.value:.4f}")
    print(f"   Thresholds: {best.user_attrs['thresholds']}")
    print(f"   Fallback params: {best.user_attrs['fallback_params']}")
    print(f"   Metrics: {best.user_attrs['metrics']}")


if __name__ == '__main__':
    main()
