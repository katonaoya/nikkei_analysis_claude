#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""マルチモデル統合スコアの閾値探索ツール"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.daily_stock_recommendation_multi import (
    prepare_candidate_scores,
    select_top_candidates,
)

DEFAULT_CONFIG_PATH = Path("config/multi_model_recommendation.json")
REQUIRED_COLUMNS = {
    "analysis_date",
    "code",
    "prediction_probability",
    "downside_probability",
    "risk_score",
    "future_return",
}


def parse_float_list(value: str) -> List[float]:
    numbers = []
    for chunk in value.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        numbers.append(float(chunk))
    return numbers


def parse_metric_weights(value: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for chunk in value.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ':' not in chunk:
            raise ValueError("--metric-weights は 'metric:weight' 形式で指定してください")
        metric, weight_str = chunk.split(':', 1)
        metric = metric.strip()
        if not metric:
            raise ValueError("メトリクス名が空です")
        weights[metric] = float(weight_str)
    if not weights:
        raise ValueError("有効なメトリクス重みが指定されていません")
    total = sum(abs(v) for v in weights.values())
    if not total:
        raise ValueError("メトリクス重みの合計が0です")
    return {k: v / total for k, v in weights.items()}


def parse_optional_float_list(value: str) -> List[Optional[float]]:
    numbers: List[Optional[float]] = []
    for chunk in value.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if chunk.lower() in {'none', 'null'}:
            numbers.append(None)
        else:
            numbers.append(float(chunk))
    return numbers


def load_candidates(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"候補データが見つかりません: {path}")

    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        sep = '\t' if suffix == ".tsv" else ','
        df = pd.read_csv(path, sep=sep)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in {".pkl", ".pickle", ".joblib"}:
        import joblib

        loaded = joblib.load(path)
        if isinstance(loaded, pd.DataFrame):
            df = loaded
        else:
            df = pd.DataFrame(loaded)
    else:
        raise ValueError(f"未対応のファイル形式です: {path.suffix}")

    if 'analysis_date' not in df:
        raise KeyError("analysis_date 列が必要です")
    if 'code' not in df:
        raise KeyError("code 列が必要です")

    df = df.copy()
    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
    df['code'] = df['code'].astype(str).str.zfill(4)

    for missing in REQUIRED_COLUMNS - set(df.columns):
        if missing == 'downside_probability' or missing == 'risk_score':
            df[missing] = 0.0
        else:
            raise KeyError(f"候補データに必要な列が不足しています: {missing}")

    return df


def evaluate_combination(
    scored_df: pd.DataFrame,
    *,
    top_n: int,
    max_per_sector: int,
    require_passed_all: bool,
    target_return: float,
    transaction_cost: float,
    fallback_max_fallback: Optional[int],
    fallback_min_passed_all: Optional[int],
    fallback_min_passed_ratio: Optional[float],
    fallback_max_per_sector: Optional[int],
    fallback_min_composite: Optional[float],
    fallback_min_up_prob: Optional[float],
    fallback_risk_margin: Optional[float],
    fallback_block_ratio: Optional[float],
    risk_threshold: Optional[float],
) -> Dict[str, float]:
    selected = select_top_candidates(
        scored_df,
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
        risk_threshold=risk_threshold,
    )

    total_selected = len(selected)
    if total_selected == 0:
        return {
            'selected_count': 0,
            'valid_count': 0,
            'precision': 0.0,
            'avg_return': 0.0,
            'avg_net_return': 0.0,
            'median_return': 0.0,
            'hit_count': 0,
            'fallback_count': 0,
            'days_with_signals': 0,
            'coverage_rate': 0.0,
        }

    returns = [rec['future_return'] for rec in selected if not np.isnan(rec.get('future_return', np.nan))]
    fallback_count = sum(not rec.get('passed_all_filters', False) for rec in selected)
    hit_count = sum(ret >= target_return for ret in returns)
    valid_count = len(returns)

    if returns:
        avg_return = float(np.mean(returns))
        median_return = float(np.median(returns))
        avg_net_return = float(np.mean([ret - transaction_cost for ret in returns]))
        precision = float(hit_count / valid_count) if valid_count else 0.0
    else:
        avg_return = median_return = avg_net_return = 0.0
        precision = 0.0

    days_with_signals = len({rec['analysis_date'] for rec in selected if rec.get('analysis_date') is not pd.NaT})

    return {
        'selected_count': total_selected,
        'valid_count': valid_count,
        'precision': precision,
        'avg_return': avg_return,
        'avg_net_return': avg_net_return,
        'median_return': median_return,
        'hit_count': hit_count,
        'fallback_count': fallback_count,
        'days_with_signals': days_with_signals,
    }


def evaluate_threshold_grid(
    candidates_df: pd.DataFrame,
    *,
    up_grid: Sequence[float],
    down_grid: Sequence[float],
    risk_grid: Sequence[float],
    weights: Dict[str, float],
    top_n: int,
    max_per_sector: int,
    require_passed_all: bool,
    target_return: float,
    transaction_cost: float,
    metric: str,
    min_valid_count: int,
    metric_weights: Optional[Dict[str, float]] = None,
    fallback_max_fallback: Optional[int] = None,
    fallback_min_passed_all: Optional[int] = None,
    fallback_min_passed_ratio_values: Optional[Sequence[Optional[float]]] = None,
    fallback_max_per_sector: Optional[int] = None,
    fallback_min_composite: Optional[float] = None,
    fallback_min_up_prob: Optional[float] = None,
    fallback_risk_margin: Optional[float] = None,
    fallback_block_ratio: Optional[float] = None,
) -> pd.DataFrame:
    total_days = candidates_df['analysis_date'].nunique()
    results: List[Dict[str, float]] = []

    ratio_values = list(fallback_min_passed_ratio_values) if fallback_min_passed_ratio_values else [None]

    for th_up in up_grid:
        for th_down in down_grid:
            for th_risk in risk_grid:
                thresholds = {'up': th_up, 'down': th_down, 'risk': th_risk}
                scored_df = prepare_candidate_scores(candidates_df, thresholds, weights)
                for ratio in ratio_values:
                    metrics = evaluate_combination(
                        scored_df,
                        top_n=top_n,
                        max_per_sector=max_per_sector,
                        require_passed_all=require_passed_all,
                        target_return=target_return,
                        transaction_cost=transaction_cost,
                        fallback_max_fallback=fallback_max_fallback,
                        fallback_min_passed_all=fallback_min_passed_all,
                        fallback_min_passed_ratio=ratio,
                        fallback_max_per_sector=fallback_max_per_sector,
                        fallback_min_composite=fallback_min_composite,
                        fallback_min_up_prob=fallback_min_up_prob,
                        fallback_risk_margin=fallback_risk_margin,
                        fallback_block_ratio=fallback_block_ratio,
                        risk_threshold=thresholds['risk'],
                    )
                    if metrics['valid_count'] < min_valid_count:
                        continue
                    coverage_rate = metrics['days_with_signals'] / total_days if total_days else 0.0
                    metrics.update({
                        'threshold_up': th_up,
                        'threshold_down': th_down,
                        'threshold_risk': th_risk,
                        'weight_up': weights['up'],
                        'weight_down': weights['down'],
                        'weight_risk': weights['risk'],
                        'coverage_rate': coverage_rate,
                        'total_days': total_days,
                        'fallback_min_passed_ratio': ratio,
                        'fallback_max_per_sector': fallback_max_per_sector,
                        'fallback_min_composite': fallback_min_composite,
                        'fallback_min_up_prob': fallback_min_up_prob,
                        'fallback_risk_margin': fallback_risk_margin,
                        'fallback_block_ratio': fallback_block_ratio,
                    })
                    results.append(metrics)

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    if metric_weights:
        df['weighted_score'] = 0.0
        for key, weight in metric_weights.items():
            if key not in df.columns:
                raise KeyError(f"メトリクス '{key}' は結果に含まれていません")
            df['weighted_score'] += df[key] * weight
        df.sort_values(['weighted_score', 'precision', 'avg_return'], ascending=False, inplace=True)
    else:
        sort_maps = {
            'precision': ['precision', 'avg_net_return', 'avg_return'],
            'avg_return': ['avg_return', 'precision', 'avg_net_return'],
            'avg_net_return': ['avg_net_return', 'precision', 'avg_return'],
        }
        sort_cols = sort_maps.get(metric, ['precision'])
        df.sort_values(sort_cols, ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def update_config(
    config_path: Path,
    *,
    threshold_up: float,
    threshold_down: float,
    threshold_risk: float,
    weight_up: float,
    weight_down: float,
    weight_risk: float,
) -> None:
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:
        config = {}

    thresholds = config.get('thresholds', {})
    thresholds.update({
        'up': threshold_up,
        'down': threshold_down,
        'risk': threshold_risk,
    })
    config['thresholds'] = thresholds

    weights_cfg = config.get('weights', {})
    weights_cfg.update({
        'up': weight_up,
        'down': weight_down,
        'risk': weight_risk,
    })
    config['weights'] = weights_cfg

    config_path.write_text(json.dumps(config, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="マルチモデル閾値のグリッドサーチ")
    parser.add_argument('--input', type=str, default='production_data/multi_model_candidates.parquet', help='候補データ（CSV/Parquet/JSON/Joblib）')
    parser.add_argument('--config', type=str, default=str(DEFAULT_CONFIG_PATH), help='設定ファイルパス')
    parser.add_argument('--threshold-up-grid', type=str, default='', help='上昇確率の評価グリッド (例: 0.40,0.44,0.48)')
    parser.add_argument('--threshold-down-grid', type=str, default='', help='下落確率の評価グリッド (例: 0.25,0.30)')
    parser.add_argument('--risk-grid', type=str, default='', help='リスクスコアの評価グリッド (例: 0.50,0.60)')
    parser.add_argument('--weight-up', type=float, default=None, help='上昇確率の重み (指定なしで設定ファイルを使用)')
    parser.add_argument('--weight-down', type=float, default=None, help='下落確率の重み (指定なしで設定ファイルを使用)')
    parser.add_argument('--weight-risk', type=float, default=None, help='リスクスコアの重み (指定なしで設定ファイルを使用)')
    parser.add_argument('--top-n', type=int, default=None, help='日次レポートの上位件数')
    parser.add_argument('--max-per-sector', type=int, default=None, help='セクター上限')
    parser.add_argument('--target-return', type=float, default=0.01, help='ヒット判定となる翌日リターン閾値')
    parser.add_argument('--transaction-cost', type=float, default=0.0, help='片道取引コスト率')
    parser.add_argument('--metric', type=str, default='precision', choices=['precision', 'avg_return', 'avg_net_return'], help='上位選定に用いるメトリクス')
    parser.add_argument('--min-valid-count', type=int, default=10, help='有効データ件数の最小値 (未満は除外)')
    parser.add_argument('--allow-fallback', action='store_true', help='閾値未達の候補でも不足分を補完する')
    parser.add_argument('--fallback-max', type=int, default=None, help='1日あたり許容するフォールバック枠数 (未指定で制限なし)')
    parser.add_argument('--fallback-min-passed', type=int, default=None, help='フォールバック適用を検討する最低合格件数 (未指定で制限なし)')
    parser.add_argument('--fallback-min-passed-ratio', type=float, default=None, help='フォールバックを検討する合格比率の上限 (例: 0.4)')
    parser.add_argument('--fallback-min-passed-ratio-grid', type=str, default=None, help="フォールバック比率をグリッド探索する場合に指定 (例: '0.3,0.4,none')")
    parser.add_argument('--fallback-max-per-sector', type=int, default=None, help='フォールバック採用をセクターごとに制限する上限')
    parser.add_argument('--fallback-min-composite', type=float, default=None, help='フォールバック採用を許可する最低統合スコア')
    parser.add_argument('--fallback-min-up-prob', type=float, default=None, help='フォールバック採用を許可する最低上昇確率')
    parser.add_argument('--fallback-risk-margin', type=float, default=None, help='リスク閾値に対するフォールバック許容マージン')
    parser.add_argument('--fallback-block-ratio', type=float, default=None, help='合格銘柄が一定比率に達した場合にフォールバックを停止する比率')
    parser.add_argument('--metric-weights', type=str, default=None, help="結果ソートに用いるメトリクス重み (例: 'precision:0.6,avg_net_return:0.3,coverage_rate:0.1')")
    parser.add_argument('--target-precision', type=float, default=None, help='目標とする Precision (差分列を出力)')
    parser.add_argument('--target-coverage', type=float, default=None, help='目標とする Coverage (差分列を出力)')
    parser.add_argument('--export-csv', type=str, default=None, help='結果をCSV出力するパス')
    parser.add_argument('--update-config', action='store_true', help='ベスト閾値で設定ファイルを更新する')
    parser.add_argument('--top-k', type=int, default=10, help='標準出力に表示する上位件数')
    args = parser.parse_args()

    config_path = Path(args.config)
    if config_path.exists():
        config_data = json.loads(config_path.read_text())
    else:
        config_data = {}

    thresholds_cfg = config_data.get('thresholds', {})
    weights_cfg = config_data.get('weights', {})
    upside_cfg = config_data.get('upside', {})
    fallback_cfg = config_data.get('fallback', {})

    default_up = thresholds_cfg.get('up', 0.44)
    default_down = thresholds_cfg.get('down', 0.30)
    default_risk = thresholds_cfg.get('risk', 0.60)

    up_grid = parse_float_list(args.threshold_up_grid) if args.threshold_up_grid else [default_up]
    down_grid = parse_float_list(args.threshold_down_grid) if args.threshold_down_grid else [default_down]
    risk_grid = parse_float_list(args.risk_grid) if args.risk_grid else [default_risk]

    weights = {
        'up': args.weight_up if args.weight_up is not None else float(weights_cfg.get('up', 1.0)),
        'down': args.weight_down if args.weight_down is not None else float(weights_cfg.get('down', 0.6)),
        'risk': args.weight_risk if args.weight_risk is not None else float(weights_cfg.get('risk', 0.4)),
    }

    top_n = args.top_n if args.top_n is not None else int(config_data.get('top_n', 5))
    max_per_sector = args.max_per_sector if args.max_per_sector is not None else int(upside_cfg.get('max_per_sector', 3))
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
        fallback_min_cfg = fallback_cfg.get('min_passed_all')
        if fallback_min_cfg is not None:
            try:
                fallback_min_passed = int(fallback_min_cfg)
            except (TypeError, ValueError):
                fallback_min_passed = None

    fallback_ratio_values: List[Optional[float]]
    if args.fallback_min_passed_ratio_grid:
        fallback_ratio_values = parse_optional_float_list(args.fallback_min_passed_ratio_grid)
    else:
        fallback_min_ratio = args.fallback_min_passed_ratio
        if fallback_min_ratio is None:
            fallback_ratio_cfg = fallback_cfg.get('min_passed_ratio')
            if fallback_ratio_cfg is not None:
                try:
                    fallback_min_ratio = float(fallback_ratio_cfg)
                except (TypeError, ValueError):
                    fallback_min_ratio = None
        fallback_ratio_values = [fallback_min_ratio]

    fallback_max_per_sector = args.fallback_max_per_sector
    if fallback_max_per_sector is None:
        fallback_max_per_sector_cfg = fallback_cfg.get('max_per_sector')
        if fallback_max_per_sector_cfg is not None:
            try:
                fallback_max_per_sector = int(fallback_max_per_sector_cfg)
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

    candidates_path = Path(args.input)
    candidates_df = load_candidates(candidates_path)

    metric_weights = parse_metric_weights(args.metric_weights) if args.metric_weights else None

    results = evaluate_threshold_grid(
        candidates_df,
        up_grid=up_grid,
        down_grid=down_grid,
        risk_grid=risk_grid,
        weights=weights,
        top_n=top_n,
        max_per_sector=max_per_sector,
        require_passed_all=require_passed_all,
        target_return=args.target_return,
        transaction_cost=args.transaction_cost,
        metric=args.metric,
        min_valid_count=args.min_valid_count,
        metric_weights=metric_weights,
        fallback_max_fallback=fallback_max,
        fallback_min_passed_all=fallback_min_passed,
        fallback_min_passed_ratio_values=fallback_ratio_values,
        fallback_max_per_sector=fallback_max_per_sector,
        fallback_min_composite=fallback_min_composite,
        fallback_min_up_prob=fallback_min_up_prob,
        fallback_risk_margin=fallback_risk_margin,
        fallback_block_ratio=fallback_block_ratio,
    )

    if results.empty:
        print("⚠️ 評価対象がありませんでした。閾値グリッドやデータ件数を見直してください。")
        return

    if args.target_precision is not None:
        results['precision_gap'] = results['precision'] - float(args.target_precision)
    if args.target_coverage is not None:
        results['coverage_gap'] = results['coverage_rate'] - float(args.target_coverage)

    if 'fallback_min_passed_ratio' in results.columns:
        results['fallback_min_passed_ratio'] = results['fallback_min_passed_ratio'].astype(float)

    top_k = max(args.top_k, 1)
    display_df = results.head(top_k)
    with pd.option_context('display.max_columns', None, 'display.width', 160):
        print(display_df.to_string(index=False, formatters={
            'precision': lambda v: f"{v*100:.2f}%",
            'avg_return': lambda v: f"{v*100:.2f}%",
            'avg_net_return': lambda v: f"{v*100:.2f}%",
            'median_return': lambda v: f"{v*100:.2f}%",
            'coverage_rate': lambda v: f"{v*100:.2f}%",
        }))

    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(export_path, index=False)
        print(f"📄 評価結果を出力しました: {export_path}")

    if args.update_config:
        best = results.iloc[0]
        update_config(
            config_path,
            threshold_up=float(best['threshold_up']),
            threshold_down=float(best['threshold_down']),
            threshold_risk=float(best['threshold_risk']),
            weight_up=float(best['weight_up']),
            weight_down=float(best['weight_down']),
            weight_risk=float(best['weight_risk']),
        )
        print(f"🛠 設定ファイルを更新しました: {config_path}")


if __name__ == '__main__':
    main()
