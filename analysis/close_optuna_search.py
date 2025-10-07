#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""終値ベースモデルのOptunaハイパーパラメータ探索"""

import argparse
import json
from datetime import datetime
from pathlib import Path
import sys

import optuna

sys.path.append(str(Path(__file__).resolve().parents[1]))
from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1
from analysis.close_param_utils import prepare_dataset, evaluate_params


DEFAULT_SEARCH_SPACE = {
    "learning_rate": (0.01, 0.1, True),
    "max_depth": (4, 10),
    "num_leaves": (16, 128),
    "min_child_samples": (10, 60),
    "subsample": (0.6, 1.0),
    "colsample_bytree": (0.6, 1.0),
    "n_estimators": (150, 500),
    "reg_alpha": (0.0, 0.5),
    "reg_lambda": (0.0, 0.5),
}

OUTPUT_PATH = Path("config/close_model_params.json")


def build_trial_params(trial: optuna.Trial) -> dict:
    params = {
        "learning_rate": trial.suggest_float("learning_rate", *DEFAULT_SEARCH_SPACE["learning_rate"][:2], log=DEFAULT_SEARCH_SPACE["learning_rate"][2]),
        "max_depth": trial.suggest_int("max_depth", *DEFAULT_SEARCH_SPACE["max_depth"]),
        "num_leaves": trial.suggest_int("num_leaves", *DEFAULT_SEARCH_SPACE["num_leaves"]),
        "min_child_samples": trial.suggest_int("min_child_samples", *DEFAULT_SEARCH_SPACE["min_child_samples"]),
        "subsample": trial.suggest_float("subsample", *DEFAULT_SEARCH_SPACE["subsample"]),
        "colsample_bytree": trial.suggest_float("colsample_bytree", *DEFAULT_SEARCH_SPACE["colsample_bytree"]),
        "n_estimators": trial.suggest_int("n_estimators", *DEFAULT_SEARCH_SPACE["n_estimators"]),
        "reg_alpha": trial.suggest_float("reg_alpha", *DEFAULT_SEARCH_SPACE["reg_alpha"]),
        "reg_lambda": trial.suggest_float("reg_lambda", *DEFAULT_SEARCH_SPACE["reg_lambda"]),
    }
    return params


def main():
    parser = argparse.ArgumentParser(description="Optuna を用いたハイパーパラメータ探索")
    parser.add_argument('--trials', type=int, default=30, help='試行回数')
    parser.add_argument('--timeout', type=int, default=None, help='探索タイムアウト（秒）')
    parser.add_argument('--study-name', type=str, default='close_optuna_search', help='Optunaスタディ名')
    parser.add_argument('--storage', type=str, default=None, help='OptunaストレージURI (省略時はインメモリ)')
    parser.add_argument('--imbalance-boost', type=float, default=1.0, help='scale_pos_weight に掛ける倍率')
    parser.add_argument('--imbalance-strategy', type=str, default='scale_pos', choices=['scale_pos', 'balanced', 'focal', 'none'], help='追加のサンプル重み戦略')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='focal戦略用ガンマ値')
    parser.add_argument('--positive-oversample-ratio', type=float, default=1.0, help='正例の単純オーバーサンプリング倍率 (>1で増幅)')
    parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'precision', 'precision_topn'], help='評価指標')
    parser.add_argument('--top-n', type=int, default=5, help='precision_topn 時の上位N')
    args = parser.parse_args()

    system = CloseReturnPrecisionSystemV1(
        imbalance_boost=args.imbalance_boost,
        imbalance_strategy=args.imbalance_strategy,
        focal_gamma=args.focal_gamma,
        positive_oversample_ratio=args.positive_oversample_ratio,
    )
    dataset = prepare_dataset(system)

    def objective(trial: optuna.Trial) -> float:
        params = build_trial_params(trial)
        score = evaluate_params(system, dataset, params, metric=args.metric, top_n=args.top_n)
        trial.set_user_attr('params', params)
        return score

    study = optuna.create_study(
        study_name=args.study_name,
        storage=args.storage,
        direction='maximize',
        load_if_exists=bool(args.storage)
    )
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    best_params = study.best_trial.user_attrs['params']
    OUTPUT_PATH.write_text(json.dumps(best_params, indent=2))

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_path = Path('analysis') / f'close_optuna_history_{timestamp}.csv'
    history = []
    for trial in study.trials:
        row = {
            'number': trial.number,
            'value': trial.value,
            'state': str(trial.state),
        }
        row.update({f"param_{k}": v for k, v in trial.params.items()})
        history.append(row)
    if history:
        import pandas as pd
        pd.DataFrame(history).to_csv(history_path, index=False)

    print("Best trial:")
    print(f"  number: {study.best_trial.number}")
    print(f"  metric: {args.metric}")
    print(f"  value : {study.best_value:.4f}")
    print(f"  params: {best_params}")
    print(f"Config updated: {OUTPUT_PATH}")
    if history:
        print(f"Trial history saved to {history_path}")


if __name__ == '__main__':
    main()
