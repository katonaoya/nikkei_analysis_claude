#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""LightGBM ハイパーパラメータの簡易グリッドサーチ"""

import argparse
import itertools
import json
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1
from analysis.close_param_utils import prepare_dataset, evaluate_params

PARAM_GRID = {
    "learning_rate": [0.02, 0.03],
    "max_depth": [6, 8],
    "num_leaves": [31, 63],
    "min_child_samples": [20, 35],
    "subsample": [0.8],
    "colsample_bytree": [0.8],
    "n_estimators": [250, 320],
    "reg_alpha": [0.05],
    "reg_lambda": [0.1]
}

OUTPUT_PATH = Path("config/close_model_params.json")


def main():
    parser = argparse.ArgumentParser(description="LightGBM ハイパーパラメータの簡易グリッドサーチ")
    parser.add_argument('--imbalance-boost', type=float, default=1.0, help='scale_pos_weight に掛ける倍率')
    parser.add_argument('--imbalance-strategy', type=str, default='scale_pos', choices=['scale_pos', 'balanced', 'focal', 'none'], help='追加のサンプル重み戦略')
    parser.add_argument('--focal-gamma', type=float, default=2.0, help='focal戦略用ガンマ値 (imbalance-strategy=focal のみ)')
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

    best_score = -1.0
    best_params = None

    keys = list(PARAM_GRID.keys())
    for values in itertools.product(*(PARAM_GRID[k] for k in keys)):
        params = dict(zip(keys, values))
        try:
            score = evaluate_params(system, dataset, params, metric=args.metric, top_n=args.top_n)
        except Exception as exc:
            print(f"skip params {params} due to error: {exc}")
            continue
        print(f"params={params} -> {args.metric.upper()}={score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params

    if best_params is None:
        print("no valid params found")
        return

    print("\nBest params:", best_params)
    print(f"Best {args.metric.upper()}: {best_score:.4f}")
    with OUTPUT_PATH.open('w') as f:
        json.dump(best_params, f, indent=2)


if __name__ == "__main__":
    main()
