#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""推奨閾値の最適化と設定ファイルへの反映"""

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1

MODEL_DIR = Path("models/enhanced_close_v1")
CONFIG_PATH = Path("config/close_recommendation_config.json")
TARGET_CONFIG_PATH = Path("config/close_threshold.json")


def load_latest_model_data():
    files = sorted(MODEL_DIR.glob("close_model_v1_*.joblib"))
    if not files:
        raise FileNotFoundError("close_model_v1_*.joblib が見つかりません")
    latest = files[-1]
    import joblib
    return joblib.load(latest), latest


def prepare_dataset(system: CloseReturnPrecisionSystemV1) -> pd.DataFrame:
    df = system.create_enhanced_features(system.load_and_integrate_data())
    df = df.sort_values(['Code', 'Date'])
    df['future_close'] = df.groupby('Code')['Close'].shift(-1)
    df['future_return'] = df['future_close'] / df['Close'] - 1
    df = df.dropna(subset=['future_return'])
    return df


def compute_probabilities(dataset: pd.DataFrame, model_data: dict) -> np.ndarray:
    feature_cols = model_data['feature_cols']
    selector = model_data.get('selector')
    scaler = model_data.get('scaler')
    model = model_data['model']

    X = dataset[feature_cols].fillna(method='ffill').fillna(0)
    if selector is not None:
        X = selector.transform(X)
    if scaler is not None:
        X = scaler.transform(X)
    proba = model.predict_proba(X)[:, 1]

    calibration = model_data.get('calibration')
    if calibration is not None:
        coef = calibration.get('coef', 0.0)
        intercept = calibration.get('intercept', 0.0)
        linear = coef * proba + intercept
        proba = 1 / (1 + np.exp(-linear))
    return proba


def evaluate_thresholds(dataset: pd.DataFrame, prob: np.ndarray, thresholds: List[float], transaction_cost: float, target_returns: List[float]) -> pd.DataFrame:
    df = dataset.copy()
    df['probability'] = prob
    df['net_return'] = df['future_return'] - transaction_cost

    results = []
    for tr in target_returns:
        for th in thresholds:
            mask = df['probability'] >= th
            subset = df[mask]
            if subset.empty:
                results.append({
                    'target_return': tr,
                    'threshold': th,
                    'count': 0,
                    'hit_rate': 0.0,
                    'avg_return': 0.0,
                    'avg_net_return': 0.0,
                })
                continue
            hits = (subset['future_return'] >= tr).mean()
            results.append({
                'target_return': tr,
                'threshold': th,
                'count': len(subset),
                'hit_rate': hits,
                'avg_return': subset['future_return'].mean(),
                'avg_net_return': subset['net_return'].mean(),
            })
    return pd.DataFrame(results)


def update_config(best_threshold: float):
    data = {
        'min_probability': best_threshold
    }
    TARGET_CONFIG_PATH.write_text(json.dumps(data, indent=2))
    if CONFIG_PATH.exists():
        config = json.loads(CONFIG_PATH.read_text())
    else:
        config = {}
    config['min_probability'] = best_threshold
    CONFIG_PATH.write_text(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description='推奨閾値の最適化')
    parser.add_argument('--thresholds', type=str, default='0.55,0.60,0.65,0.70', help='評価する確率閾値 (カンマ区切り)')
    parser.add_argument('--transaction-cost', type=float, default=0.0, help='片道ベースのコスト比率')
    parser.add_argument('--target-returns', type=str, default=None, help='評価するヒット閾値 (カンマ区切り, 例: 0.008,0.01)')
    parser.add_argument('--export-csv', type=str, default=None, help='結果をCSV出力するパス')
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(',') if x]
    system = CloseReturnPrecisionSystemV1()
    dataset = prepare_dataset(system)
    model_data, latest_model_file = load_latest_model_data()

    prob = compute_probabilities(dataset, model_data)
    y_true = dataset['Target']
    roc_auc = roc_auc_score(y_true, prob)
    pr_auc = average_precision_score(y_true, prob)

    default_target = model_data.get('target_return', system.target_return)
    if args.target_returns:
        target_returns = [float(x) for x in args.target_returns.split(',') if x]
    else:
        target_returns = [default_target]

    metrics_df = evaluate_thresholds(dataset, prob, thresholds, args.transaction_cost, target_returns)

    if metrics_df.empty:
        print("No recommendations met the specified thresholds.")
        return

    metrics_df.sort_values(['target_return', 'avg_net_return'], ascending=[True, False], inplace=True)
    best_row = metrics_df.iloc[0]
    best_threshold = float(best_row['threshold'])

    update_config(best_threshold)

    if args.export_csv:
        export_path = Path(args.export_csv)
        export_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_df.to_csv(export_path, index=False)

    print(f"Latest model: {latest_model_file.name}")
    print(f"ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}")
    print("\nThreshold metrics (sorted by net return):")
    print(metrics_df.to_string(index=False, formatters={
        'target_return': lambda v: f"{v*100:.2f}%",
        'threshold': lambda v: f"{v:.2f}",
        'hit_rate': lambda v: f"{v*100:.2f}%",
        'avg_return': lambda v: f"{v*100:.2f}%",
        'avg_net_return': lambda v: f"{v*100:.2f}%",
    }))
    print(f"\nBest threshold: {best_threshold:.2f} (target_return {best_row['target_return']*100:.2f}%, net {best_row['avg_net_return']*100:.2f}% across {int(best_row['count'])} trades)")
    print(f"Config updated: {CONFIG_PATH} & {TARGET_CONFIG_PATH}")
    if args.export_csv:
        print(f"Detailed metrics exported to {args.export_csv}")


if __name__ == '__main__':
    main()
