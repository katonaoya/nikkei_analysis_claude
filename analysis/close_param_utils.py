#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""終値モデル向けパラメータ探索ユーティリティ"""

from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score

import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1


def prepare_dataset(system: CloseReturnPrecisionSystemV1) -> pd.DataFrame:
    df = system.create_enhanced_features(system.load_and_integrate_data())
    df = df.sort_values('Date')
    return df


def compute_topn_precision(meta: pd.DataFrame, prob: np.ndarray, top_n: int) -> float:
    if top_n <= 0:
        return 0.0

    top_n = int(top_n)
    df = meta.copy()
    df = df.assign(probability=prob)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])

    total_selected = 0
    total_hits = 0
    for _, group in df.groupby('Date'):
        selected = group.sort_values('probability', ascending=False).head(top_n)
        total_selected += len(selected)
        total_hits += selected['Target'].sum()

    if total_selected == 0:
        return 0.0
    return float(total_hits) / float(total_selected)


def evaluate_params(
    system: CloseReturnPrecisionSystemV1,
    df: pd.DataFrame,
    params: Dict[str, float],
    metric: str = 'f1',
    top_n: int = 5,
) -> float:
    feature_cols = [
        col for col in df.columns
        if col not in ['Date', 'Code', 'Target']
        and str(df[col].dtype) in ['int64', 'float64', 'int32', 'float32']
    ]
    if len(feature_cols) > 30:
        counts = df[feature_cols].count()
        feature_cols = counts.nlargest(30).index.tolist()

    split_idx = int(len(df) * 0.8)
    X_train = df.iloc[:split_idx][feature_cols].fillna(method='ffill').fillna(0)
    y_train = df.iloc[:split_idx]['Target']
    X_test = df.iloc[split_idx:][feature_cols].fillna(method='ffill').fillna(0)
    y_test = df.iloc[split_idx:]['Target']
    test_meta = df.iloc[split_idx:][['Date', 'Code', 'Target']].copy()

    X_train, y_train = system._apply_positive_oversample(X_train, y_train)
    sample_weight = system._compute_sample_weights(y_train)

    selector = SelectKBest(score_func=f_classif, k=min(30, len(feature_cols)))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)

    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    X_test_scaled = scaler.transform(X_test_sel)

    scale_pos_weight = system._compute_scale_pos_weight(y_train)
    if system.imbalance_strategy and system.imbalance_strategy.lower() not in ("", "none", "scale_pos"):
        scale_pos_weight = 1.0

    model = lgb.LGBMClassifier(
        objective='binary',
        random_state=42,
        verbose=-1,
        scale_pos_weight=scale_pos_weight,
        **params
    )
    model.fit(X_train_scaled, y_train, sample_weight=sample_weight)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    metric = (metric or 'f1').lower()
    if metric == 'precision_topn':
        return compute_topn_precision(test_meta, y_pred_proba, top_n)
    if metric == 'precision':
        from sklearn.metrics import precision_score

        return precision_score(y_test, y_pred, zero_division=0)
    if metric == 'f1':
        return f1_score(y_test, y_pred, zero_division=0)

    raise ValueError(f"Unsupported metric: {metric}")
