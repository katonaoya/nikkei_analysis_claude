from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.multi_model_threshold_optimizer import (
    evaluate_threshold_grid,
    parse_metric_weights,
)


def _build_sample_candidates() -> pd.DataFrame:
    data = []
    dates = pd.date_range('2025-01-01', periods=3, freq='B')
    for day_idx, date in enumerate(dates):
        for code_idx, prob in enumerate([0.15, 0.18, 0.21]):
            data.append({
                'analysis_date': date,
                'code': f'{1000 + code_idx}',
                'prediction_probability': prob,
                'downside_probability': 0.45 + 0.01 * code_idx,
                'risk_score': 0.30 + 0.05 * code_idx,
                'future_return': 0.02 if code_idx == 2 else -0.01,
            })
    return pd.DataFrame(data)


def test_parse_metric_weights_normalizes_values():
    weights = parse_metric_weights('precision:0.6,avg_return:0.3,coverage_rate:0.1')
    assert pytest.approx(sum(weights.values()), rel=1e-9) == 1.0
    assert weights['precision'] > weights['avg_return'] > weights['coverage_rate']


def test_evaluate_threshold_grid_applies_weighted_sort():
    candidates = _build_sample_candidates()

    weights = {'up': 1.0, 'down': 0.5, 'risk': 0.4}
    result = evaluate_threshold_grid(
        candidates_df=candidates,
        up_grid=[0.18, 0.20],
        down_grid=[0.46, 0.48],
        risk_grid=[0.35],
        weights=weights,
        top_n=2,
        max_per_sector=3,
        require_passed_all=True,
        target_return=0.01,
        transaction_cost=0.001,
        metric='precision',
        min_valid_count=2,
        metric_weights={'precision': 0.6, 'avg_return': 0.4},
        fallback_min_passed_ratio_values=[None, 0.4],
        fallback_max_per_sector=1,
    )

    assert not result.empty
    assert 'weighted_score' in result.columns
    assert 'fallback_min_passed_ratio' in result.columns
    assert 'fallback_max_per_sector' in result.columns
    assert result.iloc[0]['weighted_score'] >= result.iloc[-1]['weighted_score']
