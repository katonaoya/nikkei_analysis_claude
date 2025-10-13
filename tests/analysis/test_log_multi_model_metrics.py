from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.log_multi_model_metrics import append_metrics, compute_daily_metrics, rotate_archives
from reports.daily_stock_recommendation_multi import prepare_candidate_scores, select_top_candidates


def _build_candidates() -> pd.DataFrame:
    dates = [pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-03')]
    rows = []
    for date in dates:
        rows.extend(
            [
                {
                    'analysis_date': date,
                    'code': '1001',
                    'prediction_probability': 0.6,
                    'downside_probability': 0.2,
                    'risk_score': 0.3,
                    'future_return': 0.02,
                },
                {
                    'analysis_date': date,
                    'code': '1002',
                    'prediction_probability': 0.55,
                    'downside_probability': 0.4,
                    'risk_score': 0.35,
                    'future_return': 0.0,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_compute_daily_metrics_returns_precision():
    candidates = _build_candidates()
    thresholds = {'up': 0.5, 'down': 0.5, 'risk': 0.5}
    weights = {'up': 1.0, 'down': 0.5, 'risk': 0.4}
    scored = prepare_candidate_scores(candidates, thresholds, weights)
    selections = select_top_candidates(
        scored,
        top_n=2,
        max_per_sector=3,
        require_passed_all=True,
    )

    selected_df = pd.DataFrame(selections)
    daily_df = compute_daily_metrics(selected_df, target_return=0.01)

    assert len(daily_df) == 2
    assert daily_df.loc[daily_df['analysis_date'] == pd.Timestamp('2025-01-02'), 'precision'].iloc[0] == 0.5
    assert daily_df['coverage'].eq(1.0).all()
    assert daily_df['fallback_ratio'].eq(0.0).all()


def test_append_metrics_overwrites_existing(tmp_path):
    log_path = tmp_path / 'metrics.csv'
    df1 = pd.DataFrame(
        {
            'analysis_date': [pd.Timestamp('2025-01-02')],
            'selected_count': [2],
            'valid_count': [2],
            'hit_count': [1],
            'precision': [0.5],
            'avg_return': [0.01],
            'coverage': [1.0],
            'fallback_count': [0],
            'passed_all_count': [2],
            'fallback_ratio': [0.0],
            'passed_all_ratio': [1.0],
        }
    )
    combined = append_metrics(log_path, df1)
    assert not combined.empty

    df2 = pd.DataFrame(
        {
            'analysis_date': [pd.Timestamp('2025-01-02'), pd.Timestamp('2025-01-03')],
            'selected_count': [3, 1],
            'valid_count': [3, 1],
            'hit_count': [2, 0],
            'precision': [2 / 3, 0.0],
            'avg_return': [0.02, -0.01],
            'coverage': [1.0, 1.0],
            'fallback_count': [1, 0],
            'passed_all_count': [2, 1],
            'fallback_ratio': [1 / 3, 0.0],
            'passed_all_ratio': [2 / 3, 1.0],
        }
    )
    combined = append_metrics(log_path, df2)
    assert len(combined) == 2

    stored = pd.read_csv(log_path, parse_dates=['analysis_date'])
    assert len(stored) == 2
    assert stored.loc[stored['analysis_date'] == pd.Timestamp('2025-01-02'), 'selected_count'].iloc[0] == 3


def test_rotate_archives_creates_monthly_files(tmp_path):
    log_path = tmp_path / 'metrics.csv'
    archive_dir = tmp_path / 'archive'

    df = pd.DataFrame(
        {
            'analysis_date': pd.to_datetime([
                '2025-04-01',
                '2025-05-01',
                '2025-06-01',
                '2025-07-01',
            ]),
            'selected_count': [1, 2, 3, 4],
            'valid_count': [1, 2, 3, 4],
            'hit_count': [1, 1, 2, 3],
            'precision': [1.0, 0.5, 2 / 3, 0.75],
            'avg_return': [0.01, 0.02, 0.03, 0.04],
            'coverage': [1.0, 1.0, 1.0, 1.0],
            'fallback_count': [0, 0, 1, 1],
            'passed_all_count': [1, 2, 2, 3],
            'fallback_ratio': [0.0, 0.0, 1 / 3, 0.25],
            'passed_all_ratio': [1.0, 1.0, 2 / 3, 0.75],
        }
    )

    combined = append_metrics(log_path, df)
    trimmed = rotate_archives(
        combined,
        log_path=log_path,
        archive_dir=archive_dir,
        keep_months=2,
    )

    assert (archive_dir / 'multi_model_metrics_202504.csv').exists()
    assert (archive_dir / 'multi_model_metrics_202505.csv').exists()
    assert trimmed['analysis_date'].dt.month.tolist() == [6, 7]
    stored = pd.read_csv(log_path, parse_dates=['analysis_date'])
    assert stored['analysis_date'].dt.month.tolist() == [6, 7]


def test_select_top_candidates_respects_fallback_thresholds():
    candidates = pd.DataFrame(
        [
            {
                'analysis_date': pd.Timestamp('2025-01-06'),
                'code': '1001',
                'sector': 'Tech',
                'prediction_probability': 0.62,
                'downside_probability': 0.18,
                'risk_score': 0.28,
                'future_return': 0.03,
            },
            {
                'analysis_date': pd.Timestamp('2025-01-06'),
                'code': '1002',
                'sector': 'Finance',
                'prediction_probability': 0.52,
                'downside_probability': 0.42,
                'risk_score': 0.50,
                'future_return': -0.01,
            },
        ]
    )

    thresholds = {'up': 0.55, 'down': 0.35, 'risk': 0.40}
    weights = {'up': 1.0, 'down': 0.5, 'risk': 0.4}
    scored = prepare_candidate_scores(candidates, thresholds, weights)

    baseline = select_top_candidates(
        scored,
        top_n=2,
        max_per_sector=2,
        require_passed_all=False,
        fallback_max_fallback=1,
        risk_threshold=thresholds['risk'],
    )
    assert len(baseline) == 2
    assert sum(not rec['passed_all_filters'] for rec in baseline) == 1

    gated_by_prob = select_top_candidates(
        scored,
        top_n=2,
        max_per_sector=2,
        require_passed_all=False,
        fallback_max_fallback=1,
        fallback_min_up_prob=0.60,
        risk_threshold=thresholds['risk'],
    )
    assert len(gated_by_prob) == 1
    assert all(rec['passed_all_filters'] for rec in gated_by_prob)

    gated_by_risk = select_top_candidates(
        scored,
        top_n=2,
        max_per_sector=2,
        require_passed_all=False,
        fallback_max_fallback=1,
        fallback_risk_margin=0.05,
        risk_threshold=thresholds['risk'],
    )
    assert len(gated_by_risk) == 1
    assert all(rec['passed_all_filters'] for rec in gated_by_risk)
