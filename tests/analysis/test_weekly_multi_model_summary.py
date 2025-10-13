from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.weekly_multi_model_summary import (
    MetricsSummary,
    compute_summary,
    load_metrics,
    render_markdown,
    select_recent_days,
)


def _sample_df() -> pd.DataFrame:
    dates = pd.date_range('2025-09-01', periods=6, freq='B')
    return pd.DataFrame(
        {
            'analysis_date': dates,
            'precision': [0.2, 0.4, 0.6, 0.3, 0.1, 0.5],
            'avg_return': [0.01, 0.02, -0.01, 0.00, -0.02, 0.03],
            'coverage': [1.0] * 6,
            'fallback_ratio': [1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
            'selected_count': [2, 2, 2, 2, 2, 2],
            'fallback_count': [1, 1, 1, 0, 0, 0],
        }
    )


def test_select_recent_days():
    df = _sample_df()
    recent = select_recent_days(df, days=3)
    assert recent['analysis_date'].min() == df['analysis_date'].iloc[-3]
    assert recent['analysis_date'].nunique() == 3


def test_compute_summary_values():
    df = _sample_df()
    summary = compute_summary(df)
    assert isinstance(summary, MetricsSummary)
    assert summary.days == 6
    assert summary.fallback_full_days == 1
    assert pytest.approx(summary.precision_mean, rel=1e-6) == df['precision'].mean()
    assert pytest.approx(summary.fallback_ratio_mean, rel=1e-6) == df['fallback_ratio'].mean()


def test_render_markdown_contains_key_metrics():
    df = _sample_df()
    summary = compute_summary(df)
    md = render_markdown(summary, title='Test Summary', window=14)
    assert 'Test Summary' in md
    assert 'Precision' in md
    assert 'フォールバック比率' in md


def test_load_metrics(tmp_path):
    df = _sample_df()
    csv_path = tmp_path / 'metrics.csv'
    df.to_csv(csv_path, index=False)
    loaded = load_metrics(csv_path)
    assert len(loaded) == len(df)

