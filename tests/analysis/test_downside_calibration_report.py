from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.downside_calibration_report import build_report, compute_calibration_metrics


def test_compute_calibration_metrics_basic_case():
    prob = pd.Series([0.1, 0.2, 0.8, 0.9])
    label = pd.Series([0, 0, 1, 1])
    metrics = compute_calibration_metrics(prob, label, bins=2)

    expected_brier = float(np.mean((prob - label) ** 2))
    assert metrics['brier_score'] == expected_brier
    assert metrics['count'] == 4.0
    assert metrics['positive_rate'] == 0.5


def test_build_report_returns_rows_for_each_label():
    df = pd.DataFrame(
        {
            'prob_down': [0.2, 0.8, 0.6, 0.4],
            'down_target_1pct': [0, 1, 1, 0],
            'down_target_1_5pct': [0, 1, 0, 0],
        }
    )

    report = build_report(df, label_prefix='down_target_', bins=5)
    assert set(report['label_column']) == {'down_target_1pct', 'down_target_1_5pct'}
    assert (report['count'] == 4).all()


def test_build_report_raises_when_label_missing():
    df = pd.DataFrame({'prob_down': [0.1, 0.2]})
    try:
        build_report(df, label_prefix='down_target_', bins=3)
    except ValueError as exc:
        assert 'down_target_' in str(exc)
    else:
        raise AssertionError('Expected ValueError when label columns are absent')


def test_build_report_accepts_default_down_target():
    df = pd.DataFrame({
        'prob_down': [0.3, 0.6, 0.8],
        'down_target': [0, 1, 1],
    })
    report = build_report(df, label_prefix='down_target_', bins=3)
    assert report['label_column'].tolist() == ['down_target']
