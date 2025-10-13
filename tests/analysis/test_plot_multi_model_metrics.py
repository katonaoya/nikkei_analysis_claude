from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.plot_multi_model_metrics import (
    compute_moving_average,
    filter_date_range,
    load_metrics,
)


def test_compute_moving_average():
    dates = pd.date_range('2025-10-01', periods=5, freq='B')
    df = pd.DataFrame(
        {
            'analysis_date': dates,
            'precision': [0.2, 0.4, 0.6, 0.8, 1.0],
            'fallback_ratio': [1.0, 0.8, 0.6, 0.4, 0.2],
        }
    )
    result = compute_moving_average(df, window=3)
    assert result.loc[result['analysis_date'] == dates[2], 'precision_ma'].iloc[0] == pytest.approx((0.2 + 0.4 + 0.6) / 3)
    assert result.loc[result['analysis_date'] == dates[2], 'fallback_ratio_ma'].iloc[0] == pytest.approx((1.0 + 0.8 + 0.6) / 3)


def test_filter_and_load(tmp_path):
    csv_path = tmp_path / 'metrics.csv'
    df = pd.DataFrame(
        {
            'analysis_date': pd.date_range('2025-09-01', periods=4, freq='B'),
            'precision': [0.2, 0.3, 0.4, 0.5],
            'fallback_ratio': [0.5, 0.4, 0.6, 0.7],
        }
    )
    df.to_csv(csv_path, index=False)

    loaded = load_metrics(csv_path)
    filtered = filter_date_range(loaded, '2025-09-03', '2025-09-05')
    assert len(filtered) == 2
    assert filtered['analysis_date'].min() >= pd.Timestamp('2025-09-03')


def test_compute_window_validation():
    df = pd.DataFrame({'analysis_date': [], 'precision': [], 'fallback_ratio': []})
    with pytest.raises(ValueError):
        compute_moving_average(df, 0)
