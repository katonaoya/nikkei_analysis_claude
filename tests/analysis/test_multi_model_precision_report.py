from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.multi_model_precision_report import summarise


def test_summarise_outputs_expected_metrics(tmp_path):
    data = [
        {
            'analysis_date': pd.Timestamp('2025-10-06'),
            'future_return': 0.02,
        },
        {
            'analysis_date': pd.Timestamp('2025-10-06'),
            'future_return': -0.01,
        },
        {
            'analysis_date': pd.Timestamp('2025-10-07'),
            'future_return': 0.015,
        },
    ]
    selected = pd.DataFrame(data)
    metrics = summarise(selected, total_days=4)
    assert metrics['selected_days'] == 2
    assert metrics['selected_total'] == 3
    assert metrics['precision'] == 2 / 3
    assert metrics['avg_return'] == pytest.approx((0.02 - 0.01 + 0.015) / 3)
    assert metrics['coverage_rate'] == 0.5


import pytest
