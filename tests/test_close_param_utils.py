import pandas as pd

from analysis.close_param_utils import compute_topn_precision


def test_compute_topn_precision_basic():
    data = pd.DataFrame(
        {
            'Date': ['2025-10-01'] * 5 + ['2025-10-02'] * 4,
            'Code': list(range(9)),
            'Target': [1, 0, 1, 0, 0, 0, 1, 0, 0],
        }
    )
    prob = pd.Series([0.9, 0.8, 0.7, 0.1, 0.05, 0.95, 0.6, 0.4, 0.2])

    precision = compute_topn_precision(data, prob.to_numpy(), top_n=2)
    # 1日目: 上位2は target=[1,0] -> precision 0.5
    # 2日目: 上位2は target=[0,1] -> precision 0.5 => overall 0.5
    assert precision == 0.5


def test_compute_topn_precision_handles_empty_selection():
    data = pd.DataFrame({'Date': [], 'Code': [], 'Target': []})
    prob = pd.Series([], dtype=float)

    precision = compute_topn_precision(data, prob.to_numpy(), top_n=3)
    assert precision == 0.0
