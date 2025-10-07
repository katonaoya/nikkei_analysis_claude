import pandas as pd
import pytest

from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1


def _create_system(imbalance_boost: float = 1.0, **kwargs) -> CloseReturnPrecisionSystemV1:
    # ダミーファイルパスを渡してI/Oを回避
    return CloseReturnPrecisionSystemV1(
        stock_file="tests/data/dummy_stock.parquet",
        external_file="tests/data/dummy_external.parquet",
        imbalance_boost=imbalance_boost,
        **kwargs,
    )


def test_compute_scale_pos_weight_applies_boost():
    series = pd.Series([1] * 5 + [0] * 15)
    system = _create_system(imbalance_boost=1.5)
    weight = system._compute_scale_pos_weight(series)
    expected = (15 / 5) * 1.5
    assert weight == pytest.approx(expected)


def test_compute_scale_pos_weight_handles_no_positive_samples():
    series = pd.Series([0] * 10)
    system = _create_system(imbalance_boost=0.8)
    weight = system._compute_scale_pos_weight(series)
    assert weight == pytest.approx(0.8)


def test_compute_sample_weights_balanced_strategy():
    series = pd.Series([1] * 4 + [0] * 12)
    system = _create_system(imbalance_strategy='balanced')
    weights = system._compute_sample_weights(series)
    assert weights is not None
    assert pytest.approx(weights[series == 1].mean(), rel=1e-3) == 2.0
    assert pytest.approx(weights[series == 0].mean(), rel=1e-3) == 0.6666666667


def test_apply_positive_oversample_increases_positive_count():
    X = pd.DataFrame({'f1': [0.1, 0.2, 0.3, 0.4], 'f2': [1, 2, 3, 4]})
    y = pd.Series([0, 1, 0, 1])
    system = _create_system(positive_oversample_ratio=2.0)
    X_aug, y_aug = system._apply_positive_oversample(X, y)
    assert len(y_aug) >= len(y)
    assert y_aug.sum() >= y.sum()
