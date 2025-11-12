import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

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


def test_apply_calibration_with_platt_model_object():
    probs = np.linspace(0.1, 0.9, 9)
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1])
    calibrator = LogisticRegression(solver='lbfgs', max_iter=1000)
    calibrator.fit(probs.reshape(-1, 1), labels)

    calibration_info = {
        'method': 'platt',
        'model': calibrator,
    }

    adjusted = CloseReturnPrecisionSystemV1.apply_calibration(probs, calibration_info)

    assert adjusted.shape == probs.shape
    assert np.all(adjusted >= 0) and np.all(adjusted <= 1)
    assert np.max(np.abs(adjusted - probs)) > 1e-3


def test_apply_calibration_backward_compat_dict():
    original = 0.5
    calibration_info = {'coef': 2.0, 'intercept': -1.0}
    adjusted = CloseReturnPrecisionSystemV1.apply_calibration(original, calibration_info)
    expected = 1 / (1 + np.exp(-((2.0 * original) - 1.0)))
    assert adjusted == pytest.approx(expected)


def test_apply_calibration_isotonic_method():
    probs = np.linspace(0.0, 1.0, 6)
    labels = np.array([0, 0, 0, 1, 1, 1])
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(probs, labels)

    calibration_info = {'method': 'isotonic', 'model': calibrator}
    adjusted = CloseReturnPrecisionSystemV1.apply_calibration(probs, calibration_info)

    assert adjusted.shape == probs.shape
    assert np.all(adjusted >= 0) and np.all(adjusted <= 1)
    assert adjusted[0] <= adjusted[-1]
    assert np.max(np.abs(adjusted - probs)) > 0.1
