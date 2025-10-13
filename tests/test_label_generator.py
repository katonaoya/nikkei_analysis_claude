import numpy as np
import pandas as pd

from src.features.label_generator import LabelGenerator


def _build_sample_price_frame():
    dates = pd.date_range('2025-01-01', periods=5, freq='B')
    return pd.DataFrame({
        'Code': ['1001'] * len(dates),
        'Date': dates,
        'Close': [100, 99, 101, 100, 99],
        'High': [101, 100, 102, 101, 100],
        'Low': [99, 98, 99, 98, 97]
    })


def test_create_downside_labels_defaults():
    df = _build_sample_price_frame()
    generator = LabelGenerator()

    result = generator.create_downside_labels(df, down_thresholds=-0.01)

    assert 'Return_Close_1d' in result.columns
    assert 'Target_Downside_1pct' in result.columns

    # The final horizon row is removed, so we expect len(df) - 1 rows
    assert len(result) == len(df) - 1

    # Day 1 should exactly hit -1% and therefore be labelled as downside
    assert result.loc[result['Date'] == pd.Timestamp('2025-01-01'), 'Target_Downside_1pct'].iloc[0] == 1

    # Positive return on 2025-01-03 should not be flagged
    assert result.loc[result['Date'] == pd.Timestamp('2025-01-03'), 'Target_Downside_1pct'].iloc[0] == 0


def test_create_downside_labels_strict_threshold():
    df = _build_sample_price_frame()
    generator = LabelGenerator()

    result = generator.create_downside_labels(
        df,
        down_thresholds=[-0.01, -0.015],
        allow_equal=False
    )

    strict_col = 'Target_Downside_1pct'
    deeper_col = 'Target_Downside_1_5pct'

    # With allow_equal=False, the -1% move should no longer be counted
    assert result.loc[result['Date'] == pd.Timestamp('2025-01-01'), strict_col].iloc[0] == 0

    # All returns are above -1.5%, so the deeper threshold should remain zeros
    assert result[deeper_col].sum() == 0


def test_create_downside_labels_horizon_cutoff():
    df = _build_sample_price_frame()
    generator = LabelGenerator()

    horizon_result = generator.create_downside_labels(df, down_thresholds=-0.01, horizon_days=2)

    # Two rows removed from the tail when horizon_days=2
    assert len(horizon_result) == len(df) - 2

    # Target column still exists
    assert 'Target_Downside_1pct' in horizon_result.columns


def test_create_volatility_risk_features_matches_reference():
    dates = pd.date_range('2025-02-03', periods=6, freq='B')
    close = pd.Series([100, 99, 103, 104, 98, 102], index=dates)
    high = close + [1, 0.5, 1, 1, 0.8, 1.2]
    low = close - [1, 1.2, 0.8, 0.6, 1.5, 0.9]

    df = pd.DataFrame({
        'Code': ['2002'] * len(dates),
        'Date': dates,
        'Close': close.values,
        'High': high.values,
        'Low': low.values
    })

    generator = LabelGenerator()
    result = generator.create_volatility_risk_features(df, window=3)

    expected_returns = df['Close'].pct_change()
    expected_rolling_vol = expected_returns.rolling(window=3, min_periods=2).std() * np.sqrt(252)
    expected_prev_close = df['Close'].shift(1)
    true_range = pd.concat([
        (df['High'] - df['Low']).abs(),
        (df['High'] - expected_prev_close).abs(),
        (df['Low'] - expected_prev_close).abs()
    ], axis=1).max(axis=1)
    expected_atr = true_range.rolling(window=3, min_periods=2).mean()
    expected_atr_pct = (expected_atr / df['Close']).fillna(0.0)
    expected_components = pd.concat([
        expected_rolling_vol.fillna(0.0),
        expected_atr_pct
    ], axis=1)

    scaled_components = []
    for col_name, series in expected_components.items():
        scale = np.nanpercentile(series.values, 90)
        scale = float(scale) if scale > 0 else 1.0
        scaled_components.append((series / scale).clip(lower=0))

    expected_risk = pd.concat(scaled_components, axis=1).mean(axis=1)

    assert np.allclose(result['Rolling_Volatility'], expected_rolling_vol.fillna(np.nan), equal_nan=True)
    assert np.allclose(result['ATR'], expected_atr.fillna(np.nan), equal_nan=True)
    assert np.allclose(result['Risk_Score'], expected_risk.fillna(0.0), equal_nan=True)

    # Sanity checks on bounds
    assert (result['Risk_Score'] >= 0).all()
    assert np.isfinite(result['Risk_Score']).all()
