from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.downside_risk_system_v1 import DownsideRiskSystemV1, FEATURE_COLUMNS


def _build_sample_stock_df() -> pd.DataFrame:
    dates = pd.date_range('2024-12-02', periods=45, freq='B')
    pattern = [0, 1, -1, 0, -2, -1, 0, 1, -1, -3, -2, -1, 0, 1, 2]
    records = []
    for code_idx, code in enumerate(['1001', '1002'], start=1):
        sector = 'Tech' if code == '1001' else 'Finance'
        base_price = 100 + code_idx * 5
        for offset, date in enumerate(dates):
            delta = pattern[offset % len(pattern)]
            close = base_price + delta + offset * 0.1
            records.append({
                'Code': code,
                'Date': date,
                'Open': close - 0.5,
                'High': close + 1.0,
                'Low': close - 1.5,
                'Close': close,
                'Volume': 100000 + offset * 500,
                'Sector': sector,
            })
    return pd.DataFrame(records)


def _create_system(tmp_path, stock_df: pd.DataFrame) -> DownsideRiskSystemV1:
    stock_path = tmp_path / 'stock.parquet'
    stock_df.to_parquet(stock_path, index=False)
    return DownsideRiskSystemV1(
        stock_file=str(stock_path),
        down_threshold=-0.01,
        horizon_days=1,
        models_dir=str(tmp_path / 'models'),
        production_dir=str(tmp_path / 'production'),
    )


def test_prepare_features_generates_future_return(tmp_path):
    stock_df = _build_sample_stock_df()
    system = _create_system(tmp_path, stock_df)

    features = system._prepare_features(stock_df)

    expected_cols = set(FEATURE_COLUMNS + ['future_return', 'target'])
    assert expected_cols.issubset(features.columns)
    assert features['future_return'].notna().sum() > 0
    non_na = features.dropna(subset=FEATURE_COLUMNS + ['future_return'])
    assert not non_na.empty
    assert {'future_return_2d', 'down_target_1pct_2d', 'drawdown_3pct_3d', 'no_rebound_2d'}.issubset(features.columns)


def test_predict_probabilities_outputs_expected_length(tmp_path):
    stock_df = _build_sample_stock_df()
    system = _create_system(tmp_path, stock_df)
    features = system._prepare_features(stock_df)
    bundle = system._train_model(features)
    valid_df = features.dropna(subset=FEATURE_COLUMNS + ['future_return'])
    preds = system._predict_probabilities(valid_df, bundle)
    assert len(preds) == len(valid_df)


def test_run_outputs_prediction_files(tmp_path):
    stock_df = _build_sample_stock_df()
    stock_path = tmp_path / 'stock.parquet'
    stock_df.to_parquet(stock_path, index=False)

    system = DownsideRiskSystemV1(
        stock_file=str(stock_path),
        down_threshold=-0.01,
        horizon_days=1,
        models_dir=tmp_path / 'models',
        production_dir=tmp_path / 'production',
    )

    target_date = stock_df['Date'].max()
    system.run(predict_date=str(target_date.date()), retrain=True)

    downside_path = tmp_path / 'production' / 'downside_predictions.parquet'
    risk_path = tmp_path / 'production' / 'risk_predictions.parquet'

    assert downside_path.exists()
    assert risk_path.exists()

    down_df = pd.read_parquet(downside_path)
    risk_df = pd.read_parquet(risk_path)

    target_col = system._down_target_column_name(system.down_threshold)
    assert {'analysis_date', 'code', 'prob_down', 'future_return', target_col}.issubset(down_df.columns)
    assert {'analysis_date', 'code', 'risk_score'}.issubset(risk_df.columns)


def test_run_with_multiple_thresholds_outputs_all_targets(tmp_path):
    stock_df = _build_sample_stock_df()
    stock_path = tmp_path / 'stock.parquet'
    stock_df.to_parquet(stock_path, index=False)

    thresholds = [-0.015, -0.01, -0.02]
    system = DownsideRiskSystemV1(
        stock_file=str(stock_path),
        down_threshold=-0.01,
        down_thresholds=thresholds,
        horizon_days=1,
        models_dir=tmp_path / 'models_multi',
        production_dir=tmp_path / 'production_multi',
    )

    target_date = stock_df['Date'].max()
    system.run(predict_date=str(target_date.date()), retrain=True)

    down_df = pd.read_parquet(tmp_path / 'production_multi' / 'downside_predictions.parquet')

    for threshold in thresholds:
        col_name = system._down_target_column_name(threshold)
        assert col_name in down_df.columns

    assert {'down_target_1pct_2d', 'drawdown_3pct_3d', 'no_rebound_2d'}.issubset(down_df.columns)
    assert set(down_df['down_target_1pct_2d'].unique()).issubset({0, 1})
