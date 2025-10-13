from pathlib import Path
import sys

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.build_multi_model_candidate_dataset import (
    compute_future_returns,
    merge_candidate_frames,
    CandidateBuilderConfig,
    MultiModelCandidateBuilder,
)


def test_compute_future_returns_simple_case():
    dates = pd.date_range('2025-01-01', periods=4, freq='B')
    data = []
    for idx, code in enumerate(['1001', '1002']):
        base = 100 + idx * 5
        for offset, date in enumerate(dates):
            data.append({'Code': code, 'Date': date, 'Close': base + offset})

    stock_df = pd.DataFrame(data)
    returns_df = compute_future_returns(stock_df)

    assert set(returns_df.columns) == {'analysis_date', 'code', 'future_return'}
    assert returns_df['future_return'].notna().all()

    sample = returns_df.loc[(returns_df['code'] == '1001') & (returns_df['analysis_date'] == pd.Timestamp('2025-01-01'))]
    assert pytest.approx(sample.iloc[0]['future_return'], rel=1e-9) == (101 - 100) / 100


def test_merge_candidate_frames_aligns_probabilities():
    analysis_date = pd.Timestamp('2025-03-10')
    upside_df = pd.DataFrame({
        'analysis_date': [analysis_date, analysis_date],
        'code': ['1001', '1002'],
        'prediction_probability': [0.6, 0.45],
        'company_name': ['A', 'B'],
    })

    downside_df = pd.DataFrame({
        'analysis_date': [analysis_date],
        'code': ['1001'],
        'prob_down': [0.2],
    })

    risk_df = pd.DataFrame({
        'analysis_date': [analysis_date],
        'code': ['1002'],
        'risk_score': [0.5],
    })

    returns_df = pd.DataFrame({
        'analysis_date': [analysis_date, analysis_date],
        'code': ['1001', '1002'],
        'future_return': [0.03, -0.01],
    })

    merged = merge_candidate_frames(upside_df, downside_df, risk_df, returns_df)

    assert len(merged) == 2
    record_a = merged.loc[merged['code'] == '1001'].iloc[0]
    record_b = merged.loc[merged['code'] == '1002'].iloc[0]

    assert record_a['downside_probability'] == 0.2
    assert record_a['risk_score'] == 0.0
    assert record_b['risk_score'] == 0.5
    assert record_b['downside_probability'] == 0.0
    assert record_a['future_return'] == 0.03


class _StubCloseSystem:
    def __init__(self, frames):
        self.frames = frames

    def predict_all_candidates(self, target_date_str):
        df = self.frames.get(target_date_str)
        if df is None:
            return pd.DataFrame(columns=['analysis_date', 'code', 'prediction_probability'])
        return df.copy()


class _StubDownsideSystem:
    def __init__(self, stock_df: pd.DataFrame, predictions: dict, production_dir: Path):
        self._stock_df = stock_df
        self._predictions = predictions
        self.production_dir = production_dir
        self.production_dir.mkdir(parents=True, exist_ok=True)

    def _load_stock_data(self) -> pd.DataFrame:
        return self._stock_df.copy()

    def run(self, predict_date: str, retrain: bool = False) -> None:
        pred = self._predictions[predict_date]
        downside_path = self.production_dir / 'downside_predictions.parquet'
        risk_path = self.production_dir / 'risk_predictions.parquet'
        pred['analysis_date'] = pd.to_datetime(pred['analysis_date']).dt.normalize()
        downside_cols = ['analysis_date', 'code', 'prob_down', 'future_return']
        if 'down_target_1pct' not in pred.columns:
            pred['down_target_1pct'] = (pred['future_return'] <= -0.01).astype(int)
        downside_cols.append('down_target_1pct')
        pred[downside_cols].to_parquet(downside_path, index=False)
        risk_df = pred[['analysis_date', 'code', 'risk_score']].copy()
        risk_df.to_parquet(risk_path, index=False)


def test_builder_append_output_merges_history(tmp_path):
    temp_dir = tmp_path / 'builder_tmp'
    output_path = tmp_path / 'candidates.parquet'

    dates = [pd.Timestamp('2025-02-03'), pd.Timestamp('2025-02-04')]
    close_frames = {
        date.strftime('%Y-%m-%d'): pd.DataFrame({
            'analysis_date': [date] * 2,
            'code': ['1001', '1002'],
            'prediction_probability': [0.6, 0.55],
            'company_name': ['A', 'B'],
            'sector': ['Tech', 'Finance'],
        })
        for date in dates
    }

    stock_records = []
    for code in ['1001', '1002']:
        for date in pd.date_range(dates[0] - pd.Timedelta(days=5), dates[-1] + pd.Timedelta(days=1), freq='B'):
            stock_records.append({'Code': code, 'Date': date, 'Close': 100.0})
    stock_df = pd.DataFrame(stock_records)

    predictions = {
        date.strftime('%Y-%m-%d'): pd.DataFrame({
            'analysis_date': [date] * 2,
            'code': ['1001', '1002'],
            'prob_down': [0.2, 0.3],
            'risk_score': [0.4, 0.5],
            'future_return': [0.02, -0.01],
        })
        for date in dates
    }

    close_stub = _StubCloseSystem(close_frames)
    downside_stub = _StubDownsideSystem(stock_df, predictions, temp_dir / 'production')

    config_first = CandidateBuilderConfig(
        start_date=dates[0].strftime('%Y-%m-%d'),
        end_date=dates[0].strftime('%Y-%m-%d'),
        lookback_days=1,
        max_candidates=50,
        output_path=output_path,
        temp_dir=temp_dir,
        retrain_first=True,
        config_path=Path('config/multi_model_recommendation.json'),
        append_output=False,
        down_thresholds=None,
    )

    builder_first = MultiModelCandidateBuilder(
        config=config_first,
        close_system=close_stub,
        downside_system=downside_stub,
    )
    builder_first.build()

    assert output_path.exists()
    first_df = pd.read_parquet(output_path)
    assert first_df['analysis_date'].nunique() == 1

    config_second = CandidateBuilderConfig(
        start_date=dates[1].strftime('%Y-%m-%d'),
        end_date=dates[1].strftime('%Y-%m-%d'),
        lookback_days=2,
        max_candidates=50,
        output_path=output_path,
        temp_dir=temp_dir,
        retrain_first=False,
        config_path=Path('config/multi_model_recommendation.json'),
        append_output=True,
        down_thresholds=None,
    )

    builder_second = MultiModelCandidateBuilder(
        config=config_second,
        close_system=close_stub,
        downside_system=downside_stub,
    )
    builder_second.build()

    merged_df = pd.read_parquet(output_path)
    assert merged_df['analysis_date'].nunique() == 2
    assert set(merged_df['analysis_date'].dt.normalize()) == set(dates)
