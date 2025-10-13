import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.build_multi_model_candidates import build_candidates_dataframe, build_multi_model_candidates


def test_build_candidates_dataframe_merges_optional_inputs():
    date = pd.Timestamp('2025-10-10')
    upside_df = pd.DataFrame(
        {
            'analysis_date': [date, date],
            'next_trade_date': [date, date],
            'code': ['1301', '1302'],
            'prediction_probability': [0.6, 0.5],
            'current_price': [1000, 900],
        }
    )
    downside_df = pd.DataFrame(
        {
            'analysis_date': [date],
            'code': ['1301'],
            'prob_down': [0.2],
            'future_return': [-0.015],
        }
    )
    risk_df = pd.DataFrame(
        {
            'analysis_date': [date],
            'code': ['1302'],
            'risk_score': [0.45],
        }
    )

    combined = build_candidates_dataframe(
        upside_df=upside_df,
        downside_df=downside_df,
        risk_df=risk_df,
    )

    assert 'prob_down' in combined.columns
    assert 'risk_score' in combined.columns
    row_1301 = combined[combined['code'] == '1301'].iloc[0]
    row_1302 = combined[combined['code'] == '1302'].iloc[0]
    assert row_1301['prob_down'] == 0.2
    assert row_1302['risk_score'] == 0.45


def test_build_multi_model_candidates_writes_output(tmp_path):
    date = pd.Timestamp('2025-10-10')
    upside_df = pd.DataFrame(
        {
            'analysis_date': [date],
            'next_trade_date': [date],
            'code': ['1301'],
            'prediction_probability': [0.65],
        }
    )
    downside_df = pd.DataFrame(
        {
            'analysis_date': [date],
            'code': ['1301'],
            'prob_down': [0.22],
            'future_return': [-0.01],
        }
    )
    risk_df = pd.DataFrame(
        {
            'analysis_date': [date],
            'code': ['1301'],
            'risk_score': [0.35],
        }
    )

    upside_path = tmp_path / 'upside.parquet'
    downside_path = tmp_path / 'downside.parquet'
    risk_path = tmp_path / 'risk.parquet'
    upside_df.to_parquet(upside_path, index=False)
    downside_df.to_parquet(downside_path, index=False)
    risk_df.to_parquet(risk_path, index=False)

    output_path = tmp_path / 'combined.parquet'
    combined = build_multi_model_candidates(
        date=str(date.date()),
        output=output_path,
        upside_path=upside_path,
        downside_path=downside_path,
        risk_path=risk_path,
    )

    assert output_path.exists()
    result_df = pd.read_parquet(output_path)
    assert len(result_df) == 1
    assert 'prob_down' in result_df.columns
    assert 'risk_score' in result_df.columns
    assert 'future_return' in result_df.columns
    assert combined.equals(result_df)
