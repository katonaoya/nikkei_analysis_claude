from pathlib import Path
import sys

import json
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.multi_model_optuna_search import main as optuna_main


def _build_candidates() -> pd.DataFrame:
    dates = pd.date_range('2025-01-01', periods=5, freq='B')
    records = []
    for date in dates:
        for idx, code in enumerate(['1001', '1002', '1003'], start=1):
            records.append(
                {
                    'analysis_date': date,
                    'code': code,
                    'prediction_probability': 0.2 + idx * 0.1,
                    'downside_probability': 0.1 * idx,
                    'risk_score': 0.05 * idx,
                    'future_return': 0.015 * idx if idx % 2 else -0.01,
                }
            )
    return pd.DataFrame(records)


def test_optuna_search_generates_trials(tmp_path):
    candidates = _build_candidates()
    candidate_path = tmp_path / 'candidates.parquet'
    candidates.to_parquet(candidate_path, index=False)

    config = {
        'top_n': 3,
        'thresholds': {'up': 0.18, 'down': 0.5, 'risk': 0.4},
        'upside': {'target_return': 0.01, 'max_per_sector': 3},
        'fallback': {'require_passed_all': True}
    }
    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config))

    output_path = tmp_path / 'trials.csv'
    optuna_main([
        '--input', str(candidate_path),
        '--config', str(config_path),
        '--trials', '2',
        '--weights', 'precision:0.5,coverage:0.3,fallback:0.2',
        '--output', str(output_path),
        '--penalty', '0.1',
    ])

    assert output_path.exists()
    df = pd.read_csv(output_path)
    assert not df.empty
    assert {'precision', 'coverage', 'fallback_ratio'}.issubset(df.columns)
