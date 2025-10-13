import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from systems.downside_risk_system_v1 import DownsideRiskSystemV1


def _build_stock_prices() -> pd.DataFrame:
    dates = pd.date_range('2025-09-29', periods=20, freq='B')
    records = []
    for idx, code in enumerate(['1301', '1302', '1303'], start=1):
        base = 100 + idx * 10
        for offset, date in enumerate(dates):
            price = base + np.sin(offset / 2) * (idx * 0.8)
            records.append(
                {
                    'Code': code,
                    'Date': date,
                    'Open': price - 0.5,
                    'High': price + 0.8,
                    'Low': price - 0.9,
                    'Close': price,
                    'Volume': 100000 + offset * 50,
                }
            )
    return pd.DataFrame(records)


def test_downside_risk_system_generates_outputs(tmp_path):
    stock_df = _build_stock_prices()
    stock_path = tmp_path / 'stock.parquet'
    stock_df.to_parquet(stock_path, index=False)

    models_dir = tmp_path / 'models'
    production_dir = tmp_path / 'production'

    system = DownsideRiskSystemV1(
        stock_file=str(stock_path),
        down_threshold=-0.01,
        horizon_days=1,
        models_dir=models_dir,
        production_dir=production_dir,
    )

    predict_date = stock_df['Date'].max().strftime('%Y-%m-%d')
    system.run(predict_date=predict_date, retrain=True)

    downside_path = production_dir / 'downside_predictions.parquet'
    risk_path = production_dir / 'risk_predictions.parquet'

    assert downside_path.exists()
    assert risk_path.exists()

    downside_df = pd.read_parquet(downside_path)
    risk_df = pd.read_parquet(risk_path)

    assert {'analysis_date', 'code', 'prob_down'}.issubset(downside_df.columns)
    assert {'analysis_date', 'code', 'risk_score'}.issubset(risk_df.columns)
    assert downside_df['prob_down'].between(0.0, 1.0).all()
    assert risk_df['risk_score'].between(0.0, 1.0).all()

    # 再実行（既存モデル利用）で例外が起きないことを確認
    system.run(predict_date=predict_date, retrain=False)
