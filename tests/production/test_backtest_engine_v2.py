import sys
from pathlib import Path

import matplotlib
import pandas as pd
import pytest

sys.path.append(str(Path(__file__).resolve().parents[2]))

matplotlib.use('Agg')

from production.backtest_engine_v2 import (
    BacktestEngine,
    TradeSignal,
    parse_reports,
    compute_monthly_returns,
    plot_equity_curve,
)


def _build_price_frame() -> pd.DataFrame:
    dates = pd.to_datetime(['2025-10-01', '2025-10-02', '2025-10-03', '2025-10-06'])
    data = {
        'Date': dates,
        'Code': ['123450'] * len(dates),
        'Open': [100.0, 101.0, 100.0, 100.0],
        'High': [102.0, 103.0, 104.0, 107.0],
        'Low': [99.0, 100.0, 99.0, 98.0],
        'Close': [101.0, 102.0, 103.0, 106.0],
    }
    return pd.DataFrame(data)


def test_entry_delay_days_shifts_entry_date():
    price_df = _build_price_frame()
    signal = TradeSignal(
        recommendation_date=pd.Timestamp('2025-10-01'),
        code='123450',
        company='Test Co',
        entry_price=100.0,
        probability=90.0,
    )

    engine = BacktestEngine(
        price_df=price_df,
        signals=[signal],
        position_size=100_000.0,
        profit_target=0.05,
        stop_loss=0.1,
        slippage=0.0,
        transaction_cost=0.0,
        entry_delay_days=2,
        max_holding_days=5,
    )

    equity_df, trades_df, metrics = engine.run()

    assert metrics['num_trades'] == 1
    assert trades_df.iloc[0]['entry_date'] == pd.Timestamp('2025-10-03')
    assert trades_df.iloc[0]['status'] == 'hit_target'


def test_compute_monthly_returns_extracts_correct_values():
    equity_df = pd.DataFrame(
        {
            'Date': pd.to_datetime(['2025-01-31', '2025-02-28', '2025-03-31']),
            'Equity': [1_000_000.0, 1_050_000.0, 1_000_000.0],
            'Cash': [1_000_000.0, 1_050_000.0, 1_000_000.0],
            'Exposure': [0.0, 0.0, 0.0],
            'OpenPositions': [0, 0, 0],
        }
    )

    monthly_returns = compute_monthly_returns(equity_df)

    assert list(monthly_returns['Month']) == [2, 3]
    assert monthly_returns.iloc[0]['Return'] == pytest.approx(0.05, abs=1e-6)
    assert monthly_returns.iloc[1]['Return'] == pytest.approx(-0.047619, abs=1e-6)


def test_plot_equity_curve_outputs_file(tmp_path):
    equity_df = pd.DataFrame(
        {
            'Date': pd.to_datetime(['2025-01-01', '2025-01-02', '2025-01-03']),
            'Equity': [1_000_000.0, 1_020_000.0, 1_010_000.0],
            'Cash': [1_000_000.0, 998_000.0, 990_000.0],
            'Exposure': [0.0, 22_000.0, 20_000.0],
            'OpenPositions': [0, 1, 1],
        }
    )

    output_path = tmp_path / 'equity_curve.png'
    plot_equity_curve(equity_df, output_path)

    assert output_path.exists()


def test_parse_reports_handles_section_and_table(tmp_path):
    report_text = """# Report\n\n### 1. テスト企業\n- **銘柄コード**: 1234\n- **現在価格**: 1,234円\n- **予測上昇確率**: 75.5%\n\n| 順位 | 企業名 | 銘柄コード | 現在価格 | 予測確率 | 目標価格 | 期待利益/株 |\n| 1 | テスト企業 | 1234 | 1,234円 | 75.5% | 1,300円 | +66円 |\n| 2 | テスト企業B | 5678 | 2,468円 | 65.0% | 2,600円 | +132円 |\n"""
    report_dir = tmp_path / 'reports'
    report_dir.mkdir()
    (report_dir / '2025-08-01.md').write_text(report_text, encoding='utf-8')

    signals = parse_reports(report_dir)

    assert len(signals) == 2
    codes = sorted(signal.code for signal in signals)
    assert codes == ['12340', '56780']
