import json
from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.daily_stock_recommendation_multi import MultiModelRecommendationReport
from reports.daily_stock_recommendation_multi import prepare_candidate_scores, select_top_candidates
from systems.downside_risk_system_v1 import DownsideRiskSystemV1


class StubCloseSystem:
    def __init__(self, df: pd.DataFrame, min_probability: float = 0.5, max_per_sector: int = 2) -> None:
        self._df = df
        self.min_probability = min_probability
        self.max_per_sector = max_per_sector

    def predict_all_candidates(self, target_date_str):
        return self._df.copy()


def _build_base_df(target_date: str = "2025-10-10") -> pd.DataFrame:
    analysis_date = pd.Timestamp(target_date)
    next_trade = pd.Timestamp("2025-10-13")
    data = [
        {
            "analysis_date": analysis_date,
            "next_trade_date": next_trade,
            "code": "1301",
            "company_name": "AAA",
            "sector": "Tech",
            "current_price": 1000.0,
            "target_price": 1100.0,
            "stop_loss_price": 900.0,
            "expected_return": 10.0,
            "volume": 100000.0,
            "holding_period": 1,
            "prediction_probability": prob,
        }
        for prob in (0.60, 0.50, 0.45)
    ]
    for idx, code in enumerate(("1301", "1302", "1303")):
        data[idx]["code"] = code
        data[idx]["company_name"] = f"Company{idx+1}"
        data[idx]["sector"] = "Tech" if idx < 2 else "Finance"
    return pd.DataFrame(data)


def _build_stock_prices() -> pd.DataFrame:
    dates = pd.date_range('2025-10-01', periods=12, freq='B')
    pattern = [0, 1, -1, 0, -2, -1, 0, 1, -1, 0, -1, 1]
    records = []
    for code_idx, code in enumerate(['1301', '1302', '1303'], start=1):
        base_price = 100 + code_idx * 5
        for offset, date in enumerate(dates):
            close = base_price + pattern[offset]
            records.append({
                'Code': code,
                'Date': date,
                'Open': close - 0.4,
                'High': close + 1.0,
                'Low': close - 1.2,
                'Close': close,
                'Volume': 100000 + offset * 100,
            })
    return pd.DataFrame(records)


def test_multi_model_selection_with_fallback(tmp_path):
    base_dir = tmp_path
    production_data = base_dir / "production_data"
    production_data.mkdir()

    base_df = _build_base_df()
    downside_df = pd.DataFrame(
        {
            "analysis_date": [pd.Timestamp("2025-10-10"), pd.Timestamp("2025-10-10")],
            "code": ["1301", "1302"],
            "prob_down": [0.20, 0.10],
        }
    )
    risk_df = pd.DataFrame(
        {
            "analysis_date": [pd.Timestamp("2025-10-10"), pd.Timestamp("2025-10-10"), pd.Timestamp("2025-10-10")],
            "code": ["1301", "1302", "1303"],
            "risk_score": [0.40, 0.70, 0.30],
        }
    )

    downside_path = production_data / "downside_predictions.csv"
    risk_path = production_data / "risk_predictions.csv"
    downside_df.to_csv(downside_path, index=False)
    risk_df.to_csv(risk_path, index=False)

    config = {
        "top_n": 2,
        "thresholds": {"up": 0.50, "down": 0.30, "risk": 0.50},
        "weights": {"up": 1.0, "down": 0.5, "risk": 0.3},
        "upside": {"min_probability": 0.50, "max_per_sector": 2},
        "downside": {
            "prediction_path": str(Path("production_data") / downside_path.name),
            "prob_column": "prob_down",
            "default_probability": 0.0,
        },
        "risk": {
            "prediction_path": str(Path("production_data") / risk_path.name),
            "score_column": "risk_score",
            "default_score": 0.0,
        },
        "fallback": {"require_passed_all": False, "max_fallback": 1},
        "output": {"subdir": "multi_model"},
    }

    config_path = base_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    stub_system = StubCloseSystem(base_df)
    generator = MultiModelRecommendationReport(
        str(config_path),
        base_dir=base_dir,
        close_system=stub_system,
    )

    scored = generator.get_scored_candidates("2025-10-10")
    assert {'composite_score', 'passed_up', 'passed_down', 'passed_risk'}.issubset(scored.columns)
    assert scored['analysis_date'].nunique() == 1

    report = generator.create_report("2025-10-10", top_n=2)

    assert "Company1" in report
    assert "Company3" in report  # fallback candidate
    assert "FALLBACK" in report
    assert "📊 指標サマリ" in report

    output_file = base_dir / "production_reports" / "2025-10" / "multi_model" / "2025-10-10_multi.md"
    assert output_file.exists()
    saved = output_file.read_text(encoding="utf-8")
    assert "Company1" in saved
    assert "Company3" in saved
    assert "📊 指標サマリ" in saved


def test_fallback_constraints_enforced():
    analysis_date = pd.Timestamp('2025-10-10')
    candidates = pd.DataFrame(
        [
            {
                'analysis_date': analysis_date,
                'code': '1001',
                'sector': 'Tech',
                'prediction_probability': 0.62,
                'downside_probability': 0.20,
                'risk_score': 0.35,
                'future_return': 0.02,
            },
            {
                'analysis_date': analysis_date,
                'code': '1002',
                'sector': 'Finance',
                'prediction_probability': 0.13,
                'downside_probability': 0.28,
                'risk_score': 0.45,
                'future_return': -0.01,
            },
        ]
    )

    thresholds = {'up': 0.60, 'down': 0.30, 'risk': 0.40}
    weights = {'up': 1.0, 'down': 0.5, 'risk': 0.3}
    scored = prepare_candidate_scores(candidates, thresholds, weights)

    selections = select_top_candidates(
        scored,
        top_n=2,
        max_per_sector=2,
        require_passed_all=False,
        fallback_max_fallback=1,
        fallback_min_passed_ratio=0.9,
        fallback_min_composite=-0.02,
        fallback_min_up_prob=0.14,
        fallback_risk_margin=0.02,
        fallback_block_ratio=0.6,
        risk_threshold=thresholds['risk'],
    )

    # フォールバック候補は条件未満のため除外される
    assert len(selections) == 1
    assert selections[0]['code'] == '1001'

    # 条件を緩和するとフォールバックが採用される
    relaxed = select_top_candidates(
        scored,
        top_n=2,
        max_per_sector=2,
        require_passed_all=False,
        fallback_max_fallback=1,
        fallback_min_passed_ratio=0.9,
        fallback_min_composite=-0.2,
        fallback_min_up_prob=0.10,
        fallback_risk_margin=0.10,
        fallback_block_ratio=0.6,
        risk_threshold=thresholds['risk'],
    )

    assert len(relaxed) == 2
    codes = {rec['code'] for rec in relaxed}
    assert codes == {'1001', '1002'}


def test_defaults_used_when_secondary_missing(tmp_path):
    base_df = _build_base_df()
    stub_system = StubCloseSystem(base_df)

    config = {
        "thresholds": {"up": 0.4, "down": 0.3, "risk": 0.3},
        "downside": {"default_probability": 0.25},
        "risk": {"default_score": 0.15},
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    generator = MultiModelRecommendationReport(
        str(config_path),
        base_dir=tmp_path,
        close_system=stub_system,
    )

    attached = generator._attach_secondary_scores(base_df)
    scored = generator.score_prepared_candidates(attached)
    assert scored['downside_probability'].eq(0.25).all()
    assert scored['risk_score'].eq(0.15).all()
    assert 'composite_score' in scored


def test_integration_with_generated_downside_and_risk(tmp_path):
    stock_df = _build_stock_prices()
    stock_path = tmp_path / 'stock.parquet'
    stock_df.to_parquet(stock_path, index=False)

    production_dir = tmp_path / 'production'
    models_dir = tmp_path / 'models'

    system = DownsideRiskSystemV1(
        stock_file=str(stock_path),
        down_threshold=-0.01,
        horizon_days=1,
        models_dir=models_dir,
        production_dir=production_dir,
    )

    predict_date = stock_df['Date'].max()
    system.run(predict_date=str(predict_date.date()), retrain=True)

    downside_path = production_dir / 'downside_predictions.parquet'
    risk_path = production_dir / 'risk_predictions.parquet'
    assert downside_path.exists()
    assert risk_path.exists()

    candidates_df = _build_base_df(target_date=str(predict_date.date()))
    stub_system = StubCloseSystem(candidates_df, min_probability=0.40, max_per_sector=2)

    config = {
        "top_n": 2,
        "thresholds": {"up": 0.45, "down": 0.35, "risk": 0.6},
        "weights": {"up": 1.0, "down": 0.5, "risk": 0.4},
        "upside": {"min_probability": 0.40, "max_per_sector": 2},
        "downside": {
            "prediction_path": str(Path('production') / downside_path.name),
            "prob_column": "prob_down",
            "default_probability": 0.1,
        },
        "risk": {
            "prediction_path": str(Path('production') / risk_path.name),
            "score_column": "risk_score",
            "default_score": 0.2,
        },
        "fallback": {"require_passed_all": False, "max_candidates": 50, "max_per_sector": 1},
        "output": {"subdir": "multi_model"},
    }

    config_path = tmp_path / 'config.json'
    config_path.write_text(json.dumps(config), encoding='utf-8')

    generator = MultiModelRecommendationReport(
        str(config_path),
        base_dir=tmp_path,
        close_system=stub_system,
    )

    scored_df = generator.get_scored_candidates(str(predict_date.date()))
    assert 'downside_probability' in scored_df
    assert 'risk_score' in scored_df

    report = generator.create_report(str(predict_date.date()), top_n=2)
    assert "下落リスク" in report
    assert "リスクスコア" in report

    output_file = tmp_path / 'production_reports' / predict_date.strftime('%Y-%m') / 'multi_model' / f"{predict_date.strftime('%Y-%m-%d')}_multi.md"
    assert output_file.exists()
    saved = output_file.read_text(encoding='utf-8')
    assert "FALLBACK" in saved or "PASS" in saved


def test_fallback_ratio_prevents_overfill(tmp_path):
    base_df = _build_base_df()
    base_dir = tmp_path
    production_data = base_dir / "production_data"
    production_data.mkdir()

    # Configure secondary predictions so that two candidates pass and one fails filters
    analysis_date = pd.Timestamp("2025-10-10")
    downside_df = pd.DataFrame(
        {
            "analysis_date": [analysis_date, analysis_date, analysis_date],
            "code": ["1301", "1302", "1303"],
            "prob_down": [0.20, 0.25, 0.65],
        }
    )
    risk_df = pd.DataFrame(
        {
            "analysis_date": [analysis_date, analysis_date, analysis_date],
            "code": ["1301", "1302", "1303"],
            "risk_score": [0.30, 0.35, 0.70],
        }
    )

    downside_path = production_data / "downside_predictions.csv"
    risk_path = production_data / "risk_predictions.csv"
    downside_df.to_csv(downside_path, index=False)
    risk_df.to_csv(risk_path, index=False)

    config = {
        "top_n": 3,
        "thresholds": {"up": 0.45, "down": 0.50, "risk": 0.50},
        "weights": {"up": 1.0, "down": 0.5, "risk": 0.4},
        "upside": {"min_probability": 0.40, "max_per_sector": 3},
        "downside": {
            "prediction_path": str(Path("production_data") / downside_path.name),
            "prob_column": "prob_down",
            "default_probability": 0.0,
        },
        "risk": {
            "prediction_path": str(Path("production_data") / risk_path.name),
            "score_column": "risk_score",
            "default_score": 0.0,
        },
        "fallback": {
            "require_passed_all": False,
            "max_fallback": 1,
            "min_passed_ratio": 0.5,
            "max_per_sector": 1,
        },
        "output": {"subdir": "multi_model"},
    }

    config_path = base_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    stub_system = StubCloseSystem(base_df)
    generator = MultiModelRecommendationReport(
        str(config_path),
        base_dir=base_dir,
        close_system=stub_system,
    )

    report = generator.create_report("2025-10-10", top_n=3)

    # fallback がセクター上限により制限されることを確認
    assert "Company3" not in report
    assert report.count("Company") == 2
