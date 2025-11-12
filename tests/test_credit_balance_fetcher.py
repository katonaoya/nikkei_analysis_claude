import pandas as pd
from pathlib import Path

from data_management.credit_balance_fetcher import KabutanMarginFetcher


SAMPLE_HTML = Path(__file__).parent / "data" / "sample_kabutan_margin.html"


def test_parse_margin_table_returns_expected_columns():
    html = SAMPLE_HTML.read_text(encoding="utf-8")
    df = KabutanMarginFetcher.parse_margin_table(html, "7453")
    assert set(["Code", "Date", "margin_sell_balance", "margin_buy_balance", "margin_long_short_ratio"]).issubset(df.columns)
    assert df["Code"].iloc[0] == "74530"


def test_build_daily_features_forward_fills_and_creates_deltas():
    html = SAMPLE_HTML.read_text(encoding="utf-8")
    raw = KabutanMarginFetcher.parse_margin_table(html, "7453")
    raw["Date"] = pd.to_datetime(raw["Date"])
    features = KabutanMarginFetcher.build_daily_features(raw, start_date=raw["Date"].min(), end_date=raw["Date"].max())
    assert {"margin_buy_balance", "margin_sell_balance", "margin_net_balance", "margin_buy_balance_delta_5d"}.issubset(features.columns)
    # Expect daily coverage between start and end
    expected_days = (raw["Date"].max() - raw["Date"].min()).days + 1
    assert len(features) == expected_days
