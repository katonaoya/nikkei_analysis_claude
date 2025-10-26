from pathlib import Path

import pandas as pd

from data_management.highfreq_market_feature_builder import HighFreqMarketFeatureBuilder


def test_build_features_from_sample_intraday(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text((Path(__file__).parent / "data" / "sample_highfreq_intraday.csv").read_text(), encoding="utf-8")
    df = pd.read_csv(csv_path)
    builder = HighFreqMarketFeatureBuilder(symbols={"TEST": "test"}, output_dir=tmp_path)
    features = builder.build_features({"TEST": df.assign(symbol="TEST")})
    assert "hf_test_close_return" in features.columns
    assert "hf_test_volume_ratio_last30" in features.columns
    assert len(features) == 1
