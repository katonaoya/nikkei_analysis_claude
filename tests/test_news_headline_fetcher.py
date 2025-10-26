import pandas as pd

from data_management.news_headline_fetcher import YahooNewsFetcher


def test_build_daily_features_extracts_sentiment():
    root_df = pd.DataFrame([{
        "title": "企業が上方修正を発表",
        "link": "https://example.com/article1",
        "pubDate": "Fri, 10 Oct 2025 09:00:00 GMT",
        "guid": "article1",
        "description": "業績が好調に推移している。",
        "feed_url": "https://example.com/rss",
    }, {
        "title": "景気減速で下方修正",
        "link": "https://example.com/article2",
        "pubDate": "Fri, 10 Oct 2025 12:00:00 GMT",
        "guid": "article2",
        "description": "マクロ環境の悪化が影響。",
        "feed_url": "https://example.com/rss",
    }])
    features = YahooNewsFetcher.build_daily_features(root_df)
    assert "news_count" in features.columns
    assert int(features.loc[0, "news_count"]) == 2
    assert "news_sentiment_score" in features.columns
    assert features.loc[0, "news_sentiment_score"] == 0  # one positive, one negative keyword
    assert any(col.startswith("news_embedding_") for col in features.columns)
