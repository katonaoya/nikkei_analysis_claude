#!/usr/bin/env python3
"""Yahoo!ニュース RSS fetcher and feature builder."""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

LOGGER = logging.getLogger(__name__)

DEFAULT_FEEDS = [
    "https://news.yahoo.co.jp/rss/topics/business.xml",
    "https://news.yahoo.co.jp/rss/topics/economy.xml",
    "https://news.yahoo.co.jp/rss/topics/markets.xml",
    "https://news.yahoo.co.jp/rss/topics/it.xml",
]
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; NewsFeatureBot/1.0)"

POSITIVE_KEYWORDS = ["上方修正", "増益", "最高益", "好調", "高成長", "増配", "黒字"]
NEGATIVE_KEYWORDS = ["下方修正", "減益", "赤字", "不祥事", "リコール", "減配", "損失"]
MACRO_KEYWORDS = ["日銀", "FRB", "金利", "物価", "為替", "景気", "インフレ", "GDP"]


@dataclass
class NewsFetchResult:
    raw: pd.DataFrame
    features: pd.DataFrame
    raw_path: Optional[Path] = None
    feature_path: Optional[Path] = None


class YahooNewsFetcher:
    """Fetch Yahoo! news RSS feeds and derive daily features."""

    def __init__(
        self,
        feeds: Iterable[str] = DEFAULT_FEEDS,
        output_dir: Path = Path("data/external/news_headlines"),
        session: Optional[requests.Session] = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self.feeds = list(feeds)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", user_agent)

    def fetch_feed(self, url: str) -> List[dict]:  # pragma: no cover - network
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        items = []
        for item in root.findall("channel/item"):
            items.append(
                {
                    "title": item.findtext("title"),
                    "link": item.findtext("link"),
                    "pubDate": item.findtext("pubDate"),
                    "guid": item.findtext("guid"),
                    "description": item.findtext("description"),
                    "feed_url": url,
                }
            )
        return items

    @staticmethod
    def _parse_pubdate(pubdate: str) -> dt.datetime:
        try:
            return pd.to_datetime(pubdate).to_pydatetime()
        except Exception:
            return dt.datetime.utcnow()

    def fetch_all(self) -> pd.DataFrame:  # pragma: no cover - network
        items: List[dict] = []
        for url in self.feeds:
            try:
                items.extend(self.fetch_feed(url))
            except Exception as exc:
                LOGGER.warning("Failed to fetch feed %s: %s", url, exc)
        if not items:
            raise RuntimeError("No news items fetched")
        df = pd.DataFrame(items)
        df["pubDate"] = df["pubDate"].apply(self._parse_pubdate)
        df["fetched_at"] = dt.datetime.utcnow()
        return df

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    @staticmethod
    def _count_keywords(text: str, keywords: Iterable[str]) -> int:
        if not isinstance(text, str):
            return 0
        return sum(1 for kw in keywords if kw in text)

    @staticmethod
    def build_daily_features(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            raise ValueError("news dataframe is empty")

        df = df.copy()
        df["Date"] = pd.to_datetime(df["pubDate"]).dt.tz_localize(None).dt.date
        df["title_len"] = df["title"].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df["description_len"] = df["description"].apply(lambda x: len(x) if isinstance(x, str) else 0)
        df["positive_hits"] = df["title"].apply(lambda x: YahooNewsFetcher._count_keywords(x, POSITIVE_KEYWORDS))
        df["negative_hits"] = df["title"].apply(lambda x: YahooNewsFetcher._count_keywords(x, NEGATIVE_KEYWORDS))
        df["macro_hits"] = df["title"].apply(lambda x: YahooNewsFetcher._count_keywords(x, MACRO_KEYWORDS))
        df["source_domain"] = df["link"].apply(
            lambda url: urlparse(url).netloc if isinstance(url, str) else ""
        )

        # Text embeddings (TF-IDF + SVD)
        combined_text = (
            df["title"].fillna("") + " " + df["description"].fillna("")
        ).tolist()
        if combined_text:
            vectorizer = TfidfVectorizer(max_features=512)
            tfidf_matrix = vectorizer.fit_transform(combined_text)
            n_features = tfidf_matrix.shape[1]
            n_samples = tfidf_matrix.shape[0]
            max_components = max(1, min(n_features, n_samples - 1 if n_samples > 1 else 1, 5))
            n_components = max_components
            if n_components >= 1:
                reducer = TruncatedSVD(n_components=n_components, random_state=42)
                embeddings = reducer.fit_transform(tfidf_matrix)
                for i in range(n_components):
                    df[f"embedding_{i+1}"] = embeddings[:, i]
            else:
                df["embedding_1"] = 0.0
        else:  # pragma: no cover - safety
            df["embedding_1"] = 0.0

        agg = (
            df.groupby("Date")
            .agg(
                news_count=("title", "count"),
                news_unique_sources=("source_domain", pd.Series.nunique),
                news_pos_hits=("positive_hits", "sum"),
                news_neg_hits=("negative_hits", "sum"),
                news_macro_hits=("macro_hits", "sum"),
                news_title_len_avg=("title_len", "mean"),
                news_description_len_avg=("description_len", "mean"),
            )
            .reset_index()
        )
        embedding_cols = [col for col in df.columns if col.startswith("embedding_")]
        if embedding_cols:
            emb_agg = df.groupby("Date")[embedding_cols].mean().reset_index()
            emb_agg = emb_agg.rename(columns={col: f"news_{col}" for col in embedding_cols})
            agg = agg.merge(emb_agg, on="Date", how="left")
        # Sentiment proxies
        agg["news_sentiment_score"] = agg["news_pos_hits"] - agg["news_neg_hits"]
        agg["Date"] = pd.to_datetime(agg["Date"])
        return agg

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def _save(self, df: pd.DataFrame, prefix: str, suffix: str = "parquet") -> Path:
        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"{prefix}_{timestamp}.{suffix}"
        if suffix == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return path

    def fetch_and_persist(self) -> NewsFetchResult:  # pragma: no cover - network
        raw = self.fetch_all()
        features = self.build_daily_features(raw)
        raw_path = self._save(raw, "news_raw")
        feature_path = self._save(features, "news_features")
        return NewsFetchResult(raw, features, raw_path, feature_path)


def load_latest_news_feature(directory: Path) -> Optional[Path]:
    directory = Path(directory)
    if not directory.exists():
        return None
    files = sorted(directory.glob("news_features_*.parquet"))
    return files[-1] if files else None
