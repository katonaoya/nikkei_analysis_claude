#!/usr/bin/env python3
"""CLI to refresh external data features (margin balances, news)."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path
from typing import Iterable, List

import pandas as pd

from credit_balance_fetcher import KabutanMarginFetcher, load_latest_feature_file
from news_headline_fetcher import YahooNewsFetcher, load_latest_news_feature
from highfreq_market_feature_builder import HighFreqMarketFeatureBuilder, load_latest_highfreq_feature

LOGGER = logging.getLogger("external_feature_update")


def load_codes(codes_file: Path, limit: int | None = None) -> List[str]:
    df = pd.read_csv(codes_file)
    codes = [str(code).zfill(4) for code in df["code"].tolist()]
    if limit:
        codes = codes[:limit]
    return codes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Update external data features")
    parser.add_argument(
        "--codes-file",
        type=Path,
        default=Path("data/nikkei225_codes.csv"),
        help="CSV listing target stock codes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for number of codes (debugging)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fetch even if recent feature files already exist",
    )
    parser.add_argument(
        "--hf-period",
        default="5d",
        help="Lookback period for high frequency market data",
    )
    parser.add_argument(
        "--hf-interval",
        default="5m",
        help="Interval for high frequency market data",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level",
    )
    return parser.parse_args()


def should_refresh(latest_path: Path | None, force: bool, stale_hours: int = 20) -> bool:
    if force or latest_path is None:
        return True
    age = dt.datetime.utcnow() - dt.datetime.utcfromtimestamp(latest_path.stat().st_mtime)
    return age.total_seconds() > stale_hours * 3600


def main() -> None:  # pragma: no cover - CLI glue
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))

    codes = load_codes(args.codes_file, args.limit)
    LOGGER.info("Target codes: %d", len(codes))

    margin_fetcher = KabutanMarginFetcher()
    latest_margin = load_latest_feature_file(margin_fetcher.output_dir)
    if should_refresh(latest_margin, args.force):
        LOGGER.info("Refreshing margin balance dataset")
        margin_result = margin_fetcher.fetch_and_persist(codes)
        margin_daily_path = margin_result.feature_path
    else:
        LOGGER.info("Using cached margin dataset: %s", latest_margin)
        margin_daily_path = latest_margin

    news_fetcher = YahooNewsFetcher()
    latest_news = load_latest_news_feature(news_fetcher.output_dir)
    if should_refresh(latest_news, args.force):
        LOGGER.info("Refreshing news headlines dataset")
        news_result = news_fetcher.fetch_and_persist()
        news_daily_path = news_result.feature_path
    else:
        LOGGER.info("Using cached news dataset: %s", latest_news)
        news_daily_path = latest_news

    hf_builder = HighFreqMarketFeatureBuilder()
    latest_hf = load_latest_highfreq_feature(hf_builder.output_dir)
    if should_refresh(latest_hf, args.force, stale_hours=6):
        LOGGER.info("Refreshing high frequency market dataset")
        try:
            hf_result = hf_builder.fetch_and_build(period=args.hf_period, interval=args.hf_interval)
            highfreq_path = hf_result.feature_path
        except Exception as exc:
            LOGGER.warning("High frequency fetch failed: %s", exc)
            highfreq_path = latest_hf
    else:
        LOGGER.info("Using cached high frequency dataset: %s", latest_hf)
        highfreq_path = latest_hf

    LOGGER.info("External feature update complete")
    LOGGER.info("Margin daily features: %s", margin_daily_path)
    LOGGER.info("News daily features: %s", news_daily_path)
    LOGGER.info("High frequency features: %s", highfreq_path)


if __name__ == "__main__":  # pragma: no cover
    main()
