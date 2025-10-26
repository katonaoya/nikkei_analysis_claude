#!/usr/bin/env python3
"""Kabutan credit balance data fetcher and feature builder."""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

KABUTAN_URL_TEMPLATE = "https://kabutan.jp/stock/?code={code}"
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; MarginFeatureBot/1.0)"


def to_five_digit_code(code: str) -> str:
    """Convert 4-digit securities code to J-Quants 5-digit format."""

    stripped = str(code).strip()
    if stripped.endswith("0") and len(stripped) == 5:
        return stripped
    if stripped.isdigit() and len(stripped) <= 4:
        return stripped.zfill(4) + "0"
    return stripped


@dataclass
class MarginFetchResult:
    """Container for margin fetch outputs."""

    raw_table: pd.DataFrame
    features: pd.DataFrame
    raw_path: Optional[Path] = None
    feature_path: Optional[Path] = None


class KabutanMarginFetcher:
    """Fetch margin trading statistics from Kabutan and derive features."""

    def __init__(
        self,
        output_dir: Path = Path("data/external/margin_balances"),
        session: Optional[requests.Session] = None,
        user_agent: str = DEFAULT_USER_AGENT,
        max_retries: int = 3,
        retry_backoff: float = 1.5,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.session.headers.setdefault("User-Agent", user_agent)
        self.max_retries = max(1, max_retries)
        self.retry_backoff = max(1.0, retry_backoff)

    # ------------------------------------------------------------------
    # Fetching helpers
    # ------------------------------------------------------------------
    def _request(self, code: str) -> str:
        url = KABUTAN_URL_TEMPLATE.format(code=code)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                LOGGER.debug("Fetching margin page %s (attempt %d)", url, attempt)
                resp = self.session.get(url, timeout=10)
                resp.raise_for_status()
                return resp.text
            except Exception as exc:  # pragma: no cover - network failure
                last_exc = exc
                wait = self.retry_backoff ** attempt
                LOGGER.warning(
                    "Margin fetch failed for %s (attempt %d/%d): %s", url, attempt, self.max_retries, exc
                )
                time.sleep(wait)
        raise RuntimeError(f"Failed to fetch margin page for {code}: {last_exc}")

    @staticmethod
    def parse_margin_table(html: str, code: str) -> pd.DataFrame:
        """Extract margin table from Kabutan HTML."""

        tables = pd.read_html(html)
        for table in tables:
            if set(["日付", "売り残", "買い残", "倍率"]).issubset(table.columns):
                df = table.copy()
                df.insert(0, "Code", to_five_digit_code(code))
                df.rename(
                    columns={
                        "売り残": "margin_sell_balance",
                        "買い残": "margin_buy_balance",
                        "倍率": "margin_long_short_ratio",
                    },
                    inplace=True,
                )
                df["Date"] = pd.to_datetime(df["日付"], format="%m/%d")
                df.drop(columns=["日付"], inplace=True)
                current_year = dt.datetime.now().year

                def assign_year(ts: pd.Timestamp) -> dt.date:
                    month_now = dt.datetime.now().month
                    year = current_year if ts.month <= month_now else current_year - 1
                    return dt.date(year, ts.month, ts.day)

                df["Date"] = df["Date"].apply(assign_year)
                for col in ["margin_sell_balance", "margin_buy_balance", "margin_long_short_ratio"]:
                    df[col] = (
                        df[col]
                        .astype(str)
                        .str.replace(",", "", regex=False)
                        .str.replace("千株", "", regex=False)
                        .str.strip()
                    )
                    df[col] = pd.to_numeric(df[col], errors="coerce")
                return df
        raise ValueError("信用取引テーブルが見つかりませんでした")

    def fetch_margin_for_code(self, code: str) -> pd.DataFrame:
        html = self._request(code)
        return self.parse_margin_table(html, code)

    def fetch_margin_balances(self, codes: Iterable[str]) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for code in codes:
            try:
                frames.append(self.fetch_margin_for_code(str(code)))
            except Exception as exc:  # pragma: no cover - network dependent
                LOGGER.warning("Skipping code %s due to error: %s", code, exc)
        if not frames:
            raise RuntimeError("No margin data fetched; all requests failed")
        df = pd.concat(frames, ignore_index=True)
        df["Date"] = pd.to_datetime(df["Date"])
        df.sort_values(["Code", "Date"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------
    @staticmethod
    def build_daily_features(
        raw_df: pd.DataFrame,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
    ) -> pd.DataFrame:
        if raw_df.empty:
            raise ValueError("raw_df is empty")

        feature_frames: List[pd.DataFrame] = []
        measure_cols = [
            "margin_sell_balance",
            "margin_buy_balance",
            "margin_long_short_ratio",
        ]

        for code, group in raw_df.groupby("Code"):
            group = group.sort_values("Date").copy()
            group.set_index("Date", inplace=True)

            if start_date is None:
                start = group.index.min()
            else:
                start = pd.Timestamp(start_date)
            if end_date is None:
                end = group.index.max()
            else:
                end = pd.Timestamp(end_date)
            if start > end:
                start, end = end, start

            idx = pd.date_range(start=start, end=end, freq="D")
            daily = group.reindex(idx).ffill().reset_index().rename(columns={"index": "Date"})
            if "Code" in daily.columns:
                daily["Code"] = code
            else:
                daily.insert(0, "Code", code)
            daily["margin_net_balance"] = daily["margin_buy_balance"] - daily["margin_sell_balance"]
            # Weekly deltas (5 trading days approximation)
            for col in ["margin_buy_balance", "margin_sell_balance", "margin_net_balance"]:
                daily[f"{col}_delta_5d"] = daily[col].diff(5)
            # Normalised ratio movement
            daily["margin_ratio_change_5d"] = daily["margin_long_short_ratio"].diff(5)
            feature_frames.append(daily)

        features = pd.concat(feature_frames, ignore_index=True)
        features.sort_values(["Code", "Date"], inplace=True)
        return features

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    def _save(self, df: pd.DataFrame, prefix: str, suffix: str = "parquet") -> Path:
        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"{prefix}_{timestamp}.{suffix}"
        if suffix == "parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        return path

    def fetch_and_persist(
        self,
        codes: Iterable[str],
        *,
        start_date: Optional[dt.date] = None,
        end_date: Optional[dt.date] = None,
    ) -> MarginFetchResult:
        raw = self.fetch_margin_balances(codes)
        features = self.build_daily_features(raw, start_date=start_date, end_date=end_date)
        raw_path = self._save(raw, "margin_balances_raw")
        feature_path = self._save(features, "margin_features")
        return MarginFetchResult(raw, features, raw_path, feature_path)


def load_latest_feature_file(directory: Path) -> Optional[Path]:
    """Return the newest feature file in the directory, if any."""

    directory = Path(directory)
    if not directory.exists():
        return None
    files = sorted(directory.glob("margin_features_*.parquet"))
    return files[-1] if files else None
