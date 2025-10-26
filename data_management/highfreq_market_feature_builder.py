#!/usr/bin/env python3
"""Build daily features from high frequency market data."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency used in prod runs
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

LOGGER = logging.getLogger(__name__)

SYMBOLS_DEFAULT = {
    "^N225": "nikkei225",
    "^TOPX": "topix",
    "USDJPY=X": "usdjpy",
    "NIY=F": "nikkei_futures",
}


@dataclass
class HighFreqResult:
    raw: Dict[str, pd.DataFrame]
    features: pd.DataFrame
    feature_path: Optional[Path] = None


class HighFreqMarketFeatureBuilder:
    """Fetch and aggregate 5-minute market information."""

    def __init__(
        self,
        symbols: Dict[str, str] | None = None,
        output_dir: Path = Path("data/external/highfreq_market"),
    ) -> None:
        self.symbols = symbols or SYMBOLS_DEFAULT
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Fetch utilities
    # ------------------------------------------------------------------
    def fetch_symbol(self, symbol: str, period: str = "5d", interval: str = "5m") -> pd.DataFrame:
        if yf is None:
            raise RuntimeError("yfinance is required to fetch high frequency data")
        interval_candidates = [interval]
        for candidate in ["15m", "30m", "60m", "90m", "1h"]:
            if candidate not in interval_candidates:
                interval_candidates.append(candidate)
        period_defaults = {
            "1m": "7d",
            "5m": "5d",
            "15m": "30d",
            "30m": "60d",
            "60m": "730d",
            "90m": "730d",
            "1h": "730d",
        }
        last_exc: Optional[Exception] = None
        for candidate in interval_candidates:
            try:
                period_candidate = period_defaults.get(candidate, period)
                LOGGER.info("Fetching %s (period=%s, interval=%s)", symbol, period_candidate, candidate)
                data = yf.download(symbol, period=period_candidate, interval=candidate, progress=False, auto_adjust=False)
                if data.empty:
                    continue
                df = data.reset_index().rename(columns={"Datetime": "timestamp"})
                df["symbol"] = symbol
                return df
            except Exception as exc:  # pragma: no cover - network failures
                last_exc = exc
                continue
        raise RuntimeError(f"No data returned for {symbol}" + (f" (last error: {last_exc})" if last_exc else ""))

    def fetch_all(self, period: str = "5d", interval: str = "5m") -> Dict[str, pd.DataFrame]:  # pragma: no cover - network
        frames: Dict[str, pd.DataFrame] = {}
        for symbol in self.symbols:
            try:
                frames[symbol] = self.fetch_symbol(symbol, period=period, interval=interval)
            except Exception as exc:
                LOGGER.warning("Failed to fetch %s: %s", symbol, exc)
        if not frames:
            raise RuntimeError("No high frequency data fetched")
        return frames

    # ------------------------------------------------------------------
    # Feature engineering helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _prepare(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df.sort_values("timestamp", inplace=True)
        return df

    @staticmethod
    def _build_features_for_symbol(df: pd.DataFrame, alias: str) -> pd.DataFrame:
        df = HighFreqMarketFeatureBuilder._prepare(df)
        df["date"] = df["timestamp"].dt.date
        feature_rows: List[pd.DataFrame] = []

        for date, group in df.groupby("date"):
            group = group.copy()
            group.sort_values("timestamp", inplace=True)
            group["return"] = group["Close"].pct_change().fillna(0.0)

            daily_volume = group["Volume"].sum()
            last_6 = group.tail(6)
            volume_last_30 = last_6["Volume"].sum()
            open_price = group["Open"].iloc[0]
            close_price = group["Close"].iloc[-1]
            high_price = group["High"].max()
            low_price = group["Low"].min()

            data = {
                "Date": pd.to_datetime(date),
                f"hf_{alias}_close_return": (close_price - open_price) / open_price if open_price else np.nan,
                f"hf_{alias}_range": (high_price - low_price) / open_price if open_price else np.nan,
                f"hf_{alias}_last30_return": last_6["Close"].iloc[-1] / last_6["Close"].iloc[0] - 1 if len(last_6) > 1 else 0.0,
                f"hf_{alias}_volatility": group["return"].std(),
                f"hf_{alias}_volume_ratio_last30": volume_last_30 / daily_volume if daily_volume else np.nan,
                f"hf_{alias}_volume_total": daily_volume,
            }
            feature_rows.append(pd.DataFrame([data]))

        result = pd.concat(feature_rows, ignore_index=True) if feature_rows else pd.DataFrame()
        return result

    def build_features(self, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        features: List[pd.DataFrame] = []
        for symbol, df in frames.items():
            alias = self.symbols.get(symbol, symbol.replace("^", "").lower())
            try:
                features.append(self._build_features_for_symbol(df, alias))
            except Exception as exc:
                LOGGER.warning("Failed to compute features for %s: %s", symbol, exc)
        if not features:
            raise RuntimeError("No high frequency features built")
        merged = features[0]
        for add in features[1:]:
            merged = merged.merge(add, on="Date", how="outer")
        merged.sort_values("Date", inplace=True)
        return merged

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save_features(self, df: pd.DataFrame) -> Path:
        timestamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        path = self.output_dir / f"highfreq_features_{timestamp}.parquet"
        df.to_parquet(path, index=False)
        LOGGER.info("Saved high frequency features to %s", path)
        return path

    def load_local_fallback(self) -> Optional[pd.DataFrame]:
        seed_file = self.output_dir / "highfreq_features_seed.parquet"
        if seed_file.exists():
            LOGGER.info("Using local seed high frequency features: %s", seed_file)
            return pd.read_parquet(seed_file)

        alt_path = Path("data/external_extended/nikkei225_10years_20250909_231806.parquet")
        if alt_path.exists():
            LOGGER.info("Building fallback high frequency proxies from %s", alt_path)
            df = pd.read_parquet(alt_path)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
            df.sort_values('Date', inplace=True)
            close = df['Close']
            close_return = (df['Close'] - df['Open']) / df['Open']
            range_ratio = (df['High'] - df['Low']) / df['Open']
            pct_change = close.pct_change().fillna(0.0)
            features = pd.DataFrame({
                'Date': df['Date'],
                'hf_nikkei225_close_return': close_return,
                'hf_nikkei225_range': range_ratio,
                'hf_nikkei225_last30_return': pct_change,
                'hf_nikkei225_volatility': pct_change.rolling(5, min_periods=1).std(),
                'hf_nikkei225_volume_ratio_last30': np.nan,
                'hf_nikkei225_volume_total': np.nan,
            })
            features['Date'] = pd.to_datetime(features['Date'])
            return features

        LOGGER.warning("No local fallback source for high frequency data found")
        return None

    def fetch_and_build(self, period: str = "5d", interval: str = "5m") -> HighFreqResult:  # pragma: no cover - network
        frames: Dict[str, pd.DataFrame] = {}
        features: Optional[pd.DataFrame] = None
        try:
            frames = self.fetch_all(period=period, interval=interval)
            features = self.build_features(frames)
        except Exception as exc:
            LOGGER.warning("High frequency fetch failed (%s). Attempting fallback.", exc)
            fallback_df = self.load_local_fallback()
            if fallback_df is None:
                raise
            features = fallback_df

        feature_path = self.save_features(features)
        return HighFreqResult(frames, features, feature_path)


def load_latest_highfreq_feature(directory: Path) -> Optional[Path]:
    directory = Path(directory)
    if not directory.exists():
        return None
    files = sorted(directory.glob("highfreq_features_*.parquet"))
    return files[-1] if files else None


def parse_args() -> argparse.Namespace:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description="Fetch high frequency market data and build features")
    parser.add_argument("--period", default="5d", help="Lookback period passed to yfinance")
    parser.add_argument("--interval", default="5m", help="Interval for yfinance download")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> None:  # pragma: no cover - CLI
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    builder = HighFreqMarketFeatureBuilder()
    builder.fetch_and_build(period=args.period, interval=args.interval)


if __name__ == "__main__":  # pragma: no cover
    main()
