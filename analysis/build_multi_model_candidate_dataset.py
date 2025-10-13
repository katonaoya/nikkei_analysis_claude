#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate historical multi-model candidate dataset for threshold tuning."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd

from utils.market_calendar import JapanMarketCalendar
from reports.daily_stock_recommendation_close_v1 import DailyStockRecommendationCloseV1
from reports.daily_stock_recommendation_multi import prepare_candidate_scores
from systems.downside_risk_system_v1 import DownsideRiskSystemV1


def compute_future_returns(stock_df: pd.DataFrame, horizon_days: int = 1) -> pd.DataFrame:
    df = stock_df[['Code', 'Date', 'Close']].copy()
    df['Code'] = df['Code'].astype(str).str.zfill(4)
    df['Date'] = pd.to_datetime(df['Date']).dt.normalize()
    df.sort_values(['Code', 'Date'], inplace=True)
    df['Future_Close'] = df.groupby('Code')['Close'].shift(-horizon_days)
    df['future_return'] = df['Future_Close'] / df['Close'] - 1
    df = df.dropna(subset=['future_return'])
    return df[['Code', 'Date', 'future_return']].rename(columns={'Code': 'code', 'Date': 'analysis_date'})


def merge_candidate_frames(
    upside_df: pd.DataFrame,
    downside_df: pd.DataFrame,
    risk_df: pd.DataFrame,
    returns_df: pd.DataFrame,
) -> pd.DataFrame:
    base = upside_df.copy()
    if base.empty:
        return base

    for df in (base, downside_df, risk_df, returns_df):
        if not df.empty and 'analysis_date' in df:
            df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
        if not df.empty and 'code' in df:
            df['code'] = df['code'].astype(str).str.zfill(4)

    merged = base.merge(downside_df[['analysis_date', 'code', 'prob_down']], on=['analysis_date', 'code'], how='left')
    merged = merged.merge(risk_df[['analysis_date', 'code', 'risk_score']], on=['analysis_date', 'code'], how='left')
    merged = merged.merge(returns_df, on=['analysis_date', 'code'], how='left')
    merged.rename(columns={'prob_down': 'downside_probability'}, inplace=True)
    merged['downside_probability'] = merged['downside_probability'].fillna(0.0)
    merged['risk_score'] = merged['risk_score'].fillna(0.0)
    return merged


def iter_market_days(start_date: pd.Timestamp, end_date: pd.Timestamp) -> Iterable[pd.Timestamp]:
    current = start_date
    while current <= end_date:
        if JapanMarketCalendar.is_market_open(current):
            yield pd.Timestamp(current).normalize()
        current += timedelta(days=1)


def determine_date_range(
    start_date: Optional[str],
    end_date: Optional[str],
    lookback_days: int,
) -> List[pd.Timestamp]:
    end_dt = pd.to_datetime(end_date) if end_date else pd.to_datetime(JapanMarketCalendar.get_target_date_for_analysis())
    if start_date:
        start_dt = pd.to_datetime(start_date)
        return list(iter_market_days(start_dt, end_dt))

    collected: List[pd.Timestamp] = []
    current = end_dt
    while len(collected) < lookback_days and current >= end_dt - timedelta(days=365):
        if JapanMarketCalendar.is_market_open(current):
            collected.append(pd.Timestamp(current).normalize())
        current -= timedelta(days=1)
    collected.sort()
    return collected


@dataclass
class CandidateBuilderConfig:
    start_date: Optional[str]
    end_date: Optional[str]
    lookback_days: int
    max_candidates: int
    output_path: Path
    temp_dir: Path
    retrain_first: bool
    config_path: Path
    append_output: bool
    down_thresholds: Optional[List[float]]


class MultiModelCandidateBuilder:
    def __init__(
        self,
        *,
        config: CandidateBuilderConfig,
        close_system: Optional[DailyStockRecommendationCloseV1] = None,
        downside_system: Optional[DownsideRiskSystemV1] = None,
    ) -> None:
        self.config = config
        self.output_path = config.output_path
        self.temp_dir = config.temp_dir
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.temp_production_dir = self.temp_dir / 'production'
        self.temp_production_dir.mkdir(parents=True, exist_ok=True)

        self.close_system = close_system or DailyStockRecommendationCloseV1()
        stock_file_path = getattr(self.close_system, 'stock_file', None)
        self.downside_system = downside_system or DownsideRiskSystemV1(
            stock_file=str(stock_file_path) if stock_file_path else None,
            production_dir=str(self.temp_production_dir),
            down_thresholds=config.down_thresholds,
        )

        try:
            if config.config_path.exists():
                config_dict = json.loads(config.config_path.read_text())
            else:
                config_dict = {}
        except Exception:
            config_dict = {}

        thresholds_cfg = config_dict.get('thresholds', {})
        weights_cfg = config_dict.get('weights', {})
        self.thresholds = {
            'up': float(thresholds_cfg.get('up', 0.44)),
            'down': float(thresholds_cfg.get('down', 0.30)),
            'risk': float(thresholds_cfg.get('risk', 0.60)),
        }
        self.weights = {
            'up': float(weights_cfg.get('up', 1.0)),
            'down': float(weights_cfg.get('down', 0.6)),
            'risk': float(weights_cfg.get('risk', 0.4)),
        }

    def build(self) -> Optional[pd.DataFrame]:
        date_list = determine_date_range(
            self.config.start_date,
            self.config.end_date,
            self.config.lookback_days,
        )

        if not date_list:
            print("⚠️ 対象期間に営業日が見つかりません")
            return None

        stock_df = self.downside_system._load_stock_data()
        if stock_df.empty:
            print("⚠️ 株価データが読み込めませんでした")
            return None

        returns_df = compute_future_returns(stock_df)

        collected: List[pd.DataFrame] = []
        retrain_flag = self.config.retrain_first

        for date in date_list:
            date_str = date.strftime('%Y-%m-%d')
            upside_df = self.close_system.predict_all_candidates(date_str)
            if upside_df.empty:
                continue

            self.downside_system.run(
                predict_date=date_str,
                retrain=retrain_flag,
            )
            retrain_flag = False

            downside_path = self.temp_production_dir / 'downside_predictions.parquet'
            risk_path = self.temp_production_dir / 'risk_predictions.parquet'
            downs_df = pd.read_parquet(downside_path) if downside_path.exists() else pd.DataFrame(columns=['analysis_date', 'code', 'prob_down'])
            risks_df = pd.read_parquet(risk_path) if risk_path.exists() else pd.DataFrame(columns=['analysis_date', 'code', 'risk_score'])

            merged = merge_candidate_frames(upside_df, downs_df, risks_df, returns_df)
            if merged.empty:
                continue

            merged = merged.sort_values('prediction_probability', ascending=False)
            merged = merged.head(self.config.max_candidates)
            collected.append(merged)

        if not collected:
            print("⚠️ 収集できた候補がありませんでした")
            return None

        result_df = pd.concat(collected, ignore_index=True)
        result_df['analysis_date'] = pd.to_datetime(result_df['analysis_date']).dt.normalize()
        result_df = result_df.drop_duplicates(['analysis_date', 'code'])
        try:
            result_df = prepare_candidate_scores(result_df, self.thresholds, self.weights)
        except Exception:
            pass

        if self.config.append_output and self.output_path.exists():
            try:
                existing_df = pd.read_parquet(self.output_path)
                existing_df['analysis_date'] = pd.to_datetime(existing_df['analysis_date']).dt.normalize()
                combined_df = pd.concat([existing_df, result_df], ignore_index=True)
                sort_cols = ['analysis_date']
                ascending_flags = [True]
                if 'composite_score' in combined_df.columns:
                    sort_cols.append('composite_score')
                    ascending_flags.append(False)
                combined_df = combined_df.sort_values(sort_cols, ascending=ascending_flags)
                combined_df = combined_df.drop_duplicates(['analysis_date', 'code'], keep='first')
                if not self.config.start_date and self.config.lookback_days:
                    unique_dates = sorted(combined_df['analysis_date'].unique())
                    if len(unique_dates) > self.config.lookback_days:
                        keep_dates = set(unique_dates[-self.config.lookback_days:])
                        combined_df = combined_df[combined_df['analysis_date'].isin(keep_dates)]
                result_df = combined_df.reset_index(drop=True)
            except Exception:
                pass

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_parquet(self.output_path, index=False)
        print(f"📄 候補データを出力しました: {self.output_path}")
        return result_df


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="マルチモデル候補データセット生成")
    parser.add_argument('--start-date', type=str, default=None, help='分析開始日 (YYYY-MM-DD)')
    parser.add_argument('--end-date', type=str, default=None, help='分析終了日 (YYYY-MM-DD)')
    parser.add_argument('--lookback-days', type=int, default=20, help='開始日未指定時の営業日取得数')
    parser.add_argument('--max-candidates', type=int, default=200, help='1日あたりの候補上限')
    parser.add_argument('--output', type=str, default='production_data/multi_model_candidates.parquet', help='出力ファイルパス')
    parser.add_argument('--config', type=str, default='config/multi_model_recommendation.json', help='しきい値/重みの参照設定ファイル')
    parser.add_argument('--temp-dir', type=str, default='tmp/multi_candidate_builder', help='中間生成物を保存する一時ディレクトリ')
    parser.add_argument('--no-retrain-first', action='store_true', help='最初の実行時も学習をスキップする')
    parser.add_argument('--append-output', action='store_true', help='既存の出力に追記し、重複を解消する')
    parser.add_argument('--down-thresholds', type=str, help='下落モデルに渡す閾値 (カンマ区切り)')
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = parse_args(argv)
    threshold_list = None
    if args.down_thresholds:
        threshold_list = [float(chunk.strip()) for chunk in args.down_thresholds.split(',') if chunk.strip()]
    config = CandidateBuilderConfig(
        start_date=args.start_date,
        end_date=args.end_date,
        lookback_days=max(args.lookback_days, 1),
        max_candidates=max(args.max_candidates, 1),
        output_path=Path(args.output),
        temp_dir=Path(args.temp_dir),
        retrain_first=not args.no_retrain_first,
        config_path=Path(args.config),
        append_output=bool(args.append_output),
        down_thresholds=threshold_list,
    )

    builder = MultiModelCandidateBuilder(config=config)
    builder.build()


if __name__ == '__main__':
    main()
