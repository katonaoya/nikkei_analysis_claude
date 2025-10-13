#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""マルチモデル用候補データセットを構築するユーティリティ"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from reports.daily_stock_recommendation_close_v1 import DailyStockRecommendationCloseV1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"入力ファイルが見つかりません: {path}")

    suffix = path.suffix.lower()
    if suffix in {'.parquet', '.pq'}:
        return pd.read_parquet(path)
    if suffix in {'.csv', '.tsv'}:
        sep = '\t' if suffix == '.tsv' else ','
        return pd.read_csv(path, sep=sep)
    if suffix in {'.json'}:
        return pd.read_json(path)
    if suffix in {'.pkl', '.pickle'}:
        return pd.read_pickle(path)
    raise ValueError(f"未対応のファイル形式です: {suffix}")


def build_candidates_dataframe(
    *,
    upside_df: pd.DataFrame,
    downside_df: Optional[pd.DataFrame] = None,
    risk_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    df = upside_df.copy()
    df['code'] = df['code'].astype(str).str.zfill(4)
    df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()

    if downside_df is not None and not downside_df.empty:
        merge_cols = ['analysis_date', 'code']
        keep_cols = [col for col in downside_df.columns if col in {'prob_down', 'downside_probability', 'future_return'}]
        keep_cols.extend([col for col in downside_df.columns if col.startswith('down_target_')])
        keep_cols = merge_cols + keep_cols
        down = downside_df[keep_cols].copy()
        down['analysis_date'] = pd.to_datetime(down['analysis_date']).dt.normalize()
        down['code'] = down['code'].astype(str).str.zfill(4)
        df = df.merge(down, on=merge_cols, how='left')
    else:
        df['prob_down'] = df.get('prob_down', 0.0)
        df['downside_probability'] = df.get('downside_probability', df['prob_down'])
        df['future_return'] = df.get('future_return', pd.NA)

    if risk_df is not None and not risk_df.empty:
        merge_cols = ['analysis_date', 'code']
        keep_cols = [col for col in risk_df.columns if col in {'risk_score'}]
        keep_cols = merge_cols + keep_cols
        risk = risk_df[keep_cols].copy()
        risk['analysis_date'] = pd.to_datetime(risk['analysis_date']).dt.normalize()
        risk['code'] = risk['code'].astype(str).str.zfill(4)
        df = df.merge(risk, on=merge_cols, how='left')
    else:
        df['risk_score'] = df.get('risk_score', 0.0)

    if 'downside_probability' not in df and 'prob_down' in df:
        df['downside_probability'] = df['prob_down']

    default_cols = {
        'prob_down': 0.0,
        'downside_probability': 0.0,
        'risk_score': 0.0,
    }
    for col, default in default_cols.items():
        if col not in df:
            df[col] = default
        df[col] = df[col].fillna(default)

    return df


def build_multi_model_candidates(
    *,
    date: Optional[str],
    output: Path,
    upside_path: Optional[Path] = None,
    downside_path: Optional[Path] = None,
    risk_path: Optional[Path] = None,
) -> pd.DataFrame:
    if upside_path is not None:
        upside_df = _load_table(upside_path)
    else:
        logger.info("⚙️ 終値モデルから候補を生成します")
        close_system = DailyStockRecommendationCloseV1()
        upside_df = close_system.predict_all_candidates(date)
        if upside_df.empty:
            raise RuntimeError("終値モデルの候補が取得できませんでした")

    downside_df = _load_table(downside_path) if downside_path and downside_path.exists() else None
    risk_df = _load_table(risk_path) if risk_path and risk_path.exists() else None

    combined = build_candidates_dataframe(
        upside_df=upside_df,
        downside_df=downside_df,
        risk_df=risk_df,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output, index=False)
    logger.info(
        "✅ 候補データセットを保存しました: %s (件数=%d, 日付=%s)",
        output,
        len(combined),
        ','.join(sorted({str(d.date()) for d in combined['analysis_date'].unique()})),
    )
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="マルチモデル候補データセットの構築")
    parser.add_argument('--date', type=str, help='対象日 (YYYY-MM-DD)。未指定の場合はシステムが自動判定')
    parser.add_argument('--output', type=str, default='production_data/multi_model_candidates.parquet', help='出力パス')
    parser.add_argument('--upside-path', type=str, help='終値モデル候補データを直接指定する場合のパス')
    parser.add_argument('--downside-path', type=str, default='production_data/downside_predictions.parquet', help='下落予測ファイル')
    parser.add_argument('--risk-path', type=str, default='production_data/risk_predictions.parquet', help='リスク予測ファイル')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_multi_model_candidates(
        date=args.date,
        output=Path(args.output),
        upside_path=Path(args.upside_path) if args.upside_path else None,
        downside_path=Path(args.downside_path) if args.downside_path else None,
        risk_path=Path(args.risk_path) if args.risk_path else None,
    )


if __name__ == '__main__':
    main()
