#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""マルチモデル統合推奨銘柄レポート生成スクリプト"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.market_calendar import JapanMarketCalendar
from reports.daily_stock_recommendation_close_v1 import DailyStockRecommendationCloseV1


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def prepare_candidate_scores(
    candidates: pd.DataFrame,
    thresholds: Dict[str, float],
    weights: Dict[str, float],
) -> pd.DataFrame:
    """しきい値と重みに基づき候補に評価指標を付与する"""
    df = candidates.copy()
    if df.empty:
        return df

    for col in ("prediction_probability", "downside_probability", "risk_score"):
        if col not in df:
            raise KeyError(f"候補データに必要な列がありません: {col}")

    df['passed_up'] = df['prediction_probability'] >= thresholds['up']
    df['passed_down'] = df['downside_probability'] <= thresholds['down']
    df['passed_risk'] = df['risk_score'] <= thresholds['risk']
    df['passed_all'] = df[['passed_up', 'passed_down', 'passed_risk']].all(axis=1)
    df['composite_score'] = (
        weights['up'] * df['prediction_probability']
        - weights['down'] * df['downside_probability']
        - weights['risk'] * df['risk_score']
    )

    df = df.sort_values(['analysis_date', 'composite_score'], ascending=[True, False]).reset_index(drop=True)
    return df


def select_top_candidates(
    scored_df: pd.DataFrame,
    *,
    top_n: int,
    max_per_sector: int,
    require_passed_all: bool,
    fallback_max_fallback: Optional[int] = None,
    fallback_min_passed_all: Optional[int] = None,
    fallback_min_passed_ratio: Optional[float] = None,
    fallback_max_per_sector: Optional[int] = None,
    fallback_min_composite: Optional[float] = None,
    fallback_min_up_prob: Optional[float] = None,
    fallback_risk_margin: Optional[float] = None,
    fallback_block_ratio: Optional[float] = None,
    risk_threshold: Optional[float] = None,
) -> List[Dict[str, object]]:
    """スコア付与済み候補から日付ごとに最終推奨銘柄を選定する"""
    selections: List[Dict[str, object]] = []

    def _as_float(value: Optional[float]) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    fallback_min_composite = _as_float(fallback_min_composite)
    fallback_min_up_prob = _as_float(fallback_min_up_prob)
    fallback_risk_margin = _as_float(fallback_risk_margin)
    fallback_block_ratio = _as_float(fallback_block_ratio)
    risk_threshold = _as_float(risk_threshold)

    for analysis_date, group in scored_df.groupby('analysis_date', sort=True):
        if group.empty:
            continue

        day_selections: List[Dict[str, object]] = []
        sector_limits: Dict[str, int] = {}
        fallback_sector_limits: Dict[str, int] = {}

        def day_try_append(row: pd.Series) -> bool:
            sector = row.get('sector', 'Unknown') or 'Unknown'
            current = sector_limits.get(sector, 0)
            if current >= max_per_sector:
                return False
            if any(sel['code'] == row['code'] for sel in day_selections):
                return False
            entry = {
                'analysis_date': analysis_date,
                'next_trade_date': row.get('next_trade_date'),
                'code': row.get('code'),
                'company_name': row.get('company_name'),
                'sector': row.get('sector', 'Unknown'),
                'current_price': float(row.get('current_price', np.nan)),
                'target_price': float(row.get('target_price', np.nan)),
                'stop_loss_price': float(row.get('stop_loss_price', np.nan)),
                'expected_return': float(row.get('expected_return', np.nan)),
                'volume': float(row.get('volume', np.nan)) if not pd.isna(row.get('volume')) else np.nan,
                'holding_period': row.get('holding_period', 1),
                'prediction_probability': float(row.get('prediction_probability', np.nan)),
                'downside_probability': float(row.get('downside_probability', np.nan)),
                'risk_score': float(row.get('risk_score', np.nan)),
                'composite_score': float(row.get('composite_score', np.nan)),
                'passed_up': bool(row.get('passed_up', False)),
                'passed_down': bool(row.get('passed_down', False)),
                'passed_risk': bool(row.get('passed_risk', False)),
                'passed_all_filters': bool(row.get('passed_all', False)),
                'future_return': float(row['future_return']) if 'future_return' in row.index and not pd.isna(row['future_return']) else np.nan,
            }
            day_selections.append(entry)
            sector_limits[sector] = current + 1
            return True

        # 閾値を満たす銘柄を優先
        for _, row in group[group['passed_all']].iterrows():
            if len(day_selections) >= top_n:
                break
            day_try_append(row)

        passed_all_count = len(day_selections)
        remaining = top_n - passed_all_count
        fallback_limit = None
        if fallback_max_fallback is not None and fallback_max_fallback >= 0:
            fallback_limit = int(fallback_max_fallback)

        allow_fallback = False
        if remaining > 0:
            if not require_passed_all:
                allow_fallback = True
                if fallback_limit is None:
                    fallback_limit = remaining
            else:
                target_min = fallback_min_passed_all if fallback_min_passed_all is not None else 0
                if (fallback_limit is not None and fallback_limit > 0) and passed_all_count < max(top_n, target_min):
                    allow_fallback = True

        if fallback_min_passed_all is not None and passed_all_count >= fallback_min_passed_all:
            if require_passed_all:
                allow_fallback = False

        if fallback_min_passed_ratio is not None and top_n > 0:
            try:
                ratio_threshold = float(fallback_min_passed_ratio)
            except (TypeError, ValueError):
                ratio_threshold = None
            if ratio_threshold is not None:
                current_ratio = passed_all_count / float(top_n)
                if current_ratio >= ratio_threshold:
                    allow_fallback = False

        if allow_fallback and fallback_block_ratio is not None and top_n > 0 and fallback_block_ratio >= 0:
            required = int(math.ceil(top_n * fallback_block_ratio))
            if required > 0 and passed_all_count >= required:
                allow_fallback = False

        fallback_allowance = fallback_limit
        if allow_fallback:
            if fallback_allowance is None and not require_passed_all:
                fallback_allowance = remaining
            fallback_used = 0
            if fallback_allowance is not None and fallback_allowance > 0:
                for _, row in group.iterrows():
                    if len(day_selections) >= top_n:
                        break
                    if row.get('passed_all'):
                        continue
                    if fallback_used >= fallback_allowance:
                        break
                    if fallback_max_per_sector is not None:
                        sec = row.get('sector', 'Unknown') or 'Unknown'
                        if fallback_sector_limits.get(sec, 0) >= fallback_max_per_sector:
                            continue
                    if fallback_min_composite is not None:
                        comp = row.get('composite_score')
                        if pd.isna(comp) or float(comp) < fallback_min_composite:
                            continue
                    if fallback_min_up_prob is not None:
                        up_prob = row.get('prediction_probability')
                        if pd.isna(up_prob) or float(up_prob) < fallback_min_up_prob:
                            continue
                    if fallback_risk_margin is not None and risk_threshold is not None:
                        risk_value = row.get('risk_score')
                        if pd.isna(risk_value) or float(risk_value) > risk_threshold + fallback_risk_margin:
                            continue
                    if day_try_append(row):
                        fallback_used += 1
                        if fallback_max_per_sector is not None:
                            sec = row.get('sector', 'Unknown') or 'Unknown'
                            fallback_sector_limits[sec] = fallback_sector_limits.get(sec, 0) + 1

        selections.extend(sorted(day_selections, key=lambda x: x['composite_score'], reverse=True))

    return selections


class MultiModelRecommendationReport:
    """上昇/下落/リスクモデルを統合して推奨銘柄を生成する"""

    def __init__(
        self,
        config_path: str = "config/multi_model_recommendation.json",
        *,
        base_dir: Optional[Path] = None,
        close_system: Optional[DailyStockRecommendationCloseV1] = None,
    ) -> None:
        self.base_dir = Path(base_dir) if base_dir is not None else Path(__file__).resolve().parent.parent
        self.results_dir = self.base_dir / "production_reports"
        self.config_path = Path(config_path)
        self.config = self._load_config()

        upside_cfg = self.config.get("upside", {})
        close_kwargs: Dict[str, object] = {}
        for key in ("target_return", "imbalance_boost", "min_probability", "max_per_sector"):
            if key in upside_cfg and upside_cfg[key] is not None:
                close_kwargs[key] = upside_cfg[key]
        close_config_path = upside_cfg.get("config_path", "config/close_recommendation_config.json")
        self.close_system = close_system or DailyStockRecommendationCloseV1(config_path=close_config_path, **close_kwargs)

        self.thresholds = self._resolve_thresholds()
        self.weights = self._resolve_weights()
        self.top_n_default = int(self.config.get("top_n", 5))
        fallback_cfg = self.config.get("fallback", {})
        self.require_passed_all = bool(fallback_cfg.get("require_passed_all", True))
        self.max_candidates = int(fallback_cfg.get("max_candidates", 500))
        self.fallback_max_fallback = fallback_cfg.get("max_fallback")
        self.fallback_min_passed_all = fallback_cfg.get("min_passed_all")
        self.fallback_min_passed_ratio = fallback_cfg.get("min_passed_ratio")
        self.fallback_max_per_sector = fallback_cfg.get("max_per_sector")
        self.fallback_min_composite = fallback_cfg.get("min_composite")
        self.fallback_min_up_prob = fallback_cfg.get("min_up_probability")
        self.fallback_risk_margin = fallback_cfg.get("risk_margin")
        self.fallback_block_ratio = fallback_cfg.get("block_ratio")
        if self.fallback_max_fallback is not None:
            try:
                self.fallback_max_fallback = int(self.fallback_max_fallback)
            except (TypeError, ValueError):
                logger.warning("fallback.max_fallback は整数で指定してください")
                self.fallback_max_fallback = None
        if self.fallback_min_passed_all is not None:
            try:
                self.fallback_min_passed_all = int(self.fallback_min_passed_all)
            except (TypeError, ValueError):
                logger.warning("fallback.min_passed_all は整数で指定してください")
                self.fallback_min_passed_all = None
        if self.fallback_min_passed_ratio is not None:
            try:
                self.fallback_min_passed_ratio = float(self.fallback_min_passed_ratio)
            except (TypeError, ValueError):
                logger.warning("fallback.min_passed_ratio は数値で指定してください")
                self.fallback_min_passed_ratio = None
        if self.fallback_max_per_sector is not None:
            try:
                self.fallback_max_per_sector = int(self.fallback_max_per_sector)
            except (TypeError, ValueError):
                logger.warning("fallback.max_per_sector は整数で指定してください")
                self.fallback_max_per_sector = None
        if self.fallback_min_composite is not None:
            try:
                self.fallback_min_composite = float(self.fallback_min_composite)
            except (TypeError, ValueError):
                logger.warning("fallback.min_composite は数値で指定してください")
                self.fallback_min_composite = None
        if self.fallback_min_up_prob is not None:
            try:
                self.fallback_min_up_prob = float(self.fallback_min_up_prob)
            except (TypeError, ValueError):
                logger.warning("fallback.min_up_probability は数値で指定してください")
                self.fallback_min_up_prob = None
        if self.fallback_risk_margin is not None:
            try:
                self.fallback_risk_margin = float(self.fallback_risk_margin)
            except (TypeError, ValueError):
                logger.warning("fallback.risk_margin は数値で指定してください")
                self.fallback_risk_margin = None
        if self.fallback_block_ratio is not None:
            try:
                self.fallback_block_ratio = float(self.fallback_block_ratio)
            except (TypeError, ValueError):
                logger.warning("fallback.block_ratio は数値で指定してください")
                self.fallback_block_ratio = None

        downside_cfg = self.config.get("downside", {})
        self.default_downside = float(downside_cfg.get("default_probability", 0.0))
        self.downside_clip = downside_cfg.get("clip", [0.0, 1.0])

        risk_cfg = self.config.get("risk", {})
        self.default_risk = float(risk_cfg.get("default_score", 0.0))
        self.risk_clip = risk_cfg.get("clip", [0.0, None])

        output_cfg = self.config.get("output", {})
        self.output_subdir = output_cfg.get("subdir", "multi_model")

    def _load_config(self) -> Dict[str, object]:
        if self.config_path.exists():
            try:
                return json.loads(self.config_path.read_text())
            except Exception as exc:
                logger.warning("設定ファイル読み込みに失敗したためデフォルト値を使用します: %s", exc)
        else:
            logger.info("設定ファイル %s が見つからないためデフォルト設定を使用します", self.config_path)
        return {}

    def _resolve_thresholds(self) -> Dict[str, float]:
        thresholds = self.config.get("thresholds", {})
        return {
            "up": float(thresholds.get("up", getattr(self.close_system, "min_probability", 0.5))),
            "down": float(thresholds.get("down", 0.30)),
            "risk": float(thresholds.get("risk", 0.60)),
        }

    def _resolve_weights(self) -> Dict[str, float]:
        weights = self.config.get("weights", {})
        return {
            "up": float(weights.get("up", 1.0)),
            "down": float(weights.get("down", 0.5)),
            "risk": float(weights.get("risk", 0.3)),
        }

    def create_report(self, target_date_str: Optional[str] = None, top_n: Optional[int] = None) -> str:
        if target_date_str is None:
            target_date = JapanMarketCalendar.get_target_date_for_analysis()
            target_date = pd.to_datetime(target_date)
            target_date_str = target_date.strftime('%Y-%m-%d')
            logger.info("🗓️ 自動選択された分析対象日: %s", target_date_str)
        else:
            target_date = pd.to_datetime(target_date_str)

        next_trade_date = JapanMarketCalendar.get_next_market_day(target_date)
        if top_n is None:
            top_n = self.top_n_default

        upside_df = self.close_system.predict_all_candidates(target_date_str)
        if upside_df.empty:
            logger.warning("終値ベース候補が得られなかったためマルチモデルレポートを生成できません")
            return ""

        upside_df = upside_df.head(self.max_candidates).copy()
        upside_df = self._attach_secondary_scores(upside_df)
        scored_df = self._score_candidates(upside_df)
        selected = self._select_candidates(scored_df, top_n)

        if not selected:
            logger.warning("最終的な推奨銘柄を選定できませんでした")

        report = self._build_report(
            target_date=target_date,
            next_trade_date=next_trade_date,
            scored_df=scored_df,
            selections=selected,
            top_n=top_n,
        )
        self._persist_report(report, target_date)
        return report

    def _attach_secondary_scores(self, base_df: pd.DataFrame) -> pd.DataFrame:
        base_df = base_df.copy()
        base_df['code'] = base_df['code'].astype(str).str.zfill(4)
        base_df['analysis_date'] = pd.to_datetime(base_df['analysis_date']).dt.normalize()

        downside_df = self._load_secondary_section('downside', 'downside_probability', self.default_downside)
        if not downside_df.empty:
            base_df = base_df.merge(downside_df, on=['analysis_date', 'code'], how='left')
        if 'downside_probability' not in base_df:
            base_df['downside_probability'] = self.default_downside
        base_df['downside_probability'] = base_df['downside_probability'].fillna(self.default_downside)
        if isinstance(self.downside_clip, list) and self.downside_clip:
            lower = self.downside_clip[0] if len(self.downside_clip) > 0 else None
            upper = self.downside_clip[1] if len(self.downside_clip) > 1 else None
            base_df['downside_probability'] = base_df['downside_probability'].clip(lower=lower, upper=upper)

        risk_df = self._load_secondary_section('risk', 'risk_score', self.default_risk)
        if not risk_df.empty:
            base_df = base_df.merge(risk_df, on=['analysis_date', 'code'], how='left')
        if 'risk_score' not in base_df:
            base_df['risk_score'] = self.default_risk
        base_df['risk_score'] = base_df['risk_score'].fillna(self.default_risk)
        if isinstance(self.risk_clip, list) and self.risk_clip:
            lower = self.risk_clip[0] if len(self.risk_clip) > 0 else None
            upper = self.risk_clip[1] if len(self.risk_clip) > 1 else None
            base_df['risk_score'] = base_df['risk_score'].clip(lower=lower, upper=upper)

        return base_df

    def _load_secondary_section(self, section: str, value_key: str, default_value: float) -> pd.DataFrame:
        cfg = self.config.get(section, {})
        path_str = cfg.get('prediction_path')
        if not path_str:
            logger.info("%s予測ファイルパスが未設定のため既定値 %.3f を使用", section, default_value)
            return pd.DataFrame(columns=['analysis_date', 'code', value_key])

        path = (self.base_dir / Path(path_str)).resolve() if not Path(path_str).is_absolute() else Path(path_str)
        if not path.exists():
            logger.warning("%s予測ファイルが見つかりません: %s", section, path)
            return pd.DataFrame(columns=['analysis_date', 'code', value_key])

        try:
            if path.suffix in ('.parquet', '.pq'):
                df = pd.read_parquet(path)
            elif path.suffix in ('.csv', '.tsv'):
                sep = '\t' if path.suffix == '.tsv' else ','
                df = pd.read_csv(path, sep=sep)
            elif path.suffix in ('.json',):
                df = pd.read_json(path)
            elif path.suffix in ('.joblib', '.pkl'):
                import joblib
                loaded = joblib.load(path)
                if isinstance(loaded, pd.DataFrame):
                    df = loaded
                else:
                    df = pd.DataFrame(loaded)
            else:
                logger.warning("未対応のファイル形式のため読み込みをスキップします: %s", path)
                return pd.DataFrame(columns=['analysis_date', 'code', value_key])
        except Exception as exc:
            logger.warning("%s予測ファイルの読み込みに失敗しました: %s", section, exc)
            return pd.DataFrame(columns=['analysis_date', 'code', value_key])

        date_col = cfg.get('date_column', 'analysis_date')
        code_col = cfg.get('code_column', 'code')
        value_col = cfg.get('value_column') or cfg.get('prob_column') or cfg.get('score_column')
        if value_col is None:
            logger.warning("%s設定で値列が指定されていないため読み込みをスキップします", section)
            return pd.DataFrame(columns=['analysis_date', 'code', value_key])

        extra_columns = cfg.get('extra_columns', [])
        default_extra = []
        if section == 'downside':
            default_extra.append('future_return')
        required_cols = {date_col, code_col, value_col}
        missing_cols = required_cols - set(df.columns)
        if missing_cols:
            logger.warning("%s予測ファイルに必要な列が不足しています: %s", section, ', '.join(sorted(missing_cols)))
            return pd.DataFrame(columns=['analysis_date', 'code', value_key])

        rename_map = {date_col: 'analysis_date', code_col: 'code', value_col: value_key}
        df = df.rename(columns=rename_map)
        keep_cols = ['analysis_date', 'code', value_key]
        candidates_extra = list(dict.fromkeys(extra_columns + default_extra))
        for col in candidates_extra:
            col_name = rename_map.get(col, col)
            if col in df.columns:
                keep_cols.append(col)
            elif col_name in df.columns:
                keep_cols.append(col_name)
        if section == 'downside':
            keep_cols.extend([col for col in df.columns if col.startswith('down_target_')])
        keep_cols = list(dict.fromkeys(keep_cols))
        df = df[keep_cols].copy()
        df['analysis_date'] = pd.to_datetime(df['analysis_date']).dt.normalize()
        df['code'] = df['code'].astype(str).str.zfill(4)
        df = df.sort_values(['analysis_date', 'code'])
        df = df.drop_duplicates(['analysis_date', 'code'], keep='last')
        return df

    def _score_candidates(self, df: pd.DataFrame) -> pd.DataFrame:
        return prepare_candidate_scores(df, self.thresholds, self.weights)

    def _select_candidates(self, df: pd.DataFrame, top_n: int) -> List[Dict[str, object]]:
        return select_top_candidates(
            df,
            top_n=top_n,
            max_per_sector=getattr(self.close_system, 'max_per_sector', 3),
            require_passed_all=self.require_passed_all,
            fallback_max_fallback=self.fallback_max_fallback,
            fallback_min_passed_all=self.fallback_min_passed_all,
            fallback_min_passed_ratio=self.fallback_min_passed_ratio,
            fallback_max_per_sector=self.fallback_max_per_sector,
            fallback_min_composite=self.fallback_min_composite,
            fallback_min_up_prob=self.fallback_min_up_prob,
            fallback_risk_margin=self.fallback_risk_margin,
            fallback_block_ratio=self.fallback_block_ratio,
            risk_threshold=self.thresholds.get('risk'),
        )

    def score_prepared_candidates(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """既に外部予測を付加済みの候補にスコアを再計算する"""
        return self._score_candidates(candidates_df)

    def get_scored_candidates(self, target_date_str: Optional[str] = None) -> pd.DataFrame:
        """対象日の候補に下落・リスク情報を付与しスコアを付けて返却"""
        base_df = self.close_system.predict_all_candidates(target_date_str)
        if base_df.empty:
            return base_df
        enriched = self._attach_secondary_scores(base_df)
        return self._score_candidates(enriched)

    def _build_report(self, target_date: pd.Timestamp, next_trade_date: pd.Timestamp, scored_df: pd.DataFrame, selections: List[Dict[str, object]], top_n: int) -> str:
        passed_all_count = int(scored_df['passed_all'].sum())
        total_candidates = len(scored_df)
        metric_summary = self._summarize_metrics(scored_df, selections)

        summary_lines = [
            f"📈 日次株価予測レポート（マルチモデル統合）",
            "=====================================",
            "",
            f"📅 基準日付: {target_date.date()}",
            f"📅 推奨取引日: {next_trade_date.strftime('%Y-%m-%d')}",
            f"🏆 推奨銘柄数: {len(selections)}銘柄 (TOP {top_n})",
            f"⚙️ 上昇閾値: {self.thresholds['up']:.0%}",
            f"⚠️ 下落閾値: {self.thresholds['down']:.0%}",
            f"🌪️ リスク閾値: {self.thresholds['risk']:.2f}",
            f"🔢 候補数: {total_candidates}銘柄 / フィルタ通過: {passed_all_count}銘柄",
            "",
            f"📏 スコア重み: 上昇={self.weights['up']:.2f}, 下落={self.weights['down']:.2f}, リスク={self.weights['risk']:.2f}",
            "=====================================",
            "🎯 推奨銘柄一覧",
            "=====================================",
        ]

        if metric_summary:
            summary_lines.extend(metric_summary)

        recent_metrics = self._summarize_recent_metrics()
        if recent_metrics:
            summary_lines.extend(recent_metrics)

        if not selections:
            summary_lines.append("\n❌ 推奨条件を満たす銘柄がありませんでした。")
        else:
            for idx, rec in enumerate(selections, 1):
                filter_mark = "PASS" if rec['passed_all_filters'] else "FALLBACK"
                if np.isnan(rec['volume']):
                    volume_display = ""
                else:
                    volume_display = f"{int(rec['volume']):,}株"
                summary_lines.append(
                    (
                        f"\n{idx}位: {rec['company_name']} ({rec['code']})\n"
                        f"  🏢 セクター: {rec['sector']}\n"
                        f"  🎯 上昇確率: {rec['prediction_probability']:.1%}\n"
                        f"  ⚠️ 下落リスク: {rec['downside_probability']:.1%}\n"
                        f"  🌪️ リスクスコア: {rec['risk_score']:.2f}\n"
                        f"  🧮 統合スコア: {rec['composite_score']:.3f}\n"
                        f"  ✅ フィルタ判定: {filter_mark} (UP={'◎' if rec['passed_up'] else '×'}, DOWN={'◎' if rec['passed_down'] else '×'}, RISK={'◎' if rec['passed_risk'] else '×'})\n"
                        f"  💰 現在価格: ¥{rec['current_price']:,.0f}\n"
                        f"  📈 目標価格: ¥{rec['target_price']:,.0f} (+{rec['expected_return']:.1f}%)\n"
                        f"  📉 損切価格: ¥{rec['stop_loss_price']:,.0f} (-{rec['expected_return']:.1f}%)\n"
                        f"  📊 出来高: {volume_display}\n"
                        f"  ⏰ 推奨保有: {rec['holding_period']}日間\n"
                    )
                )

        summary_lines.extend([
            "=====================================",
            "📊 システム情報",
            "=====================================",
            "🤖 上昇モデル: Close-to-Close Precision System V1",
            "⚠️ 下落モデル: 設定ファイル参照 (値未提供時は既定値 0 を使用)",
            "🌪️ リスクモデル: 設定ファイル参照 (値未提供時は既定値 0 を使用)",
            f"📅 レポート生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])

        return "\n".join(summary_lines)

    def _persist_report(self, report: str, target_date: pd.Timestamp) -> None:
        if not report:
            return
        target_month = target_date.strftime('%Y-%m')
        output_dir = self.results_dir / target_month / self.output_subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        report_file = output_dir / f"{target_date.strftime('%Y-%m-%d')}_multi.md"
        report_file.write_text(report, encoding='utf-8')
        logger.info("📄 マルチモデルレポート保存: %s", report_file)

    def _summarize_metrics(self, scored_df: pd.DataFrame, selections: List[Dict[str, object]]) -> List[str]:
        if scored_df.empty:
            return []

        def fmt_percent(value: float) -> str:
            return f"{value*100:.1f}%" if np.isfinite(value) else "N/A"

        def fmt_decimal(value: float) -> str:
            return f"{value:.3f}" if np.isfinite(value) else "N/A"

        summary = ["", "📊 指標サマリ", "-------------------------------------"]

        summary.append(
            "  ・全候補 平均上昇確率: " + fmt_percent(float(scored_df['prediction_probability'].mean()))
        )
        summary.append(
            "  ・全候補 平均下落確率: " + fmt_percent(float(scored_df['downside_probability'].mean()))
        )
        summary.append(
            "  ・全候補 平均リスク: " + fmt_decimal(float(scored_df['risk_score'].mean()))
        )
        summary.append(
            "  ・全候補 平均統合スコア: " + fmt_decimal(float(scored_df['composite_score'].mean()))
        )

        if selections:
            sel_df = pd.DataFrame(selections)
            summary.append(
                "  ・推奨銘柄 平均上昇確率: " + fmt_percent(float(sel_df['prediction_probability'].mean()))
            )
            summary.append(
                "  ・推奨銘柄 平均下落確率: " + fmt_percent(float(sel_df['downside_probability'].mean()))
            )
            summary.append(
                "  ・推奨銘柄 平均リスク: " + fmt_decimal(float(sel_df['risk_score'].mean()))
            )
            summary.append(
                "  ・推奨銘柄 平均統合スコア: " + fmt_decimal(float(sel_df['composite_score'].mean()))
            )

            if 'future_return' in sel_df.columns and sel_df['future_return'].notna().any():
                summary.append(
                    "  ・推奨銘柄 平均翌日リターン: " + fmt_percent(float(sel_df['future_return'].dropna().mean()))
                )

            if 'passed_all_filters' in sel_df.columns:
                fallback_count = int(sel_df['passed_all_filters'].eq(False).sum())
                summary.append(f"  ・フォールバック採用件数: {fallback_count}件")

        return summary

    def _summarize_recent_metrics(self, lookback_days: int = 14) -> List[str]:
        log_path = self.base_dir / 'production_data' / 'multi_model_metrics.csv'
        if not log_path.exists():
            return []

        try:
            metrics_df = pd.read_csv(log_path, parse_dates=['analysis_date'])
        except Exception as exc:
            logger.warning("日次メトリクスログの読み込みに失敗しました: %s", exc)
            return []

        if metrics_df.empty or 'analysis_date' not in metrics_df:
            return []

        metrics_df = metrics_df.dropna(subset=['analysis_date']).sort_values('analysis_date')
        unique_dates = metrics_df['analysis_date'].unique()
        if lookback_days > 0 and len(unique_dates) > lookback_days:
            keep_dates = set(unique_dates[-lookback_days:])
            metrics_df = metrics_df[metrics_df['analysis_date'].isin(keep_dates)]

        if metrics_df.empty:
            return []

        lines = ["", f"📈 精度推移 (直近{min(lookback_days, len(unique_dates))}営業日)", "-------------------------------------"]
        tail_df = metrics_df.tail(lookback_days if lookback_days > 0 else len(metrics_df))

        for _, row in tail_df.iterrows():
            date_val = row['analysis_date']
            precision = row.get('precision')
            avg_return = row.get('avg_return')
            coverage = row.get('coverage')
            selected = int(row.get('selected_count', 0))

            lines.append(
                "  ・{date}: Precision {prec} / AvgReturn {ret} / Coverage {cov} / Picks {count}".format(
                    date=date_val.strftime('%Y-%m-%d'),
                    prec=f"{precision * 100:.1f}%" if pd.notna(precision) else "N/A",
                    ret=f"{avg_return * 100:.2f}%" if pd.notna(avg_return) else "N/A",
                    cov=f"{coverage * 100:.0f}%" if pd.notna(coverage) else "N/A",
                    count=selected,
                )
            )

        return lines if len(lines) > 3 else []


def main():
    parser = argparse.ArgumentParser(description="マルチモデル統合推奨銘柄レポート生成")
    parser.add_argument("--date", type=str, help="対象日付 (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=None, help="上位N銘柄")
    parser.add_argument("--config", type=str, default="config/multi_model_recommendation.json", help="設定ファイルパス")
    args = parser.parse_args()

    report_generator = MultiModelRecommendationReport(args.config)
    report = report_generator.create_report(args.date, args.top)
    if report:
        print(report)


if __name__ == "__main__":
    main()
