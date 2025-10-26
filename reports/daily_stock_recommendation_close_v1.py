#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
終値ベース推奨銘柄システム
最新学習済みモデルの指標に同期
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from datetime import datetime, timedelta
import joblib
import argparse
import json
import sys
from typing import Dict

sys.path.append(str(Path(__file__).parent.parent))
from utils.market_calendar import JapanMarketCalendar
from systems.enhanced_close_return_system_v1 import CloseReturnPrecisionSystemV1

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class DailyStockRecommendationCloseV1:
    """終値ベース推奨銘柄システム"""

    def __init__(self, target_return: float = 0.01, imbalance_boost: float = 1.0, min_probability: float = None, max_per_sector: int = None, config_path: str = "config/close_recommendation_config.json"):
        self.model_dir = Path("models")
        self.data_dir = Path("data")
        self.results_dir = Path("production_reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.config = self._load_config(config_path)
        if target_return is None:
            target_return = self.config.get("target_return", 0.01)
        if min_probability is None:
            min_probability = self.config.get("min_probability", 0.60)
        if max_per_sector is None:
            max_per_sector = self.config.get("max_per_sector", 3)
        self.strict_min_probability = self.config.get("strict_min_probability", False)

        # モデルコンポーネント
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.model_accuracy = None
        self.model_topn_precision = None
        self.model_avg_selected = None
        self.cluster_topn_precision = None
        self.baseline_topn_precision = None
        self.model_mode = "single"
        self.ensemble_base_models = {}
        self.ensemble_meta_model = None
        self.base_model_names = []
        self.cluster_meta_models: Dict[str, Dict[str, object]] = {}
        self.code_cluster_map: Dict[str, str] = {}
        self.pipeline = CloseReturnPrecisionSystemV1(target_return=target_return, imbalance_boost=imbalance_boost)
        
        # 会社名マッピング
        self.company_names = {}
        self.company_sectors = {}
        self.calibration = None
        self.imbalance_strategy = getattr(self.pipeline, 'imbalance_strategy', 'scale_pos')
        self.focal_gamma = getattr(self.pipeline, 'focal_gamma', 2.0)
        self.positive_oversample_ratio = getattr(self.pipeline, 'positive_oversample_ratio', 1.0)
        self._load_company_names()
        self._load_close_model()
        self.target_return = target_return
        self.imbalance_boost = imbalance_boost
        self.min_probability = min_probability
        self.max_per_sector = max_per_sector
        self.last_candidates = []
        self.last_generation_stats = {}

    def _load_config(self, path: str) -> dict:
        cfg_path = Path(path)
        if cfg_path.exists():
            try:
                return json.loads(cfg_path.read_text())
            except Exception as exc:
                logger.warning(f"設定ファイル読み込み失敗: {exc}")
        return {}

    def _load_company_names(self):
        """会社名マッピングを読み込み"""
        try:
            # CSVファイルから会社名を読み込み
            csv_file = Path("docment/ユーザー情報/nikkei225_matched_companies_20250909_230026.csv")
            if csv_file.exists():
                df = pd.read_csv(csv_file, encoding='utf-8-sig')
                for _, row in df.iterrows():
                    code = str(row['target_code'])
                    name = row['target_name'].replace('（株）', '').replace('(株)', '')
                    self.company_names[code] = name
                    sector = row.get('sector') if 'sector' in row else None
                    if isinstance(sector, str) and sector:
                        self.company_sectors[code] = sector
                    else:
                        self.company_sectors[code] = 'Unknown'
                logger.info(f"✅ 会社名マッピング読み込み完了: {len(self.company_names)}社")
            else:
                logger.warning("会社名CSVファイルが見つかりません")
        except Exception as e:
            logger.error(f"会社名読み込みエラー: {e}")
    
    def _get_company_name(self, code):
        """銘柄コードから会社名を取得"""
        return self.company_names.get(str(code), f"銘柄{code}")

    def _align_feature_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [col for col in self.feature_names if col not in df.columns]
        if missing:
            for col in missing:
                df[col] = 0.0
            logger.debug("欠損特徴量 %s を0で補完", ", ".join(missing))
        return df

    @staticmethod
    def _predict_model_proba(model, features: np.ndarray) -> np.ndarray:
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)
            if isinstance(proba, tuple):
                proba = proba[0]
            return proba[:, 1]
        if hasattr(model, 'decision_function'):
            decision = model.decision_function(features)
            return 1.0 / (1.0 + np.exp(-decision))
        preds = model.predict(features)
        return preds.astype(float)

    def _predict_probabilities(self, feature_df: pd.DataFrame) -> np.ndarray:
        aligned_df = self._align_feature_columns(feature_df.copy())
        matrix = aligned_df[self.feature_names].values.astype(np.float32)

        if self.model_mode == "ensemble" and self.ensemble_base_models and self.ensemble_meta_model:
            base_outputs = []
            for name in self.base_model_names:
                model = self.ensemble_base_models.get(name)
                if model is None:
                    continue
                base_outputs.append(self._predict_model_proba(model, matrix))
            if not base_outputs:
                raise ValueError("アンサンブル用のベースモデル出力が得られませんでした")
            stacked = np.column_stack(base_outputs)
            ensemble_scores = self.ensemble_meta_model.predict_proba(stacked)[:, 1]
            if self.calibration is not None:
                ensemble_scores = self.calibration.predict(ensemble_scores)

            if self.cluster_meta_models and self.code_cluster_map:
                codes = feature_df.get('Code')
                if codes is None:
                    codes = feature_df.get('Stock')
                if codes is not None:
                    codes = codes.astype(str).tolist()
                    adjusted = ensemble_scores.copy()
                    for idx, code in enumerate(codes):
                        cluster = self.code_cluster_map.get(code)
                        if not cluster:
                            continue
                        info = self.cluster_meta_models.get(cluster)
                        if not info:
                            continue
                        model = info.get('model')
                        if model is None:
                            continue
                        prob = model.predict_proba(stacked[idx:idx+1])[:, 1]
                        calibrator = info.get('calibrator')
                        if calibrator is not None:
                            prob = calibrator.predict(prob)
                        adjusted[idx] = prob[0]
                    return adjusted

            return ensemble_scores

        features = matrix
        if self.selector is not None:
            features = self.selector.transform(features)
        if self.scaler is not None:
            features = self.scaler.transform(features)
        proba = self.model.predict_proba(features)[:, 1]
        return CloseReturnPrecisionSystemV1.apply_calibration(proba, self.calibration)
    
    def _load_close_model(self):
        """終値ベースモデルを読み込み"""
        try:
            ensemble_path = Path("models/ensemble_close_v2/latest_ensemble_model.joblib")
            if ensemble_path.exists():
                try:
                    ensemble_data = joblib.load(ensemble_path)
                    base_models = ensemble_data.get('base_models', {})
                    meta_model = ensemble_data.get('meta_model')
                    feature_cols = ensemble_data.get('feature_cols', [])

                    if base_models and meta_model and feature_cols:
                        self.model_mode = "ensemble"
                        self.ensemble_base_models = base_models
                        self.ensemble_meta_model = meta_model
                        self.base_model_names = list(base_models.keys())
                        self.feature_names = feature_cols
                        holdout_metrics = (
                            ensemble_data.get('holdout_metrics', {})
                            if isinstance(ensemble_data, dict)
                            else {}
                        )
                        ensemble_holdout = holdout_metrics.get('ensemble', {}) if isinstance(holdout_metrics, dict) else {}
                        self.model_accuracy = ensemble_holdout.get('ensemble_accuracy')
                        top_n = ensemble_data.get('top_n', 3)
                        self.model_topn_precision = ensemble_holdout.get(f'ensemble_top{top_n}_precision')
                        self.model_avg_selected = ensemble_holdout.get(f'ensemble_top{top_n}_avg_selected')
                        self.calibration = ensemble_data.get('calibrator')
                        self.cluster_meta_models = ensemble_data.get('cluster_meta_models', {})
                        self.code_cluster_map = ensemble_data.get('code_cluster_map', {})
                        self.cluster_topn_precision = ensemble_data.get('cluster_holdout_top3_precision')
                        self.baseline_topn_precision = ensemble_data.get('baseline_holdout_top3_precision')
                        self.model = None
                        self.scaler = None
                        self.selector = None
                        logger.info("✅ 終値ベースアンサンブルモデル読み込み完了: %s", ensemble_path.name)
                        logger.info("📊 アンサンブル特徴量数: %d", len(self.feature_names))
                        return
                    logger.warning("アンサンブルモデルの構造が不完全なため、従来モデルを使用します")
                except Exception as exc:
                    logger.warning("アンサンブルモデル読み込み失敗: %s", exc)

            # 終値ベースモデルファイルを探す
            model_files = list(self.model_dir.glob("enhanced_close_v1/*close_model_v1*.joblib"))
            if not model_files:
                raise FileNotFoundError("終値ベースモデルが見つかりません")
            
            # 最新の終値ベースモデルを使用
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model_data = joblib.load(latest_model)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.selector = model_data.get('selector')
            self.feature_names = model_data['feature_cols']
            self.model_accuracy = model_data.get('accuracy')
            self.calibration = model_data.get('calibration')
            model_target_return = model_data.get('target_return')
            if model_target_return is not None and abs(model_target_return - self.pipeline.target_return) > 1e-6:
                logger.info(f"target_return updated from model: {model_target_return:.4f}")
                self.pipeline.target_return = model_target_return
                self.target_return = model_target_return

            model_imbalance_boost = model_data.get('imbalance_boost')
            if model_imbalance_boost is not None:
                if abs(model_imbalance_boost - self.pipeline.imbalance_boost) > 1e-6:
                    logger.info(f"imbalance_boost updated from model: {model_imbalance_boost:.3f}")
                self.pipeline.imbalance_boost = model_imbalance_boost
                self.imbalance_boost = model_imbalance_boost

            for attr in ("imbalance_strategy", "focal_gamma", "positive_oversample_ratio"):
                model_value = model_data.get(attr)
                if model_value is not None:
                    setattr(self.pipeline, attr, model_value)
                    setattr(self, attr, model_value)

            logger.info(f"✅ 終値ベースモデル読み込み完了: {latest_model.name}")
            logger.info(f"📊 特徴量数: {len(self.feature_names)}")
            if self.model_accuracy is not None:
                logger.info(f"📈 モデル精度: {self.model_accuracy:.4f}")
            
        except Exception as e:
            logger.error(f"終値ベースモデル読み込みエラー: {e}")
            raise
    
    def _prepare_feature_frame(self, target_date: pd.Timestamp) -> pd.DataFrame:
        """学習パイプラインと同一ロジックで特徴量を取得"""
        try:
            df = self.pipeline.load_and_integrate_data()
            df['Date'] = pd.to_datetime(df['Date'])
            df = df[df['Date'] <= target_date].copy()
            logger.info(
                "✅ 特徴量データ読み込み完了: %s件 (最新: %s)",
                f"{len(df):,}",
                df['Date'].max().strftime('%Y-%m-%d') if not df.empty else 'N/A'
            )
            return df
        except Exception as e:
            logger.error(f"特徴量データ準備エラー: {e}")
            raise
    
    def generate_recommendations(self, target_date_str=None, top_n=5):
        """推奨銘柄を生成"""
        self.last_candidates = []
        self.last_generation_stats = {}
        try:
            if target_date_str is None:
                # 営業日ベースで分析対象日を決定
                target_date = JapanMarketCalendar.get_target_date_for_analysis()
                target_date_str = str(target_date)
                logger.info(f"🗓️ 自動選択された分析対象日: {target_date_str}")
            
            target_date = pd.to_datetime(target_date_str)
            next_date = JapanMarketCalendar.get_next_market_day(target_date)
            
            logger.info(f"🚀 {target_date_str}の推奨銘柄分析開始...")
            
            feature_df = self._prepare_feature_frame(target_date)
            target_data = feature_df[feature_df['Date'] == target_date].copy()
            
            if len(target_data) == 0:
                logger.warning(f"対象日 {target_date_str} のデータが見つかりません")
                return []

            logger.info(f"📊 対象日の銘柄数: {len(target_data)}銘柄")

            target_data = target_data.replace([np.inf, -np.inf], np.nan)
            target_data = target_data.ffill().fillna(0)

            probabilities = self._predict_probabilities(target_data)
            target_data = target_data.assign(prediction_probability=probabilities)
            
            recommendations = []
            fallback_candidates = []
            
            for _, row in target_data.iterrows():
                try:
                    code = row['Code']
                    prediction_proba = float(row['prediction_probability'])
                    target_return = float(getattr(self.pipeline, 'target_return', 0.01))
                    candidate = {
                        'code': code,
                        'company_name': self._get_company_name(code),
                        'prediction_probability': prediction_proba,
                        'current_price': row['Close'],
                        'volume': row['Volume'],
                        'target_price': row['Close'] * (1 + target_return),
                        'stop_loss_price': row['Close'] * (1 - target_return),
                        'expected_return': target_return * 100,
                        'holding_period': 1,
                        'sector': self.company_sectors.get(str(code), 'Unknown'),
                        'passed_threshold': prediction_proba >= self.min_probability,
                        'threshold': self.min_probability,
                    }

                    fallback_candidates.append(candidate)
                    if candidate['passed_threshold']:
                        recommendations.append(candidate)
                
                except Exception as e:
                    logger.debug(f"銘柄 {code} の予測エラー: {e}")
                    continue
            
            # 確信度でソート
            recommendations.sort(key=lambda x: x['prediction_probability'], reverse=True)

            selected = []
            sector_counts = {}

            def try_append(candidate):
                sector = candidate.get('sector', 'Unknown')
                if sector_counts.get(sector, 0) >= self.max_per_sector:
                    return False
                if any(existing['code'] == candidate['code'] for existing in selected):
                    return False
                selected.append(candidate)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                return True

            for rec in recommendations:
                if len(selected) >= top_n:
                    break
                try_append(rec)

            if len(selected) < top_n:
                fallback_sorted = sorted(
                    fallback_candidates,
                    key=lambda x: x['prediction_probability'],
                    reverse=True,
                )
                for cand in fallback_sorted:
                    if len(selected) >= top_n:
                        break
                    if self.strict_min_probability and not cand['passed_threshold']:
                        continue
                    if cand['passed_threshold']:
                        try_append(cand)

            if not self.strict_min_probability and len(selected) < top_n:
                fallback_sorted = sorted(
                    fallback_candidates,
                    key=lambda x: x['prediction_probability'],
                    reverse=True,
                )
                for cand in fallback_sorted:
                    if len(selected) >= top_n:
                        break
                    if any(existing['code'] == cand['code'] for existing in selected):
                        continue
                    selected.append(cand)

            fallback_sorted = sorted(
                fallback_candidates,
                key=lambda x: x['prediction_probability'],
                reverse=True,
            )
            limit = max(top_n, 10)
            self.last_candidates = fallback_sorted[:limit]
            self.last_generation_stats = {
                'threshold': self.min_probability,
                'top_n': top_n,
                'total_candidates': len(fallback_candidates),
                'above_threshold': sum(1 for cand in fallback_candidates if cand['passed_threshold']),
                'max_probability': fallback_sorted[0]['prediction_probability'] if fallback_sorted else None,
            }

            recommendations = selected
            
            logger.info(f"✅ 推奨銘柄生成完了: {len(recommendations)}銘柄")
            return recommendations
        
        except Exception as e:
            logger.error(f"推奨銘柄生成エラー: {e}")
            return []
    
    def create_report(self, target_date_str=None, top_n=None):
        """レポート作成"""
        if target_date_str is None:
            # 営業日ベースで分析対象日を決定
            target_date = JapanMarketCalendar.get_target_date_for_analysis()
            target_date_str = str(target_date)
            logger.info(f"🗓️ 自動選択された分析対象日: {target_date_str}")
        
        target_date = pd.to_datetime(target_date_str)
        next_date = JapanMarketCalendar.get_next_market_day(target_date)
        if top_n is None:
            top_n = self.config.get('top_n', 3)
        
        recommendations = self.generate_recommendations(target_date_str, top_n)
        
        # レポート生成
        model_accuracy_display = "N/A"
        if self.model_accuracy is not None:
            model_accuracy_display = f"{self.model_accuracy * 100:.2f}%"

        model_label = "Close-to-Close Precision Ensemble V2" if self.model_mode == "ensemble" else "Close-to-Close Precision System V1"

        topn_display = "N/A"
        if self.model_topn_precision is not None:
            topn_display = f"{self.model_topn_precision * 100:.2f}%"
        cluster_display = "N/A"
        if self.cluster_topn_precision is not None:
            cluster_display = f"{self.cluster_topn_precision * 100:.2f}%"
        baseline_display = "N/A"
        if self.baseline_topn_precision is not None:
            baseline_display = f"{self.baseline_topn_precision * 100:.2f}%"

        topn_display = "N/A"
        if self.model_topn_precision is not None:
            topn_display = f"{self.model_topn_precision * 100:.2f}%"

        report = f"""📈 日次株価予測レポート（終値ベースモデル対応）
=====================================

📅 基準日付: {target_date_str}
📅 推奨取引日: {next_date.strftime('%Y-%m-%d')}
🏆 推奨銘柄数: {len(recommendations)}銘柄 (TOP {top_n})
⚙️ モデル精度: {model_accuracy_display} ({model_label})
🎯 Top{top_n} 的中率: {topn_display} / クラスタ調整後 {cluster_display} (従来 {baseline_display})
📈 判定閾値: {getattr(self.pipeline, 'target_return', 0.01)*100:.1f}% (終値→終値)
🎯 推奨閾値: 翌営業日終値が+{getattr(self.pipeline, 'target_return', 0.01)*100:.1f}%以上になる確率 {self.min_probability*100:.0f}%以上

=====================================
🎯 推奨銘柄一覧
=====================================
"""
        
        if not recommendations:
            report += "\n❌ 推奨条件を満たす銘柄がありませんでした。\n"
            if self.strict_min_probability and self.last_candidates:
                stats = self.last_generation_stats or {}
                report += "\n-------------------------------------\n📊 閾値検証サマリー\n-------------------------------------\n"
                threshold = stats.get('threshold')
                if threshold is not None:
                    report += f"\n- 閾値: {threshold*100:.0f}%"
                else:
                    report += "\n- 閾値: N/A"
                report += f"\n- 評価銘柄数: {stats.get('total_candidates', len(self.last_candidates))}銘柄"
                report += f"\n- 閾値到達: {stats.get('above_threshold', 0)}銘柄"
                max_prob = stats.get('max_probability')
                if max_prob is not None:
                    report += f"\n- 最高確率: {max_prob*100:.1f}%"
                report += "\n\n閾値未達の上位候補:\n"
                limit = max(stats.get('top_n', 5), 10)
                for idx, cand in enumerate(self.last_candidates, 1):
                    note = "閾値達成" if cand.get('passed_threshold') else "閾値未達"
                    report += f"\n{idx}位: {cand['company_name']} ({cand['code']})\n"
                    report += f"  🎯 予測確率: {cand['prediction_probability']*100:.1f}% ({note})\n"
                    report += f"  📏 閾値: {self.min_probability*100:.0f}%\n"
                    report += f"  💰 現在価格: ¥{cand['current_price']:,.0f}\n"
                    report += f"  📊 出来高: {cand['volume']:,}株\n"
                    report += f"  🏢 セクター: {cand.get('sector', 'Unknown')}\n"
                    if idx >= limit:
                        break
        else:
            for i, rec in enumerate(recommendations, 1):
                note = ""
                if not rec.get('passed_threshold', True):
                    note = " (閾値未達)"
                report += f"""
{i}位: {rec['company_name']} ({rec['code']})
  💰 現在価格: ¥{rec['current_price']:,.0f}
  📈 目標価格: ¥{rec['target_price']:,.0f} (+{rec['expected_return']:.1f}%)
  📉 損切価格: ¥{rec['stop_loss_price']:,.0f} (-{rec['expected_return']:.1f}%)
  🎯 予測確率: {rec['prediction_probability']:.1%}{note}
  📏 判定閾値: {rec.get('threshold', self.min_probability):.0%}
  🏢 セクター: {rec.get('sector', 'Unknown')}
  📊 出来高: {rec['volume']:,}株
  ⏰ 推奨保有: {rec['holding_period']}日間
"""
        
        report += f"""
=====================================
📊 システム情報
=====================================
🤖 使用モデル: {model_label}
🕒 判定条件: 前日終値→翌日終値で+{self.pipeline.target_return*100:.1f}%
🎯 モデル精度: {model_accuracy_display}
📊 特徴量数: {len(self.feature_names)}個
📅 レポート生成: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 月別フォルダ作成とレポートファイル保存
        target_month = target_date.strftime('%Y-%m')
        month_dir = self.results_dir / target_month
        month_dir.mkdir(parents=True, exist_ok=True)
        
        report_file = month_dir / f"{target_date_str}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 レポート保存: {report_file}")
        return report

def main():
    parser = argparse.ArgumentParser(description="終値ベース推奨銘柄システム")
    parser.add_argument("--date", type=str, help="対象日付 (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=None, help="上位N銘柄")
    parser.add_argument("--target-return", type=float, default=None, help="終値ベース判定閾値 (例: 0.8%→0.008)")
    parser.add_argument("--imbalance-boost", type=float, default=1.0, help="scale_pos_weight に掛ける倍率")
    parser.add_argument("--min-probability", type=float, default=None, help="推奨に用いる最低予測確率")
    parser.add_argument("--max-per-sector", type=int, default=None, help="セクターあたりの上限銘柄数")

    args = parser.parse_args()

    system = DailyStockRecommendationCloseV1(target_return=args.target_return, imbalance_boost=args.imbalance_boost, min_probability=args.min_probability, max_per_sector=args.max_per_sector)
    report = system.create_report(args.date, args.top)
    print(report)

if __name__ == "__main__":
    main()
