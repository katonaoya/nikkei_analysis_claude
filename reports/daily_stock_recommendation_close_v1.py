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

        # モデルコンポーネント
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        self.model_accuracy = None
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
    
    def _load_close_model(self):
        """終値ベースモデルを読み込み"""
        try:
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
            
            recommendations = []
            
            for _, row in target_data.iterrows():
                try:
                    code = row['Code']
                    
                    # 終値ベースモデルの特徴量を抽出（欠損列は0で補完）
                    feature_values = []
                    missing_cols = []
                    for col in self.feature_names:
                        if col in row:
                            feature_values.append(row[col])
                        else:
                            feature_values.append(0.0)
                            missing_cols.append(col)

                    if missing_cols:
                        logger.debug(
                            "銘柄 %s: 欠損特徴量 %s を0で補完",
                            code,
                            ", ".join(missing_cols)
                        )

                    features = pd.DataFrame([feature_values], columns=self.feature_names)

                    # スケーリング
                    if self.scaler is not None:
                        features = self.scaler.transform(features)

                    # 特徴量選択
                    if self.selector is not None:
                        features = self.selector.transform(features)
                    
                    # 予測
                    prediction_proba = self.model.predict_proba(features)[0][1]
                    if self.calibration is not None:
                        coef = self.calibration.get('coef', 0.0)
                        intercept = self.calibration.get('intercept', 0.0)
                        linear = coef * prediction_proba + intercept
                        prediction_proba = 1 / (1 + np.exp(-linear))
                    
                    # 推奨条件（デフォルト60%以上）
                    if prediction_proba >= self.min_probability:
                        target_return = float(getattr(self.pipeline, 'target_return', 0.01))
                        recommendations.append({
                            'code': code,
                            'company_name': self._get_company_name(code),
                            'prediction_probability': prediction_proba,
                            'current_price': row['Close'],
                            'volume': row['Volume'],
                            'target_price': row['Close'] * (1 + target_return),
                            'stop_loss_price': row['Close'] * (1 - target_return),
                            'expected_return': target_return * 100,
                            'holding_period': 1,
                            'sector': self.company_sectors.get(str(code), 'Unknown')
                        })
                
                except Exception as e:
                    logger.debug(f"銘柄 {code} の予測エラー: {e}")
                    continue
            
            # 確信度でソート
            recommendations.sort(key=lambda x: x['prediction_probability'], reverse=True)

            selected = []
            sector_counts = {}
            for rec in recommendations:
                sector = rec.get('sector', 'Unknown')
                if sector_counts.get(sector, 0) >= self.max_per_sector:
                    continue
                selected.append(rec)
                sector_counts[sector] = sector_counts.get(sector, 0) + 1
                if len(selected) >= top_n:
                    break
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
            top_n = self.config.get('top_n', 5)
        
        recommendations = self.generate_recommendations(target_date_str, top_n)
        
        # レポート生成
        model_accuracy_display = "N/A"
        if self.model_accuracy is not None:
            model_accuracy_display = f"{self.model_accuracy * 100:.2f}%"

        report = f"""📈 日次株価予測レポート（終値ベースモデル対応）
=====================================

📅 基準日付: {target_date_str}
📅 推奨取引日: {next_date.strftime('%Y-%m-%d')}
🏆 推奨銘柄数: {len(recommendations)}銘柄 (TOP {top_n})
⚙️ モデル精度: {model_accuracy_display} (Close-to-Close Precision System V1)
📈 判定閾値: {getattr(self.pipeline, 'target_return', 0.01)*100:.1f}% (終値→終値)
🎯 推奨閾値: 翌営業日終値が+{getattr(self.pipeline, 'target_return', 0.01)*100:.1f}%以上になる確率 {self.min_probability*100:.0f}%以上

=====================================
🎯 推奨銘柄一覧
=====================================
"""
        
        if not recommendations:
            report += "\n❌ 推奨条件を満たす銘柄がありませんでした。\n"
        else:
            for i, rec in enumerate(recommendations, 1):
                report += f"""
{i}位: {rec['company_name']} ({rec['code']})
  💰 現在価格: ¥{rec['current_price']:,.0f}
  📈 目標価格: ¥{rec['target_price']:,.0f} (+{rec['expected_return']:.1f}%)
  📉 損切価格: ¥{rec['stop_loss_price']:,.0f} (-{rec['expected_return']:.1f}%)
  🎯 予測確率: {rec['prediction_probability']:.1%}
  🏢 セクター: {rec.get('sector', 'Unknown')}
  📊 出来高: {rec['volume']:,}株
  ⏰ 推奨保有: {rec['holding_period']}日間
"""
        
        report += f"""
=====================================
📊 システム情報
=====================================
🤖 使用モデル: Close-to-Close Precision System V1
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
