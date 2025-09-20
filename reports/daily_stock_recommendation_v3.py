#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V3モデル対応推奨銘柄システム
78.6%精度のEnhanced Precision System V3に完全対応
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

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class DailyStockRecommendationV3:
    """V3モデル対応推奨銘柄システム"""
    
    def __init__(self):
        self.model_dir = Path("models")
        self.data_dir = Path("data")
        self.results_dir = Path("production_reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # モデルコンポーネント
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_names = None
        
        # 会社名マッピング
        self.company_names = {}
        self._load_company_names()
        self._load_v3_model()
    
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
                logger.info(f"✅ 会社名マッピング読み込み完了: {len(self.company_names)}社")
            else:
                logger.warning("会社名CSVファイルが見つかりません")
        except Exception as e:
            logger.error(f"会社名読み込みエラー: {e}")
    
    def _get_company_name(self, code):
        """銘柄コードから会社名を取得"""
        return self.company_names.get(str(code), f"銘柄{code}")
    
    def _load_v3_model(self):
        """V3モデルを読み込み"""
        try:
            # V3モデルファイルを探す
            model_files = list(self.model_dir.glob("enhanced_v3/*enhanced_model_v3*.joblib"))
            if not model_files:
                raise FileNotFoundError("V3モデルが見つかりません")
            
            # 最新のV3モデルを使用
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model_data = joblib.load(latest_model)
            
            self.model = model_data['model']
            self.scaler = model_data.get('scaler')
            self.selector = model_data.get('selector')
            self.feature_names = model_data['feature_cols']
            
            logger.info(f"✅ V3モデル読み込み完了: {latest_model.name}")
            logger.info(f"📊 特徴量数: {len(self.feature_names)}")
            
        except Exception as e:
            logger.error(f"V3モデル読み込みエラー: {e}")
            raise
    
    def _load_stock_data(self, target_date):
        """株価データを読み込み"""
        try:
            # 最新の株価データファイルを探す
            data_files = list(self.data_dir.glob("processed/nikkei225_complete_*.parquet"))
            if not data_files:
                raise FileNotFoundError("株価データが見つかりません")
            
            latest_data = max(data_files, key=lambda x: x.stat().st_mtime)
            df = pd.read_parquet(latest_data)
            df['Date'] = pd.to_datetime(df['Date'])
            
            # 対象日までのデータをフィルタ
            df = df[df['Date'] <= target_date].copy()
            
            logger.info(f"✅ 株価データ読み込み完了: {len(df):,}件 (最新: {df['Date'].max().strftime('%Y-%m-%d')})")
            return df
        
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            raise
    
    def _create_v3_features(self, df):
        """V3モデルと同じ特徴量を作成"""
        df = df.copy()
        df = df.sort_values(['Code', 'Date'])
        
        enhanced_df_list = []
        
        for code in df['Code'].unique():
            code_df = df[df['Code'] == code].copy()
            
            if len(code_df) < 30:
                continue
            
            # V3モデルの特徴量を正確に再現
            code_df['Returns'] = code_df['Close'].pct_change(fill_method=None)
            code_df['MA_5'] = code_df['Close'].rolling(5).mean()
            code_df['MA_20'] = code_df['Close'].rolling(20).mean()
            code_df['Volatility'] = code_df['Returns'].rolling(20).std()
            
            # RSI
            delta = code_df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            code_df['RSI'] = 100 - (100 / (1 + rs))
            
            enhanced_df_list.append(code_df)
        
        if not enhanced_df_list:
            raise ValueError("処理可能な銘柄データがありません")
        
        enhanced_df = pd.concat(enhanced_df_list, ignore_index=True)
        
        # 無限値・欠損値処理
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.fillna(method='ffill').fillna(0)
        
        logger.info(f"✅ V3特徴量作成完了: {len(enhanced_df):,}件")
        return enhanced_df
    
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
            
            # データ読み込み
            df = self._load_stock_data(target_date)
            
            # V3特徴量作成
            enhanced_df = self._create_v3_features(df)
            
            # 対象日のデータを取得
            target_data = enhanced_df[enhanced_df['Date'] == target_date].copy()
            
            if len(target_data) == 0:
                logger.warning(f"対象日 {target_date_str} のデータが見つかりません")
                return []
            
            logger.info(f"📊 対象日の銘柄数: {len(target_data)}銘柄")
            
            recommendations = []
            
            for _, row in target_data.iterrows():
                try:
                    code = row['Code']
                    
                    # V3モデルの特徴量のみを抽出
                    features = row[self.feature_names].values.reshape(1, -1)
                    
                    # スケーリング
                    if self.scaler is not None:
                        features = self.scaler.transform(features)
                    
                    # 特徴量選択
                    if self.selector is not None:
                        features = self.selector.transform(features)
                    
                    # 予測
                    prediction_proba = self.model.predict_proba(features)[0][1]
                    
                    # 推奨条件（60%以上）
                    if prediction_proba >= 0.60:
                        recommendations.append({
                            'code': code,
                            'company_name': self._get_company_name(code),
                            'prediction_probability': prediction_proba,
                            'current_price': row['Close'],
                            'volume': row['Volume'],
                            'target_price': row['Close'] * 1.07,
                            'stop_loss_price': row['Close'] * 0.95,
                            'expected_return': 7.0,
                            'holding_period': 10,
                        })
                
                except Exception as e:
                    logger.debug(f"銘柄 {code} の予測エラー: {e}")
                    continue
            
            # 確信度でソート
            recommendations.sort(key=lambda x: x['prediction_probability'], reverse=True)
            recommendations = recommendations[:top_n]
            
            logger.info(f"✅ 推奨銘柄生成完了: {len(recommendations)}銘柄")
            return recommendations
        
        except Exception as e:
            logger.error(f"推奨銘柄生成エラー: {e}")
            return []
    
    def create_report(self, target_date_str=None, top_n=5):
        """レポート作成"""
        if target_date_str is None:
            # 営業日ベースで分析対象日を決定
            target_date = JapanMarketCalendar.get_target_date_for_analysis()
            target_date_str = str(target_date)
            logger.info(f"🗓️ 自動選択された分析対象日: {target_date_str}")
        
        target_date = pd.to_datetime(target_date_str)
        next_date = JapanMarketCalendar.get_next_market_day(target_date)
        
        recommendations = self.generate_recommendations(target_date_str, top_n)
        
        # レポート生成
        report = f"""📈 日次株価予測レポート（V3モデル対応）
=====================================

📅 基準日付: {target_date_str}
📅 推奨取引日: {next_date.strftime('%Y-%m-%d')}
🏆 推奨銘柄数: {len(recommendations)}銘柄 (TOP {top_n})
⚙️ モデル精度: 78.6% (Enhanced Precision System V3)
🎯 推奨閾値: 60%以上の予測確信度

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
  📉 損切価格: ¥{rec['stop_loss_price']:,.0f} (-5.0%)
  🎯 予測確率: {rec['prediction_probability']:.1%}
  📊 出来高: {rec['volume']:,}株
  ⏰ 推奨保有: {rec['holding_period']}日間
"""
        
        report += f"""
=====================================
📊 システム情報
=====================================
🤖 使用モデル: Enhanced Precision System V3
🎯 モデル精度: 78.6%
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
    parser = argparse.ArgumentParser(description="V3モデル対応推奨銘柄システム")
    parser.add_argument("--date", type=str, help="対象日付 (YYYY-MM-DD)")
    parser.add_argument("--top", type=int, default=5, help="上位N銘柄")
    
    args = parser.parse_args()
    
    system = DailyStockRecommendationV3()
    report = system.create_report(args.date, args.top)
    print(report)

if __name__ == "__main__":
    main()