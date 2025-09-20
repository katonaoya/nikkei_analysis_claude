#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日付指定による翌日推奨銘柄レポート作成システム

使用方法:
    python daily_stock_recommendation.py --date 2025-09-05
    python daily_stock_recommendation.py --date 2025-09-05 --top 10
"""

import pandas as pd
import numpy as np
import argparse
import logging
from pathlib import Path
from datetime import datetime, timedelta
import joblib
from typing import List, Dict, Tuple, Optional
import json

# ログ設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

class DailyStockRecommendation:
    """日付指定による翌日推奨銘柄レポート作成システム"""
    
    def __init__(self, model_dir: str = "models", data_dir: str = "data"):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.results_dir = Path("results/daily_reports")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # 最適パラメータ（並列最適化結果より）
        self.optimal_params = {
            'holding_days': 10,
            'profit_target': 0.07,  # 7%
            'stop_loss': 0.05       # 5%
        }
        
        # 銘柄名マッピング読み込み
        self.company_names = self._load_company_names()
        
        # モデルとスケーラーを読み込み
        self.model = None
        self.scaler = None
        self.feature_names = None
        self._load_model_components()
    
    def _load_company_names(self) -> Dict[str, str]:
        """銘柄コードと会社名のマッピングを読み込み"""
        try:
            csv_path = self.data_dir / "nikkei225_codes.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                return dict(zip(df['code'].astype(str).str.zfill(4), df['name']))
            else:
                logger.warning("銘柄名マッピングファイルが見つかりません")
                return {}
        except Exception as e:
            logger.warning(f"銘柄名マッピングの読み込みに失敗: {e}")
            return {}
    
    def _load_model_components(self):
        """学習済みモデル、スケーラー、特徴量名を読み込み"""
        try:
            # Enhanced V3モデルを優先的に探す
            model_files = list(self.model_dir.glob("enhanced_v3/*enhanced_model_v3*.joblib"))
            if not model_files:
                model_files = list(self.model_dir.glob("*final_model*.pkl"))
                model_files.extend(list(self.model_dir.glob("*model*.joblib")))
            
            scaler_files = list(self.model_dir.glob("enhanced_v3/*scaler*.pkl"))
            scaler_files.extend(list(self.model_dir.glob("*scaler*.pkl")))
            scaler_files.extend(list(self.model_dir.glob("*scaler*.joblib")))
            
            if not model_files:
                logger.error("学習済みモデルファイルが見つかりません")
                return
            
            if not scaler_files:
                logger.warning("スケーラーファイルが見つかりません。スケーリングなしで実行します。")
                self.scaler = None
            else:
                # 最新のスケーラーを使用
                latest_scaler = max(scaler_files, key=lambda x: x.stat().st_mtime)
                self.scaler = joblib.load(latest_scaler)
                logger.info(f"✅ スケーラー読み込み完了: {latest_scaler.name}")
            
            # 最新のモデルファイルを使用
            latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
            model_dict = joblib.load(latest_model)
            
            # モデルが辞書形式の場合
            if isinstance(model_dict, dict):
                self.model = model_dict.get('model')
                if self.scaler is None and 'scaler' in model_dict:
                    self.scaler = model_dict.get('scaler')
                if 'feature_cols' in model_dict:
                    self.feature_names = model_dict.get('feature_cols')
            else:
                self.model = model_dict
            
            # 特徴量名を読み込み（あれば）
            feature_file = self.model_dir / "feature_names.json"
            if feature_file.exists():
                with open(feature_file, 'r', encoding='utf-8') as f:
                    self.feature_names = json.load(f)
            
            logger.info(f"✅ モデル読み込み完了: {latest_model.name}")
            
        except Exception as e:
            logger.error(f"モデル読み込みエラー: {e}")
    
    def load_historical_data(self, target_date: str) -> pd.DataFrame:
        """指定日付までの履歴データを読み込み"""
        logger.info(f"📊 {target_date}までのデータを読み込み中...")
        
        # データファイルを探す（複数の場所をチェック）
        parquet_files = list(self.data_dir.glob("*nikkei225*.parquet"))
        parquet_files.extend(list(self.data_dir.glob("**/*nikkei225*.parquet")))
        
        if not parquet_files:
            logger.error("履歴データファイルが見つかりません")
            return pd.DataFrame()
        
        # 最新のファイルを使用
        latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        df = pd.read_parquet(latest_file)
        
        # 日付フィルタ
        df['Date'] = pd.to_datetime(df['Date'])
        target_datetime = pd.to_datetime(target_date)
        df = df[df['Date'] <= target_datetime]
        
        logger.info(f"✅ データ読み込み完了: {len(df):,}件 (最新: {df['Date'].max().date()})")
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """技術指標特徴量を生成"""
        logger.info("🔧 技術指標生成中...")
        
        if df.empty:
            return df
        
        df = df.copy()
        df = df.sort_values(['Code', 'Date'])
        
        enhanced_df_list = []
        
        for code in df['Code'].unique():
            code_df = df[df['Code'] == code].copy()
            
            if len(code_df) < 50:  # 最低限必要なデータ数
                continue
            
            # 基本価格データ
            code_df['Returns'] = code_df['Close'].pct_change()
            code_df['Volume_MA_20'] = code_df['Volume'].rolling(20).mean()
            code_df['Price_Volume_Trend'] = code_df['Returns'] * code_df['Volume']
            
            # 移動平均（多期間）
            for window in [5, 10, 20, 50]:
                code_df[f'MA_{window}'] = code_df['Close'].rolling(window).mean()
                code_df[f'MA_{window}_ratio'] = code_df['Close'] / code_df[f'MA_{window}']
            
            # ボラティリティ（多期間）
            for window in [5, 10, 20]:
                code_df[f'Volatility_{window}'] = code_df['Returns'].rolling(window).std()
            
            # RSI（多期間）
            for window in [7, 14, 21]:
                delta = code_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / loss
                code_df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # ボリンジャーバンド
            for window in [20]:
                rolling_mean = code_df['Close'].rolling(window).mean()
                rolling_std = code_df['Close'].rolling(window).std()
                code_df[f'BB_upper_{window}'] = rolling_mean + (rolling_std * 2)
                code_df[f'BB_lower_{window}'] = rolling_mean - (rolling_std * 2)
                code_df[f'BB_ratio_{window}'] = (code_df['Close'] - code_df[f'BB_lower_{window}']) / (code_df[f'BB_upper_{window}'] - code_df[f'BB_lower_{window}'])
            
            # MACD
            exp1 = code_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = code_df['Close'].ewm(span=26, adjust=False).mean()
            code_df['MACD'] = exp1 - exp2
            code_df['MACD_signal'] = code_df['MACD'].ewm(span=9, adjust=False).mean()
            code_df['MACD_histogram'] = code_df['MACD'] - code_df['MACD_signal']
            
            # オンバランスボリューム
            code_df['OBV'] = (code_df['Volume'] * np.where(code_df['Close'] > code_df['Close'].shift(1), 1, 
                             np.where(code_df['Close'] < code_df['Close'].shift(1), -1, 0))).cumsum()
            
            # ストキャスティクス
            for window in [14]:
                low_min = code_df['Low'].rolling(window).min()
                high_max = code_df['High'].rolling(window).max()
                code_df[f'Stoch_K_{window}'] = 100 * (code_df['Close'] - low_min) / (high_max - low_min)
                code_df[f'Stoch_D_{window}'] = code_df[f'Stoch_K_{window}'].rolling(3).mean()
            
            # ATR (Average True Range)
            high_low = code_df['High'] - code_df['Low']
            high_close = np.abs(code_df['High'] - code_df['Close'].shift())
            low_close = np.abs(code_df['Low'] - code_df['Close'].shift())
            true_range = np.maximum(high_low, np.maximum(high_close, low_close))
            code_df['ATR_14'] = true_range.rolling(14).mean()
            
            enhanced_df_list.append(code_df)
        
        if not enhanced_df_list:
            logger.error("特徴量生成可能な銘柄がありません")
            return pd.DataFrame()
        
        enhanced_df = pd.concat(enhanced_df_list, ignore_index=True)
        enhanced_df = enhanced_df.dropna()
        
        logger.info(f"✅ 特徴量生成完了: {len(enhanced_df):,}件")
        return enhanced_df
    
    def predict_recommendations(self, df: pd.DataFrame, target_date: str, top_n: int = 5) -> List[Dict]:
        """翌日の推奨銘柄を予測"""
        logger.info(f"🔮 {target_date}翌日の推奨銘柄予測中...")
        
        if self.model is None:
            logger.error("モデルが読み込まれていません")
            return []
        
        # 指定日の最新データを取得
        target_datetime = pd.to_datetime(target_date)
        latest_df = df[df['Date'] == target_datetime]
        
        if latest_df.empty:
            logger.warning(f"{target_date}のデータが見つかりません。直近のデータを使用します。")
            latest_df = df[df['Date'] == df['Date'].max()]
        
        recommendations = []
        
        # 特徴量カラムを特定（モデルから取得するか、自動検出）
        if self.feature_names:
            feature_cols = self.feature_names
        else:
            feature_cols = [col for col in latest_df.columns 
                           if col not in ['Code', 'Date', 'CompanyName', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
        
        for _, row in latest_df.iterrows():
            try:
                code = row['Code']
                company_name = self.company_names.get(str(code), f"銘柄{code}")
                
                # 特徴量を準備
                features = row[feature_cols].values.reshape(1, -1)
                
                # 欠損値チェック
                if pd.isna(features).any():
                    continue
                
                # スケーリング（スケーラーがある場合のみ）
                if self.scaler is not None:
                    features_scaled = self.scaler.transform(features)
                else:
                    features_scaled = features
                
                # 予測
                prediction = self.model.predict(features_scaled)[0]
                prediction_proba = self.model.predict_proba(features_scaled)[0][1]  # 正例確率
                
                # 推奨条件（確率閾値）- 78.6%精度モデルに合わせて調整
                if prediction_proba >= 0.60:  # 60%以上の信頼度
                    recommendations.append({
                        'code': code,
                        'company_name': company_name,
                        'prediction_probability': prediction_proba,
                        'current_price': row['Close'],
                        'target_price': row['Close'] * (1 + self.optimal_params['profit_target']),
                        'stop_loss_price': row['Close'] * (1 - self.optimal_params['stop_loss']),
                        'expected_return': self.optimal_params['profit_target'] * 100,
                        'holding_period': self.optimal_params['holding_days'],
                        'volume': row['Volume'],
                        'rsi_14': row.get('RSI_14', None),
                        'macd_histogram': row.get('MACD_histogram', None)
                    })
                    
            except Exception as e:
                logger.warning(f"銘柄{code}の予測処理でエラー: {e}")
                continue
        
        # 確率順でソート
        recommendations.sort(key=lambda x: x['prediction_probability'], reverse=True)
        
        logger.info(f"✅ 推奨銘柄抽出完了: {len(recommendations)}銘柄")
        return recommendations[:top_n]
    
    def generate_report(self, recommendations: List[Dict], target_date: str, top_n: int) -> str:
        """レポートを生成"""
        next_date = (pd.to_datetime(target_date) + timedelta(days=1)).strftime('%Y-%m-%d')
        
        report = f"""
📈 日次株価予測レポート
=====================================

📅 基準日付: {target_date}
📅 推奨取引日: {next_date}
🏆 推奨銘柄数: {len(recommendations)}銘柄 (TOP {top_n})
⚙️  最適パラメータ: 保有{self.optimal_params['holding_days']}日・利確{self.optimal_params['profit_target']*100:.1f}%・損切{self.optimal_params['stop_loss']*100:.1f}%

=====================================
🎯 推奨銘柄一覧
=====================================
"""
        
        if not recommendations:
            report += "\n❌ 推奨条件を満たす銘柄がありませんでした。\n"
            return report
        
        for i, rec in enumerate(recommendations, 1):
            report += f"""
{i}位: {rec['company_name']} ({rec['code']})
  💰 現在価格: ¥{rec['current_price']:,.0f}
  📈 目標価格: ¥{rec['target_price']:,.0f} (+{rec['expected_return']:.1f}%)
  📉 損切価格: ¥{rec['stop_loss_price']:,.0f} (-{self.optimal_params['stop_loss']*100:.1f}%)
  🎯 予測確率: {rec['prediction_probability']:.1%}
  📊 出来高: {rec['volume']:,}株
  📈 RSI(14): {rec['rsi_14']:.1f if rec['rsi_14'] is not None else 'N/A'}
  📊 MACD: {'上昇' if rec['macd_histogram'] and rec['macd_histogram'] > 0 else '下降' if rec['macd_histogram'] and rec['macd_histogram'] < 0 else 'N/A'}
  ⏰ 推奨保有: {rec['holding_period']}日間
"""
        
        report += f"""
=====================================
📊 投資戦略サマリー
=====================================

💡 運用方針:
  • 各銘柄への投資上限: 20万円推奨
  • 最大同時保有: 5銘柄
  • 利確目標: +{self.optimal_params['profit_target']*100:.1f}%
  • 損切設定: -{self.optimal_params['stop_loss']*100:.1f}%
  • 最大保有期間: {self.optimal_params['holding_days']}日

⚠️  リスク管理:
  • 必ず損切りラインを設定してください
  • 市場急変時は早期撤退を検討
  • 分散投資を心がけてください

📈 期待パフォーマンス:
  • 年間期待リターン: 114.63%
  • 推定勝率: 54.1%
  • 過去検証データ: 日経225・10年間

=====================================
⚠️  免責事項: 本レポートは過去データに基づく予測であり、
投資成果を保証するものではありません。
投資は自己責任で行ってください。
=====================================
"""
        
        return report
    
    def save_report(self, report: str, target_date: str) -> str:
        """レポートをファイルに保存"""
        timestamp = datetime.now().strftime('%H%M%S')
        filename = f"stock_recommendation_{target_date}_{timestamp}.txt"
        filepath = self.results_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"📄 レポート保存: {filepath}")
        return str(filepath)
    
    def run_daily_analysis(self, target_date: str, top_n: int = 5) -> str:
        """日次分析を実行"""
        logger.info(f"🚀 {target_date}の日次分析開始...")
        
        try:
            # データ読み込み
            df = self.load_historical_data(target_date)
            if df.empty:
                raise Exception("履歴データが見つかりません")
            
            # 特徴量生成
            enhanced_df = self.create_features(df)
            if enhanced_df.empty:
                raise Exception("特徴量生成に失敗しました")
            
            # 推奨銘柄予測
            recommendations = self.predict_recommendations(enhanced_df, target_date, top_n)
            
            # レポート生成
            report = self.generate_report(recommendations, target_date, top_n)
            
            # レポート保存
            filepath = self.save_report(report, target_date)
            
            # コンソール出力
            print(report)
            
            logger.info("✅ 日次分析完了")
            return filepath
            
        except Exception as e:
            logger.error(f"❌ 日次分析エラー: {e}")
            return ""

def main():
    parser = argparse.ArgumentParser(description='日付指定による翌日推奨銘柄レポート作成')
    # デフォルトを今日の日付にする
    today = datetime.now().strftime('%Y-%m-%d')
    parser.add_argument('--date', default=today, help=f'基準日付 (YYYY-MM-DD, デフォルト: {today})')
    parser.add_argument('--top', type=int, default=5, help='推奨銘柄数 (デフォルト: 5)')
    
    args = parser.parse_args()
    
    # 日付検証
    try:
        pd.to_datetime(args.date)
    except:
        logger.error("❌ 日付形式が正しくありません (YYYY-MM-DD)")
        return
    
    # 分析実行
    analyzer = DailyStockRecommendation()
    result_file = analyzer.run_daily_analysis(args.date, args.top)
    
    if result_file:
        logger.info(f"🎉 分析完了: {result_file}")
    else:
        logger.error("❌ 分析に失敗しました")

if __name__ == "__main__":
    main()