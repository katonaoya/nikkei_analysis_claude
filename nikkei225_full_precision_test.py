#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日経225全銘柄×10年間完全データでの精度検証システム
530,744件の完全データセットでLightGBMモデルの精度を最終検証
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
import joblib
import warnings
from typing import Tuple, Optional

# 機械学習関連
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, classification_report
import lightgbm as lgb

warnings.filterwarnings('ignore')

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Nikkei225FullPrecisionTest:
    """日経225全銘柄×10年間完全データでの精度検証"""
    
    def __init__(self):
        """初期化"""
        # 最新の完全データファイルを自動検出
        self.data_dir = Path("data/nikkei225_full")
        self.model_dir = Path("models")
        self.model_dir.mkdir(exist_ok=True)
        
        # 拡張LightGBMパラメータ（大規模データ用に最適化）
        self.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,
            'max_depth': 8,
            'min_child_samples': 30,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'learning_rate': 0.03,  # 大規模データ用に小さく
            'reg_alpha': 0.1,       # L1正則化追加
            'reg_lambda': 0.1,      # L2正則化追加
            'random_state': 42,
            'verbose': -1
        }
        
        logger.info("日経225全銘柄×10年間完全データ精度検証システム初期化完了")
    
    def load_latest_full_data(self) -> pd.DataFrame:
        """最新の完全データセットを読み込み"""
        if not self.data_dir.exists():
            raise FileNotFoundError(f"データディレクトリが見つかりません: {self.data_dir}")
        
        # 最新のparquetファイルを検索
        parquet_files = list(self.data_dir.glob("nikkei225_full_*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"完全データファイルが見つかりません: {self.data_dir}")
        
        latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
        logger.info(f"📁 最新データファイル読み込み: {latest_file.name}")
        
        df = pd.read_parquet(latest_file)
        
        # データ情報表示
        logger.info(f"📊 読み込み完了:")
        logger.info(f"  総レコード数: {len(df):,}件")
        logger.info(f"  銘柄数: {df['Code'].nunique()}銘柄")
        logger.info(f"  期間: {df['Date'].min()} ～ {df['Date'].max()}")
        
        return df
    
    def create_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """拡張技術指標の生成（大規模データ用）"""
        logger.info("🔧 拡張技術指標生成開始...")
        
        df = df.copy()
        df = df.sort_values(['Code', 'Date'])
        
        enhanced_df_list = []
        
        for code in df['Code'].unique():
            code_df = df[df['Code'] == code].copy()
            
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
            
            enhanced_df_list.append(code_df)
        
        enhanced_df = pd.concat(enhanced_df_list, ignore_index=True)
        
        # 目的変数の作成
        enhanced_df['Target'] = 0
        for code in enhanced_df['Code'].unique():
            mask = enhanced_df['Code'] == code
            code_data = enhanced_df[mask].copy()
            # 翌日の高値が前日終値から1%以上上昇
            next_high = code_data['High'].shift(-1)
            prev_close = code_data['Close'].shift(1)
            enhanced_df.loc[mask, 'Target'] = (next_high / prev_close > 1.01).astype(int)
        
        # NaNを除去
        enhanced_df = enhanced_df.dropna()
        
        logger.info(f"✅ 特徴量生成完了:")
        logger.info(f"  処理後レコード数: {len(enhanced_df):,}件")
        logger.info(f"  特徴量数: {len([col for col in enhanced_df.columns if col not in ['Code', 'Date', 'CompanyName', 'Target']])}個")
        logger.info(f"  正例率: {enhanced_df['Target'].mean():.3f}")
        
        return enhanced_df
    
    def advanced_time_series_validation(self, df: pd.DataFrame) -> Tuple[float, dict]:
        """拡張時系列バックテスト（30日間）"""
        logger.info("🎯 拡張時系列バックテスト開始（30日間）...")
        
        # 特徴量とターゲット分離
        feature_cols = [col for col in df.columns if col not in ['Code', 'Date', 'CompanyName', 'Target']]
        
        # 日付列をdatetime型に変換
        df = df.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        
        # 日付でソート
        df_sorted = df.sort_values('Date')
        
        # テスト期間設定（直近30日間）
        latest_date = df_sorted['Date'].max()
        test_start_date = latest_date - pd.Timedelta(days=30)
        
        # 訓練・テストデータ分割
        train_df = df_sorted[df_sorted['Date'] < test_start_date].copy()
        test_df = df_sorted[df_sorted['Date'] >= test_start_date].copy()
        
        logger.info(f"📅 訓練期間: {train_df['Date'].min()} ～ {train_df['Date'].max()}")
        logger.info(f"📅 テスト期間: {test_df['Date'].min()} ～ {test_df['Date'].max()}")
        logger.info(f"📊 訓練データ: {len(train_df):,}件")
        logger.info(f"📊 テストデータ: {len(test_df):,}件")
        
        if len(test_df) == 0:
            logger.error("❌ テストデータが存在しません")
            return 0.0, {}
        
        # 特徴量準備
        X_train = train_df[feature_cols]
        y_train = train_df['Target']
        X_test = test_df[feature_cols]
        y_test = test_df['Target']
        
        # 特徴量選択（上位30個）
        logger.info("🔍 特徴量選択中...")
        selector = SelectKBest(score_func=f_classif, k=min(30, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        selected_features = [feature_cols[i] for i in selector.get_support(indices=True)]
        logger.info(f"✅ 選択された特徴量: {len(selected_features)}個")
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # LightGBMモデル訓練
        logger.info("🤖 拡張LightGBMモデル訓練中...")
        model = lgb.LGBMClassifier(**self.model_params)
        model.fit(X_train_scaled, y_train)
        
        # 予測実行
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 日別バックテスト
        test_df_copy = test_df.copy()
        test_df_copy['PredProba'] = pred_proba
        
        daily_results = []
        unique_dates = sorted(test_df_copy['Date'].unique())
        
        for test_date in unique_dates:
            daily_data = test_df_copy[test_df_copy['Date'] == test_date]
            
            if len(daily_data) < 3:
                continue
            
            # 上位3銘柄選択
            top3_indices = daily_data['PredProba'].nlargest(3).index
            selected_predictions = daily_data.loc[top3_indices]
            
            # 実際の結果
            actual_results = selected_predictions['Target'].values
            precision = np.mean(actual_results)
            
            daily_results.append({
                'date': test_date,
                'precision': precision,
                'n_correct': np.sum(actual_results),
                'n_total': len(actual_results),
                'selected_codes': selected_predictions['Code'].tolist(),
                'probabilities': selected_predictions['PredProba'].tolist()
            })
        
        if not daily_results:
            logger.error("❌ 有効なテストデータが見つかりません")
            return 0.0, {}
        
        # 総合精度計算
        total_correct = sum(result['n_correct'] for result in daily_results)
        total_predictions = sum(result['n_total'] for result in daily_results)
        overall_precision = total_correct / total_predictions if total_predictions > 0 else 0.0
        
        # 詳細統計
        daily_precisions = [result['precision'] for result in daily_results]
        stats = {
            'overall_precision': overall_precision,
            'total_correct': total_correct,
            'total_predictions': total_predictions,
            'test_days': len(daily_results),
            'mean_daily_precision': np.mean(daily_precisions),
            'std_daily_precision': np.std(daily_precisions),
            'min_daily_precision': np.min(daily_precisions),
            'max_daily_precision': np.max(daily_precisions),
            'selected_features': selected_features,
            'daily_results': daily_results[-5:]  # 最新5日分
        }
        
        logger.info("="*60)
        logger.info("📊 拡張バックテスト結果（日経225全データ）")
        logger.info("="*60)
        logger.info(f"🎯 総合精度: {overall_precision:.4f} ({overall_precision*100:.2f}%)")
        logger.info(f"✅ 的中数: {total_correct}/{total_predictions}")
        logger.info(f"📅 テスト期間: {len(daily_results)}日間")
        logger.info(f"📈 日次精度: {np.mean(daily_precisions):.4f}±{np.std(daily_precisions):.4f}")
        
        return overall_precision, stats
    
    def save_final_model_and_results(self, df: pd.DataFrame, precision: float, stats: dict) -> str:
        """最終モデルと結果を保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # モデル再訓練
        feature_cols = [col for col in df.columns if col not in ['Code', 'Date', 'CompanyName', 'Target']]
        X = df[feature_cols]
        y = df['Target']
        
        selector = SelectKBest(score_func=f_classif, k=min(30, len(feature_cols)))
        X_selected = selector.fit_transform(X, y)
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X_selected)
        
        final_model = lgb.LGBMClassifier(**self.model_params)
        final_model.fit(X_scaled, y)
        
        # 保存
        model_filename = f"nikkei225_full_model_{len(df)}records_{precision:.4f}precision_{timestamp}.joblib"
        model_path = self.model_dir / model_filename
        
        joblib.dump({
            'model': final_model,
            'scaler': scaler,
            'selector': selector,
            'feature_cols': feature_cols,
            'precision': precision,
            'stats': stats,
            'data_info': {
                'total_records': len(df),
                'stocks': df['Code'].nunique(),
                'period': f"{df['Date'].min()} - {df['Date'].max()}"
            }
        }, model_path)
        
        logger.info(f"💾 最終モデル保存: {model_filename}")
        return str(model_path)
    
    def run_complete_validation(self) -> dict:
        """完全な精度検証を実行"""
        logger.info("🚀 日経225全銘柄×10年間完全データ精度検証開始")
        
        try:
            # 1. データ読み込み
            df = self.load_latest_full_data()
            
            # 2. 特徴量生成
            enhanced_df = self.create_advanced_features(df)
            
            # 3. 精度検証
            precision, stats = self.advanced_time_series_validation(enhanced_df)
            
            # 4. モデル保存
            model_path = self.save_final_model_and_results(enhanced_df, precision, stats)
            
            # 5. 最終結果
            final_results = {
                'precision': precision,
                'precision_percent': precision * 100,
                'data_records': len(enhanced_df),
                'data_stocks': enhanced_df['Code'].nunique(),
                'model_path': model_path,
                'stats': stats
            }
            
            logger.info("="*60)
            logger.info("🎉 日経225全銘柄×10年間完全データ検証完了")
            logger.info("="*60)
            logger.info(f"📊 最終精度: {precision:.4f} ({precision*100:.2f}%)")
            logger.info(f"📈 データ規模: {len(enhanced_df):,}件 ({enhanced_df['Code'].nunique()}銘柄)")
            logger.info(f"🎯 60%目標: {'✅ 達成' if precision >= 0.60 else '❌ 未達成'}")
            
            return final_results
            
        except Exception as e:
            logger.error(f"❌ 検証エラー: {str(e)}")
            raise


def main():
    """メイン実行関数"""
    logger.info("🚀 日経225全銘柄×10年間完全データ精度検証システム開始")
    
    try:
        validator = Nikkei225FullPrecisionTest()
        results = validator.run_complete_validation()
        
        logger.info("="*60)
        logger.info("📊 最終結果サマリー")
        logger.info("="*60)
        logger.info(f"🎯 達成精度: {results['precision_percent']:.2f}%")
        logger.info(f"📊 使用データ: {results['data_records']:,}件 ({results['data_stocks']}銘柄)")
        logger.info(f"💾 保存モデル: {Path(results['model_path']).name}")
        
        # ベースラインとの比較
        baseline_precision = 0.5758  # 既存ベースライン
        improvement = (results['precision'] - baseline_precision) / baseline_precision * 100
        
        logger.info(f"📈 ベースライン比較: {baseline_precision:.4f} → {results['precision']:.4f}")
        logger.info(f"📊 改善率: {improvement:+.1f}%")
        
        if results['precision'] >= 0.60:
            logger.info("🎉 60%目標達成！実用レベルの精度を実現")
        else:
            logger.info(f"⚠️  60%目標未達成（現在{results['precision_percent']:.2f}%）")
            
    except Exception as e:
        logger.error(f"❌ システムエラー: {str(e)}")
        raise


if __name__ == "__main__":
    main()