#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
日経225完全データセット高精度学習・検証システム
95.45%精度モデルと同様のパターンで最新データセット（542,143件）を使用して学習・検証
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import os
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ML関連
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Nikkei225CompletePrecisionTrainer:
    """日経225完全データセット高精度学習・検証システム"""
    
    def __init__(self, data_file: str = None):
        """初期化"""
        # 最新の日経225完全データセットを使用
        if data_file is None:
            data_file = "data/processed/nikkei225_complete_225stocks_20250909_230649.parquet"
        
        self.data_file = data_file
        self.df = None
        self.model = None
        self.scaler = None
        self.selector = None
        self.feature_cols = None
        
        # 95.45%精度モデルと同じパラメータ
        self.model_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 300,           # 大規模データ用
            'max_depth': 8,                # 複雑パターン学習
            'min_child_samples': 30,       # 過学習防止
            'subsample': 0.8,              # データサンプリング
            'colsample_bytree': 0.8,       # 特徴量サンプリング
            'learning_rate': 0.03,         # 大規模データ用
            'reg_alpha': 0.1,              # L1正則化
            'reg_lambda': 0.1,             # L2正則化
            'random_state': 42,            # 再現性確保
            'verbose': -1                  # ログ抑制
        }
        
        logger.info(f"データファイル: {data_file}")
    
    def load_and_prepare_data(self):
        """データ読み込みと前処理"""
        logger.info("データ読み込み開始...")
        
        try:
            self.df = pd.read_parquet(self.data_file)
            logger.info(f"データ読み込み完了: {len(self.df):,}件, {self.df['Code'].nunique()}銘柄")
            
            # 日付型変換
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            # データ範囲確認
            logger.info(f"データ期間: {self.df['Date'].min()} 〜 {self.df['Date'].max()}")
            
            # 基本統計
            logger.info(f"銘柄数: {self.df['Code'].nunique()}")
            logger.info(f"平均レコード数/銘柄: {len(self.df)/self.df['Code'].nunique():.0f}")
            
            return True
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return False
    
    def create_enhanced_features(self):
        """95.45%精度モデルと同様の拡張特徴量作成"""
        logger.info("拡張特徴量作成開始...")
        
        enhanced_df = self.df.copy()
        
        # 銘柄別に特徴量計算
        result_dfs = []
        
        for code in enhanced_df['Code'].unique():
            code_df = enhanced_df[enhanced_df['Code'] == code].copy()
            code_df = code_df.sort_values('Date')
            
            # 基本リターンとボリューム
            code_df['Returns'] = code_df['Close'].pct_change(fill_method=None)
            code_df['Volume_MA_20'] = code_df['Volume'].rolling(20).mean()
            code_df['Price_Volume_Trend'] = code_df['Returns'] * code_df['Volume']
            
            # 移動平均（4種類）
            for window in [5, 10, 20, 50]:
                code_df[f'MA_{window}'] = code_df['Close'].rolling(window).mean()
                code_df[f'MA_{window}_ratio'] = code_df['Close'] / code_df[f'MA_{window}']
            
            # ボラティリティ（3種類）
            for window in [5, 10, 20]:
                code_df[f'Volatility_{window}'] = code_df['Returns'].rolling(window).std()
            
            # RSI（3種類）
            for window in [7, 14, 21]:
                delta = code_df['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / loss
                code_df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # ボリンジャーバンド（20日）
            for window in [20]:
                rolling_mean = code_df['Close'].rolling(window).mean()
                rolling_std = code_df['Close'].rolling(window).std()
                code_df[f'BB_upper_{window}'] = rolling_mean + (rolling_std * 2)
                code_df[f'BB_lower_{window}'] = rolling_mean - (rolling_std * 2)
                code_df[f'BB_ratio_{window}'] = (code_df['Close'] - code_df[f'BB_lower_{window}']) / (code_df[f'BB_upper_{window}'] - code_df[f'BB_lower_{window}'])
            
            # MACD（3指標）
            exp1 = code_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = code_df['Close'].ewm(span=26, adjust=False).mean()
            code_df['MACD'] = exp1 - exp2
            code_df['MACD_signal'] = code_df['MACD'].ewm(span=9, adjust=False).mean()
            code_df['MACD_histogram'] = code_df['MACD'] - code_df['MACD_signal']
            
            # オンバランスボリューム（OBV）
            code_df['OBV'] = (code_df['Volume'] * np.where(code_df['Close'] > code_df['Close'].shift(1), 1, 
                             np.where(code_df['Close'] < code_df['Close'].shift(1), -1, 0))).cumsum()
            
            # ストキャスティクス（14日）
            for window in [14]:
                low_min = code_df['Low'].rolling(window).min()
                high_max = code_df['High'].rolling(window).max()
                code_df[f'Stoch_K_{window}'] = 100 * (code_df['Close'] - low_min) / (high_max - low_min)
                code_df[f'Stoch_D_{window}'] = code_df[f'Stoch_K_{window}'].rolling(3).mean()
            
            result_dfs.append(code_df)
        
        # 結合
        enhanced_df = pd.concat(result_dfs, ignore_index=True)
        
        # 目的変数作成（95.45%精度の核心）
        logger.info("目的変数作成...")
        enhanced_df['Target'] = 0
        
        for code in enhanced_df['Code'].unique():
            mask = enhanced_df['Code'] == code
            code_data = enhanced_df[mask].copy()
            # 翌日の高値が前日終値から1%以上上昇
            next_high = code_data['High'].shift(-1)    # 翌日高値
            prev_close = code_data['Close'].shift(1)   # 前日終値
            enhanced_df.loc[mask, 'Target'] = (next_high / prev_close > 1.01).astype(int)
        
        # 欠損値除去
        enhanced_df = enhanced_df.dropna()
        
        self.df = enhanced_df
        logger.info(f"拡張特徴量作成完了: {len(self.df):,}件")
        
        # 正例率確認
        positive_rate = self.df['Target'].mean()
        logger.info(f"正例率: {positive_rate:.3f} ({positive_rate:.1%})")
        
        return enhanced_df
    
    def prepare_features_and_target(self):
        """特徴量とターゲットの準備"""
        logger.info("特徴量とターゲット準備...")
        
        # 特徴量カラム選択（非数値列除外）
        exclude_cols = ['Date', 'Code', 'CompanyName', 'MatchMethod', 'ApiCode', 'Target']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        
        # 数値型のみ選択
        numeric_cols = self.df[self.feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = numeric_cols
        
        logger.info(f"使用特徴量数: {len(self.feature_cols)}")
        
        X = self.df[self.feature_cols]
        y = self.df['Target']
        
        # 無限値やNaN除去
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X, y
    
    def train_and_validate_model(self):
        """95.45%精度モデルと同様の学習・検証"""
        logger.info("モデル学習・検証開始...")
        
        # 特徴量準備
        X, y = self.prepare_features_and_target()
        
        # 時系列分割（95.45%精度モデルと同様）
        df_sorted = self.df.sort_values('Date')
        latest_date = df_sorted['Date'].max()
        test_start_date = latest_date - pd.Timedelta(days=30)  # 30日間テスト
        
        logger.info(f"訓練期間: 〜 {test_start_date}")
        logger.info(f"テスト期間: {test_start_date} 〜 {latest_date}")
        
        # 訓練・テストデータ分割
        train_mask = df_sorted['Date'] < test_start_date
        test_mask = df_sorted['Date'] >= test_start_date
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"訓練データ: {len(X_train):,}件")
        logger.info(f"テストデータ: {len(X_test):,}件")
        
        # 特徴量選択（上位30特徴量）
        logger.info("特徴量選択...")
        self.selector = SelectKBest(score_func=f_classif, k=30)
        X_train_selected = self.selector.fit_transform(X_train, y_train)
        X_test_selected = self.selector.transform(X_test)
        
        # スケーリング
        logger.info("スケーリング...")
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train_selected)
        X_test_scaled = self.scaler.transform(X_test_selected)
        
        # モデル学習
        logger.info("LightGBMモデル学習...")
        self.model = LGBMClassifier(**self.model_params)
        self.model.fit(X_train_scaled, y_train)
        
        # 予測
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # 日別精度検証（95.45%精度モデルと同様）
        return self.evaluate_daily_precision(df_sorted[test_mask], y_pred_proba)
    
    def evaluate_daily_precision(self, test_df, pred_proba):
        """日別精度評価（95.45%精度モデルと同様）"""
        logger.info("日別精度評価開始...")
        
        test_df_copy = test_df.copy()
        test_df_copy['PredProba'] = pred_proba
        
        # 営業日別に評価
        unique_dates = sorted(test_df_copy['Date'].unique())
        daily_results = []
        total_predictions = 0
        total_correct = 0
        
        logger.info(f"検証期間: {unique_dates[0]} 〜 {unique_dates[-1]} ({len(unique_dates)}営業日)")
        
        for test_date in unique_dates:
            daily_data = test_df_copy[test_df_copy['Date'] == test_date]
            
            if len(daily_data) < 3:
                continue
            
            # 上位3銘柄選択（質重視戦略）
            top3_indices = daily_data['PredProba'].nlargest(3).index
            selected_predictions = daily_data.loc[top3_indices]
            
            # 実際の結果
            actual_results = selected_predictions['Target'].values
            n_correct = np.sum(actual_results)
            n_total = len(actual_results)
            daily_precision = n_correct / n_total if n_total > 0 else 0
            
            daily_results.append({
                'date': test_date,
                'n_correct': n_correct,
                'n_total': n_total,
                'precision': daily_precision,
                'selected_codes': selected_predictions['Code'].tolist()
            })
            
            total_correct += n_correct
            total_predictions += n_total
            
            logger.info(f"{test_date.strftime('%Y-%m-%d')}: {n_correct}/{n_total} = {daily_precision:.1%} "
                       f"[{', '.join(selected_predictions['Code'].astype(str).tolist())}]")
        
        # 総合精度計算
        overall_precision = total_correct / total_predictions if total_predictions > 0 else 0
        
        logger.info(f"\n🎉 検証結果:")
        logger.info(f"検証営業日数: {len(daily_results)}日間")
        logger.info(f"総予測数: {total_predictions}件")
        logger.info(f"的中数: {total_correct}件")
        logger.info(f"総合精度: {overall_precision:.4f} ({overall_precision:.2%})")
        
        return {
            'overall_precision': overall_precision,
            'daily_results': daily_results,
            'total_correct': total_correct,
            'total_predictions': total_predictions,
            'n_days': len(daily_results)
        }
    
    def save_model(self, results):
        """モデルと結果保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        precision_str = f"{results['overall_precision']:.4f}".replace('.', '')
        
        # modelsディレクトリ作成
        os.makedirs("models", exist_ok=True)
        
        # モデルファイル名
        model_file = f"models/nikkei225_complete_model_{len(self.df)}records_{precision_str}precision_{timestamp}.joblib"
        
        # 保存データ
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'selector': self.selector,
            'feature_cols': self.feature_cols,
            'precision': results['overall_precision'],
            'results': results,
            'data_info': {
                'total_records': len(self.df),
                'n_companies': self.df['Code'].nunique(),
                'data_period': f"{self.df['Date'].min()} - {self.df['Date'].max()}",
                'model_params': self.model_params
            }
        }
        
        # 保存
        joblib.dump(model_data, model_file)
        logger.info(f"モデル保存完了: {model_file}")
        
        return model_file
    
    def run_complete_training(self):
        """完全な学習・検証実行"""
        logger.info("🚀 日経225完全データセット高精度学習・検証開始!")
        
        try:
            # データ読み込み
            if not self.load_and_prepare_data():
                return None
            
            # 特徴量作成
            self.create_enhanced_features()
            
            # 学習・検証
            results = self.train_and_validate_model()
            
            # モデル保存
            model_file = self.save_model(results)
            
            # 結果サマリー
            logger.info(f"\n🎯 最終結果:")
            logger.info(f"データセット: {len(self.df):,}件 ({self.df['Code'].nunique()}銘柄)")
            logger.info(f"検証精度: {results['overall_precision']:.4f} ({results['overall_precision']:.2%})")
            logger.info(f"検証期間: {results['n_days']}営業日")
            logger.info(f"予測成功: {results['total_correct']}/{results['total_predictions']}件")
            logger.info(f"保存先: {model_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"学習・検証エラー: {e}")
            return None

def main():
    """メイン実行"""
    trainer = Nikkei225CompletePrecisionTrainer()
    results = trainer.run_complete_training()
    
    if results:
        print(f"\n✅ 学習・検証完了!")
        print(f"📊 達成精度: {results['overall_precision']:.2%}")
        print(f"📈 検証実績: {results['total_correct']}/{results['total_predictions']}件")
        print(f"📅 検証期間: {results['n_days']}営業日間")
    else:
        print("\n❌ 学習・検証に失敗しました")

if __name__ == "__main__":
    main()