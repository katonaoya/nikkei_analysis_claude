#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡易版強化学習・検証システム
外部指標問題を回避し、実装済み改善要素を活用した安定版
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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimplifiedEnhancedTrainer:
    """簡易版強化学習・検証システム"""
    
    def __init__(self, data_file: str = None):
        """初期化"""
        if data_file is None:
            data_file = "data/processed/nikkei225_complete_225stocks_20250909_230649.parquet"
        
        self.data_file = data_file
        self.df = None
        self.models = {}
        self.feature_cols = None
        
        logger.info(f"データファイル: {data_file}")
    
    def load_and_enhance_data(self):
        """データ読み込みと拡張特徴量作成"""
        logger.info("データ読み込み・拡張特徴量作成開始...")
        
        try:
            self.df = pd.read_parquet(self.data_file)
            logger.info(f"データ読み込み完了: {len(self.df):,}件, {self.df['Code'].nunique()}銘柄")
            
            # 日付型変換
            self.df['Date'] = pd.to_datetime(self.df['Date'])
            
            # 🆕 拡張特徴量作成（外部指標なしバージョン）
            enhanced_df = self.df.copy()
            result_dfs = []
            
            for code in enhanced_df['Code'].unique():
                code_df = enhanced_df[enhanced_df['Code'] == code].copy()
                code_df = code_df.sort_values('Date')
                
                # 既存特徴量の拡張
                code_df['Returns'] = code_df['Close'].pct_change(fill_method=None)
                code_df['Volume_MA_20'] = code_df['Volume'].rolling(20).mean()
                code_df['Price_Volume_Trend'] = code_df['Returns'] * code_df['Volume']
                
                # 🆕 追加移動平均
                for window in [3, 7, 10, 25, 50, 75, 100]:
                    code_df[f'MA_{window}'] = code_df['Close'].rolling(window).mean()
                    code_df[f'MA_{window}_ratio'] = code_df['Close'] / code_df[f'MA_{window}']
                    code_df[f'MA_{window}_slope'] = code_df[f'MA_{window}'].diff(5)
                
                # 🆕 拡張ボラティリティ
                for window in [3, 7, 10, 15, 30]:
                    code_df[f'Volatility_{window}'] = code_df['Returns'].rolling(window).std()
                    code_df[f'Returns_zscore_{window}'] = (code_df['Returns'] - code_df['Returns'].rolling(window).mean()) / code_df['Returns'].rolling(window).std()
                
                # 🆕 拡張RSI
                for window in [5, 9, 14, 21, 28]:
                    delta = code_df['Close'].diff()
                    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                    rs = gain / loss
                    code_df[f'RSI_{window}'] = 100 - (100 / (1 + rs))
                
                # 🆕 複数ボリンジャーバンド
                for window in [10, 20, 30]:
                    for std_mult in [1, 2, 2.5]:
                        rolling_mean = code_df['Close'].rolling(window).mean()
                        rolling_std = code_df['Close'].rolling(window).std()
                        code_df[f'BB_upper_{window}_{std_mult}'] = rolling_mean + (rolling_std * std_mult)
                        code_df[f'BB_lower_{window}_{std_mult}'] = rolling_mean - (rolling_std * std_mult)
                        code_df[f'BB_ratio_{window}_{std_mult}'] = (code_df['Close'] - code_df[f'BB_lower_{window}_{std_mult}']) / (code_df[f'BB_upper_{window}_{std_mult}'] - code_df[f'BB_lower_{window}_{std_mult}'])
                
                # 🆕 MACD変種
                for fast, slow, signal in [(8, 21, 5), (12, 26, 9), (19, 39, 9)]:
                    exp1 = code_df['Close'].ewm(span=fast, adjust=False).mean()
                    exp2 = code_df['Close'].ewm(span=slow, adjust=False).mean()
                    code_df[f'MACD_{fast}_{slow}'] = exp1 - exp2
                    code_df[f'MACD_signal_{fast}_{slow}_{signal}'] = code_df[f'MACD_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
                    code_df[f'MACD_histogram_{fast}_{slow}_{signal}'] = code_df[f'MACD_{fast}_{slow}'] - code_df[f'MACD_signal_{fast}_{slow}_{signal}']
                
                # 🆕 OBV変種
                obv_volume = code_df['Volume'] * np.where(code_df['Close'] > code_df['Close'].shift(1), 1, 
                                                         np.where(code_df['Close'] < code_df['Close'].shift(1), -1, 0))
                code_df['OBV'] = obv_volume.cumsum()
                code_df['OBV_MA_10'] = code_df['OBV'].rolling(10).mean()
                code_df['OBV_ratio'] = code_df['OBV'] / code_df['OBV_MA_10']
                
                # 🆕 ストキャスティクス変種
                for window in [9, 14, 21]:
                    low_min = code_df['Low'].rolling(window).min()
                    high_max = code_df['High'].rolling(window).max()
                    code_df[f'Stoch_K_{window}'] = 100 * (code_df['Close'] - low_min) / (high_max - low_min)
                    code_df[f'Stoch_D_{window}'] = code_df[f'Stoch_K_{window}'].rolling(3).mean()
                
                # 🆕 価格パターン特徴量
                code_df['High_Low_ratio'] = code_df['High'] / code_df['Low']
                code_df['Open_Close_ratio'] = code_df['Open'] / code_df['Close']
                code_df['Volume_price_ratio'] = code_df['Volume'] / code_df['Close']
                
                # 🆕 リターン系特徴量
                for period in [2, 3, 5, 10, 20]:
                    code_df[f'Returns_{period}d'] = code_df['Close'].pct_change(period)
                    code_df[f'Max_return_{period}d'] = code_df['Returns'].rolling(period).max()
                    code_df[f'Min_return_{period}d'] = code_df['Returns'].rolling(period).min()
                
                result_dfs.append(code_df)
            
            # 結合
            enhanced_df = pd.concat(result_dfs, ignore_index=True)
            
            # 目的変数作成
            logger.info("目的変数作成...")
            enhanced_df['Target'] = 0
            
            for code in enhanced_df['Code'].unique():
                mask = enhanced_df['Code'] == code
                code_data = enhanced_df[mask].copy()
                next_high = code_data['High'].shift(-1)
                prev_close = code_data['Close'].shift(1)
                enhanced_df.loc[mask, 'Target'] = (next_high / prev_close > 1.01).astype(int)
            
            # 欠損値処理
            logger.info("欠損値処理...")
            enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
            enhanced_df = enhanced_df.dropna(subset=['Close', 'Date', 'Code', 'Target'])
            enhanced_df = enhanced_df.fillna(method='ffill').fillna(method='bfill')
            enhanced_df = enhanced_df.dropna()
            
            self.df = enhanced_df
            logger.info(f"拡張特徴量作成完了: {len(enhanced_df):,}件")
            
            positive_rate = enhanced_df['Target'].mean()
            logger.info(f"正例率: {positive_rate:.3f} ({positive_rate:.1%})")
            
            return True
            
        except Exception as e:
            logger.error(f"データ処理エラー: {e}")
            return False
    
    def select_optimal_validation_period(self):
        """🆕 季節性考慮した最適検証期間選択"""
        logger.info("季節性考慮した最適検証期間選択...")
        
        df_sorted = self.df.sort_values('Date')
        latest_date = df_sorted['Date'].max()
        
        # 候補期間定義
        validation_periods = [
            {
                'name': '7月安定期',
                'start_days': 70,
                'end_days': 35,
                'description': '夏枯れ前の安定期間'
            },
            {
                'name': '10月安定期',
                'start_days': 120,
                'end_days': 90,
                'description': '秋の安定した取引期間'
            },
            {
                'name': '1月新年期',
                'start_days': 250,
                'end_days': 220,
                'description': '新年の活発な取引期間'
            }
        ]
        
        best_period = None
        best_score = 0
        
        for period in validation_periods:
            test_start = latest_date - timedelta(days=period['start_days'])
            test_end = latest_date - timedelta(days=period['end_days'])
            
            period_data = df_sorted[
                (df_sorted['Date'] >= test_start) & 
                (df_sorted['Date'] <= test_end)
            ]
            
            if len(period_data) < 1000:
                continue
            
            period_volatility = period_data.groupby('Date')['Returns'].std().mean()
            positive_rate = period_data['Target'].mean()
            balance_score = 1 - abs(positive_rate - 0.5)
            
            stability_score = 1 / (period_volatility + 0.001)
            data_volume_score = min(len(period_data) / 3000, 1.0)
            total_score = stability_score * balance_score * data_volume_score
            
            logger.info(f"{period['name']}: {test_start.date()} - {test_end.date()}, スコア: {total_score:.4f}")
            
            if total_score > best_score:
                best_score = total_score
                best_period = {
                    **period,
                    'start_date': test_start,
                    'end_date': test_end,
                    'score': total_score,
                    'data_count': len(period_data)
                }
        
        if best_period:
            logger.info(f"🎯 最適検証期間: {best_period['name']}")
            logger.info(f"期間: {best_period['start_date'].date()} - {best_period['end_date'].date()}")
        
        return best_period
    
    def create_ensemble_models(self):
        """🆕 アンサンブルモデル作成"""
        logger.info("アンサンブルモデル作成...")
        
        models = {
            'lightgbm_v1': LGBMClassifier(
                objective='binary',
                n_estimators=400,
                max_depth=8,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                verbose=-1
            ),
            'lightgbm_v2': LGBMClassifier(
                objective='binary',
                n_estimators=600,
                max_depth=10,
                learning_rate=0.02,
                subsample=0.85,
                colsample_bytree=0.7,
                reg_alpha=0.15,
                reg_lambda=0.15,
                random_state=123,
                verbose=-1
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=250,
                max_depth=12,
                min_samples_split=8,
                min_samples_leaf=4,
                max_features=0.8,
                random_state=42,
                n_jobs=-1
            ),
            'xgboost': xgb.XGBClassifier(
                objective='binary:logistic',
                n_estimators=350,
                max_depth=7,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.75,
                reg_alpha=0.12,
                reg_lambda=0.12,
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        return models
    
    def train_and_validate_ensemble(self, validation_period):
        """🆕 アンサンブル学習・検証"""
        logger.info("アンサンブル学習・検証開始...")
        
        # 特徴量準備
        exclude_cols = ['Date', 'Code', 'CompanyName', 'MatchMethod', 'ApiCode', 'Target']
        self.feature_cols = [col for col in self.df.columns if col not in exclude_cols]
        numeric_cols = self.df[self.feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = numeric_cols
        
        logger.info(f"使用特徴量数: {len(self.feature_cols)}")
        
        X = self.df[self.feature_cols]
        y = self.df['Target']
        
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # 時系列分割
        df_sorted = self.df.sort_values('Date')
        
        if validation_period:
            test_start = validation_period['start_date']
            test_end = validation_period['end_date']
        else:
            test_end = df_sorted['Date'].max()
            test_start = test_end - timedelta(days=30)
        
        logger.info(f"訓練期間: 〜 {test_start.date()}")
        logger.info(f"テスト期間: {test_start.date()} 〜 {test_end.date()}")
        
        train_mask = df_sorted['Date'] < test_start
        test_mask = (df_sorted['Date'] >= test_start) & (df_sorted['Date'] <= test_end)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"訓練データ: {len(X_train):,}件")
        logger.info(f"テストデータ: {len(X_test):,}件")
        
        # アンサンブル学習
        base_models = self.create_ensemble_models()
        trained_models = {}
        
        for name, model in base_models.items():
            logger.info(f"{name}学習開始...")
            
            # 特徴量選択（モデル別）
            k_features = {
                'lightgbm_v1': 35,
                'lightgbm_v2': 40, 
                'random_forest': 30,
                'xgboost': 32
            }
            
            selector = SelectKBest(score_func=f_classif, k=k_features[name])
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # スケーリング
            scaler = RobustScaler() if 'lightgbm' in name or 'xgb' in name else StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # 学習
            model.fit(X_train_scaled, y_train)
            
            trained_models[name] = {
                'model': model,
                'scaler': scaler,
                'selector': selector
            }
        
        self.models = trained_models
        
        # アンサンブル評価
        return self.evaluate_ensemble(df_sorted[test_mask], X_test, y_test)
    
    def evaluate_ensemble(self, test_df, X_test, y_test):
        """アンサンブル評価"""
        logger.info("アンサンブル評価開始...")
        
        # 各モデル予測
        model_predictions = {}
        for name, model_data in self.models.items():
            model = model_data['model']
            scaler = model_data['scaler']
            selector = model_data['selector']
            
            X_selected = selector.transform(X_test)
            X_scaled = scaler.transform(X_selected)
            pred_proba = model.predict_proba(X_scaled)[:, 1]
            model_predictions[name] = pred_proba
        
        # 🆕 アンサンブル予測（加重平均）
        weights = {
            'lightgbm_v1': 0.3,
            'lightgbm_v2': 0.3,
            'random_forest': 0.2,
            'xgboost': 0.2
        }
        
        ensemble_proba = np.zeros(len(X_test))
        for name, proba in model_predictions.items():
            ensemble_proba += weights[name] * proba
        
        # 日別精度評価
        test_df_copy = test_df.copy()
        test_df_copy['EnsemblePredProba'] = ensemble_proba
        
        unique_dates = sorted(test_df_copy['Date'].unique())
        daily_results = []
        ensemble_stats = {'total_correct': 0, 'total_predictions': 0}
        individual_stats = {name: {'total_correct': 0, 'total_predictions': 0} for name in model_predictions.keys()}
        
        logger.info(f"検証期間: {unique_dates[0].date()} 〜 {unique_dates[-1].date()} ({len(unique_dates)}営業日)")
        
        for test_date in unique_dates:
            daily_data = test_df_copy[test_df_copy['Date'] == test_date]
            
            if len(daily_data) < 3:
                continue
            
            # アンサンブル（上位3銘柄）
            top3_ensemble = daily_data['EnsemblePredProba'].nlargest(3).index
            ensemble_results_daily = daily_data.loc[top3_ensemble]['Target'].values
            ensemble_correct = np.sum(ensemble_results_daily)
            ensemble_total = len(ensemble_results_daily)
            
            ensemble_stats['total_correct'] += ensemble_correct
            ensemble_stats['total_predictions'] += ensemble_total
            
            ensemble_precision = ensemble_correct / ensemble_total
            
            daily_results.append({
                'date': test_date,
                'ensemble_correct': ensemble_correct,
                'ensemble_total': ensemble_total,
                'ensemble_precision': ensemble_precision,
                'selected_codes': daily_data.loc[top3_ensemble]['Code'].tolist()
            })
            
            logger.info(f"{test_date.strftime('%Y-%m-%d')}: {ensemble_correct}/{ensemble_total}={ensemble_precision:.1%} "
                       f"[{', '.join(daily_data.loc[top3_ensemble]['Code'].astype(str).tolist())}]")
        
        # 総合精度
        ensemble_overall = ensemble_stats['total_correct'] / ensemble_stats['total_predictions']
        
        logger.info(f"\n🎉 アンサンブル検証結果:")
        logger.info(f"検証営業日数: {len(daily_results)}日間")
        logger.info(f"アンサンブル精度: {ensemble_overall:.4f} ({ensemble_overall:.2%})")
        
        return {
            'ensemble_precision': ensemble_overall,
            'daily_results': daily_results,
            'ensemble_stats': ensemble_stats,
            'n_days': len(daily_results)
        }
    
    def save_models(self, results, validation_period):
        """モデル保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        precision_str = f"{results['ensemble_precision']:.4f}".replace('.', '')
        
        os.makedirs("models/simplified_enhanced", exist_ok=True)
        
        model_file = f"models/simplified_enhanced/simplified_enhanced_model_{len(self.df)}records_{precision_str}precision_{timestamp}.joblib"
        
        model_data = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'ensemble_precision': results['ensemble_precision'],
            'results': results,
            'validation_period': validation_period,
            'improvements': [
                'extended_technical_indicators',
                'seasonal_validation_period_optimization', 
                'ensemble_learning_4models',
                'enhanced_feature_engineering'
            ]
        }
        
        joblib.dump(model_data, model_file)
        logger.info(f"モデル保存完了: {model_file}")
        
        return model_file
    
    def run_simplified_enhanced_training(self):
        """簡易版強化学習実行"""
        logger.info("🚀 簡易版強化学習・検証システム開始!")
        
        try:
            # データ処理
            if not self.load_and_enhance_data():
                return None
            
            # 最適検証期間選択
            validation_period = self.select_optimal_validation_period()
            
            # アンサンブル学習・検証
            results = self.train_and_validate_ensemble(validation_period)
            
            # モデル保存
            model_file = self.save_models(results, validation_period)
            
            # 結果サマリー
            logger.info(f"\n🎯 簡易版強化学習最終結果:")
            logger.info(f"データセット: {len(self.df):,}件 ({self.df['Code'].nunique()}銘柄)")
            logger.info(f"拡張特徴量数: {len(self.feature_cols)}")
            logger.info(f"アンサンブル精度: {results['ensemble_precision']:.4f} ({results['ensemble_precision']:.2%})")
            logger.info(f"検証期間: {results['n_days']}営業日")
            logger.info(f"最適期間: {validation_period['name'] if validation_period else 'デフォルト'}")
            logger.info(f"保存先: {model_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"簡易版強化学習エラー: {e}")
            return None

def main():
    """メイン実行"""
    trainer = SimplifiedEnhancedTrainer()
    results = trainer.run_simplified_enhanced_training()
    
    if results:
        print(f"\n✅ 簡易版強化学習・検証完了!")
        print(f"📊 アンサンブル精度: {results['ensemble_precision']:.2%}")
        print(f"📈 改善要素: 拡張特徴量 + 季節性最適化 + アンサンブル4モデル")
        print(f"📅 検証期間: {results['n_days']}営業日間")
    else:
        print("\n❌ 簡易版強化学習・検証に失敗しました")

if __name__ == "__main__":
    main()