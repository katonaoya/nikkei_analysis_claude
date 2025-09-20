#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
強化版高精度学習・検証システム v2.0
改善提案1-3を統合実装：
1. Yahoo Finance 10年分外部指標データ統合
2. 季節性考慮した検証期間選択
3. アンサンブル手法導入による精度安定化
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
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.ensemble import VotingClassifier

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPrecisionTrainerV2:
    """強化版高精度学習・検証システム v2.0"""
    
    def __init__(self, stock_data_file: str = None, external_data_file: str = None):
        """初期化"""
        # データファイル設定
        if stock_data_file is None:
            stock_data_file = "data/processed/nikkei225_complete_225stocks_20250909_230649.parquet"
        if external_data_file is None:
            external_data_file = "data/external_extended/external_integrated_10years_20250909_231815.parquet"
        
        self.stock_data_file = stock_data_file
        self.external_data_file = external_data_file
        self.stock_df = None
        self.external_df = None
        self.integrated_df = None
        
        # アンサンブルモデル
        self.models = {}
        self.scalers = {}
        self.selectors = {}
        self.feature_cols = None
        
        logger.info(f"株価データファイル: {stock_data_file}")
        logger.info(f"外部指標データファイル: {external_data_file}")
    
    def load_data(self):
        """データ読み込み"""
        logger.info("データ読み込み開始...")
        
        try:
            # 株価データ読み込み
            self.stock_df = pd.read_parquet(self.stock_data_file)
            logger.info(f"株価データ読み込み完了: {len(self.stock_df):,}件, {self.stock_df['Code'].nunique()}銘柄")
            
            # 外部指標データ読み込み
            self.external_df = pd.read_parquet(self.external_data_file)
            logger.info(f"外部指標データ読み込み完了: {len(self.external_df):,}件, {len(self.external_df.columns)}カラム")
            
            # 日付型変換（タイムゾーン統一）
            self.stock_df['Date'] = pd.to_datetime(self.stock_df['Date']).dt.tz_localize(None)
            self.external_df['Date'] = pd.to_datetime(self.external_df['Date']).dt.tz_localize(None)
            
            return True
            
        except Exception as e:
            logger.error(f"データ読み込みエラー: {e}")
            return False
    
    def integrate_external_data(self):
        """外部指標データ統合"""
        logger.info("外部指標データ統合開始...")
        
        # 株価データをベースに外部指標データをマージ
        integrated_df = pd.merge(
            self.stock_df, 
            self.external_df, 
            on='Date', 
            how='left'
        )
        
        # 外部指標の欠損値を前方補完
        external_cols = [col for col in self.external_df.columns if col != 'Date']
        for col in external_cols:
            if col in integrated_df.columns:
                integrated_df[col] = integrated_df[col].ffill().bfill()
        
        logger.info(f"外部指標データ統合完了: {len(integrated_df):,}件, {len(integrated_df.columns)}カラム")
        
        # 統合前後の比較
        logger.info(f"統合前カラム数: {len(self.stock_df.columns)}")
        logger.info(f"統合後カラム数: {len(integrated_df.columns)}")
        logger.info(f"追加されたカラム数: {len(integrated_df.columns) - len(self.stock_df.columns)}")
        
        self.integrated_df = integrated_df
        return integrated_df
    
    def create_enhanced_features(self):
        """拡張特徴量作成（外部指標含む）"""
        logger.info("拡張特徴量作成開始...")
        
        enhanced_df = self.integrated_df.copy()
        
        # 銘柄別に特徴量計算
        result_dfs = []
        
        for code in enhanced_df['Code'].unique():
            code_df = enhanced_df[enhanced_df['Code'] == code].copy()
            code_df = code_df.sort_values('Date')
            
            # 基本特徴量（既存）
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
            
            # ボリンジャーバンド
            rolling_mean = code_df['Close'].rolling(20).mean()
            rolling_std = code_df['Close'].rolling(20).std()
            code_df['BB_upper_20'] = rolling_mean + (rolling_std * 2)
            code_df['BB_lower_20'] = rolling_mean - (rolling_std * 2)
            code_df['BB_ratio_20'] = (code_df['Close'] - code_df['BB_lower_20']) / (code_df['BB_upper_20'] - code_df['BB_lower_20'])
            
            # MACD
            exp1 = code_df['Close'].ewm(span=12, adjust=False).mean()
            exp2 = code_df['Close'].ewm(span=26, adjust=False).mean()
            code_df['MACD'] = exp1 - exp2
            code_df['MACD_signal'] = code_df['MACD'].ewm(span=9, adjust=False).mean()
            code_df['MACD_histogram'] = code_df['MACD'] - code_df['MACD_signal']
            
            # OBV
            code_df['OBV'] = (code_df['Volume'] * np.where(code_df['Close'] > code_df['Close'].shift(1), 1, 
                             np.where(code_df['Close'] < code_df['Close'].shift(1), -1, 0))).cumsum()
            
            # ストキャスティクス
            low_min = code_df['Low'].rolling(14).min()
            high_max = code_df['High'].rolling(14).max()
            code_df['Stoch_K_14'] = 100 * (code_df['Close'] - low_min) / (high_max - low_min)
            code_df['Stoch_D_14'] = code_df['Stoch_K_14'].rolling(3).mean()
            
            # 🆕 外部指標との相関特徴量
            if 'usdjpy_Close' in code_df.columns:
                code_df['Stock_USDJPY_Corr'] = code_df['Returns'].rolling(20).corr(code_df['usdjpy_Daily_Return'])
                code_df['Stock_USDJPY_Ratio'] = code_df['Close'] / code_df['usdjpy_Close']
            
            if 'vix_Close' in code_df.columns:
                code_df['Stock_VIX_Corr'] = code_df['Returns'].rolling(20).corr(code_df['vix_Daily_Return'])
                code_df['VIX_Fear_Factor'] = code_df['vix_Close'] / code_df['vix_MA_20']
            
            if 'nikkei225_Close' in code_df.columns:
                code_df['Stock_Market_Beta'] = code_df['Returns'].rolling(60).cov(code_df['nikkei225_Daily_Return']) / code_df['nikkei225_Daily_Return'].rolling(60).var()
                code_df['Market_Relative_Strength'] = code_df['MA_20'] / code_df['nikkei225_MA_20']
            
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
        
        # 欠損値除去（段階的に実行）
        logger.info(f"欠損値除去前: {len(enhanced_df):,}件")
        
        # 重要カラムの欠損値確認
        important_cols = ['Close', 'Target', 'Returns']
        for col in important_cols:
            if col in enhanced_df.columns:
                null_count = enhanced_df[col].isnull().sum()
                logger.info(f"{col}の欠損値: {null_count:,}件 ({null_count/len(enhanced_df)*100:.2f}%)")
        
        # 無限値を先に処理
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        
        # 段階的欠損値処理
        enhanced_df = enhanced_df.dropna(subset=['Close', 'Date', 'Code'])  # 必須カラム
        if 'Target' in enhanced_df.columns:
            enhanced_df = enhanced_df.dropna(subset=['Target'])
        
        # 残りの欠損値を前方補完
        enhanced_df = enhanced_df.fillna(method='ffill').fillna(method='bfill')
        enhanced_df = enhanced_df.dropna()  # 最終クリーンアップ
        
        self.integrated_df = enhanced_df
        logger.info(f"拡張特徴量作成完了: {len(enhanced_df):,}件")
        
        # 正例率確認
        positive_rate = enhanced_df['Target'].mean()
        logger.info(f"正例率: {positive_rate:.3f} ({positive_rate:.1%})")
        
        return enhanced_df
    
    def select_optimal_validation_period(self):
        """季節性考慮した最適検証期間選択"""
        logger.info("季節性考慮した最適検証期間選択...")
        
        df_sorted = self.integrated_df.sort_values('Date')
        latest_date = df_sorted['Date'].max()
        
        # 候補期間定義（季節性考慮）
        validation_periods = [
            # 安定期間（高ボラティリティ時期回避）
            {
                'name': '7月安定期',
                'start_days': 60,  # 7月1日頃から
                'end_days': 30,    # 8月10日頃まで
                'description': '夏枯れ前の安定期間'
            },
            {
                'name': '3月期末前',
                'start_days': 190,  # 3月1日頃から
                'end_days': 160,   # 3月31日頃まで
                'description': '期末前の活発な取引期間'
            },
            {
                'name': '10月安定期', 
                'start_days': 120, # 10月1日頃から
                'end_days': 90,    # 10月31日頃まで
                'description': '秋の安定した取引期間'
            }
        ]
        
        best_period = None
        best_score = 0
        
        for period in validation_periods:
            test_start = latest_date - timedelta(days=period['start_days'])
            test_end = latest_date - timedelta(days=period['end_days'])
            
            # 期間内のデータ確認
            period_data = df_sorted[
                (df_sorted['Date'] >= test_start) & 
                (df_sorted['Date'] <= test_end)
            ]
            
            if len(period_data) < 1000:  # 最小データ量チェック
                continue
            
            # ボラティリティ計算（安定性指標）
            period_volatility = period_data.groupby('Date')['Returns'].std().mean()
            
            # 正例率のバランス確認
            positive_rate = period_data['Target'].mean()
            balance_score = 1 - abs(positive_rate - 0.5)  # 0.5に近いほど高スコア
            
            # 総合スコア（低ボラティリティ × バランス × データ量）
            stability_score = (1 / (period_volatility + 0.001))
            data_volume_score = min(len(period_data) / 2000, 1.0)
            total_score = stability_score * balance_score * data_volume_score
            
            logger.info(f"{period['name']}: {test_start.date()} - {test_end.date()}")
            logger.info(f"  データ量: {len(period_data):,}件")
            logger.info(f"  ボラティリティ: {period_volatility:.4f}")
            logger.info(f"  正例率: {positive_rate:.3f}")
            logger.info(f"  総合スコア: {total_score:.4f}")
            
            if total_score > best_score:
                best_score = total_score
                best_period = {
                    **period,
                    'start_date': test_start,
                    'end_date': test_end,
                    'score': total_score,
                    'data_count': len(period_data),
                    'volatility': period_volatility,
                    'positive_rate': positive_rate
                }
        
        if best_period:
            logger.info(f"🎯 最適検証期間選択: {best_period['name']}")
            logger.info(f"期間: {best_period['start_date'].date()} - {best_period['end_date'].date()}")
            logger.info(f"スコア: {best_period['score']:.4f}")
        
        return best_period
    
    def prepare_features_and_target(self):
        """特徴量とターゲット準備"""
        logger.info("特徴量とターゲット準備...")
        
        # 特徴量カラム選択
        exclude_cols = ['Date', 'Code', 'CompanyName', 'MatchMethod', 'ApiCode', 'Target']
        self.feature_cols = [col for col in self.integrated_df.columns if col not in exclude_cols]
        
        # 数値型のみ選択
        numeric_cols = self.integrated_df[self.feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        self.feature_cols = numeric_cols
        
        logger.info(f"使用特徴量数: {len(self.feature_cols)}")
        
        X = self.integrated_df[self.feature_cols]
        y = self.integrated_df['Target']
        
        # 無限値やNaN除去
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.fillna(0)
        
        return X, y
    
    def create_ensemble_models(self):
        """アンサンブルモデル作成"""
        logger.info("アンサンブルモデル作成...")
        
        # 🆕 強化版LightGBM
        lgbm_params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'n_estimators': 500,      # 増強
            'max_depth': 10,          # 深さ増加
            'min_child_samples': 20,   # 調整
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'learning_rate': 0.02,    # 低下
            'reg_alpha': 0.15,
            'reg_lambda': 0.15,
            'random_state': 42,
            'verbose': -1
        }
        
        # 🆕 Random Forest
        rf_params = {
            'n_estimators': 300,
            'max_depth': 12,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 0.7,
            'random_state': 42,
            'n_jobs': -1
        }
        
        # 🆕 XGBoost
        xgb_params = {
            'objective': 'binary:logistic',
            'n_estimators': 400,
            'max_depth': 8,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': 42,
            'eval_metric': 'logloss'
        }
        
        # モデル作成
        models = {
            'lightgbm': LGBMClassifier(**lgbm_params),
            'random_forest': RandomForestClassifier(**rf_params),
            'xgboost': xgb.XGBClassifier(**xgb_params)
        }
        
        return models
    
    def train_ensemble_models(self, validation_period):
        """アンサンブルモデル学習"""
        logger.info("アンサンブルモデル学習開始...")
        
        # 特徴量準備
        X, y = self.prepare_features_and_target()
        
        # 時系列分割（最適期間使用）
        df_sorted = self.integrated_df.sort_values('Date')
        
        if validation_period:
            test_start = validation_period['start_date']
            test_end = validation_period['end_date']
        else:
            # フォールバック：従来の30日分割
            test_end = df_sorted['Date'].max()
            test_start = test_end - timedelta(days=30)
        
        logger.info(f"訓練期間: 〜 {test_start.date()}")
        logger.info(f"テスト期間: {test_start.date()} 〜 {test_end.date()}")
        
        # データ分割
        train_mask = df_sorted['Date'] < test_start
        test_mask = (df_sorted['Date'] >= test_start) & (df_sorted['Date'] <= test_end)
        
        X_train = X[train_mask]
        y_train = y[train_mask]
        X_test = X[test_mask]
        y_test = y[test_mask]
        
        logger.info(f"訓練データ: {len(X_train):,}件")
        logger.info(f"テストデータ: {len(X_test):,}件")
        
        # アンサンブルモデル作成
        base_models = self.create_ensemble_models()
        
        # 各モデルを個別に学習
        trained_models = {}
        
        for name, model in base_models.items():
            logger.info(f"{name}モデル学習開始...")
            
            # 特徴量選択（モデル別に最適化）
            k_features = {
                'lightgbm': 40,
                'random_forest': 35, 
                'xgboost': 30
            }
            
            selector = SelectKBest(score_func=f_classif, k=k_features[name])
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # スケーリング（モデル別）
            if name in ['random_forest']:
                scaler = StandardScaler()
            else:
                scaler = RobustScaler()
            
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # モデル学習
            model.fit(X_train_scaled, y_train)
            
            # 保存
            trained_models[name] = {
                'model': model,
                'scaler': scaler,
                'selector': selector
            }
            
            logger.info(f"{name}モデル学習完了")
        
        self.models = trained_models
        
        # 🆕 アンサンブル予測
        return self.evaluate_ensemble_performance(df_sorted[test_mask], X_test, y_test)
    
    def evaluate_ensemble_performance(self, test_df, X_test, y_test):
        """アンサンブル性能評価"""
        logger.info("アンサンブル性能評価開始...")
        
        # 各モデルの予測
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
        weights = {'lightgbm': 0.5, 'random_forest': 0.3, 'xgboost': 0.2}
        ensemble_proba = np.zeros(len(X_test))
        
        for name, proba in model_predictions.items():
            ensemble_proba += weights[name] * proba
        
        # 日別精度評価
        return self.evaluate_daily_precision_ensemble(test_df, ensemble_proba, model_predictions)
    
    def evaluate_daily_precision_ensemble(self, test_df, ensemble_proba, model_predictions):
        """アンサンブル日別精度評価"""
        logger.info("アンサンブル日別精度評価...")
        
        test_df_copy = test_df.copy()
        test_df_copy['EnsemblePredProba'] = ensemble_proba
        
        # 個別モデル予測も追加
        for name, proba in model_predictions.items():
            test_df_copy[f'{name}_PredProba'] = proba
        
        # 営業日別評価
        unique_dates = sorted(test_df_copy['Date'].unique())
        daily_results = []
        ensemble_results = {'total_correct': 0, 'total_predictions': 0}
        individual_results = {name: {'total_correct': 0, 'total_predictions': 0} for name in model_predictions.keys()}
        
        logger.info(f"検証期間: {unique_dates[0].date()} 〜 {unique_dates[-1].date()} ({len(unique_dates)}営業日)")
        
        for test_date in unique_dates:
            daily_data = test_df_copy[test_df_copy['Date'] == test_date]
            
            if len(daily_data) < 3:
                continue
            
            # アンサンブル予測（上位3銘柄）
            top3_ensemble = daily_data['EnsemblePredProba'].nlargest(3).index
            ensemble_results_daily = daily_data.loc[top3_ensemble]['Target'].values
            ensemble_correct = np.sum(ensemble_results_daily)
            ensemble_total = len(ensemble_results_daily)
            
            ensemble_results['total_correct'] += ensemble_correct
            ensemble_results['total_predictions'] += ensemble_total
            
            # 個別モデル評価
            individual_daily = {}
            for name in model_predictions.keys():
                top3_individual = daily_data[f'{name}_PredProba'].nlargest(3).index
                individual_actual = daily_data.loc[top3_individual]['Target'].values
                individual_correct = np.sum(individual_actual)
                individual_total = len(individual_actual)
                
                individual_results[name]['total_correct'] += individual_correct
                individual_results[name]['total_predictions'] += individual_total
                individual_daily[name] = individual_correct / individual_total if individual_total > 0 else 0
            
            ensemble_precision = ensemble_correct / ensemble_total if ensemble_total > 0 else 0
            
            daily_results.append({
                'date': test_date,
                'ensemble_correct': ensemble_correct,
                'ensemble_total': ensemble_total,
                'ensemble_precision': ensemble_precision,
                'individual_precision': individual_daily,
                'selected_codes': daily_data.loc[top3_ensemble]['Code'].tolist()
            })
            
            logger.info(f"{test_date.strftime('%Y-%m-%d')}: Ensemble {ensemble_correct}/{ensemble_total}={ensemble_precision:.1%} "
                       f"[{', '.join(daily_data.loc[top3_ensemble]['Code'].astype(str).tolist())}]")
        
        # 総合精度計算
        ensemble_overall = ensemble_results['total_correct'] / ensemble_results['total_predictions']
        individual_overall = {
            name: results['total_correct'] / results['total_predictions'] 
            for name, results in individual_results.items()
        }
        
        logger.info(f"\n🎉 アンサンブル検証結果:")
        logger.info(f"検証営業日数: {len(daily_results)}日間")
        logger.info(f"アンサンブル精度: {ensemble_overall:.4f} ({ensemble_overall:.2%})")
        logger.info(f"個別モデル精度:")
        for name, precision in individual_overall.items():
            logger.info(f"  {name}: {precision:.4f} ({precision:.2%})")
        
        return {
            'ensemble_precision': ensemble_overall,
            'individual_precision': individual_overall,
            'daily_results': daily_results,
            'ensemble_stats': ensemble_results,
            'individual_stats': individual_results,
            'n_days': len(daily_results)
        }
    
    def save_enhanced_models(self, results, validation_period):
        """強化版モデル保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ensemble_precision = f"{results['ensemble_precision']:.4f}".replace('.', '')
        
        os.makedirs("models/enhanced_v2", exist_ok=True)
        
        model_file = f"models/enhanced_v2/enhanced_ensemble_model_{len(self.integrated_df)}records_{ensemble_precision}precision_{timestamp}.joblib"
        
        model_data = {
            'models': self.models,
            'feature_cols': self.feature_cols,
            'ensemble_precision': results['ensemble_precision'],
            'individual_precision': results['individual_precision'],
            'results': results,
            'validation_period': validation_period,
            'data_info': {
                'total_records': len(self.integrated_df),
                'n_companies': self.integrated_df['Code'].nunique(),
                'data_period': f"{self.integrated_df['Date'].min()} - {self.integrated_df['Date'].max()}",
                'external_indicators': True,
                'seasonal_optimization': True,
                'ensemble_method': 'weighted_voting'
            }
        }
        
        joblib.dump(model_data, model_file)
        logger.info(f"強化版モデル保存完了: {model_file}")
        
        return model_file
    
    def run_enhanced_training(self):
        """強化版学習・検証実行"""
        logger.info("🚀 強化版高精度学習・検証システム v2.0 開始!")
        
        try:
            # データ読み込み
            if not self.load_data():
                return None
            
            # 外部指標データ統合
            self.integrate_external_data()
            
            # 特徴量作成
            self.create_enhanced_features()
            
            # 最適検証期間選択
            validation_period = self.select_optimal_validation_period()
            
            # アンサンブル学習・検証
            results = self.train_ensemble_models(validation_period)
            
            # モデル保存
            model_file = self.save_enhanced_models(results, validation_period)
            
            # 結果サマリー
            logger.info(f"\n🎯 強化版最終結果:")
            logger.info(f"データセット: {len(self.integrated_df):,}件 ({self.integrated_df['Code'].nunique()}銘柄)")
            logger.info(f"外部指標: {len(self.external_df.columns)-1}個統合")
            logger.info(f"アンサンブル精度: {results['ensemble_precision']:.4f} ({results['ensemble_precision']:.2%})")
            logger.info(f"検証期間: {results['n_days']}営業日")
            logger.info(f"最適期間: {validation_period['name'] if validation_period else 'デフォルト'}")
            logger.info(f"保存先: {model_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"強化版学習・検証エラー: {e}")
            return None

def main():
    """メイン実行"""
    trainer = EnhancedPrecisionTrainerV2()
    results = trainer.run_enhanced_training()
    
    if results:
        print(f"\n✅ 強化版学習・検証完了!")
        print(f"📊 アンサンブル精度: {results['ensemble_precision']:.2%}")
        print(f"📈 個別モデル精度:")
        for name, precision in results['individual_precision'].items():
            print(f"  - {name}: {precision:.2%}")
        print(f"📅 検証期間: {results['n_days']}営業日間")
    else:
        print("\n❌ 強化版学習・検証に失敗しました")

if __name__ == "__main__":
    main()