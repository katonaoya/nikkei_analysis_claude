#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Precision System V3
外部データ統合 + 厳密なバックテストによる改善版

改善点：
1. 外部経済指標データの統合（USD/JPY, VIX, 日経225指数等）
2. ウォークフォワード最適化による厳密なバックテスト
3. 複雑性を抑えたシンプルな実装
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime, timedelta
import logging
import warnings
import os
from pathlib import Path

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedPrecisionSystemV3:
    """外部データ統合 + 厳密バックテスト版"""
    
    def __init__(self, stock_file: str = None, external_file: str = None):
        """初期化"""
        # デフォルトファイルパス
        if stock_file is None:
            stock_file = "data/processed/nikkei225_complete_225stocks_20250909_230649.parquet"
        if external_file is None:
            external_file = "data/external_extended/external_integrated_10years_20250909_231815.parquet"
            
        self.stock_file = stock_file
        self.external_file = external_file
        
        # 保存ディレクトリ
        self.output_dir = Path("models/enhanced_v3")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("🎯 Enhanced Precision System V3 初期化完了")
        logger.info(f"株価データ: {self.stock_file}")
        logger.info(f"外部データ: {self.external_file}")
    
    def load_and_integrate_data(self) -> pd.DataFrame:
        """データ読み込みと統合"""
        logger.info("📊 データ読み込み開始...")
        
        # 株価データ読み込み
        stock_df = pd.read_parquet(self.stock_file)
        logger.info(f"株価データ: {len(stock_df):,}件, {stock_df['Code'].nunique()}銘柄")
        
        # 外部データ読み込み（ファイルが存在する場合のみ）
        external_df = None
        if os.path.exists(self.external_file):
            try:
                external_df = pd.read_parquet(self.external_file)
                logger.info(f"外部データ: {len(external_df):,}件, {len(external_df.columns)}カラム")
            except Exception as e:
                logger.warning(f"外部データ読み込みエラー: {e}")
                external_df = None
        else:
            logger.warning(f"外部データファイルが見つかりません: {self.external_file}")
        
        # 日付型統一
        stock_df['Date'] = pd.to_datetime(stock_df['Date']).dt.tz_localize(None)
        
        # 外部データとの統合
        if external_df is not None:
            try:
                external_df['Date'] = pd.to_datetime(external_df['Date']).dt.tz_localize(None)
                
                # 重要な外部指標のみ選択（複雑性を抑制）
                important_external_cols = ['Date']
                for col in external_df.columns:
                    if any(key in col.lower() for key in ['usdjpy', 'vix', 'nikkei225_close', 'sp500_close']):
                        important_external_cols.append(col)
                
                if len(important_external_cols) > 1:
                    external_selected = external_df[important_external_cols].copy()
                    stock_df = pd.merge(stock_df, external_selected, on='Date', how='left')
                    logger.info(f"外部データ統合完了: {len(important_external_cols)-1}指標")
                else:
                    logger.warning("重要な外部指標が見つかりませんでした")
                    
            except Exception as e:
                logger.warning(f"外部データ統合エラー: {e}")
        
        logger.info(f"統合後データ: {len(stock_df):,}件, {len(stock_df.columns)}カラム")
        return stock_df
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """拡張特徴量作成（シンプル版）"""
        logger.info("🔥 拡張特徴量エンジニアリング開始...")
        
        enhanced_df = df.copy()
        
        # 銘柄別に特徴量作成
        for code in enhanced_df['Code'].unique():
            mask = enhanced_df['Code'] == code
            code_data = enhanced_df[mask].copy().sort_values('Date')
            
            # 基本特徴量
            code_data['Returns'] = code_data['Close'].pct_change()
            code_data['Log_Returns'] = np.log(code_data['Close'] / code_data['Close'].shift(1))
            code_data['High_Low_Ratio'] = code_data['High'] / code_data['Low']
            
            # 移動平均（複数期間）
            for window in [5, 10, 20, 50]:
                code_data[f'MA_{window}'] = code_data['Close'].rolling(window).mean()
                code_data[f'MA_{window}_ratio'] = code_data['Close'] / code_data[f'MA_{window}']
            
            # ボラティリティ
            for window in [5, 20]:
                code_data[f'Volatility_{window}'] = code_data['Returns'].rolling(window).std()
            
            # RSI
            for window in [14, 30]:
                delta = code_data['Close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
                rs = gain / loss
                code_data[f'RSI_{window}'] = 100 - (100 / (1 + rs))
            
            # ボリンジャーバンド
            rolling_mean = code_data['Close'].rolling(20).mean()
            rolling_std = code_data['Close'].rolling(20).std()
            code_data['BB_upper'] = rolling_mean + (rolling_std * 2)
            code_data['BB_lower'] = rolling_mean - (rolling_std * 2)
            code_data['BB_ratio'] = (code_data['Close'] - code_data['BB_lower']) / (code_data['BB_upper'] - code_data['BB_lower'])
            
            # MACD
            exp1 = code_data['Close'].ewm(span=12).mean()
            exp2 = code_data['Close'].ewm(span=26).mean()
            code_data['MACD'] = exp1 - exp2
            code_data['MACD_signal'] = code_data['MACD'].ewm(span=9).mean()
            code_data['MACD_histogram'] = code_data['MACD'] - code_data['MACD_signal']
            
            # ボリューム特徴量
            code_data['Volume_MA_20'] = code_data['Volume'].rolling(20).mean()
            code_data['Volume_ratio'] = code_data['Volume'] / code_data['Volume_MA_20']
            
            # 外部データとの相関特徴量（外部データがある場合）
            for col in code_data.columns:
                if any(key in col.lower() for key in ['usdjpy', 'vix', 'nikkei225', 'sp500']):
                    if code_data[col].notna().sum() > 100:  # 十分なデータがある場合のみ
                        # 外部指標との比率
                        code_data[f'{col}_change'] = code_data[col].pct_change()
                        # 5日移動平均
                        code_data[f'{col}_MA5'] = code_data[col].rolling(5).mean()
            
            enhanced_df.loc[mask] = code_data
        
        # 目的変数作成
        logger.info("目的変数作成...")
        enhanced_df['Target'] = 0
        for code in enhanced_df['Code'].unique():
            mask = enhanced_df['Code'] == code
            code_data = enhanced_df[mask].copy()
            next_high = code_data['High'].shift(-1)
            prev_close = code_data['Close'].shift(1)
            enhanced_df.loc[mask, 'Target'] = (next_high / prev_close > 1.01).astype(int)
        
        # 無限値・欠損値処理
        enhanced_df = enhanced_df.replace([np.inf, -np.inf], np.nan)
        enhanced_df = enhanced_df.dropna(subset=['Close', 'Date', 'Code', 'Target'])
        
        logger.info(f"🔥 特徴量作成完了: {len(enhanced_df):,}件")
        logger.info(f"特徴量数: {len(enhanced_df.columns)}カラム")
        logger.info(f"正例率: {enhanced_df['Target'].mean():.3f}")
        
        return enhanced_df
    
    def walk_forward_optimization(self, df: pd.DataFrame, initial_train_size: int = 252*3) -> list:
        """ウォークフォワード最適化"""
        logger.info("📈 ウォークフォワード最適化開始...")
        
        # 日付でソート
        df_sorted = df.sort_values(['Date', 'Code']).copy()
        unique_dates = sorted(df_sorted['Date'].unique())
        
        results = []
        step_size = 21  # 月次リバランス
        
        # 特徴量カラム選択
        feature_cols = [col for col in df_sorted.columns 
                       if col not in ['Date', 'Code', 'Target'] and 
                       df_sorted[col].dtype in ['int64', 'float64']]
        
        logger.info(f"使用特徴量数: {len(feature_cols)}")
        
        for i in range(initial_train_size, len(unique_dates) - step_size, step_size):
            try:
                # 期間設定
                train_end_idx = i
                test_start_idx = i
                test_end_idx = min(i + step_size, len(unique_dates))
                
                train_dates = unique_dates[:train_end_idx]
                test_dates = unique_dates[test_start_idx:test_end_idx]
                
                # データ分割
                train_df = df_sorted[df_sorted['Date'].isin(train_dates)]
                test_df = df_sorted[df_sorted['Date'].isin(test_dates)]
                
                if len(train_df) == 0 or len(test_df) == 0:
                    continue
                
                # 特徴量・目的変数分離
                X_train = train_df[feature_cols]
                y_train = train_df['Target']
                X_test = test_df[feature_cols]
                y_test = test_df['Target']
                
                # 欠損値処理
                X_train = X_train.fillna(method='ffill').fillna(0)
                X_test = X_test.fillna(method='ffill').fillna(0)
                
                # 特徴量選択
                selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_cols)))
                X_train_selected = selector.fit_transform(X_train, y_train)
                X_test_selected = selector.transform(X_test)
                
                # スケーリング
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train_selected)
                X_test_scaled = scaler.transform(X_test_selected)
                
                # モデル学習
                model = lgb.LGBMClassifier(
                    objective='binary',
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    random_state=42,
                    verbose=-1
                )
                
                model.fit(X_train_scaled, y_train)
                
                # 予測
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # 評価
                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, zero_division=0)
                recall = recall_score(y_test, y_pred, zero_division=0)
                f1 = f1_score(y_test, y_pred, zero_division=0)
                
                result = {
                    'period': f"{train_dates[-1].strftime('%Y-%m-%d')} to {test_dates[-1].strftime('%Y-%m-%d')}",
                    'train_size': len(train_df),
                    'test_size': len(test_df),
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'positive_rate': y_test.mean()
                }
                
                results.append(result)
                logger.info(f"期間 {result['period']}: 精度={accuracy:.4f}, 適合率={precision:.4f}")
                
            except Exception as e:
                logger.warning(f"期間 {i} でエラー: {e}")
                continue
        
        return results
    
    def train_final_model(self, df: pd.DataFrame) -> dict:
        """最終モデル学習"""
        logger.info("🤖 最終モデル学習開始...")
        
        # 特徴量準備
        feature_cols = [col for col in df.columns 
                       if col not in ['Date', 'Code', 'Target'] and 
                       df.dtype in ['int64', 'float64']]
        
        X = df[feature_cols].fillna(method='ffill').fillna(0)
        y = df['Target']
        
        # 時系列分割（最後20%をテスト用）
        df_sorted = df.sort_values('Date')
        split_idx = int(len(df_sorted) * 0.8)
        
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        # 特徴量選択
        selector = SelectKBest(score_func=f_classif, k=min(50, len(feature_cols)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # モデル学習
        model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=300,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # 予測・評価
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        logger.info(f"🎯 最終モデル性能:")
        logger.info(f"  精度: {accuracy:.4f}")
        logger.info(f"  適合率: {precision:.4f}")
        logger.info(f"  再現率: {recall:.4f}")
        logger.info(f"  F1スコア: {f1:.4f}")
        
        # モデル保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_data = {
            'model': model,
            'scaler': scaler,
            'selector': selector,
            'feature_cols': feature_cols,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_size': len(X_train),
            'test_size': len(X_test)
        }
        
        model_file = self.output_dir / f"enhanced_model_v3_{accuracy:.4f}acc_{timestamp}.joblib"
        joblib.dump(model_data, model_file)
        logger.info(f"🎯 モデル保存完了: {model_file}")
        
        return model_data
    
    def run_enhanced_system(self):
        """拡張システム実行"""
        logger.info("🚀 Enhanced Precision System V3 実行開始!")
        
        try:
            # データ統合
            df = self.load_and_integrate_data()
            
            # 特徴量作成
            enhanced_df = self.create_enhanced_features(df)
            
            # ウォークフォワード最適化
            wfo_results = self.walk_forward_optimization(enhanced_df)
            
            # 最終モデル学習
            final_model = self.train_final_model(enhanced_df)
            
            # 結果統計
            if wfo_results:
                wfo_accuracies = [r['accuracy'] for r in wfo_results]
                wfo_mean_acc = np.mean(wfo_accuracies)
                wfo_std_acc = np.std(wfo_accuracies)
                
                logger.info(f"\n📊 ウォークフォワード最適化結果:")
                logger.info(f"  期間数: {len(wfo_results)}")
                logger.info(f"  平均精度: {wfo_mean_acc:.4f} ± {wfo_std_acc:.4f}")
                logger.info(f"  最高精度: {max(wfo_accuracies):.4f}")
                logger.info(f"  最低精度: {min(wfo_accuracies):.4f}")
            
            # 結果保存
            results = {
                'final_model_accuracy': final_model['accuracy'],
                'wfo_mean_accuracy': wfo_mean_acc if wfo_results else 0,
                'wfo_std_accuracy': wfo_std_acc if wfo_results else 0,
                'wfo_results': wfo_results,
                'data_size': len(enhanced_df),
                'feature_count': len(final_model['feature_cols']),
                'external_data_integrated': os.path.exists(self.external_file)
            }
            
            results_file = self.output_dir / f"enhanced_results_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib"
            joblib.dump(results, results_file)
            
            logger.info(f"🎉 Enhanced Precision System V3 完了!")
            logger.info(f"最終精度: {final_model['accuracy']:.4f}")
            logger.info(f"データ統合: {'✅' if results['external_data_integrated'] else '❌'}")
            logger.info(f"結果保存: {results_file}")
            
            return results
            
        except Exception as e:
            logger.error(f"システム実行エラー: {e}")
            return None

def main():
    """メイン実行"""
    system = EnhancedPrecisionSystemV3()
    results = system.run_enhanced_system()
    
    if results:
        print(f"\n✅ Enhanced Precision System V3 実行完了!")
        print(f"📊 最終精度: {results['final_model_accuracy']:.4f}")
        if results['wfo_mean_accuracy'] > 0:
            print(f"📈 ウォークフォワード平均精度: {results['wfo_mean_accuracy']:.4f}")
        print(f"📁 データ統合: {'成功' if results['external_data_integrated'] else '外部データなし'}")
        print(f"📊 データ量: {results['data_size']:,}件")
        print(f"🔧 特徴量数: {results['feature_count']}個")
    else:
        print("\n❌ システム実行に失敗しました")

if __name__ == "__main__":
    main()