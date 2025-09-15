"""
時系列安定性検証システム
時期による性能変動を分析し、安定性を評価
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime, timedelta
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TemporalStabilityTester:
    """時系列安定性テスト"""
    
    def __init__(self):
        self.df = None
        self.load_data()
    
    def load_data(self):
        """データ読み込み"""
        data_file = Path("data/nikkei225_full_data/nikkei225_full_10years_175stocks_20250831_020101.pkl")
        self.df = pd.read_pickle(data_file)
        
        # 基本前処理
        self.df = self.df.sort_values(['Code', 'Date']).reset_index(drop=True)
        self.df['close_price'] = pd.to_numeric(self.df['Close'], errors='coerce')
        self.df['daily_return'] = self.df.groupby('Code')['close_price'].pct_change(fill_method=None)
        self.df['next_day_return'] = self.df.groupby('Code')['close_price'].pct_change(fill_method=None).shift(-1)
        self.df['target'] = (self.df['next_day_return'] >= 0.01).astype(int)
        
        # 日付型変換
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        logger.info(f"データ読み込み完了: {len(self.df):,}レコード")
        logger.info(f"期間: {self.df['Date'].min().date()} ～ {self.df['Date'].max().date()}")
    
    def create_stable_features(self):
        """安定性重視の特徴量作成"""
        df_features = self.df.copy()
        
        # 基本的で安定した特徴量のみ
        
        # 移動平均比率
        for window in [10, 20, 50]:
            sma = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
            df_features[f'sma_ratio_{window}'] = df_features['close_price'] / sma
        
        # RSI
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df_features['rsi_14'] = df_features.groupby('Code')['close_price'].transform(calc_rsi)
        
        # ボラティリティ（複数期間）
        for window in [10, 20]:
            df_features[f'volatility_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
        
        # モメンタム
        for period in [5, 10]:
            df_features[f'momentum_{period}'] = df_features.groupby('Code')['close_price'].transform(
                lambda x: x.pct_change(period, fill_method=None)
            )
        
        # 過去リターン
        for lag in [1, 2, 3]:
            df_features[f'return_lag_{lag}'] = df_features.groupby('Code')['daily_return'].shift(lag)
        
        # 市場相対性能
        market_return = df_features.groupby('Date')['daily_return'].mean()
        df_features['market_return'] = df_features['Date'].map(market_return)
        df_features['excess_return'] = df_features['daily_return'] - df_features['market_return']
        
        feature_cols = [col for col in df_features.columns 
                       if col.startswith(('sma_ratio', 'rsi', 'volatility', 'momentum', 
                                        'return_lag', 'excess_return'))]
        
        return df_features, feature_cols
    
    def test_temporal_windows(self, df_features, feature_cols):
        """時期別ウィンドウテスト"""
        logger.info("=== 時期別ウィンドウテスト開始 ===")
        
        # データを年別に分割
        df_features['year'] = df_features['Date'].dt.year
        years = sorted(df_features['year'].unique())
        
        logger.info(f"対象年度: {years[0]} ～ {years[-1]}")
        
        results = []
        
        # 3年ウィンドウで学習し、1年で検証を繰り返す
        for test_year in years[3:]:  # 2018年から開始
            train_years = [test_year - 3, test_year - 2, test_year - 1]
            
            logger.info(f"\n訓練期間: {train_years} → 検証: {test_year}")
            
            # 訓練・検証データ分割
            train_mask = df_features['year'].isin(train_years)
            test_mask = df_features['year'] == test_year
            
            df_train = df_features[train_mask]
            df_test = df_features[test_mask]
            
            if len(df_train) < 10000 or len(df_test) < 1000:
                logger.warning(f"{test_year}: データ不足")
                continue
            
            # 特徴量・ターゲット準備
            X_train = df_train[feature_cols].fillna(0)
            y_train = df_train['target']
            X_test = df_test[feature_cols].fillna(0)
            y_test = df_test['target']
            
            # 有効データのみ
            train_valid = ~(y_train.isna() | X_train.isna().any(axis=1))
            test_valid = ~(y_test.isna() | X_test.isna().any(axis=1))
            
            X_train, y_train = X_train[train_valid], y_train[train_valid]
            X_test, y_test = X_test[test_valid], y_test[test_valid]
            
            # スケーリング
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル訓練
            model = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_split=10,
                random_state=42, n_jobs=-1
            )
            model.fit(X_train_scaled, y_train)
            
            # 予測・評価
            proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 複数閾値で評価
            year_results = {
                'test_year': test_year,
                'train_size': len(X_train),
                'test_size': len(X_test),
                'target_rate': y_test.mean()
            }
            
            for threshold in [0.5, 0.6, 0.7, 0.8]:
                predictions = (proba >= threshold).astype(int)
                
                if predictions.sum() > 0:
                    precision = precision_score(y_test, predictions)
                    recall = recall_score(y_test, predictions)
                    
                    year_results[f'precision_{threshold}'] = precision
                    year_results[f'recall_{threshold}'] = recall
                    year_results[f'predictions_{threshold}'] = predictions.sum()
                else:
                    year_results[f'precision_{threshold}'] = 0
                    year_results[f'recall_{threshold}'] = 0
                    year_results[f'predictions_{threshold}'] = 0
            
            results.append(year_results)
            
            # 結果ログ
            best_precision = max([year_results[f'precision_{t}'] for t in [0.5, 0.6, 0.7, 0.8]])
            logger.info(f"{test_year}年: 最高精度 {best_precision:.3f}, ターゲット率 {y_test.mean():.1%}")
        
        return results
    
    def test_market_conditions(self, df_features, feature_cols):
        """市場環境別テスト"""
        logger.info("=== 市場環境別テスト開始 ===")
        
        # 市場全体のリターンを計算
        daily_market_return = df_features.groupby('Date')['daily_return'].mean()
        
        # 市場環境を分類
        market_volatility = daily_market_return.rolling(30).std()
        market_trend = daily_market_return.rolling(30).mean()
        
        conditions = []
        
        for date, vol, trend in zip(daily_market_return.index, market_volatility, market_trend):
            if pd.isna(vol) or pd.isna(trend):
                condition = 'Unknown'
            elif vol > market_volatility.quantile(0.75):
                condition = 'High_Volatility'
            elif vol < market_volatility.quantile(0.25):
                condition = 'Low_Volatility'
            elif trend > 0.001:  # 0.1%以上
                condition = 'Bull_Market'
            elif trend < -0.001:  # -0.1%以下
                condition = 'Bear_Market'
            else:
                condition = 'Sideways'
            
            conditions.append(condition)
        
        # データに市場環境を追加
        market_conditions = pd.DataFrame({
            'Date': daily_market_return.index,
            'market_condition': conditions
        })
        
        df_features = df_features.merge(market_conditions, on='Date', how='left')
        
        # 環境別テスト
        condition_results = {}
        
        for condition in ['Bull_Market', 'Bear_Market', 'High_Volatility', 'Low_Volatility', 'Sideways']:
            logger.info(f"\n市場環境: {condition}")
            
            condition_data = df_features[df_features['market_condition'] == condition]
            
            if len(condition_data) < 5000:
                logger.info(f"データ不足 ({len(condition_data)}件)")
                continue
            
            # 訓練・検証分割（時系列順）
            split_point = int(len(condition_data) * 0.8)
            train_data = condition_data.iloc[:split_point]
            test_data = condition_data.iloc[split_point:]
            
            X_train = train_data[feature_cols].fillna(0)
            y_train = train_data['target']
            X_test = test_data[feature_cols].fillna(0)
            y_test = test_data['target']
            
            # 有効データのみ
            train_valid = ~(y_train.isna() | X_train.isna().any(axis=1))
            test_valid = ~(y_test.isna() | X_test.isna().any(axis=1))
            
            X_train, y_train = X_train[train_valid], y_train[train_valid]
            X_test, y_test = X_test[test_valid], y_test[test_valid]
            
            if len(X_train) < 1000 or len(X_test) < 200:
                logger.info(f"分割後データ不足")
                continue
            
            # スケーリング・モデル訓練
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            model = RandomForestClassifier(n_estimators=200, max_depth=10,
                                         random_state=42, n_jobs=-1)
            model.fit(X_train_scaled, y_train)
            
            proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 最適閾値での評価
            best_score = 0
            best_threshold = 0.5
            
            for threshold in [0.5, 0.6, 0.7, 0.8]:
                predictions = (proba >= threshold).astype(int)
                if predictions.sum() > 0:
                    precision = precision_score(y_test, predictions)
                    if precision > best_score:
                        best_score = precision
                        best_threshold = threshold
            
            condition_results[condition] = {
                'sample_count': len(condition_data),
                'target_rate': condition_data['target'].mean(),
                'best_precision': best_score,
                'best_threshold': best_threshold
            }
            
            logger.info(f"{condition}: 精度{best_score:.3f}, サンプル{len(condition_data):,}件")
        
        return condition_results
    
    def run_stability_analysis(self):
        """安定性分析の実行"""
        logger.info("=== 時系列安定性分析開始 ===")
        
        # 特徴量作成
        df_features, feature_cols = self.create_stable_features()
        logger.info(f"安定性重視特徴量: {len(feature_cols)}個")
        
        # 1. 時期別ウィンドウテスト
        temporal_results = self.test_temporal_windows(df_features, feature_cols)
        
        # 2. 市場環境別テスト
        market_results = self.test_market_conditions(df_features, feature_cols)
        
        # 安定性評価
        logger.info("\n=== 安定性評価 ===")
        
        if temporal_results:
            # 年別精度の統計
            precisions_60 = [r.get('precision_0.6', 0) for r in temporal_results if r.get('precision_0.6', 0) > 0]
            precisions_70 = [r.get('precision_0.7', 0) for r in temporal_results if r.get('precision_0.7', 0) > 0]
            
            if precisions_60:
                logger.info(f"閾値0.6での年別精度:")
                logger.info(f"  平均: {np.mean(precisions_60):.3f}")
                logger.info(f"  標準偏差: {np.std(precisions_60):.3f}")
                logger.info(f"  最小-最大: {np.min(precisions_60):.3f} - {np.max(precisions_60):.3f}")
            
            if precisions_70:
                logger.info(f"閾値0.7での年別精度:")
                logger.info(f"  平均: {np.mean(precisions_70):.3f}")
                logger.info(f"  標準偏差: {np.std(precisions_70):.3f}")
                logger.info(f"  最小-最大: {np.min(precisions_70):.3f} - {np.max(precisions_70):.3f}")
        
        # 市場環境別精度の統計
        if market_results:
            market_precisions = [r['best_precision'] for r in market_results.values()]
            logger.info(f"\n市場環境別精度:")
            logger.info(f"  平均: {np.mean(market_precisions):.3f}")
            logger.info(f"  標準偏差: {np.std(market_precisions):.3f}")
            
            for condition, results in market_results.items():
                logger.info(f"  {condition}: {results['best_precision']:.3f}")
        
        # 安定性スコア計算
        stability_score = 0
        
        if temporal_results and precisions_60:
            temporal_stability = 1 / (1 + np.std(precisions_60))  # 標準偏差が小さいほど高スコア
            stability_score += temporal_stability * 0.6
            
        if market_results:
            market_stability = 1 / (1 + np.std(market_precisions))
            stability_score += market_stability * 0.4
        
        logger.info(f"\n総合安定性スコア: {stability_score:.3f}")
        
        return {
            'temporal_results': temporal_results,
            'market_results': market_results,
            'stability_score': stability_score,
            'feature_count': len(feature_cols)
        }


def main():
    """メイン実行"""
    logger.info("時系列安定性検証開始")
    
    try:
        tester = TemporalStabilityTester()
        results = tester.run_stability_analysis()
        
        print(f"\n=== 時系列安定性分析結果 ===")
        print(f"安定性スコア: {results['stability_score']:.3f}")
        print(f"使用特徴量: {results['feature_count']}個")
        
        if results['stability_score'] > 0.7:
            print("✅ 高い時系列安定性")
        elif results['stability_score'] > 0.5:
            print("⚠️  中程度の安定性")
        else:
            print("❌ 安定性に課題あり")
        
        return results
        
    except Exception as e:
        logger.error(f"エラー: {e}")
        raise


if __name__ == "__main__":
    main()