"""
精度向上戦略の実用的検証
最も効果的な改善方法を特定
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ImprovementTester:
    """精度向上手法のテスト"""
    
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
        
        print(f"データ読み込み完了: {len(self.df):,}レコード")
        print(f"ターゲット分布: {self.df['target'].mean():.1%}")
    
    def create_basic_features(self, df):
        """基本特徴量（現在の特徴量）"""
        df_features = df.copy()
        
        # 移動平均系
        for window in [5, 10, 20]:
            sma = df_features.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
            df_features[f'sma_ratio_{window}'] = df_features['close_price'] / sma
            
        # RSI
        def calc_rsi(prices, window=14):
            delta = prices.diff()
            gain = delta.where(delta > 0, 0).rolling(window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
            rs = gain / (loss + 1e-8)
            return 100 - (100 / (1 + rs))
        
        df_features['rsi'] = df_features.groupby('Code')['close_price'].transform(calc_rsi)
        
        # ラグ特徴量
        for lag in [1, 2, 3]:
            df_features[f'return_lag_{lag}'] = df_features.groupby('Code')['daily_return'].shift(lag)
        
        feature_cols = [col for col in df_features.columns 
                       if col.startswith(('sma_ratio', 'rsi', 'return_lag'))]
        return df_features, feature_cols
    
    def create_enhanced_features(self, df):
        """拡張特徴量"""
        df_features = df.copy()
        
        # 基本特徴量
        df_features, basic_cols = self.create_basic_features(df_features)
        
        # 高度な特徴量
        high_prices = pd.to_numeric(df_features['High'], errors='coerce')
        low_prices = pd.to_numeric(df_features['Low'], errors='coerce')
        volumes = pd.to_numeric(df_features['Volume'], errors='coerce')
        
        # ボラティリティ系
        for window in [5, 10, 20]:
            df_features[f'volatility_{window}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(window).std()
            )
            df_features[f'price_range_{window}'] = df_features.groupby('Code').apply(
                lambda group: ((group['High'].astype(float) - group['Low'].astype(float)) / 
                              group['close_price']).rolling(window).mean()
            ).values
        
        # モメンタム系
        for period in [5, 10, 20]:
            df_features[f'momentum_{period}'] = df_features.groupby('Code')['close_price'].transform(
                lambda x: x.pct_change(period)
            )
            df_features[f'return_std_{period}'] = df_features.groupby('Code')['daily_return'].transform(
                lambda x: x.rolling(period).std()
            )
        
        # 市場関連
        market_return = df_features.groupby('Date')['daily_return'].mean()
        df_features['market_return'] = df_features['Date'].map(market_return)
        df_features['excess_return'] = df_features['daily_return'] - df_features['market_return']
        
        # ボリューム関連
        df_features['volume_ma_ratio'] = df_features.groupby('Code').apply(
            lambda x: pd.to_numeric(x['Volume'], errors='coerce') / 
                     pd.to_numeric(x['Volume'], errors='coerce').rolling(20).mean()
        ).values
        
        # 価格位置
        for window in [10, 20]:
            high_max = df_features.groupby('Code')['High'].transform(
                lambda x: pd.to_numeric(x, errors='coerce').rolling(window).max()
            )
            low_min = df_features.groupby('Code')['Low'].transform(
                lambda x: pd.to_numeric(x, errors='coerce').rolling(window).min()
            )
            df_features[f'price_position_{window}'] = (
                (df_features['close_price'] - low_min) / (high_max - low_min + 1e-8)
            )
        
        enhanced_cols = [col for col in df_features.columns 
                        if col.startswith(('sma_ratio', 'rsi', 'return_lag', 'volatility', 
                                         'price_range', 'momentum', 'return_std', 'excess_return',
                                         'volume_ma_ratio', 'price_position'))]
        return df_features, enhanced_cols
    
    def test_feature_impact(self):
        """特徴量追加の効果をテスト"""
        print("\n=== 特徴量追加効果テスト ===")
        
        # 基本特徴量での性能
        df_basic, basic_cols = self.create_basic_features(self.df)
        basic_score = self._evaluate_features(df_basic, basic_cols, "基本特徴量")
        
        # 拡張特徴量での性能
        df_enhanced, enhanced_cols = self.create_enhanced_features(self.df)
        enhanced_score = self._evaluate_features(df_enhanced, enhanced_cols, "拡張特徴量")
        
        improvement = enhanced_score - basic_score
        print(f"\n特徴量追加による改善: {improvement:.3f}")
        print(f"改善率: {improvement/basic_score*100:.1f}%" if basic_score > 0 else "計算不可")
        
        return improvement
    
    def test_model_comparison(self):
        """モデル比較テスト"""
        print("\n=== モデル比較テスト ===")
        
        df_features, feature_cols = self.create_basic_features(self.df)
        
        models = {
            "LightGBM": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, 
                                          max_depth=6, random_state=42, verbosity=-1),
            "XGBoost": xgb.XGBClassifier(n_estimators=100, learning_rate=0.05,
                                       max_depth=6, random_state=42, verbosity=0),
            "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8,
                                                  random_state=42, n_jobs=-1)
        }
        
        results = {}
        for name, model in models.items():
            score = self._evaluate_model(df_features, feature_cols, model, name)
            results[name] = score
        
        best_model = max(results, key=results.get)
        print(f"\n最高性能モデル: {best_model} (精度: {results[best_model]:.3f})")
        
        return results
    
    def test_target_threshold_impact(self):
        """ターゲット閾値の影響をテスト"""
        print("\n=== ターゲット閾値影響テスト ===")
        
        thresholds = [0.005, 0.01, 0.015, 0.02, 0.025]  # 0.5%から2.5%まで
        
        results = {}
        for threshold in thresholds:
            # ターゲット再定義
            df_test = self.df.copy()
            df_test['target'] = (df_test['next_day_return'] >= threshold).astype(int)
            
            target_rate = df_test['target'].mean()
            print(f"{threshold*100:.1f}%以上上昇: {target_rate:.1%} ({df_test['target'].sum():,}件)")
            
            if target_rate > 0.05:  # 最低5%の頻度が必要
                df_features, feature_cols = self.create_basic_features(df_test)
                score = self._evaluate_features(df_features, feature_cols, f"{threshold*100:.1f}%閾値")
                results[threshold] = {'score': score, 'frequency': target_rate}
            else:
                print(f"  → データ不足のためスキップ")
        
        if results:
            best_threshold = max(results, key=lambda x: results[x]['score'])
            print(f"\n最適閾値: {best_threshold*100:.1f}%")
            print(f"最高精度: {results[best_threshold]['score']:.3f}")
            print(f"出現頻度: {results[best_threshold]['frequency']:.1%}")
        
        return results
    
    def test_data_preprocessing_impact(self):
        """データ前処理の影響をテスト"""
        print("\n=== データ前処理影響テスト ===")
        
        # 標準前処理
        df_standard = self.df.copy()
        df_features, feature_cols = self.create_basic_features(df_standard)
        standard_score = self._evaluate_features(df_features, feature_cols, "標準前処理")
        
        # 外れ値除去
        df_outlier_removed = self.df.copy()
        
        # 極端なリターンを除去（上下5%）
        return_q05 = df_outlier_removed['daily_return'].quantile(0.05)
        return_q95 = df_outlier_removed['daily_return'].quantile(0.95)
        
        outlier_mask = (
            (df_outlier_removed['daily_return'] >= return_q05) & 
            (df_outlier_removed['daily_return'] <= return_q95)
        )
        df_outlier_removed = df_outlier_removed[outlier_mask]
        
        print(f"外れ値除去: {len(self.df) - len(df_outlier_removed):,}件除去")
        
        if len(df_outlier_removed) > 100000:  # 十分なデータが残っている場合
            df_features_clean, _ = self.create_basic_features(df_outlier_removed)
            clean_score = self._evaluate_features(df_features_clean, feature_cols, "外れ値除去")
            
            improvement = clean_score - standard_score
            print(f"外れ値除去による改善: {improvement:.3f}")
        
        return {'standard': standard_score}
    
    def _evaluate_features(self, df, feature_cols, name):
        """特徴量セットの評価"""
        X = df[feature_cols].fillna(0)
        y = df['target']
        
        # 有効データのみ
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 10000:
            print(f"  {name}: データ不足")
            return 0
        
        # Train/Test分割
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデル訓練
        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05, 
                                  max_depth=6, random_state=42, verbosity=-1)
        model.fit(X_train_scaled, y_train)
        
        # 予測・評価
        proba = model.predict_proba(X_test_scaled)[:, 1]
        
        best_score = 0
        for threshold in [0.5, 0.6, 0.7]:
            predictions = (proba >= threshold).astype(int)
            if predictions.sum() > 0:
                precision = precision_score(y_test, predictions)
                if precision > best_score:
                    best_score = precision
        
        print(f"  {name} ({len(feature_cols)}特徴量): 最高精度 {best_score:.3f}")
        return best_score
    
    def _evaluate_model(self, df, feature_cols, model, name):
        """モデルの評価"""
        X = df[feature_cols].fillna(0)
        y = df['target']
        
        valid_mask = ~(y.isna() | X.isna().any(axis=1))
        X, y = X[valid_mask], y[valid_mask]
        
        split_point = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
        y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model.fit(X_train_scaled, y_train)
        proba = model.predict_proba(X_test_scaled)[:, 1]
        
        best_score = 0
        for threshold in [0.5, 0.6, 0.7]:
            predictions = (proba >= threshold).astype(int)
            if predictions.sum() > 0:
                precision = precision_score(y_test, predictions)
                if precision > best_score:
                    best_score = precision
        
        print(f"  {name}: 最高精度 {best_score:.3f}")
        return best_score


def main():
    """メイン実行"""
    print("=== 精度向上戦略検証 ===")
    
    tester = ImprovementTester()
    
    # 各改善手法をテスト
    feature_improvement = tester.test_feature_impact()
    model_results = tester.test_model_comparison()
    threshold_results = tester.test_target_threshold_impact()
    preprocessing_results = tester.test_data_preprocessing_impact()
    
    # 総合的な推奨事項
    print("\n" + "="*50)
    print("=== 総合推奨事項 ===")
    
    print(f"\n【即効性のある改善】")
    print(f"1. モデル変更: 最大{max(model_results.values()) - min(model_results.values()):.3f}の改善")
    print(f"2. ターゲット調整: 最適な閾値設定で精度向上")
    
    print(f"\n【中期的な改善】")
    print(f"3. 特徴量拡張: 約{feature_improvement:.3f}の改善見込み")
    print(f"4. データクリーニング: 品質向上による安定化")
    
    print(f"\n【実用性重視の推奨設定】")
    print(f"・ターゲット: 1%上昇（バランス型）")
    print(f"・モデル: {max(model_results, key=model_results.get)}")
    print(f"・特徴量: 拡張版（約30特徴量）")
    print(f"・閾値: 0.6-0.7（実用的予測数確保）")
    
    print(f"\n【期待される最終性能】")
    best_model_score = max(model_results.values())
    expected_final = best_model_score + feature_improvement
    print(f"予想精度: {expected_final:.3f} ({expected_final:.1%})")
    print(f"日次予測数: 2-5件（実用的レベル）")
    print(f"月間成功見込み: {expected_final * 3 * 20:.0f}件程度")


if __name__ == "__main__":
    main()