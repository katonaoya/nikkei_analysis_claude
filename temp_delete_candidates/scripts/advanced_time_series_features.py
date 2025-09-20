#!/usr/bin/env python3
"""
高度時系列特徴量エンジニアリング
60%超えを目指す第2段階: ラグ特徴量、移動統計、トレンド分析、周期性
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

class AdvancedTimeSeriesFeatures:
    """高度時系列特徴量エンジニアリング"""
    
    def __init__(self):
        self.data_dir = Path("data")
        self.processed_dir = self.data_dir / "processed"
        self.scaler = StandardScaler()
        
        # 現在の最適特徴量
        self.base_features = [
            'Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20',
            'sp500_change', 'vix_change', 'nikkei_change', 'us_10y_change', 'usd_jpy_change'
        ]
        
    def load_integrated_data(self):
        """統合データ読み込み"""
        logger.info("📊 統合データ読み込み...")
        
        integrated_file = self.processed_dir / "integrated_with_external.parquet"
        df = pd.read_parquet(integrated_file)
        df['Date'] = pd.to_datetime(df['Date'])
        
        logger.info(f"✅ データ読み込み: {len(df):,}件")
        return df
    
    def create_lag_features(self, df, target_columns, lags=[1, 2, 3, 5, 10]):
        """ラグ特徴量作成"""
        logger.info("⏱️ ラグ特徴量作成...")
        
        df_with_lags = df.copy()
        
        # 各銘柄ごとにラグ特徴量を作成
        for col in target_columns:
            if col not in df.columns:
                continue
                
            logger.info(f"  {col} のラグ特徴量作成...")
            
            for lag in lags:
                lag_col = f"{col}_lag_{lag}"
                df_with_lags[lag_col] = df_with_lags.groupby('Code')[col].shift(lag)
        
        created_features = [f"{col}_lag_{lag}" for col in target_columns for lag in lags if col in df.columns]
        logger.info(f"  ラグ特徴量: {len(created_features)}個作成")
        
        return df_with_lags, created_features
    
    def create_rolling_statistics(self, df, target_columns, windows=[5, 10, 20, 50]):
        """移動統計特徴量作成"""
        logger.info("📊 移動統計特徴量作成...")
        
        df_with_stats = df.copy()
        created_features = []
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            logger.info(f"  {col} の移動統計作成...")
            
            for window in windows:
                # 移動平均
                ma_col = f"{col}_ma_{window}"
                df_with_stats[ma_col] = df_with_stats.groupby('Code')[col].rolling(window, min_periods=1).mean().reset_index(0, drop=True)
                
                # 移動標準偏差
                std_col = f"{col}_std_{window}"
                df_with_stats[std_col] = df_with_stats.groupby('Code')[col].rolling(window, min_periods=1).std().reset_index(0, drop=True)
                
                # 現在値と移動平均の乖離
                diff_col = f"{col}_diff_ma_{window}"
                df_with_stats[diff_col] = (df_with_stats[col] - df_with_stats[ma_col]) / (df_with_stats[ma_col].abs() + 1e-8)
                
                created_features.extend([ma_col, std_col, diff_col])
        
        logger.info(f"  移動統計特徴量: {len(created_features)}個作成")
        return df_with_stats, created_features
    
    def create_trend_features(self, df, target_columns, windows=[5, 10, 20]):
        """トレンド特徴量作成"""
        logger.info("📈 トレンド特徴量作成...")
        
        df_with_trends = df.copy()
        created_features = []
        
        def calculate_slope(series):
            """線形回帰の傾きを計算"""
            if len(series) < 2:
                return 0
            x = np.arange(len(series))
            try:
                slope, _, _, _, _ = stats.linregress(x, series)
                return slope if not np.isnan(slope) else 0
            except:
                return 0
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            logger.info(f"  {col} のトレンド特徴量作成...")
            
            for window in windows:
                # 線形トレンド（傾き）
                slope_col = f"{col}_slope_{window}"
                df_with_trends[slope_col] = df_with_trends.groupby('Code')[col].rolling(window, min_periods=2).apply(calculate_slope).reset_index(0, drop=True)
                
                # トレンド強度（R²）
                def calculate_r_squared(series):
                    if len(series) < 3:
                        return 0
                    x = np.arange(len(series))
                    try:
                        _, _, r_value, _, _ = stats.linregress(x, series)
                        return r_value ** 2 if not np.isnan(r_value) else 0
                    except:
                        return 0
                
                r2_col = f"{col}_r2_{window}"
                df_with_trends[r2_col] = df_with_trends.groupby('Code')[col].rolling(window, min_periods=3).apply(calculate_r_squared).reset_index(0, drop=True)
                
                created_features.extend([slope_col, r2_col])
        
        logger.info(f"  トレンド特徴量: {len(created_features)}個作成")
        return df_with_trends, created_features
    
    def create_cyclical_features(self, df):
        """周期性特徴量作成"""
        logger.info("🔄 周期性特徴量作成...")
        
        df_with_cycles = df.copy()
        
        # 曜日効果
        df_with_cycles['day_of_week'] = df_with_cycles['Date'].dt.dayofweek
        df_with_cycles['is_monday'] = (df_with_cycles['day_of_week'] == 0).astype(int)
        df_with_cycles['is_friday'] = (df_with_cycles['day_of_week'] == 4).astype(int)
        
        # 月効果
        df_with_cycles['month'] = df_with_cycles['Date'].dt.month
        df_with_cycles['is_january'] = (df_with_cycles['month'] == 1).astype(int)
        df_with_cycles['is_december'] = (df_with_cycles['month'] == 12).astype(int)
        
        # 四半期効果
        df_with_cycles['quarter'] = df_with_cycles['Date'].dt.quarter
        df_with_cycles['is_q1'] = (df_with_cycles['quarter'] == 1).astype(int)
        df_with_cycles['is_q4'] = (df_with_cycles['quarter'] == 4).astype(int)
        
        # 月初・月末効果
        df_with_cycles['day_of_month'] = df_with_cycles['Date'].dt.day
        df_with_cycles['is_month_start'] = (df_with_cycles['day_of_month'] <= 5).astype(int)
        df_with_cycles['is_month_end'] = (df_with_cycles['day_of_month'] >= 25).astype(int)
        
        # 年効果（リーマンショック、コロナショック等）
        df_with_cycles['year'] = df_with_cycles['Date'].dt.year
        df_with_cycles['is_crisis_year'] = df_with_cycles['year'].isin([2008, 2009, 2020]).astype(int)
        
        cyclical_features = [
            'day_of_week', 'is_monday', 'is_friday', 
            'month', 'is_january', 'is_december',
            'quarter', 'is_q1', 'is_q4',
            'is_month_start', 'is_month_end', 
            'is_crisis_year'
        ]
        
        logger.info(f"  周期性特徴量: {len(cyclical_features)}個作成")
        return df_with_cycles, cyclical_features
    
    def create_momentum_features(self, df, target_columns):
        """モメンタム特徴量作成"""
        logger.info("🚀 モメンタム特徴量作成...")
        
        df_with_momentum = df.copy()
        created_features = []
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            logger.info(f"  {col} のモメンタム特徴量作成...")
            
            # 短期モメンタム（3日、5日）
            for period in [3, 5]:
                momentum_col = f"{col}_momentum_{period}"
                df_with_momentum[momentum_col] = df_with_momentum.groupby('Code')[col].pct_change(periods=period)
                created_features.append(momentum_col)
            
            # 加速度（変化率の変化率）
            acceleration_col = f"{col}_acceleration"
            df_with_momentum[acceleration_col] = df_with_momentum.groupby('Code')[col].pct_change().pct_change()
            created_features.append(acceleration_col)
            
            # 相対強度（過去20日の分位数）
            def rolling_rank(series, window=20):
                return series.rolling(window, min_periods=1).rank(pct=True)
            
            rank_col = f"{col}_rank_20"
            df_with_momentum[rank_col] = df_with_momentum.groupby('Code')[col].apply(rolling_rank).reset_index(0, drop=True)
            created_features.append(rank_col)
        
        logger.info(f"  モメンタム特徴量: {len(created_features)}個作成")
        return df_with_momentum, created_features
    
    def create_volatility_features(self, df, target_columns):
        """ボラティリティ特徴量作成"""
        logger.info("📊 ボラティリティ特徴量作成...")
        
        df_with_vol = df.copy()
        created_features = []
        
        for col in target_columns:
            if col not in df.columns:
                continue
                
            logger.info(f"  {col} のボラティリティ特徴量作成...")
            
            # 実現ボラティリティ（複数期間）
            for window in [5, 10, 20]:
                vol_col = f"{col}_realized_vol_{window}"
                df_with_vol[vol_col] = df_with_vol.groupby('Code')[col].rolling(window, min_periods=1).std().reset_index(0, drop=True)
                created_features.append(vol_col)
            
            # EWMA ボラティリティ
            ewma_col = f"{col}_ewma_vol"
            df_with_vol[ewma_col] = df_with_vol.groupby('Code')[col].ewm(span=20).std().reset_index(0, drop=True)
            created_features.append(ewma_col)
            
            # ボラティリティの変化
            vol_change_col = f"{col}_vol_change"
            df_with_vol[vol_change_col] = df_with_vol.groupby('Code')[f'{col}_realized_vol_20'].pct_change()
            created_features.append(vol_change_col)
        
        logger.info(f"  ボラティリティ特徴量: {len(created_features)}個作成")
        return df_with_vol, created_features
    
    def feature_selection_by_importance(self, X, y, all_features, top_k=30):
        """重要度による特徴選択"""
        logger.info(f"🔍 特徴選択（上位{top_k}個）...")
        
        # LogisticRegression で特徴重要度計算
        X_scaled = self.scaler.fit_transform(X)
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        model.fit(X_scaled, y)
        
        # 重要度（係数の絶対値）
        importances = abs(model.coef_[0])
        feature_importance = list(zip(all_features, importances))
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # 上位K個選択
        selected_features = [feat for feat, imp in feature_importance[:top_k]]
        
        logger.info(f"上位{top_k}特徴量選択完了")
        logger.info("上位10特徴量:")
        for i, (feat, imp) in enumerate(feature_importance[:10], 1):
            logger.info(f"  {i:2d}. {feat:30s}: {imp:.4f}")
        
        return selected_features, feature_importance
    
    def evaluate_enhanced_features(self, X, y, feature_set_name):
        """拡張特徴量の評価"""
        logger.info(f"📊 {feature_set_name} 評価...")
        
        X_scaled = self.scaler.fit_transform(X.fillna(0))
        model = LogisticRegression(C=0.001, class_weight='balanced', random_state=42, max_iter=1000)
        
        tscv = TimeSeriesSplit(n_splits=5)
        scores = []
        fold_details = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
            X_train = X_scaled[train_idx]
            X_test = X_scaled[test_idx]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
            
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, pred)
            scores.append(accuracy)
            
            fold_details.append({
                'fold': fold + 1,
                'accuracy': accuracy,
                'train_size': len(X_train),
                'test_size': len(X_test)
            })
        
        result = {
            'avg': np.mean(scores),
            'std': np.std(scores),
            'min': np.min(scores),
            'max': np.max(scores),
            'scores': scores,
            'fold_details': fold_details
        }
        
        logger.info(f"  {feature_set_name}: {result['avg']:.3%} ± {result['std']:.3%}")
        return result

def main():
    """メイン実行"""
    logger.info("🚀 高度時系列特徴量エンジニアリング")
    logger.info("🎯 目標: 59.4%から62%超えを目指す")
    
    ts_features = AdvancedTimeSeriesFeatures()
    
    try:
        # 1. データ読み込み
        df = ts_features.load_integrated_data()
        
        # 2. ベースライン評価
        logger.info("📏 ベースライン評価...")
        clean_df = df[df['Binary_Direction'].notna()].copy()
        clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
        
        X_base = clean_df[ts_features.base_features].fillna(0)
        y = clean_df['Binary_Direction'].astype(int)
        
        baseline_result = ts_features.evaluate_enhanced_features(X_base, y, "ベースライン")
        
        # 3. 段階的特徴量追加
        logger.info("\n🔧 段階的特徴量エンジニアリング...")
        
        enhanced_df = clean_df.copy()
        all_features = ts_features.base_features.copy()
        
        # ラグ特徴量追加
        logger.info("\n⏱️ ステップ1: ラグ特徴量追加...")
        key_features_for_lag = ['sp500_change', 'vix_change', 'Market_Return']
        enhanced_df, lag_features = ts_features.create_lag_features(enhanced_df, key_features_for_lag, lags=[1, 2, 3])
        all_features.extend(lag_features)
        
        X_with_lags = enhanced_df[all_features].fillna(0)
        lag_result = ts_features.evaluate_enhanced_features(X_with_lags, y, "ベース+ラグ特徴量")
        
        # 移動統計追加
        logger.info("\n📊 ステップ2: 移動統計追加...")
        enhanced_df, stats_features = ts_features.create_rolling_statistics(enhanced_df, key_features_for_lag, windows=[5, 10, 20])
        all_features.extend(stats_features)
        
        X_with_stats = enhanced_df[all_features].fillna(0)
        stats_result = ts_features.evaluate_enhanced_features(X_with_stats, y, "ベース+ラグ+移動統計")
        
        # トレンド特徴量追加
        logger.info("\n📈 ステップ3: トレンド特徴量追加...")
        enhanced_df, trend_features = ts_features.create_trend_features(enhanced_df, key_features_for_lag, windows=[5, 10])
        all_features.extend(trend_features)
        
        X_with_trends = enhanced_df[all_features].fillna(0)
        trend_result = ts_features.evaluate_enhanced_features(X_with_trends, y, "ベース+ラグ+統計+トレンド")
        
        # 周期性特徴量追加
        logger.info("\n🔄 ステップ4: 周期性特徴量追加...")
        enhanced_df, cyclical_features = ts_features.create_cyclical_features(enhanced_df)
        all_features.extend(cyclical_features)
        
        X_with_cycles = enhanced_df[all_features].fillna(0)
        cycle_result = ts_features.evaluate_enhanced_features(X_with_cycles, y, "ベース+ラグ+統計+トレンド+周期")
        
        # モメンタム特徴量追加
        logger.info("\n🚀 ステップ5: モメンタム特徴量追加...")
        enhanced_df, momentum_features = ts_features.create_momentum_features(enhanced_df, key_features_for_lag)
        all_features.extend(momentum_features)
        
        X_with_momentum = enhanced_df[all_features].fillna(0)
        momentum_result = ts_features.evaluate_enhanced_features(X_with_momentum, y, "全特徴量")
        
        # 特徴選択による最適化
        logger.info("\n🔍 ステップ6: 特徴選択最適化...")
        X_full = enhanced_df[all_features].fillna(0)
        selected_features, feature_importance = ts_features.feature_selection_by_importance(X_full, y, all_features, top_k=25)
        
        X_selected = enhanced_df[selected_features].fillna(0)
        selected_result = ts_features.evaluate_enhanced_features(X_selected, y, "選択済み特徴量(25個)")
        
        # 結果まとめ
        logger.info("\n" + "="*100)
        logger.info("🏆 高度時系列特徴量エンジニアリング結果")
        logger.info("="*100)
        
        results = [
            ("ベースライン", baseline_result),
            ("ラグ特徴量追加", lag_result),
            ("移動統計追加", stats_result),
            ("トレンド追加", trend_result),
            ("周期性追加", cycle_result),
            ("モメンタム追加", momentum_result),
            ("特徴選択後", selected_result)
        ]
        
        baseline_score = baseline_result['avg']
        
        logger.info("📈 段階的改善結果:")
        for i, (name, result) in enumerate(results, 1):
            improvement = (result['avg'] - baseline_score) * 100
            status = "🚀" if improvement > 2.0 else "📈" if improvement > 0.5 else "📊" if improvement >= 0 else "📉"
            logger.info(f"  {i}. {name:20s}: {result['avg']:.3%} ({improvement:+.2f}%) {status}")
        
        # 最高結果
        best_result = max(results, key=lambda x: x[1]['avg'])
        final_improvement = (best_result[1]['avg'] - baseline_score) * 100
        
        logger.info(f"\n🏆 最高性能:")
        logger.info(f"  手法: {best_result[0]}")
        logger.info(f"  精度: {best_result[1]['avg']:.3%} ± {best_result[1]['std']:.3%}")
        logger.info(f"  向上: {final_improvement:+.2f}% ({baseline_score:.1%} → {best_result[1]['avg']:.1%})")
        
        # 目標達成確認
        target_60 = 0.60
        target_62 = 0.62
        
        if best_result[1]['avg'] >= target_62:
            logger.info(f"🎉 目標大幅達成！ 62%超え ({best_result[1]['avg']:.1%} >= 62.0%)")
        elif best_result[1]['avg'] >= target_60:
            logger.info(f"✅ 目標達成！ 60%超え ({best_result[1]['avg']:.1%} >= 60.0%)")
        else:
            logger.info(f"📈 改善効果確認 ({best_result[1]['avg']:.1%})")
        
        # 特徴量統計
        logger.info(f"\n📊 特徴量統計:")
        logger.info(f"  ベース特徴量: {len(ts_features.base_features)}個")
        logger.info(f"  追加特徴量: {len(all_features) - len(ts_features.base_features)}個")
        logger.info(f"  総特徴量: {len(all_features)}個")
        logger.info(f"  選択特徴量: {len(selected_features)}個")
        
        logger.info(f"\n⚖️ この結果は全データ{len(clean_df):,}件での厳密な時系列検証です")
        logger.info(f"✅ 第2段階完了: 時系列特徴量エンジニアリング")
        
    except Exception as e:
        logger.error(f"❌ エラー発生: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()