#!/usr/bin/env python3
"""
マーケットデータのみでの60%精度テスト
J-Quants認証なしで実行可能
"""

import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

from yahoo_market_data import YahooMarketData
from loguru import logger

class MarketDataOnlyTest:
    """マーケットデータのみでの精度テスト"""
    
    def __init__(self):
        self.base_data_file = "data/processed/integrated_with_external.parquet"
    
    def load_base_data(self) -> pd.DataFrame:
        """ベースデータ読み込み"""
        try:
            df = pd.read_parquet(self.base_data_file)
            
            # カラム統一
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
            if 'code' in df.columns:
                df['Stock'] = df['code'].astype(str)
            
            logger.success(f"✅ ベースデータ読み込み: {len(df)}件")
            return df
        except Exception as e:
            logger.error(f"❌ ベースデータ読み込み失敗: {e}")
            return pd.DataFrame()
    
    def integrate_market_data_only(self, base_df: pd.DataFrame) -> pd.DataFrame:
        """マーケットデータのみ統合"""
        logger.info("🔄 マーケットデータ統合中...")
        
        try:
            # Yahoo Financeからマーケットデータ取得
            market_data = YahooMarketData()
            data_dict = market_data.get_all_market_data(period="2y")
            
            if not data_dict:
                logger.warning("マーケットデータが取得できませんでした")
                return base_df
            
            # マーケット特徴量生成
            market_features = market_data.calculate_market_features(data_dict)
            
            if market_features.empty:
                logger.warning("マーケット特徴量が生成できませんでした")
                return base_df
            
            # ベースデータとマージ
            enhanced_df = base_df.merge(market_features, on='Date', how='left')
            
            # 前方補完で欠損値を埋める
            market_cols = [col for col in market_features.columns if col != 'Date']
            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(method='ffill')
            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(method='bfill')
            enhanced_df[market_cols] = enhanced_df[market_cols].fillna(0)
            
            logger.success(f"✅ マーケットデータ統合完了: {len(market_cols)}特徴量追加")
            return enhanced_df
            
        except Exception as e:
            logger.error(f"❌ マーケットデータ統合失敗: {e}")
            return base_df
    
    def create_enhanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """拡張特徴量生成"""
        logger.info("🔧 拡張特徴量生成中...")
        
        enhanced_df = df.copy()
        
        # ターゲット生成
        enhanced_df = enhanced_df.sort_values(['Stock', 'Date'])
        enhanced_df['next_high'] = enhanced_df.groupby('Stock')['high'].shift(-1)
        enhanced_df['Target'] = (enhanced_df['next_high'] > enhanced_df['close'] * 1.01).astype(int)
        
        # 既存特徴量の改良
        for stock, stock_df in enhanced_df.groupby('Stock'):
            stock_mask = enhanced_df['Stock'] == stock
            stock_data = enhanced_df[stock_mask].sort_values('Date')
            
            # RSI改良
            if len(stock_data) > 20:
                delta = stock_data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss.replace(0, 1)
                rsi = 100 - (100 / (1 + rs))
                enhanced_df.loc[stock_mask, 'Enhanced_RSI'] = rsi
                
                # RSI divergence
                enhanced_df.loc[stock_mask, 'RSI_Divergence'] = rsi - rsi.rolling(5).mean()
            
            # 複合モメンタム指標
            if len(stock_data) > 30:
                short_ma = stock_data['close'].rolling(5).mean()
                long_ma = stock_data['close'].rolling(20).mean()
                enhanced_df.loc[stock_mask, 'MA_Cross_Signal'] = (short_ma > long_ma).astype(int)
                enhanced_df.loc[stock_mask, 'MA_Distance'] = (short_ma - long_ma) / long_ma
            
            # ボリンジャーバンド風指標
            if len(stock_data) > 20:
                ma = stock_data['close'].rolling(20).mean()
                std = stock_data['close'].rolling(20).std()
                enhanced_df.loc[stock_mask, 'BB_Position'] = (stock_data['close'] - ma) / (std * 2)
        
        # マーケット関連特徴量の改良
        if 'nikkei225_close' in enhanced_df.columns:
            # 市場との相関強度
            enhanced_df['Market_Sync'] = enhanced_df['close'].pct_change().rolling(20).corr(enhanced_df['nikkei225_return_1d'])
            
            # 相対パフォーマンス
            enhanced_df['Relative_Performance'] = enhanced_df['close'].pct_change() - enhanced_df['nikkei225_return_1d']
        
        if 'vix_close' in enhanced_df.columns:
            # VIX レジーム別特徴量
            enhanced_df['VIX_Regime'] = pd.cut(enhanced_df['vix_close'], 
                                             bins=[0, 15, 25, 100], 
                                             labels=[0, 1, 2]).astype(int)
        
        # 欠損値処理
        enhanced_df = enhanced_df.fillna(method='ffill').fillna(0)
        
        logger.success(f"✅ 拡張特徴量生成完了")
        return enhanced_df
    
    def run_market_enhanced_test(self) -> bool:
        """マーケットデータ拡張テスト実行"""
        logger.info("🎯 マーケットデータ拡張による60%精度テスト開始")
        
        # データ準備
        base_df = self.load_base_data()
        if base_df.empty:
            return False
        
        # 主要銘柄に限定（処理速度向上）
        stock_counts = base_df['Stock'].value_counts()
        major_stocks = stock_counts[stock_counts >= 200].head(100).index.tolist()
        base_df = base_df[base_df['Stock'].isin(major_stocks)]
        
        logger.info(f"対象銘柄: {len(major_stocks)}銘柄")
        
        # マーケットデータ統合
        enhanced_df = self.integrate_market_data_only(base_df)
        
        # 拡張特徴量生成
        final_df = self.create_enhanced_features(enhanced_df)
        
        # 特徴量選択
        feature_cols = []
        for col in final_df.columns:
            if col not in ['Date', 'Stock', 'Target', 'next_high'] and final_df[col].dtype in ['int64', 'float64']:
                feature_cols.append(col)
        
        logger.info(f"使用特徴量数: {len(feature_cols)}")
        
        # テスト実行
        df_sorted = final_df.sort_values(['Stock', 'Date'])
        unique_dates = sorted(df_sorted['Date'].unique())
        test_dates = unique_dates[-20:]  # 最新20日
        
        strategies = []
        
        # === 戦略1: マーケット拡張LightGBM ===
        logger.info("🚀 戦略1: マーケット拡張LightGBM")
        
        strategy1_preds = []
        strategy1_actuals = []
        
        for test_date in test_dates[-10:]:
            train_data = df_sorted[df_sorted['Date'] < test_date]
            test_data = df_sorted[df_sorted['Date'] == test_date]
            
            train_clean = train_data.dropna(subset=['Target'] + feature_cols)
            test_clean = test_data.dropna(subset=['Target'] + feature_cols)
            
            if len(train_clean) < 500 or len(test_clean) < 3:
                continue
            
            X_train = train_clean[feature_cols]
            y_train = train_clean['Target']
            X_test = test_clean[feature_cols]
            y_test = test_clean['Target']
            
            # 特徴量選択
            selector = SelectKBest(score_func=f_classif, k=min(20, len(feature_cols)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            # スケーリング
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # モデル学習
            model = lgb.LGBMClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.08,
                subsample=0.85,
                colsample_bytree=0.85,
                random_state=42,
                verbose=-1
            )
            
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            
            # 上位3銘柄選択
            n_select = min(3, len(probs))
            top_indices = np.argsort(probs)[-n_select:]
            
            selected_actuals = y_test.iloc[top_indices].values
            strategy1_preds.extend([1] * len(selected_actuals))
            strategy1_actuals.extend(selected_actuals)
        
        if strategy1_preds:
            precision1 = sum(strategy1_actuals) / len(strategy1_actuals)
            strategies.append(('マーケット拡張LightGBM', precision1, len(strategy1_preds)))
            logger.info(f"  結果: {precision1:.2%}")
        
        # === 戦略2: アンサンブル上位2銘柄 ===
        logger.info("🔥 戦略2: マーケット拡張アンサンブル")
        
        strategy2_preds = []
        strategy2_actuals = []
        
        for test_date in test_dates[-10:]:
            train_data = df_sorted[df_sorted['Date'] < test_date]
            test_data = df_sorted[df_sorted['Date'] == test_date]
            
            train_clean = train_data.dropna(subset=['Target'] + feature_cols)
            test_clean = test_data.dropna(subset=['Target'] + feature_cols)
            
            if len(train_clean) < 500 or len(test_clean) < 2:
                continue
            
            X_train = train_clean[feature_cols]
            y_train = train_clean['Target']
            X_test = test_clean[feature_cols]
            y_test = test_clean['Target']
            
            # 特徴量選択
            selector = SelectKBest(score_func=f_classif, k=min(15, len(feature_cols)))
            X_train_selected = selector.fit_transform(X_train, y_train)
            X_test_selected = selector.transform(X_test)
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train_selected)
            X_test_scaled = scaler.transform(X_test_selected)
            
            # アンサンブル
            models = [
                lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1),
                RandomForestClassifier(n_estimators=100, max_depth=4, random_state=43)
            ]
            
            ensemble_probs = []
            for model in models:
                model.fit(X_train_scaled, y_train)
                probs = model.predict_proba(X_test_scaled)[:, 1]
                ensemble_probs.append(probs)
            
            avg_probs = np.mean(ensemble_probs, axis=0)
            
            # 上位2銘柄選択
            n_select = min(2, len(avg_probs))
            top_indices = np.argsort(avg_probs)[-n_select:]
            
            selected_actuals = y_test.iloc[top_indices].values
            strategy2_preds.extend([1] * len(selected_actuals))
            strategy2_actuals.extend(selected_actuals)
        
        if strategy2_preds:
            precision2 = sum(strategy2_actuals) / len(strategy2_actuals)
            strategies.append(('マーケット拡張アンサンブル', precision2, len(strategy2_preds)))
            logger.info(f"  結果: {precision2:.2%}")
        
        # 結果表示
        print("\\n" + "="*70)
        print("🎯 マーケットデータ拡張による60%精度テスト結果")
        print("="*70)
        
        print(f"{'戦略名':<25} {'精度':<12} {'選択数':<8} {'60%達成'}")
        print("-"*55)
        
        best_precision = 0
        success = False
        
        for name, precision, count in sorted(strategies, key=lambda x: x[1], reverse=True):
            status = "✅ YES" if precision >= 0.60 else "❌ NO"
            print(f"{name:<25} {precision:<12.2%} {count:<8d} {status}")
            
            if precision >= 0.60:
                success = True
            if precision > best_precision:
                best_precision = precision
        
        if success:
            print(f"\\n🎉 【60%精度達成成功！】")
            print(f"マーケットデータの統合により60%を達成しました！")
        else:
            print(f"\\n📊 結果分析:")
            print(f"最高精度: {best_precision:.2%}")
            print(f"従来の56%から{best_precision-0.56:.1%}ポイント改善")
            if best_precision >= 0.58:
                print("ファンダメンタルデータ追加で60%達成可能")
            else:
                print("さらなるデータ統合が必要")
        
        return success

# 実行
if __name__ == "__main__":
    test = MarketDataOnlyTest()
    success = test.run_market_enhanced_test()
    
    if success:
        print("\\n🎉 マーケットデータ統合により60%精度達成成功！")
    else:
        print("\\n⚠️ ファンダメンタルデータ追加で60%達成を目指しましょう")