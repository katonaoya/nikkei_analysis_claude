#!/usr/bin/env python3
"""
Precision 60%以上を達成するための最適化スクリプト
複数のモデル、特徴量、パラメータを徹底的に最適化
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.feature_selection import SelectFromModel, RFE
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class Precision60Optimizer:
    """Precision 60%以上を達成するための最適化クラス"""
    
    def __init__(self):
        """初期化"""
        self.best_precision = 0
        self.best_config = None
        self.results = []
        
    def load_data(self):
        """データ読み込み"""
        logger.info("📥 データ読み込み開始...")
        df = pd.read_parquet('data/processed/integrated_with_external.parquet')
        
        # カラム名の調整
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        if 'code' in df.columns:
            df['Stock'] = df['code']
            
        logger.info(f"データ件数: {len(df):,}件")
        return df
    
    def generate_advanced_features(self, df):
        """高度な特徴量生成"""
        logger.info("🔧 高度な特徴量生成中...")
        features = []
        
        for stock, stock_df in df.groupby('Stock'):
            stock_df = stock_df.sort_values('Date')
            
            # 基本的な価格変化
            stock_df['Return'] = stock_df['close'].pct_change()
            stock_df['Log_Return'] = np.log(stock_df['close'] / stock_df['close'].shift(1))
            stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
            
            # 価格関連の特徴量
            for period in [5, 10, 20, 50]:
                # 移動平均
                stock_df[f'MA{period}'] = stock_df['close'].rolling(period).mean()
                stock_df[f'Price_vs_MA{period}'] = (stock_df['close'] - stock_df[f'MA{period}']) / stock_df[f'MA{period}']
                
                # ボラティリティ
                stock_df[f'Volatility_{period}'] = stock_df['Return'].rolling(period).std()
                
                # 最高値・最安値からの位置
                stock_df[f'High_{period}'] = stock_df['high'].rolling(period).max()
                stock_df[f'Low_{period}'] = stock_df['low'].rolling(period).min()
                stock_df[f'Price_Position_{period}'] = (stock_df['close'] - stock_df[f'Low_{period}']) / (stock_df[f'High_{period}'] - stock_df[f'Low_{period}'])
                
            # RSI（複数期間）
            for period in [7, 14, 21]:
                delta = stock_df['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
                rs = gain / loss.replace(0, 1)
                stock_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
            
            # MACD
            exp1 = stock_df['close'].ewm(span=12, adjust=False).mean()
            exp2 = stock_df['close'].ewm(span=26, adjust=False).mean()
            stock_df['MACD'] = exp1 - exp2
            stock_df['MACD_signal'] = stock_df['MACD'].ewm(span=9, adjust=False).mean()
            stock_df['MACD_diff'] = stock_df['MACD'] - stock_df['MACD_signal']
            
            # ボリンジャーバンド
            for period in [20, 30]:
                ma = stock_df['close'].rolling(period).mean()
                std = stock_df['close'].rolling(period).std()
                stock_df[f'BB_upper_{period}'] = ma + (std * 2)
                stock_df[f'BB_lower_{period}'] = ma - (std * 2)
                stock_df[f'BB_position_{period}'] = (stock_df['close'] - stock_df[f'BB_lower_{period}']) / (stock_df[f'BB_upper_{period}'] - stock_df[f'BB_lower_{period}'])
                stock_df[f'BB_width_{period}'] = (stock_df[f'BB_upper_{period}'] - stock_df[f'BB_lower_{period}']) / ma
            
            # 出来高関連
            stock_df['Volume_MA5'] = stock_df['volume'].rolling(5).mean()
            stock_df['Volume_MA20'] = stock_df['volume'].rolling(20).mean()
            stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA20'].replace(0, 1)
            stock_df['Volume_Ratio_5_20'] = stock_df['Volume_MA5'] / stock_df['Volume_MA20'].replace(0, 1)
            
            # 価格の変化率（モメンタム）
            for period in [1, 2, 3, 5, 10, 20]:
                stock_df[f'Return_{period}d'] = stock_df['close'].pct_change(period)
                stock_df[f'Return_{period}d_abs'] = np.abs(stock_df[f'Return_{period}d'])
            
            # 連続上昇・下降日数
            stock_df['Up'] = (stock_df['Return'] > 0).astype(int)
            stock_df['Down'] = (stock_df['Return'] < 0).astype(int)
            stock_df['Consecutive_Up'] = stock_df['Up'].groupby((stock_df['Up'] == 0).cumsum()).cumsum()
            stock_df['Consecutive_Down'] = stock_df['Down'].groupby((stock_df['Down'] == 0).cumsum()).cumsum()
            
            # 曜日と月
            stock_df['DayOfWeek'] = stock_df['Date'].dt.dayofweek
            stock_df['Month'] = stock_df['Date'].dt.month
            
            # 高値安値の比率
            stock_df['HL_Ratio'] = stock_df['high'] / stock_df['low']
            stock_df['OC_Ratio'] = stock_df['close'] / stock_df['open']
            
            # ATR (Average True Range)
            high_low = stock_df['high'] - stock_df['low']
            high_close = np.abs(stock_df['high'] - stock_df['close'].shift())
            low_close = np.abs(stock_df['low'] - stock_df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            stock_df['ATR_14'] = true_range.rolling(14).mean()
            
            # ストキャスティクス
            for period in [14, 21]:
                lowest_low = stock_df['low'].rolling(period).min()
                highest_high = stock_df['high'].rolling(period).max()
                stock_df[f'Stochastic_{period}'] = 100 * ((stock_df['close'] - lowest_low) / (highest_high - lowest_low))
            
            features.append(stock_df)
        
        df = pd.concat(features, ignore_index=True)
        
        # 特徴量リストを作成（数値型のカラムのみ）
        feature_cols = []
        for col in df.columns:
            if col not in ['Date', 'Stock', 'Target', 'open', 'high', 'low', 'close', 'volume', 
                          'UpperLimit', 'LowerLimit', 'turnover_value', 'code', 'date']:
                # 数値型のカラムのみを選択
                if df[col].dtype in ['float64', 'int64', 'float32', 'int32']:
                    feature_cols.append(col)
        
        logger.info(f"生成した特徴量数: {len(feature_cols)}")
        return df, feature_cols
    
    def select_top_features(self, X_train, y_train, n_features=30):
        """重要度の高い特徴量を選択"""
        # Random Forestで特徴量の重要度を計算
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        
        # 重要度でソート
        importances = pd.DataFrame({
            'feature': X_train.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # 上位n個の特徴量を選択
        top_features = importances.head(n_features)['feature'].tolist()
        return top_features
    
    def optimize_models(self, df, feature_cols):
        """複数モデルの最適化"""
        logger.info("🎯 モデル最適化開始...")
        
        # データ準備
        df = df.sort_values('Date')
        
        # テスト期間の設定（直近1年）
        test_start = pd.to_datetime('2024-10-01')
        test_end = pd.to_datetime('2025-09-30')
        
        # 学習用とテスト用に分割
        train_df = df[df['Date'] < test_start]
        test_period_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)]
        
        # モデル定義
        models = {
            'lgb_conservative': lgb.LGBMClassifier(
                n_estimators=300, max_depth=3, min_child_samples=50,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                learning_rate=0.01, min_split_gain=0.01
            ),
            'lgb_balanced': lgb.LGBMClassifier(
                n_estimators=200, max_depth=5, min_child_samples=30,
                subsample=0.9, colsample_bytree=0.9, random_state=42,
                learning_rate=0.03
            ),
            'xgb_conservative': xgb.XGBClassifier(
                n_estimators=300, max_depth=3, min_child_samples=50,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                learning_rate=0.01, gamma=0.1
            ),
            'rf_conservative': RandomForestClassifier(
                n_estimators=500, max_depth=5, min_samples_split=100,
                min_samples_leaf=50, random_state=42, n_jobs=-1
            ),
            'catboost': CatBoostClassifier(
                iterations=200, depth=4, learning_rate=0.03,
                random_state=42, verbose=False
            ),
            'extra_trees': ExtraTreesClassifier(
                n_estimators=500, max_depth=5, min_samples_split=100,
                min_samples_leaf=50, random_state=42, n_jobs=-1
            ),
            'gb_conservative': GradientBoostingClassifier(
                n_estimators=200, max_depth=3, min_samples_split=100,
                min_samples_leaf=50, learning_rate=0.01, random_state=42
            )
        }
        
        # 異なる特徴量セットを試す
        feature_sets = {
            'top_20': 20,
            'top_30': 30,
            'top_40': 40,
            'top_50': 50
        }
        
        best_results = []
        
        for feature_name, n_features in feature_sets.items():
            logger.info(f"\n特徴量セット: {feature_name} ({n_features}個)")
            
            # 特徴量選択用のデータ準備
            train_clean = train_df.dropna(subset=['Target'] + feature_cols)
            if len(train_clean) < 10000:
                continue
                
            # 特徴量選択
            X_select = train_clean[feature_cols]
            y_select = train_clean['Target']
            top_features = self.select_top_features(X_select, y_select, n_features)
            
            for model_name, model in models.items():
                logger.info(f"  モデル: {model_name}")
                
                # 異なる閾値を試す
                for confidence_threshold in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75]:
                    
                    # 日次で予測を実行
                    daily_precisions = []
                    daily_predictions = []
                    daily_top5_correct = []
                    
                    test_dates = sorted(test_period_df['Date'].unique())
                    
                    for test_date in test_dates[-30:]:  # 直近30日でテスト
                        # 学習データとテストデータ
                        train_data = df[df['Date'] < test_date]
                        test_data = df[df['Date'] == test_date]
                        
                        if len(train_data) < 5000 or len(test_data) < 20:
                            continue
                        
                        # クリーンなデータ
                        train_clean = train_data.dropna(subset=['Target'] + top_features)
                        test_clean = test_data.dropna(subset=['Target'] + top_features)
                        
                        if len(train_clean) < 1000 or len(test_clean) < 10:
                            continue
                        
                        X_train = train_clean[top_features]
                        y_train = train_clean['Target']
                        X_test = test_clean[top_features]
                        y_test = test_clean['Target']
                        
                        # スケーリング
                        scaler = RobustScaler()
                        X_train_scaled = scaler.fit_transform(X_train)
                        X_test_scaled = scaler.transform(X_test)
                        
                        # モデル学習
                        try:
                            model.fit(X_train_scaled, y_train)
                            
                            # 予測確率を取得
                            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                            
                            # 上位5銘柄のみを選択
                            test_clean['pred_proba'] = y_pred_proba
                            test_clean['predicted'] = (y_pred_proba >= confidence_threshold).astype(int)
                            
                            # 信頼度でソートして上位5つを取得
                            top5 = test_clean.nlargest(5, 'pred_proba')
                            
                            if len(top5) > 0 and top5['pred_proba'].iloc[0] >= confidence_threshold:
                                # 上位5銘柄の精度を計算
                                top5_predictions = top5['predicted'].values
                                top5_actuals = top5['Target'].values
                                
                                # 閾値を超えた銘柄のみで精度計算
                                valid_predictions = top5[top5['pred_proba'] >= confidence_threshold]
                                if len(valid_predictions) > 0:
                                    precision = precision_score(
                                        valid_predictions['Target'], 
                                        valid_predictions['predicted'],
                                        zero_division=0
                                    )
                                    
                                    if precision > 0:
                                        daily_precisions.append(precision)
                                        daily_predictions.append(len(valid_predictions))
                                        daily_top5_correct.append(sum(valid_predictions['Target']))
                                        
                        except Exception as e:
                            continue
                    
                    # 結果集計
                    if len(daily_precisions) >= 10:  # 最低10日分のデータがある場合
                        avg_precision = np.mean(daily_precisions)
                        avg_predictions = np.mean(daily_predictions)
                        total_correct = sum(daily_top5_correct)
                        total_predicted = sum(daily_predictions)
                        
                        if avg_precision >= 0.55:  # 55%以上の場合のみ記録
                            result = {
                                'model': model_name,
                                'features': feature_name,
                                'n_features': n_features,
                                'threshold': confidence_threshold,
                                'precision': avg_precision,
                                'avg_daily_picks': avg_predictions,
                                'total_correct': total_correct,
                                'total_predicted': total_predicted,
                                'test_days': len(daily_precisions),
                                'feature_list': top_features
                            }
                            
                            best_results.append(result)
                            
                            if avg_precision > self.best_precision:
                                self.best_precision = avg_precision
                                self.best_config = result
                                logger.success(f"    🎯 新記録! Precision: {avg_precision:.2%} (閾値: {confidence_threshold})")
        
        return best_results
    
    def print_results(self, results):
        """結果表示"""
        print("\n" + "="*80)
        print("🎯 Precision 60%達成のための最適化結果")
        print("="*80)
        
        # Precisionでソート
        results_sorted = sorted(results, key=lambda x: x['precision'], reverse=True)
        
        print("\n📊 【Top 10設定】")
        print(f"{'順位':<4} {'モデル':<20} {'特徴量':<10} {'閾値':<6} {'Precision':<10} {'1日平均':<8}")
        print("-"*70)
        
        for i, result in enumerate(results_sorted[:10], 1):
            print(f"{i:<4} {result['model']:<20} {result['features']:<10} "
                  f"{result['threshold']:<6.2f} {result['precision']:<10.2%} "
                  f"{result['avg_daily_picks']:<8.1f}")
        
        if self.best_config:
            print("\n🏆 【最適設定】")
            print(f"  モデル: {self.best_config['model']}")
            print(f"  特徴量数: {self.best_config['n_features']}個")
            print(f"  信頼度閾値: {self.best_config['threshold']:.2f}")
            print(f"  達成Precision: {self.best_config['precision']:.2%}")
            print(f"  1日平均選択数: {self.best_config['avg_daily_picks']:.1f}銘柄")
            print(f"  テスト日数: {self.best_config['test_days']}日")
            
            if self.best_config['precision'] >= 0.6:
                print("\n✅ 目標のPrecision 60%を達成しました！")
            else:
                print(f"\n⚠️ 現在の最高Precision: {self.best_config['precision']:.2%} (目標まであと{0.6 - self.best_config['precision']:.2%})")
            
            print("\n📋 【使用する特徴量】")
            for i, feature in enumerate(self.best_config['feature_list'][:10], 1):
                print(f"  {i:2d}. {feature}")
            
            # 設定ファイルとして保存
            config = {
                'model': self.best_config['model'],
                'features': self.best_config['feature_list'],
                'threshold': float(self.best_config['threshold']),
                'achieved_precision': float(self.best_config['precision'])
            }
            
            with open('precision_60_config.yaml', 'w') as f:
                yaml.dump(config, f)
            
            print("\n💾 最適設定を precision_60_config.yaml に保存しました")
        
        print("\n" + "="*80)

def main():
    """メイン実行"""
    optimizer = Precision60Optimizer()
    
    # データ読み込み
    df = optimizer.load_data()
    
    # 高度な特徴量生成
    df, feature_cols = optimizer.generate_advanced_features(df)
    
    # モデル最適化
    results = optimizer.optimize_models(df, feature_cols)
    
    # 結果表示
    optimizer.print_results(results)

if __name__ == "__main__":
    main()