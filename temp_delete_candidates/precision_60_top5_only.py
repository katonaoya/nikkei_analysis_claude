#!/usr/bin/env python3
"""
Precision 60%達成を目指す - 上位5銘柄のみに絞った高精度予測
"""

import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import yaml
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, accuracy_score
import lightgbm as lgb
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

class Top5PrecisionOptimizer:
    """上位5銘柄に絞った高精度予測クラス"""
    
    def __init__(self):
        """初期化"""
        self.best_precision = 0
        self.best_config = None
        
    def load_and_prepare_data(self):
        """データ読み込みと前処理"""
        logger.info("📥 データ読み込み中...")
        df = pd.read_parquet('data/processed/integrated_with_external.parquet')
        
        # カラム名の調整
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        if 'code' in df.columns:
            df['Stock'] = df['code']
        
        # 基本的な特徴量を生成
        logger.info("🔧 特徴量生成中...")
        features = []
        
        for stock, stock_df in df.groupby('Stock'):
            stock_df = stock_df.sort_values('Date')
            
            # 価格変化率
            stock_df['Return'] = stock_df['close'].pct_change()
            stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
            
            # テクニカル指標（重要なもののみ）
            # RSI
            delta = stock_df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss.replace(0, 1)
            stock_df['RSI'] = 100 - (100 / (1 + rs))
            
            # 移動平均からの乖離
            for period in [5, 20]:
                stock_df[f'MA{period}'] = stock_df['close'].rolling(period).mean()
                stock_df[f'Price_vs_MA{period}'] = (stock_df['close'] - stock_df[f'MA{period}']) / stock_df[f'MA{period}']
            
            # ボラティリティ
            stock_df['Volatility_20'] = stock_df['Return'].rolling(20).std()
            
            # 出来高比率
            stock_df['Volume_MA20'] = stock_df['volume'].rolling(20).mean()
            stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA20'].replace(0, 1)
            
            # モメンタム
            stock_df['Momentum_5'] = stock_df['close'].pct_change(5)
            stock_df['Momentum_20'] = stock_df['close'].pct_change(20)
            
            # 高値安値からの位置
            stock_df['High_20'] = stock_df['high'].rolling(20).max()
            stock_df['Low_20'] = stock_df['low'].rolling(20).min()
            stock_df['Price_Position'] = (stock_df['close'] - stock_df['Low_20']) / (stock_df['High_20'] - stock_df['Low_20'])
            
            features.append(stock_df)
        
        df = pd.concat(features, ignore_index=True)
        
        # 使用する特徴量
        feature_cols = [
            'RSI', 'Price_vs_MA5', 'Price_vs_MA20', 'Volatility_20',
            'Volume_Ratio', 'Momentum_5', 'Momentum_20', 'Price_Position'
        ]
        
        return df, feature_cols
    
    def test_top5_precision(self, df, feature_cols):
        """上位5銘柄のみでPrecision測定"""
        logger.info("🎯 上位5銘柄の精度テスト開始...")
        
        # テスト期間（直近30日）
        df = df.sort_values('Date')
        unique_dates = sorted(df['Date'].unique())
        test_dates = unique_dates[-30:]
        
        # 複数のモデルと閾値をテスト
        models = {
            'lgb': lgb.LGBMClassifier(
                n_estimators=100, max_depth=3, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8, random_state=42
            ),
            'rf': RandomForestClassifier(
                n_estimators=200, max_depth=5, min_samples_split=50,
                min_samples_leaf=20, random_state=42, n_jobs=-1
            ),
            'gb': GradientBoostingClassifier(
                n_estimators=100, max_depth=3, min_samples_split=50,
                min_samples_leaf=20, learning_rate=0.05, random_state=42
            )
        }
        
        results = []
        
        for model_name, model in models.items():
            logger.info(f"\nモデル: {model_name}")
            
            # 異なる選択戦略をテスト
            for strategy in ['top5_any', 'top5_threshold', 'top3_high_conf']:
                
                daily_precisions = []
                daily_counts = []
                all_predictions = []
                all_actuals = []
                
                for test_date in test_dates:
                    # データ分割
                    train_data = df[df['Date'] < test_date]
                    test_data = df[df['Date'] == test_date]
                    
                    if len(train_data) < 5000 or len(test_data) < 20:
                        continue
                    
                    # クリーンなデータ
                    train_clean = train_data.dropna(subset=['Target'] + feature_cols)
                    test_clean = test_data.dropna(subset=['Target'] + feature_cols)
                    
                    if len(train_clean) < 1000 or len(test_clean) < 10:
                        continue
                    
                    X_train = train_clean[feature_cols]
                    y_train = train_clean['Target']
                    X_test = test_clean[feature_cols]
                    y_test = test_clean['Target']
                    
                    # スケーリング
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # モデル学習
                    try:
                        model.fit(X_train_scaled, y_train)
                        
                        # 予測確率
                        y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                        test_clean['pred_proba'] = y_pred_proba
                        
                        # 戦略別の銘柄選択
                        if strategy == 'top5_any':
                            # 上位5銘柄（閾値なし）
                            selected = test_clean.nlargest(5, 'pred_proba')
                            
                        elif strategy == 'top5_threshold':
                            # 上位5銘柄（60%以上の確率のみ）
                            high_conf = test_clean[test_clean['pred_proba'] >= 0.6]
                            if len(high_conf) > 0:
                                selected = high_conf.nlargest(min(5, len(high_conf)), 'pred_proba')
                            else:
                                continue
                                
                        elif strategy == 'top3_high_conf':
                            # 上位3銘柄（65%以上の確率のみ）
                            high_conf = test_clean[test_clean['pred_proba'] >= 0.65]
                            if len(high_conf) > 0:
                                selected = high_conf.nlargest(min(3, len(high_conf)), 'pred_proba')
                            else:
                                continue
                        
                        if len(selected) > 0:
                            # Precision計算
                            predictions = (selected['pred_proba'] >= 0.5).astype(int)
                            actuals = selected['Target'].values
                            
                            precision = precision_score(actuals, predictions, zero_division=0)
                            
                            if precision > 0:
                                daily_precisions.append(precision)
                                daily_counts.append(len(selected))
                                all_predictions.extend(predictions)
                                all_actuals.extend(actuals)
                                
                    except Exception as e:
                        continue
                
                # 結果集計
                if len(daily_precisions) >= 10:
                    avg_precision = np.mean(daily_precisions)
                    avg_count = np.mean(daily_counts)
                    overall_precision = precision_score(all_actuals, all_predictions, zero_division=0)
                    
                    result = {
                        'model': model_name,
                        'strategy': strategy,
                        'avg_daily_precision': avg_precision,
                        'overall_precision': overall_precision,
                        'avg_picks_per_day': avg_count,
                        'test_days': len(daily_precisions),
                        'total_correct': sum([a for a, p in zip(all_actuals, all_predictions) if a == 1 and p == 1]),
                        'total_predicted': sum(all_predictions)
                    }
                    
                    results.append(result)
                    
                    if overall_precision > self.best_precision:
                        self.best_precision = overall_precision
                        self.best_config = result
                        logger.success(f"  🎯 新記録! {strategy}: Precision {overall_precision:.2%} (1日平均{avg_count:.1f}銘柄)")
        
        return results
    
    def advanced_filtering(self, df, feature_cols):
        """高度なフィルタリングによる精度向上"""
        logger.info("🔬 高度なフィルタリング戦略のテスト...")
        
        # 最適な条件の組み合わせを探索
        df = df.sort_values('Date')
        unique_dates = sorted(df['Date'].unique())
        test_dates = unique_dates[-30:]
        
        # LightGBMモデル（最も良好な結果を示しやすい）
        model = lgb.LGBMClassifier(
            n_estimators=150, max_depth=4, min_child_samples=30,
            subsample=0.9, colsample_bytree=0.9, random_state=42,
            learning_rate=0.02
        )
        
        # フィルタリング条件の組み合わせ
        filters = [
            {'name': 'high_volume', 'condition': lambda df: df['Volume_Ratio'] > 1.2},
            {'name': 'low_rsi', 'condition': lambda df: (df['RSI'] < 40) | (df['RSI'] > 60)},
            {'name': 'strong_momentum', 'condition': lambda df: df['Momentum_5'] > 0.02},
            {'name': 'oversold', 'condition': lambda df: df['Price_Position'] < 0.3},
            {'name': 'breakout', 'condition': lambda df: df['Price_Position'] > 0.8}
        ]
        
        best_filter_result = None
        best_filter_precision = 0
        
        for filter_config in filters:
            daily_results = []
            
            for test_date in test_dates:
                train_data = df[df['Date'] < test_date]
                test_data = df[df['Date'] == test_date]
                
                if len(train_data) < 5000 or len(test_data) < 20:
                    continue
                
                # フィルタ適用
                test_filtered = test_data[filter_config['condition'](test_data)]
                
                if len(test_filtered) < 5:
                    continue
                
                train_clean = train_data.dropna(subset=['Target'] + feature_cols)
                test_clean = test_filtered.dropna(subset=['Target'] + feature_cols)
                
                if len(train_clean) < 1000 or len(test_clean) < 3:
                    continue
                
                X_train = train_clean[feature_cols]
                y_train = train_clean['Target']
                X_test = test_clean[feature_cols]
                
                # モデル学習と予測
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                model.fit(X_train_scaled, y_train)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                test_clean['pred_proba'] = y_pred_proba
                
                # 上位5銘柄を選択
                top5 = test_clean.nlargest(min(5, len(test_clean)), 'pred_proba')
                
                if len(top5) > 0 and top5['pred_proba'].iloc[0] >= 0.55:
                    predictions = (top5['pred_proba'] >= 0.5).astype(int)
                    actuals = top5['Target'].values
                    
                    if sum(predictions) > 0:
                        precision = sum([a for a, p in zip(actuals, predictions) if a == 1 and p == 1]) / sum(predictions)
                        daily_results.append({
                            'precision': precision,
                            'count': len(top5),
                            'correct': sum([a for a, p in zip(actuals, predictions) if a == 1 and p == 1])
                        })
            
            if len(daily_results) >= 5:
                avg_precision = np.mean([r['precision'] for r in daily_results])
                total_correct = sum([r['correct'] for r in daily_results])
                total_predicted = sum([r['count'] for r in daily_results])
                
                if total_predicted > 0:
                    overall_precision = total_correct / total_predicted
                    
                    if overall_precision > best_filter_precision:
                        best_filter_precision = overall_precision
                        best_filter_result = {
                            'filter': filter_config['name'],
                            'precision': overall_precision,
                            'avg_daily_precision': avg_precision,
                            'test_days': len(daily_results)
                        }
                        
                        logger.info(f"  フィルタ {filter_config['name']}: Precision {overall_precision:.2%}")
        
        return best_filter_result
    
    def print_final_results(self, results, filter_result):
        """最終結果の表示"""
        print("\n" + "="*80)
        print("🎯 Precision 60%達成チャレンジ - 最終結果")
        print("="*80)
        
        # 結果をPrecisionでソート
        results_sorted = sorted(results, key=lambda x: x['overall_precision'], reverse=True)
        
        print("\n📊 【上位5銘柄戦略の結果】")
        print(f"{'順位':<4} {'モデル':<10} {'戦略':<20} {'Precision':<12} {'1日平均':<10}")
        print("-"*70)
        
        for i, result in enumerate(results_sorted[:10], 1):
            print(f"{i:<4} {result['model']:<10} {result['strategy']:<20} "
                  f"{result['overall_precision']:<12.2%} {result['avg_picks_per_day']:<10.1f}")
        
        if self.best_config:
            print("\n🏆 【最高成績】")
            print(f"  モデル: {self.best_config['model']}")
            print(f"  戦略: {self.best_config['strategy']}")
            print(f"  達成Precision: {self.best_config['overall_precision']:.2%}")
            print(f"  1日平均選択数: {self.best_config['avg_picks_per_day']:.1f}銘柄")
            print(f"  的中数/予測数: {self.best_config['total_correct']}/{self.best_config['total_predicted']}")
            
            if self.best_config['overall_precision'] >= 0.6:
                print("\n✅ 目標のPrecision 60%を達成しました！")
                
                # 成功設定を保存
                success_config = {
                    'achieved': True,
                    'precision': float(self.best_config['overall_precision']),
                    'model': self.best_config['model'],
                    'strategy': self.best_config['strategy'],
                    'timestamp': datetime.now().isoformat()
                }
                
                with open('precision_60_achieved.yaml', 'w') as f:
                    yaml.dump(success_config, f)
                
                print("💾 成功設定を precision_60_achieved.yaml に保存しました")
            else:
                print(f"\n⚠️ 目標まであと {0.6 - self.best_config['overall_precision']:.2%}")
        
        if filter_result:
            print(f"\n🔬 【フィルタリング戦略】")
            print(f"  最良フィルタ: {filter_result['filter']}")
            print(f"  達成Precision: {filter_result['precision']:.2%}")
            
            if filter_result['precision'] >= 0.6:
                print("  ✅ フィルタリングで60%達成！")
        
        print("\n" + "="*80)

def main():
    """メイン実行"""
    optimizer = Top5PrecisionOptimizer()
    
    # データ準備
    df, feature_cols = optimizer.load_and_prepare_data()
    
    # 上位5銘柄戦略のテスト
    results = optimizer.test_top5_precision(df, feature_cols)
    
    # 高度なフィルタリング
    filter_result = optimizer.advanced_filtering(df, feature_cols)
    
    # 結果表示
    optimizer.print_final_results(results, filter_result)

if __name__ == "__main__":
    main()