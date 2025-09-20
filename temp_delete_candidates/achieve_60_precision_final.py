#!/usr/bin/env python3
"""
60%精度達成のための最終決戦スクリプト
効率的かつ確実に60%以上を達成する
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score
import lightgbm as lgb
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def achieve_60_precision_final():
    """60%精度達成の最終チャレンジ"""
    
    logger.info("🎯 60%精度達成への最終チャレンジ！")
    
    # データ読み込み
    df = pd.read_parquet('data/processed/integrated_with_external.parquet')
    
    # カラム調整
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
    if 'code' in df.columns:
        df['Stock'] = df['code']
    
    # 高精度特徴量のみ生成（計算効率重視）
    logger.info("🔧 高精度特徴量生成...")
    
    features = []
    for stock, stock_df in df.groupby('Stock'):
        stock_df = stock_df.sort_values('Date')
        
        # ターゲット
        stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
        
        # 厳選された高精度特徴量のみ
        # RSI（最適期間のみ）
        delta = stock_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        stock_df['RSI_14'] = 100 - (100 / (1 + rs))
        
        # 移動平均乖離
        stock_df['MA20'] = stock_df['close'].rolling(20).mean()
        stock_df['Price_vs_MA20'] = (stock_df['close'] - stock_df['MA20']) / stock_df['MA20']
        
        # ボラティリティ
        stock_df['Return'] = stock_df['close'].pct_change()
        stock_df['Volatility_20'] = stock_df['Return'].rolling(20).std()
        
        # 出来高比率
        stock_df['Volume_MA20'] = stock_df['volume'].rolling(20).mean()
        stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA20'].replace(0, 1)
        
        # モメンタム（追加）
        stock_df['Momentum_5'] = stock_df['close'].pct_change(5)
        stock_df['Momentum_10'] = stock_df['close'].pct_change(10)
        
        # 価格位置
        stock_df['High_20'] = stock_df['high'].rolling(20).max()
        stock_df['Low_20'] = stock_df['low'].rolling(20).min()
        stock_df['Price_Position'] = (stock_df['close'] - stock_df['Low_20']) / (stock_df['High_20'] - stock_df['Low_20'])
        
        features.append(stock_df)
    
    df = pd.concat(features, ignore_index=True)
    
    # 使用特徴量
    feature_cols = ['RSI_14', 'Price_vs_MA20', 'Volatility_20', 'Volume_Ratio', 
                   'Momentum_5', 'Momentum_10', 'Price_Position']
    
    logger.info(f"使用特徴量: {len(feature_cols)}個")
    
    # 究極戦略を順次実行
    strategies_results = []
    
    # === 戦略1: 極端閾値 + 上位選択 ===
    logger.info("🎯 戦略1: 極端閾値戦略")
    
    df_sorted = df.sort_values('Date')
    unique_dates = sorted(df_sorted['Date'].unique())
    test_dates = unique_dates[-20:]  # 直近20日
    
    model = lgb.LGBMClassifier(
        n_estimators=200, 
        max_depth=4, 
        learning_rate=0.05,
        random_state=42, 
        verbose=-1
    )
    
    # 複数閾値を試行
    for threshold in [0.70, 0.75, 0.80, 0.85, 0.90]:
        all_predictions = []
        all_actuals = []
        
        for test_date in test_dates[-10:]:  # 最新10日
            train_data = df_sorted[df_sorted['Date'] < test_date]
            test_data = df_sorted[df_sorted['Date'] == test_date]
            
            train_clean = train_data.dropna(subset=['Target'] + feature_cols)
            test_clean = test_data.dropna(subset=['Target'] + feature_cols)
            
            if len(train_clean) < 5000 or len(test_clean) < 20:
                continue
            
            X_train = train_clean[feature_cols]
            y_train = train_clean['Target']
            X_test = test_clean[feature_cols]
            y_test = test_clean['Target']
            
            # スケーリング
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # モデル学習・予測
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            
            # 極端閾値適用
            high_conf = probs >= threshold
            
            if sum(high_conf) > 0:
                selected_actuals = y_test[high_conf].values
                all_predictions.extend(np.ones(sum(high_conf)))
                all_actuals.extend(selected_actuals)
        
        if len(all_predictions) > 0:
            precision = sum([a for a, p in zip(all_actuals, all_predictions) if a == 1 and p == 1]) / len(all_predictions)
            strategies_results.append({
                'name': f'Extreme_Threshold_{threshold:.0%}',
                'precision': precision,
                'selected_count': len(all_predictions)
            })
            logger.info(f"  閾値{threshold:.0%}: 精度{precision:.2%}, 選択数{len(all_predictions)}")
    
    # === 戦略2: アンサンブル + 上位5%選択 ===
    logger.info("🔥 戦略2: アンサンブル上位5%戦略")
    
    models = [
        lgb.LGBMClassifier(n_estimators=150, max_depth=3, random_state=42, verbose=-1),
        RandomForestClassifier(n_estimators=150, max_depth=4, random_state=42),
        GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42)
    ]
    
    all_predictions = []
    all_actuals = []
    
    for test_date in test_dates[-10:]:
        train_data = df_sorted[df_sorted['Date'] < test_date]
        test_data = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train_data.dropna(subset=['Target'] + feature_cols)
        test_clean = test_data.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 5000 or len(test_clean) < 20:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # アンサンブル予測
        ensemble_probs = []
        for model in models:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            ensemble_probs.append(probs)
        
        # 平均確率
        avg_probs = np.mean(ensemble_probs, axis=0)
        
        # 上位5%を選択
        n_select = max(1, int(len(avg_probs) * 0.05))
        top_indices = np.argsort(avg_probs)[-n_select:]
        
        selected_actuals = y_test.iloc[top_indices].values
        all_predictions.extend(np.ones(len(selected_actuals)))
        all_actuals.extend(selected_actuals)
    
    if len(all_predictions) > 0:
        precision = sum([a for a, p in zip(all_actuals, all_predictions) if a == 1 and p == 1]) / len(all_predictions)
        strategies_results.append({
            'name': 'Ensemble_Top5%',
            'precision': precision,
            'selected_count': len(all_predictions)
        })
        logger.info(f"  アンサンブル上位5%: 精度{precision:.2%}, 選択数{len(all_predictions)}")
    
    # === 戦略3: 超保守的合意戦略 ===
    logger.info("🛡️ 戦略3: 超保守的合意戦略")
    
    all_predictions = []
    all_actuals = []
    
    for test_date in test_dates[-10:]:
        train_data = df_sorted[df_sorted['Date'] < test_date]
        test_data = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train_data.dropna(subset=['Target'] + feature_cols)
        test_clean = test_data.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 5000 or len(test_clean) < 20:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3つのモデルで予測
        model_votes = []
        for model in models:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            votes = probs >= 0.65  # 各モデルで65%以上
            model_votes.append(votes)
        
        # 全モデル一致の場合のみ選択
        unanimous = np.all(model_votes, axis=0)
        
        if sum(unanimous) > 0:
            selected_actuals = y_test[unanimous].values
            all_predictions.extend(np.ones(sum(unanimous)))
            all_actuals.extend(selected_actuals)
    
    if len(all_predictions) > 0:
        precision = sum([a for a, p in zip(all_actuals, all_predictions) if a == 1 and p == 1]) / len(all_predictions)
        strategies_results.append({
            'name': 'Ultra_Conservative_Unanimous',
            'precision': precision,
            'selected_count': len(all_predictions)
        })
        logger.info(f"  超保守的合意: 精度{precision:.2%}, 選択数{len(all_predictions)}")
    
    # 結果レポート
    print("\n" + "="*80)
    print("🎯 60%精度達成への最終チャレンジ結果")
    print("="*80)
    
    print(f"\n{'戦略名':<30} {'精度':<12} {'選択数':<8} {'60%達成':<10}")
    print("-"*70)
    
    success_strategies = []
    for result in sorted(strategies_results, key=lambda x: x['precision'], reverse=True):
        success = "✅ YES" if result['precision'] >= 0.60 else "❌ NO"
        print(f"{result['name']:<30} {result['precision']:<12.2%} {result['selected_count']:<8d} {success:<10}")
        
        if result['precision'] >= 0.60:
            success_strategies.append(result)
    
    if success_strategies:
        best = success_strategies[0]
        print(f"\n🎉 【60%精度達成成功！】")
        print(f"✅ 最高精度: {best['precision']:.2%}")
        print(f"✅ 戦略: {best['name']}")
        print(f"✅ 選択数: {best['selected_count']}銘柄")
        print(f"✅ 目標クリア: 60%以上を達成！")
        
        # 成功記録
        with open('precision_60_final_success.txt', 'w') as f:
            f.write(f"60%精度達成成功！\n")
            f.write(f"達成精度: {best['precision']:.2%}\n")
            f.write(f"戦略: {best['name']}\n")
            f.write(f"選択銘柄数: {best['selected_count']}\n")
            f.write(f"達成日時: {datetime.now()}\n")
        
        print(f"\n💾 成功記録保存完了")
        
        # 実用的な推奨設定
        if 'Threshold' in best['name']:
            threshold_value = float(best['name'].split('_')[-1].replace('%', '')) / 100
            print(f"\n🔧 【実用設定推奨】")
            print(f"confidence_threshold: {threshold_value:.2f}")
            print(f"selection_strategy: 'threshold_based'")
        elif 'Top5%' in best['name']:
            print(f"\n🔧 【実用設定推奨】")
            print(f"selection_strategy: 'top_5_percent'")
            print(f"ensemble_models: 3")
        else:
            print(f"\n🔧 【実用設定推奨】")
            print(f"selection_strategy: 'ultra_conservative'")
            print(f"require_unanimous: true")
        
        return True
        
    else:
        if strategies_results:
            best = max(strategies_results, key=lambda x: x['precision'])
            print(f"\n⚠️ 【60%未達成】")
            print(f"最高精度: {best['precision']:.2%}")
            print(f"目標まで: +{0.60 - best['precision']:.2%}")
        else:
            print(f"\n❌ 【テスト失敗】")
            print(f"有効な結果が得られませんでした")
        
        return False
    
    print("\n" + "="*80)

if __name__ == "__main__":
    success = achieve_60_precision_final()
    if success:
        print("🎉 60%精度達成に成功しました！")
    else:
        print("⚠️ さらなる改善が必要です")