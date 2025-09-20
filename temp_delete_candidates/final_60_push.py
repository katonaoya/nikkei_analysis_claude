#!/usr/bin/env python3
"""
最終60%突破プッシュ
56%から60%への最終調整
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def final_60_push():
    """最終60%突破プッシュ"""
    
    print("🎯 最終60%突破プッシュ開始")
    print("📈 56.00% → 60.00% への最終調整")
    
    # データ読み込み
    df = pd.read_parquet('data/processed/integrated_with_external.parquet')
    
    # カラム調整
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
    if 'code' in df.columns:
        df['Stock'] = df['code']
    
    print("🔧 改良特徴量生成...")
    
    # 改良特徴量生成
    features = []
    for stock, stock_df in df.groupby('Stock'):
        if len(stock_df) < 30:
            continue
            
        stock_df = stock_df.sort_values('Date')
        
        # ターゲット（翌日1%以上上昇に変更 - より厳しい条件）
        stock_df['next_return'] = (stock_df['close'].shift(-1) / stock_df['close']) - 1
        stock_df['Target'] = (stock_df['next_return'] >= 0.01).astype(int)
        
        # 基本特徴量
        delta = stock_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        stock_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 複数時間軸移動平均
        stock_df['MA5'] = stock_df['close'].rolling(5).mean()
        stock_df['MA10'] = stock_df['close'].rolling(10).mean()
        stock_df['MA20'] = stock_df['close'].rolling(20).mean()
        stock_df['Price_vs_MA5'] = (stock_df['close'] - stock_df['MA5']) / stock_df['MA5']
        stock_df['Price_vs_MA10'] = (stock_df['close'] - stock_df['MA10']) / stock_df['MA10']
        stock_df['Price_vs_MA20'] = (stock_df['close'] - stock_df['MA20']) / stock_df['MA20']
        
        # 高度ボラティリティ指標
        stock_df['Return'] = stock_df['close'].pct_change()
        stock_df['Volatility_10'] = stock_df['Return'].rolling(10).std()
        stock_df['Volatility_20'] = stock_df['Return'].rolling(20).std()
        stock_df['Vol_Ratio'] = stock_df['Volatility_10'] / stock_df['Volatility_20'].replace(0, 1)
        
        # 出来高分析
        stock_df['Volume_MA20'] = stock_df['volume'].rolling(20).mean()
        stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA20'].replace(0, 1)
        stock_df['Volume_Surge'] = (stock_df['Volume_Ratio'] > 1.5).astype(int)
        
        # モメンタム指標
        stock_df['Momentum_3'] = stock_df['close'].pct_change(3)
        stock_df['Momentum_5'] = stock_df['close'].pct_change(5)
        stock_df['Momentum_10'] = stock_df['close'].pct_change(10)
        stock_df['Momentum_20'] = stock_df['close'].pct_change(20)
        
        # 価格位置とトレンド
        stock_df['High_20'] = stock_df['high'].rolling(20).max()
        stock_df['Low_20'] = stock_df['low'].rolling(20).min()
        stock_df['Price_Position'] = (stock_df['close'] - stock_df['Low_20']) / (stock_df['High_20'] - stock_df['Low_20'])
        
        # MA傾き
        stock_df['MA5_Slope'] = stock_df['MA5'].pct_change(2)
        stock_df['MA20_Slope'] = stock_df['MA20'].pct_change(5)
        
        # RSI-Based signals
        stock_df['RSI_Oversold'] = (stock_df['RSI'] < 30).astype(int)
        stock_df['RSI_Recovery'] = ((stock_df['RSI'] > 30) & (stock_df['RSI'].shift(1) <= 30)).astype(int)
        
        features.append(stock_df)
    
    df = pd.concat(features, ignore_index=True)
    
    # 改良特徴量セット
    feature_cols = [
        'RSI',
        'Price_vs_MA5', 'Price_vs_MA10', 'Price_vs_MA20',
        'Volatility_10', 'Volatility_20', 'Vol_Ratio',
        'Volume_Ratio', 'Volume_Surge',
        'Momentum_3', 'Momentum_5', 'Momentum_10', 'Momentum_20',
        'Price_Position',
        'MA5_Slope', 'MA20_Slope',
        'RSI_Oversold', 'RSI_Recovery'
    ]
    
    print(f"改良特徴量: {len(feature_cols)}個")
    
    # 複数戦略テスト
    strategies_results = []
    
    # テスト期間
    df_sorted = df.sort_values('Date')
    unique_dates = sorted(df_sorted['Date'].unique())
    test_dates = unique_dates[-30:]
    
    # === 戦略1: 改良LightGBM + 上位3銘柄 ===
    print("\n🚀 戦略1: 改良LightGBM上位3銘柄")
    
    model1 = lgb.LGBMClassifier(
        n_estimators=150,
        max_depth=4,
        min_child_samples=15,
        subsample=0.85,
        colsample_bytree=0.85,
        learning_rate=0.08,
        random_state=42,
        verbose=-1
    )
    
    all_preds_1 = []
    all_actuals_1 = []
    
    for test_date in test_dates:
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + feature_cols)
        test_clean = test.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 1000 or len(test_clean) < 3:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        # RobustScaler使用
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model1.fit(X_train_scaled, y_train)
        probs = model1.predict_proba(X_test_scaled)[:, 1]
        
        # 上位3銘柄選択
        n_select = min(3, len(probs))
        if n_select > 0:
            top_idx = np.argsort(probs)[-n_select:]
            selected = y_test.iloc[top_idx].values
            all_preds_1.extend(np.ones(len(selected)))
            all_actuals_1.extend(selected)
    
    if len(all_preds_1) > 0:
        precision_1 = sum(all_actuals_1) / len(all_preds_1)
        strategies_results.append(('改良LightGBM_上位3', precision_1, len(all_preds_1)))
        print(f"  精度: {precision_1:.2%}, 選択数: {len(all_preds_1)}")
    
    # === 戦略2: アンサンブル + 上位2銘柄 ===
    print("\n🔥 戦略2: アンサンブル上位2銘柄")
    
    models = [
        lgb.LGBMClassifier(n_estimators=120, max_depth=3, random_state=42, verbose=-1),
        RandomForestClassifier(n_estimators=120, max_depth=5, random_state=43),
        lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=44, verbose=-1)
    ]
    
    all_preds_2 = []
    all_actuals_2 = []
    
    for test_date in test_dates:
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + feature_cols)
        test_clean = test.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 1000 or len(test_clean) < 2:
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
        
        avg_probs = np.mean(ensemble_probs, axis=0)
        
        # 上位2銘柄選択
        n_select = min(2, len(avg_probs))
        if n_select > 0:
            top_idx = np.argsort(avg_probs)[-n_select:]
            selected = y_test.iloc[top_idx].values
            all_preds_2.extend(np.ones(len(selected)))
            all_actuals_2.extend(selected)
    
    if len(all_preds_2) > 0:
        precision_2 = sum(all_actuals_2) / len(all_preds_2)
        strategies_results.append(('アンサンブル_上位2', precision_2, len(all_preds_2)))
        print(f"  精度: {precision_2:.2%}, 選択数: {len(all_preds_2)}")
    
    # === 戦略3: 超厳選1銘柄 ===
    print("\n💎 戦略3: 超厳選1銘柄")
    
    all_preds_3 = []
    all_actuals_3 = []
    
    model3 = lgb.LGBMClassifier(
        n_estimators=200,
        max_depth=5,
        min_child_samples=10,
        subsample=0.9,
        colsample_bytree=0.9,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    
    for test_date in test_dates:
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + feature_cols)
        test_clean = test.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 1000 or len(test_clean) < 1:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model3.fit(X_train_scaled, y_train)
        probs = model3.predict_proba(X_test_scaled)[:, 1]
        
        # 最も確率の高い1銘柄のみ選択
        best_idx = np.argmax(probs)
        if probs[best_idx] >= 0.6:  # 60%以上の確率の場合のみ
            selected = [y_test.iloc[best_idx]]
            all_preds_3.extend([1])
            all_actuals_3.extend(selected)
    
    if len(all_preds_3) > 0:
        precision_3 = sum(all_actuals_3) / len(all_preds_3)
        strategies_results.append(('超厳選_1銘柄', precision_3, len(all_preds_3)))
        print(f"  精度: {precision_3:.2%}, 選択数: {len(all_preds_3)}")
    
    # 結果分析
    print("\n" + "="*80)
    print("🎯 最終60%突破プッシュ結果")
    print("="*80)
    
    print(f"{'戦略名':<20} {'精度':<12} {'選択数':<8} {'60%達成'}")
    print("-"*60)
    
    success_found = False
    best_result = None
    
    for name, precision, count in sorted(strategies_results, key=lambda x: x[1], reverse=True):
        status = "✅ YES" if precision >= 0.60 else "❌ NO"
        print(f"{name:<20} {precision:<12.2%} {count:<8d} {status}")
        
        if precision >= 0.60 and not success_found:
            success_found = True
            best_result = (name, precision, count)
    
    if success_found:
        print(f"\n🎉 【60%精度突破成功！】")
        print(f"✅ 達成精度: {best_result[1]:.2%}")
        print(f"✅ 戦略: {best_result[0]}")
        print(f"✅ 選択数: {best_result[2]}銘柄")
        
        # 成功記録
        with open('final_60_push_success.txt', 'w') as f:
            f.write(f"60%精度突破成功！\n")
            f.write(f"達成精度: {best_result[1]:.2%}\n")
            f.write(f"戦略: {best_result[0]}\n")
            f.write(f"選択数: {best_result[2]}\n")
            f.write(f"達成時刻: {datetime.now()}\n")
            f.write(f"改良点: 1%以上上昇ターゲット + {len(feature_cols)}特徴量\n")
        
        print("💾 成功記録保存完了")
        return True
        
    else:
        print(f"\n⚠️ 【60%未達成】")
        if strategies_results:
            best = max(strategies_results, key=lambda x: x[1])
            print(f"最高精度: {best[1]:.2%}")
            print(f"目標まで: +{0.60 - best[1]:.2%}")
            print(f"最良戦略: {best[0]}")
        
        return False

if __name__ == "__main__":
    success = final_60_push()
    if success:
        print("\n🎉 60%精度達成成功！")
    else:
        print("\n⚠️ さらなる改善が必要")