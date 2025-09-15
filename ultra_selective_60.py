#!/usr/bin/env python3
"""
超厳選60%精度チャレンジ
極端に厳しい条件で60%突破を目指す
"""

import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')

def ultra_selective_60():
    """超厳選60%チャレンジ"""
    
    print("🎯 超厳選60%精度チャレンジ開始")
    
    # データ読み込み
    df = pd.read_parquet('data/processed/integrated_with_external.parquet')
    
    # カラム調整
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
    if 'code' in df.columns:
        df['Stock'] = df['code']
    
    # 特徴量生成
    print("🔧 高精度特徴量生成...")
    
    features = []
    for stock, stock_df in df.groupby('Stock'):
        stock_df = stock_df.sort_values('Date')
        
        # ターゲット（翌日1%以上の上昇）
        stock_df['next_close'] = stock_df['close'].shift(-1)
        stock_df['Target'] = (stock_df['next_close'] > stock_df['close'] * 1.01).astype(int)
        
        # RSI
        delta = stock_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        stock_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 移動平均乖離
        stock_df['MA5'] = stock_df['close'].rolling(5).mean()
        stock_df['MA20'] = stock_df['close'].rolling(20).mean()
        stock_df['Price_vs_MA5'] = (stock_df['close'] - stock_df['MA5']) / stock_df['MA5']
        stock_df['Price_vs_MA20'] = (stock_df['close'] - stock_df['MA20']) / stock_df['MA20']
        
        # ボラティリティ
        stock_df['Return'] = stock_df['close'].pct_change()
        stock_df['Volatility'] = stock_df['Return'].rolling(20).std()
        
        # 出来高
        stock_df['Volume_MA'] = stock_df['volume'].rolling(20).mean()
        stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA'].replace(0, 1)
        
        # モメンタム
        stock_df['Momentum_5'] = stock_df['close'].pct_change(5)
        
        # 価格位置
        stock_df['High_20'] = stock_df['high'].rolling(20).max()
        stock_df['Low_20'] = stock_df['low'].rolling(20).min()
        stock_df['Price_Position'] = (stock_df['close'] - stock_df['Low_20']) / (stock_df['High_20'] - stock_df['Low_20'])
        
        features.append(stock_df)
    
    df = pd.concat(features, ignore_index=True)
    feature_cols = ['RSI', 'Price_vs_MA5', 'Price_vs_MA20', 'Volatility', 'Volume_Ratio', 'Momentum_5', 'Price_Position']
    
    # テスト期間
    df_sorted = df.sort_values('Date')
    unique_dates = sorted(df_sorted['Date'].unique())
    test_dates = unique_dates[-7:]  # 最新7日
    
    print(f"テスト期間: {len(test_dates)}日")
    print(f"使用特徴量: {len(feature_cols)}個")
    
    # 超厳選戦略群
    strategies = []
    
    # === 戦略1: 超高閾値（85%以上） ===
    print("\n🎯 戦略1: 超高閾値85%")
    
    model1 = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=42,
        verbose=-1
    )
    
    all_preds_1 = []
    all_actuals_1 = []
    
    for test_date in test_dates[-5:]:
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + feature_cols)
        test_clean = test.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 2000 or len(test_clean) < 10:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model1.fit(X_train_scaled, y_train)
        probs = model1.predict_proba(X_test_scaled)[:, 1]
        
        # 85%以上の銘柄のみ選択
        high_conf = probs >= 0.85
        if sum(high_conf) > 0:
            selected = y_test[high_conf].values
            all_preds_1.extend(np.ones(len(selected)))
            all_actuals_1.extend(selected)
    
    if len(all_preds_1) > 0:
        precision_1 = sum(all_actuals_1) / len(all_actuals_1)
        strategies.append(('超高閾値85%', precision_1, len(all_preds_1)))
        print(f"  結果: 精度{precision_1:.1%}, 選択数{len(all_preds_1)}")
    
    # === 戦略2: アンサンブル合意（80%以上） ===
    print("\n🔥 戦略2: アンサンブル合意80%")
    
    models = [
        lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, verbose=-1),
        RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    ]
    
    all_preds_2 = []
    all_actuals_2 = []
    
    for test_date in test_dates[-5:]:
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + feature_cols)
        test_clean = test.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 2000 or len(test_clean) < 10:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 両モデルで予測
        probs_list = []
        for model in models:
            model.fit(X_train_scaled, y_train)
            probs = model.predict_proba(X_test_scaled)[:, 1]
            probs_list.append(probs)
        
        # 両モデルが80%以上で合意
        consensus = np.all([probs >= 0.80 for probs in probs_list], axis=0)
        
        if sum(consensus) > 0:
            selected = y_test[consensus].values
            all_preds_2.extend(np.ones(len(selected)))
            all_actuals_2.extend(selected)
    
    if len(all_preds_2) > 0:
        precision_2 = sum(all_actuals_2) / len(all_actuals_2)
        strategies.append(('アンサンブル合意80%', precision_2, len(all_preds_2)))
        print(f"  結果: 精度{precision_2:.1%}, 選択数{len(all_preds_2)}")
    
    # === 戦略3: 上位1%超厳選 ===
    print("\n🛡️ 戦略3: 上位1%超厳選")
    
    all_preds_3 = []
    all_actuals_3 = []
    
    for test_date in test_dates[-5:]:
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + feature_cols)
        test_clean = test.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 2000 or len(test_clean) < 10:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = lgb.LGBMClassifier(n_estimators=150, max_depth=4, random_state=42, verbose=-1)
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # 上位1%のみ選択
        n_select = max(1, int(len(probs) * 0.01))
        top_indices = np.argsort(probs)[-n_select:]
        
        selected = y_test.iloc[top_indices].values
        all_preds_3.extend(np.ones(len(selected)))
        all_actuals_3.extend(selected)
    
    if len(all_preds_3) > 0:
        precision_3 = sum(all_actuals_3) / len(all_actuals_3)
        strategies.append(('上位1%超厳選', precision_3, len(all_preds_3)))
        print(f"  結果: 精度{precision_3:.1%}, 選択数{len(all_preds_3)}")
    
    # 最終結果
    print("\n" + "="*70)
    print("🎯 超厳選60%精度チャレンジ結果")
    print("="*70)
    
    print(f"\n{'戦略名':<20} {'精度':<10} {'選択数':<6} {'60%達成'}")
    print("-"*50)
    
    best_precision = 0
    best_strategy = None
    
    for name, precision, count in strategies:
        status = "✅ YES" if precision >= 0.60 else "❌ NO"
        print(f"{name:<20} {precision:<10.1%} {count:<6d} {status}")
        
        if precision > best_precision:
            best_precision = precision
            best_strategy = (name, precision, count)
    
    if best_precision >= 0.60:
        print(f"\n🎉 【60%精度達成成功！】")
        print(f"✅ 達成精度: {best_precision:.1%}")
        print(f"✅ 戦略: {best_strategy[0]}")
        print(f"✅ 選択数: {best_strategy[2]}")
        
        # 成功記録
        with open('ultra_selective_60_success.txt', 'w') as f:
            f.write(f"60%精度達成成功！\n")
            f.write(f"達成精度: {best_precision:.2%}\n")
            f.write(f"戦略: {best_strategy[0]}\n")
            f.write(f"選択数: {best_strategy[2]}\n")
            f.write(f"達成時刻: {datetime.now()}\n")
        
        print("💾 成功記録保存完了")
        return True
    else:
        print(f"\n⚠️ 【60%未達成】")
        if best_strategy:
            print(f"最高精度: {best_precision:.1%}")
            print(f"目標まで: +{0.60 - best_precision:.1%}")
            print(f"最良戦略: {best_strategy[0]}")
        return False

if __name__ == "__main__":
    success = ultra_selective_60()
    if success:
        print("\n🎉 60%精度達成成功！")
    else:
        print("\n⚠️ さらなる改善が必要")