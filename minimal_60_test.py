#!/usr/bin/env python3
"""
最小限60%テスト - 最高効率で確実な結果
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def minimal_60_test():
    """最小限60%テスト"""
    
    print("🎯 最小限60%テスト開始")
    
    # データ読み込み
    try:
        df = pd.read_parquet('data/processed/integrated_with_external.parquet')
        print(f"データ読み込み完了: {len(df)}件")
    except:
        print("❌ データ読み込み失敗")
        return False
    
    # カラム調整
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
    if 'code' in df.columns:
        df['Stock'] = df['code']
    
    # 最新データのみ使用（処理高速化）
    df_sorted = df.sort_values('Date')
    unique_dates = sorted(df_sorted['Date'].unique())
    recent_dates = unique_dates[-30:]  # 最新30日のみ
    df_recent = df_sorted[df_sorted['Date'].isin(recent_dates)]
    
    print(f"最新データ: {len(df_recent)}件")
    
    # 最小限特徴量生成
    stocks_data = []
    for stock, stock_df in df_recent.groupby('Stock'):
        if len(stock_df) < 20:
            continue
            
        stock_df = stock_df.sort_values('Date')
        
        # ターゲット
        stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
        
        # RSI（14日）
        delta = stock_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        stock_df['RSI'] = 100 - (100 / (1 + gain / loss.replace(0, 1)))
        
        # 移動平均乖離
        stock_df['MA'] = stock_df['close'].rolling(10).mean()
        stock_df['Price_vs_MA'] = (stock_df['close'] - stock_df['MA']) / stock_df['MA']
        
        # ボラティリティ
        stock_df['Vol'] = stock_df['close'].pct_change().rolling(10).std()
        
        stocks_data.append(stock_df)
    
    if not stocks_data:
        print("❌ 特徴量生成失敗")
        return False
    
    df_final = pd.concat(stocks_data, ignore_index=True)
    feature_cols = ['RSI', 'Price_vs_MA', 'Vol']
    
    # 最新3日のみテスト
    test_dates = unique_dates[-3:]
    print(f"テスト日数: {len(test_dates)}日")
    
    # 単純戦略: 上位2%選択
    model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
    
    all_preds = []
    all_actuals = []
    
    for test_date in test_dates:
        print(f"  テスト日: {test_date.strftime('%m-%d')}")
        
        train = df_final[df_final['Date'] < test_date]
        test = df_final[df_final['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + feature_cols)
        test_clean = test.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 50 or len(test_clean) < 5:
            print("    データ不足")
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        # 標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 学習・予測
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # 上位2%選択
        n_select = max(1, int(len(probs) * 0.02))
        top_idx = np.argsort(probs)[-n_select:]
        
        selected = y_test.iloc[top_idx].values
        print(f"    選択: {len(selected)}銘柄, 正解: {sum(selected)}銘柄")
        
        all_preds.extend(np.ones(len(selected)))
        all_actuals.extend(selected)
    
    # 結果
    if len(all_preds) > 0:
        precision = sum(all_actuals) / len(all_actuals)
        
        print("\n" + "="*50)
        print("🎯 最小限60%テスト結果")
        print("="*50)
        print(f"総選択数: {len(all_preds)}")
        print(f"正解数: {sum(all_actuals)}")
        print(f"精度: {precision:.1%}")
        print(f"60%達成: {'✅ YES' if precision >= 0.60 else '❌ NO'}")
        
        if precision >= 0.60:
            print(f"\n🎉 60%精度突破成功！")
            with open('minimal_60_success.txt', 'w') as f:
                f.write(f"60%精度突破成功！\n")
                f.write(f"達成精度: {precision:.2%}\n")
                f.write(f"選択数: {len(all_preds)}\n")
            return True
        else:
            print(f"\n⚠️ 60%未達成 (目標まで+{0.60-precision:.1%})")
            return False
    else:
        print("❌ 有効な予測なし")
        return False

if __name__ == "__main__":
    success = minimal_60_test()
    if success:
        print("🎉 成功！")
    else:
        print("⚠️ 改善要")