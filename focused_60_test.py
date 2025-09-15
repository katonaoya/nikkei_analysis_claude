#!/usr/bin/env python3
"""
効率的な60%精度テスト - タイムアウト回避版
最小限のテストで確実に結果を出す
"""

import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def focused_60_test():
    """効率的な60%精度テスト"""
    
    print("🎯 効率的60%精度テスト開始")
    
    # データ読み込み
    df = pd.read_parquet('data/processed/integrated_with_external.parquet')
    
    # カラム調整
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
    if 'code' in df.columns:
        df['Stock'] = df['code']
    
    # 高速特徴量生成（最小限）
    print("🔧 特徴量生成...")
    
    # 株式別に特徴量計算
    features = []
    for stock, stock_df in df.groupby('Stock'):
        stock_df = stock_df.sort_values('Date')
        
        # ターゲット
        stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
        
        # RSI（14日）
        delta = stock_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        stock_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 移動平均乖離率
        stock_df['MA20'] = stock_df['close'].rolling(20).mean()
        stock_df['Price_vs_MA20'] = (stock_df['close'] - stock_df['MA20']) / stock_df['MA20']
        
        # ボラティリティ
        stock_df['Return'] = stock_df['close'].pct_change()
        stock_df['Volatility'] = stock_df['Return'].rolling(20).std()
        
        # 出来高比率
        stock_df['Volume_MA'] = stock_df['volume'].rolling(20).mean()
        stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA'].replace(0, 1)
        
        features.append(stock_df)
    
    df = pd.concat(features, ignore_index=True)
    feature_cols = ['RSI', 'Price_vs_MA20', 'Volatility', 'Volume_Ratio']
    
    # テスト期間（最新5日のみ）
    df_sorted = df.sort_values('Date')
    unique_dates = sorted(df_sorted['Date'].unique())
    test_dates = unique_dates[-5:]  # 最新5日のみで高速テスト
    
    print(f"テスト期間: {len(test_dates)}日")
    
    # 戦略: 上位3%選択（超厳選）
    print("🚀 超厳選戦略実行...")
    
    model = lgb.LGBMClassifier(
        n_estimators=50,  # 高速化
        max_depth=3,
        random_state=42,
        verbose=-1
    )
    
    all_predictions = []
    all_actuals = []
    daily_results = []
    
    for i, test_date in enumerate(test_dates):
        print(f"  日付 {i+1}/{len(test_dates)}: {test_date.strftime('%Y-%m-%d')}")
        
        train_data = df_sorted[df_sorted['Date'] < test_date]
        test_data = df_sorted[df_sorted['Date'] == test_date]
        
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
        
        # 学習・予測
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # 上位3%選択（超厳選）
        n_select = max(1, int(len(probs) * 0.03))
        top_indices = np.argsort(probs)[-n_select:]
        
        selected_actuals = y_test.iloc[top_indices].values
        selected_probs = probs[top_indices]
        
        # 結果記録
        all_predictions.extend(np.ones(len(selected_actuals)))
        all_actuals.extend(selected_actuals)
        
        # 日別結果
        if len(selected_actuals) > 0:
            daily_precision = sum(selected_actuals) / len(selected_actuals)
            daily_results.append({
                'date': test_date.strftime('%Y-%m-%d'),
                'selected': len(selected_actuals),
                'correct': sum(selected_actuals),
                'precision': daily_precision,
                'avg_confidence': np.mean(selected_probs)
            })
            print(f"    選択数: {len(selected_actuals)}, 正解数: {sum(selected_actuals)}, 精度: {daily_precision:.1%}")
    
    # 最終結果
    if len(all_predictions) > 0:
        overall_precision = sum([a for a, p in zip(all_actuals, all_predictions) if a == 1 and p == 1]) / len(all_predictions)
        
        print("\n" + "="*60)
        print("🎯 効率的60%精度テスト結果")
        print("="*60)
        print(f"総選択銘柄数: {len(all_predictions)}")
        print(f"正解数: {sum(all_actuals)}")
        print(f"全体精度: {overall_precision:.2%}")
        print(f"60%達成: {'✅ YES' if overall_precision >= 0.60 else '❌ NO'}")
        
        if overall_precision >= 0.60:
            print(f"\n🎉 60%精度突破成功！")
            print(f"達成精度: {overall_precision:.2%}")
            
            # 成功記録
            with open('focused_60_success.txt', 'w') as f:
                f.write(f"60%精度突破成功！\n")
                f.write(f"達成精度: {overall_precision:.2%}\n")
                f.write(f"戦略: 上位3%超厳選\n")
                f.write(f"選択数: {len(all_predictions)}\n")
                f.write(f"達成時刻: {datetime.now()}\n")
            
            print("💾 成功記録保存完了")
            return True
        else:
            print(f"\n⚠️ 60%未達成")
            print(f"目標まで: +{0.60 - overall_precision:.2%}")
            return False
    else:
        print("❌ テスト失敗: 有効な予測なし")
        return False

if __name__ == "__main__":
    success = focused_60_test()
    if success:
        print("\n🎉 60%精度達成成功！")
    else:
        print("\n⚠️ さらなる改善が必要")