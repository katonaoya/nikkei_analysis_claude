#!/usr/bin/env python3
"""
57.93%ベストプラクティス再現
ドキュメント記載の成功手法を正確に再現
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def replicate_best_practice():
    """57.93%ベストプラクティス再現"""
    
    print("🎯 57.93%ベストプラクティス再現開始")
    print("📋 ドキュメント記載の成功手法を正確に再現")
    
    # データ読み込み
    df = pd.read_parquet('data/processed/integrated_with_external.parquet')
    print(f"データ読み込み: {len(df)}件")
    
    # カラム調整
    if 'date' in df.columns:
        df['Date'] = pd.to_datetime(df['date'])
    if 'code' in df.columns:
        df['Stock'] = df['code']
    
    print("🔧 ベストプラクティス特徴量生成...")
    print("使用特徴量: RSI, Price_vs_MA5, Price_vs_MA20, Volatility, Volume_Ratio, Momentum_5, Momentum_20, Price_Position")
    
    # ベストプラクティス特徴量生成（8個）
    features = []
    for stock, stock_df in df.groupby('Stock'):
        if len(stock_df) < 30:  # 十分なデータがある銘柄のみ
            continue
            
        stock_df = stock_df.sort_values('Date')
        
        # ターゲット（翌日の終値が当日より高い）
        stock_df['Target'] = (stock_df['close'].shift(-1) > stock_df['close']).astype(int)
        
        # 1. RSI (14日)
        delta = stock_df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        stock_df['RSI'] = 100 - (100 / (1 + rs))
        
        # 2. Price_vs_MA5 (5日移動平均乖離率)
        stock_df['MA5'] = stock_df['close'].rolling(5).mean()
        stock_df['Price_vs_MA5'] = (stock_df['close'] - stock_df['MA5']) / stock_df['MA5']
        
        # 3. Price_vs_MA20 (20日移動平均乖離率)
        stock_df['MA20'] = stock_df['close'].rolling(20).mean()
        stock_df['Price_vs_MA20'] = (stock_df['close'] - stock_df['MA20']) / stock_df['MA20']
        
        # 4. Volatility (20日ボラティリティ)
        stock_df['Return'] = stock_df['close'].pct_change()
        stock_df['Volatility'] = stock_df['Return'].rolling(20).std()
        
        # 5. Volume_Ratio (出来高比率)
        stock_df['Volume_MA20'] = stock_df['volume'].rolling(20).mean()
        stock_df['Volume_Ratio'] = stock_df['volume'] / stock_df['Volume_MA20'].replace(0, 1)
        
        # 6. Momentum_5 (5日モメンタム)
        stock_df['Momentum_5'] = stock_df['close'].pct_change(5)
        
        # 7. Momentum_20 (20日モメンタム)
        stock_df['Momentum_20'] = stock_df['close'].pct_change(20)
        
        # 8. Price_Position (価格帯での位置)
        stock_df['High_20'] = stock_df['high'].rolling(20).max()
        stock_df['Low_20'] = stock_df['low'].rolling(20).min()
        stock_df['Price_Position'] = (stock_df['close'] - stock_df['Low_20']) / (stock_df['High_20'] - stock_df['Low_20'])
        
        features.append(stock_df)
    
    df = pd.concat(features, ignore_index=True)
    
    # ベストプラクティス特徴量（8個）
    feature_cols = [
        'RSI',
        'Price_vs_MA5', 
        'Price_vs_MA20',
        'Volatility',
        'Volume_Ratio',
        'Momentum_5',
        'Momentum_20',
        'Price_Position'
    ]
    
    print(f"特徴量生成完了: {len(feature_cols)}個")
    
    # ベストプラクティスモデル設定
    model = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1
    )
    
    print("🚀 ベストプラクティス戦略実行: 上位5銘柄選択")
    
    # 時系列バックテスト（30日間）
    df_sorted = df.sort_values('Date')
    unique_dates = sorted(df_sorted['Date'].unique())
    test_dates = unique_dates[-30:]  # 直近30日間
    
    print(f"テスト期間: {len(test_dates)}日間")
    
    all_predictions = []
    all_actuals = []
    daily_results = []
    
    for i, test_date in enumerate(test_dates):
        if i % 10 == 0:
            print(f"  進捗: {i+1}/{len(test_dates)} ({(i+1)/len(test_dates)*100:.0f}%)")
        
        # 1. 当日より前のデータで学習
        train_data = df_sorted[df_sorted['Date'] < test_date]
        test_data = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train_data.dropna(subset=['Target'] + feature_cols)
        test_clean = test_data.dropna(subset=['Target'] + feature_cols)
        
        if len(train_clean) < 1000 or len(test_clean) < 5:
            continue
        
        X_train = train_clean[feature_cols]
        y_train = train_clean['Target']
        X_test = test_clean[feature_cols]
        y_test = test_clean['Target']
        
        # 2. 特徴量標準化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # 3. モデル学習
        model.fit(X_train_scaled, y_train)
        
        # 4. 予測確率取得
        pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        # 5. 上位5銘柄選択（ベストプラクティス）
        n_select = min(5, len(pred_proba))
        if n_select > 0:
            top_indices = np.argsort(pred_proba)[-n_select:]
            selected_actuals = y_test.iloc[top_indices].values
            selected_probs = pred_proba[top_indices]
            
            # 結果記録
            all_predictions.extend(np.ones(len(selected_actuals)))
            all_actuals.extend(selected_actuals)
            
            # 日別結果
            daily_precision = sum(selected_actuals) / len(selected_actuals) if len(selected_actuals) > 0 else 0
            daily_results.append({
                'date': test_date.strftime('%Y-%m-%d'),
                'selected': len(selected_actuals),
                'correct': sum(selected_actuals),
                'precision': daily_precision,
                'avg_prob': np.mean(selected_probs)
            })
    
    # 最終結果計算
    if len(all_predictions) > 0:
        precision = sum([a for a, p in zip(all_actuals, all_predictions) if a == 1 and p == 1]) / len(all_predictions)
        
        print("\n" + "="*80)
        print("🎯 ベストプラクティス再現結果")
        print("="*80)
        print(f"📊 戦略名: LightGBM + 上位5銘柄選択")
        print(f"📈 1日平均選択数: {len(all_predictions)/len(test_dates):.1f}銘柄")
        print(f"🎯 テスト期間: 直近30日間")
        print(f"📏 総選択銘柄数: {len(all_predictions)}")
        print(f"✅ 正解数: {sum(all_actuals)}")
        print(f"🎖️ **精度: {precision:.2%}**")
        
        # ベストプラクティス比較
        target_precision = 0.5793  # 57.93%
        if precision >= target_precision:
            print(f"🎉 ベストプラクティス達成！ ({target_precision:.2%}以上)")
        else:
            print(f"⚠️ ベストプラクティス未達 (目標: {target_precision:.2%})")
        
        # 60%達成確認
        if precision >= 0.60:
            print(f"🚀 【60%精度突破成功！】")
            print(f"✅ 目標クリア: {precision:.2%} ≥ 60.00%")
            
            # 成功記録
            with open('best_practice_60_success.txt', 'w') as f:
                f.write(f"60%精度突破成功！\n")
                f.write(f"達成精度: {precision:.2%}\n")
                f.write(f"戦略: ベストプラクティス再現\n")
                f.write(f"選択数: {len(all_predictions)}\n")
                f.write(f"達成時刻: {datetime.now()}\n")
                f.write(f"使用特徴量: {', '.join(feature_cols)}\n")
            
            print("💾 成功記録保存完了")
            return True
        else:
            print(f"⚠️ 60%未達成 (目標まで: +{0.60 - precision:.2%})")
        
        # 詳細分析
        print(f"\n📊 詳細分析:")
        successful_days = len([r for r in daily_results if r['precision'] > 0.5])
        print(f"成功日数: {successful_days}/{len(daily_results)} ({successful_days/len(daily_results)*100:.1f}%)")
        avg_daily_precision = np.mean([r['precision'] for r in daily_results])
        print(f"日別平均精度: {avg_daily_precision:.2%}")
        
        return precision >= 0.60
        
    else:
        print("❌ テスト失敗: 有効な予測なし")
        return False

if __name__ == "__main__":
    success = replicate_best_practice()
    if success:
        print("\n🎉 60%精度達成成功！")
    else:
        print("\n⚠️ さらなる改善が必要")