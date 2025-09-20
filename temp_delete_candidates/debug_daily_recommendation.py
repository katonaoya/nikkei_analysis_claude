#!/usr/bin/env python3
"""
daily_stock_recommendation.pyの予測結果を詳細にデバッグ
"""

import pandas as pd
import numpy as np
from datetime import datetime
import joblib
from pathlib import Path

def debug_prediction_results(target_date='2025-08-15'):
    """予測結果の詳細を調査"""
    print(f"=== {target_date}の予測結果詳細調査 ===")
    
    # モデル読み込み
    model_dir = Path("models")
    model_files = list(model_dir.glob("*model*.joblib"))
    if not model_files:
        print("❌ モデルファイルが見つかりません")
        return
    
    latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
    model = joblib.load(latest_model)
    print(f"✅ モデル読み込み: {latest_model.name}")
    
    # データ読み込み
    data_dir = Path("data")
    parquet_files = list(data_dir.glob("**/*nikkei225*.parquet"))
    if not parquet_files:
        print("❌ データファイルが見つかりません")
        return
    
    latest_file = max(parquet_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_parquet(latest_file)
    df['Date'] = pd.to_datetime(df['Date'])
    target_datetime = pd.to_datetime(target_date)
    df = df[df['Date'] <= target_datetime]
    
    print(f"✅ データ読み込み: {len(df):,}件")
    
    # 最新日のデータを取得
    latest_date = df['Date'].max()
    latest_df = df[df['Date'] == latest_date]
    
    print(f"📊 最新日付: {latest_date.date()}")
    print(f"📊 最新日データ: {len(latest_df)}銘柄")
    
    # 基本統計を確認
    print(f"\n📈 基本価格統計:")
    print(f"   Close価格範囲: {latest_df['Close'].min():.0f} ~ {latest_df['Close'].max():.0f}円")
    print(f"   平均価格: {latest_df['Close'].mean():.0f}円")
    
    # 特徴量カラムを確認
    feature_cols = [col for col in latest_df.columns 
                   if col not in ['Code', 'Date', 'CompanyName', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']]
    
    print(f"\n🔧 特徴量数: {len(feature_cols)}個")
    print("特徴量一覧:", feature_cols[:10], "..." if len(feature_cols) > 10 else "")
    
    # サンプル銘柄で予測実行
    prediction_results = []
    
    for i, (_, row) in enumerate(latest_df.head(10).iterrows()):
        try:
            code = row['Code']
            
            # 特徴量準備
            features = row[feature_cols].values.reshape(1, -1)
            
            # 欠損値チェック
            if pd.isna(features).any():
                print(f"⚠️ {code}: 欠損値あり")
                continue
            
            # 予測実行
            prediction_proba = model.predict_proba(features)[0][1]
            
            prediction_results.append({
                'code': code,
                'price': row['Close'],
                'probability': prediction_proba
            })
            
            print(f"📊 {code}: {row['Close']:.0f}円 → {prediction_proba:.3f} ({prediction_proba*100:.1f}%)")
            
        except Exception as e:
            print(f"❌ {code}: エラー - {e}")
    
    # 全体の予測確率分布を確認
    if prediction_results:
        probabilities = [r['probability'] for r in prediction_results]
        print(f"\n📊 予測確率統計 (サンプル{len(probabilities)}銘柄):")
        print(f"   最小確率: {min(probabilities):.3f} ({min(probabilities)*100:.1f}%)")
        print(f"   最大確率: {max(probabilities):.3f} ({max(probabilities)*100:.1f}%)")
        print(f"   平均確率: {np.mean(probabilities):.3f} ({np.mean(probabilities)*100:.1f}%)")
        print(f"   50%以上: {sum(1 for p in probabilities if p >= 0.50)}銘柄")
        print(f"   55%以上: {sum(1 for p in probabilities if p >= 0.55)}銘柄")
        print(f"   60%以上: {sum(1 for p in probabilities if p >= 0.60)}銘柄")
    
    print(f"\n💡 推奨対応:")
    if max(probabilities) < 0.50:
        print("   - 全銘柄が50%未満 → 閾値を45%に下げる")
    elif max(probabilities) < 0.55:
        print("   - 最大確率が55%未満 → 閾値を50%に下げる")
    else:
        print("   - システム正常動作中")

if __name__ == "__main__":
    debug_prediction_results('2025-08-15')