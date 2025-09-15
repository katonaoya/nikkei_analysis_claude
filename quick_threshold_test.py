#!/usr/bin/env python3
"""
信頼度閾値別精度テスト（高速版）
既存のtest_ai_accuracy.pyを改良して5つの閾値で比較
"""

import pandas as pd
import numpy as np
import yaml
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score
from loguru import logger
import warnings
warnings.filterwarnings('ignore')

def quick_threshold_comparison():
    """高速閾値比較テスト"""
    
    # 設定読み込み
    config_path = Path("production_config.yaml")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    optimal_features = config['features']['optimal_features']
    
    # データ読み込み
    logger.info("📥 データ読み込み...")
    data_dir = Path(config['data']['processed_dir'])
    integrated_file = data_dir / config['data']['integrated_file']
    df = pd.read_parquet(integrated_file)
    
    # カラム調整
    if 'Target' not in df.columns and 'Binary_Direction' in df.columns:
        df['Target'] = df['Binary_Direction']
    if 'Stock' not in df.columns and 'Code' in df.columns:
        df['Stock'] = df['Code']
    
    logger.info(f"データ件数: {len(df):,}件")
    
    # テスト期間の設定（直近15日でテスト）
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    unique_dates = sorted(df['Date'].unique())
    test_dates = unique_dates[-15:]  # 高速化のため15日に短縮
    
    # テスト対象の閾値
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70]
    
    logger.info("🎯 閾値別テスト開始...")
    
    results = {}
    
    for threshold in thresholds:
        logger.info(f"  閾値 {threshold:.0%} テスト中...")
        
        all_predictions = []
        all_actuals = []
        daily_counts = []
        
        for test_date in test_dates:
            # データ分割
            train_data = df[df['Date'] < test_date]
            test_data = df[df['Date'] == test_date]
            
            if len(train_data) < 1000 or len(test_data) < 10:
                continue
            
            # クリーンデータ
            train_clean = train_data[['Date', 'Stock', 'Target'] + optimal_features].dropna()
            test_clean = test_data[['Date', 'Stock', 'Target'] + optimal_features].dropna()
            
            if len(train_clean) == 0 or len(test_clean) == 0:
                continue
            
            X_train = train_clean[optimal_features]
            y_train = train_clean['Target']
            X_test = test_clean[optimal_features]
            y_test = test_clean['Target']
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 高速なLogisticRegressionを使用
            model = LogisticRegression(random_state=42, max_iter=500)
            model.fit(X_train_scaled, y_train)
            
            # 予測確率
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
            
            # 閾値でフィルタリング
            high_conf_indices = y_pred_proba >= threshold
            
            if sum(high_conf_indices) > 0:
                # 閾値以上の銘柄は全て「上昇」予測
                selected_predictions = np.ones(sum(high_conf_indices))
                selected_actuals = y_test[high_conf_indices]
                
                all_predictions.extend(selected_predictions)
                all_actuals.extend(selected_actuals)
                daily_counts.append(sum(high_conf_indices))
        
        # 結果計算
        if len(all_predictions) > 0:
            precision = sum([a for a, p in zip(all_actuals, all_predictions) if a == 1 and p == 1]) / len(all_predictions)
            avg_daily = np.mean(daily_counts) if daily_counts else 0
            
            results[threshold] = {
                'precision': precision,
                'total_selected': len(all_predictions),
                'total_correct': sum(all_actuals),
                'avg_daily_picks': avg_daily,
                'test_days': len([d for d in daily_counts if d > 0])
            }
        else:
            results[threshold] = {
                'precision': 0,
                'total_selected': 0,
                'total_correct': 0,
                'avg_daily_picks': 0,
                'test_days': 0
            }
    
    # 結果表示
    print("\n" + "="*80)
    print("📊 信頼度閾値別精度比較テスト結果（直近15日間）")
    print("="*80)
    
    print(f"\n{'閾値':<8} {'精度':<10} {'総選択':<8} {'的中数':<8} {'1日平均':<10} {'取引日数':<8}")
    print("-"*65)
    
    for threshold in thresholds:
        r = results[threshold]
        print(f"{threshold:.0%}      {r['precision']:<10.2%} "
              f"{r['total_selected']:<8d} {r['total_correct']:<8d} "
              f"{r['avg_daily_picks']:<10.1f} {r['test_days']:<8d}")
    
    print("\n📈 【詳細分析】")
    
    for threshold in thresholds:
        r = results[threshold]
        if r['total_selected'] > 0:
            print(f"\n🎯 閾値 {threshold:.0%}:")
            print(f"  • 精度: {r['precision']:.2%}")
            print(f"  • 総選択数: {r['total_selected']}銘柄")
            print(f"  • 的中数: {r['total_correct']}銘柄")
            print(f"  • 1日平均選択数: {r['avg_daily_picks']:.1f}銘柄")
            print(f"  • 取引が発生した日数: {r['test_days']}/15日")
            
            if r['test_days'] > 0:
                frequency = r['test_days'] / 15 * 100
                print(f"  • 取引頻度: {frequency:.1f}%")
        else:
            print(f"\n🎯 閾値 {threshold:.0%}: 選択された銘柄なし")
    
    # 推奨事項
    print(f"\n💡 【推奨事項】")
    
    # 60%以上の精度を達成した閾値
    good_results = [(t, r) for t, r in results.items() 
                   if r['precision'] >= 0.60 and r['total_selected'] > 5]
    
    if good_results:
        best_threshold, best_result = max(good_results, key=lambda x: x[1]['precision'])
        print(f"✅ 60%以上達成: 閾値 {best_threshold:.0%}")
        print(f"   → 精度: {best_result['precision']:.2%}")
        print(f"   → 1日平均: {best_result['avg_daily_picks']:.1f}銘柄")
        print(f"   → この設定を推奨します")
        
        # 設定更新の提案
        if abs(best_threshold - config['system']['confidence_threshold']) > 0.01:
            print(f"\n🔧 設定更新提案:")
            print(f"   現在: {config['system']['confidence_threshold']:.0%}")
            print(f"   推奨: {best_threshold:.0%}")
    else:
        # 最も精度の高い閾値を推奨
        best_threshold = max(thresholds, key=lambda t: results[t]['precision'])
        best_result = results[best_threshold]
        print(f"📍 現状最良: 閾値 {best_threshold:.0%}")
        print(f"   → 精度: {best_result['precision']:.2%}")
        print(f"   → 1日平均: {best_result['avg_daily_picks']:.1f}銘柄")
        
        if best_result['precision'] < 0.60:
            print(f"   → 60%目標まで: +{0.60 - best_result['precision']:.2%}")
    
    print(f"\n📋 【運用ガイド】")
    print("• 高精度重視: 65-70%閾値（週1-2回の厳選取引）")
    print("• バランス型: 55-60%閾値（週2-3回の安定取引）")
    print("• 頻度重視: 50-55%閾値（ほぼ毎日だが精度は控えめ）")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    quick_threshold_comparison()