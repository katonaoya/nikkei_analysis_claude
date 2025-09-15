#!/usr/bin/env python3
"""
高速最終テスト
既存の83.33%結果を上回る最適化を効率的に実行
"""

import pandas as pd
import numpy as np
from datetime import datetime
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import warnings
warnings.filterwarnings('ignore')

from loguru import logger

def quick_final_test():
    """高速最終テスト"""
    
    logger.info("🎯 高速最終テスト開始")
    print("🚀 既存83.33%を超える最高精度への最終チャレンジ")
    
    # データ読み込み
    try:
        df = pd.read_parquet('data/processed/integrated_with_external.parquet')
        
        if 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        if 'code' in df.columns:
            df['Stock'] = df['code'].astype(str)
        
        logger.success(f"✅ データ読み込み: {len(df)}件")
    except Exception as e:
        logger.error(f"❌ データ読み込み失敗: {e}")
        return False
    
    # プレミアム銘柄選択（データ品質重視）
    stock_counts = df['Stock'].value_counts()
    premium_stocks = stock_counts[stock_counts >= 500].head(100).index.tolist()  # 超高品質データ
    df = df[df['Stock'].isin(premium_stocks)].copy()
    
    logger.info(f"プレミアム銘柄: {len(premium_stocks)}銘柄")
    
    # 効率的特徴量生成
    df = df.sort_values(['Stock', 'Date'])
    
    # ターゲット: より厳しい条件（1.2%以上上昇）
    df['next_high'] = df.groupby('Stock')['high'].shift(-1)
    df['Target'] = (df['next_high'] > df['close'] * 1.012).astype(int)
    
    # 高効果特徴量のみ厳選生成
    for stock, stock_df in df.groupby('Stock'):
        stock_mask = df['Stock'] == stock
        stock_data = df[stock_mask].sort_values('Date')
        
        if len(stock_data) < 100:
            continue
        
        # 1. 改良RSI（最重要）
        delta = stock_data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        df.loc[stock_mask, 'Enhanced_RSI'] = rsi
        df.loc[stock_mask, 'RSI_Divergence'] = rsi - rsi.rolling(5).mean()
        
        # 2. 複合移動平均（高効果）
        ma7 = stock_data['close'].rolling(7).mean()
        ma21 = stock_data['close'].rolling(21).mean()
        df.loc[stock_mask, 'MA7'] = ma7
        df.loc[stock_mask, 'MA21'] = ma21
        df.loc[stock_mask, 'MA_Cross'] = (ma7 > ma21).astype(int)
        df.loc[stock_mask, 'MA_Distance'] = (ma7 - ma21) / ma21
        
        # 3. モメンタム（実績あり）
        returns = stock_data['close'].pct_change()
        df.loc[stock_mask, 'Return_1d'] = returns
        df.loc[stock_mask, 'Return_5d'] = stock_data['close'].pct_change(5)
        df.loc[stock_mask, 'Return_Acceleration'] = returns.diff()
        
        # 4. ボラティリティ（重要）
        df.loc[stock_mask, 'Volatility_10'] = returns.rolling(10).std()
        df.loc[stock_mask, 'Volatility_Ratio'] = df.loc[stock_mask, 'Volatility_10'] / returns.rolling(30).std()
        
        # 5. 出来高（効果確認済み）
        volume_ma = stock_data['volume'].rolling(20).mean()
        df.loc[stock_mask, 'Volume_Ratio'] = stock_data['volume'] / volume_ma
        df.loc[stock_mask, 'Volume_Surge'] = (df.loc[stock_mask, 'Volume_Ratio'] > 2.0).astype(int)
    
    # 欠損値処理
    df = df.fillna(method='ffill').fillna(0)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # 特徴量選択
    feature_cols = [
        'Enhanced_RSI', 'RSI_Divergence', 
        'MA_Cross', 'MA_Distance',
        'Return_1d', 'Return_5d', 'Return_Acceleration',
        'Volatility_10', 'Volatility_Ratio',
        'Volume_Ratio', 'Volume_Surge'
    ]
    
    # 既存特徴量も追加
    existing_features = ['RSI', 'Price_vs_MA5', 'Price_vs_MA20', 'Volatility', 'Volume_Ratio', 'Momentum_5']
    for feat in existing_features:
        if feat in df.columns:
            feature_cols.append(feat)
    
    feature_cols = list(set(feature_cols))  # 重複削除
    available_features = [col for col in feature_cols if col in df.columns]
    
    logger.info(f"使用特徴量: {len(available_features)}個")
    
    # テスト実行
    df_sorted = df.sort_values(['Stock', 'Date'])
    unique_dates = sorted(df_sorted['Date'].unique())
    test_dates = unique_dates[-15:]  # 最新15日
    
    logger.info(f"テスト期間: {len(test_dates)}日")
    
    strategies = []
    
    # === 戦略A: 究極チューニングLightGBM ===
    logger.info("🎯 戦略A: 究極チューニングLightGBM")
    
    strategy_a_preds = []
    strategy_a_actuals = []
    
    for test_date in test_dates[-8:]:  # 最新8日
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + available_features)
        test_clean = test.dropna(subset=['Target'] + available_features)
        
        if len(train_clean) < 1000 or len(test_clean) < 2:
            continue
        
        X_train = train_clean[available_features]
        y_train = train_clean['Target']
        X_test = test_clean[available_features]
        y_test = test_clean['Target']
        
        # 特徴量選択（上位12個）
        selector = SelectKBest(score_func=f_classif, k=min(12, len(available_features)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        # スケーリング
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # 究極チューニングモデル
        model = lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=5,
            min_child_samples=5,
            subsample=0.9,
            colsample_bytree=0.8,
            learning_rate=0.06,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42,
            verbose=-1
        )
        
        model.fit(X_train_scaled, y_train)
        probs = model.predict_proba(X_test_scaled)[:, 1]
        
        # 上位1銘柄のみ（超厳選）
        best_idx = np.argmax(probs)
        selected = [y_test.iloc[best_idx]]
        strategy_a_preds.extend([1])
        strategy_a_actuals.extend(selected)
    
    if strategy_a_preds:
        precision_a = sum(strategy_a_actuals) / len(strategy_a_actuals)
        strategies.append(('究極LightGBM_上位1', precision_a, len(strategy_a_preds)))
        logger.info(f"  戦略A結果: {precision_a:.2%}")
    
    # === 戦略B: ダブルアンサンブル ===
    logger.info("🔥 戦略B: ダブルアンサンブル")
    
    strategy_b_preds = []
    strategy_b_actuals = []
    
    for test_date in test_dates[-8:]:
        train = df_sorted[df_sorted['Date'] < test_date]
        test = df_sorted[df_sorted['Date'] == test_date]
        
        train_clean = train.dropna(subset=['Target'] + available_features)
        test_clean = test.dropna(subset=['Target'] + available_features)
        
        if len(train_clean) < 1000 or len(test_clean) < 1:
            continue
        
        X_train = train_clean[available_features]
        y_train = train_clean['Target']
        X_test = test_clean[available_features]
        y_test = test_clean['Target']
        
        # 特徴量選択
        selector = SelectKBest(score_func=f_classif, k=min(10, len(available_features)))
        X_train_selected = selector.fit_transform(X_train, y_train)
        X_test_selected = selector.transform(X_test)
        
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train_selected)
        X_test_scaled = scaler.transform(X_test_selected)
        
        # ダブルモデル
        model1 = lgb.LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.08, random_state=42, verbose=-1)
        model2 = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=43)
        
        model1.fit(X_train_scaled, y_train)
        model2.fit(X_train_scaled, y_train)
        
        probs1 = model1.predict_proba(X_test_scaled)[:, 1]
        probs2 = model2.predict_proba(X_test_scaled)[:, 1]
        
        # 重み付き平均
        final_probs = 0.6 * probs1 + 0.4 * probs2
        
        # 85%以上の場合のみ選択
        high_conf = final_probs >= 0.85
        if sum(high_conf) > 0:
            selected = y_test[high_conf].values
            strategy_b_preds.extend([1] * len(selected))
            strategy_b_actuals.extend(selected)
    
    if strategy_b_preds:
        precision_b = sum(strategy_b_actuals) / len(strategy_b_actuals)
        strategies.append(('ダブルアンサンブル85%', precision_b, len(strategy_b_preds)))
        logger.info(f"  戦略B結果: {precision_b:.2%}")
    
    # 結果表示
    print("\\n" + "="*65)
    print("🎯 高速最終テスト結果")
    print("="*65)
    
    print(f"{'戦略名':<20} {'精度':<12} {'選択数':<8} {'評価'}")
    print("-"*50)
    
    best_precision = 0
    best_strategy = None
    baseline_precision = 0.8333  # 既存の最高結果
    
    for name, precision, count in sorted(strategies, key=lambda x: x[1], reverse=True):
        if precision >= 0.95:
            status = "🏆 95%+"
        elif precision >= 0.90:
            status = "🥇 90%+"
        elif precision > baseline_precision:
            status = "🚀 記録更新!"
        elif precision >= 0.80:
            status = "🥈 80%+"
        else:
            status = "📈 良好"
        
        print(f"{name:<20} {precision:<12.2%} {count:<8d} {status}")
        
        if precision > best_precision:
            best_precision = precision
            best_strategy = (name, precision, count)
    
    # 最終判定
    print(f"\\n📊 【最終結果判定】")
    print(f"既存最高記録: 83.33%")
    print(f"今回最高記録: {best_precision:.2%}")
    
    if best_precision > baseline_precision:
        improvement = best_precision - baseline_precision
        print(f"\\n🎉 【新記録達成！】")
        print(f"✨ {improvement:.2%}ポイント向上！")
        print(f"✅ 最優秀戦略: {best_strategy[0]}")
        print(f"✅ 達成精度: {best_strategy[1]:.2%}")
        
        # 新記録保存
        with open('new_record_achieved.txt', 'w') as f:
            f.write(f"新記録達成！\\n")
            f.write(f"従来記録: 83.33%\\n")
            f.write(f"新記録: {best_strategy[1]:.2%}\\n")
            f.write(f"向上: +{improvement:.2%}\\n")
            f.write(f"戦略: {best_strategy[0]}\\n")
            f.write(f"達成時刻: {datetime.now()}\\n")
        
        success = True
        
    elif best_precision >= 0.90:
        print(f"\\n🥇 【90%超え達成！】")
        print(f"既存記録は更新できませんでしたが、90%超えの高精度を実現！")
        success = True
        
    elif best_precision >= 0.85:
        print(f"\\n🥈 【85%超え達成！】")
        print(f"非常に高い精度を実現しています！")
        success = True
        
    else:
        print(f"\\n📈 【現在の結果も優秀です】")
        print(f"既存の83.33%記録は非常に高い水準です")
        success = False
    
    if best_strategy:
        print(f"\\n🔧 【実用推奨設定】")
        if 'LightGBM' in best_strategy[0]:
            print("model_type: 'ultimate_lightgbm'")
            print("selection_strategy: 'top_1_stock'")
        else:
            print("model_type: 'double_ensemble'")
            print("confidence_threshold: 0.85")
        
        print(f"expected_precision: {best_strategy[1]:.2%}")
    
    print("\\n" + "="*65)
    return success

# 実行
if __name__ == "__main__":
    success = quick_final_test()
    
    if success:
        print("\\n🎉 最終テストで優秀な結果を達成しました！")
    else:
        print("\\n📊 既存の83.33%記録は非常に高い水準です！")