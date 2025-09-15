"""
100%精度の現実性チェック
実際のモデル予測を詳細分析して問題点を特定
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def realistic_precision_test():
    """現実的な精度テストと問題点特定"""
    logger.info("=== 現実的精度テスト開始 ===")
    
    # データ読み込み
    data_file = Path("data/nikkei225_full_data/nikkei225_full_10years_175stocks_20250831_020101.pkl")
    df = pd.read_pickle(data_file)
    
    # 基本前処理（前回と同じ）
    df = df.sort_values(['Code', 'Date']).reset_index(drop=True)
    df['close_price'] = pd.to_numeric(df['Close'], errors='coerce')
    df['daily_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None)
    df['next_day_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None).shift(-1)
    
    # ターゲット作成（2%以上上昇）
    df['target'] = (df['next_day_return'] >= 0.02).astype(int)
    
    print(f"データサイズ: {len(df):,}")
    print(f"ターゲット分布: {df['target'].mean():.1%} ({df['target'].sum():,}件)")
    
    # シンプルな特徴量のみ作成（処理速度向上のため）
    windows = [5, 10, 20]
    
    for window in windows:
        # 移動平均比率
        sma = df.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
        df[f'price_to_sma_{window}'] = df['close_price'] / sma
        
        # ボラティリティ
        df[f'volatility_{window}'] = df.groupby('Code')['daily_return'].transform(
            lambda x: x.rolling(window).std()
        )
    
    # RSI
    def calc_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = df.groupby('Code')['close_price'].transform(calc_rsi)
    
    # ラグ特徴量
    for lag in range(1, 4):
        df[f'return_lag_{lag}'] = df.groupby('Code')['daily_return'].shift(lag)
    
    # 特徴量準備
    feature_cols = [col for col in df.columns if col.startswith(('price_to_sma', 'volatility', 'rsi', 'return_lag'))]
    
    print(f"使用特徴量数: {len(feature_cols)}")
    
    X = df[feature_cols].fillna(0)
    y = df['target']
    dates = df['Date']
    
    # NaN除去
    valid_mask = ~(y.isna() | X.isna().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    dates = dates[valid_mask]
    
    print(f"有効データ: {len(X):,}")
    
    # 時系列分割で現実的テスト
    tscv = TimeSeriesSplit(n_splits=3, gap=5)  # 3フォールドで高速化
    results = []
    
    scaler = RobustScaler()
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"Fold {fold + 1}/3 実行中...")
        
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # スケーリング
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # シンプルなLightGBMで検証
        model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=-1
        )
        
        model.fit(X_train_scaled, y_train)
        
        # 予測確率
        proba = model.predict_proba(X_val_scaled)[:, 1]
        
        # 複数の閾値で評価
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
        
        fold_results = []
        
        for threshold in thresholds:
            predictions = (proba >= threshold).astype(int)
            
            if predictions.sum() > 0:
                precision = precision_score(y_val, predictions)
                recall = recall_score(y_val, predictions)
                
                # 混同行列
                tn, fp, fn, tp = confusion_matrix(y_val, predictions).ravel()
                
                fold_results.append({
                    'fold': fold + 1,
                    'threshold': threshold,
                    'precision': precision,
                    'recall': recall,
                    'predictions': predictions.sum(),
                    'true_positives': tp,
                    'false_positives': fp,
                    'true_negatives': tn,
                    'false_negatives': fn,
                    'total_positive': y_val.sum(),
                    'total_samples': len(y_val)
                })
                
                print(f"  Fold {fold+1}, 閾値{threshold:.2f}: "
                      f"精度={precision:.3f}, 再現率={recall:.3f}, "
                      f"予測数={predictions.sum()}, TP={tp}, FP={fp}")
            else:
                print(f"  Fold {fold+1}, 閾値{threshold:.2f}: 予測なし")
        
        results.extend(fold_results)
    
    # 結果分析
    print(f"\n=== 詳細結果分析 ===")
    
    # 閾値別統計
    df_results = pd.DataFrame(results)
    
    if len(df_results) > 0:
        summary = df_results.groupby('threshold').agg({
            'precision': ['mean', 'std', 'min', 'max'],
            'recall': ['mean', 'std'],
            'predictions': ['mean', 'sum'],
            'true_positives': 'sum',
            'false_positives': 'sum'
        }).round(4)
        
        print("閾値別統計:")
        print(summary)
        
        # 100%精度の閾値を特定
        perfect_precision = df_results[df_results['precision'] == 1.0]
        
        if len(perfect_precision) > 0:
            print(f"\n=== 100%精度の詳細 ===")
            print(f"100%精度を達成した閾値: {perfect_precision['threshold'].unique()}")
            print(f"該当フォールド数: {len(perfect_precision)}")
            print(f"平均予測数: {perfect_precision['predictions'].mean():.1f}")
            print(f"最大予測数: {perfect_precision['predictions'].max()}")
            print(f"最小予測数: {perfect_precision['predictions'].min()}")
            
            print(f"\n100%精度時の詳細:")
            for _, row in perfect_precision.iterrows():
                print(f"  Fold {row['fold']}, 閾値{row['threshold']:.2f}: "
                      f"予測{row['predictions']}件, TP={row['true_positives']}, FP={row['false_positives']}")
                      
            # 問題点分析
            avg_predictions = perfect_precision['predictions'].mean()
            total_positives = df_results['total_positive'].iloc[0] if len(df_results) > 0 else 0
            
            print(f"\n=== 問題点分析 ===")
            print(f"1. 極少予測数: 平均{avg_predictions:.1f}件の予測（全体の{avg_predictions/total_positives*100:.2f}%）")
            print(f"2. 高い閾値: 0.7以上の極端に高い閾値を使用")
            print(f"3. 実用性の問題: 日々{avg_predictions:.0f}件程度の投資機会のみ")
            
            recall_at_100_precision = perfect_precision['recall'].mean()
            print(f"4. 再現率犠牲: 100%精度時の平均再現率={recall_at_100_precision:.1%}")
            
        else:
            print(f"100%精度は達成されませんでした（より現実的な結果）")
    
    return results


def analyze_overfitting_risk():
    """オーバーフィッティングリスク分析"""
    print(f"\n=== オーバーフィッティングリスク分析 ===")
    
    print("100%精度が示す潜在的な問題:")
    print("1. データリーケージ: 未来の情報が特徴量に混入している可能性")
    print("2. 極端な閾値設定: 超保守的な予測により見かけ上の高精度")
    print("3. 時系列分割の問題: 訓練期間と検証期間の重複")
    print("4. 特徴量の時間依存性: 過去の期間に特化した学習")
    print("5. 統計的有意性の欠如: 少ない予測数による偶然の一致")
    
    print(f"\n実運用における想定リスク:")
    print("1. 環境変化への適応不足: 市場変動に対する脆弱性")  
    print("2. 機会損失: 極端に少ない投資機会")
    print("3. 精度低下: 新しいデータでの性能劣化")
    print("4. 実用性の欠如: 日次1-2件の予測では投資戦略として不十分")


if __name__ == "__main__":
    results = realistic_precision_test()
    analyze_overfitting_risk()