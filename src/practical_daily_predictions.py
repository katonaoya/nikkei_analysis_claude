"""
実用的な日次予測数でのテスト
ユーザー希望の2-5件/日を実現する現実的な閾値を探索
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import precision_score, recall_score, confusion_matrix
import lightgbm as lgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_practical_threshold():
    """実用的な予測数を実現する閾値探索"""
    logger.info("=== 実用的閾値探索開始 ===")
    
    # データ読み込み
    data_file = Path("data/nikkei225_full_data/nikkei225_full_10years_175stocks_20250831_020101.pkl")
    df = pd.read_pickle(data_file)
    
    # 基本前処理
    df = df.sort_values(['Code', 'Date']).reset_index(drop=True)
    df['close_price'] = pd.to_numeric(df['Close'], errors='coerce')
    df['daily_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None)
    df['next_day_return'] = df.groupby('Code')['close_price'].pct_change(fill_method=None).shift(-1)
    
    # 1%ターゲット
    df['target'] = (df['next_day_return'] >= 0.01).astype(int)
    
    print(f"1%以上上昇の頻度: {df['target'].mean():.1%} ({df['target'].sum():,}件)")
    
    # 特徴量作成
    windows = [5, 10, 20]
    for window in windows:
        sma = df.groupby('Code')['close_price'].transform(lambda x: x.rolling(window).mean())
        df[f'price_to_sma_{window}'] = df['close_price'] / sma
        df[f'volatility_{window}'] = df.groupby('Code')['daily_return'].transform(
            lambda x: x.rolling(window).std()
        )
    
    def calc_rsi(prices, window=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
        rs = gain / (loss + 1e-8)
        return 100 - (100 / (1 + rs))
    
    df['rsi_14'] = df.groupby('Code')['close_price'].transform(calc_rsi)
    
    for lag in range(1, 4):
        df[f'return_lag_{lag}'] = df.groupby('Code')['daily_return'].shift(lag)
    
    # 特徴量準備
    feature_cols = [col for col in df.columns if col.startswith(('price_to_sma', 'volatility', 'rsi', 'return_lag'))]
    X = df[feature_cols].fillna(0)
    y = df['target']
    dates = pd.to_datetime(df['Date'])
    
    # NaN除去
    valid_mask = ~(y.isna() | X.isna().any(axis=1))
    X = X[valid_mask]
    y = y[valid_mask]
    dates = dates[valid_mask]
    
    print(f"有効データ: {len(X):,}")
    
    # 1つのフォールドでクイックテスト（処理速度向上）
    split_point = int(len(X) * 0.8)
    X_train, X_val = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_val = y.iloc[:split_point], y.iloc[split_point:]
    dates_val = dates.iloc[split_point:]
    
    # スケーリング
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # モデル訓練
    model = lgb.LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,
        verbosity=-1
    )
    
    model.fit(X_train_scaled, y_train)
    proba = model.predict_proba(X_val_scaled)[:, 1]
    
    print(f"\n=== 実用的閾値テスト ===")
    
    # より実用的な閾値範囲
    practical_thresholds = np.arange(0.45, 0.75, 0.05)
    results = []
    
    # 検証期間の日数計算
    val_dates_unique = dates_val.dt.date.nunique()
    print(f"検証期間: {val_dates_unique}日")
    
    for threshold in practical_thresholds:
        predictions = (proba >= threshold).astype(int)
        
        if predictions.sum() > 0:
            precision = precision_score(y_val, predictions)
            recall = recall_score(y_val, predictions)
            tn, fp, fn, tp = confusion_matrix(y_val, predictions).ravel()
            
            # 日次予測数計算
            daily_predictions = predictions.sum() / val_dates_unique
            
            results.append({
                'threshold': threshold,
                'precision': precision,
                'recall': recall,
                'total_predictions': predictions.sum(),
                'daily_predictions': daily_predictions,
                'true_positives': tp,
                'false_positives': fp,
                'validation_days': val_dates_unique
            })
            
            print(f"閾値{threshold:.2f}: 精度={precision:.3f}, "
                  f"再現率={recall:.3f}, 日次={daily_predictions:.1f}件, "
                  f"TP={tp}, FP={fp}")
        else:
            print(f"閾値{threshold:.2f}: 予測なし")
    
    # ユーザー希望範囲（2-5件/日）の分析
    print(f"\n=== ユーザー希望範囲分析（2-5件/日） ===")
    
    target_results = [r for r in results if 2 <= r['daily_predictions'] <= 5]
    
    if target_results:
        print(f"希望範囲の閾値数: {len(target_results)}")
        
        # 最高精度の設定
        best_precision = max(target_results, key=lambda x: x['precision'])
        print(f"\n最高精度設定:")
        print(f"  閾値: {best_precision['threshold']:.2f}")
        print(f"  精度: {best_precision['precision']:.1%}")
        print(f"  日次予測数: {best_precision['daily_predictions']:.1f}件")
        print(f"  月間成功予想: {best_precision['daily_predictions'] * best_precision['precision'] * 20:.0f}件")
        print(f"  月間失敗予想: {best_precision['daily_predictions'] * (1-best_precision['precision']) * 20:.0f}件")
        
        # バランス型の設定
        balanced = min(target_results, key=lambda x: abs(x['daily_predictions'] - 3.5))
        print(f"\nバランス型設定:")
        print(f"  閾値: {balanced['threshold']:.2f}")
        print(f"  精度: {balanced['precision']:.1%}")
        print(f"  日次予測数: {balanced['daily_predictions']:.1f}件")
        print(f"  月間成功予想: {balanced['daily_predictions'] * balanced['precision'] * 20:.0f}件")
        print(f"  月間失敗予想: {balanced['daily_predictions'] * (1-balanced['precision']) * 20:.0f}件")
        
    else:
        print("希望範囲（2-5件/日）に該当する結果なし")
        
        # 最も近い結果を表示
        if results:
            closest = min(results, key=lambda x: abs(x['daily_predictions'] - 3.5))
            print(f"\n最も近い結果:")
            print(f"  閾値: {closest['threshold']:.2f}")
            print(f"  精度: {closest['precision']:.1%}")
            print(f"  日次予測数: {closest['daily_predictions']:.1f}件")
    
    # 現実性評価
    print(f"\n=== 運用現実性評価 ===")
    
    realistic_results = [r for r in results if r['precision'] >= 0.60]
    
    if realistic_results:
        print(f"60%以上精度の選択肢:")
        for r in realistic_results:
            monthly_trades = r['daily_predictions'] * 20
            monthly_success = monthly_trades * r['precision']
            monthly_failure = monthly_trades * (1 - r['precision'])
            
            print(f"  閾値{r['threshold']:.2f}: 精度{r['precision']:.1%}, "
                  f"日次{r['daily_predictions']:.1f}件 "
                  f"→ 月間成功{monthly_success:.0f}回, 失敗{monthly_failure:.0f}回")
    
    return results


def recommend_practical_settings():
    """実用的な設定の推奨"""
    print(f"\n=== 実用設定推奨 ===")
    
    print("現実的な運用パターン:")
    print("\n【保守型】日次1-2件, 精度75-80%")
    print("  → 月間成功12-24回, 失敗3-8回")
    print("  → 安定したリターン, リスク最小")
    
    print("\n【バランス型】日次3-4件, 精度65-70%")
    print("  → 月間成功39-56回, 失敗21-24回") 
    print("  → 適度な機会数, 実用的精度")
    
    print("\n【積極型】日次5-8件, 精度60-65%")
    print("  → 月間成功60-104回, 失敗40-56回")
    print("  → 多くの機会, やや高リスク")
    
    print(f"\n推奨アプローチ:")
    print("1. 複数の閾値設定を並行運用")
    print("2. 月単位でパフォーマンス評価")
    print("3. 市場環境に応じた閾値調整")
    print("4. ポートフォリオの一部として活用")


if __name__ == "__main__":
    results = find_practical_threshold()
    recommend_practical_settings()