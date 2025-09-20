#!/usr/bin/env python3
"""
51.4%達成の再確認と詳細分析
"""

import pandas as pd
import numpy as np
from pathlib import Path
from loguru import logger
import sys
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ログ設定
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", level="INFO")

def main():
    """51.4%達成の詳細検証"""
    logger.info("🎯 51.4%精度の詳細検証")
    
    # データ読み込み
    data_dir = Path("data")
    processed_dir = data_dir / "processed"
    processed_files = list(processed_dir.glob("*.parquet"))
    
    if not processed_files:
        logger.error("❌ 処理済みデータが見つかりません")
        return
        
    df = pd.read_parquet(processed_files[0])
    logger.info(f"✅ データ読み込み: {len(df):,}件")
    
    # 最適特徴量（究極のテストで判明）
    optimal_features = ['Market_Breadth', 'Market_Return', 'Volatility_20', 'Price_vs_MA20']
    
    # データ準備
    clean_df = df[df['Binary_Direction'].notna()].copy()
    clean_df = clean_df.sort_values(['Date', 'Code']).reset_index(drop=True)
    
    # 特徴量存在確認
    missing = [f for f in optimal_features if f not in clean_df.columns]
    if missing:
        logger.error(f"❌ 不足特徴量: {missing}")
        return
        
    X = clean_df[optimal_features].fillna(0)
    y = clean_df['Binary_Direction'].astype(int)
    
    logger.info(f"検証データ: {len(clean_df):,}件")
    logger.info(f"使用特徴量: {optimal_features}")
    
    # 最適構成での検証
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 最適パラメータ（究極のテストで判明）
    model = LogisticRegression(
        C=0.001, 
        class_weight='balanced', 
        random_state=42, 
        max_iter=1000,
        solver='lbfgs'
    )
    
    # 5分割時系列評価
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    detailed_results = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X_scaled)):
        X_train = X_scaled[train_idx]
        X_test = X_scaled[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, pred)
        scores.append(accuracy)
        
        detailed_results.append({
            'fold': fold + 1,
            'accuracy': accuracy,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'class_dist': y_test.value_counts().to_dict()
        })
        
        logger.info(f"  Fold {fold+1}: {accuracy:.1%} (Train: {len(X_train):,}, Test: {len(X_test):,})")
    
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    # 結果詳細
    logger.info("\n" + "="*80)
    logger.info("🎯 51.4%精度検証結果")
    logger.info("="*80)
    logger.info(f"平均精度: {avg_score:.3%}")
    logger.info(f"標準偏差: {std_score:.3%}")
    logger.info(f"範囲: {min(scores):.1%} - {max(scores):.1%}")
    
    # 各Foldの詳細
    logger.info("\n📊 Fold別詳細:")
    for result in detailed_results:
        logger.info(f"  Fold {result['fold']}: {result['accuracy']:.1%}")
        logger.info(f"    Train: {result['train_size']:,}, Test: {result['test_size']:,}")
        logger.info(f"    クラス分布: {result['class_dist']}")
    
    # 目標達成確認
    target = 0.514  # 51.4%
    if avg_score >= target:
        logger.info(f"\n✅ 51.4%達成確認！ ({avg_score:.1%} >= {target:.1%})")
    else:
        logger.warning(f"\n⚠️ 51.4%未達成 ({avg_score:.1%} < {target:.1%})")
        logger.info(f"差: {(target - avg_score)*100:.2f}%")
    
    # 特徴量重要度
    logger.info("\n🔍 特徴量重要度:")
    importances = abs(model.coef_[0])
    feature_importance = list(zip(optimal_features, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for feature, importance in feature_importance:
        logger.info(f"  {feature:20s}: {importance:.4f}")
    
    logger.info(f"\n⚖️ この結果は全データ{len(clean_df):,}件での厳密な5分割時系列検証です")
    
if __name__ == "__main__":
    main()