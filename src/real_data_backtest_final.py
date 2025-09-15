"""
実データを使用した完全なバックテストスクリプト
J-Quantsから取得した確実な実データのみを使用
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.features.feature_engineer import FeatureEngineer
from src.models.ensemble_model import EnsembleModel
from src.evaluation.time_series_validator import TimeSeriesValidator
from src.evaluation.precision_evaluator import PrecisionEvaluator
from src.evaluation.market_analyzer import MarketAnalyzer
from src.evaluation.trading_simulator import TradingSimulator

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_real_jquants_data():
    """
    J-Quantsから取得した実データを読み込み
    """
    data_dir = Path("data/real_jquants_data")
    
    # 最新のファイルを見つける
    if not data_dir.exists():
        raise FileNotFoundError("実データディレクトリが存在しません。先にjquants_real_data_fetcher.pyを実行してください。")
    
    pickle_files = list(data_dir.glob("nikkei225_real_data_*.pkl"))
    if not pickle_files:
        raise FileNotFoundError("実データファイルが見つかりません。先にjquants_real_data_fetcher.pyを実行してください。")
    
    # 最新ファイルを取得
    latest_file = max(pickle_files, key=os.path.getctime)
    logger.info(f"実データファイル読み込み: {latest_file}")
    
    df = pd.read_pickle(latest_file)
    
    # データ確認
    logger.info(f"実データ確認:")
    logger.info(f"  - 総レコード数: {len(df):,}件")
    logger.info(f"  - 銘柄数: {df['symbol'].nunique()}銘柄")
    logger.info(f"  - 期間: {df['date'].min().date()} ～ {df['date'].max().date()}")
    logger.info(f"  - カラム: {list(df.columns)}")
    logger.info(f"  - ターゲット分布: {df['target'].mean():.1%}")
    
    return df

def run_real_data_backtest():
    """
    実データを使用した完全なバックテスト実行
    """
    logger.info("=== 100%実データバックテスト開始 ===")
    
    try:
        # Step 1: 実データ読み込み
        df = load_real_jquants_data()
        
        # データクリーニング：重複除去とインデックスリセット
        logger.info("データクリーニング実行中...")
        df = df.drop_duplicates(subset=['date', 'symbol']).reset_index(drop=True)
        df = df.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # 必要なカラムのみ使用（特徴量作成に必要な基本カラム）
        required_cols = ['date', 'symbol', 'close_price', 'open_price', 'high_price', 
                        'low_price', 'volume', 'daily_return', 'next_day_return', 'target']
        
        # 存在するカラムのみ保持
        available_cols = [col for col in required_cols if col in df.columns]
        df_clean = df[available_cols].copy()
        
        # FeatureEngineerが期待する形式に変換
        df_clean = df_clean.rename(columns={
            'date': 'Date',
            'symbol': 'Code', 
            'close_price': 'Close',
            'open_price': 'Open',
            'high_price': 'High',
            'low_price': 'Low',
            'volume': 'Volume'
        })
        
        logger.info(f"クリーニング後データ: {len(df_clean)}件、{len(df_clean.columns)}カラム")
        
        # Step 2: 特徴量作成
        logger.info("特徴量エンジニアリング実行中...")
        feature_engineer = FeatureEngineer()
        df_with_features = feature_engineer.generate_features(df_clean)
        
        logger.info(f"特徴量作成後:")
        logger.info(f"  - レコード数: {len(df_with_features):,}件")
        logger.info(f"  - 特徴量数: {df_with_features.shape[1]}カラム")
        logger.info(f"  - 欠損値除去後: {df_with_features.dropna().shape[0]:,}件")
        
        # Step 3: データ分割
        df_clean = df_with_features.dropna()
        
        # 時系列順にソート
        df_clean = df_clean.sort_values(['date', 'symbol']).reset_index(drop=True)
        
        # 特徴量カラム特定
        feature_columns = [col for col in df_clean.columns 
                          if col not in ['date', 'symbol', 'target', 'close_price', 'daily_return', 'next_day_return']]
        
        logger.info(f"使用特徴量: {len(feature_columns)}個")
        logger.info(f"期間: {df_clean['date'].min().date()} ～ {df_clean['date'].max().date()}")
        
        # Step 4: 時系列バリデーション
        logger.info("時系列クロスバリデーション実行中...")
        ts_validator = TimeSeriesValidator(n_splits=5, gap_days=5)
        
        splits = ts_validator.create_time_series_splits(
            df_clean[feature_columns + ['date']], 
            df_clean['target'], 
            date_col='date'
        )
        
        logger.info(f"分割数: {len(splits)}分割")
        
        # Step 5: モデル訓練・評価
        logger.info("アンサンブルモデル訓練・評価中...")
        ensemble_model = EnsembleModel()
        precision_evaluator = PrecisionEvaluator()
        
        all_predictions = []
        fold_results = []
        
        for fold, (train_idx, test_idx) in enumerate(splits, 1):
            logger.info(f"  Fold {fold}/{len(splits)} 実行中...")
            
            X_train = df_clean.iloc[train_idx][feature_columns]
            y_train = df_clean.iloc[train_idx]['target']
            X_test = df_clean.iloc[test_idx][feature_columns]
            y_test = df_clean.iloc[test_idx]['target']
            
            # モデル訓練
            ensemble_model.fit(X_train, y_train)
            
            # 予測
            y_pred_proba = ensemble_model.predict_proba(X_test)[:, 1]
            
            # 予測結果保存
            test_data = df_clean.iloc[test_idx].copy()
            test_data['prediction_proba'] = y_pred_proba
            test_data['fold'] = fold
            all_predictions.append(test_data)
            
            # Fold結果評価
            returns_data = df_clean.iloc[test_idx]['next_day_return']
            metrics = precision_evaluator.calculate_precision_metrics(
                y_test, y_pred_proba, returns=returns_data
            )
            
            fold_results.append({
                'fold': fold,
                'precision': metrics.precision,
                'recall': metrics.recall,
                'f1_score': metrics.f1_score,
                'avg_return': metrics.avg_return,
                'sharpe_ratio': metrics.sharpe_ratio,
                'samples': len(test_idx)
            })
            
            logger.info(f"    Precision: {metrics.precision:.3f}, Recall: {metrics.recall:.3f}")
        
        # Step 6: 総合結果集計
        predictions_df = pd.concat(all_predictions, ignore_index=True)
        
        logger.info("=== バックテスト結果集計 ===")
        
        # 全期間での評価
        overall_metrics = precision_evaluator.calculate_precision_metrics(
            predictions_df['target'],
            predictions_df['prediction_proba'],
            returns=predictions_df['next_day_return']
        )
        
        logger.info(f"全期間Precision: {overall_metrics.precision:.3f}")
        logger.info(f"全期間Recall: {overall_metrics.recall:.3f}")
        logger.info(f"全期間F1-Score: {overall_metrics.f1_score:.3f}")
        logger.info(f"平均リターン: {overall_metrics.avg_return:.4f}")
        logger.info(f"シャープレシオ: {overall_metrics.sharpe_ratio:.3f}")
        
        # Fold別結果
        fold_df = pd.DataFrame(fold_results)
        logger.info(f"Fold平均Precision: {fold_df['precision'].mean():.3f} ± {fold_df['precision'].std():.3f}")
        logger.info(f"Fold平均Recall: {fold_df['recall'].mean():.3f} ± {fold_df['recall'].std():.3f}")
        
        # Step 7: 市場環境分析
        logger.info("市場環境分析実行中...")
        market_analyzer = MarketAnalyzer()
        
        market_df = market_analyzer.classify_market_regime(
            predictions_df,
            date_col='date',
            price_col='close_price',
            volume_col='volume'
        )
        
        # 市場環境別パフォーマンス
        if 'market_regime' in market_df.columns:
            regime_performance = market_df.groupby('market_regime').agg({
                'target': 'mean',
                'prediction_proba': 'mean',
                'next_day_return': 'mean'
            })
            logger.info("市場環境別パフォーマンス:")
            for regime, data in regime_performance.iterrows():
                logger.info(f"  {regime}: Target={data['target']:.1%}, Avg_Return={data['next_day_return']:.4f}")
        
        # Step 8: トレーディングシミュレーション
        logger.info("トレーディングシミュレーション実行中...")
        trading_simulator = TradingSimulator(
            initial_capital=1000000,  # 100万円
            transaction_cost=0.001,   # 0.1%
            max_positions=3,          # 最大3銘柄
            min_confidence=0.75       # Precision ≥ 0.75目標
        )
        
        sim_results = trading_simulator.run_simulation(
            predictions_df,
            predictions_df,  # price_dataとpredictions_dfは同じ
            date_col='date',
            symbol_col='symbol',
            min_confidence=0.75
        )
        
        logger.info("=== トレーディングシミュレーション結果 ===")
        logger.info(f"総リターン: {sim_results['total_return']:.1%}")
        logger.info(f"年率リターン: {sim_results['annual_return']:.1%}")
        logger.info(f"シャープレシオ: {sim_results['sharpe_ratio']:.3f}")
        logger.info(f"最大ドローダウン: {sim_results['max_drawdown']:.1%}")
        logger.info(f"勝率: {sim_results['win_rate']:.1%}")
        logger.info(f"総取引数: {sim_results['total_trades']}回")
        
        # Step 9: 結果保存
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 予測結果保存
        results_dir = Path("data/backtest_results")
        results_dir.mkdir(exist_ok=True)
        
        predictions_df.to_pickle(results_dir / f"real_data_predictions_{timestamp}.pkl")
        fold_df.to_csv(results_dir / f"fold_results_{timestamp}.csv", index=False)
        
        # 最終まとめ
        logger.info("=== 実データバックテスト完了 ===")
        logger.info(f"✅ データソース: 100%実データ (J-Quants API)")
        logger.info(f"✅ レコード数: {len(predictions_df):,}件")
        logger.info(f"✅ 銘柄数: {predictions_df['symbol'].nunique()}銘柄")
        logger.info(f"✅ 期間: {predictions_df['date'].min().date()} ～ {predictions_df['date'].max().date()}")
        logger.info(f"✅ 全期間Precision: {overall_metrics.precision:.3f}")
        logger.info(f"✅ 目標達成: {'○' if overall_metrics.precision >= 0.75 else '×'} (目標: ≥0.750)")
        logger.info(f"✅ シミュレーション年率リターン: {sim_results['annual_return']:.1%}")
        
        return {
            'predictions': predictions_df,
            'fold_results': fold_df,
            'overall_metrics': overall_metrics,
            'simulation_results': sim_results,
            'market_analysis': market_df
        }
        
    except Exception as e:
        logger.error(f"実データバックテスト失敗: {str(e)}")
        raise

if __name__ == "__main__":
    results = run_real_data_backtest()